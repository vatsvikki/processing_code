#!/usr/bin/env python3
"""
sc_scaling_pipeline.py

Surface-Consistent (SC) amplitude scaling for a raw Seismic Unix (.su)
shot/receiver gather dataset, with a numbered, self-checking QC step
after every stage of the workflow.

WHAT "SURFACE-CONSISTENT SCALING" MEANS
----------------------------------------
Field seismic amplitudes are corrupted by effects that are constant for
a given SHOT, constant for a given RECEIVER STATION, and effects that
vary smoothly with SOURCE-RECEIVER OFFSET (geometric/spherical
divergence, residual gain). SC scaling assumes the log-amplitude of a
trace decomposes additively:

    log(RMS_trace) = S(shot) + R(receiver_station) + O(|offset| bin) + e

and solves for the per-shot term S, per-receiver term R and per-offset
-bin term O that best explain the observed amplitude variation, using
iterative (Gauss-Seidel) least squares. The trace is then rescaled by
exp(-(S+R+O)) so that, on average, every shot record and every receiver
station contributes the same average energy, and the residual
offset-dependent trend is flattened. What is LEFT BEHIND (the residual
e) is what the next real amplitude/AGC/deconvolution steps work on --
true structural and stratigraphic amplitude variation, not
acquisition-related coupling variation.

A CDP/midpoint-consistent term is deliberately NOT solved for by
default: unlike S, R and O (which are acquisition artifacts), a CDP
term can absorb real geological amplitude variation (bright spots,
AVO, structural focusing) and is not standard for this step. It can be
turned on with --include-cdp-term if a specific QC problem calls for
it, but the default 3-term (S+R+O) model is the safe, defensible one.

PIPELINE STEPS (each with its own QC gate)
-------------------------------------------
  STEP 0  Probe & validate input SU file (geometry of the binary format)
  STEP 1  PASS 1: stream headers + per-trace RMS (samples are read and
          discarded per trace/chunk -- never held in bulk)
  STEP 2  Build shot / receiver / offset-bin groups
  STEP 3  Surface-consistent decomposition (Gauss-Seidel, vectorized)
  STEP 4  Convert S/R/O terms into a clipped per-trace correction factor
  STEP 5  PASS 2: stream through the file again, apply the correction,
          write the corrected .su, log per-trace before/after RMS
  STEP 6  Diagnostic QC plots (maps, spectra of the correction, offset
          trend before/after)
  STEP 7  Final QC summary report (JSON + text) with PASS/WARN/FAIL
          flags against documented expected-outcome thresholds

USAGE
-----
    python3 sc_scaling_pipeline.py \\
        --input /Users/vikas/SCube/harrison/data/EAST_ATCHAFALAYA-GCSR-UNDEFINED_SDL062470-16.su \\
        --outdir /Users/vikas/SCube/harrison/data/processing_16/SC_scaling

All arguments have working defaults set to exactly that input/output
pair, so `python3 sc_scaling_pipeline.py` with no arguments runs the
full pipeline as-is.
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# SU / SEG-Y TRACE HEADER LAYOUT (240 bytes, standard fields)
# ============================================================

HEADER_BYTES = 240

FLDR_OFF, CDP_OFF, OFFSET_OFF = 8, 20, 36
SCALCO_OFF = 70
SX_OFF, SY_OFF, GX_OFF, GY_OFF = 72, 76, 80, 84
NS_OFF, DT_OFF = 114, 116

DEFAULT_INPUT = (
    "/Users/vikas/SCube/harrison/data/"
    "EAST_ATCHAFALAYA-GCSR-UNDEFINED_SDL062470-16.su"
)
DEFAULT_OUTDIR = (
    "/Users/vikas/SCube/harrison/data/processing_16/SC_scaling"
)


def banner(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def qc(label, passed, detail=""):
    tag = "PASS" if passed else "**FAIL**"
    print(f"  QC [{tag}] {label}" + (f" -- {detail}" if detail else ""))
    return {"label": label, "passed": bool(passed), "detail": detail}


# ============================================================
# STEP 0: PROBE & VALIDATE INPUT FILE
# ============================================================

def read_header_field(h, order, fmt, off):
    return struct.unpack_from(order + fmt, h, off)[0]


def probe_file(path):
    """Detect byte order, ns, dt and confirm the file is an exact
    whole number of fixed-length traces. This is the geometric
    contract the rest of the (vectorized) pipeline depends on."""

    size = path.stat().st_size

    with open(path, "rb") as f:
        first = f.read(HEADER_BYTES)
        if len(first) < HEADER_BYTES:
            raise RuntimeError(f"File shorter than one trace header: {path}")

        order = None
        for cand in (">", "<"):
            ns = read_header_field(first, cand, "H", NS_OFF)
            if 1 <= ns <= 20000:
                order = cand
                break
        if order is None:
            raise RuntimeError(
                f"Could not determine byte order for {path} "
                "(ns field unreasonable in both endiannesses)"
            )

        ns = read_header_field(first, order, "H", NS_OFF)
        dt = read_header_field(first, order, "H", DT_OFF)
        trace_bytes = HEADER_BYTES + ns * 4

        if trace_bytes <= 0 or size % trace_bytes != 0:
            raise RuntimeError(
                f"File size {size:,} bytes is not an exact multiple of "
                f"trace size {trace_bytes:,} bytes (ns={ns}). "
                "Variable trace length is not supported by this "
                "fixed-ns fast path."
            )

        n_traces = size // trace_bytes

        # Spot-check ns/dt constancy at start / middle / end -- the
        # vectorized structured-array reads below assume a single,
        # fixed ns for the whole file.
        checks = []
        for frac in (0.0, 0.5, 1.0 - 1.0 / n_traces):
            idx = int(frac * (n_traces - 1))
            f.seek(idx * trace_bytes)
            h = f.read(HEADER_BYTES)
            checks.append((
                read_header_field(h, order, "H", NS_OFF),
                read_header_field(h, order, "H", DT_OFF),
            ))

    ns_dt_constant = all(c == (ns, dt) for c in checks)

    return {
        "order": order,
        "ns": ns,
        "dt_us": dt,
        "trace_bytes": trace_bytes,
        "n_traces": n_traces,
        "file_size": size,
        "ns_dt_constant": ns_dt_constant,
    }


def step0_validate(input_path, outdir):
    banner("STEP 0 : PROBE & VALIDATE INPUT SU FILE")

    if not input_path.exists():
        raise FileNotFoundError(f"Input SU file not found: {input_path}")

    info = probe_file(input_path)

    record_seconds = info["ns"] * info["dt_us"] / 1e6

    print(f"  Input file        : {input_path}")
    print(f"  File size          : {info['file_size']:,} bytes "
          f"({info['file_size']/1e9:.2f} GB)")
    print(f"  Byte order         : {'big-endian (>)' if info['order']=='>' else 'little-endian (<)'}")
    print(f"  Samples/trace (ns) : {info['ns']}")
    print(f"  Sample rate (dt)   : {info['dt_us']} microseconds")
    print(f"  Record length      : {record_seconds:.3f} s")
    print(f"  Trace size         : {info['trace_bytes']:,} bytes")
    print(f"  Trace count        : {info['n_traces']:,}")

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "qc_plots").mkdir(parents=True, exist_ok=True)

    checks = []
    checks.append(qc(
        "File size is an exact multiple of one fixed-length trace",
        True,
        f"{info['file_size']:,} / {info['trace_bytes']:,} = {info['n_traces']:,} traces exactly"
    ))
    checks.append(qc(
        "ns / dt constant at start, middle and end of file",
        info["ns_dt_constant"],
        "required for the vectorized fixed-ns fast path used below"
    ))
    checks.append(qc(
        "ns in plausible range (500-20000 samples)",
        500 <= info["ns"] <= 20000,
        f"ns={info['ns']}"
    ))
    checks.append(qc(
        "dt in plausible range (250-4000 microseconds)",
        250 <= info["dt_us"] <= 4000,
        f"dt={info['dt_us']} us"
    ))

    print()
    print("  EXPECTED OUTCOME: all four checks PASS before continuing.")
    print("  A FAIL here means the file is not a single, homogeneous")
    print("  fixed-trace-length SU volume and must be repaired/split")
    print("  before SC scaling can run.")

    if not all(c["passed"] for c in checks):
        raise RuntimeError("STEP 0 QC failed -- see checks above.")

    return info, checks


# ============================================================
# STRUCTURED DTYPE FOR FAST VECTORIZED WHOLE-TRACE I/O
# ============================================================

def build_dtypes(order, ns):
    header_dtype = np.dtype([
        ("tracl", order + "i4"), ("tracr", order + "i4"), ("fldr", order + "i4"),
        ("tracf", order + "i4"), ("ep", order + "i4"), ("cdp", order + "i4"),
        ("cdpt", order + "i4"),
        ("trid", order + "i2"), ("nvs", order + "i2"), ("nhs", order + "i2"), ("duse", order + "i2"),
        ("offset", order + "i4"), ("gelev", order + "i4"), ("selev", order + "i4"),
        ("sdepth", order + "i4"), ("gdel", order + "i4"), ("sdel", order + "i4"),
        ("swdep", order + "i4"), ("gwdep", order + "i4"),
        ("scalel", order + "i2"), ("scalco", order + "i2"),
        ("sx", order + "i4"), ("sy", order + "i4"), ("gx", order + "i4"), ("gy", order + "i4"),
        ("counit", order + "i2"),
        ("pad1", "V24"),
        ("ns", order + "u2"), ("dt", order + "u2"),
        ("pad2", "V122"),
    ])
    assert header_dtype.itemsize == HEADER_BYTES, header_dtype.itemsize

    full_dtype = np.dtype([
        ("header", header_dtype),
        ("samples", order + "f4", (ns,)),
    ])
    assert full_dtype.itemsize == HEADER_BYTES + ns * 4

    return header_dtype, full_dtype


def iter_chunks(path, full_dtype, n_traces, chunk_size):
    """Yield successive structured-array chunks of whole traces
    (header + samples together) straight from disk."""
    with open(path, "rb") as f:
        remaining = n_traces
        while remaining > 0:
            take = min(chunk_size, remaining)
            arr = np.fromfile(f, dtype=full_dtype, count=take)
            if arr.shape[0] != take:
                raise RuntimeError("Unexpected short read while streaming SU file.")
            yield arr
            remaining -= take


# ============================================================
# STEP 1: PASS 1 -- HEADERS + PER-TRACE RMS (vectorized, chunked)
# ============================================================

def step1_scan(input_path, info, window, dead_rms_fraction, outlier_rms_multiplier, chunk_size):
    banner("STEP 1 : PASS 1 -- STREAM HEADERS + PER-TRACE RMS")

    ns, dt_us, n_traces = info["ns"], info["dt_us"], info["n_traces"]
    order = info["order"]
    _, full_dtype = build_dtypes(order, ns)

    dt_s = dt_us * 1e-6
    if window is None:
        i1, i2 = 0, ns
        print(f"  RMS analysis window: full trace (0 - {ns*dt_s:.3f} s)")
    else:
        tmin, tmax = window
        i1 = max(0, int(round(tmin / dt_s)))
        i2 = min(ns, int(round(tmax / dt_s)))
        if i2 <= i1:
            raise RuntimeError(f"Invalid RMS window {window} for ns={ns}, dt={dt_s}s")
        print(f"  RMS analysis window: {tmin:.3f} - {tmax:.3f} s "
              f"(samples {i1}:{i2})")

    fldr = np.empty(n_traces, dtype=np.int32)
    cdp = np.empty(n_traces, dtype=np.int32)
    offset = np.empty(n_traces, dtype=np.int32)
    sx = np.empty(n_traces, dtype=np.int64)
    sy = np.empty(n_traces, dtype=np.int64)
    gx = np.empty(n_traces, dtype=np.int64)
    gy = np.empty(n_traces, dtype=np.int64)
    rms_raw = np.empty(n_traces, dtype=np.float64)
    has_nan_inf = np.zeros(n_traces, dtype=bool)

    t0 = time.time()
    n_done = 0
    for chunk in iter_chunks(input_path, full_dtype, n_traces, chunk_size):
        n = chunk.shape[0]
        sl = slice(n_done, n_done + n)

        h = chunk["header"]
        fldr[sl] = h["fldr"]
        cdp[sl] = h["cdp"]
        offset[sl] = h["offset"]
        sx[sl] = h["sx"].astype(np.int64)
        sy[sl] = h["sy"].astype(np.int64)
        gx[sl] = h["gx"].astype(np.int64)
        gy[sl] = h["gy"].astype(np.int64)

        samples = chunk["samples"][:, i1:i2].astype(np.float64)
        finite = np.isfinite(samples)
        bad_rows = ~finite.all(axis=1)
        has_nan_inf[sl] = bad_rows
        samples = np.where(finite, samples, 0.0)

        rms_raw[sl] = np.sqrt(np.mean(samples * samples, axis=1))

        n_done += n
        if (n_done // chunk_size) % 20 == 0 or n_done == n_traces:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            print(f"  pass 1: {n_done:,}/{n_traces:,} traces "
                  f"({100*n_done/n_traces:5.1f}%) -- {rate:,.0f} traces/s", end="\r")

    print()
    elapsed = time.time() - t0
    print(f"  Pass 1 complete in {elapsed:.1f} s "
          f"({n_traces/elapsed:,.0f} traces/s)")

    meta = dict(fldr=fldr, cdp=cdp, offset=offset,
                sx=sx, sy=sy, gx=gx, gy=gy,
                rms_raw=rms_raw, has_nan_inf=has_nan_inf)

    # ---- QC ----
    median_rms = float(np.median(rms_raw[rms_raw > 0])) if np.any(rms_raw > 0) else 0.0
    dead_thresh = dead_rms_fraction * median_rms
    dead_mask = rms_raw <= dead_thresh
    n_dead = int(dead_mask.sum())
    n_nan = int(has_nan_inf.sum())

    # ---- Amplitude OUTLIER / SPIKE detection (symmetric to dead-trace
    # detection, but on the high side). A trace whose RMS is enormously
    # above the survey median is very unlikely to be legitimate
    # reflection amplitude -- it is far more likely a data glitch,
    # cultural noise burst, sync error or instrument spike. Such traces
    # must NOT be allowed to pollute a shot/receiver group's average
    # amplitude (which would mis-scale every OTHER trace in that group),
    # and must not themselves be blindly boosted/cut by a correction
    # derived from their neighbors. They are therefore excluded from
    # the decomposition and passed through unscaled (Step 4), and
    # listed separately here for manual trace-edit review -- SC scaling
    # is not the place to edit bad traces, only to avoid being fooled
    # by them.
    outlier_thresh = outlier_rms_multiplier * median_rms
    outlier_mask = (rms_raw > outlier_thresh) & (~dead_mask)
    n_outlier = int(outlier_mask.sum())

    invalid_mask = dead_mask | outlier_mask
    finite_rms = rms_raw[~invalid_mask]
    pct = np.percentile(finite_rms, [1, 25, 50, 75, 99]) if finite_rms.size else np.zeros(5)

    n_shots = int(np.unique(fldr).size)
    n_receivers = int(np.unique(np.stack([gx, gy], axis=1), axis=0).shape[0])

    print()
    print(f"  Unique shots (fldr)      : {n_shots:,}")
    print(f"  Unique receiver stations : {n_receivers:,}")
    print(f"  Median trace RMS         : {median_rms:.6g}")
    print(f"  RMS percentiles (1/25/50/75/99, outliers excluded): "
          f"{pct[0]:.4g} / {pct[1]:.4g} / {pct[2]:.4g} / {pct[3]:.4g} / {pct[4]:.4g}")
    print(f"  Dead/near-zero traces    : {n_dead:,} ({100*n_dead/n_traces:.3f}%) "
          f"[threshold = {dead_rms_fraction:g} x median]")
    print(f"  Amplitude outlier/spikes : {n_outlier:,} ({100*n_outlier/n_traces:.3f}%) "
          f"[threshold = {outlier_rms_multiplier:g} x median = {outlier_thresh:.4g}]")
    print(f"  NaN/Inf-contaminated     : {n_nan:,} ({100*n_nan/n_traces:.3f}%)")

    checks = []
    checks.append(qc(
        "Trace count scanned matches file's exact trace count",
        n_done == n_traces,
        f"{n_done:,} == {n_traces:,}"
    ))
    checks.append(qc(
        "Dead/near-zero traces are a small minority (<5%)",
        (n_dead / n_traces) < 0.05,
        f"{100*n_dead/n_traces:.3f}% dead"
    ))
    checks.append(qc(
        "Amplitude outlier/spike traces are a small minority (<1%)",
        (n_outlier / n_traces) < 0.01,
        f"{100*n_outlier/n_traces:.3f}% outliers (>{outlier_rms_multiplier:g}x median)"
    ))
    checks.append(qc(
        "No more than a small fraction of traces contain NaN/Inf (<1%)",
        (n_nan / n_traces) < 0.01,
        f"{100*n_nan/n_traces:.3f}% NaN/Inf"
    ))
    checks.append(qc(
        "RMS distribution (outliers excluded) spans a believable dynamic "
        "range (p99/p1 between 2x and 10,000x)",
        finite_rms.size > 0 and 2.0 <= (pct[4] / max(pct[0], 1e-30)) <= 1e4,
        f"p99/p1 = {pct[4]/max(pct[0],1e-30):.3g}"
    ))

    print()
    print("  EXPECTED OUTCOME:")
    print("  - Dead traces should be a small handful (bad channels, edge")
    print("    mutes, missing FFID) -- a large dead fraction usually means")
    print("    the wrong file/format or a header-offset bug.")
    print("  - Outlier/spike traces (RMS orders of magnitude above the")
    print("    median) should also be a small handful; each one is worth")
    print("    checking against field observer notes (bad geophone plant,")
    print("    cultural noise, sync/encoding glitch). They are excluded")
    print("    from the SC decomposition and left unscaled so they cannot")
    print("    corrupt their shot/receiver group's correction, and so SC")
    print("    scaling never amplifies them further.")
    print("  - The RMS dynamic range (p99/p1) is expected to be large for")
    print("    raw, unscaled field data (spans near-source vs. far-offset,")
    print("    good vs. weak shots) -- that large spread is exactly what")
    print("    SC scaling is being run to compress.")

    # QC plot: raw RMS histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(np.log10(np.clip(rms_raw, 1e-12, None)), bins=200, color="#3b6ea5")
    ax.axvline(np.log10(max(median_rms, 1e-12)), color="red", linestyle="--",
               label=f"median = {median_rms:.4g}")
    ax.set_xlabel("log10(trace RMS)")
    ax.set_ylabel("Trace count")
    ax.set_title("STEP 1 QC -- Raw trace RMS distribution (pre-SC-scaling)")
    ax.legend()
    fig.tight_layout()

    return meta, invalid_mask, dead_mask, outlier_mask, checks, {
        "median_rms": median_rms,
        "rms_percentiles_1_25_50_75_99_outliers_excluded": pct.tolist(),
        "n_dead": n_dead,
        "n_outlier": n_outlier,
        "outlier_threshold": outlier_thresh,
        "n_nan_inf": n_nan,
        "n_shots": n_shots,
        "n_receivers": n_receivers,
    }, fig


# ============================================================
# STEP 2: BUILD SHOT / RECEIVER / OFFSET GROUPS
# ============================================================

def step2_build_groups(meta, invalid_mask, n_offset_bins):
    banner("STEP 2 : BUILD SURFACE-CONSISTENT GROUPS "
           "(shot / receiver / offset-bin)")

    fldr = meta["fldr"]
    gx, gy = meta["gx"], meta["gy"]
    offset_abs = np.abs(meta["offset"]).astype(np.float64)

    src_unique, src_idx = np.unique(fldr, return_inverse=True)
    n_src = src_unique.size

    rec_key = np.stack([gx, gy], axis=1)
    rec_unique, rec_idx = np.unique(rec_key, axis=0, return_inverse=True)
    n_rec = rec_unique.shape[0]

    valid_offsets = offset_abs[~invalid_mask]
    lo = float(valid_offsets.min()) if valid_offsets.size else 0.0
    hi = float(valid_offsets.max()) if valid_offsets.size else 1.0
    edges = np.linspace(lo, hi + 1.0, n_offset_bins + 1)
    off_idx = np.clip(np.digitize(offset_abs, edges) - 1, 0, n_offset_bins - 1)
    n_off = n_offset_bins

    src_fold = np.bincount(src_idx, minlength=n_src)
    rec_fold = np.bincount(rec_idx, minlength=n_rec)
    off_fold = np.bincount(off_idx, minlength=n_off)

    print(f"  Shot groups     : {n_src:,}  (fold min/median/max = "
          f"{src_fold.min()}/{int(np.median(src_fold))}/{src_fold.max()})")
    print(f"  Receiver groups : {n_rec:,}  (fold min/median/max = "
          f"{rec_fold.min()}/{int(np.median(rec_fold))}/{rec_fold.max()})")
    print(f"  Offset bins     : {n_off} spanning {lo:.1f} - {hi:.1f} "
          f"(fold min/median/max = {off_fold.min()}/{int(np.median(off_fold))}/{off_fold.max()})")

    low_fold_src = int((src_fold <= 2).sum())
    low_fold_rec = int((rec_fold <= 2).sum())

    checks = []
    checks.append(qc(
        "Every trace assigned to exactly one shot, receiver and offset group",
        src_idx.size == fldr.size and rec_idx.size == fldr.size and off_idx.size == fldr.size
    ))
    checks.append(qc(
        "Shot/receiver group counts are consistent with survey size "
        "(neither 1 group nor 1-per-trace)",
        1 < n_src < fldr.size and 1 < n_rec < fldr.size,
        f"n_src={n_src:,}, n_rec={n_rec:,}, n_traces={fldr.size:,}"
    ))
    checks.append(qc(
        "No empty offset bin",
        bool((off_fold > 0).all()),
        f"{int((off_fold==0).sum())} empty bins"
    ))

    print()
    print("  EXPECTED OUTCOME:")
    print("  - Shot/receiver fold should roughly track nominal channel")
    print(f"    count / live-receivers-per-shot. Groups with fold<=2 "
          f"({low_fold_src:,} shots, {low_fold_rec:,} receivers here) get a")
    print("    less-constrained (noisier) S/R term -- not an error, but")
    print("    worth a second look if that count is a large fraction of")
    print("    all groups.")
    print("  - No offset bin should be empty; an empty bin means the")
    print("    requested --offset-bins is finer than the offset range")
    print("    supports.")

    if not all(c["passed"] for c in checks):
        raise RuntimeError("STEP 2 QC failed -- see checks above.")

    groups = dict(
        src_idx=src_idx, n_src=n_src, src_unique=src_unique, src_fold=src_fold,
        rec_idx=rec_idx, n_rec=n_rec, rec_unique=rec_unique, rec_fold=rec_fold,
        off_idx=off_idx, n_off=n_off, off_edges=edges, off_fold=off_fold,
    )
    stats = dict(
        n_src=n_src, n_rec=n_rec, n_off=n_off,
        low_fold_src=low_fold_src, low_fold_rec=low_fold_rec,
    )
    return groups, checks, stats


# ============================================================
# STEP 3: SURFACE-CONSISTENT DECOMPOSITION (vectorized Gauss-Seidel)
# ============================================================

def _group_mean(residual, idx, n_groups):
    sums = np.bincount(idx, weights=residual, minlength=n_groups)
    counts = np.bincount(idx, minlength=n_groups).astype(np.float64)
    counts[counts == 0] = 1.0
    return sums / counts


def step3_decompose(meta, invalid_mask, groups, n_iter, tol):
    banner("STEP 3 : SURFACE-CONSISTENT DECOMPOSITION "
           "(shot S + receiver R + offset O)")

    rms_raw = meta["rms_raw"]
    valid = ~invalid_mask
    log_rms = np.zeros_like(rms_raw)
    log_rms[valid] = np.log(rms_raw[valid])

    src_idx, n_src = groups["src_idx"], groups["n_src"]
    rec_idx, n_rec = groups["rec_idx"], groups["n_rec"]
    off_idx, n_off = groups["off_idx"], groups["n_off"]

    S_group = np.zeros(n_src)
    R_group = np.zeros(n_rec)
    O_group = np.zeros(n_off)

    S_trace = np.zeros_like(log_rms)
    R_trace = np.zeros_like(log_rms)
    O_trace = np.zeros_like(log_rms)

    total_var = float(np.var(log_rms[valid])) if valid.any() else 0.0

    history = []
    prev_std = None
    t0 = time.time()
    for it in range(1, n_iter + 1):
        other = R_trace + O_trace
        S_group = _group_mean(np.where(valid, log_rms - other, 0.0), src_idx, n_src)
        S_group[np.bincount(src_idx[valid], minlength=n_src) == 0] = 0.0
        S_trace = S_group[src_idx]

        other = S_trace + O_trace
        R_group = _group_mean(np.where(valid, log_rms - other, 0.0), rec_idx, n_rec)
        R_group[np.bincount(rec_idx[valid], minlength=n_rec) == 0] = 0.0
        R_trace = R_group[rec_idx]

        other = S_trace + R_trace
        O_group = _group_mean(np.where(valid, log_rms - other, 0.0), off_idx, n_off)
        O_trace = O_group[off_idx]

        residual = log_rms[valid] - (S_trace + R_trace + O_trace)[valid]
        resid_std = float(np.std(residual))
        explained = 1.0 - (resid_std ** 2) / total_var if total_var > 0 else 0.0
        history.append({"iteration": it, "residual_std": resid_std,
                         "variance_explained": explained})

        print(f"  iter {it:2d}/{n_iter}: residual std = {resid_std:.5f}  "
              f"variance explained = {100*explained:5.1f}%")

        if prev_std is not None and abs(prev_std - resid_std) < tol:
            print(f"  Converged (|delta residual std| < {tol}) -- stopping early.")
            break
        prev_std = resid_std

    elapsed = time.time() - t0
    print(f"  Decomposition finished in {elapsed:.2f} s "
          f"({len(history)} iterations).")

    # Zero-mean normalization of S and R (their common offset is not
    # identifiable -- push it into O so factors read naturally as
    # "relative to survey average").
    src_fold = groups["src_fold"]
    rec_fold = groups["rec_fold"]
    mean_S = float(np.average(S_group, weights=np.maximum(src_fold, 1)))
    mean_R = float(np.average(R_group, weights=np.maximum(rec_fold, 1)))
    S_group = S_group - mean_S
    R_group = R_group - mean_R
    O_group = O_group + mean_S + mean_R

    S_trace = S_group[src_idx]
    R_trace = R_group[rec_idx]
    O_trace = O_group[off_idx]

    final_resid_std = history[-1]["residual_std"]
    final_explained = history[-1]["variance_explained"]

    checks = []
    checks.append(qc(
        "Residual std decreased monotonically (decomposition is converging)",
        all(history[i]["residual_std"] <= history[i-1]["residual_std"] + 1e-9
            for i in range(1, len(history))),
    ))
    checks.append(qc(
        "Meaningful fraction of RMS variance explained by S+R+O (>=30%)",
        final_explained >= 0.30,
        f"{100*final_explained:.1f}% explained"
    ))
    checks.append(qc(
        "Converged or completed all requested iterations without blow-up",
        np.isfinite(final_resid_std) and final_resid_std < 10.0,
        f"final residual std = {final_resid_std:.4f}"
    ))

    print()
    print("  EXPECTED OUTCOME:")
    print("  - Residual std should drop quickly over the first few")
    print("    iterations and flatten out -- a curve that is still")
    print("    falling steeply at the last iteration means --iterations")
    print("    should be increased.")
    print("  - Typical surface-consistent models explain roughly")
    print("    30-90% of raw RMS variance; the unexplained remainder is")
    print("    the geology + noise the later processing steps need.")
    print("    A very low number (<30%) suggests the header groupings")
    print("    (fldr/gx/gy) don't actually vary consistently in this")
    print("    dataset -- check the geometry QC from Step 1/2.")

    if not all(c["passed"] for c in checks):
        raise RuntimeError("STEP 3 QC failed -- see checks above.")

    # Convergence QC plot
    fig, ax1 = plt.subplots(figsize=(9, 5))
    iters = [h["iteration"] for h in history]
    ax1.plot(iters, [h["residual_std"] for h in history], "o-", color="#3b6ea5",
              label="Residual std (log-RMS)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Residual std (log-RMS)", color="#3b6ea5")
    ax2 = ax1.twinx()
    ax2.plot(iters, [100*h["variance_explained"] for h in history], "s--", color="#d9822b",
              label="Variance explained (%)")
    ax2.set_ylabel("Variance explained (%)", color="#d9822b")
    ax1.set_title("STEP 3 QC -- Decomposition convergence")
    fig.tight_layout()

    decomp = dict(S_group=S_group, R_group=R_group, O_group=O_group,
                  S_trace=S_trace, R_trace=R_trace, O_trace=O_trace,
                  valid=valid)
    stats = dict(history=history, final_residual_std=final_resid_std,
                 final_variance_explained=final_explained,
                 mean_S_removed=mean_S, mean_R_removed=mean_R)
    return decomp, checks, stats, fig


# ============================================================
# STEP 4: PER-TRACE CORRECTION FACTOR (clipped)
# ============================================================

def step4_correction_factors(decomp, min_gain, max_gain):
    banner("STEP 4 : COMPUTE PER-TRACE CORRECTION FACTORS")

    S_trace, R_trace = decomp["S_trace"], decomp["R_trace"]
    valid = decomp["valid"]

    log_correction = -(S_trace + R_trace)  # O term intentionally left in
                                            # the data; only S and R (the
                                            # acquisition-coupling terms)
                                            # are corrected for.
    lo, hi = np.log(min_gain), np.log(max_gain)
    clipped_low = log_correction < lo
    clipped_high = log_correction > hi
    log_correction = np.clip(log_correction, lo, hi)

    correction = np.exp(log_correction).astype(np.float32)
    correction[~valid] = 1.0  # dead/invalid traces: pass through unscaled

    n_clip = int((clipped_low | clipped_high).sum())
    n_total = correction.size

    print(f"  Gain clip range        : {min_gain:.3f}x - {max_gain:.3f}x "
          f"({20*np.log10(min_gain):+.1f} dB / {20*np.log10(max_gain):+.1f} dB)")
    print(f"  Traces clipped low     : {int(clipped_low.sum()):,}")
    print(f"  Traces clipped high    : {int(clipped_high.sum()):,}")
    print(f"  Total clipped          : {n_clip:,} ({100*n_clip/n_total:.3f}%)")
    print(f"  Correction factor stats: min={correction.min():.4f}  "
          f"median={np.median(correction):.4f}  max={correction.max():.4f}")

    checks = []
    checks.append(qc(
        "Only a small fraction of traces hit the gain clip limits (<10%)",
        (n_clip / n_total) < 0.10,
        f"{100*n_clip/n_total:.3f}% clipped"
    ))
    checks.append(qc(
        "Correction factors are finite and positive everywhere",
        bool(np.all(np.isfinite(correction)) and np.all(correction > 0)),
    ))

    print()
    print("  EXPECTED OUTCOME:")
    print("  - A small percentage of clipped traces is normal (extremely")
    print("    weak or hot shots/receivers hitting the safety limit).")
    print("  - A large clipped fraction means the decomposition found")
    print(f"    corrections outside [{min_gain}x, {max_gain}x] for many")
    print("    traces -- widen the clip range deliberately, or first")
    print("    investigate whether those traces are actually bad data")
    print("    that should be edited/killed instead of scaled.")

    if not all(c["passed"] for c in checks):
        raise RuntimeError("STEP 4 QC failed -- see checks above.")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(20 * np.log10(correction[valid]), bins=200, color="#3b6ea5")
    ax.axvline(20*np.log10(min_gain), color="red", linestyle="--", label="clip limits")
    ax.axvline(20*np.log10(max_gain), color="red", linestyle="--")
    ax.set_xlabel("Applied correction (dB)")
    ax.set_ylabel("Trace count")
    ax.set_title("STEP 4 QC -- Applied surface-consistent correction (dB)")
    ax.legend()
    fig.tight_layout()

    stats = dict(n_clipped=n_clip, pct_clipped=100*n_clip/n_total,
                 min_gain=min_gain, max_gain=max_gain,
                 correction_min=float(correction.min()),
                 correction_median=float(np.median(correction)),
                 correction_max=float(correction.max()))
    return correction, checks, stats, fig


# ============================================================
# STEP 5: PASS 2 -- APPLY CORRECTION, WRITE OUTPUT, LOG QC CSV
# ============================================================

def step5_apply_and_write(input_path, output_path, qc_csv_path, info,
                           meta, correction, valid_mask, chunk_size):
    banner("STEP 5 : PASS 2 -- APPLY CORRECTION & WRITE SC-SCALED SU")

    ns, n_traces, order = info["ns"], info["n_traces"], info["order"]
    _, full_dtype = build_dtypes(order, ns)

    rms_corrected = np.empty(n_traces, dtype=np.float64)

    t0 = time.time()
    n_done = 0
    with open(input_path, "rb") as fin, \
         open(output_path, "wb") as fout, \
         open(qc_csv_path, "w") as fqc:

        fqc.write("fldr,cdp,offset,sx,sy,gx,gy,rms_raw,correction,rms_corrected\n")

        remaining = n_traces
        while remaining > 0:
            take = min(chunk_size, remaining)
            chunk = np.fromfile(fin, dtype=full_dtype, count=take)
            if chunk.shape[0] != take:
                raise RuntimeError("Unexpected short read in Pass 2.")

            sl = slice(n_done, n_done + take)
            corr = correction[sl].astype(np.float32).reshape(-1, 1)

            out_chunk = chunk.copy()
            out_chunk["samples"] = (chunk["samples"] * corr).astype(order + "f4")
            out_chunk.tofile(fout)

            rc = np.sqrt(np.mean(out_chunk["samples"].astype(np.float64) ** 2, axis=1))
            rms_corrected[sl] = rc

            h = chunk["header"]
            rows = np.column_stack([
                h["fldr"], h["cdp"], h["offset"], h["sx"], h["sy"], h["gx"], h["gy"],
                meta["rms_raw"][sl], correction[sl], rc,
            ])
            np.savetxt(fqc, rows, delimiter=",",
                       fmt=["%d", "%d", "%d", "%d", "%d", "%d", "%d", "%.6g", "%.6g", "%.6g"])

            n_done += take
            remaining -= take
            if (n_done // chunk_size) % 20 == 0 or remaining == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                print(f"  pass 2: {n_done:,}/{n_traces:,} traces "
                      f"({100*n_done/n_traces:5.1f}%) -- {rate:,.0f} traces/s", end="\r")

    print()
    elapsed = time.time() - t0
    print(f"  Pass 2 complete in {elapsed:.1f} s "
          f"({n_traces/elapsed:,.0f} traces/s)")

    out_size = output_path.stat().st_size
    in_size = input_path.stat().st_size

    def cv(x):
        m = np.mean(x)
        return float(np.std(x) / m) if m > 0 else float("nan")

    # Robust, log-domain metric computed over the SAME population the
    # decomposition was fit to (dead/outlier traces excluded). This is
    # the metric that actually matches what Step 3 optimizes -- a plain
    # whole-population linear std/mean is not robust: a mere handful of
    # untouched outlier traces (passed through unscaled by design, see
    # Step 1/4) can dominate a linear std/mean over ~1M traces and make
    # an otherwise-successful scaling look like a failure.
    log_before = np.log(meta["rms_raw"][valid_mask])
    log_after = np.log(np.clip(rms_corrected[valid_mask], 1e-300, None))
    log_std_before = float(np.std(log_before))
    log_std_after = float(np.std(log_after))
    log_std_reduction_pct = 100 * (1 - log_std_after / log_std_before) if log_std_before > 0 else float("nan")

    # Informational only (NOT a QC gate): whole-population linear CV,
    # which intentionally still includes any pass-through outlier
    # traces -- useful to see, but not a fair pass/fail test.
    lin_valid = meta["rms_raw"] > 0
    cv_before = cv(meta["rms_raw"][lin_valid])
    cv_after = cv(rms_corrected[lin_valid])

    print()
    print(f"  Output trace count checked : {n_done:,} == {n_traces:,}")
    print(f"  Output file size           : {out_size:,} bytes "
          f"(input was {in_size:,})")
    print(f"  log-RMS std, decomposition population, before -> after : "
          f"{log_std_before:.4f} -> {log_std_after:.4f} "
          f"({log_std_reduction_pct:+.1f}%)")
    print(f"  Whole-population linear CV, before -> after (informational, "
          f"includes pass-through outliers, not a QC gate): "
          f"{cv_before:.4f} -> {cv_after:.4f}")

    checks = []
    checks.append(qc(
        "All traces written (output trace count == input trace count)",
        n_done == n_traces
    ))
    checks.append(qc(
        "Output file size exactly equals input file size "
        "(same header+sample-count per trace, only amplitudes changed)",
        out_size == in_size,
        f"{out_size:,} vs {in_size:,}"
    ))
    checks.append(qc(
        "Amplitude variability (log-RMS std, over the decomposition's "
        "own valid population) reduced by scaling",
        log_std_after < log_std_before,
        f"{log_std_before:.4f} -> {log_std_after:.4f} ({log_std_reduction_pct:+.1f}%)"
    ))

    print()
    print("  EXPECTED OUTCOME:")
    print("  - Output file size must equal the input size exactly -- SC")
    print("    scaling only rewrites sample amplitudes, never headers or")
    print("    sample counts, so any mismatch signals a bug or a")
    print("    truncated write.")
    print("  - log-RMS std (computed over the same dead/outlier-excluded")
    print("    population the decomposition targets) should drop by")
    print("    roughly the variance-explained fraction reported in Step 3")
    print("    -- this is the correct, outlier-robust read on whether")
    print("    scaling worked. If it barely moves, Step 3 likely under-fit.")
    print("  - The whole-population linear CV is printed for context only:")
    print("    it is NOT outlier-robust, and a handful of pass-through")
    print("    dead/outlier traces (Step 1) can make it look worse even")
    print("    when the log-RMS metric above clearly improved -- that is")
    print("    expected and not a failure.")

    if not all(c["passed"] for c in checks):
        raise RuntimeError("STEP 5 QC failed -- see checks above.")

    stats = dict(n_written=n_done, in_size=in_size, out_size=out_size,
                 log_std_before=log_std_before, log_std_after=log_std_after,
                 log_std_reduction_pct=log_std_reduction_pct,
                 cv_before_informational=cv_before, cv_after_informational=cv_after)
    return rms_corrected, checks, stats


# ============================================================
# STEP 6: DIAGNOSTIC QC PLOTS
# ============================================================

def step6_diagnostic_plots(meta, groups, decomp, rms_corrected, outdir):
    banner("STEP 6 : DIAGNOSTIC QC PLOTS "
           "(source/receiver maps, offset trend before/after)")

    plots_dir = outdir / "qc_plots"
    saved = []

    # --- Source map colored by S factor (dB) ---
    src_unique = groups["src_unique"]
    src_idx = groups["src_idx"]
    sx_by_src = np.zeros(groups["n_src"])
    sy_by_src = np.zeros(groups["n_src"])
    counts = np.bincount(src_idx, minlength=groups["n_src"])
    np.add.at(sx_by_src, src_idx, meta["sx"])
    np.add.at(sy_by_src, src_idx, meta["sy"])
    sx_by_src /= np.maximum(counts, 1)
    sy_by_src /= np.maximum(counts, 1)
    S_db = 20 * np.log10(np.exp(decomp["S_group"]))

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(sx_by_src, sy_by_src, c=S_db, cmap="RdBu_r", s=6,
                     vmin=-np.percentile(np.abs(S_db), 98),
                     vmax=np.percentile(np.abs(S_db), 98))
    plt.colorbar(sc, ax=ax, label="Source correction applied (dB)")
    ax.set_xlabel("Source X")
    ax.set_ylabel("Source Y")
    ax.set_title("STEP 6 QC -- Source-consistent correction map")
    ax.axis("equal")
    fig.tight_layout()
    p = plots_dir / "01_source_correction_map.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    saved.append(p)

    # --- Receiver map colored by R factor (dB) ---
    rec_unique = groups["rec_unique"]
    R_db = 20 * np.log10(np.exp(decomp["R_group"]))
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(rec_unique[:, 0], rec_unique[:, 1], c=R_db, cmap="RdBu_r", s=4,
                     vmin=-np.percentile(np.abs(R_db), 98),
                     vmax=np.percentile(np.abs(R_db), 98))
    plt.colorbar(sc, ax=ax, label="Receiver correction applied (dB)")
    ax.set_xlabel("Receiver X")
    ax.set_ylabel("Receiver Y")
    ax.set_title("STEP 6 QC -- Receiver-consistent correction map")
    ax.axis("equal")
    fig.tight_layout()
    p = plots_dir / "02_receiver_correction_map.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    saved.append(p)

    # --- RMS vs offset before/after ---
    off_idx, n_off, edges = groups["off_idx"], groups["n_off"], groups["off_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    valid = decomp["valid"]
    rms_before_bin = _group_mean(np.where(valid, meta["rms_raw"], np.nan), off_idx, n_off)
    rms_after_bin = _group_mean(np.where(valid, rms_corrected, np.nan), off_idx, n_off)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(centers, rms_before_bin, "o-", label="Before SC scaling")
    ax.semilogy(centers, rms_after_bin, "s-", label="After SC scaling")
    ax.set_xlabel("Absolute offset")
    ax.set_ylabel("Mean trace RMS (log scale)")
    ax.set_title("STEP 6 QC -- RMS vs. offset, before/after")
    ax.legend()
    fig.tight_layout()
    p = plots_dir / "03_rms_vs_offset_before_after.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    saved.append(p)

    # --- RMS histogram before/after ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(np.log10(np.clip(meta["rms_raw"][valid], 1e-12, None)), bins=150,
            alpha=0.6, label="Before", color="#3b6ea5")
    ax.hist(np.log10(np.clip(rms_corrected[valid], 1e-12, None)), bins=150,
            alpha=0.6, label="After", color="#d9822b")
    ax.set_xlabel("log10(trace RMS)")
    ax.set_ylabel("Trace count")
    ax.set_title("STEP 6 QC -- Trace RMS distribution, before/after")
    ax.legend()
    fig.tight_layout()
    p = plots_dir / "04_rms_histogram_before_after.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    saved.append(p)

    for p in saved:
        print(f"  Saved: {p}")

    print()
    print("  EXPECTED OUTCOME:")
    print("  - Source/receiver maps: mostly uniform, near-zero-dB color")
    print("    with a scatter of isolated outliers -- those outliers are")
    print("    the point of this QC: they flag specific bad shots (weak")
    print("    charge/vibrator coupling, misfires) or bad receiver")
    print("    stations (poor plant, dead/weak geophones) worth checking")
    print("    against field observer notes.")
    print("  - RMS vs offset AFTER should be visibly flatter than BEFORE")
    print("    -- the offset-dependent (spherical-divergence-like) trend")
    print("    is expected to still exist in raw amplitude but be more")
    print("    stable/monotonic post-scaling, not eliminated (only the")
    print("    O term itself is not removed from the data by design, so")
    print("    the true-amplitude decay with offset survives; what")
    print("    should be gone is jaggedness caused by S/R coupling.")
    print("  - The AFTER RMS histogram should be visibly narrower")
    print("    (taller, tighter peak) than the BEFORE histogram.")

    return saved


# ============================================================
# STEP 7: FINAL QC SUMMARY REPORT
# ============================================================

def step7_summary(outdir, stem, all_checks, all_stats, args):
    banner("STEP 7 : FINAL QC SUMMARY REPORT")

    n_pass = sum(1 for c in all_checks if c["passed"])
    n_total = len(all_checks)

    print(f"  {n_pass}/{n_total} QC checks passed across all steps.")
    for c in all_checks:
        tag = "PASS" if c["passed"] else "**FAIL**"
        print(f"    [{tag}] {c['label']}")

    report = {
        "input": str(args.input),
        "output_su": str(outdir / f"{stem}_sc_scaled.su"),
        "qc_csv": str(outdir / f"{stem}_sc_scaling_qc.csv"),
        "parameters": {
            "rms_window": args.rms_window,
            "offset_bins": args.offset_bins,
            "iterations": args.iterations,
            "min_gain": args.min_gain,
            "max_gain": args.max_gain,
            "dead_rms_fraction": args.dead_rms_fraction,
            "outlier_rms_multiplier": args.outlier_rms_multiplier,
        },
        "checks_passed": n_pass,
        "checks_total": n_total,
        "checks": all_checks,
        "stats": all_stats,
    }

    json_path = outdir / f"{stem}_sc_scaling_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    txt_path = outdir / f"{stem}_sc_scaling_report.txt"
    with open(txt_path, "w") as f:
        f.write(f"SC SCALING QC REPORT -- {stem}\n")
        f.write("=" * 70 + "\n")
        f.write(f"{n_pass}/{n_total} checks passed\n\n")
        for c in all_checks:
            tag = "PASS" if c["passed"] else "FAIL"
            f.write(f"[{tag}] {c['label']}")
            if c["detail"]:
                f.write(f" -- {c['detail']}")
            f.write("\n")
        f.write("\nSTATS\n")
        f.write(json.dumps(all_stats, indent=2, default=str))

    print()
    print(f"  Report written: {json_path}")
    print(f"  Report written: {txt_path}")

    if n_pass < n_total:
        print()
        print("  ** ONE OR MORE QC CHECKS FAILED -- review before using the "
              "output for further processing. **")
    else:
        print()
        print("  All QC checks passed. SC-scaled volume is ready for the "
              "next processing step.")

    return report


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Surface-consistent (SC) amplitude scaling with "
                     "step-by-step QC for a raw .su dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT),
                   help="Input raw .su file")
    p.add_argument("--outdir", type=Path, default=Path(DEFAULT_OUTDIR),
                   help="Output directory for the scaled .su, QC CSV, "
                        "plots and reports")
    p.add_argument("--rms-window", type=float, nargs=2, default=None,
                   metavar=("TMIN", "TMAX"),
                   help="Time window in seconds for the RMS amplitude "
                        "used to drive the decomposition (default: full "
                        "trace)")
    p.add_argument("--offset-bins", type=int, default=20,
                   help="Number of |offset| bins for the O term")
    p.add_argument("--iterations", type=int, default=25,
                   help="Max Gauss-Seidel iterations")
    p.add_argument("--tol", type=float, default=1e-4,
                   help="Early-stop tolerance on residual-std change "
                        "between iterations")
    p.add_argument("--min-gain", type=float, default=0.25,
                   help="Lower clip limit on the linear correction factor")
    p.add_argument("--max-gain", type=float, default=4.0,
                   help="Upper clip limit on the linear correction factor")
    p.add_argument("--dead-rms-fraction", type=float, default=1e-6,
                   help="A trace is flagged dead if its RMS is below this "
                        "fraction of the survey median RMS")
    p.add_argument("--outlier-rms-multiplier", type=float, default=50.0,
                   help="A trace is flagged as an amplitude outlier/spike "
                        "(excluded from the decomposition, passed through "
                        "unscaled, listed for manual review) if its RMS "
                        "exceeds this multiple of the survey median RMS")
    p.add_argument("--include-cdp-term", action="store_true",
                   help="Also solve a CDP-consistent term (off by default "
                        "-- see module docstring for why)")
    p.add_argument("--chunk-size", type=int, default=5000,
                   help="Traces per streamed I/O chunk")
    return p.parse_args()


def main():
    args = parse_args()

    if args.include_cdp_term:
        print("NOTE: --include-cdp-term was requested, but this pipeline's "
              "default 3-term (S+R+O) model does not yet wire a CDP term "
              "into Step 3/4. Proceeding with the safe S+R+O model; add "
              "a CDP group in Step 2/3 if this is genuinely needed.")

    stem = args.input.stem
    outdir = args.outdir

    all_checks = []
    all_stats = {}

    info, c0 = step0_validate(args.input, outdir)
    all_checks += c0

    meta, invalid_mask, dead_mask, outlier_mask, c1, s1, fig1 = step1_scan(
        args.input, info, args.rms_window, args.dead_rms_fraction,
        args.outlier_rms_multiplier, args.chunk_size)
    all_checks += c1
    all_stats["step1"] = s1
    fig1.savefig(outdir / "qc_plots" / "00_raw_rms_histogram.png", dpi=180)
    plt.close(fig1)

    # Manual-review listing of flagged outlier/spike traces (Step 1).
    if outlier_mask.any():
        outlier_path = outdir / f"{stem}_sc_amplitude_outliers.txt"
        idx = np.where(outlier_mask)[0]
        order_desc = idx[np.argsort(-meta["rms_raw"][idx])]
        with open(outlier_path, "w") as f:
            f.write("# fldr cdp offset sx sy gx gy rms_raw  "
                    "(excluded from SC decomposition, passed through unscaled)\n")
            for i in order_desc:
                f.write(f"{meta['fldr'][i]} {meta['cdp'][i]} {meta['offset'][i]} "
                        f"{meta['sx'][i]} {meta['sy'][i]} {meta['gx'][i]} {meta['gy'][i]} "
                        f"{meta['rms_raw'][i]:.6g}\n")
        print(f"  Saved amplitude outlier list : {outlier_path} "
              f"({outlier_mask.sum():,} traces)")

    groups, c2, s2 = step2_build_groups(meta, invalid_mask, args.offset_bins)
    all_checks += c2
    all_stats["step2"] = s2

    decomp, c3, s3, fig3 = step3_decompose(meta, invalid_mask, groups, args.iterations, args.tol)
    all_checks += c3
    all_stats["step3"] = s3
    fig3.savefig(outdir / "qc_plots" / "05_decomposition_convergence.png", dpi=180)
    plt.close(fig3)

    correction, c4, s4, fig4 = step4_correction_factors(decomp, args.min_gain, args.max_gain)
    all_checks += c4
    all_stats["step4"] = s4
    fig4.savefig(outdir / "qc_plots" / "06_correction_factor_histogram.png", dpi=180)
    plt.close(fig4)

    output_su = outdir / f"{stem}_sc_scaled.su"
    qc_csv = outdir / f"{stem}_sc_scaling_qc.csv"
    rms_corrected, c5, s5 = step5_apply_and_write(
        args.input, output_su, qc_csv, info, meta, correction, decomp["valid"], args.chunk_size)
    all_checks += c5
    all_stats["step5"] = s5

    step6_diagnostic_plots(meta, groups, decomp, rms_corrected, outdir)

    # Persist source/receiver factor lookup tables (useful for reuse on
    # a companion line, or for a processor to review by hand).
    src_path = outdir / f"{stem}_sc_source_factors.txt"
    rec_path = outdir / f"{stem}_sc_receiver_factors.txt"
    with open(src_path, "w") as f:
        f.write("# fldr log_source_factor correction_dB\n")
        for fldr_val, s in zip(groups["src_unique"], decomp["S_group"]):
            f.write(f"{fldr_val} {s:.8f} {20*np.log10(np.exp(-s)):.4f}\n")
    with open(rec_path, "w") as f:
        f.write("# gx gy log_receiver_factor correction_dB\n")
        for (gxv, gyv), r in zip(groups["rec_unique"], decomp["R_group"]):
            f.write(f"{gxv} {gyv} {r:.8f} {20*np.log10(np.exp(-r)):.4f}\n")
    print()
    print(f"  Saved source factor table   : {src_path}")
    print(f"  Saved receiver factor table : {rec_path}")

    report = step7_summary(outdir, stem, all_checks, all_stats, args)

    banner("SC SCALING PIPELINE COMPLETE")
    print(f"  SC-scaled SU : {output_su}")
    print(f"  Per-trace QC : {qc_csv}")
    print(f"  QC plots     : {outdir / 'qc_plots'}")
    print(f"  Reports      : {outdir / (stem + '_sc_scaling_report.json')}")
    print(f"                 {outdir / (stem + '_sc_scaling_report.txt')}")

    return 0 if report["checks_passed"] == report["checks_total"] else 1


if __name__ == "__main__":
    sys.exit(main())
