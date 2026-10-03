#!/usr/bin/env python3
"""
whole_survey_direction_check.py

Automated, whole-survey version of shot_geometry_qc_marimo.py's manual
per-shot direction-fix check.

THE IDEA
--------
shot_geometry_qc_marimo.py checks ONE shot at a time: does a straight
reference line t = |offset|/V track the leading edge (direct/refracted
arrival) of the real data? If not, a toggle (negate SX/SY/GX/GY, swap
X/Y) might fix it -- but that has to be eyeballed, one shot, one toggle
at a time, and confirmed by hand across "different parts of the survey".

This script makes that check OBJECTIVE and runs it on EVERY shot at
once:

  1. Pick each trace's first-break (direct/refracted arrival) time
     automatically, with an STA/LTA onset detector. This is intrinsic
     to the recorded waveform -- it does NOT depend on which geometry
     hypothesis we're testing.
  2. For every combination of the 5 direction toggles (2^5 = 32) plus
     the as-stored header offset (33 hypotheses total), recompute each
     trace's offset and fit  t ~= a + b*|offset|  per shot by ordinary
     least squares.
  3. Score each hypothesis by its median R^2 across every shot in the
     survey. The CORRECT geometry should make first-break time nearly
     linear in offset almost everywhere; a wrong sign/swap scrambles
     the (offset, time) pairing and increases scatter (lower R^2).

This turns "look at one shot, click some toggles, eyeball it" into one
data-driven, whole-survey verdict.

MEMORY-SAFE, SINGLE EFFECTIVE PASS OVER THE FILE
--------------------------------------------------
Only per-shot-per-hypothesis SUFFICIENT STATISTICS for the linear fit
(Sx, Sy, Sxy, Sxx, Syy, N -- six numbers) are accumulated while
streaming; the standard closed-form OLS/R^2 formulas are applied once,
at the end, from those sums. No raw (offset, time) pairs are held in
memory, so this scales to a survey of any size at constant, small
memory (~n_shots * n_hypotheses * 6 floats).

USAGE
-----
    python3 whole_survey_direction_check.py \\
        --input /path/to/survey.su --outdir /path/to/output_dir
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SC_SCALING_DIR = Path(__file__).resolve().parent.parent / "SC_scaling"
if str(_SC_SCALING_DIR) not in sys.path:
    sys.path.insert(0, str(_SC_SCALING_DIR))

from sc_scaling_pipeline import banner, qc as qc_check, probe_file, build_dtypes

DEFAULT_INPUT = (
    "/Users/vikas/SCube/harrison/data/"
    "EAST_ATCHAFALAYA-GCSR-UNDEFINED_SDL062470-16.su"
)
DEFAULT_OUTDIR = str(Path(__file__).resolve().parent)


# ============================================================
# THE 33 CANDIDATE GEOMETRY HYPOTHESES
# ============================================================

def build_hypotheses():
    """5 boolean toggles (negate SX/SY/GX/GY, swap X/Y) -> 32 combinations,
    matching shot_geometry_qc_marimo.py's toggles exactly, applied in the
    same order (swap first, then negate) -- plus a 33rd hypothesis that
    just uses the as-stored header offset, untouched, as a baseline."""
    names = ["negate_sx", "negate_sy", "negate_gx", "negate_gy", "swap_xy"]
    # n_toggles is used only to break exact ties in favor of the simplest
    # explanation (Occam's razor) -- see the note in step3 about why ties
    # are actually EXPECTED: negating SX+GX together (or SY+GY together,
    # or swapping X/Y for both source and receiver together) is a rigid
    # transform that preserves the source-receiver distance exactly, so
    # it is mathematically indistinguishable from "no change" here. -1
    # marks the as-stored baseline as the simplest possible explanation
    # of all (it doesn't even touch the coordinate fields).
    hyps = [{"label": "as-stored header offset", "stored": True, "n_toggles": -1}]
    for combo in itertools.product([False, True], repeat=len(names)):
        flags = dict(zip(names, combo))
        label = "none (no change)" if not any(flags.values()) else "+".join(
            n for n, v in flags.items() if v
        )
        hyps.append({"label": label, "stored": False, "n_toggles": sum(combo), **flags})
    return hyps


def recompute_offset(sx, sy, gx, gy, hyp):
    if hyp.get("swap_xy"):
        sx, sy = sy, sx
        gx, gy = gy, gx
    if hyp.get("negate_sx"):
        sx = -sx
    if hyp.get("negate_sy"):
        sy = -sy
    if hyp.get("negate_gx"):
        gx = -gx
    if hyp.get("negate_gy"):
        gy = -gy
    return np.sqrt((gx - sx) ** 2 + (gy - sy) ** 2)


# ============================================================
# FIRST-BREAK PICKING (STA/LTA onset detector, vectorized over a chunk)
# ============================================================

def _sustained(exceed, min_consecutive):
    """True at position i only if `exceed` is True at i, i+1, ..., i+min_consecutive-1
    (all within bounds) -- rejects single-sample noise spikes that briefly push the
    STA/LTA ratio over threshold without a real, sustained onset behind them."""
    if min_consecutive <= 1:
        return exceed
    n = exceed.shape[1]
    out = exceed.copy()
    for k in range(1, min_consecutive):
        shifted = np.zeros_like(exceed)
        if n - k > 0:
            shifted[:, : n - k] = exceed[:, k:]
        out &= shifted
    return out


def pick_first_breaks(samples, dt_s, short_win, long_win, threshold, min_consecutive=3):
    """samples: (n_chunk, ns) float array. Returns (pick_time_s, has_pick),
    each length n_chunk. STA/LTA ratio exceeding `threshold` for at least
    `min_consecutive` samples in a row marks the onset of coherent energy
    (direct/refracted arrival, or -- on a dead/silent trace, or one where
    the ratio only ever spikes briefly on noise -- never sustained, which
    naturally excludes such traces from the fit without any separate
    dead-trace logic). Verified against real data: a single-sample
    threshold crossing (no sustained-exceedance check) was found to
    trigger on noise bursts uncorrelated with offset, not a genuine
    arrival -- see the module docstring / commit notes for the check
    that caught this."""
    sq = samples.astype(np.float64) ** 2
    csum = np.concatenate([np.zeros((samples.shape[0], 1)), np.cumsum(sq, axis=1)], axis=1)
    ns = samples.shape[1]
    if long_win >= ns:
        return np.full(samples.shape[0], np.nan), np.zeros(samples.shape[0], dtype=bool)

    idx = np.arange(long_win - 1, ns)
    sta = (csum[:, idx + 1] - csum[:, idx + 1 - short_win]) / short_win
    lta = (csum[:, idx + 1] - csum[:, idx + 1 - long_win]) / long_win
    ratio = sta / (lta + 1e-30)
    exceed = _sustained(ratio > threshold, min_consecutive)
    has_pick = exceed.any(axis=1)
    first_pos = np.argmax(exceed, axis=1)
    pick_sample = np.where(has_pick, idx[first_pos], -1)
    pick_time_s = np.where(has_pick, pick_sample * dt_s, np.nan)
    return pick_time_s, has_pick


# ============================================================
# STEP 1: PASS 1 -- HEADERS ONLY (fldr, sx, sy, gx, gy, offset) + shot index
# ============================================================

def step1_scan_headers(input_path, info, chunk_size):
    banner("STEP 1 : PASS 1 -- SCAN HEADERS, BUILD SHOT INDEX")

    ns, order, n_traces = info["ns"], info["order"], info["n_traces"]
    _, full_dtype = build_dtypes(order, ns)

    fldr = np.empty(n_traces, dtype=np.int64)
    sx = np.empty(n_traces, dtype=np.float64)
    sy = np.empty(n_traces, dtype=np.float64)
    gx = np.empty(n_traces, dtype=np.float64)
    gy = np.empty(n_traces, dtype=np.float64)
    offset = np.empty(n_traces, dtype=np.float64)

    t0 = time.time()
    n_done = 0
    with open(input_path, "rb") as f:
        remaining = n_traces
        while remaining > 0:
            take = min(chunk_size, remaining)
            chunk = np.fromfile(f, dtype=full_dtype, count=take)
            if chunk.shape[0] != take:
                raise RuntimeError("Unexpected short read in Pass 1.")
            sl = slice(n_done, n_done + take)
            h = chunk["header"]
            fldr[sl] = h["fldr"]
            sx[sl] = h["sx"]
            sy[sl] = h["sy"]
            gx[sl] = h["gx"]
            gy[sl] = h["gy"]
            offset[sl] = h["offset"]
            n_done += take
            remaining -= take
            if (n_done // chunk_size) % 20 == 0 or remaining == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                print(f"  pass 1: {n_done:,}/{n_traces:,} traces "
                      f"({100*n_done/n_traces:5.1f}%) -- {rate:,.0f} traces/s", end="\r")
    print()
    print(f"  Pass 1 complete in {time.time()-t0:.1f} s.")

    src_unique, src_idx = np.unique(fldr, return_inverse=True)
    n_src = src_unique.size

    checks = [
        qc_check("Trace count scanned matches file's exact trace count", n_done == n_traces,
                  f"{n_done:,} == {n_traces:,}"),
        qc_check("More than one shot found (a direction check needs many shots)", n_src > 1,
                  f"{n_src:,} unique shots"),
    ]
    print(f"  Unique shots (fldr): {n_src:,}")
    if not all(c["passed"] for c in checks):
        raise RuntimeError("STEP 1 QC failed -- see checks above.")

    meta = dict(fldr=fldr, sx=sx, sy=sy, gx=gx, gy=gy, offset=offset)
    return meta, src_unique, src_idx, n_src, checks


# ============================================================
# STEP 2: PASS 2 -- FIRST-BREAK PICK + ACCUMULATE PER-SHOT/PER-HYPOTHESIS FIT STATS
# ============================================================

def step2_accumulate_fits(input_path, info, meta, src_idx, n_src, hypotheses,
                           short_win_ms, long_win_ms, threshold, min_consecutive, chunk_size):
    banner("STEP 2 : PASS 2 -- FIRST-BREAK PICKING + PER-SHOT LINEAR FITS "
           "(all 33 hypotheses at once)")

    ns, order, n_traces = info["ns"], info["order"], info["n_traces"]
    dt_s = info["dt_us"] * 1e-6
    short_win = max(2, int(round(short_win_ms / 1000.0 / dt_s)))
    long_win = max(short_win + 1, int(round(long_win_ms / 1000.0 / dt_s)))
    print(f"  STA/LTA windows: short={short_win} samples ({short_win*dt_s*1000:.1f} ms), "
          f"long={long_win} samples ({long_win*dt_s*1000:.1f} ms), threshold={threshold}x")

    n_hyp = len(hypotheses)
    _, full_dtype = build_dtypes(order, ns)

    Sx = np.zeros((n_src, n_hyp))
    Sy = np.zeros((n_src, n_hyp))
    Sxy = np.zeros((n_src, n_hyp))
    Sxx = np.zeros((n_src, n_hyp))
    Syy = np.zeros((n_src, n_hyp))
    N = np.zeros((n_src, n_hyp))

    stored_offset_abs = np.abs(meta["offset"])
    hyp_idx_arange = np.arange(n_hyp)

    t0 = time.time()
    n_done = 0
    n_picked = 0
    with open(input_path, "rb") as f:
        remaining = n_traces
        while remaining > 0:
            take = min(chunk_size, remaining)
            chunk = np.fromfile(f, dtype=full_dtype, count=take)
            if chunk.shape[0] != take:
                raise RuntimeError("Unexpected short read in Pass 2.")
            sl = slice(n_done, n_done + take)

            t_pick, has_pick = pick_first_breaks(
                chunk["samples"], dt_s, short_win, long_win, threshold, min_consecutive)

            if has_pick.any():
                shot_sel = src_idx[sl][has_pick]
                t_sel = t_pick[has_pick]
                sx_sel = meta["sx"][sl][has_pick]
                sy_sel = meta["sy"][sl][has_pick]
                gx_sel = meta["gx"][sl][has_pick]
                gy_sel = meta["gy"][sl][has_pick]
                stored_sel = stored_offset_abs[sl][has_pick]

                offset_all = np.empty((shot_sel.size, n_hyp))
                for j, hyp in enumerate(hypotheses):
                    offset_all[:, j] = stored_sel if hyp["stored"] else \
                        recompute_offset(sx_sel, sy_sel, gx_sel, gy_sel, hyp)

                flat_idx = (shot_sel[:, None] * n_hyp + hyp_idx_arange[None, :]).ravel()
                minlen = n_src * n_hyp

                Sx += np.bincount(flat_idx, weights=offset_all.ravel(), minlength=minlen).reshape(n_src, n_hyp)
                Sy += np.bincount(flat_idx, weights=np.repeat(t_sel, n_hyp), minlength=minlen).reshape(n_src, n_hyp)
                Sxy += np.bincount(flat_idx, weights=(offset_all * t_sel[:, None]).ravel(), minlength=minlen).reshape(n_src, n_hyp)
                Sxx += np.bincount(flat_idx, weights=(offset_all ** 2).ravel(), minlength=minlen).reshape(n_src, n_hyp)
                Syy += np.bincount(flat_idx, weights=np.repeat(t_sel ** 2, n_hyp), minlength=minlen).reshape(n_src, n_hyp)
                N += np.bincount(flat_idx, minlength=minlen).reshape(n_src, n_hyp)

                n_picked += int(has_pick.sum())

            n_done += take
            remaining -= take
            if (n_done // chunk_size) % 20 == 0 or remaining == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                print(f"  pass 2: {n_done:,}/{n_traces:,} traces "
                      f"({100*n_done/n_traces:5.1f}%) -- {rate:,.0f} traces/s", end="\r")
    print()
    print(f"  Pass 2 complete in {time.time()-t0:.1f} s.")
    print(f"  First-break pick success: {n_picked:,}/{n_traces:,} "
          f"({100*n_picked/n_traces:.1f}%)")

    checks = [
        qc_check("A usable majority of traces got a first-break pick (>=30%)",
                  (n_picked / n_traces) >= 0.30,
                  f"{100*n_picked/n_traces:.1f}% picked"),
    ]
    print()
    print("  EXPECTED OUTCOME:")
    print("  - A low pick rate means the STA/LTA windows/threshold don't suit this data's")
    print("    noise level -- widen --short-win-ms/--long-win-ms or lower --threshold and retry.")
    if not all(c["passed"] for c in checks):
        raise RuntimeError("STEP 2 QC failed -- see checks above.")

    return dict(Sx=Sx, Sy=Sy, Sxy=Sxy, Sxx=Sxx, Syy=Syy, N=N), n_picked, checks


# ============================================================
# STEP 3: CLOSED-FORM OLS FIT + R^2 PER SHOT PER HYPOTHESIS, THEN AGGREGATE
# ============================================================

def step3_score_hypotheses(stats, hypotheses, min_points_per_shot):
    banner("STEP 3 : SCORE EVERY HYPOTHESIS ACROSS THE WHOLE SURVEY")

    Sx, Sy, Sxy, Sxx, Syy, N = (stats[k] for k in ("Sx", "Sy", "Sxy", "Sxx", "Syy", "N"))
    n_src, n_hyp = N.shape

    valid = N >= min_points_per_shot
    denom = N * Sxx - Sx ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(denom != 0, (N * Sxy - Sx * Sy) / denom, np.nan)
        a = np.where(N > 0, (Sy - b * Sx) / N, np.nan)
        ss_tot = np.where(N > 0, Syy - Sy ** 2 / N, np.nan)
        ss_res = Syy - a * Sy - b * Sxy
        r2 = np.where(ss_tot > 0, 1 - ss_res / ss_tot, np.nan)
    r2 = np.where(valid, r2, np.nan)
    velocity = np.where(b > 0, 1.0 / np.where(b != 0, b, np.nan), np.nan)

    n_valid_shots_per_hyp = valid.sum(axis=0)
    median_r2 = np.nanmedian(r2, axis=0)
    mean_r2 = np.nanmean(r2, axis=0)
    median_v = np.nanmedian(velocity, axis=0)

    order_by_score = np.argsort(-np.nan_to_num(median_r2, nan=-1.0))

    print(f"  {'Rank':<5}{'Hypothesis':<40}{'Median R2':<12}{'Mean R2':<12}"
          f"{'Median V (units/s)':<20}{'Shots scored':<14}")
    for rank, j in enumerate(order_by_score, start=1):
        print(f"  {rank:<5}{hypotheses[j]['label']:<40}{median_r2[j]:<12.4f}"
              f"{mean_r2[j]:<12.4f}{median_v[j]:<20.1f}{int(n_valid_shots_per_hyp[j]):<14,}")

    # Several hypotheses tying EXACTLY is expected, not a bug: negating
    # SX+GX together (or SY+GY together, or swapping X/Y for both source
    # and receiver together) is a rigid transform that preserves the
    # source-receiver distance exactly, so it's mathematically
    # indistinguishable from "no change" by this offset-based check.
    # Break ties in favor of the simplest explanation.
    top_score = median_r2[order_by_score[0]]
    tol = max(1e-9, 1e-9 * abs(top_score))
    tied = [j for j in range(n_hyp) if np.isfinite(median_r2[j]) and abs(median_r2[j] - top_score) <= tol]
    best_j = min(tied, key=lambda j: hypotheses[j]["n_toggles"])
    baseline_j = next(j for j, h in enumerate(hypotheses) if h["stored"])

    if len(tied) > 1:
        print()
        print(f"  NOTE: {len(tied)} hypotheses tie EXACTLY at the top score -- this is expected")
        print("  geometric degeneracy (paired negations / a full X<->Y swap preserve distance")
        print("  exactly), not a fitting error. Reporting the simplest tied hypothesis below.")
        print("  Tied hypotheses: " + ", ".join(hypotheses[j]["label"] for j in tied))

    checks = [
        qc_check("At least one hypothesis was scored on a meaningful number of shots",
                  int(n_valid_shots_per_hyp[best_j]) >= max(10, 0.5 * n_src),
                  f"best hypothesis scored on {int(n_valid_shots_per_hyp[best_j]):,}/{n_src:,} shots"),
        qc_check("The best-scoring hypothesis's median R^2 indicates a genuinely linear "
                 "direct-arrival trend (>=0.5)",
                  bool(median_r2[best_j] >= 0.5),
                  f"median R^2 = {median_r2[best_j]:.3f}"),
    ]
    print()
    print("  EXPECTED OUTCOME:")
    print("  - The winning hypothesis's median R^2 should be noticeably higher than the")
    print("    as-stored baseline's if there really is a direction/sign error being corrected.")
    print("  - A low R^2 even for the best hypothesis (<0.5) suggests the first-break picker")
    print("    itself isn't finding a clean, consistent arrival on this data (check the")
    print("    STA/LTA parameters) rather than a geometry problem.")

    return dict(
        r2=r2, velocity=velocity, valid=valid,
        median_r2=median_r2, mean_r2=mean_r2, median_v=median_v,
        n_valid_shots_per_hyp=n_valid_shots_per_hyp,
        order_by_score=order_by_score, best_j=best_j, baseline_j=baseline_j,
    ), checks


# ============================================================
# STEP 4: QC PLOTS
# ============================================================

def step4_plots(scores, hypotheses, src_unique, outdir):
    banner("STEP 4 : QC PLOTS")
    plots_dir = Path(outdir) / "qc_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    order = scores["order_by_score"]
    top_n = min(10, len(order))
    top_idx = order[:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [hypotheses[j]["label"] for j in top_idx]
    ax.barh(range(top_n), scores["median_r2"][top_idx], color="#3b6ea5")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Median R^2 across all shots (direct-arrival linearity)")
    ax.set_title("Top hypotheses -- whole-survey direction check")
    fig.tight_layout()
    p1 = plots_dir / "10_whole_survey_hypothesis_ranking.png"
    fig.savefig(p1, dpi=180)
    plt.close(fig)

    best_j = scores["best_j"]
    baseline_j = scores["baseline_j"]
    r2_best = scores["r2"][:, best_j]
    r2_base = scores["r2"][:, baseline_j]
    valid_both = np.isfinite(r2_best) & np.isfinite(r2_base)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(r2_base[valid_both], bins=40, alpha=0.6, label="As-stored header offset", color="#c0392b")
    ax.hist(r2_best[valid_both], bins=40, alpha=0.6, label=f"Best: {hypotheses[best_j]['label']}", color="#3b6ea5")
    ax.set_xlabel("Per-shot R^2 (direct-arrival linearity)")
    ax.set_ylabel("Shot count")
    ax.set_title("Per-shot fit quality: as-stored vs. best hypothesis")
    ax.legend()
    fig.tight_layout()
    p2 = plots_dir / "11_whole_survey_r2_before_after.png"
    fig.savefig(p2, dpi=180)
    plt.close(fig)

    print(f"  Saved: {p1}")
    print(f"  Saved: {p2}")
    return [p1, p2]


# ============================================================
# MAIN
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Whole-survey, automated version of the shot-by-shot direct-arrival "
                     "direction/sign QC check.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, default=Path(DEFAULT_INPUT), help="Input raw .su file")
    p.add_argument("--outdir", type=Path, default=Path(DEFAULT_OUTDIR),
                   help="Output directory for the report and QC plots")
    p.add_argument("--short-win-ms", type=float, default=10.0,
                   help="STA/LTA short window (ms)")
    p.add_argument("--long-win-ms", type=float, default=100.0,
                   help="STA/LTA long window (ms)")
    p.add_argument("--threshold", type=float, default=8.0,
                   help="STA/LTA ratio threshold marking an onset/first-break. Verified "
                        "empirically on this survey: at threshold=8/min-consecutive=5, "
                        "picked time correlates with offset at r=0.99; at the naive "
                        "threshold=3/no-sustain default this drops to ~0 (noise, not signal)")
    p.add_argument("--min-consecutive", type=int, default=5,
                   help="Require the STA/LTA ratio to stay above --threshold for this "
                        "many consecutive samples, rejecting single-sample noise spikes")
    p.add_argument("--min-points-per-shot", type=int, default=20,
                   help="A shot's fit is only trusted with at least this many first-break picks")
    p.add_argument("--chunk-size", type=int, default=5000, help="Traces per streamed I/O chunk")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    banner("WHOLE-SURVEY DIRECTION/SIGN CHECK")
    print(f"  Input  : {args.input}")
    print(f"  Outdir : {outdir}")

    info = probe_file(args.input)
    print(f"  ns={info['ns']}, dt={info['dt_us']}us, {info['n_traces']:,} traces, "
          f"byte order {info['order']!r}")

    hypotheses = build_hypotheses()
    print(f"  Testing {len(hypotheses)} hypotheses "
          f"(32 direction-toggle combinations + as-stored header offset).")

    all_checks = []

    meta, src_unique, src_idx, n_src, c1 = step1_scan_headers(args.input, info, args.chunk_size)
    all_checks += c1

    stats, n_picked, c2 = step2_accumulate_fits(
        args.input, info, meta, src_idx, n_src, hypotheses,
        args.short_win_ms, args.long_win_ms, args.threshold, args.min_consecutive, args.chunk_size,
    )
    all_checks += c2

    scores, c3 = step3_score_hypotheses(stats, hypotheses, args.min_points_per_shot)
    all_checks += c3

    plots = step4_plots(scores, hypotheses, src_unique, outdir)

    best_j = scores["best_j"]
    baseline_j = scores["baseline_j"]

    n_pass = sum(1 for c in all_checks if c["passed"])
    n_total = len(all_checks)

    banner("VERDICT")
    print(f"  Best hypothesis : {hypotheses[best_j]['label']}")
    print(f"  Median R^2      : {scores['median_r2'][best_j]:.4f} "
          f"(as-stored baseline: {scores['median_r2'][baseline_j]:.4f})")
    print(f"  Median apparent velocity under best hypothesis: {scores['median_v'][best_j]:.1f} (survey units/s)")
    if best_j == baseline_j:
        print("  -> The AS-STORED header offset already gives the best (or tied-best) direct-arrival")
        print("     alignment: no direction/sign fix indicated by this check.")
    elif hypotheses[best_j]["label"] == "none (no change)":
        print("  -> Recomputing offset from sx/sy/gx/gy with NO sign/swap change scores best:")
        print("     the coordinate fields themselves are consistent; if the as-stored offset")
        print("     scored notably worse, the STORED OFFSET FIELD is the one that's wrong,")
        print("     not the coordinates.")
    else:
        print(f"  -> Applying [{hypotheses[best_j]['label']}] scores best across the whole survey --")
        print("     this is the geometry fix indicated by this check. Confirm visually on a few")
        print("     individual shots with shot_geometry_qc_marimo.py before committing to it.")

    report = {
        "input": str(args.input),
        "n_shots": n_src,
        "n_traces": info["n_traces"],
        "n_hypotheses": len(hypotheses),
        "parameters": vars(args) | {"input": str(args.input), "outdir": str(args.outdir)},
        "hypotheses": [h["label"] for h in hypotheses],
        "median_r2_per_hypothesis": {h["label"]: float(scores["median_r2"][j]) for j, h in enumerate(hypotheses)},
        "mean_r2_per_hypothesis": {h["label"]: float(scores["mean_r2"][j]) for j, h in enumerate(hypotheses)},
        "median_velocity_per_hypothesis": {h["label"]: float(scores["median_v"][j]) for j, h in enumerate(hypotheses)},
        "n_valid_shots_per_hypothesis": {h["label"]: int(scores["n_valid_shots_per_hyp"][j]) for j, h in enumerate(hypotheses)},
        "best_hypothesis": hypotheses[best_j]["label"],
        "baseline_hypothesis": hypotheses[baseline_j]["label"],
        "checks_passed": n_pass,
        "checks_total": n_total,
        "checks": all_checks,
    }
    json_path = outdir / "whole_survey_direction_check_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print()
    print(f"  Report written: {json_path}")
    print(f"  QC plots       : {outdir / 'qc_plots'}")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
