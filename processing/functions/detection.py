"""Dead / bad trace detection on one shot gather (numpy only).

Idea
----
For every trace we measure a few amplitude statistics inside an analysis window
(RMS, peak-to-peak, fraction of exactly-zero samples, crest factor, DC ratio).
Because seismic amplitude decays strongly with offset, a trace is NOT judged
against the whole gather but against a *reference RMS*: the running median of
the RMS of its neighbouring live traces (neighbours in the chosen reference order --
by offset by default).

DEAD  = no usable signal at all
        (flat/zero trace, mostly-zero trace, or RMS ~ 0 vs. neighbours)
BAD   = signal exists but is not trustworthy
        (too strong = noisy, too weak, spiky, large DC offset)

A trace that is dead is reported as dead only.  Setting any threshold to 0
switches that single test off.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .segy_io import ShotGather, trace_order


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def time_window_samples(ns: int, dt_ms: float, t_start_ms: float, t_end_ms: float) -> slice:
    """t_end_ms <= 0 means 'to the end of the trace'."""
    record_ms = ns * dt_ms
    if t_start_ms >= record_ms:
        raise ValueError(f"analysis window starts at {t_start_ms:g} ms but the record is only {record_ms:g} ms long")
    i0 = int(round(max(t_start_ms, 0.0) / dt_ms))
    i1 = ns if t_end_ms <= 0 else min(int(round(t_end_ms / dt_ms)), ns)
    if i1 - i0 < 2:
        raise ValueError("analysis window must contain at least 2 samples (check start / end time)")
    return slice(i0, i1)


def trace_metrics(data: np.ndarray, win: slice) -> dict[str, np.ndarray]:
    """Per-trace statistics inside the analysis window. data: [ntr, ns]."""
    w = data[:, win]
    nonfinite = ~np.isfinite(w)
    n_nonfinite = nonfinite.sum(axis=1)
    if n_nonfinite.any():
        w = np.where(nonfinite, 0.0, w)

    rms = np.sqrt(np.mean(np.square(w, dtype=np.float64), axis=1))
    mx, mn = w.max(axis=1).astype(np.float64), w.min(axis=1).astype(np.float64)
    max_abs = np.maximum(np.abs(mx), np.abs(mn))
    mean = w.mean(axis=1, dtype=np.float64)
    std = w.std(axis=1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        crest = np.where(rms > 0, max_abs / rms, 0.0)
        dc_ratio = np.where(std > 0, np.abs(mean) / std, np.where(np.abs(mean) > 0, np.inf, 0.0))
    return {
        "rms": rms,
        "p2p": mx - mn,
        "max_abs": max_abs,
        "zero_frac": (w == 0).mean(axis=1),
        "crest": crest,
        "dc_ratio": dc_ratio,
        "n_nonfinite": n_nonfinite,
    }


def reference_rms(rms: np.ndarray, valid: np.ndarray, order: np.ndarray, window: int) -> np.ndarray:
    """Running median of `rms` over `window` neighbouring valid traces.

    Neighbours are taken along `order`; the result is returned in the original
    (file) trace order.  window <= 1 (or >= ntr) -> one global median.
    """
    ntr = len(rms)
    v = np.where(valid, rms, np.nan)[order]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN slices are handled below
        glob = np.nanmedian(v) if np.isfinite(v).any() else np.nan
        if window <= 1 or window >= ntr:
            ref_sorted = np.full(ntr, glob)
        else:
            window += (window % 2 == 0)              # odd
            pad = window // 2
            padded = np.pad(v, pad, mode="edge")
            ref_sorted = np.nanmedian(sliding_window_view(padded, window), axis=1)
            ref_sorted = np.where(np.isfinite(ref_sorted), ref_sorted, glob)
    ref = np.empty(ntr)
    ref[order] = ref_sorted
    return ref


def neighbour_correlation(
    data: np.ndarray,
    valid: np.ndarray,
    rec_line: np.ndarray,
    rec_stn: np.ndarray,
    win: slice,
    *,
    neighbours: int = 2,
    max_lag: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """How well every trace resembles its neighbours on the same receiver line.

    Neighbours are the `neighbours` nearest *valid* traces on each side along the receiver line
    (sorted by line, station).  For every pair the normalised cross-correlation is searched over
    +/- `max_lag` samples (adjacent traces are shifted by the moveout).  A trace's values are the
    median over its neighbours, so one bad neighbour does not spoil a good trace.

    Returns (score, flipped):
      score   median of the best *positive* correlation (about 0..1; high = looks like its neighbours)
      flipped median of the best correlation with the trace turned upside down (= -most negative value);
              a trace whose polarity is reversed has a low score but a high flipped value
    Both are NaN for a trace with no valid neighbour (invalid trace, or alone on its line).
    """
    ntr = data.shape[0]
    x = np.asarray(data[:, win], np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    norm = np.sqrt((x * x).sum(axis=1))
    order = np.lexsort((rec_stn, rec_line))
    order = order[valid[order]]                       # valid traces in receiver order
    line = rec_line[order]
    w = x.shape[1]
    max_lag = int(max(0, min(max_lag, w // 4)))

    best = np.full((ntr, 2 * neighbours), np.nan)     # best positive corr against each neighbour slot
    flip = np.full((ntr, 2 * neighbours), np.nan)     # best corr of the polarity-reversed trace
    for d in range(1, neighbours + 1):
        a, b = order[:-d], order[d:]
        same = line[:-d] == line[d:]
        a, b = a[same], b[same]
        if not len(a):
            continue
        xa, xb = x[a], x[b]
        cc = np.empty((len(a), 2 * max_lag + 1))
        for k, lag in enumerate(range(-max_lag, max_lag + 1)):
            if lag >= 0:
                cc[:, k] = np.einsum("ij,ij->i", xa[:, lag:], xb[:, :w - lag])
            else:
                cc[:, k] = np.einsum("ij,ij->i", xa[:, :w + lag], xb[:, -lag:])
        with np.errstate(divide="ignore", invalid="ignore"):
            cc /= (norm[a] * norm[b])[:, None]
        cc = np.nan_to_num(cc)
        pos, neg = cc.max(axis=1), -cc.min(axis=1)
        best[a, d - 1], best[b, neighbours + d - 1] = pos, pos
        flip[a, d - 1], flip[b, neighbours + d - 1] = neg, neg
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN rows: no neighbour
        return np.nanmedian(best, axis=1), np.nanmedian(flip, axis=1)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def _add(reasons: list[list[str]], mask: np.ndarray, text) -> None:
    for i in np.flatnonzero(mask):
        reasons[i].append(text(i) if callable(text) else text)


@dataclass
class QCResult:
    ffid: int
    order: np.ndarray                    # display permutation (positions -> file idx)
    dead: np.ndarray                     # bool [ntr]  (file order)
    bad: np.ndarray                      # bool [ntr]
    reasons: list[list[str]]
    metrics: dict[str, np.ndarray]
    ref_rms: np.ndarray
    params: dict = field(default_factory=dict)

    @property
    def flagged(self) -> np.ndarray:
        return self.dead | self.bad


def run_qc(
    gather: ShotGather,
    *,
    check_dead: bool = True,
    check_bad: bool = True,
    sort_by: str = "file",
    ref_sort: str = "offset",
    ref_window: int = 21,
    t_start_ms: float = 0.0,
    t_end_ms: float = 0.0,
    # dead tests
    dead_p2p_max: float = 0.0,
    dead_zero_frac: float = 0.9,
    dead_rel_rms: float = 0.0,
    # bad tests
    noisy_ratio: float = 5.0,
    weak_ratio: float = 0.1,
    spike_crest: float = 0.0,
    dc_ratio_max: float = 0.0,
    corr_ratio_min: float = 0.0,
    corr_neighbours: int = 2,
    corr_lag_ms: float = 40.0,
) -> QCResult:
    """Detect dead and/or bad traces in `gather`.

    sort_by        display order of the result (default "file" = common-shot / recorded order)
    ref_sort       order in which neighbours are taken for the reference RMS (default "offset":
                   steadiest reference, independent of how the gather is displayed)

    dead_p2p_max   dead if peak-to-peak amplitude <= this (0 -> exactly flat/zero trace)
    dead_zero_frac dead if >= this fraction of samples are exactly 0   (0 = off)
    dead_rel_rms   dead if RMS < this x reference RMS                  (0 = off)
    noisy_ratio    bad  if RMS > this x reference RMS                  (0 = off)
    weak_ratio     bad  if RMS < this x reference RMS                  (0 = off)
    spike_crest    bad  if max|amp| / RMS > this (isolated spike)      (0 = off)
    dc_ratio_max   bad  if |mean| / std > this (DC offset)             (0 = off)
    corr_ratio_min bad  if the correlation with its neighbours on the receiver line is below this x the
                   local reference (median over traces at similar offsets)   (0 = off)
    corr_neighbours  neighbours on each side that a trace is correlated with
    corr_lag_ms      time shift searched between neighbours (moveout), +/- this many ms
    """
    data = gather.data
    ntr = gather.ntr
    order = trace_order(gather, sort_by)
    win = time_window_samples(gather.ns, gather.dt_ms, t_start_ms, t_end_ms)
    m = trace_metrics(data, win)
    reasons: list[list[str]] = [[] for _ in range(ntr)]

    # ---- dead -----------------------------------------------------------
    dead = np.zeros(ntr, bool)
    if check_dead:
        flat = m["p2p"] <= dead_p2p_max
        _add(reasons, flat, "flat / zero trace")
        dead |= flat
        if dead_zero_frac > 0:
            zf = m["zero_frac"] >= dead_zero_frac
            _add(reasons, zf & ~flat, lambda i: f"{m['zero_frac'][i]:.1%} zero samples")
            dead |= zf
        nf = m["n_nonfinite"] > 0.5 * (win.stop - win.start)
        _add(reasons, nf & ~dead, "NaN / Inf samples")
        dead |= nf

    # reference amplitude from live traces only
    valid = ~dead & (m["rms"] > 0)
    ref_order = trace_order(gather, ref_sort)
    ref = reference_rms(m["rms"], valid, ref_order, ref_window)
    ratio = np.divide(m["rms"], ref, out=np.zeros(ntr), where=ref > 0)

    if check_dead and dead_rel_rms > 0:
        rel = (ratio < dead_rel_rms) & ~dead
        _add(reasons, rel, lambda i: f"RMS {ratio[i]:.2g}x ref (< {dead_rel_rms:g}x)")
        dead |= rel

    # ---- bad ------------------------------------------------------------
    bad = np.zeros(ntr, bool)
    corr_metrics: dict[str, np.ndarray] = {}
    if check_bad:
        alive = ~dead
        tests = []
        if noisy_ratio > 0:
            tests.append((ratio > noisy_ratio, lambda i: f"noisy: RMS {ratio[i]:.1f}x ref"))
        if weak_ratio > 0:
            tests.append((ratio < weak_ratio, lambda i: f"weak: RMS {ratio[i]:.2g}x ref"))
        if spike_crest > 0:
            tests.append((m["crest"] > spike_crest, lambda i: f"spike: crest {m['crest'][i]:.0f}"))
        if dc_ratio_max > 0:
            tests.append((m["dc_ratio"] > dc_ratio_max, lambda i: f"DC offset: |mean|/std {m['dc_ratio'][i]:.1f}"))
        if corr_ratio_min > 0:
            ok = alive & (m["rms"] > 0)                   # dead traces are neither tested nor used as neighbours
            lag = int(round(corr_lag_ms / gather.dt_ms))
            score, flip = neighbour_correlation(
                data, ok, gather.headers["rec_line"], gather.headers["rec_stn"], win,
                neighbours=int(corr_neighbours), max_lag=lag)
            ok &= np.isfinite(score)
            corr_ref = reference_rms(np.nan_to_num(score), ok, ref_order, ref_window)
            corr_ratio = np.divide(score, corr_ref, out=np.ones(ntr), where=corr_ref > 0)
            corr_metrics = {"corr": score, "corr_ref": corr_ref, "corr_ratio": corr_ratio}
            tests.append((
                (corr_ratio < corr_ratio_min) & ok,
                lambda i: f"low correlation: {score[i]:.2f} = {corr_ratio[i]:.2f}x local {corr_ref[i]:.2f}"
                          + (" (polarity reversed?)" if flip[i] - score[i] > 0.4 else ""),
            ))
        for mask, text in tests:
            mask = mask & alive
            _add(reasons, mask, text)
            bad |= mask

    return QCResult(
        ffid=gather.ffid, order=order, dead=dead, bad=bad, reasons=reasons,
        metrics={**m, "ratio": ratio, **corr_metrics}, ref_rms=ref,
        params=dict(sort_by=sort_by, ref_sort=ref_sort, ref_window=ref_window, t_start_ms=t_start_ms, t_end_ms=t_end_ms,
                    dead_p2p_max=dead_p2p_max, dead_zero_frac=dead_zero_frac, dead_rel_rms=dead_rel_rms,
                    noisy_ratio=noisy_ratio, weak_ratio=weak_ratio, spike_crest=spike_crest,
                    dc_ratio_max=dc_ratio_max, corr_ratio_min=corr_ratio_min, corr_neighbours=corr_neighbours,
                    corr_lag_ms=corr_lag_ms),
    )


def flagged_table(gather: ShotGather, res: QCResult) -> list[dict]:
    """One row per flagged trace, in display order (for a GUI table / CSV)."""
    h = gather.headers
    pos_of = np.empty(gather.ntr, int)
    pos_of[res.order] = np.arange(gather.ntr)
    rows = []
    for i in res.order:
        if not (res.dead[i] or res.bad[i]):
            continue
        rows.append({
            "position": int(pos_of[i]) + 1,
            "file_trace": int(gather.i0 + i + 1),
            "channel": int(h["chan"][i]),
            "rec_line": int(h["rec_line"][i]),
            "rec_stn": int(h["rec_stn"][i]),
            "offset": int(h["offset"][i]),
            "class": "DEAD" if res.dead[i] else "BAD",
            "rms": float(f"{res.metrics['rms'][i]:.4g}"),
            "rms/ref": float(f"{res.metrics['ratio'][i]:.3g}"),
            "reason": "; ".join(res.reasons[i]),
        })
    return rows
