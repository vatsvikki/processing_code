"""Sort and QC of the CMP sort (no UI).

The CMP sort is the one the brute stack used: every trace goes to the CMP bin (CDP gather) of the IL / XL grid its
source-receiver midpoint falls in, and inside a bin the traces are ordered by offset.  `sort_order` gives that order for the
whole file (an index, no samples are read).  `run_sort_qc` then checks that the sort is PROPER:

  * geometry   - every trace in exactly one bin, the traces outside the bins, fold, holes, near / far offset coverage;
  * integrity  - bins in order, offsets ascending inside every bin, no duplicate traces, offset header = offset computed from
                 the coordinates, no zero coordinates;
  * data       - traces that neighbour in offset inside a CMP gather must look alike (reflections continue across offset),
                 far more than traces paired at random.  A wrongly sorted or mixed gather fails this.
The CDP number of the trace header is reported next to the bins, but the sort follows the bins.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry, stacking
from .fold import Bins

SORT_KEYS = {"offset": "offset", "CMP bin (cdp)": "cdp", "CMP bin, then offset": "cdp_offset"}   # label -> key
OK, WARN, INFO = "✅ OK", "⚠ check", "ℹ info"


def bin_ids(bins: Bins) -> np.ndarray:
    """CMP bin number of every trace (row * columns + column); traces outside the bins get the largest number."""
    n_i, n_j = bins.shape
    return np.where(bins.ii >= 0, bins.ii * n_j + bins.jj, n_i * n_j)


def sort_order(tab: dict, key: str, bins: Bins) -> np.ndarray:
    """File trace indices in sorted order: "offset" (absolute header offset), "cdp" (by CMP bin) or "cdp_offset" (by CMP bin,
    and by offset inside a bin).  Ties keep the file order."""
    off, bid = np.abs(tab["offset"]), bin_ids(bins)
    if key == "offset":
        return np.argsort(off, kind="stable")
    if key == "cdp":
        return np.argsort(bid, kind="stable")
    if key == "cdp_offset":
        return np.lexsort((off, bid))
    raise ValueError(f"unknown sort key {key!r}; use one of {list(SORT_KEYS.values())}")


@dataclass
class SortQC:
    rows: list[dict]                          # Status | Check | Result | Meaning
    fold: np.ndarray                          # fold of every bin with data
    min_off: np.ndarray                       # smallest / largest |offset| of every bin with data
    max_off: np.ndarray
    diff_pct: np.ndarray                      # (|offset header| - offset from the coordinates) as % of the latter, per trace
    coh_adjacent: np.ndarray                  # correlation of neighbouring offsets inside sampled CMP gathers
    coh_random: np.ndarray                    # the same for randomly paired traces of the same gathers
    unit: str
    low_limit: float                          # fold below this counts as low
    tol_pct: float
    order: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))   # the sort order (file trace indices)
    n_warn: int = 0


def _best_corr(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    """Largest normalised cross-correlation of a and b over shifts of up to +-lag samples."""
    a, b = a - a.mean(), b - b.mean()
    best = -1.0
    for s in range(-lag, lag + 1):
        x, y = (a[s:], b[:len(b) - s]) if s >= 0 else (a[:s], b[-s:])
        d = float(np.linalg.norm(x) * np.linalg.norm(y))
        if d > 0:
            best = max(best, float(x @ y) / d)
    return best


def gather_coherence(path: str, bins: Bins, n_bins: int = 40, seed: int = 7) -> tuple[np.ndarray, np.ndarray, int]:
    """Correlation of offset-neighbours vs randomly paired traces, in `n_bins` random high-fold CMP gathers (a window in the
    middle of the record, shifts up to +-40 ms).  In a properly sorted CMP gather the neighbours are far more alike."""
    rng = np.random.default_rng(seed)
    counts = bins.counts
    cand = np.argwhere(counts >= max(10, int(np.median(counts[counts > 0]))))
    if len(cand) == 0:
        return np.zeros(0), np.zeros(0), 0
    pick = cand[rng.choice(len(cand), min(n_bins, len(cand)), replace=False)]
    adj, rnd = [], []
    for i, j in pick:
        cg = stacking.cmp_gather(path, bins, float(bins.il_of(i)), float(bins.xl_of(j)))
        ns, dt = cg.data.shape[1], cg.dt_ms
        w0, w1 = int(0.2 * ns), int(min(0.75 * ns, 3000 / dt + 250))
        d, lag, n = cg.data[:, w0:w1], max(1, int(round(40 / dt))), cg.ntr
        adj += [_best_corr(d[k], d[k + 1], lag) for k in range(n - 1)]
        perm = rng.permutation(n)
        rnd += [_best_corr(d[perm[k]], d[perm[k + 1]], lag) for k in range(n - 1)]
    return np.array(adj), np.array(rnd), len(pick)


def run_sort_qc(path: str, bins: Bins, sort_key: str = "cdp_offset", low_fold_pct: float = 25.0,
                offset_tol_pct: float = 2.0) -> SortQC:
    """QC of the CMP sort of `path` into `bins`, and the sort order for `sort_key`."""
    t = geometry.read_trace_table(path)
    unit = geometry.read_geometry(path, 60).unit
    n = len(t["cdp"])
    n_i, n_j = bins.shape
    counts = bins.counts
    live = counts[counts > 0]
    inside = bins.ii >= 0
    rows: list[dict] = []

    def add(check, result, status, meaning):
        rows.append({"Status": status, "Check": check, "Result": result, "Meaning": meaning})

    # ---- 1. geometry of the binning ------------------------------------------------------------------------------------
    n_out = int((~inside).sum())
    add("Traces sorted into bins", f"{int(inside.sum()):,} of {n:,} traces in {live.size:,} bins", INFO,
        "Every trace goes to the CMP bin its source-receiver midpoint falls in (the bins of the fold map).")
    add("Traces outside the bins", f"{n_out:,}  ({100 * n_out / n:.3f} %)", OK if n_out <= 0.005 * n else WARN,
        "Midpoints outside the corner points of the IL / XL grid; a few at the edge are normal, many mean the corner points "
        "are too small.")
    med = float(np.median(live))
    low = live < low_fold_pct / 100.0 * med
    add("Fold (mean / median / maximum)", f"{live.mean():.1f} / {med:.0f} / {int(live.max())}", INFO, "Traces per bin.")
    add(f"Low-fold bins (< {low_fold_pct:g} % of the median)", f"{int(low.sum()):,}  ({100 * low.mean():.1f} % of the bins with data)",
        OK if low.mean() <= 0.15 else WARN, "Bins that would stack poorly; the edges of a survey are normally low.")
    first = np.array([np.flatnonzero(r)[0] if r.any() else -1 for r in counts > 0])
    last = np.array([np.flatnonzero(r)[-1] if r.any() else -1 for r in counts > 0])
    span = int(sum(l - f + 1 for f, l in zip(first, last) if f >= 0))
    holes = span - int((counts > 0).sum())
    add("Holes in the coverage", f"{holes:,} empty bins inside the rows  ({100 * holes / max(span, 1):.2f} %)",
        OK if holes <= 0.02 * span else WARN, "Empty bins between filled bins of the same inline: gaps in the fold.")

    # ---- 2. offsets of every bin ------------------------------------------------------------------------------------------
    off = np.abs(t["offset"])
    idx = np.flatnonzero(inside)
    bid = bin_ids(bins)[idx]
    o = np.argsort(bid, kind="stable")
    bid_s, off_s = bid[o], off[idx][o]
    starts = np.r_[0, np.flatnonzero(np.diff(bid_s)) + 1]
    min_off, max_off = np.minimum.reduceat(off_s, starts), np.maximum.reduceat(off_s, starts)
    u = f" {unit}" if unit else ""
    near = min_off > 2.0 * np.median(min_off)
    far = max_off < 0.5 * np.median(max_off)
    add("Near-offset coverage", f"median smallest offset {np.median(min_off):,.0f}{u}; {100 * near.mean():.1f} % of the bins "
        f"start at more than twice that", OK if near.mean() <= 0.05 else WARN, "Bins without near traces (a hole around the source).")
    add("Far-offset coverage", f"median largest offset {np.median(max_off):,.0f}{u}; {100 * far.mean():.1f} % of the bins end "
        f"below half of that", OK if far.mean() <= 0.05 else WARN, "Bins without far traces: the NMO velocity is then poorly constrained.")

    # ---- 3. integrity of the sort itself ----------------------------------------------------------------------------------
    proper = sort_order(t, "cdp_offset", bins)                          # the CMP sort: bin, then offset
    pb, po = bin_ids(bins)[proper], off[proper]
    same = pb[1:] == pb[:-1]
    bad_bins = int((np.diff(pb) < 0).sum())
    bad_off = int((same & (np.diff(po) < 0)).sum())
    add("Bins in order, offsets ascending inside every bin", f"{bad_bins:,} bin-order and {bad_off:,} offset-order violations "
        f"in the sorted file", OK if bad_bins == 0 and bad_off == 0 else WARN,
        "After the sort the bin number never decreases and the offset never decreases inside a bin.")
    dup = n - len(np.unique(t["key"]))
    add("Duplicate traces (same FFID + channel)", f"{dup:,}", OK if dup == 0 else WARN,
        "The same trace twice would be counted twice in every stack.")
    xy = np.maximum(t["offset_xy"], 1.0)
    diff_pct = (np.abs(t["offset"]) - t["offset_xy"]) / xy * 100.0
    bad = np.abs(diff_pct) > offset_tol_pct
    add(f"Offset header vs coordinates (> {offset_tol_pct:g} %)", f"{int(bad.sum()):,} traces  ({100 * bad.mean():.3f} %)",
        OK if bad.mean() <= 0.005 else WARN,
        "The offset word against the distance between source and receiver: a wrong offset breaks the sort by offset.")
    zero = int(t["zero_xy"].sum())
    add("Traces with zero source or receiver coordinates", f"{zero:,}", OK if zero == 0 else WARN,
        "Missing coordinates put the midpoint at the origin - those traces land outside the bins.")
    cd = t["cdp"][idx][o]
    pair = np.unique(bid_s.astype(np.int64) * (1 << 32) + (cd - cd.min()))
    per_bin = np.unique(pair >> 32, return_counts=True)[1]
    one = float(np.mean(per_bin == 1))
    add("Header CDP inside a bin", f"{100 * one:.1f} % of the bins hold one header CDP number ({len(np.unique(t['cdp'])):,} CDP "
        f"numbers in the file)", INFO if one < 1 else OK,
        "The sort follows the bins, not the CDP header: if the header CDP numbering uses another bin layout, the traces of one "
        "CMP bin carry different CDP numbers - that is not an error of the sort.")

    # ---- 4. the data: do the sorted neighbours look alike? -----------------------------------------------------------------
    adj, rnd, n_gath = gather_coherence(path, bins)
    if len(adj):
        ratio = float(np.mean(adj) / max(np.mean(rnd), 1e-6))
        add("Neighbouring offsets look alike (data)", f"correlation {np.mean(adj):.2f} for offset neighbours against "
            f"{np.mean(rnd):.2f} for random pairs ({ratio:.1f}x) in {n_gath} sampled gathers",
            OK if ratio >= 1.5 and np.mean(adj) >= 0.15 else WARN,
            "In a properly sorted CMP gather the reflections continue from one offset to the next, so neighbours correlate far "
            "more than random pairs. A wrong sort, or bins that mix different midpoints, would not.")

    # ---- 5. the order chosen ----------------------------------------------------------------------------------------------
    order = sort_order(t, sort_key, bins)
    k = {"offset": off, "cdp": bin_ids(bins), "cdp_offset": bin_ids(bins)}[sort_key][order]
    add("Sort order chosen", f"{n:,} traces in order of {[l for l, v in SORT_KEYS.items() if v == sort_key][0]}",
        OK if bool((np.diff(k) >= 0).all()) else WARN, "The key never decreases along the sorted order.")
    qc = SortQC(rows, live.astype(np.int64), min_off.astype(float), max_off.astype(float), diff_pct.astype(np.float32), adj, rnd,
                unit, low_fold_pct / 100.0 * med, offset_tol_pct, order)
    qc.n_warn = sum(r["Status"] == WARN for r in rows)
    return qc
