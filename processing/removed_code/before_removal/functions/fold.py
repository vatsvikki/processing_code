"""Fold (traces per bin) from the source-receiver midpoints of every trace (no UI, no plotting).

Bins are the cells of the IL / XL grid when there is one (one cell per IL and XL number, centred on the whole numbers),
otherwise square X / Y bins of `bin_size`.  Fold = number of traces whose midpoint falls in the bin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry
from .grid import Grid


@dataclass
class FoldMap:
    counts: np.ndarray               # [n_rows, n_cols] traces per bin (0 = no data)
    node_x: np.ndarray               # [n_rows + 1, n_cols + 1] corner coordinates of the bins
    node_y: np.ndarray
    kind: str                        # "IL / XL grid" | "X / Y bins"
    bin_desc: str                    # e.g. "150.0 ft x 150.0 ft"
    alignment: str                   # where the bins sit relative to whole IL / XL numbers
    traces: int                      # traces counted
    outside: int                     # traces whose midpoint fell outside the bins
    unit: str

    @property
    def filled(self) -> np.ndarray:
        return self.counts[self.counts > 0]


def auto_bin_size(geom) -> float:
    """Square bin for a file without an IL / XL grid: half the median distance between neighbouring receivers on a
    receiver line, rounded to a round number (150 ft for 300 ft receivers)."""
    steps = []
    for line in np.unique(geom.rec_line):
        m = geom.rec_line == line
        o = np.argsort(geom.rec_stn[m])
        x, y = geom.rec_x[m][o], geom.rec_y[m][o]
        d = np.hypot(np.diff(x), np.diff(y))
        steps += list(d[d > 0])
    if not steps:
        return 25.0
    half = float(np.median(steps)) / 2
    mag = 10.0 ** np.floor(np.log10(half))
    return float(round(half / (mag / 2)) * (mag / 2)) or 25.0


_ALIGN = {"auto": None, "bin centres on whole IL / XL": [(0.0, 0.0)], "bin edges on whole IL / XL": [(0.5, 0.5)]}
ALIGN_CHOICES = list(_ALIGN)


def _cv(counts: np.ndarray) -> float:
    live = counts[counts > 0]
    return float(live.std() / live.mean()) if len(live) else 1e9


@dataclass
class Bins:
    """Every trace assigned to a CMP bin of the IL / XL grid."""
    ii: np.ndarray                   # per trace: bin row (IL direction), -1 = outside the bins
    jj: np.ndarray                   # per trace: bin column (XL direction)
    il0: int                         # IL number of row 0 (before the half-bin shift)
    xl0: int
    si: float                        # half-bin shifts chosen (0 or 0.5)
    sj: float
    counts: np.ndarray               # [n_i, n_j] traces per bin
    alignment: str
    grid: Grid

    @property
    def shape(self) -> tuple[int, int]:
        return self.counts.shape

    def il_of(self, i):              # IL number at the centre of bin row i
        return self.il0 + np.asarray(i) + self.si

    def xl_of(self, j):
        return self.xl0 + np.asarray(j) + self.sj

    def locate(self, il: float, xl: float) -> tuple[int, int]:
        """Bin row / column containing IL / XL number (nearest bin centre)."""
        return (int(np.floor(il - self.si + 0.5 + 1e-9)) - self.il0, int(np.floor(xl - self.sj + 0.5 + 1e-9)) - self.xl0)


def assign_bins(path: str, grid: Grid, align: str = "auto") -> Bins:
    """Put every trace (by its source-receiver midpoint) into a bin of the IL / XL grid, one bin per IL x XL number.
    `align` as for compute_fold: "auto" keeps the half-bin shift with the smoothest fold."""
    mx, my = geometry.read_midpoints(path)
    il, xl = grid.ilxl(mx, my)
    (il0, il1), (xl0, xl1) = (tuple(int(round(v)) for v in r) for r in (grid.il_range, grid.xl_range))
    shifts = _ALIGN.get(align) or [(0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5)]

    def one(si, sj):
        # nearest bin centre; midpoints exactly half-way go up, always (np.rint would send them to the even one)
        ii = np.floor(il - si + 0.5 + 1e-9).astype(np.int64) - il0
        jj = np.floor(xl - sj + 0.5 + 1e-9).astype(np.int64) - xl0
        n_i, n_j = il1 - il0 + 1 + (si > 0), xl1 - xl0 + 1 + (sj > 0)
        inside = (ii >= 0) & (ii < n_i) & (jj >= 0) & (jj < n_j)
        counts = np.bincount(ii[inside] * n_j + jj[inside], minlength=n_i * n_j).reshape(n_i, n_j)
        ii[~inside] = -1
        jj[~inside] = -1
        return counts, ii, jj, si, sj

    counts, ii, jj, si, sj = min((one(si, sj) for si, sj in shifts), key=lambda r: (_cv(r[0]), r[3], r[4]))
    where = "bin centres on whole IL / XL" if (si, sj) == (0, 0) else (
        "bin edges on whole IL / XL" if (si, sj) == (0.5, 0.5) else
        f"bins shifted half a bin in {'IL' if si else ''}{' and ' if si and sj else ''}{'XL' if sj else ''}")
    return Bins(ii, jj, il0, xl0, si, sj, counts, where + (" (chosen automatically)" if align == "auto" else ""), grid)


def compute_fold(path: str, grid: Grid | None = None, bin_size: float = 150.0, align: str = "auto") -> FoldMap:
    """Fold of every trace.  `align` decides where the bins sit relative to whole IL / XL numbers (or, without a
    grid, to the X / Y origin): "auto" tries all four half-bin shifts and keeps the smoothest fold - the one that
    puts the midpoints at bin centres instead of on bin boundaries, where they would alias into stripes."""
    unit = geometry.read_geometry(path, 60).unit
    u = f" {unit}" if unit else ""
    shifts = _ALIGN.get(align) or [(0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5)]

    if grid is not None:
        bins = assign_bins(path, grid, align)
        n_i, n_j = bins.shape
        node_x, node_y = grid.xy((bins.il0 - 0.5 + bins.si + np.arange(n_i + 1))[:, None],
                                 (bins.xl0 - 0.5 + bins.sj + np.arange(n_j + 1))[None, :])
        inside = bins.ii >= 0
        return FoldMap(bins.counts, node_x, node_y, "IL / XL grid",
                       f"{grid.il_spacing:,.1f}{u} x {grid.xl_spacing:,.1f}{u}  (one IL x one XL)",
                       bins.alignment, int(inside.sum()), int((~inside).sum()), unit)

    mx, my = geometry.read_midpoints(path)
    b = max(float(bin_size), 1e-6)

    def count_xy(sx, sy):
        xe = np.floor(mx.min() / b - sx) * b + sx * b + np.arange(0, int((mx.max() - mx.min()) / b) + 3) * b
        ye = np.floor(my.min() / b - sy) * b + sy * b + np.arange(0, int((my.max() - my.min()) / b) + 3) * b
        return np.histogram2d(mx, my, bins=[xe, ye])[0].astype(np.int64).T, xe, ye, sx, sy

    counts, xe, ye, sx, sy = min((count_xy(sx, sy) for sx, sy in shifts), key=lambda r: (_cv(r[0]), r[3], r[4]))
    node_x, node_y = np.meshgrid(xe, ye)
    where = "bins from the X / Y origin" if (sx, sy) == (0, 0) else f"bins shifted by half a bin ({sx:g}, {sy:g})"
    return FoldMap(counts, node_x, node_y, "X / Y bins", f"{b:,.1f}{u} x {b:,.1f}{u}",
                   where + (" (chosen automatically)" if align == "auto" else ""), int(counts.sum()), 0, unit)


def summary_rows(f: FoldMap) -> list[dict]:
    live = f.filled
    return [
        {"Item": "Bins", "Value": f"{f.kind}, {f.bin_desc}"},
        {"Item": "Bin alignment", "Value": f.alignment},
        {"Item": "Bins with data", "Value": f"{len(live):,}  (of {f.counts.size:,})"},
        {"Item": "Traces counted", "Value": f"{f.traces:,}" + (f"  ({f.outside:,} outside the bins)" if f.outside else "")},
        {"Item": "Fold - maximum", "Value": f"{int(live.max()):,}" if len(live) else "-"},
        {"Item": "Fold - mean over bins with data", "Value": f"{live.mean():,.1f}" if len(live) else "-"},
        {"Item": "Fold - median", "Value": f"{float(np.median(live)):,.0f}" if len(live) else "-"},
        {"Item": "Fold - minimum", "Value": f"{int(live.min()):,}" if len(live) else "-"},
    ]
