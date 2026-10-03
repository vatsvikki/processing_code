"""Inline / crossline (IL / XL) grid of the survey (no UI, no plotting).

Where the grid comes from (first that works):
  * corner points typed by the user, one per line:  `IL, XL, X, Y`  (or just `IL, XL` when the headers give a
    usable grid to place them),
  * the RECEIVER words of the trace headers: receiver line (RECLN, byte 173) = IL and receiver station (RECSTN,
    byte 181) = XL, fitted against the receiver coordinates,
  * the inline_no / crossline_no words (bytes 189 / 193), fitted against the trace midpoints.
If none of them forms a regular grid the caller asks the user for corner points.

Both end up as an affine map  [x, y] = a @ [1, IL, XL].  Corner points the user gave are saved (per SEG-Y file, in
~/.trace_qc/geometry_grids.json, or $TRACE_QC_HOME) and offered again the next time that file is used.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import segy_io
from .geometry import Geometry

CORNER_HELP = ("One corner per line:  IL, XL, X, Y  (3 or 4 corners; X, Y in the file's coordinate units), "
               "separated by tabs, spaces or commas - a table pasted from a spreadsheet works, heading lines and a "
               "last line that repeats the first corner are ignored. IL, XL alone also works when the trace headers "
               "give a usable grid. Lines starting with # are ignored.")

# The corner-point table as it is usually supplied (this survey's bin grid, 150 ft x 150 ft, coordinates in ft).
# Shown greyed-out in the empty corner-points box of the Acquisition Geometry tool, and parsed as it stands.
CORNER_EXAMPLE = (
    "IL\tXL\tcoords (ft)\t\n"
    "\t\tx-coords\ty-coords\n"
    "1001\t1001\t1971388.1\t428622\n"
    "1001\t1686\t2074138.1\t428622\n"
    "1404\t1686\t2074138.1\t489072\n"
    "1404\t1001\t1971388.1\t489072\n"
    "1001\t1001\t1971388.1\t428622\n"
)


@dataclass
class Grid:
    a: np.ndarray                                    # [x, y] = a @ [1, il, xl]   (2 x 3)
    source: str                                      # "corner points" | "trace headers"
    corners: list[tuple[float, float, float, float]] # (il, xl, x, y) of the survey outline, in drawing order
    rms: float                                       # fit error of the points used, coordinate units
    n_points: int
    outliers: int = 0                                # points left out of a header fit because they sit off the grid

    def xy(self, il, xl):
        il, xl = np.asarray(il, float), np.asarray(xl, float)
        return self.a[0, 0] + self.a[0, 1] * il + self.a[0, 2] * xl, self.a[1, 0] + self.a[1, 1] * il + self.a[1, 2] * xl

    def ilxl(self, x, y):
        m = self.a[:, 1:]
        d = np.stack([np.asarray(x, float) - self.a[0, 0], np.asarray(y, float) - self.a[1, 0]])
        il, xl = np.linalg.solve(m, d.reshape(2, -1))
        return il.reshape(np.shape(x)), xl.reshape(np.shape(x))

    @property
    def il_spacing(self) -> float:                   # distance between neighbouring inlines (IL + 1)
        return float(np.hypot(*self.a[:, 1]))

    @property
    def xl_spacing(self) -> float:                   # distance between neighbouring crosslines (XL + 1)
        return float(np.hypot(*self.a[:, 2]))

    @property
    def inline_azimuth(self) -> float:               # direction along an inline (XL increasing), degrees from north
        return float(np.degrees(np.arctan2(self.a[0, 2], self.a[1, 2])) % 360)

    @property
    def il_range(self) -> tuple[float, float]:
        v = [c[0] for c in self.corners]
        return min(v), max(v)

    @property
    def xl_range(self) -> tuple[float, float]:
        v = [c[1] for c in self.corners]
        return min(v), max(v)


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------
def _fit(il, xl, x, y) -> tuple[np.ndarray, float]:
    """Least-squares affine map (il, xl) -> (x, y); returns (2x3 matrix, rms error).  Raises if the points are collinear."""
    A = np.column_stack([np.ones(len(il)), il, xl]).astype(float)
    if np.linalg.matrix_rank(A) < 3:
        raise ValueError("the points lie on one line (or repeat) - they cannot define a grid")
    cx, *_ = np.linalg.lstsq(A, np.asarray(x, float), rcond=None)
    cy, *_ = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)
    err = np.hypot(A @ cx - x, A @ cy - y)
    return np.vstack([cx, cy]), float(np.sqrt((err ** 2).mean()))


def _outline(grid_a: np.ndarray, il0, il1, xl0, xl1) -> list[tuple[float, float, float, float]]:
    out = []
    for il, xl in ((il0, xl0), (il1, xl0), (il1, xl1), (il0, xl1)):
        out.append((float(il), float(xl), float(grid_a[0, 0] + grid_a[0, 1] * il + grid_a[0, 2] * xl),
                    float(grid_a[1, 0] + grid_a[1, 1] * il + grid_a[1, 2] * xl)))
    return out


def _fit_robust(il, xl, x, y, rounds: int = 4) -> tuple[np.ndarray, float, int, int]:
    """Affine fit that ignores points sitting well off the grid (a few mis-positioned receivers must not spoil it).
    Returns (matrix, rms of the points kept, points kept, points left out)."""
    il, xl, x, y = (np.asarray(v, float) for v in (il, xl, x, y))
    a, _ = _fit(il, xl, x, y)
    A = np.column_stack([np.ones(len(il)), il, xl])
    keep = np.ones(len(il), bool)
    for _ in range(rounds):
        err = np.hypot(A @ a[0] - x, A @ a[1] - y)
        spacing = min(np.hypot(*a[:, 1]), np.hypot(*a[:, 2]))
        new = err <= max(4.0 * float(np.median(err[keep])), 0.02 * spacing)
        if new.sum() < 3 or (new == keep).all():
            break
        try:
            a, _ = _fit(il[new], xl[new], x[new], y[new])
        except ValueError:
            break
        keep = new
    err = np.hypot(A @ a[0] - x, A @ a[1] - y)
    return a, float(np.sqrt((err[keep] ** 2).mean())), int(keep.sum()), int((~keep).sum())


def _header_candidate(name: str, words: str, il, xl, x, y, unit: str) -> tuple[Grid | None, str]:
    """A grid from one pair of header words, or (None, why not)."""
    il, xl = np.asarray(il, float), np.asarray(xl, float)
    if len(il) < 3 or il.min() == il.max() or xl.min() == xl.max():
        return None, f"{words} are zero or constant"
    try:
        a, rms, n_kept, n_out = _fit_robust(il, xl, x, y)
    except ValueError as e:
        return None, f"{words} {e}"
    g = Grid(a, name, _outline(a, il.min(), il.max(), xl.min(), xl.max()), rms, n_kept, n_out)
    u = f" {unit}" if unit else ""
    if min(g.il_spacing, g.xl_spacing) <= 0 or rms > 0.25 * min(g.il_spacing, g.xl_spacing) or n_out > 0.1 * len(il):
        return None, (f"{words} do not form a regular grid over the positions (best fit is off by {rms:,.0f}{u} on "
                      f"average{f', {n_out:,} of {len(il):,} points off it' if n_out else ''}, against a grid spacing of "
                      f"{g.il_spacing:,.1f}{u} x {g.xl_spacing:,.1f}{u})")
    return g, ""


def grid_from_headers(geom: Geometry) -> tuple[Grid | None, str]:
    """The IL / XL grid the trace headers describe, or (None, why not).  Tried in this order:
    receiver line / station (RECLN / RECSTN) against the receiver positions, then the inline / crossline words
    against the trace midpoints."""
    tried = []
    for name, words, il, xl, x, y in (
        ("receiver headers", "the receiver line / station words (bytes 173 / 181)",
         geom.rec_line, geom.rec_stn, geom.rec_x, geom.rec_y),
        ("trace headers", "the inline / crossline words (bytes 189 / 193)", geom.hdr_il, geom.hdr_xl, geom.hdr_x, geom.hdr_y),
    ):
        g, why = _header_candidate(name, words, il, xl, x, y, geom.unit)
        if g is not None:
            return g, ""
        tried.append(why)
    return None, "; ".join(tried)


CORNER_COLUMNS = ("IL", "XL", "X", "Y")


def _cell(v) -> float | None:
    """Number in a table cell; None for an empty cell.  Raises ValueError for text that is not a number."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().replace(" ", "")
        if not s or s.lower() in ("nan", "none", "null"):
            return None
        return float(s)
    f = float(v)
    return None if f != f else f                      # NaN = empty cell


def parse_corners(source) -> list[tuple[float, ...]]:
    """Corner points -> tuples of 4 (IL, XL, X, Y) or 2 (IL, XL) numbers, repeated corners dropped.
    `source` is either the rows of the corner table (dicts keyed IL / XL / X / Y, or lists) or text with one corner per
    line.  Raises ValueError naming the bad row / line."""
    if isinstance(source, str):
        return _parse_text(source)
    rows = []
    for n, r in enumerate(source or [], 1):
        if isinstance(r, dict):
            low = {str(k).strip().lower(): v for k, v in r.items()}
            cells = [low.get(c.lower()) for c in CORNER_COLUMNS]
        else:
            cells = (list(r) + [None] * 4)[:4]
        try:
            il, xl, x, y = (_cell(c) for c in cells)
        except (TypeError, ValueError):
            raise ValueError(f"row {n}: numbers only (IL, XL, X, Y)") from None
        if il is None and xl is None and x is None and y is None:
            continue                                  # an empty row of the table
        if il is None or xl is None:
            raise ValueError(f"row {n}: both IL and XL are needed")
        if (x is None) != (y is None):
            raise ValueError(f"row {n}: X and Y must both be filled in, or both left empty")
        row = (il, xl) if x is None else (il, xl, x, y)
        if row not in rows:
            rows.append(row)
    return rows


def _parse_text(text: str) -> list[tuple[float, ...]]:
    """'IL, XL, X, Y' (or 'IL, XL') lines -> tuples of 4 (or 2) numbers.  Heading lines without any number (as in a
    table pasted from a spreadsheet) and repeated corners are skipped.  Raises ValueError naming the bad line."""
    rows = []
    for n, line in enumerate(str(text).splitlines(), 1):
        s = line.split("#")[0].strip()
        if not s:
            continue
        vals = []
        for t in (t for t in re.split(r"[,;\s]+", s) if t):
            try:
                vals.append(float(t))
            except ValueError:
                vals.append(None)
        if all(v is None for v in vals):             # a heading such as "IL  XL  coords (ft)"
            continue
        if None in vals:
            raise ValueError(f"line {n} ('{line.strip()}'): numbers only, e.g. 1000, 2000, 2016000.5, 446399")
        if len(vals) not in (2, 4):
            raise ValueError(f"line {n} ('{line.strip()}'): expected IL, XL, X, Y (or IL, XL) - got {len(vals)} numbers")
        if tuple(vals) not in rows:                  # a table that closes the polygon repeats its first corner
            rows.append(tuple(vals))
    return rows


def grid_from_corners(source, header_grid: Grid | None = None, header_reason: str = "") -> Grid:
    """Grid from the corner points the user gave (table rows or text).  Raises ValueError with a message the user
    can act on."""
    rows = parse_corners(source)
    if len(rows) < 3:
        raise ValueError(f"{len(rows)} corner point(s) given - at least 3 are needed (all 4 for a full outline)")
    with_xy = [r for r in rows if len(r) == 4]
    if len(with_xy) >= 3:
        arr = np.array(with_xy)
        try:
            a, rms = _fit(arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3])
        except ValueError as e:
            raise ValueError(f"corner points: {e}") from None
        n_used = len(with_xy)
    elif header_grid is not None:
        a, rms, n_used = header_grid.a, header_grid.rms, header_grid.n_points
    else:
        raise ValueError("X and Y are needed for at least 3 corners: " + (header_reason or "no usable header grid"))
    pts = []
    for r in rows:                                   # corners given without X, Y are placed with the fitted map
        x, y = (r[2], r[3]) if len(r) == 4 else (a[0, 0] + a[0, 1] * r[0] + a[0, 2] * r[1], a[1, 0] + a[1, 1] * r[0] + a[1, 2] * r[1])
        pts.append((float(r[0]), float(r[1]), float(x), float(y)))
    cx, cy = np.mean([p[2] for p in pts]), np.mean([p[3] for p in pts])
    pts.sort(key=lambda p: np.arctan2(p[3] - cy, p[2] - cx))         # draw the outline around the centre
    return Grid(a, "corner points", pts, rms, n_used)


# ---------------------------------------------------------------------------
# the corner table (what the user sees and edits)
# ---------------------------------------------------------------------------
def _shown(v: float, digits: int = 1):
    """A number as the table shows it: whole numbers without a decimal point."""
    return int(round(v)) if abs(v - round(v)) < 1e-9 else round(float(v), digits)


def table_rows(corners, n_min: int = 0) -> list[dict]:
    """Corner tuples (IL, XL[, X, Y]) -> rows for the table widget, padded with empty rows up to `n_min`
    (empty cells are "", which the table shows blank)."""
    rows = []
    for c in corners:
        rows.append({"IL": _shown(c[0], 3), "XL": _shown(c[1], 3),
                     "X": _shown(c[2]) if len(c) == 4 else "", "Y": _shown(c[3]) if len(c) == 4 else ""})
    while len(rows) < n_min:
        rows.append({k: "" for k in CORNER_COLUMNS})
    return rows


def corners_to_text(corners) -> str:
    """Corner tuples -> the text form that is saved: one `IL, XL, X, Y` line per corner."""
    return "\n".join(", ".join(f"{v:.12g}" for v in c) for c in corners)


def same_corners(corners, grid: Grid, tol: float = 0.06) -> bool:
    """True when the corner points are the corners of `grid` (X, Y compared to `tol`, as the table shows them
    rounded to 0.1) - i.e. the table still holds the values it was filled with from the headers."""
    if len(corners) != len(grid.corners):
        return False
    left = list(grid.corners)
    for c in corners:
        hit = next((g for g in left if abs(g[0] - c[0]) < 1e-6 and abs(g[1] - c[1]) < 1e-6
                    and (len(c) < 4 or (abs(g[2] - c[2]) <= tol and abs(g[3] - c[3]) <= tol))), None)
        if hit is None:
            return False
        left.remove(hit)
    return True


def corner_table(path: str) -> list[dict]:
    """The rows the corner table starts with for this SEG-Y file: the table saved earlier, else the corners of the
    grid the headers describe (change any value to use your own IL / XL), else empty rows."""
    try:
        saved = parse_corners(load_saved(path))
    except ValueError:
        saved = []
    if saved:
        return table_rows(saved)                      # (more rows: the table's "+ New row")
    from . import geometry                            # imported here: geometry is heavy and only needed for this
    g, _ = grid_from_headers(geometry.read_geometry(path, 60))
    return table_rows([(c[0], c[1], round(c[2], 1), round(c[3], 1)) for c in g.corners]) if g else table_rows([], 4)


# ---------------------------------------------------------------------------
# saved corner points (per SEG-Y file)
# ---------------------------------------------------------------------------
def store_path() -> Path:
    return Path(os.environ.get("TRACE_QC_HOME") or Path.home() / ".trace_qc") / "geometry_grids.json"


def _read_store() -> dict:
    try:
        d = json.loads(store_path().read_text())
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def load_saved(path: str) -> str:
    """The corner-point text saved for this SEG-Y file, or ''."""
    return str(_read_store().get(segy_io.norm_path(path), {}).get("corners", ""))


def save_corners(path: str, text: str) -> bool:
    """Remember the corner points for this file.  True when something new was written."""
    text = text.strip()
    if load_saved(path).strip() == text:
        return False
    store = _read_store()
    store[segy_io.norm_path(path)] = {"corners": text, "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    p = store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")           # write beside it, then replace: never half a file
    with os.fdopen(fd, "w") as f:
        json.dump(store, f, indent=2)
    os.replace(tmp, p)
    return True


def forget_saved(path: str) -> bool:
    store = _read_store()
    if store.pop(segy_io.norm_path(path), None) is None:
        return False
    p = store_path()
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(store, f, indent=2)
    os.replace(tmp, p)
    return True


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def grid_rows(g: Grid, unit: str = "", spread=None) -> list[dict]:
    u = f" {unit}" if unit else ""
    il0, il1 = g.il_range
    xl0, xl1 = g.xl_range
    rows = [
        {"Item": "Grid from", "Value": g.source + (f"  (fit error {g.rms:,.2f}{u} over {g.n_points:,} points"
                                                   + (f", {g.outliers:,} off-grid points ignored" if g.outliers else "") + ")"
                                                   if g.n_points > 3 or g.rms > 1e-6 else "")},
        {"Item": "Inline range (IL)", "Value": f"{il0:g} - {il1:g}"},
        {"Item": "Crossline range (XL)", "Value": f"{xl0:g} - {xl1:g}"},
        {"Item": "Distance between inlines (IL + 1)", "Value": f"{g.il_spacing:,.2f}{u}"},
        {"Item": "Distance between crosslines (XL + 1)", "Value": f"{g.xl_spacing:,.2f}{u}"},
        {"Item": "Inline direction (XL increasing), azimuth", "Value": f"{g.inline_azimuth:.1f}° from north"},
    ]
    for k, (il, xl, x, y) in enumerate(g.corners, 1):
        rows.append({"Item": f"Corner {k}", "Value": f"IL {il:g}, XL {xl:g}   at X {x:,.1f}, Y {y:,.1f}"})
    if spread is not None:
        sil, sxl = g.ilxl(spread.src_x, spread.src_y)
        ril, rxl = g.ilxl(spread.rec_x, spread.rec_y)
        rows.append({"Item": f"FFID {spread.ffid} source", "Value": f"IL {float(sil):.1f}, XL {float(sxl):.1f}"})
        rows.append({"Item": f"FFID {spread.ffid} receivers", "Value":
                     f"IL {ril.min():.1f} - {ril.max():.1f}, XL {rxl.min():.1f} - {rxl.max():.1f}"})
    return rows
