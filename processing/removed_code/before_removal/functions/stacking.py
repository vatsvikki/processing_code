"""CMP gathers, NMO correction and stacking (a brute stack) from the CMP bins of the IL / XL grid (no UI, no plotting).

Workflow: `fold.assign_bins` puts every trace into a CMP bin; `cmp_gather` reads the traces of one bin (they come from
many shots); `nmo_correct` flattens the reflections with a velocity function v(t0); `stack_line` does that for every bin
of one inline or crossline and sums, giving a stack section.  The velocity function is a plain list of (time ms,
velocity) picks, the same for the whole line - a brute stack.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import segy_io
from .fold import Bins
from .segy_io import TRACE_HEADER_FIELDS, header_word

VEL_COLUMNS = ("Time (ms)", "Velocity")


# ---------------------------------------------------------------------------
# velocity function
# ---------------------------------------------------------------------------
def default_velocity_rows(unit: str = "ft") -> list[dict]:
    """A plausible starting function (time ms, velocity in unit/s) to edit: fast enough to flatten the main events."""
    pairs = ([(0, 5000), (1000, 7000), (2500, 9500), (5000, 12500), (8000, 15000)] if unit == "ft"
             else [(0, 1500), (1000, 2100), (2500, 2900), (5000, 3800), (8000, 4600)])
    head = f"Velocity ({unit}/s)" if unit else "Velocity"
    return [{"Time (ms)": t, head: v} for t, v in pairs]


def parse_velocity(source) -> tuple[np.ndarray, np.ndarray]:
    """Velocity table (rows with the time in ms in the first column and the velocity in the second; or text with one
    `time velocity` pair per line) -> (times ms ascending, velocities).  Empty rows are skipped.  Raises ValueError."""
    if isinstance(source, str):
        source = [line.replace(",", " ").split() for line in source.splitlines() if line.split("#")[0].strip()]
    pairs = []
    for n, r in enumerate(source or [], 1):
        cells = list(r.values()) if isinstance(r, dict) else list(r)
        cells = [None if (c is None or (isinstance(c, str) and not c.strip()) or (isinstance(c, float) and c != c)) else c
                 for c in cells[:2]] + [None] * (2 - len(cells[:2]))
        if cells[0] is None and cells[1] is None:
            continue
        try:
            t, v = float(cells[0]), float(cells[1])
        except (TypeError, ValueError):
            raise ValueError(f"velocity row {n}: both the time (ms) and the velocity must be numbers") from None
        if v <= 0 or t < 0:
            raise ValueError(f"velocity row {n}: time must be >= 0 and velocity > 0")
        pairs.append((t, v))
    if not pairs:
        raise ValueError("the velocity table is empty - give at least one time / velocity pair")
    pairs.sort()
    t = np.array([p[0] for p in pairs])
    if len(t) > 1 and (np.diff(t) <= 0).any():
        raise ValueError("two velocity picks have the same time")
    return t, np.array([p[1] for p in pairs])


def velocity_at(t_ms: np.ndarray, vel_t: np.ndarray, vel_v: np.ndarray) -> np.ndarray:
    """Velocity function sampled at times t_ms (linear between the picks, constant outside)."""
    return np.interp(t_ms, vel_t, vel_v)


# ---------------------------------------------------------------------------
# velocity model file: picks at several locations (IL / XL, X / Y or CDP numbers)
# ---------------------------------------------------------------------------
_ROLES = {"il": "il", "inline": "il", "in-line": "il", "iline": "il", "xl": "xl", "xline": "xl", "crossline": "xl",
          "cross-line": "xl", "x": "x", "easting": "x", "y": "y", "northing": "y", "cdp": "cdp", "cmp": "cdp",
          "ensemble": "cdp", "t": "t", "time": "t", "time_ms": "t", "twt": "t", "tnmo": "t", "t0": "t",
          "v": "v", "vel": "v", "velocity": "v", "vnmo": "v", "vrms": "v", "vstack": "v"}
MODEL_HELP = ("Text / CSV file with a header row naming the columns: the location as IL + XL, or X + Y, or CDP, then TIME "
              "(ms) and VELOCITY, one pick per row, e.g.  IL, XL, TIME, VELOCITY.  Several picks per location give its "
              "function; between locations the velocity is interpolated (nearest four, by distance).  A file with only "
              "TIME and VELOCITY is a 1-D function.")


def read_velocity_file(path: str) -> dict[str, np.ndarray]:
    """Read a velocity file -> {role: array} with roles il, xl, x, y, cdp (those present), t (ms) and v.
    Raises ValueError telling the user what is wrong."""
    p = os.path.expanduser(str(path).strip())
    if not os.path.isfile(p):
        raise ValueError(f"velocity model file not found: {p}")
    lines = [ln.rstrip("\n") for ln in open(p, errors="replace") if ln.strip() and not ln.lstrip().startswith("#")]
    if len(lines) < 2:
        raise ValueError("the velocity model file needs a header row and at least one pick")

    def split(ln, delim):
        return [c.strip() for c in (ln.split(delim) if delim else ln.split())]

    head = lines[0]
    delim = "\t" if "\t" in head else "," if "," in head else ";" if ";" in head else None
    names = []
    for c in split(head, delim):
        c = "".join(ch for ch in c.lower().replace(" ", "_") if ch not in "()[]")
        c = c.split("_")[0] if c.split("_")[0] in _ROLES and c not in _ROLES else c
        names.append(_ROLES.get(c))
    if "t" not in names or "v" not in names:
        raise ValueError("the header row must name a TIME column (time / time_ms / twt) and a VELOCITY column "
                         f"(velocity / vel / vnmo); found: {split(head, delim)}")
    cols: dict[str, list] = {r: [] for r in dict.fromkeys(n for n in names if n)}
    for k, ln in enumerate(lines[1:], 2):
        cells = split(ln, delim)
        if len(cells) < len(names):
            raise ValueError(f"line {k} of the velocity file has {len(cells)} values, the header has {len(names)}")
        try:
            for role, cell in zip(names, cells):
                if role:
                    cols[role].append(float(cell))
        except ValueError:
            raise ValueError(f"line {k} of the velocity file: '{ln.strip()[:60]}' - numbers only") from None
    out = {r: np.array(v) for r, v in cols.items()}
    if out["t"].max() < 20:                                   # seconds, not ms
        out["t"] = out["t"] * 1000.0
    if (out["v"] <= 0).any() or (out["t"] < 0).any():
        raise ValueError("the velocity file has a negative time or a velocity that is not positive")
    return out


@dataclass
class VelocityModel:
    """Velocity functions picked at several locations, placed in X / Y; `field` interpolates between them."""
    keys: str                            # "IL / XL" | "X / Y" | "CDP"  - how the locations were given
    x: np.ndarray                        # location of every pick location, X / Y of the data
    y: np.ndarray
    picks: list                          # per location: (times ms, velocities)
    labels: list                         # per location: e.g. "IL 100 / XL 200"
    coverage: dict                       # fraction of the picks that fall inside the data for each key type
    source: str

    def field(self, xs, ys, t_ms) -> np.ndarray:
        """Velocity [len(xs), len(t_ms)]: the picked functions resampled to t_ms and combined by inverse-distance
        weighting of the nearest four locations."""
        xs, ys = np.atleast_1d(np.asarray(xs, float)), np.atleast_1d(np.asarray(ys, float))
        F = np.array([np.interp(t_ms, t, v) for t, v in self.picks], np.float32)
        if len(F) == 1:
            return np.broadcast_to(F[0], (len(xs), len(t_ms)))
        d = np.hypot(xs[:, None] - self.x[None, :], ys[:, None] - self.y[None, :])
        nn = np.argsort(d, axis=1)[:, :min(4, len(F))]
        w = 1.0 / np.maximum(np.take_along_axis(d, nn, 1), 1e-6) ** 2
        w /= w.sum(axis=1, keepdims=True)
        return (w[:, :, None].astype(np.float32) * F[nn]).sum(axis=1)


@dataclass
class Velocity:
    """What the NMO uses: one 1-D function for everything, or a model that varies over the survey."""
    vel_t: np.ndarray | None = None
    vel_v: np.ndarray | None = None
    model: VelocityModel | None = None

    def at(self, xs, ys, t_ms) -> np.ndarray:
        if self.model is not None:
            return self.model.field(xs, ys, t_ms)
        return np.broadcast_to(np.interp(t_ms, self.vel_t, self.vel_v).astype(np.float32), (len(np.atleast_1d(xs)), len(t_ms)))

    def describe(self, unit: str = "") -> str:
        u = f"{unit}/s" if unit else "length/s"
        if self.model is None:
            return f"1-D function, {len(self.vel_t)} picks, {self.vel_v.min():,.0f} - {self.vel_v.max():,.0f} {u}"
        cov = ", ".join(f"{k} {100 * f:.0f} %" for k, f in self.model.coverage.items())
        return (f"velocity model {Path(self.model.source).name}: {len(self.model.picks)} locations placed by "
                f"{self.model.keys} (matches with the data: {cov})")


def one_d(vel_t, vel_v) -> Velocity:
    return Velocity(vel_t=np.asarray(vel_t, float), vel_v=np.asarray(vel_v, float))


def build_velocity(raw: dict[str, np.ndarray], source: str, grid, path: str) -> Velocity:
    """Turn a parsed velocity file into a Velocity.  Locations are matched to the data by whichever information the
    file has AND the data has: IL / XL (through the IL / XL grid), X / Y (the trace coordinates) or CDP numbers (the
    CDP header word); the key that matches most picks is used.  A file without locations is a 1-D function."""
    from . import geometry

    keysets = [k for k in (("il", "xl"), ("x", "y"), ("cdp",)) if all(r in raw for r in k)]
    if not keysets:                                   # no location: one function for everything
        o = np.argsort(raw["t"], kind="stable")
        t, v = raw["t"][o], raw["v"][o]
        _, first = np.unique(t, return_index=True)
        return one_d(t[first], v[first])
    mx, my = geometry.read_midpoints(path)
    cover, xy = {}, {}
    if ("il", "xl") in keysets:
        (il0, il1), (xl0, xl1) = grid.il_range, grid.xl_range
        cover["IL / XL"] = float(np.mean((raw["il"] >= il0 - 1) & (raw["il"] <= il1 + 1) & (raw["xl"] >= xl0 - 1) & (raw["xl"] <= xl1 + 1)))
        xy["IL / XL"] = grid.xy(raw["il"], raw["xl"])
    if ("x", "y") in keysets:
        px, py = 0.02 * (mx.max() - mx.min()), 0.02 * (my.max() - my.min())
        cover["X / Y"] = float(np.mean((raw["x"] >= mx.min() - px) & (raw["x"] <= mx.max() + px) &
                                        (raw["y"] >= my.min() - py) & (raw["y"] <= my.max() + py)))
        xy["X / Y"] = (raw["x"], raw["y"])
    if ("cdp",) in keysets:
        cdp = geometry.read_cdp(path)
        vals, inv = np.unique(cdp, return_inverse=True)
        cx, cy = np.bincount(inv, weights=mx) / np.bincount(inv), np.bincount(inv, weights=my) / np.bincount(inv)
        pos = np.searchsorted(vals, raw["cdp"])
        found = (pos < len(vals)) & (vals[np.clip(pos, 0, len(vals) - 1)] == raw["cdp"])
        cover["CDP"] = float(found.mean())
        xy["CDP"] = (np.where(found, cx[np.clip(pos, 0, len(vals) - 1)], np.nan), np.where(found, cy[np.clip(pos, 0, len(vals) - 1)], np.nan))
    best = max(cover, key=lambda k: cover[k])
    if cover[best] < 0.5:
        raise ValueError("the locations in the velocity file do not match the data: " +
                         ", ".join(f"{k} {100 * f:.0f} % of the picks inside" for k, f in cover.items()) +
                         ". Check that the numbering is the one of the IL / XL grid in use (corner points) or use X / Y.")
    x, y = xy[best]
    ok = np.isfinite(x) & np.isfinite(y)
    loc = np.stack([np.round(x[ok], 3), np.round(y[ok], 3)], axis=1)
    uniq, inv = np.unique(loc, axis=0, return_inverse=True)
    t_all, v_all = raw["t"][ok], raw["v"][ok]
    label_cols = {"IL / XL": ("il", "xl"), "X / Y": ("x", "y"), "CDP": ("cdp",)}[best]
    picks, labels = [], []
    for k in range(len(uniq)):
        m = np.flatnonzero(inv.ravel() == k)
        o = np.argsort(t_all[m], kind="stable")
        t, v = t_all[m][o], v_all[m][o]
        _, first = np.unique(t, return_index=True)
        picks.append((t[first], v[first]))
        r0 = np.flatnonzero(ok)[m[0]]
        labels.append(" / ".join(f"{n.upper()} {raw[n][r0]:g}" for n in label_cols))
    return Velocity(model=VelocityModel(best, uniq[:, 0], uniq[:, 1], picks, labels, cover, source))


# ---------------------------------------------------------------------------
# NMO
# ---------------------------------------------------------------------------
def nmo_correct(data: np.ndarray, offsets: np.ndarray, dt_ms: float, vel_t: np.ndarray | None, vel_v: np.ndarray | None,
                stretch_pct: float = 30.0, v_samples: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Normal-moveout correction of traces [n, ns] recorded at `offsets` [n] (same length unit as the velocity):
    output sample t0 takes the input at t = sqrt(t0^2 + (x / v(t0))^2), linear interpolation.
    The velocity is the 1-D function (vel_t, vel_v) or, when `v_samples` [n, ns] (or [ns]) is given, that array.
    Samples stretched by more than `stretch_pct` % (0 = no mute) or falling beyond the record are muted.
    Returns (corrected traces, live mask [n, ns] = samples that may enter a stack)."""
    n, ns = data.shape
    t0 = np.arange(ns) * dt_ms / 1000.0                                   # s
    v = np.asarray(v_samples if v_samples is not None else velocity_at(np.arange(ns) * dt_ms, vel_t, vel_v), float)
    v = v[None, :] if v.ndim == 1 else v
    tq = np.sqrt(t0[None, :] ** 2 + (np.abs(offsets)[:, None] / v) ** 2)              # s
    pos = tq / (dt_ms / 1000.0)                                           # fractional input sample
    i0 = np.floor(pos).astype(np.int64)
    live = i0 < ns - 1
    frac = (pos - i0).astype(np.float32)
    i0 = np.clip(i0, 0, ns - 2)
    a = np.take_along_axis(data, i0, axis=1)
    b = np.take_along_axis(data, i0 + 1, axis=1)
    out = a * (1.0 - frac) + b * frac
    if stretch_pct and stretch_pct > 0:
        live &= ~(tq > (1.0 + stretch_pct / 100.0) * t0[None, :])       # stretch = tq / t0 - 1; at t0 = 0 any offset is infinite
    out[~live] = 0.0
    return out.astype(np.float32, copy=False), live


# ---------------------------------------------------------------------------
# the flow: separate stages that hand their result on (a session per SEG-Y file)
# ---------------------------------------------------------------------------
@dataclass
class CmpSorted:
    """Result of the CMP Sort stage: every trace of the file assigned to a CMP bin of an IL / XL grid.  (No samples are
    read or copied - the index is what the later stages read the traces by.)"""
    path: str
    grid: object                     # the IL / XL grid the bins belong to (grid.Grid)
    bins: Bins
    key: tuple                       # cache key of the bins

    def describe(self) -> str:
        n_i, n_j = self.bins.shape
        return f"{n_i} x {n_j} bins, {self.bins.alignment}, IL / XL grid from {self.grid.source}"


@dataclass
class NmoSetup:
    """Result of the NMO Correction stage: the velocity and the mute that the Stack stage will apply."""
    velocity: "Velocity"
    stretch_pct: float
    note: str


_FLOW: dict[str, dict] = {}


def flow_state(path: str) -> dict:
    """What the stages of the flow have produced for this file so far: keys "sorted" (CmpSorted), "nmo" (NmoSetup),
    "stack" (StackResult) and "saved" (list of files).  A stage that runs again drops everything after it; a changed
    file starts a new flow.  Each stage reads only the stage before it - no stage calls another."""
    p = segy_io.norm_path(path)
    mtime = os.path.getmtime(p)
    st = _FLOW.get(p)
    if st is None or st.get("mtime") != mtime:
        st = _FLOW[p] = {"mtime": mtime}
    return st


FLOW_ORDER = ("sorted", "nmo", "stack", "saved")


def restart_from(path: str, stage: str) -> dict:
    """Forget the results of every stage AFTER `stage` (it has just been run again) and return the flow state."""
    st = flow_state(path)
    for k in FLOW_ORDER[FLOW_ORDER.index(stage) + 1:]:
        st.pop(k, None)
    return st


# ---------------------------------------------------------------------------
# CMP gather of one bin
# ---------------------------------------------------------------------------
@dataclass
class CmpGather:
    il: float                        # IL / XL numbers at the bin centre
    xl: float
    trace_index: np.ndarray          # file trace indices (0-based), sorted by offset
    data: np.ndarray                 # float32 [n, ns], sorted by offset
    offsets: np.ndarray              # signed header offsets, same order
    ffid: np.ndarray                 # shot of each trace
    dt_ms: float
    headers: dict[str, np.ndarray]   # the header words plot_gather needs

    @property
    def ntr(self) -> int:
        return self.data.shape[0]


def _offset(raw: np.ndarray, order: str) -> np.ndarray:
    return header_word(raw, *TRACE_HEADER_FIELDS["offset"], order)


def best_bin(bins: Bins) -> tuple[int, int]:
    i, j = np.unravel_index(int(bins.counts.argmax()), bins.counts.shape)
    return int(i), int(j)


def cmp_gather(path: str, bins: Bins, il: float = 0, xl: float = 0) -> CmpGather:
    """The traces of the bin containing IL / XL (0, 0 = the bin with the highest fold), read from the file and sorted
    by offset.  Raises LookupError if the bin is empty or outside the survey."""
    sgy = segy_io.open_segy(path)
    if il == 0 and xl == 0:
        i, j = best_bin(bins)
    else:
        i, j = bins.locate(il, xl)
    n_i, n_j = bins.shape
    if not (0 <= i < n_i and 0 <= j < n_j) or bins.counts[i, j] == 0:
        raise LookupError(f"no traces in the bin IL {il:g} / XL {xl:g}"
                          f" (IL runs {bins.il_of(0):g}-{bins.il_of(n_i - 1):g}, XL {bins.xl_of(0):g}-{bins.xl_of(n_j - 1):g})")
    idx = np.flatnonzero((bins.ii == i) & (bins.jj == j))
    raw, data = sgy.read_traces_at(idx)
    off = _offset(raw, sgy.order)
    order = np.argsort(np.abs(off), kind="stable")
    raw, data, idx, off = raw[order], data[order], idx[order], off[order]
    words = {k: header_word(raw, *TRACE_HEADER_FIELDS[k], sgy.order) for k in ("ffid", "chan", "offset", "rec_line", "rec_stn", "cdp")}
    return CmpGather(float(bins.il_of(i)), float(bins.xl_of(j)), idx, data, off, words["ffid"], sgy.dt_ms, words)


# ---------------------------------------------------------------------------
# stack of one line
# ---------------------------------------------------------------------------
@dataclass
class StackResult:
    section: np.ndarray              # float32 [n_pos, ns]
    positions: np.ndarray            # XL numbers (inline stack) or IL numbers (crossline stack) of the columns
    fold: np.ndarray                 # traces stacked in each column
    line_kind: str                   # "IL" | "XL"
    line_no: float                   # the IL (or XL) number of the line
    dt_ms: float
    velocity: Velocity               # what was used
    vel_field: np.ndarray            # float32 [n_pos, ns]: the velocity applied at every position
    stretch_pct: float
    unit: str
    alignment: str
    n_traces: int

    @property
    def across(self) -> str:         # name of the number that runs along the section
        return "XL" if self.line_kind == "IL" else "IL"


def auto_line(bins: Bins, kind: str) -> int:
    """Row (inline) / column (crossline) index with the most traces."""
    return int(np.argmax(bins.counts.sum(axis=1 if kind == "IL" else 0)))


def line_index(bins: Bins, kind: str, number: float) -> int:
    """Row / column index of IL (or XL) `number`; 0 = the line with the most traces."""
    if not number:
        return auto_line(bins, kind)
    i, j = bins.locate(number, number)
    idx = i if kind == "IL" else j
    lim = bins.shape[0 if kind == "IL" else 1]
    if not 0 <= idx < lim:
        lo, hi = (bins.il_of(0), bins.il_of(lim - 1)) if kind == "IL" else (bins.xl_of(0), bins.xl_of(lim - 1))
        raise LookupError(f"{kind} {number:g} is outside the survey ({kind} runs {lo:g}-{hi:g})")
    return idx


def stack_line(path: str, bins: Bins, kind: str, number: float, velocity: Velocity, stretch_pct: float = 30.0,
               chunk: int = 400) -> StackResult:
    """NMO-correct and stack every bin along one inline (kind "IL") or crossline ("XL").  `velocity` is a 1-D function
    for the whole line or a model that gives each bin its own function (placed by the bin's X / Y)."""
    sgy = segy_io.open_segy(path)
    k = line_index(bins, kind, number)
    inline = kind == "IL"
    across = bins.jj if inline else bins.ii
    on_line = (bins.ii == k) if inline else (bins.jj == k)
    idx = np.flatnonzero(on_line)
    if len(idx) == 0:
        raise LookupError(f"no traces on {kind} {(bins.il_of(k) if inline else bins.xl_of(k)):g}")
    lo, hi = int(across[idx].min()), int(across[idx].max())
    n_pos = hi - lo + 1
    ns = sgy.ns
    acc = np.zeros((n_pos, ns), np.float32)
    hits = np.zeros((n_pos, ns), np.float32)
    ax_num = bins.xl_of(np.arange(lo, hi + 1)) if inline else bins.il_of(np.arange(lo, hi + 1))
    bx, by = bins.grid.xy(np.full(n_pos, bins.il_of(k)) if inline else ax_num, ax_num if inline else np.full(n_pos, bins.xl_of(k)))
    vfield = np.asarray(velocity.at(bx, by, np.arange(ns) * sgy.dt_ms), np.float32)              # [n_pos, ns]
    fold = np.bincount(across[idx] - lo, minlength=n_pos)
    order = np.argsort(idx, kind="stable")                        # read in file order
    idx = idx[order]
    for c0 in range(0, len(idx), chunk):
        sel = idx[c0:c0 + chunk]
        raw, data = sgy.read_traces_at(sel)
        col = across[sel] - lo
        out, live = nmo_correct(data, _offset(raw, sgy.order).astype(np.float64), sgy.dt_ms, None, None, stretch_pct,
                                v_samples=vfield[col])
        o = np.argsort(col, kind="stable")
        col, w, m = col[o], out[o], live[o].astype(np.float32)
        starts = np.r_[0, np.flatnonzero(np.diff(col)) + 1]
        acc[col[starts]] += np.add.reduceat(w, starts, axis=0)
        hits[col[starts]] += np.add.reduceat(m, starts, axis=0)
    section = np.divide(acc, hits, out=np.zeros_like(acc), where=hits > 0)
    if inline:
        pos, line_no = bins.xl_of(np.arange(lo, hi + 1)), float(bins.il_of(k))
    else:
        pos, line_no = bins.il_of(np.arange(lo, hi + 1)), float(bins.xl_of(k))
    from . import geometry
    return StackResult(section, np.asarray(pos, float), fold, kind, line_no, sgy.dt_ms, velocity, vfield,
                       float(stretch_pct), geometry.read_geometry(path, 60).unit, bins.alignment, int(len(idx)))


# ---------------------------------------------------------------------------
# products: the stack as SEG-Y, the velocity function as text
# ---------------------------------------------------------------------------
def velocity_tag(velocity: Velocity, stretch_pct: float) -> str:
    """Short code of a velocity (function or model file) + mute, so files of different velocities never share a name."""
    if velocity.model is None:
        key = ",".join(f"{t:g}:{v:g}" for t, v in zip(velocity.vel_t, velocity.vel_v)) + f"|{stretch_pct:g}"
        return "v" + hashlib.md5(key.encode()).hexdigest()[:6]
    src = velocity.model.source
    st = os.stat(src) if os.path.isfile(src) else None
    key = f"{src}|{st.st_mtime if st else 0}|{st.st_size if st else 0}|{stretch_pct:g}"
    return "vm" + hashlib.md5(key.encode()).hexdigest()[:6]


def write_velocity(out_path: str, res: StackResult, source_path: str) -> str:
    """The velocity a stack used, as CSV with a comment header: the function for a 1-D velocity; for a model the
    velocity at every position along the line, every 200 ms."""
    u = f"{res.unit}/s" if res.unit else "length/s"
    v = res.velocity
    lines = [
        "# NMO velocity used for the stack" + (" (brute stack: one function for the whole line)" if v.model is None else " (velocity model)"),
        f"# source file: {source_path}",
        f"# line: {res.line_kind} {res.line_no:g}   ({len(res.positions)} {res.across} bins, {res.n_traces:,} traces)",
        f"# velocity unit: {u}; time in ms; " + ("linear between picks, constant outside" if v.model is None else
                                               f"model file {v.model.source}, placed by {v.model.keys}"),
        f"# NMO stretch mute: {res.stretch_pct:g} %",
    ]
    if v.model is None:
        lines += ["time_ms,velocity"] + [f"{t:g},{x:g}" for t, x in zip(v.vel_t, v.vel_v)]
    else:
        step = max(1, int(round(200 / res.dt_ms)))
        lines.append(f"{res.across},time_ms,velocity")
        for k, pos in enumerate(res.positions):
            lines += [f"{pos:g},{i * res.dt_ms:g},{res.vel_field[k, i]:.1f}" for i in range(0, res.vel_field.shape[1], step)]
    Path(out_path).write_text("\n".join(lines) + "\n")
    return out_path


def write_stack_segy(out_path: str, res: StackResult, source_path: str, fmt: str = "ibm") -> str:
    """The stack section as a SEG-Y file: the source file's text and binary headers (with notes on the stack in the
    free text cards), then one trace per position along the line, with the IL / XL, the fold (bytes 33-34) and the
    bin-centre X / Y in the trace header."""
    from .segy_write import build_head, pack_traces

    sgy = segy_io.open_segy(source_path)
    notes = [f"BRUTE STACK {res.line_kind} {res.line_no:g}, {len(res.positions)} TRACES ALONG {res.across}",
             ("NMO V(T) " + (f"{len(res.velocity.vel_t)} PICKS" if res.velocity.model is None else
                             f"MODEL {len(res.velocity.model.picks)} LOCS") + f", STRETCH MUTE {res.stretch_pct:g} PCT")]
    n, ns = res.section.shape
    raw = np.zeros((n, segy_io.TRACE_HEADER_BYTES), np.uint8)

    def put(byte1: int, code: str, values):
        size = np.dtype(code).itemsize
        raw[:, byte1 - 1: byte1 - 1 + size] = np.ascontiguousarray(
            np.asarray(values).astype(sgy.order + code)).view(np.uint8).reshape(n, size)

    inline = res.line_kind == "IL"
    up = lambda v: np.floor(np.asarray(v, float) + 0.5)          # bin centres at x.5: round half UP (np.round would repeat numbers)
    il = np.full(n, res.line_no) if inline else res.positions
    xl = res.positions if inline else np.full(n, res.line_no)
    seq = np.arange(1, n + 1)
    put(1, "i4", seq); put(5, "i4", seq)
    put(9, "i4", up(np.full(n, res.line_no)))                     # "field record" = the line number
    put(21, "i4", up(res.positions))                              # ensemble number = position along the line
    put(25, "i4", np.ones(n)); put(29, "i2", np.ones(n)); put(31, "i2", np.ones(n))
    put(33, "i2", np.minimum(res.fold, 32767))                    # number of traces stacked = fold
    put(115, "i2", np.full(n, ns)); put(117, "i2", np.full(n, round(res.dt_ms * 1000)))
    put(189, "i4", up(il)); put(193, "i4", up(xl))
    body = pack_traces(raw, res.section, fmt, sgy.order)
    with open(out_path, "wb") as f:
        f.write(bytes(build_head(sgy, notes, fmt)))
        body.tofile(f)
    return out_path
