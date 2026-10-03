"""CDP sort of the whole survey (as in the brute-stack notebook, cdp_nmo_stack_marimo.py, "2. CDP sorting"): every
trace sorted by CDP number, then by |offset| within each CDP, and the gather of one CDP read from the file.

Only the trace headers are read for the sort (once per file version, cached), so changing the CDP shown is quick.
No UI here - the "CDP Sort" display step of the Flow (steps.py) shows the results.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path

import numpy as np

from . import segy_io
from .progress import report
from .geometry import _REC_X, _REC_Y, _SRC_X, _SRC_Y, _xy
from .segy_io import TRACE_HEADER_FIELDS, ShotGather, header_word

_CHUNK = 100_000                                  # trace headers per read (~24 MB)


@dataclass
class CdpIndex:
    order: np.ndarray        # trace indices sorted by CDP, then |offset| (the CDP-sorted survey)
    cdps: np.ndarray         # the distinct CDP numbers, ascending
    first: np.ndarray        # position in `order` of each CDP's first trace
    fold: np.ndarray         # traces per CDP
    off_min: np.ndarray      # smallest / largest |offset| per CDP
    off_max: np.ndarray
    mid_x: np.ndarray        # mean source-receiver midpoint per CDP (coordinate scalar applied)
    mid_y: np.ndarray
    traces: int
    offsets: np.ndarray | None = None   # |offset| of every trace in `order` (each CDP's nearest first)

    def position(self, cdp: int) -> int:
        """Index (into cdps) of the CDP shown for a requested number: 0 = the highest fold, else the nearest CDP."""
        if not cdp:
            return int(np.argmax(self.fold))
        k = int(np.searchsorted(self.cdps, cdp))
        if k >= len(self.cdps) or (k > 0 and abs(self.cdps[k - 1] - cdp) <= abs(self.cdps[k] - cdp)):
            k -= 1
        return max(k, 0)

    def traces_of(self, k: int) -> np.ndarray:
        """File trace indices of CDP number cdps[k], nearest offset first."""
        return self.order[self.first[k]:self.first[k] + self.fold[k]]


def _index_file(path: str, mtime: float) -> Path:
    """Where the index of this file version is kept (output/index/), so it is read from the headers only once."""
    from .saving import DEFAULT_DIR
    key = hashlib.sha1(f"{path}|{os.path.getsize(path)}|{mtime}".encode()).hexdigest()[:16]
    return Path(DEFAULT_DIR) / "index" / f"{Path(path).stem}_{key}.npz"


@lru_cache(maxsize=2)
def _index(path: str, mtime: float) -> CdpIndex:
    store = _index_file(path, mtime)
    try:
        z = np.load(store)
        return CdpIndex(z["order"], z["cdps"], z["first"], z["fold"], z["off_min"], z["off_max"], z["mid_x"],
                        z["mid_y"], int(z["traces"]), z["offsets"])
    except (OSError, KeyError, ValueError):
        pass
    idx = _scan(path)
    try:
        store.parent.mkdir(parents=True, exist_ok=True)
        tmp = store.with_suffix(".tmp.npz")
        np.savez(tmp, order=idx.order, cdps=idx.cdps, first=idx.first, fold=idx.fold, off_min=idx.off_min,
                 off_max=idx.off_max, mid_x=idx.mid_x, mid_y=idx.mid_y, traces=idx.traces, offsets=idx.offsets)
        os.replace(tmp, store)
    except OSError:
        pass                                          # no room to keep it: it is simply read again next time
    return idx


def _scan(path: str) -> CdpIndex:
    """Read every trace header and sort (the slow part, a few minutes on a large file)."""
    sgy = segy_io.open_segy(path)
    n = sgy.ntraces
    cdp, off = np.empty(n, np.int64), np.empty(n, np.int64)
    mx, my = np.empty(n, np.float64), np.empty(n, np.float64)
    for i0 in range(0, n, _CHUNK):
        raw = sgy.read_headers(i0, i0 + _CHUNK)
        s = slice(i0, i0 + len(raw))
        cdp[s] = header_word(raw, *TRACE_HEADER_FIELDS["cdp"], sgy.order)
        off[s] = np.abs(header_word(raw, *TRACE_HEADER_FIELDS["offset"], sgy.order))
        sx, sy = _xy(raw, sgy.order, _SRC_X, _SRC_Y)
        gx, gy = _xy(raw, sgy.order, _REC_X, _REC_Y)
        mx[s], my[s] = (sx + gx) / 2.0, (sy + gy) / 2.0
        report(i0 + len(raw), n, "Reading every trace header (CDP sort of the whole survey)")
    # CDP first, then |offset| within a CDP (lexsort: the last key is the primary one) - a plain argsort on the CDP
    # alone would leave each CDP's traces in recorded order instead of near-to-far
    order = np.lexsort((off, cdp))
    cdp_s, off_s = cdp[order], off[order]
    cdps, first, fold = np.unique(cdp_s, return_index=True, return_counts=True)
    last = first + fold - 1
    sum_x = np.add.reduceat(mx[order], first)
    sum_y = np.add.reduceat(my[order], first)
    return CdpIndex(order, cdps, first, fold, off_s[first], off_s[last], sum_x / fold, sum_y / fold, n, off_s)


def cdp_index(path: str) -> CdpIndex:
    sgy = segy_io.open_segy(path)
    return _index(sgy.path, os.path.getmtime(sgy.path))


def read_cdp_gather(path: str, idx: CdpIndex, k: int) -> ShotGather:
    """The traces of CDP cdps[k] straight from the file (nearest offset first), as a gather the Flow can plot."""
    sgy = segy_io.open_segy(path)
    raw, data = sgy.read_traces_at(idx.traces_of(k))
    headers = {name: header_word(raw, byte, code, sgy.order) for name, (byte, code) in TRACE_HEADER_FIELDS.items()}
    return ShotGather(int(idx.cdps[k]), 0, data, headers, sgy.dt_ms)


@lru_cache(maxsize=2)
def _top_labels(path: str, mtime: float, n: int) -> tuple[str, ...]:
    from . import grid as grid_mod, nmo
    idx = _index(path, mtime)
    order = np.argsort(-idx.fold, kind="stable")[:n]
    tr = None
    try:
        tr = nmo.fit_corners(grid_mod.parse_corners(grid_mod.corner_table(path)))
    except Exception:
        tr = None
    out = []
    for k in order:
        ilxl = ""
        if tr is not None:
            xl, il = nmo.to_data_ilxl(tr, float(idx.mid_x[k]), float(idx.mid_y[k]))
            ilxl = f"IL {il:.0f}, XL {xl:.0f}, "
        out.append(f"CDP {int(idx.cdps[k])}  ({ilxl}fold {int(idx.fold[k])})")
    return tuple(out)


def top_cdp_labels(path: str, n: int = 200) -> list[str]:
    """The n CDPs with the highest fold, as "CDP 421506  (IL 198, XL 407, fold 28)" - IL / XL from this file's grid
    (its saved corner table, or the headers' IL / XL) when it has one. Reads every trace header the first time."""
    sgy = segy_io.open_segy(path)
    return list(_top_labels(sgy.path, os.path.getmtime(sgy.path), n))


def cdp_from_label(label) -> int:
    """The CDP number of a label made by top_cdp_labels (0 when there is none)."""
    import re
    m = re.match(r"\s*CDP\s+(-?\d+)", str(label or ""))
    return int(m.group(1)) if m else 0
