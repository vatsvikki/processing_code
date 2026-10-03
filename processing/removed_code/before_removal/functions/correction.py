"""Dead-trace correction: interpolate or remove (no UI, no plotting)."""
from __future__ import annotations

import numpy as np

from .segy_io import ShotGather

MODES = ("interpolate", "remove")


def correct_dead(gather: ShotGather, dead: np.ndarray, mode: str = "interpolate") -> tuple[ShotGather, np.ndarray]:
    """Return (corrected gather, bool mask over the corrected gather for marking on a plot).

    remove       dead traces are deleted, the gather gets shorter; the mask marks the trace that now
                 sits at each removal point (the first kept trace after a removed one)
    interpolate  each dead trace is replaced by the linear interpolation of the nearest live
                 traces to its left and right *on the same receiver line* (file/channel order).
                 At the end of a line, or if a line has no live trace, the nearest live
                 trace is copied instead.
    """
    if mode not in MODES:
        raise ValueError(f"unknown correction {mode!r}; use one of {MODES}")
    dead = np.asarray(dead, bool)
    if mode == "remove":
        keep = ~dead
        if not keep.any():
            raise ValueError("every trace is dead - nothing left after removal")
        hdr = {k: v[keep] for k, v in gather.headers.items()}
        mask = np.zeros(int(keep.sum()), bool)
        after = np.cumsum(keep)                        # for a dead trace: index of the next kept trace
        mask[np.minimum(after[dead], len(mask) - 1)] = True
        return ShotGather(gather.ffid, gather.i0, gather.data[keep], hdr, gather.dt_ms), mask

    data = gather.data.copy()
    live = np.flatnonzero(~dead)
    if not len(live) and dead.any():
        raise ValueError("every trace is dead - nothing to interpolate from")
    line = gather.headers["rec_line"]
    for i in np.flatnonzero(dead):
        same = live[line[live] == line[i]]
        pool = same if len(same) else live
        k = np.searchsorted(pool, i)
        left = pool[k - 1] if k > 0 else None
        right = pool[k] if k < len(pool) else None
        if left is not None and right is not None:
            w = (i - left) / (right - left)
            data[i] = (1.0 - w) * data[left] + w * data[right]
        else:
            data[i] = data[left if left is not None else right]
    return ShotGather(gather.ffid, gather.i0, data, gather.headers, gather.dt_ms), dead.copy()
