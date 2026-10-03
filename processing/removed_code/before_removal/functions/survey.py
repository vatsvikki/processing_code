"""Dead / bad trace scan of a whole SEG-Y file, shot by shot (no UI, no plotting)."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

from . import detection, segy_io


@dataclass
class SurveyResult:
    path: str
    per_shot: list[dict]             # ffid, traces, dead, bad, good
    flagged: list[dict]              # one row per dead / bad trace in the scanned shots
    params: dict

    @property
    def n_traces(self) -> int:
        return sum(r["traces"] for r in self.per_shot)

    @property
    def n_dead(self) -> int:
        return sum(r["dead"] for r in self.per_shot)

    @property
    def n_bad(self) -> int:
        return sum(r["bad"] for r in self.per_shot)


def _scan_shot(path: str, ffid: int, i0: int, n: int, qc_kw: dict) -> tuple[dict, list[dict]]:
    sgy = segy_io.open_segy(path)
    hdr, data = sgy.read_traces(i0, i0 + n)
    g = segy_io.ShotGather(ffid=ffid, i0=i0, data=data, headers=hdr, dt_ms=sgy.dt_ms)
    res = detection.run_qc(g, **qc_kw)
    n_dead, n_bad = int(res.dead.sum()), int(res.bad.sum())
    rows = [{"ffid": ffid, **r} for r in detection.flagged_table(g, res)]
    return {"ffid": ffid, "traces": n, "dead": n_dead, "bad": n_bad, "good": n - n_dead - n_bad}, rows


@lru_cache(maxsize=2)
def _scan_cached(path: str, mtime: float, ffid_from: int, ffid_to: int, workers: int, qc_items: tuple) -> SurveyResult:
    qc_kw = dict(qc_items)
    idx = segy_io.shot_index(path)
    sel = [k for k in range(len(idx)) if ffid_from <= idx.ffids[k] <= ffid_to]
    if not sel:
        raise LookupError(f"no shots with FFID between {ffid_from} and {ffid_to} "
                          f"(file has {int(idx.ffids[0])} to {int(idx.ffids[-1])})")
    jobs = [(path, int(idx.ffids[k]), int(idx.first[k]), int(idx.count[k]), qc_kw) for k in sel]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(pool.map(lambda a: _scan_shot(*a), jobs))
    return SurveyResult(
        path=path,
        per_shot=[r[0] for r in results],
        flagged=[row for r in results for row in r[1]],
        params={"ffid_from": ffid_from, "ffid_to": ffid_to, **qc_kw},
    )


def scan_survey(path: str, *, ffid_from: int = 0, ffid_to: int = 0, workers: int = 0, **qc_kw) -> SurveyResult:
    """Run dead + bad QC on every shot with ffid_from <= FFID <= ffid_to (0 = first / last shot).

    qc_kw are the `detection.run_qc` thresholds.  Results are cached in memory, so re-running
    with the same settings is instant.
    """
    path = os.path.expanduser(str(path).strip())
    idx = segy_io.shot_index(path)
    lo = int(ffid_from) if ffid_from and ffid_from > 0 else int(idx.ffids[0])
    hi = int(ffid_to) if ffid_to and ffid_to > 0 else int(idx.ffids[-1])
    workers = int(workers) if workers and workers > 0 else min(8, os.cpu_count() or 1)
    qc_kw.pop("sort_by", None)
    return _scan_cached(path, os.path.getmtime(path), lo, hi, workers, tuple(sorted(qc_kw.items())))
