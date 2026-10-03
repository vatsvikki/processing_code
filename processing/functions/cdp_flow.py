"""The CDP part of a flow: "NMO Correction" and "CDP Stack" work on CDP gathers made of the traces the shot steps above
them produced (decon, filters ...), not on the raw file. No UI here. (The CDP sort of the raw file is the 🛠 Tools ->
CDP Sort tool, cdp_sort_tool.py; write_cdp_sorted is its writer.)

A flow is   [shot steps ...]  then  [CDP steps ...]   - processing steps must come before the first CDP step.

* Whole data (BatchJob): the shot steps are run on every shot and written to an intermediate file in the output
  folder (kept per flow, so the CDP steps and later runs reuse it); then the last CDP step makes the product - the
  NMO-corrected CDP gathers or the stack.
* Preview (one CDP on screen): the CDP's traces come from that intermediate file when the whole-data run made it, else
  the shots the CDP's traces come from are processed on the spot (a few seconds per CDP).
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from . import cdp_sort, segy_io
from .pipeline import get_step, run_steps
from .progress import report
from .saving import DEFAULT_DIR

CDP_STEP_KEYS = ("nmo_correction_step", "cdp_stack_step")


def is_cdp_step(key: str) -> bool:
    return key in CDP_STEP_KEYS


def processing_steps(specs: list[dict]) -> list[dict]:
    """The steps that change the data (no Display / CDP steps) - what the CDP gathers are made from - each with ALL its
    parameters filled in (defaults for those not given), so the same flow always has the same intermediate file however
    it was written down."""
    from .pipeline import _params_for
    return [{"step": s["step"], "params": _params_for(get_step(s["step"]), s.get("params"))}
            for s in specs or [] if not is_cdp_step(s["step"]) and get_step(s["step"]).category != "Display"]


def split_flow(specs: list[dict]) -> tuple[list[dict], list[dict], list[str]]:
    """(shot steps before the first CDP step, the CDP steps, problems): a processing step after a CDP step is a
    problem - the CDP gathers are made from what the shot steps produced."""
    first = next((i for i, s in enumerate(specs) if is_cdp_step(s["step"])), len(specs))
    shot, cdp, problems = list(specs[:first]), [], []
    for s in specs[first:]:
        if is_cdp_step(s["step"]):
            cdp.append(s)
        elif get_step(s["step"]).category != "Display":
            problems.append(f"'{get_step(s['step']).label}' comes after a CDP step - move it above "
                            f"'{get_step(specs[first]['step']).label}' (CDP gathers are made from what the shot steps "
                            "above them produced)")
    return shot, cdp, problems


def _flow_signature(path: str, specs: list[dict], ffid_from: int = 0, ffid_to: int = 0) -> str:
    sgy = segy_io.open_segy(path)
    key = json.dumps([sgy.path, os.path.getmtime(sgy.path), sgy.ntraces, specs, int(ffid_from or 0), int(ffid_to or 0)],
                     sort_keys=True, default=str)
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def processed_path(path: str, specs: list[dict], ffid_from: int = 0, ffid_to: int = 0) -> str:
    """Where the whole-data run keeps the output of these processing steps (one file per flow and FFID range)."""
    sig = _flow_signature(path, processing_steps(specs), ffid_from, ffid_to)
    return str(Path(DEFAULT_DIR) / "flows" / f"{Path(path).stem}_{sig}.sgy")


def processed_file(p: str) -> str | None:
    """The finished file of the intermediate `p` (the whole-data result of some processing steps), or None.
    Usually `p` itself; when that result is the flow's final output (written to the user's output file) the
    record next to `p` points there - and is trusted only while that file is unchanged."""
    try:
        m = json.loads(Path(p + ".json").read_text())
    except (OSError, ValueError):
        return None
    f = m.get("file") or p
    if not m.get("done") or not os.path.exists(f):
        return None
    if m.get("file") and (os.path.getsize(f), int(os.path.getmtime(f))) != (m.get("size"), m.get("mtime")):
        return None                                      # the output file was replaced since
    return f


def processed_ready(p: str) -> bool:
    return processed_file(p) is not None


def mark_processed(p: str, info: dict, file: str | None = None) -> None:
    """Record that the intermediate `p` is finished (its data in `file` when that is not `p` itself)."""
    extra = {"file": file, "size": os.path.getsize(file), "mtime": int(os.path.getmtime(file))} if file else {}
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p + ".json").write_text(json.dumps({"done": True, **info, **extra}, indent=2, default=str))


def find_processed(path: str, specs: list[dict]) -> str | None:
    """A finished intermediate file of these processing steps, for any FFID range (the whole file first)."""
    full = processed_path(path, specs)
    hit = processed_file(full)
    if hit:
        return hit
    stem = Path(full).name.rsplit("_", 1)[0]
    want = processing_steps(specs)
    for meta in sorted(Path(full).parent.glob(f"{stem}_*.sgy.json")):
        try:
            m = json.loads(meta.read_text())
        except (OSError, ValueError):
            continue
        if m.get("source") == segy_io.open_segy(path).path and m.get("steps") == want:
            hit = processed_file(str(meta)[:-5])
            if hit:
                return hit
    return None


def resumable(proc: list[dict]) -> bool:
    """Can a later step carry on from the saved result of these processing steps? Not when a QC step among them
    flagged traces that no Correct Dead Traces after it used yet: the flags are not stored in the file."""
    last_fix = max((i for i, s in enumerate(proc) if s["step"] == "correct_dead"), default=-1)
    return not any(get_step(s["step"]).category == "QC" for s in proc[last_fix + 1:])


def best_start(path: str, proc: list[dict], ffid_from: int = 0, ffid_to: int = 0) -> tuple[str | None, int]:
    """(file, j): the saved whole-data result of the longest first j processing steps a run can start from
    (None, 0 = start from the raw file)."""
    for j in range(len(proc), 0, -1):
        if j < len(proc) and not resumable(proc[:j]):
            continue
        f = processed_file(processed_path(path, proc[:j], ffid_from, ffid_to))
        if f:
            return f, j
    return None, 0


def drop_stale(path: str, proc: list[dict]) -> list[str]:
    """Delete the intermediate files of this input that no step of the current flow can use any more (made by
    other settings / steps): each is as big as the input. Results of the current flow's first steps are kept."""
    src = segy_io.open_segy(path).path
    folder = Path(DEFAULT_DIR) / "flows"
    gone = []
    for meta in folder.glob(f"{Path(path).stem}_*.sgy.json"):
        try:
            m = json.loads(meta.read_text())
        except (OSError, ValueError):
            continue
        steps = m.get("steps") or []
        if m.get("source") != src or steps == proc[:len(steps)]:
            continue
        data = Path(str(meta)[:-5])
        for f in (data, meta, Path(str(data)[:-4] + "_report.csv")):
            try:
                f.unlink()
            except OSError:
                pass
        gone.append(str(data))
    return gone


# ---------------------------------------------------------------------------------------------------------------------
# preview: the CDP gathers of the data as processed by the steps above
# ---------------------------------------------------------------------------------------------------------------------
_SHOT_CACHE: OrderedDict = OrderedDict()


def _processed_shot(path: str, ffid: int, specs: list[dict], sig: str):
    key = (path, sig, int(ffid))
    hit = _SHOT_CACHE.get(key)
    if hit is not None:
        _SHOT_CACHE.move_to_end(key)
        return hit
    final = run_steps(segy_io.read_shot(path, int(ffid)), specs, path=path, batch=True)[-1].state
    hit = (final.gather, np.asarray(final.src))
    _SHOT_CACHE[key] = hit
    while len(_SHOT_CACHE) > 96:
        _SHOT_CACHE.popitem(last=False)
    return hit


class CdpSource:
    """Where a CDP step gets its CDP gathers from, for the flow above it (state.flow)."""

    def __init__(self, path: str, flow_so_far: list[dict]):
        self.path = path
        self.specs = processing_steps(flow_so_far)
        self.labels = [get_step(s["step"]).label for s in self.specs]
        self.processed = find_processed(path, self.specs) if self.specs else None
        self.mode = "raw" if not self.specs else ("processed" if self.processed else "on the spot")
        self.file = self.processed or path
        self.idx = cdp_sort.cdp_index(self.file)
        self._sig = _flow_signature(path, self.specs) if self.specs else ""

    def describe(self) -> str:
        if self.mode == "raw":
            return "CDP gathers of the raw data (no processing step above this one)."
        steps = " → ".join(self.labels)
        if self.mode == "processed":
            return f"CDP gathers of the data processed by {steps} (the whole-data run's result)."
        return (f"CDP gathers of the data processed by {steps} - the shots of this CDP were processed on the spot "
                "(run the flow on the whole data once to make every CDP instant).")

    def gather(self, k: int) -> segy_io.ShotGather:
        """CDP number cdps[k], nearest offset first, with the steps above applied."""
        if self.mode != "on the spot":
            return cdp_sort.read_cdp_gather(self.file, self.idx, k)
        trace_idx = self.idx.traces_of(k)
        si = segy_io.shot_index(self.path)
        shot_pos = np.searchsorted(si.first, trace_idx, side="right") - 1
        rows, datas, heads = [], [], []
        shots = np.unique(shot_pos)
        for n, sp in enumerate(shots, 1):
            g, src = _processed_shot(self.path, int(si.ffids[sp]), self.specs, self._sig)
            where = {int(s): r for r, s in enumerate(src)}
            for j in np.flatnonzero(shot_pos == sp):
                r = where.get(int(trace_idx[j] - si.first[sp]))
                if r is not None:                      # (a trace a step removed is not in the gather)
                    rows.append(j)
                    datas.append(g.data[r])
                    heads.append({key: v[r] for key, v in g.headers.items()})
            report(n, len(shots), f"Processing the {len(shots)} shots of CDP {int(self.idx.cdps[k])}")
        order = np.argsort(rows, kind="stable")        # back to nearest-offset-first
        data = np.vstack([datas[i] for i in order]) if datas else np.zeros((0, segy_io.open_segy(self.path).ns), np.float32)
        keys = heads[0].keys() if heads else ()
        headers = {key: np.array([heads[i][key] for i in order]) for key in keys}
        return segy_io.ShotGather(int(self.idx.cdps[k]), 0, data, headers, segy_io.open_segy(self.path).dt_ms)


# ---------------------------------------------------------------------------------------------------------------------
# whole data: the CDP products
# ---------------------------------------------------------------------------------------------------------------------
def _records(sgy: segy_io.SegyFile) -> np.memmap:
    return np.memmap(sgy.path, dtype=np.uint8, mode="r", offset=sgy.data_start, shape=(sgy.ntraces, sgy.trace_bytes))


def write_cdp_sorted(src: str, idx, out: str, head: bytes, chunk: int = 5000) -> None:
    """The traces of `src` in CDP order (by CDP, nearest offset first), bytes unchanged."""
    sgy = segy_io.open_segy(src)
    rec = _records(sgy)
    with open(out, "wb") as f:
        f.write(head)
        for a in range(0, len(idx.order), chunk):
            sel = idx.order[a:a + chunk]
            o = np.argsort(sel)                         # read in file order, write in CDP order
            block = np.empty((len(sel), sgy.trace_bytes), np.uint8)
            block[o] = rec[sel[o]]
            block.tofile(f)
            report(min(len(idx.order), a + chunk), len(idx.order), "Writing the CDP-sorted gathers")
    del rec


def write_nmo_gathers(src: str, idx, vs, mute_pct: float, out: str, head: bytes, fmt: str) -> None:
    """The CDP gathers of `src` (CDP order, nearest offset first), NMO-corrected with each CDP's velocity and the
    stretch mute; headers unchanged."""
    from . import nmo
    from .segy_write import encode_samples
    sgy = segy_io.open_segy(src)
    near = vs.nearest_traces(idx.mid_x, idx.mid_y) if vs.scan is not None else None
    with open(out, "wb") as f:
        f.write(head)
        k = 0
        n_cdp = len(idx.cdps)
        while k < n_cdp:
            k1 = min(n_cdp, k + 500)
            sel = idx.order[idx.first[k]:idx.first[k1 - 1] + idx.fold[k1 - 1]]
            raw, data = sgy.read_traces_at(sel)
            cdp_of = np.repeat(np.arange(k, k1), idx.fold[k:k1])
            if near is not None:
                vel = np.empty((len(sel), sgy.ns))
                for c in range(k, k1):
                    vel[cdp_of == c] = vs.model_trace_velocity(int(near[c]))[0]
            else:
                vel = vs.function
            off = segy_io.header_word(raw, 37, "i4", sgy.order)
            amp, _ = nmo.nmo_many(data.astype(np.float64), nmo.cdp_offsets(off, vs.unit), vs.full_t, vel, mute_pct / 100.0)
            rec = np.empty((len(sel), 240 + 4 * sgy.ns), np.uint8)
            rec[:, :240] = raw
            rec[:, 240:] = np.ascontiguousarray(encode_samples(amp.astype(np.float32), fmt, sgy.order)).view(np.uint8).reshape(len(sel), -1)
            rec.tofile(f)
            k = k1
            report(k, n_cdp, "Writing the NMO-corrected CDP gathers")
