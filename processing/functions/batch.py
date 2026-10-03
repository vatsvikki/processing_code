"""Apply a flow to every shot of a SEG-Y file and write the result as ONE SEG-Y file (no UI).

    job = BatchJob(path, steps, out_path, fmt="ibm", ffid_from=398, ffid_to=1982)
    job.start()                      # runs in a background thread
    job.snapshot()                   # {"status": "running", "done": 312, "total": 1291, ...} - poll for progress
    job.cancel()

The file is too big to hold in memory (15 GB), so shots are processed in parallel but written one by one,
in FFID order, as they finish.  The output goes to `<out>.part` and is renamed only when every shot is done,
so a cancelled / failed job never leaves something that looks like a finished file.

What is written: the source's EBCDIC / binary headers (a processing note in the blank text cards, the sample
format code updated), and for every trace that survives the flow its original 240-byte trace header + the
processed samples.  A per-shot CSV report (traces in / out and each step's note) is written next to it.
"""
from __future__ import annotations

import csv
import os
import shutil
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import nmo as _nmo

from . import cdp_flow, cdp_sort, segy_io, stack as stack_mod, steps as _steps, velocity_setup  # noqa: F401
from .pipeline import _params_for, get_step, run_steps
from .progress import Cancelled, reporting
from .segy_write import FORMATS, build_head, pack_traces

_STATES = ("idle", "running", "done", "cancelled", "error")


@dataclass
class BatchResult:
    out_path: str | None             # None: only the shot steps' result, kept in output/flows
    report_path: str | None
    shots: int
    traces_in: int
    traces_out: int
    bytes_out: int
    seconds: float
    notes: list[str] = field(default_factory=list)


def describe_flow(steps: list[dict]) -> list[str]:
    """['1. Detect Dead Traces', ...] for messages and the text header."""
    return [f"{i}. {get_step(s['step']).label}" for i, s in enumerate(steps, 1)]


def select_shots(path: str, ffid_from: int = 0, ffid_to: int = 0) -> tuple[segy_io.ShotIndex, list[int]]:
    """Shot index and the positions in it of the shots with ffid_from <= FFID <= ffid_to (0 = first / last)."""
    idx = segy_io.shot_index(path)
    lo = int(ffid_from) if ffid_from and ffid_from > 0 else int(idx.ffids[0])
    hi = int(ffid_to) if ffid_to and ffid_to > 0 else int(idx.ffids[-1])
    sel = [k for k in range(len(idx)) if lo <= idx.ffids[k] <= hi]
    if not sel:
        raise LookupError(f"no shots with FFID between {lo} and {hi} (file has {int(idx.ffids[0])} to {int(idx.ffids[-1])})")
    return idx, sel


def _refuse_source(out: str, source: str) -> None:
    """A job reads the source while it writes: the output must never be the same file."""
    src = os.path.abspath(os.path.expanduser(str(source).strip()))
    if out == src or (os.path.exists(out) and os.path.exists(src) and os.path.samefile(out, src)):
        raise ValueError("the output file must differ from the input file - choose another name")


def resolve_output(out: str, source: str, default_name: str) -> str:
    """The output file path from what the user typed.

    Same handling as the input path box: `~` is expanded; a folder (existing, or ending in "/") gets
    `default_name` inside it; a name without an extension gets ".sgy"; empty = next to the source file.
    Raises if it would be the source file itself.
    """
    src = os.path.abspath(os.path.expanduser(str(source).strip()))
    text = os.path.expanduser(str(out or "").strip())
    if not text:
        text = os.path.join(os.path.dirname(src), default_name)
    elif text.endswith(("/", os.sep)) or os.path.isdir(text):
        text = os.path.join(text, default_name)
    elif not os.path.splitext(text)[1]:
        text += ".sgy"
    text = os.path.abspath(text)
    _refuse_source(text, src)
    return text


def estimate(path: str, ffid_from: int = 0, ffid_to: int = 0) -> dict:
    """{'shots', 'traces', 'gigabytes'} that a job over this FFID range would read and write."""
    sgy = segy_io.open_segy(path)
    idx, sel = select_shots(path, ffid_from, ffid_to)
    traces = int(sum(idx.count[k] for k in sel))
    return {"shots": len(sel), "traces": traces, "gigabytes": (traces * sgy.trace_bytes + sgy.data_start) / 1e9}


class BatchJob:
    def __init__(self, path: str, steps: list[dict], out_path: str | None, *, fmt: str = "ibm",
                 ffid_from: int = 0, ffid_to: int = 0, workers: int = 0, overwrite: bool = False):
        """out_path = the flow's product (the processed shots, or the last CDP step's NMO gathers / stack).
        out_path=None: only the shot steps are run, and their result is kept in the output folder (output/flows) for
        the next steps - a run starts from the furthest such result already there, so a step done once on the whole
        data is never done again."""
        if fmt not in FORMATS:
            raise ValueError(f"unknown sample format {fmt!r}; use one of {FORMATS}")
        if not steps:
            raise ValueError("the flow has no steps")
        describe_flow(steps)                                   # unknown step keys fail here, not in the thread
        # a flow with CDP steps: the shot steps go to an intermediate file, then the LAST CDP step makes the product
        self.shot_steps, self.cdp_steps, problems = cdp_flow.split_flow(steps)
        if problems:
            raise ValueError("; ".join(problems))
        self.product = self.cdp_steps[-1]["step"] if self.cdp_steps else ""
        self.path = os.path.expanduser(str(path).strip())
        if out_path is not None:                               # never let a job write onto the file it reads
            out_path = os.path.abspath(os.path.expanduser(str(out_path)))
            _refuse_source(out_path, self.path)
        self.steps, self.out_path, self.fmt = steps, out_path, fmt
        self.ffid_from, self.ffid_to, self.overwrite = ffid_from, ffid_to, overwrite
        self.workers = int(workers) if workers and workers > 0 else min(8, os.cpu_count() or 1)
        self.status, self.message = "idle", ""
        self.unit, self.phase, self.cdps = "shots", "", 0
        self.done = self.total = self.traces_in = self.traces_out = 0
        self.result: BatchResult | None = None
        self.title = ""                                        # what the GUI calls this run ("up to 3. Bandpass ...")
        self.resumed = ""                                      # the saved result it started from, if any
        self._t0 = self._t1 = 0.0
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    # -- control ------------------------------------------------------------
    def start(self) -> "BatchJob":
        self.status, self.message = "running", "starting ..."
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="batch-job", daemon=True)
        self._thread.start()
        return self

    def cancel(self) -> None:
        self._cancel.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return self.status == "running"

    def snapshot(self) -> dict:
        end = self._t1 if self._t1 else time.perf_counter()
        elapsed = end - self._t0 if self._t0 else 0.0
        eta = elapsed * (self.total - self.done) / self.done if self.done and self.running else None
        return {"status": self.status, "message": self.message, "done": self.done, "total": self.total,
                "unit": self.unit, "phase": self.phase, "product": self.product, "cdps": self.cdps,
                "traces_in": self.traces_in, "traces_out": self.traces_out, "elapsed": elapsed, "eta": eta,
                "result": self.result, "out_path": self.out_path}

    # -- the work -----------------------------------------------------------
    def _work(self, sgy: segy_io.SegyFile, idx: segy_io.ShotIndex, k: int):
        i0, n = int(idx.first[k]), int(idx.count[k])
        raw_headers, hdr, data = sgy.read_block(i0, i0 + n)
        gather = segy_io.ShotGather(ffid=int(idx.ffids[k]), i0=i0, data=data, headers=hdr, dt_ms=sgy.dt_ms)
        try:
            stages = run_steps(gather, self._remaining, path=self.path, batch=True)
        except Exception as e:
            raise type(e)(f"FFID {gather.ffid}: {e}") from e
        final = stages[-1].state
        packed = None
        if self._writing and final.gather.ntr:
            packed = pack_traces(raw_headers[final.src], final.gather.headers, final.gather.data, self._pack_fmt,
                                 sgy.order)
        return {"ffid": gather.ffid, "in": n, "out": final.gather.ntr, "packed": packed,
                "notes": " | ".join(f"{s.label}: {s.note}" for s in stages[1:]) or "(saved result copied)"}

    def _head_lines(self, sgy: segy_io.SegyFile, idx: segy_io.ShotIndex, sel: list[int]) -> list[str]:
        """Text-header cards (max 4): what was done, on which shots, and the deconvolution settings."""
        lo, hi = int(idx.ffids[sel[0]]), int(idx.ffids[sel[-1]])
        labels = []
        for s in self.steps:
            mode = (s.get("params") or {}).get("mode")
            labels.append(get_step(s["step"]).label.upper().replace(" TRACES", "") + (f" ({mode.upper()})" if mode else ""))
        lines = [f"PROCESSED FFID {lo}-{hi} ({len(sel)} SHOTS)", "FLOW: " + " > ".join(labels)]
        for s in self.steps:
            if s["step"] == "spiking_decon":
                pr = {p.key: (s.get("params") or {}).get(p.key, p.default) for p in get_step("spiking_decon").params}
                end = "END" if pr["design_end_ms"] <= 0 else f"{pr['design_end_ms']:g}"
                lines.append(f"SPIKING DECON: OPERATOR {pr['operator_ms']:g} MS, PREWHITE {pr['prewhite_pct']:g} %, "
                             f"DESIGN {pr['design_start_ms']:g}-{end} MS" + ("" if pr["balance"] else ", NO RMS BALANCE"))
        return lines

    # -- the CDP part of a flow (phase 2) ----------------------------------------------
    def _progress(self, done, total, title):
        if self._cancel.is_set():
            raise Cancelled()
        cdps = any(w in title for w in ("Finishing", "full stack to SEG-Y", "NMO-corrected"))
        self.phase, self.message, self.unit = title, title, "CDPs" if cdps else "traces"
        self.done, self.total = done, total

    def _velocity(self, src, spec, chained_nmo=None):
        """The velocity setup and mute of a CDP step (its own settings, or those of an NMO step above it)."""
        st = get_step(spec["step"])
        p = _params_for(st, spec.get("params"))
        if chained_nmo is not None:
            pn = _params_for(get_step("nmo_correction_step"), chained_nmo.get("params"))
            p = {**p, **{k: pn[k] for k in _steps._VEL_KEYS}, "vel_source": pn["vel_source"],
                 "stretch_mute_pct": pn["stretch_mute_pct"]}
        s = segy_io.open_segy(src)
        full_t = np.arange(s.ns) * (s.dt_ms / 1000.0)
        vs = velocity_setup.build(self.path, full_t, source=p["vel_source"], **{k: p[k] for k in _steps._VEL_KEYS})
        if vs.error:
            who = st.label if chained_nmo is None else f"NMO Correction (its velocity is used by {st.label})"
            raise ValueError(f"{who}: " + vs.error.replace("**", ""))
        return vs, float(p["stretch_mute_pct"]), p

    def _cdp_phase(self, src, part):
        """Phase 2: the product of the last CDP step, from `src` (the shot steps' result, or the raw file)."""
        with reporting(lambda d, t, title: self._progress(d, t, title)):
            idx = cdp_sort.cdp_index(src)
            s = segy_io.open_segy(src)
            lines = [f"CDP PRODUCT: {get_step(self.product).label.upper()} ({len(idx.cdps)} CDPS)",
                     "FLOW: " + " > ".join(get_step(x["step"]).label.upper() for x in self.steps)]
            if self.product == "nmo_correction_step":
                vs, mute, _ = self._velocity(src, self.cdp_steps[-1])
                cdp_flow.write_nmo_gathers(src, idx, vs, mute, part, bytes(build_head(s, lines, self.fmt)), self.fmt)
            else:                                                # the stack of what the steps above produced
                chained = next((x for x in reversed(self.cdp_steps[:-1]) if x["step"] == "nmo_correction_step"), None)
                if chained is not None:                          # NMO-corrected gathers: NMO once, as that step says
                    vs, mute, _ = self._velocity(src, chained)
                else:                                            # CDP gathers: their plain mean
                    vs = velocity_setup.build(self.path, np.arange(s.ns) * (s.dt_ms / 1000.0),
                                              source=velocity_setup.NONE)
                    mute = 0.0
                fs = stack_mod.full_stack(src, idx, vs, mute, None, {"raw": False, "post_nmo": False, "stack": False})
                il, xl = (stack_mod.cdp_ilxl(idx, vs.data_transform) if vs.data_transform is not None
                          else (np.zeros(len(idx.cdps), np.int64), np.zeros(len(idx.cdps), np.int64)))
                stack_mod.export_segy(fs, il, xl, part, s.order)
            # what was written: CDP-ordered traces, or one stacked trace per CDP
            self.traces_out = len(idx.cdps) if self.product == "cdp_stack_step" else int(len(idx.order))
            self.cdps = len(idx.cdps)

    def _run(self) -> None:
        part, fh = None, None
        try:
            final_out = os.path.expanduser(self.out_path) if self.out_path is not None else None
            # the shot steps: carry on from the furthest whole-data result of the first of them already saved
            proc = cdp_flow.processing_steps(self.shot_steps)
            cache = cdp_flow.processed_path(self.path, self.shot_steps, self.ffid_from, self.ffid_to) if proc else None
            start, j = cdp_flow.best_start(self.path, proc, self.ffid_from, self.ffid_to) if proc else (None, 0)
            self._remaining = proc[j:]
            if j:
                self.resumed = f"the saved whole-data result of the first {j} step(s)"
            src = start or self.path                              # the file the shots are read from
            same = (not self.cdp_steps and final_out is not None and start is not None
                    and os.path.abspath(start) == final_out)
            if not proc or same or (not self._remaining and (self.cdp_steps or final_out is None)):
                phase1_out = None                                 # nothing (new) to do per shot
                if final_out is None:
                    self.message = ("already run on the whole data - its result is kept" if proc
                                    else "no processing step - nothing to run on the shots")
            elif self.cdp_steps or final_out is None:
                phase1_out = cache                                # kept for the next steps (output/flows)
            else:
                phase1_out = final_out                            # the flow's product: the processed shots
            cdp_src = cache if (proc and self.cdp_steps) else self.path
            if (self.cdp_steps and final_out is not None and os.path.exists(final_out) and not self.overwrite):
                raise FileExistsError(f"{final_out} already exists - choose another name or tick 'overwrite'")
            if same:
                self.message = "already run on the whole data - the output file is this flow's result"
            sgy = segy_io.open_segy(src)
            idx, sel = select_shots(src, self.ffid_from, self.ffid_to)
            self.total = len(sel)
            rows: list[dict] = []
            run_phase1 = phase1_out is not None
            self._writing = run_phase1
            if phase1_out is not None:
                out = os.path.expanduser(phase1_out)
                if os.path.exists(out) and not self.overwrite and out != cache:
                    raise FileExistsError(f"{out} already exists - choose another name or tick 'overwrite'")
                os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
                if out == cache:                                  # this flow's old results nobody can use any more
                    cdp_flow.drop_stale(self.path, proc)
                need = int(sum(idx.count[k] for k in sel)) * sgy.trace_bytes + sgy.data_start
                free = shutil.disk_usage(os.path.dirname(os.path.abspath(out))).free
                if free < need * 1.02 + 50e6:
                    raise OSError(f"not enough disk space for the output: needs about {need / 1e9:.1f} GB, "
                                  f"{free / 1e9:.1f} GB free in {os.path.dirname(os.path.abspath(out))}")
                part = out + ".part"
                fh = open(part, "wb")
                fh.write(build_head(segy_io.open_segy(self.path), self._head_lines(sgy, idx, sel),
                                    self.fmt if out == final_out else "ieee"))
            self._pack_fmt = self.fmt if phase1_out is not None and phase1_out == final_out else "ieee"
            if run_phase1:
                self.message = f"processing {len(sel)} shots with {self.workers} threads"
            self.phase = "shot steps"

            with (ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="shot") if run_phase1
                  else _nothing()) as pool:
                todo = iter(sel)
                pending: deque = deque()

                def submit() -> None:
                    k = next(todo, None)
                    if k is not None:
                        pending.append(pool.submit(self._work, sgy, idx, k))

                for _ in range(2 * self.workers if run_phase1 else 0):  # a bounded window keeps memory flat
                    submit()
                while pending:
                    if self._cancel.is_set():
                        for f in pending:
                            f.cancel()
                        break
                    res = pending.popleft().result()               # raises the failing shot's error
                    if fh is not None and res["packed"] is not None:
                        res["packed"].tofile(fh)
                    self.done += 1
                    self.traces_in += res["in"]
                    self.traces_out += res["out"]
                    rows.append({"ffid": res["ffid"], "traces_in": res["in"], "traces_out": res["out"],
                                 "removed": res["in"] - res["out"], "notes": res["notes"]})
                    submit()
            if fh is not None:
                fh.close()
                fh = None

            if self._cancel.is_set():
                self._t1 = time.perf_counter()
                self.status, self.message = "cancelled", f"cancelled after {self.done} of {self.total} shots"
                return
            report = None
            if phase1_out is not None:
                out = os.path.expanduser(phase1_out)
                os.replace(part, out)
                part = None
                info = {"source": segy_io.open_segy(self.path).path, "steps": proc, "ffid_from": self.ffid_from,
                        "ffid_to": self.ffid_to, "shots": self.done, "traces": self.traces_out}
                # remembered for the next steps: the kept file, or the user's output when that is this result
                if out == cache or not cdp_flow.processed_ready(cache):    # (a kept file stays the record)
                    cdp_flow.mark_processed(cache, info, file=None if out == cache else out)
                report = os.path.splitext(out)[0] + "_report.csv"
                with open(report, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["ffid", "traces_in", "traces_out", "removed", "notes"])
                    w.writeheader()
                    w.writerows(rows)
            if self.cdp_steps and final_out is not None:
                part = final_out + ".part"
                self._cdp_phase(cdp_flow.processed_file(cdp_src) if cdp_src != self.path else self.path, part)
                os.replace(part, final_out)
                part = None
            self._t1 = time.perf_counter()
            self.result = BatchResult(
                out_path=os.path.expanduser(self.out_path) if self.out_path else None, report_path=report,
                shots=self.done, traces_in=self.traces_in, traces_out=self.traces_out,
                bytes_out=(os.path.getsize(os.path.expanduser(self.out_path)) if self.out_path
                           else os.path.getsize(cache) if cache and os.path.exists(cache) else 0),
                seconds=self._t1 - self._t0, notes=describe_flow(self.steps))
            if not self.message.startswith(("already", "no processing")):
                self.message = "finished"
            self.status = "done"
        except Cancelled:
            self._t1 = time.perf_counter()
            self.status, self.message = "cancelled", f"cancelled during: {self.phase}"
        except Exception as e:                                       # shown by the GUI, never a stack trace
            self._t1 = time.perf_counter()
            self.status, self.message = "error", f"{type(e).__name__}: {e}"
        finally:
            if part is not None:                                     # cancelled / failed: no half-written file left behind
                try:
                    if fh is not None:
                        fh.close()
                except Exception:
                    pass
                try:
                    os.remove(part)
                except OSError:
                    pass


class _nothing:
    """Stand-in for the thread pool when the shot steps need not run (their result is already there)."""
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
