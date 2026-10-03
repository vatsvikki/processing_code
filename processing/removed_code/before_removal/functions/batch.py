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

from . import segy_io, steps as _steps  # noqa: F401  (steps registers itself)
from .pipeline import _params_for, get_step, run_steps
from .segy_write import FORMATS, build_head, pack_traces

_STATES = ("idle", "running", "done", "cancelled", "error")


@dataclass
class BatchResult:
    out_path: str | None             # None for a dry run
    report_path: str | None
    shots: int
    traces_in: int
    traces_out: int
    bytes_out: int
    seconds: float
    notes: list[str] = field(default_factory=list)
    extra_outputs: list[str] = field(default_factory=list)   # files written by Save Data steps (checkpoints)


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
        """out_path=None is a dry run: everything is computed and counted, nothing is written."""
        if fmt not in FORMATS:
            raise ValueError(f"unknown sample format {fmt!r}; use one of {FORMATS}")
        if not steps:
            raise ValueError("the flow has no steps")
        describe_flow(steps)                                   # unknown step keys fail here, not in the thread
        self.path = os.path.expanduser(str(path).strip())
        if out_path is not None:                               # never let a job write onto the file it reads
            out_path = os.path.abspath(os.path.expanduser(str(out_path)))
            _refuse_source(out_path, self.path)
        self.steps, self.out_path, self.fmt = steps, out_path, fmt
        # Save Data steps: (index of the step, its settings, the file the data of every shot at that point goes to)
        self.checkpoints: list[tuple[int, dict, str]] = []
        for i, s in enumerate(steps):
            if s["step"] == "save_data_step":
                prm = _params_for(get_step("save_data_step"), s.get("params"))
                target = _steps.save_target(self.path, steps[:i], prm, None)
                if out_path is not None:
                    _refuse_source(target, self.path)
                    if os.path.abspath(target) == out_path:
                        raise ValueError(f"the Save Data step {i + 1} would write the same file as the final output ({target})")
                self.checkpoints.append((i, prm, target))
        self.ffid_from, self.ffid_to, self.overwrite = ffid_from, ffid_to, overwrite
        self.workers = int(workers) if workers and workers > 0 else min(8, os.cpu_count() or 1)
        self.status, self.message = "idle", ""
        self.done = self.total = self.traces_in = self.traces_out = 0
        self.result: BatchResult | None = None
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
                "traces_in": self.traces_in, "traces_out": self.traces_out, "elapsed": elapsed, "eta": eta,
                "result": self.result, "out_path": self.out_path}

    # -- the work -----------------------------------------------------------
    def _work(self, sgy: segy_io.SegyFile, idx: segy_io.ShotIndex, k: int):
        i0, n = int(idx.first[k]), int(idx.count[k])
        raw_headers, hdr, data = sgy.read_block(i0, i0 + n)
        gather = segy_io.ShotGather(ffid=int(idx.ffids[k]), i0=i0, data=data, headers=hdr, dt_ms=sgy.dt_ms)
        try:
            stages = run_steps(gather, self.steps, path=self.path, batch=True)
        except Exception as e:
            raise type(e)(f"FFID {gather.ffid}: {e}") from e
        final = stages[-1].state
        packed = None
        if self.out_path is not None and final.gather.ntr:
            packed = pack_traces(raw_headers[final.src], final.gather.data, self.fmt, sgy.order)
        ck = {}                                                # the data at every Save Data step
        if self.out_path is not None:
            for i, prm, _ in self.checkpoints:
                st = stages[i + 1].state
                if st.gather.ntr:
                    ck[i] = pack_traces(raw_headers[st.src], st.gather.data, prm["fmt"], sgy.order)
        return {"ffid": gather.ffid, "in": n, "out": final.gather.ntr, "packed": packed, "ck": ck,
                "notes": " | ".join(f"{s.label}: {s.note}" for s in stages[1:])}

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

    def _run(self) -> None:
        part, fh = None, None
        cps: dict[int, object] = {}                                  # open checkpoint files by step index
        try:
            sgy = segy_io.open_segy(self.path)
            idx, sel = select_shots(self.path, self.ffid_from, self.ffid_to)
            self.total = len(sel)
            rows: list[dict] = []
            if self.out_path is not None:
                out = os.path.expanduser(self.out_path)
                if os.path.exists(out) and not self.overwrite:
                    raise FileExistsError(f"{out} already exists - choose another name or tick 'overwrite'")
                os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
                need = int(sum(idx.count[k] for k in sel)) * sgy.trace_bytes + sgy.data_start
                free = shutil.disk_usage(os.path.dirname(os.path.abspath(out))).free
                if free < need * 1.02 + 50e6:
                    raise OSError(f"not enough disk space for the output: needs about {need / 1e9:.1f} GB, "
                                  f"{free / 1e9:.1f} GB free in {os.path.dirname(os.path.abspath(out))}")
                part = out + ".part"
                fh = open(part, "wb")
                fh.write(build_head(sgy, self._head_lines(sgy, idx, sel), self.fmt))
                for i, prm, target in self.checkpoints:
                    if os.path.exists(target) and not self.overwrite:
                        raise FileExistsError(f"{target} (Save Data step {i + 1}) already exists - choose another name or "
                                              "tick 'overwrite'")
                    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
                    cps[i] = open(target + ".part", "wb")
                    cps[i].write(build_head(sgy, self._head_lines(sgy, idx, sel), prm["fmt"]))
            self.message = f"processing {len(sel)} shots with {self.workers} threads"

            with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="shot") as pool:
                todo = iter(sel)
                pending: deque = deque()

                def submit() -> None:
                    k = next(todo, None)
                    if k is not None:
                        pending.append(pool.submit(self._work, sgy, idx, k))

                for _ in range(2 * self.workers):                  # a bounded window keeps memory flat
                    submit()
                while pending:
                    if self._cancel.is_set():
                        for f in pending:
                            f.cancel()
                        break
                    res = pending.popleft().result()               # raises the failing shot's error
                    if fh is not None and res["packed"] is not None:
                        res["packed"].tofile(fh)
                    for i, blk in res["ck"].items():
                        blk.tofile(cps[i])
                    self.done += 1
                    self.traces_in += res["in"]
                    self.traces_out += res["out"]
                    rows.append({"ffid": res["ffid"], "traces_in": res["in"], "traces_out": res["out"],
                                 "removed": res["in"] - res["out"], "notes": res["notes"]})
                    submit()
            if fh is not None:
                fh.close()
                fh = None
            for f in cps.values():
                f.close()

            self._t1 = time.perf_counter()
            if self._cancel.is_set():
                self.status, self.message = "cancelled", f"cancelled after {self.done} of {self.total} shots"
                return
            report, extra = None, []
            if self.out_path is not None:
                out = os.path.expanduser(self.out_path)
                os.replace(part, out)
                part = None
                for i, prm, target in self.checkpoints:
                    os.replace(target + ".part", target)
                    extra.append(target)
                report = os.path.splitext(out)[0] + "_report.csv"
                with open(report, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["ffid", "traces_in", "traces_out", "removed", "notes"])
                    w.writeheader()
                    w.writerows(rows)
            self.result = BatchResult(
                out_path=os.path.expanduser(self.out_path) if self.out_path else None, report_path=report,
                shots=self.done, traces_in=self.traces_in, traces_out=self.traces_out,
                bytes_out=(os.path.getsize(os.path.expanduser(self.out_path)) if self.out_path else 0),
                seconds=self._t1 - self._t0, notes=describe_flow(self.steps), extra_outputs=extra)
            self.status = "done"
            self.message = "finished" if self.out_path else "finished (dry run, nothing written)"
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
            for i, prm, target in self.checkpoints:                  # a failed / cancelled job leaves no checkpoint files either
                try:
                    if i in cps:
                        cps[i].close()
                except Exception:
                    pass
                try:
                    os.remove(target + ".part")
                except OSError:
                    pass
