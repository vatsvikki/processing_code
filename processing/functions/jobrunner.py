"""Whole-data runs as background processes (no UI).

A run started here does not live inside the app: it is its own process on the machine (a new session, so closing
the app, the SSH connection or the laptop does not stop it). It works in a folder of its own,
output/jobs/<id>/, where it keeps

    spec.json     what to run (the BatchJob arguments)
    status.json   its progress, rewritten every second (BatchJob.snapshot() + the process id and a heartbeat)
    cancel        created by cancel() - the run stops at the next shot / CDP block
    run.log       anything the process printed (errors)

The app follows a run through a RemoteJob handle - the same methods as a BatchJob (snapshot, cancel, join,
running) - and finds a run still going when it is opened again (find_running).

    python -m functions.jobrunner <job folder>          (what start_job launches)
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from .saving import DEFAULT_DIR

JOBS_DIR = Path(DEFAULT_DIR) / "jobs"
_PKG_ROOT = Path(__file__).resolve().parent.parent          # the folder that holds the functions package


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=str))
    os.replace(tmp, path)                                     # (a reader never sees half a file)


def start_job(spec: dict) -> "RemoteJob":
    """Start a whole-data run in the background. spec: path, steps, out_path, fmt, ffid_from, ffid_to, overwrite,
    title, key."""
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = JOBS_DIR / f"{stamp}-{os.getpid() % 10000:04d}{int(time.time() * 1000) % 1000:03d}"
    folder.mkdir()
    _write_json(folder / "spec.json", spec)
    _write_json(folder / "status.json", {"status": "running", "message": "starting ...", "done": 0, "total": 0,
                                         "elapsed": 0.0, "heartbeat": time.time(), "pid": None})
    with open(folder / "run.log", "ab") as log:
        proc = subprocess.Popen([sys.executable, "-m", "functions.jobrunner", str(folder)], cwd=str(_PKG_ROOT),
                                stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
    status = json.loads((folder / "status.json").read_text())
    status["pid"] = proc.pid
    _write_json(folder / "status.json", status)
    return RemoteJob(folder)


class RemoteJob:
    """The app's handle on a background run: snapshot() / cancel() / join() / running, like a BatchJob."""

    def __init__(self, folder):
        self.folder = Path(folder)
        spec = json.loads((self.folder / "spec.json").read_text())
        self.path, self.steps, self.out_path = spec["path"], spec["steps"], spec.get("out_path")
        self.ffid_from, self.ffid_to = spec.get("ffid_from", 0), spec.get("ffid_to", 0)
        self.title, self.key = spec.get("title", ""), spec.get("key", "")
        self.resumed = ""

    def _status(self) -> dict:
        try:
            st = json.loads((self.folder / "status.json").read_text())
        except (OSError, ValueError):
            return {"status": "running", "message": "starting ...", "done": 0, "total": 0, "elapsed": 0.0}
        if st.get("status") == "running" and not _alive(st.get("pid")) and time.time() - st.get("heartbeat", 0) > 15:
            st["status"] = "error"                            # the process is gone without a last word
            st["message"] = ("the run stopped unexpectedly (process ended - e.g. the machine restarted or the disk "
                             f"is full); see {self.folder / 'run.log'}")
        return st

    def snapshot(self) -> dict:
        st = self._status()
        self.resumed = st.get("resumed", "")
        r = st.get("result")
        return {**{"unit": "shots", "phase": "", "product": "", "cdps": 0, "traces_in": 0, "traces_out": 0,
                   "eta": None, "out_path": self.out_path, "step_text": "", "steps_total": len(self.steps)},
                **st, "result": SimpleNamespace(**r) if isinstance(r, dict) else None}

    @property
    def running(self) -> bool:
        return self._status().get("status") == "running"

    def cancel(self) -> None:
        (self.folder / "cancel").touch()

    def join(self, timeout: float | None = None) -> None:
        t0 = time.time()
        while self.running and (timeout is None or time.time() - t0 < timeout):
            time.sleep(0.5)


def _alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def find_running() -> RemoteJob | None:
    """The newest run that is still going (to show it again when the app is reopened)."""
    if not JOBS_DIR.is_dir():
        return None
    for folder in sorted(JOBS_DIR.iterdir(), reverse=True):
        if (folder / "spec.json").exists():
            job = RemoteJob(folder)
            if job.running:
                return job
    return None


def _main(folder: str) -> None:
    """The background process: run the BatchJob, keep status.json up to date, stop on the cancel file."""
    from .batch import BatchJob
    folder = Path(folder)
    spec = json.loads((folder / "spec.json").read_text())
    job = BatchJob(spec["path"], spec["steps"], spec.get("out_path"), fmt=spec.get("fmt", "ibm"),
                   ffid_from=spec.get("ffid_from", 0), ffid_to=spec.get("ffid_to", 0),
                   overwrite=bool(spec.get("overwrite")))
    job.title = spec.get("title", "")
    stop = threading.Event()

    def report() -> None:
        while True:
            snap = job.snapshot()
            r = snap.get("result")
            snap["result"] = dataclasses.asdict(r) if r is not None else None
            snap.update(pid=os.getpid(), heartbeat=time.time())
            if snap["status"] == "idle":
                snap["status"] = "running"
            try:
                _write_json(folder / "status.json", snap)
            except OSError:
                pass                                          # (e.g. a full disk: try again next second)
            if (folder / "cancel").exists():
                job.cancel()
            if stop.is_set():
                return
            time.sleep(1.0)

    reporter = threading.Thread(target=report, daemon=True)
    job.status = "running"
    job._t0 = time.perf_counter()
    reporter.start()
    job._run()                                                # in this process, to the end
    stop.set()
    reporter.join(5)


if __name__ == "__main__":
    _main(sys.argv[1])
