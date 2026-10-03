"""Progress of long functions (the whole-survey header scan, the full stack), for a GUI to show - no UI here.

A GUI (or a background job) wraps a run in `reporting(fn)`; the functions call `report(done, total, title)`;
fn(done, total, title) draws it. The reporter belongs to the thread that set it, so a whole-data job running in the
background and the preview on screen each see only their own progress. A reporter may raise (e.g. to cancel a job):
that stops the work.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading

_local = threading.local()


@contextmanager
def reporting(fn):
    old = getattr(_local, "reporter", None)
    _local.reporter = fn
    try:
        yield
    finally:
        _local.reporter = old


class Cancelled(Exception):
    """Raised by a reporter to stop the work (a cancelled job)."""


def report(done: int, total: int, title: str = "") -> None:
    fn = getattr(_local, "reporter", None)
    if fn is None:
        return
    try:
        fn(int(done), int(total), title)
    except Cancelled:
        raise
    except Exception:
        pass                                          # a drawing problem never stops the work
