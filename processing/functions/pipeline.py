"""Processing pipeline: an ordered list of steps run on one shot gather (no UI).

A *step* is `fn(state, **params) -> (new_state, note)` registered with @step.  The state
carries the gather plus the dead / bad flags found so far, so later steps can use what an
earlier step found (e.g. "Correct Dead Traces" acts on the traces "Detect Dead Traces" flagged).

A flow is plain data - a list of {"step": key, "params": {...}} - so a GUI, a script or a
JSON file can define it:

    stages = run_steps(gather, [{"step": "detect_dead"},
                                {"step": "correct_dead", "params": {"mode": "interpolate"}},
                                {"step": "spiking_decon", "params": {"operator_ms": 200}}])
    stages[-1].state.gather        # final gather; stages[0] is the untouched input
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from .detection import QCResult
from .registry import Param
from .segy_io import ShotGather, trace_order


@dataclass
class PipeState:
    gather: ShotGather
    dead: np.ndarray                         # bool [ntr] flagged dead so far
    bad: np.ndarray                          # bool [ntr] flagged bad so far
    dead_checked: bool = False               # has a detect step looked for dead traces yet?
    marks: np.ndarray | None = None          # traces to highlight on plots (interpolated / removal points)
    marks_label: str = ""
    src: np.ndarray | None = None            # index in the source shot of every trace (traces get removed)
    path: str = ""                           # the source SEG-Y file (a step that writes a file needs its headers)
    batch: bool = False                      # True inside a whole-data run (BatchJob)
    flow: list = field(default_factory=list) # the steps applied so far: [{"step": key, "params": {...}}]
    log: list = field(default_factory=list)  # "label: note" of each step so far
    view_extra: list | None = None           # a "display" step's own Outputs (a map, not a gather) to show for this
                                             # stage instead of the usual gather + spectrum figures; reset to None
                                             # before every step runs, so it only applies to the step that set it
    figs: list | None = None                 # extra figures of a processing step (F-K spectrum, Radon panel) shown below
                                             # the gather for this stage; reset before every step like view_extra

    @classmethod
    def initial(cls, gather: ShotGather, path: str = "", batch: bool = False) -> "PipeState":
        return cls(gather, np.zeros(gather.ntr, bool), np.zeros(gather.ntr, bool), src=np.arange(gather.ntr),
                   path=path, batch=batch)


@dataclass
class Step:
    key: str
    label: str
    description: str
    params: list[Param]
    run: Callable[..., tuple[PipeState, str]]
    order: int = 100
    category: str = "Processing"             # "QC" steps only flag traces; the Flow tool offers the Processing ones


@dataclass
class Stage:
    label: str                               # "2. Spiking Decon"
    state: PipeState
    note: str = ""
    seconds: float = 0.0


_STEPS: dict[str, Step] = {}


def step(label: str, description: str = "", params: list[Param] | tuple = (), order: int = 100, category: str = "Processing"):
    """Register `fn(state, **params) -> (new_state, note)` as a pipeline step."""
    def deco(fn):
        _STEPS[fn.__name__] = Step(fn.__name__, label, description, list(params), fn, order, category)
        return fn
    return deco


def get_steps(category: str | None = None) -> list[Step]:
    """The registered steps in order; category="Processing" leaves out the QC steps that only flag traces."""
    return sorted((s for s in _STEPS.values() if category is None or s.category == category), key=lambda s: (s.order, s.label))


def get_step(key: str) -> Step:
    try:
        return _STEPS[key]
    except KeyError:
        raise KeyError(f"unknown pipeline step {key!r}; available: {', '.join(_STEPS)}") from None


def _params_for(st: Step, given: dict | None) -> dict:
    """Defaults for every parameter, overridden by `given`, cast to the parameter's type."""
    cast = {"int": int, "float": float, "bool": bool}
    out = {}
    for p in st.params:
        v = (given or {}).get(p.key)
        v = p.default if v is None else v
        out[p.key] = cast.get(p.kind, lambda x: x)(v)
    return out


def run_steps(gather: ShotGather, specs: list[dict], path: str = "", batch: bool = False,
              start: list[Stage] | None = None) -> list[Stage]:
    """Run the flow; returns one Stage per step, preceded by stage 0 = the untouched input.
    `path` is the source SEG-Y file (needed by steps that write files); batch=True inside a whole-data run.
    start = stages already computed for the first steps of `specs` (stage 0 first): the flow carries on from the last
    of them instead of starting again from the input."""
    stages = list(start) if start else [Stage("0. Input (raw)", PipeState.initial(gather, path, batch),
                                              f"{gather.ntr} traces")]
    for i, spec in enumerate(specs, 1):
        if i < len(stages):                     # already computed (start)
            continue
        st = get_step(spec["step"])
        t0 = time.perf_counter()
        try:
            new_state, note = st.run(replace(stages[-1].state, view_extra=None, figs=None), **_params_for(st, spec.get("params")))
        except Exception as e:
            raise type(e)(f"step {i} ({st.label}): {e}") from e
        prev = stages[-1].state
        new_state = replace(new_state, flow=prev.flow + [{"step": st.key, "params": dict(spec.get("params") or {})}],
                            log=prev.log + [f"{st.label}: {note}"])
        stages.append(Stage(f"{i}. {st.label}", new_state, note, time.perf_counter() - t0))
    return stages


def flag_result(state: PipeState, sort_by: str = "file") -> QCResult | None:
    """The state's dead / bad flags as a QCResult so `plotting.plot_gather` can mark them (None if none)."""
    if not (state.dead.any() or state.bad.any()):
        return None
    g = state.gather
    return QCResult(
        ffid=g.ffid, order=trace_order(g, sort_by), dead=state.dead, bad=state.bad,
        reasons=[[] for _ in range(g.ntr)], metrics={}, ref_rms=np.zeros(g.ntr),
    )
