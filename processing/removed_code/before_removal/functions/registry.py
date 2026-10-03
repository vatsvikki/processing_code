"""Plug-in registry: how a function describes itself to ANY GUI.

A tool function is an ordinary Python function

    def my_tool(path: str, **params) -> list[Output]

decorated with @register(...).  The decorator records its label, description
and a list of `Param`s.  A GUI never has to know what the function does: it
lists `get_functions()` as buttons, builds a parameter form from `.params`,
calls `.run(path, **values)` and renders the returned `Output`s (markdown /
table / image).  To add a new function you only write it and decorate it --
no UI code changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Param:
    key: str                         # keyword argument name of the function
    label: str                       # text shown next to the input
    kind: str = "float"              # "int" | "float" | "bool" | "choice" | "text" | "textarea" (several lines) |
                                     # "table" (default = list of row dicts, edited as a table) | "ffid"
                                     # "ffid" = pick one of the FFIDs that really exist in the file (slider)
    default: Any = 0
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] | None = None
    help: str = ""
    group: str = ""                  # widgets with the same group are shown under one heading
    live: bool = False               # True: shown outside the Run form, re-runs the tool as soon as it changes
    auto: str = ""                   # default read from the file header instead of `default`, see context.default_for
    example: str = ""                # "textarea" only: sample input shown greyed-out while the box is empty
    info: str = ""                   # full explanation for the (i) hover; empty = param_help.PARAM_INFO[key] (else `help`)


@dataclass
class Output:
    """One result item.  kind: "markdown" | "table" | "image" | "data" (not drawn, only saved)."""
    kind: str
    title: str = ""
    content: Any = None              # markdown: str | image: PNG bytes | table: rows (list of dicts)
    columns: list[str] | None = None # table only
    mono: tuple[str, ...] = ()       # table only: columns to show in a monospace font
    paginate: bool = False           # table only: long, sortable table (else a plain static table)
    figure: Any = None               # image only: the matplotlib Figure, so it can be saved as PDF / SVG / any dpi
    zoom: bool = False               # image only: a gather (x = trace position, y = time ms) a GUI may let the user
                                     # box-select to set trace_min / trace_max / t_min_ms / t_max_ms


def markdown(title: str, text: str) -> Output:
    return Output("markdown", title, text)


def table(title: str, columns: list[str], rows: list[dict], *, mono=(), paginate=False) -> Output:
    return Output("table", title, rows, columns=columns, mono=tuple(mono), paginate=paginate)


def data(title: str, obj: Any) -> Output:
    """Processed data a GUI does not draw but can offer to save (obj: segy_write.ProcessedShot)."""
    return Output("data", title, obj)


def image(title: str, png: bytes, figure: Any = None, zoom: bool = False) -> Output:
    return Output("image", title, png, figure=figure, zoom=zoom)


@dataclass
class QCFunction:
    key: str
    label: str
    description: str
    params: list[Param]
    run: Callable[..., list[Output]]
    order: int = 100
    autorun: bool = True             # False: heavy tool, the GUI waits for an explicit Run press
    kind: str = "tool"               # "pipeline": the GUI offers a flow builder (steps, order, per-step parameters)
                                     # "flowstep": a processing function - the GUI adds it to the flow when it is clicked
    step_key: str = ""               # kind "flowstep": the pipeline step (pipeline.get_step) this function is
    category: str = "General"        # the toolbar groups functions by this, see CATEGORY_ORDER

    def defaults(self) -> dict:
        return {p.key: p.default for p in self.params}


_REGISTRY: dict[str, QCFunction] = {}


def register(label: str, description: str = "", params: list[Param] | tuple = (), order: int = 100,
             autorun: bool = True, kind: str = "tool", category: str = "General", step_key: str = ""):
    def deco(fn):
        _REGISTRY[fn.__name__] = QCFunction(fn.__name__, label, description, list(params), fn, order, autorun, kind,
                                        category=category, step_key=step_key)
        return fn
    return deco


CATEGORY_ORDER = ("Data", "Display", "QC", "Processing")      # (the flow builder, category "Flow", is not on the top bar)


def get_categories() -> list[tuple[str, list[QCFunction]]]:
    """[(category, [functions])] in toolbar order; categories not in CATEGORY_ORDER come last."""
    fs = get_functions()
    names = [c for c in CATEGORY_ORDER if any(f.category == c for f in fs)]
    names += sorted({f.category for f in fs} - set(names))
    return [(c, [f for f in fs if f.category == c]) for c in names]


def get_functions() -> list[QCFunction]:
    """All registered functions in toolbar order."""
    return sorted(_REGISTRY.values(), key=lambda f: (f.order, f.label))
