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

from . import plotting


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
    choices_auto: str = ""           # "choice" only: the options come from the loaded file, see context.choices_for
    show_if: dict | None = None      # shown only while other parameters of the same function have these values:
                                     # {"vel_source": "Velocity model SEG-Y"} (a value or a list of values per key)

    def shown(self, values: dict) -> bool:
        """Whether this parameter is shown, given the current values of its function's parameters."""
        for key, want in (self.show_if or {}).items():
            allowed = want if isinstance(want, (list, tuple, set)) else [want]
            if values.get(key) not in allowed:
                return False
        return True
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


def figure(title: str, fig, zoom: bool = False) -> Output:
    """Image output that keeps its Figure (so Save can write PDF / SVG / high-dpi PNG).

    zoom=True marks a shot gather that a GUI may let the user zoom by dragging a box on it."""
    return image(title, plotting.figure_to_png(fig), figure=fig, zoom=zoom)


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
    category: str = "General"        # the toolbar groups functions by this, see CATEGORY_ORDER

    def defaults(self) -> dict:
        return {p.key: p.default for p in self.params}


_REGISTRY: dict[str, QCFunction] = {}


def register(label: str, description: str = "", params: list[Param] | tuple = (), order: int = 100,
             autorun: bool = True, kind: str = "tool", category: str = "General"):
    def deco(fn):
        _REGISTRY[fn.__name__] = QCFunction(fn.__name__, label, description, list(params), fn, order, autorun, kind, category)
        return fn
    return deco


CATEGORY_ORDER = ("Data", "Display", "QC", "Processing")


def get_categories() -> list[tuple[str, list[QCFunction]]]:
    """[(category, [functions])] in toolbar order; categories not in CATEGORY_ORDER come last."""
    fs = get_functions()
    names = [c for c in CATEGORY_ORDER if any(f.category == c for f in fs)]
    names += sorted({f.category for f in fs} - set(names))
    return [(c, [f for f in fs if f.category == c]) for c in names]


def get_functions() -> list[QCFunction]:
    """All registered functions in toolbar order."""
    return sorted(_REGISTRY.values(), key=lambda f: (f.order, f.label))
