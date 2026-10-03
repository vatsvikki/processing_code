"""functions -- GUI-independent SEG-Y shot-gather QC functions.

    from functions import get_functions
    for f in get_functions():          # buttons / menu entries
        print(f.label, [p.key for p in f.params])
    outputs = get_functions()[0].run("/path/file.sgy")

Sub-modules: segy_io (reading), detection (dead/bad tests), plotting (figures),
ebcdic (header tables), correction (dead-trace interpolate / remove), batch (whole-file flow, in the background),
context (file facts for GUI widgets), saving (PNG / PDF / SVG / CSV export), registry (plug-in mechanism),
tools (the registered tools).
"""
from . import tools  # noqa: F401  (importing registers the tools)
from . import cdp_recalc  # noqa: F401  (standalone CDP Recalculate tool)
from . import cdp_sort_tool  # noqa: F401  (standalone CDP Sort tool)
from . import notebook_tools  # noqa: F401  (whole notebooks in the Tools menu, e.g. Shot Geometry QC)
from .batch import BatchJob, estimate as estimate_batch, resolve_output
from .context import FileContext, choices_for, default_for, file_context
from .pipeline import get_steps, run_steps
from .registry import Output, Param, QCFunction, get_categories, get_functions, register
from .saving import DEFAULT_DIR, FORMATS, plot_tag, save_data, save_outputs
from .segy_write import FORMATS as SEGY_FORMATS

__all__ = ["BatchJob", "estimate_batch", "resolve_output", "DEFAULT_DIR", "FORMATS", "SEGY_FORMATS", "save_data", "FileContext", "Output", "Param", "QCFunction", "choices_for", "default_for", "file_context",
           "get_categories", "get_functions", "get_steps", "plot_tag", "register", "run_steps", "save_outputs"]
