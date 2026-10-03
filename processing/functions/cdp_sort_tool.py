"""CDP Sort - a tool of the 🛠 Tools menu (opens in its own tab): the brute-stack notebook's whole-survey CDP sort of the
SEG-Y file. Every trace sorted by CDP number, then by |offset| within each CDP; the gather of one CDP; the list of all
CDPs with fold, offsets, midpoint and IL / XL; optionally a CDP-sorted copy of the file. Independent of the Flow.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import cdp_flow, cdp_sort, geometry, plotting, segy_io
from . import grid as grid_mod
from .registry import Param, figure, markdown, register, table
from .segy_write import build_head


@register("CDP Sort", "Whole-survey CDP sort of the file (as in the brute-stack notebook): every trace by CDP, then by "
          "offset; the gather of any CDP, the list of all CDPs with fold, offsets and IL / XL, and optionally a "
          "CDP-sorted copy of the file. Independent of the Flow.",
          params=[
              Param("cdp_number", "CDP to show  (0 = the highest fold)", "int", 0, min=0, step=1, group="CDP",
                    help="A CDP number from the table of all CDPs; a number not in the file shows the nearest CDP"),
              Param("sort_fig_height", "Figure height, inches", "float", 8.0, min=3, max=20, step=0.5, group="CDP",
                    help="Height of the CDP gather plot; shorter = it fits a small screen (the width follows the page)"),
              Param("write_sorted", "Write a CDP-sorted copy of the file", "bool", False, group="Output file",
                    help="Every trace of the file in CDP order (by CDP, nearest offset first), headers and samples "
                         "unchanged - the input of a CDP-domain processing package. Reads the whole file"),
              Param("sorted_out", "Output SEG-Y file  (empty = <input name>_cdp_sorted.sgy next to the input)", "text", "",
                    group="Output file", help="Full path of the CDP-sorted copy (on the machine the app runs on); a name "
                                              "without an extension gets .sgy. Never the input file itself"),
          ],
          order=11, autorun=False, category="Tools")
def cdp_sort_tool(path: str, cdp_number: int = 0, sort_fig_height: float = 8.0, write_sorted: bool = False,
                  sorted_out: str = ""):
    sgy = segy_io.open_segy(path)
    idx = cdp_sort.cdp_index(sgy.path)
    k = idx.position(int(cdp_number))
    cdp, fold = int(idx.cdps[k]), int(idx.fold[k])
    cg = cdp_sort.read_cdp_gather(sgy.path, idx, k)

    # IL / XL of every CDP (from its mean midpoint) when a grid is known: the corner table saved for this file or the
    # header IL / XL - the same grid the Acquisition Geometry / Fold maps of the Flow use
    il = xl = None
    try:
        from .steps import _resolve_grid
        grid, _ = _resolve_grid(sgy.path, geometry.read_geometry(sgy.path, 60), grid_mod.corner_table(sgy.path), False, 60)
        if grid is not None:
            il, xl = (np.rint(v).astype(np.int64) for v in grid.ilxl(idx.mid_x, idx.mid_y))
    except Exception:
        il = xl = None

    asked = int(cdp_number)
    which = (" (the CDP with the highest fold)" if not asked else
             "" if asked == cdp else f" (the nearest CDP to {asked:,} - that number is not in the file)")
    md = (f"**{idx.traces:,} traces** sorted into **{len(idx.cdps):,} CDPs** "
          f"({int(idx.cdps[0]):,} – {int(idx.cdps[-1]):,}), by CDP and then by offset within each CDP · fold "
          f"{int(idx.fold.min())} – {int(idx.fold.max())} (median {int(np.median(idx.fold))})\n\n"
          f"Showing **CDP {cdp:,}**{which}: **{fold} traces**, offset {int(idx.off_min[k]):,} – {int(idx.off_max[k]):,}"
          + (f", IL {int(il[k])} / XL {int(xl[k])}" if il is not None else "") + ".\n\n"
          "This is the raw file. The Flow's NMO Correction and CDP Stack make their own CDP gathers - of the data the "
          "steps above them produced - so they need no sort step.")
    out = [markdown("CDP sort", md)]

    if write_sorted:
        p = Path(sgy.path)
        dest = os.path.abspath(os.path.expanduser(sorted_out.strip() or str(p.with_name(f"{p.stem}_cdp_sorted.sgy"))))
        if not os.path.splitext(dest)[1]:
            dest += ".sgy"
        if os.path.exists(dest) and os.path.samefile(dest, sgy.path):
            raise ValueError("the output file is the input file - choose another name")
        fmt = {1: "ibm", 5: "ieee"}.get(sgy.fmt)
        lines = [f"CDP SORTED: {len(idx.cdps)} CDPS, BY CDP THEN |OFFSET|"]
        head = bytes(build_head(sgy, lines, fmt)) if fmt else open(sgy.path, "rb").read(sgy.data_start)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        cdp_flow.write_cdp_sorted(sgy.path, idx, dest + ".part", head)
        os.replace(dest + ".part", dest)
        out.append(markdown("Output file", f"✅ Wrote **{idx.traces:,} traces** in CDP order to `{dest}` "
                                           f"({os.path.getsize(dest) / 1e9:.2f} GB) - headers and samples unchanged."))

    fig = plotting.plot_gather(cg, label_by="offset", fig_height=sort_fig_height,
                               title=f"CDP {cdp} gather  -  {fold} traces, sorted by offset")
    out.append(figure(f"CDP {cdp} gather", fig))
    cols = ["CDP", "Fold", "Offset min", "Offset max", "Midpoint X", "Midpoint Y"] + (["IL", "XL"] if il is not None else [])
    rows = []
    for j in range(len(idx.cdps)):
        r = {"CDP": int(idx.cdps[j]), "Fold": int(idx.fold[j]), "Offset min": int(idx.off_min[j]),
             "Offset max": int(idx.off_max[j]), "Midpoint X": round(float(idx.mid_x[j]), 1),
             "Midpoint Y": round(float(idx.mid_y[j]), 1)}
        if il is not None:
            r["IL"], r["XL"] = int(il[j]), int(xl[j])
        rows.append(r)
    out.append(table(f"All CDPs, sorted ({len(idx.cdps):,})", cols, rows, paginate=True))
    return out
