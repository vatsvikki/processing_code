"""The "CDP Sort" Flow step, moved to the 🛠 Tools menu on request (2026-10-03) - functions/cdp_sort_tool.py.
Kept for reference only; nothing imports this file."""

@step("CDP Sort", "Whole-survey CDP sort (as in the brute-stack notebook): every trace sorted by CDP, then by offset "
      "within each CDP; shows the gather of one CDP (read from the file, nearest offset first) and the list of all "
      "CDPs with their fold. The first run reads all trace headers; after that changing the CDP is quick. Does not "
      "change the gather.",
      params=[_FIG_HEIGHT,
              Param("cdp_number", "CDP to show  (0 = the highest fold)", "int", 0, min=0, step=1, live=True, group="CDP",
                    help="A CDP number from the table below; a number that is not in the file shows the nearest CDP")],
      order=72, category="Display")
def cdp_sort_step(state: PipeState, fig_height: float = 8.0, cdp_number: int = 0):
    if state.batch:            # nothing displays this inside a whole-data run - skip it for every shot
        return state, "CDP sort not built (whole-data run)"
    src = cdp_flow.CdpSource(state.path, state.flow)       # the data as the steps above left it
    idx = src.idx
    k = idx.position(int(cdp_number))
    cdp, fold = int(idx.cdps[k]), int(idx.fold[k])
    cg = src.gather(k)

    # IL / XL of every CDP (from its mean midpoint) when a grid is known: the header IL / XL or the corner table saved
    # for this file - the same grid the Acquisition Geometry / Fold maps use
    grid = None
    try:
        geom = geometry.read_geometry(state.path, 60)
        grid, _ = _resolve_grid(state.path, geom, grid_mod.corner_table(state.path), False, 60)
    except Exception:
        grid = None
    il = xl = None
    if grid is not None:
        il, xl = (np.rint(v).astype(np.int64) for v in grid.ilxl(idx.mid_x, idx.mid_y))

    asked = int(cdp_number)
    which = (" (the CDP with the highest fold)" if not asked else
             "" if asked == cdp else f" (the nearest CDP to {asked:,} - that number is not in the file)")
    md = (f"**{idx.traces:,} traces** sorted into **{len(idx.cdps):,} CDPs** "
          f"({int(idx.cdps[0]):,} – {int(idx.cdps[-1]):,}), by CDP and then by offset within each CDP · fold "
          f"{int(idx.fold.min())} – {int(idx.fold.max())} (median {int(np.median(idx.fold))})\n\n"
          f"Showing **CDP {cdp:,}**{which}: **{fold} traces**, offset {int(idx.off_min[k]):,} – "
          f"{int(idx.off_max[k]):,}"
          + (f", IL {int(il[k])} / XL {int(xl[k])}" if il is not None else "")
          + ".\n\n" + src.describe())
    fig = plotting.plot_gather(cg, label_by="offset", fig_height=fig_height,
                               title=f"CDP {cdp} gather  -  {fold} traces, sorted by offset (from the file)")
    cols = ["CDP", "Fold", "Offset min", "Offset max", "Midpoint X", "Midpoint Y"] + (["IL", "XL"] if il is not None else [])
    rows = []
    for j in range(len(idx.cdps)):
        r = {"CDP": int(idx.cdps[j]), "Fold": int(idx.fold[j]), "Offset min": int(idx.off_min[j]),
             "Offset max": int(idx.off_max[j]), "Midpoint X": round(float(idx.mid_x[j]), 1),
             "Midpoint Y": round(float(idx.mid_y[j]), 1)}
        if il is not None:
            r["IL"], r["XL"] = int(il[j]), int(xl[j])
        rows.append(r)
    out = [markdown("CDP sort", md), figure(f"CDP {cdp} gather", fig),
           table(f"All CDPs, sorted ({len(idx.cdps):,})", cols, rows, paginate=True)]
    return replace(state, view_extra=out), f"CDP {cdp} shown ({fold} traces, {len(idx.cdps):,} CDPs)"
