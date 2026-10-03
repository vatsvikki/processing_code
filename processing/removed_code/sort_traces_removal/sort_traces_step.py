"""The "Sort Traces" Flow step, removed from the Flow on request (2026-10-01). Kept for reference only - nothing
imports this file. It went in functions/steps.py (with its explanation in functions/param_help.FUNCTION_INFO, below),
and the gather plot labelled sorted traces through PipeState.sorted_by -> plot_gather(label_by=...) in
functions/tools._flow_outputs. For a whole-survey sort into CDP gathers use the "CDP Sort" display step.
"""

_SORT_LABELS = {"offset": "offset", "channel": "channel", "receiver": "receiver line, station", "cdp": "CDP",
                "cdp_offset": "CDP, offset", "file": "file order"}


@step("Sort Traces", "Reorder the traces of the gather by a header key (the 'Trace order in gather' display setting "
      "is separate - this changes the data itself, so Save / Apply to whole data also get the new order).",
      params=[Param("sort_by", "Sort by", "choice", "offset", live=True, choices=list(_SORT_LABELS), group="Sort",
                    help="offset = ascending |offset|; channel = channel number; receiver = receiver line, then "
                         "station; cdp = CDP (ensemble) number; cdp_offset = by CDP, then by |offset| within it; "
                         "file = back to the order recorded in the file"),
              Param("descending", "Descending order", "bool", False, live=True, group="Sort",
                    help="Reverse the order above (largest / last key first, and within a composite key like "
                         "'CDP, offset' both parts reverse together)")],
      order=58)
def sort_traces(state: PipeState, sort_by: str = "offset", descending: bool = False):
    g = state.gather
    # "file": back to the order recorded in the file - by each trace's position in the source shot (state.src), not
    # the gather's current order, so it also undoes an earlier Sort Traces step
    order = np.argsort(state.src, kind="stable") if sort_by == "file" else trace_order(g, sort_by)
    if descending:
        order = order[::-1]
    new_g = ShotGather(g.ffid, g.i0, g.data[order], {k: v[order] for k, v in g.headers.items()}, g.dt_ms)
    new_state = replace(
        state, gather=new_g, dead=state.dead[order], bad=state.bad[order], src=state.src[order],
        marks=state.marks[order] if state.marks is not None else None,
        sorted_by="" if sort_by == "file" and not descending else sort_by,
    )
    note = f"sorted by {_SORT_LABELS.get(sort_by, sort_by)}" + (", descending" if descending else "")
    return new_state, note


FUNCTION_INFO_ENTRY = {
    "sort_traces": r"""
**What it does.** No maths - it computes a permutation of the trace positions from a header key and reorders the
gather's samples and headers by it (`functions/segy_io.trace_order`, the same key every other tool's *Trace order in
gather* display setting uses - but this step changes the data itself, not just how it is drawn):

- **offset** - ascending $|\text{offset}|$ (near to far)
- **channel** - ascending channel number (the header, not the position in the shot)
- **receiver** - receiver line, then station within the line (a stable two-level sort)
- **cdp** - ascending CDP (ensemble) number, the header word, not a bin computed from a grid
- **cdp, offset** - by CDP first, then by $|\text{offset}|$ within each CDP - of this shot's traces only (for the
  whole survey sorted into CDP gathers, use the **CDP Sort** display function)
- **file** - back to the order the traces were recorded in (undoes any of the above, also after an earlier Sort
  Traces step)

After this step the gather plot labels the traces by the key they were sorted on (2nd row of the trace axis).

**Descending** reverses the resulting order - for a composite key (*cdp, offset*) this reverses both parts together
(descending CDP, and within a CDP, descending offset), not just the outer key.

**Where to put it in the flow matters.** Steps after this one see the new order, including *Save data (SEG-Y)* and
*Apply to whole data* (each output trace still carries its own original 240-byte header, just written in the new
order). It also changes what "neighbour" means for a step that looks at position, not geometry - in particular
*Correct Dead Traces*' interpolation uses the nearest **live** trace on the same receiver line *by its current
position*; sorted by offset first, that finds the offset-nearest live trace on the line rather than the
physically-nearest one. Put *Sort Traces* **after** *Correct Dead Traces* (and any other position-sensitive step)
unless you specifically want that.
""",
}
