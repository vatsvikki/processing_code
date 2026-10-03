import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full", app_title="Processing Tool", css_file="custom.css")


@app.cell
def _():
    import os
    import sys
    import html
    from pathlib import Path
    from types import SimpleNamespace

    import marimo as mo

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from functions import segy_io
    from functions.param_help import FUNCTION_INFO, PARAM_INFO, WIDGET_INFO, info_text
    from functions import (DEFAULT_DIR, FORMATS, SEGY_FORMATS, BatchJob, choices_for, default_for, estimate_batch, file_context,
                              get_categories, get_functions, get_steps, plot_tag, resolve_output, save_outputs)

    def tip(text):
        """The (i) marker: hover it for the full explanation of the input it stands next to."""
        t = html.escape(str(text), quote=True).replace("\n", "&#10;")
        return f'<span class="qc-i" title="{t}">ⓘ</span>' if t else ""

    # File opened at start: $SEGY_FILE (set in run_app.sh / .env); empty -> type a path into the box.
    DEFAULT_FILE = os.environ.get("SEGY_FILE", "").strip()

    # Every function registered in functions becomes a toolbar button.
    # This file contains ONLY UI code -- to add a tool, write a decorated
    # function in functions/tools.py; nothing here needs to change.
    functions = {f.key: f for f in get_functions()}

    # parameter values used last, so e.g. the FFID carries over when you switch
    # from one function to another
    remembered = {}
    # outputs of the last run, for the Save button
    last = {}
    # zoom windows visited (for "Previous view")
    zoom_hist = []
    # the file the Load Data tool has read: its normalised path, the outputs it produced, the Load-button count
    # already handled, and the error of the last attempt
    loaded = {"path": None, "outs": None, "seen": 0, "error": None}
    # the whole-data job (a BatchJob runs in its own thread; the GUI only polls it)
    jobs = {}
    # the flow built in the Flow tool: the function of every place (its length = the number of functions), the parameter
    # values per place and function, and the stage shown (None = the last one)
    pipe_mem = {"slots": [], "params": {}, "view": None}
    return (BatchJob, DEFAULT_DIR, DEFAULT_FILE, FORMATS, Path, os, SEGY_FORMATS, SimpleNamespace, choices_for, default_for, estimate_batch,
            file_context, functions, get_categories, get_steps, html, info_text, jobs, last, loaded, mo, pipe_mem, plot_tag,
            remembered, resolve_output, save_outputs, segy_io, tip, zoom_hist, FUNCTION_INFO, PARAM_INFO,
            WIDGET_INFO)


@app.cell
def _(functions, mo):
    active, set_active = mo.state(next(iter(functions)), allow_self_loops=True)
    save_msg, set_save_msg = mo.state("")
    zoom_version, set_zoom_version = mo.state(0)     # bumped to rebuild the range widgets after a zoom
    load_version, set_load_version = mo.state(0)                         # bumped when Load Data has finished
    job_version, set_job_version = mo.state(0, allow_self_loops=True)    # bumped when a whole-data job starts / is dismissed
    pipe_version, set_pipe_version = mo.state(0, allow_self_loops=True)  # bumped when a function is added / moved / removed
    run_version, set_run_version = mo.state(0, allow_self_loops=True)    # bumped by a ▶ of the flow: compute now
    add_open, set_add_open = mo.state(False, allow_self_loops=True)      # the "add a function" picker is open
    return (active, add_open, job_version, load_version, pipe_version, run_version, save_msg, set_active,
            set_add_open, set_job_version, set_load_version, set_pipe_version, set_run_version, set_save_msg,
            set_zoom_version, zoom_version)


@app.cell
def _(mo):
    # long functions (the whole-survey header read, the full stack ...) report their progress through
    # functions.progress; `with show_progress():` around a run turns those reports into progress bars in the cell
    from contextlib import contextmanager
    from functions import progress

    @contextmanager
    def show_progress():
        bar = {}

        def _report(done, total, title):
            pct = int(100 * done / max(total, 1))
            if bar.get("title") != title:                 # a new phase: a new bar
                if bar.get("cm") is not None:
                    bar["cm"].__exit__(None, None, None)
                bar["cm"] = mo.status.progress_bar(total=100, title=title, show_rate=False, show_eta=True,
                                                   remove_on_exit=True)
                bar.update(bar=bar["cm"].__enter__(), title=title, pct=0)
            if pct > bar["pct"]:
                bar["bar"].update(increment=pct - bar["pct"], subtitle=f"{done:,} / {total:,}")
                bar["pct"] = pct

        try:
            with progress.reporting(_report):
                yield
        finally:
            if bar.get("cm") is not None:
                bar["cm"].__exit__(None, None, None)

    return (show_progress,)


@app.cell
def _(active, functions, get_categories, html, mo, os, path_input):
    # ---- TOP: the title strip, the 🛠 Tools menu and the active function's label.  The Data / Display / Processing /
    # QC functions are picked from the Flow card's ＋ Add a function list (left panel); the functions registered with
    # category="Tools" (standalone utilities, e.g. CDP Recalculate) are in the Tools menu here, and each opens in its
    # own browser tab (this app with ?tool=<key>&segy=<path>, see the TOOL WINDOW cells) - a separate session, so it
    # never touches this window's flow or loaded data.
    # The category label (e.g. "Display") still comes from @register(..., category=...).
    from urllib.parse import urlencode as _urlencode

    _cat_of = {_f.key: _n for _n, _fs in get_categories() for _f in _fs}
    _flow_key = next(_f.key for _f in functions.values() if _f.kind == "pipeline")
    _now = active()
    _active_cat = "" if _now == _flow_key else _cat_of.get(_now, "")
    _tools = [_f for _f in functions.values() if _f.category == "Tools"]
    # notebook tools: when run_app.sh started the notebooks server (port of this page + 1, NOTEBOOK_TOOLS=1), link
    # straight to the notebook there - a normal marimo page, with its own progress bars; else embed it in a tab here
    from pathlib import Path as _Path

    _nb_base = None
    _req = mo.app_meta().request
    if os.environ.get("NOTEBOOK_TOOLS") == "1" and _req is not None:
        _u = _req.url
        # the request is the page's websocket (ws / wss): the page itself is http / https
        _scheme = {"ws": "http", "wss": "https"}.get(_u.get("scheme") or "http", _u.get("scheme") or "http")
        _port = _u.get("port") or (443 if _scheme == "https" else 80)
        _nb_base = f"{_scheme}://{_u.get('hostname') or 'localhost'}:{int(_port) + 1}/"

    def _href(_f):
        _segy = path_input.value.strip()
        if _f.kind == "notebook" and _nb_base:
            return _nb_base + "?" + _urlencode({"file": _Path(_f.run()).name, "segy": _segy})
        return "?" + _urlencode({"tool": _f.key, "segy": _segy})

    _links = "".join(
        f'<a href="{html.escape(_href(_f))}" target="_blank" '
        f'rel="noopener"><b>{html.escape(_f.label)} ↗</b><small>{html.escape(_f.description)}</small></a>'
        for _f in _tools
    )
    _menu = (f'<details class="qc-menu"><summary>🛠 Tools</summary><div class="qc-menu-list">{_links}</div></details>'
             if _tools else "")
    topbar = mo.Html(
        '<div class="qc-top">'
        '<div class="qc-strip"><span class="qc-title">Processing Tool</span>' + _menu
        + f'<span class="qc-active">{html.escape(_active_cat) + " › " if _active_cat else ""}'
        f'<b>{html.escape(functions[_now].label)}</b></span></div>'
        "</div>"
    )
    return (topbar,)


@app.cell
def _(functions, mo):
    # ---- TOOL WINDOW: this browser tab was opened from the 🛠 Tools menu (?tool=<key>&segy=<path>; not "file",
    # which marimo itself uses to pick the notebook) -> the page shows
    # only that tool (its own file box, parameters, results), nothing of the flow.  tool_window = None: normal page.
    _qp = mo.query_params()
    _k = _qp.get("tool")
    tool_window = functions[_k] if isinstance(_k, str) and _k in functions and functions[_k].category == "Tools" else None
    tool_window_file = str(_qp.get("segy") or "") if tool_window is not None else ""
    return tool_window, tool_window_file


@app.cell
def _(WIDGET_INFO, default_for, html, info_text, mo, tip, tool_window, tool_window_file):
    # the tool tab's inputs: the file box, then the tool's parameters under their group headings in a compact
    # two-column grid (text boxes / checkboxes span both columns); the (i) next to each input has its explanation
    tw_path, tw_form = None, None
    if tool_window is not None and tool_window.kind != "notebook":
        tw_path = mo.ui.text(value=tool_window_file, label="SEG-Y file " + tip(WIDGET_INFO["segy_file"]), full_width=True)
        _elements, _parts, _group = {}, [], None
        for _p in tool_window.params:
            if _p.group != _group:
                if _group is not None:
                    _parts.append("</div>")
                _group = _p.group
                _parts.append(f'<div class="qc-form-group">{html.escape(_group)}</div><div class="qc-form-grid">')
            _val = default_for(_p, None, 0)
            _label = html.escape(_p.label) + " " + tip(info_text(_p))
            if _p.kind in ("int", "float"):
                _w = mo.ui.number(start=_p.min, stop=_p.max, step=_p.step or (1 if _p.kind == "int" else None),
                                  value=int(_val) if _p.kind == "int" else float(_val), label=_label, full_width=True)
            elif _p.kind == "bool":
                _w = mo.ui.checkbox(value=bool(_val), label=_label)
            elif _p.kind == "choice":
                _w = mo.ui.dropdown(options=_p.choices, value=_val, label=_label, full_width=True)
            else:
                _w = mo.ui.text(value=str(_val), label=_label, full_width=True)
            _elements[_p.key] = _w
            _wide = " qc-span2" if _p.kind in ("bool", "text", "textarea") else ""
            _parts.append(f'<div class="qc-form-cell{_wide}">{{{_p.key}}}</div>')
        if _group is not None:
            _parts.append("</div>")
        if _elements:
            tw_form = mo.ui.batch(mo.Html("".join(_parts)), _elements).form(
                submit_button_label="▶  Run", submit_button_tooltip=WIDGET_INFO["run_button"], bordered=False)
    return tw_form, tw_path


@app.cell
def _(Path, mo, render):
    # cards of the tool tab: one per result - its title with the download (CSV / full-resolution PNG) on the right,
    # then the table / text / interactive plotly figure
    import csv as _csv
    import dataclasses as _dc
    import io as _io

    TOOL_CARD = {"border": "1px solid rgba(128,128,128,.28)", "border-radius": "12px", "padding": "14px 16px",
                 "background": "rgba(128,128,128,.05)", "min-width": "0", "box-sizing": "border-box"}

    def _csv_bytes(o):
        _buf = _io.StringIO()
        _w = _csv.DictWriter(_buf, fieldnames=o.columns, extrasaction="ignore")
        _w.writeheader()
        _w.writerows(o.content)
        return _buf.getvalue().encode()

    def tool_card(o, stem):
        """One result of a Tools-window run as a card (None for "data" outputs, which a tool tab does not show)."""
        _action, _body = None, None
        if o.kind == "plotly":
            _c = o.content
            _action = mo.download(data=_c["png"], filename=_c["filename"], mimetype="image/png", label=_c["label"])
            _body = mo.vstack(([mo.md(f'<span style="opacity:.7">{_c["note"]}</span>')] if _c.get("note") else []) + [
                mo.ui.plotly(_c["fig"](), config={"displayModeBar": True, "toImageButtonOptions": {
                    "filename": Path(_c["filename"]).stem, "format": "png", "scale": 2}})], gap=0.3)
            return mo.vstack([mo.hstack([mo.md(f"#### {o.title}"), _action], justify="space-between", align="center",
                                        gap=1, wrap=True), _body], gap=0.5).style(
                {**TOOL_CARD, "width": "fit-content", "max-width": "100%"})
        if o.kind == "data":
            return None
        if o.kind == "table":
            _slug = "".join(_ch if _ch.isalnum() else "_" for _ch in o.title.lower()).strip("_")[:40]
            _action = mo.download(data=lambda _o=o: _csv_bytes(_o), filename=f"{stem}_{_slug}.csv", mimetype="text/csv",
                                  label="💾 CSV")
        _body = render(_dc.replace(o, title=""))
        _head = mo.hstack([mo.md(f"#### {o.title}")] + ([_action] if _action is not None else []),
                          justify="space-between", align="center", gap=1)
        return mo.vstack([_head, _body], gap=0.4).style(TOOL_CARD)

    return TOOL_CARD, tool_card


@app.cell
def _(FUNCTION_INFO, Path, TOOL_CARD, coerce, html, mo, show_progress, tool_card, tool_window, tw_form, tw_path):
    # the tool tab's page: title strip; inputs left; summary / corner points right.  The plotly figures and the long
    # tables come after it, each in its own cell (marimo limits the size of each cell's output): tool_plots, tool_tail
    tool_page, tool_plots, tool_tail, tool_stem = None, [], [], ""
    if tool_window is not None and tool_window.kind == "notebook":
        tool_page = mo.Html(
            '<div class="qc-tool"><div class="qc-top"><div class="qc-strip"><span class="qc-title">Processing Tool</span>'
            f'<span class="qc-active">Tools › <b>{html.escape(tool_window.label)}</b></span></div></div></div>'
        )
    elif tool_window is not None:
        tool_stem = f"{Path(tw_path.value.strip() or 'data').stem}_{tool_window.key}"
        if tw_form is not None and tw_form.value is None:
            _right = mo.callout(mo.md(f"**{tool_window.label}** — {tool_window.description}\n\n"
                                      "Check the file and the inputs on the left, then press **▶ Run**."), kind="info")
        else:
            try:
                # long work (e.g. the whole-survey header read) shows a progress bar while the tool runs
                with show_progress(), mo.status.spinner(title=f"Running {tool_window.label} ..."):
                    _outs = tool_window.run(tw_path.value,
                                            **coerce(tool_window.params, (tw_form.value if tw_form else {})))
                _cards = []
                for _o in _outs:
                    if _o.kind == "plotly":
                        tool_plots.append(_o)
                    elif _o.kind == "table" and _o.paginate:
                        tool_tail.append(_o)
                    elif (_c := tool_card(_o, tool_stem)) is not None:
                        _cards.append(_c)
                _right = mo.vstack(_cards, gap=1)
            except Exception as _e:  # show the problem instead of a stack trace
                _right = mo.callout(mo.md(f"**{type(_e).__name__}:** {_e}"), kind="danger")

        _explain = FUNCTION_INFO.get(tool_window.key)
        _left = mo.vstack([mo.md(f"#### 🛠 {tool_window.label}"),
                           mo.md(f'<span style="opacity:.7">{html.escape(tool_window.description)}</span>'),
                           tw_path] + ([tw_form] if tw_form is not None else [])
                          + ([mo.accordion({"ⓘ  Full explanation (what it does, the maths, the inputs)": mo.md(_explain)})]
                             if _explain else []), gap=0.8).style(TOOL_CARD)
        # a normal scrolling page (not the fixed-height qc-root): the plots and long tables follow it
        tool_page = mo.Html(
            '<div class="qc-tool"><div class="qc-top"><div class="qc-strip"><span class="qc-title">Processing Tool</span>'
            f'<span class="qc-active">Tools › <b>{html.escape(tool_window.label)}</b></span></div></div>'
            f'<div class="qc-tool-panels"><div>{_left}</div><div style="min-width:0">{_right}</div></div></div>'
        )
    return tool_page, tool_plots, tool_stem, tool_tail


@app.cell
def _(DEFAULT_FILE, WIDGET_INFO, mo, tip):
    path_input = mo.ui.text(value=DEFAULT_FILE, label="SEG-Y file " + tip(WIDGET_INFO["segy_file"]), full_width=True)
    return (path_input,)


@app.cell
def _(WIDGET_INFO, mo):
    # counter button: every press bumps the value, which re-runs the loader cell below
    load_button = mo.ui.button(
        label="▶  Load data", kind="success", value=0, on_click=lambda v: (v or 0) + 1,
        tooltip=WIDGET_INFO["load_button"],
    )
    return (load_button,)


@app.cell
def _(functions, load_button, loaded, mo, path_input, segy_io, set_load_version):
    # ---- LOAD DATA: runs once per press of the button, with a progress bar --------------------------
    # The file is not touched before this: typing a path only changes the box.  Everything else (FFID slider,
    # tools) works on the file recorded in `loaded`.
    if load_button.value != loaded["seen"]:
        loaded["seen"] = load_button.value
        _loader = next(_f for _f in functions.values() if _f.kind == "loader")
        _path = path_input.value
        _done = [0]
        with mo.status.progress_bar(
            total=100, title="Loading data", subtitle="Opening the file ...", show_rate=False, remove_on_exit=True,
        ) as _bar:
            def _progress(frac, text=""):
                _target = int(frac * 100)
                if _target > _done[0]:
                    _bar.update(increment=_target - _done[0], subtitle=text)
                    _done[0] = _target

            try:
                loaded.update(outs=_loader.run(_path, progress=_progress), path=segy_io.norm_path(_path), error=None)
            except Exception as _e:
                loaded.update(outs=None, path=None, error=f"**{type(_e).__name__}:** {_e}")
        set_load_version(lambda n: n + 1)
    return


@app.cell
def _(file_context, load_version, loaded, path_input, pipe_mem, remembered, segy_io):
    # ---- what the file header says: real FFIDs, shot sizes, record length ---
    # only for the file that Load Data has read (a different path in the box -> "not loaded" again)
    load_version()
    remembered.clear()                       # new file -> forget the old file's values
    pipe_mem["params"].clear()
    ctx, ctx_error, not_loaded = None, None, False
    if loaded["path"] is not None and loaded["path"] == segy_io.norm_path(path_input.value):
        try:
            ctx = file_context(path_input.value)
        except Exception as _e:
            ctx_error = f"**{type(_e).__name__}:** {_e}"
    else:
        not_loaded = True
    return ctx, ctx_error, not_loaded


@app.cell
def _(ctx, mo):
    # ---- SURVEY GRID (data IL / XL corner points): asked once, in the Data card, for the loaded file - and used by
    # every function that needs the data's IL / XL (Acquisition Geometry / Fold, NMO Correction, CDP Stack, the CDP lists)
    from functions import grid as survey_grid_store
    grid_reset = None
    if ctx is not None:
        _path = ctx.sgy.path
        grid_reset = mo.ui.button(label="↺  Use the header IL / XL", value=0,
                                  on_click=lambda v, _p=_path: (survey_grid_store.forget_saved(_p), (v or 0) + 1)[1],
                                  tooltip="Forget the table saved for this file and go back to the IL / XL of the headers")
    return grid_reset, survey_grid_store


@app.cell
def _(ctx, grid_reset, mo, survey_grid_store):
    survey_grid = None
    if ctx is not None:
        grid_reset.value if grid_reset is not None else None    # (rebuilt after "Use the header IL / XL")
        _rows = survey_grid_store.corner_table(ctx.sgy.path)
        survey_grid = mo.ui.data_editor([dict(_r) for _r in _rows] + [{"IL": "", "XL": "", "X": "", "Y": ""}
                                                                     for _ in range(max(0, 4 - len(_rows)))],
                                        label="Corner points: IL, XL, X, Y - one per row")
    return (survey_grid,)


@app.cell
def _(ctx, survey_grid):
    # the table as typed: saved for this file when it differs from the headers' grid (dropped when it equals them);
    # grid_rev changes with every edit, so what depends on the grid on screen runs again
    grid_msg, grid_rev = "", 0
    if ctx is not None and survey_grid is not None:
        from functions import cdp_sort as _cdp_sort, geometry as _geometry
        from functions.steps import _resolve_grid
        try:
            _g, _notes = _resolve_grid(ctx.sgy.path, _geometry.read_geometry(ctx.sgy.path, 60), survey_grid.value, False, 60)
            grid_msg = "  \n".join(_notes)
            if _g is not None:
                grid_msg += (f"  \nGrid: IL {_g.il_range[0]:.0f} – {_g.il_range[1]:.0f}, XL {_g.xl_range[0]:.0f} – "
                             f"{_g.xl_range[1]:.0f}, bins {_g.il_spacing:,.1f} × {_g.xl_spacing:,.1f}")
        except Exception as _e:
            grid_msg = f"⚠ **Survey grid not used:** {_e}"
        _cdp_sort._top_labels.cache_clear()                  # the CDP lists show IL / XL: with this grid from now on
        grid_rev = hash(repr(survey_grid.value))
    return grid_msg, grid_rev


@app.cell
def _(active, functions):
    func = functions[active()]
    return (func,)


@app.cell
def _(html, info_text, mo, tip):
    # ---- widget factory: one widget per Param, grouped under headings -------
    def panel(params, value_of, live, on_change=None, skip_groups=(), choices_of=None):
        # choices_of(p): the options of a "choice" widget (default p.choices) - e.g. read from the loaded file
        _elements, _lines, _group = {}, [], None
        _now = {_q.key: value_of(_q) for _q in params if _q.kind != "ffid"}
        for _p in params:
            if _p.kind == "ffid" or not _p.shown(_now):        # (a setting for another choice: not shown)
                continue
            if _p.group != _group:
                _group = _p.group
                if _group and _group not in skip_groups:
                    _lines.append(f"**{html.escape(_group)}**")
            _val = value_of(_p)
            if _p.kind in ("int", "float"):
                # header-derived defaults (e.g. last trace) are 0 until a file is loaded: keep them inside the limits
                if _p.min is not None:
                    _val = max(_val, _p.min)
                if _p.max is not None:
                    _val = min(_val, _p.max)
            _label = html.escape(_p.label) + " " + tip(info_text(_p))       # (i): the full explanation on hover
            if _p.kind == "int":
                _w = mo.ui.number(start=_p.min, stop=_p.max, step=_p.step or 1, value=int(_val), label=_label, debounce=live)
            elif _p.kind == "float":
                _w = mo.ui.number(start=_p.min, stop=_p.max, step=_p.step, value=float(_val), label=_label, debounce=live)
            elif _p.kind == "bool":
                _w = mo.ui.checkbox(value=bool(_val), label=_label)
            elif _p.kind == "choice":
                _opts = list(choices_of(_p) if choices_of is not None else _p.choices or [])
                if _val not in _opts:                 # e.g. a list read from another file: start at its first entry
                    _val = _opts[0] if _opts else None
                _w = mo.ui.dropdown(options=_opts, value=_val, label=_label)
            elif _p.kind == "textarea":
                _w = mo.ui.text_area(value=str(_val), label=_label, rows=6, full_width=True, placeholder=_p.example)
            elif _p.kind == "table":                  # rows keyed by column name, edited in place
                _w = mo.ui.data_editor([dict(_r) for _r in _val], label=_label, column_sizing_mode="fit")
            else:
                _w = mo.ui.text(value=str(_val), label=_label)
            _elements[_p.key] = _w
            _hint = f'<br/><small style="opacity:.65">{html.escape(_p.help)}</small>' if _p.help else ""
            _lines.append("{" + _p.key + "}" + _hint)
        if not _elements:
            return None
        return mo.ui.batch(mo.md("\n\n".join(_lines)), _elements, on_change=on_change)

    return (panel,)


@app.cell
def _(get_steps):
    # the functions the flow can use: the Processing ones, the two QC steps (Detect Dead / Bad Traces) that only
    # flag traces for the steps after them, and the Display ones (Plot Shot Gather, Acquisition Geometry / Fold)
    # that show a map / gather for the current stage instead of changing it - all in the order they'd usually run
    # (detect, then correct / process, then a display function wherever it is useful to look).
    pipe_steps = {_s.label: _s for _s in get_steps() if _s.category in ("Processing", "QC", "Display")}
    return (pipe_steps,)


@app.cell
def _(FUNCTION_INFO, SimpleNamespace, WIDGET_INFO, add_open, functions, html, mo, pipe_mem, pipe_steps, pipe_version,
       set_active, set_add_open, set_pipe_version, set_run_version):
    # ---- FLOW: a persistent card (left panel, always) - not tied to which function is selected above, and not
    # gated on data being loaded: its ＋ list is also where Load Data and EBCDIC & Headers now live (there is no
    # Data class on the top bar any more), so it must stay reachable before a file is loaded too.
    # A ＋ at the end of the list opens the picker; click a function there to add it at the end (the ＋ then appears
    # again, after the new function, ready to add the next one).  On each function: ▲ ▼ move it, the red − removes it.
    # Any interaction switches the right panel to the flow's own result (see the RIGHT cell, func.kind == "pipeline").
    # The list lives in pipe_mem["slots"]; every change bumps pipe_version so this cell rebuilds its buttons.
    # flow_toggle / flow_add / pipe_up / pipe_down / pipe_remove are globals on purpose: marimo drops buttons nothing
    # refers to any more, and a click on such a button would do nothing.
    pipe_slots, flow_toggle, flow_add, pipe_up, pipe_down, pipe_remove = None, None, None, [], [], []
    pipe_run, run_all = [], None
    if True:
        pipe_version()
        pipe_mem["slots"][:] = [_l for _l in pipe_mem["slots"] if _l in pipe_steps]
        _flow_key = next(_f.key for _f in functions.values() if _f.kind == "pipeline")

        def _changed():
            pipe_mem["view"] = None                      # a changed flow shows its last stage again
            set_active(_flow_key)                         # show the flow's own result on the right
            set_pipe_version(lambda n: n + 1)

        def _add(_label):
            if len(pipe_mem["slots"]) < 12:
                pipe_mem["slots"].append(_label)
                set_add_open(False)                       # picked: close the list, the ＋ shows again at the new end
                _changed()

        def _remove(_i):
            _slots = pipe_mem["slots"]
            if 0 <= _i < len(_slots):
                del _slots[_i]
                # the parameters of the later functions move up one place with them
                _kept = {(_k[0] - (_k[0] > _i), _k[1]): _v for _k, _v in pipe_mem["params"].items() if _k[0] != _i}
                pipe_mem["params"].clear()
                pipe_mem["params"].update(_kept)
                _changed()

        def _move(_i, _d):
            _slots, _j = pipe_mem["slots"], _i + _d
            if 0 <= _i < len(_slots) and 0 <= _j < len(_slots):
                _slots[_i], _slots[_j] = _slots[_j], _slots[_i]
                _swap = {_i: _j, _j: _i}                 # each function takes its parameters along
                _moved = {(_swap.get(_k[0], _k[0]), _k[1]): _v for _k, _v in pipe_mem["params"].items()}
                pipe_mem["params"].clear()
                pipe_mem["params"].update(_moved)
                _changed()

        def _toggle_add(_):
            set_active(_flow_key)
            set_add_open(lambda v: not v)

        def _open_tool(_k):
            # a Data / Display function (Load Data, EBCDIC & Headers, Plot Shot Gather, Acquisition Geometry /
            # Fold): opens on its own, on the right - it is not a flow step (no state to chain), so it is not
            # added to the flow
            set_active(_k)
            set_add_open(False)

        _n, _open = len(pipe_mem["slots"]), add_open()
        pipe_slots = SimpleNamespace(value=list(pipe_mem["slots"]))       # the functions, top to bottom
        pipe_up = [mo.ui.button(label="▲", tooltip=WIDGET_INFO["step_up"], disabled=_i == 0,
                                on_click=lambda _, _i=_i: _move(_i, -1)) for _i in range(_n)]
        pipe_down = [mo.ui.button(label="▼", tooltip=WIDGET_INFO["step_down"], disabled=_i == _n - 1,
                                  on_click=lambda _, _i=_i: _move(_i, 1)) for _i in range(_n)]
        pipe_remove = [mo.ui.button(label="−", kind="danger", tooltip=WIDGET_INFO["step_remove"],
                                    on_click=lambda _, _i=_i: _remove(_i)) for _i in range(_n)]

        def _run(_k):
            # ▶: compute the flow up to stage _k now on the selected shot and show that stage; nothing runs before a ▶
            # is pressed. ▶ Run flow (_k = -1) also runs the whole flow on the whole data in one pass (the RIGHT cell
            # starts it, it has the parameters): every shot read once, all functions in memory, the result written once
            _labels = [_l for _l in pipe_mem["slots"] if _l in pipe_steps]
            if _k < 0:
                pipe_mem["whole_to"] = len(_labels)
            _k = len(_labels) if _k < 0 else _k
            pipe_mem["run_to"] = _k
            pipe_mem["view"] = f"{_k}. {_labels[_k - 1]}" if _k > 0 else "0. Input (raw)"
            set_active(_flow_key)
            set_run_version(lambda n: n + 1)

        pipe_run = [mo.ui.button(label="▶", kind="success", tooltip=WIDGET_INFO["step_run"],
                                 on_click=lambda _, _i=_i: _run(_i + 1)) for _i in range(_n)]
        run_all = mo.ui.button(label="▶  Run flow", kind="success", tooltip=WIDGET_INFO["run_all"], disabled=_n == 0,
                               on_click=lambda _: _run(-1))
        flow_toggle = mo.ui.button(
            label="✕  Cancel" if _open else "＋  Add a function", kind="neutral" if _open else "success",
            disabled=_n >= 12 and not _open, tooltip=WIDGET_INFO["flow_add"],
            on_click=_toggle_add,
        )
        if _open:
            # a separate list, one block per function: its button (full width, on its own line, so a long name never
            # wraps inside the button), its short description, and a clickable ⓘ that opens the full explanation -
            # the maths it uses and how each of its parameters changes the result (functions/param_help.FUNCTION_INFO).
            # Click the button itself (not the ⓘ) to add that function and close the list.
            _addcard = {
                "border": "1px dashed rgba(43,154,102,.5)", "border-radius": "10px", "padding": "10px 12px",
                "background": "rgba(43,154,102,.06)", "min-width": "0", "box-sizing": "border-box",
            }

            def _block(_label, _key, _description, _onclick):
                return mo.vstack(
                    [
                        mo.ui.button(label=_label, kind="neutral", full_width=True, tooltip=_description, on_click=_onclick),
                        mo.md(f'<span style="opacity:.75;font-size:.85rem">{html.escape(_description)}</span>'),
                        mo.accordion({"ⓘ  Full explanation (maths & parameters)":
                                     mo.md(FUNCTION_INFO.get(_key, "No further explanation yet."))}),
                    ],
                    gap=0.3,
                )

            # one flat list, Load Data first (it has to happen before anything else works), then the rest of the
            # Processing / QC / Display functions (click = add to the flow - Display ones show a map / gather for
            # that stage without changing it), then EBCDIC & Headers (click = open on its own - it reads / reports
            # on the shot, it does not fit into a flow stage).
            _standalone_fns = sorted((_f for _f in functions.values() if _f.category == "Data"),
                                     key=lambda _f: _f.order)
            _load_key = next((_f.key for _f in _standalone_fns if _f.kind == "loader"), None)
            _load, _standalone_fns = (
                next((_f for _f in _standalone_fns if _f.key == _load_key), None),
                [_f for _f in _standalone_fns if _f.key != _load_key],
            )
            _blocks = []
            if _load is not None:
                _blocks.append(_block(_load.label, _load.key, _load.description, lambda _, _k=_load.key: _open_tool(_k)))
            _blocks += [_block(_l, _s.key, _s.description, lambda _, _l=_l: _add(_l)) for _l, _s in pipe_steps.items()]
            _blocks += [_block(_f.label, _f.key, _f.description, lambda _, _k=_f.key: _open_tool(_k)) for _f in _standalone_fns]
            flow_add = mo.vstack([mo.md("**Pick a function:**")] + _blocks, gap=1.0).style(_addcard)
    return flow_add, flow_toggle, pipe_down, pipe_remove, pipe_run, pipe_slots, pipe_up, run_all


@app.cell
def _(FUNCTION_INFO, choices_for, ctx, default_for, functions, mo, panel, pipe_mem, pipe_slots, pipe_steps, set_active,
      set_pipe_version):
    # parameters of every function in the flow (the values are remembered per place and function), plus its full
    # explanation (maths & how the parameters affect the result) - both collapsed by default, in one accordion.
    pipe_params, pipe_cards = None, {}
    if pipe_slots is not None:
        _panels, _explain = {}, {}
        _flow_key = next(_f.key for _f in functions.values() if _f.kind == "pipeline")
        for _i, _label in enumerate(pipe_slots.value):
            _st = pipe_steps.get(_label)
            if _st is None:
                continue

            def _value(p, _i=_i, _k=_st.key):
                return pipe_mem["params"].get((_i, _k), {}).get(p.key, default_for(p, ctx, 0))

            def _changed(v, _i=_i, _k=_st.key, _st=_st):
                # merged into what was remembered, so the values of settings hidden now come back when shown again
                _before = {_q.key: _value(_q, _i, _k) for _q in _st.params}
                pipe_mem["params"][(_i, _k)] = {**pipe_mem["params"].get((_i, _k), {}), **dict(v)}
                # a setting other settings depend on (e.g. Velocity for NMO) changed: rebuild the card with the right
                # settings showing, and keep it open
                _controls = {_c for _q in _st.params for _c in (_q.show_if or {})}
                if any(_c in v and v[_c] != _before.get(_c) for _c in _controls):
                    pipe_mem["expand"] = str(_i)
                    set_pipe_version(lambda n: n + 1)
                set_active(_flow_key)

            if _st.params:
                # the widgets must be shown through the dictionary (pipe_params[key]), or edits would not reach
                # pipe_params.value
                _panels[str(_i)] = panel(_st.params, _value, True, on_change=_changed,
                                         choices_of=lambda _p: choices_for(_p, ctx))
            _explain[str(_i)] = mo.md(FUNCTION_INFO.get(_st.key, "No further explanation yet."))
        pipe_params = mo.ui.dictionary(_panels)
        # collapsible cards so three functions do not fill the whole left panel; the card whose settings just changed
        # shape stays open (marimo >= 0.25 can; an older one shows it closed)
        import inspect as _inspect
        _can_expand = "expanded" in _inspect.signature(mo.accordion).parameters
        _open = pipe_mem.pop("expand", None)
        pipe_cards = {
            _k: mo.accordion({**({"⚙  parameters": pipe_params[_k]} if _k in _panels else {}),
                              "ⓘ  full explanation (maths & parameters)": _explain[_k]},
                             **({"expanded": ["⚙  parameters"]} if _can_expand and _open == _k and _k in _panels else {}))
            for _k in _explain
        }
    return pipe_cards, pipe_params


@app.cell
def _(WIDGET_INFO, functions, mo, pipe_mem, pipe_slots, pipe_steps, run_version, set_active, tip):
    # which stage of the flow to look at (and save): the input or the result after any function
    run_version()                             # (rebuilt after a ▶, which also sets the stage it ran to)
    pipe_view = None
    if pipe_slots is not None and pipe_slots.value:
        _names = ["0. Input (raw)"] + [
            f"{_k}. {_label}" for _k, _label in enumerate([_l for _l in pipe_slots.value if _l in pipe_steps], 1)
        ]
        _flow_key = next(_f.key for _f in functions.values() if _f.kind == "pipeline")

        def _view_changed(v):
            pipe_mem.update(view=v)
            set_active(_flow_key)

        pipe_view = mo.ui.dropdown(
            options=_names, label="View stage " + tip(WIDGET_INFO["view_stage"]),
            value=pipe_mem["view"] if pipe_mem["view"] in _names else _names[-1],
            on_change=_view_changed,
        )
    return (pipe_view,)


@app.cell
def _(PARAM_INFO, ctx, func, mo, remembered, tip):
    # ---- FFID scroll bar: only the FFIDs that exist in the file -------------
    ffid_slider = None
    if ctx is not None and any(_p.kind == "ffid" for _p in func.params):
        _cur = int(remembered.get("ffid") or 0)
        _cur = ctx.index.nearest(_cur) if _cur > 0 else ctx.ffids[0]
        ffid_slider = mo.ui.slider(
            steps=ctx.ffids, value=_cur, debounce=True, show_value=True, include_input=True, full_width=True,
            label="Shot FFID " + tip(PARAM_INFO["ffid"]), on_change=lambda v: remembered.update(ffid=v),
        )
    return (ffid_slider,)


@app.cell
def _(ctx, default_for, ffid_slider, func, panel, remembered, zoom_hist, zoom_version):
    # ---- LIVE parameters: change -> the tool re-runs at once ----------------
    # Their starting values come from the header: last time = ns * dt, last trace
    # = number of traces in the selected shot.  Trace range resets with the shot.
    # The "Display" group (clip / AGC / time & trace range / figure height) is split out here and shown, in the
    # layout cell, as its own collapsible, tinted card, set apart from the tool's own parameters.
    zoom_version()                            # rebuild after a mouse zoom / reset (new values in `remembered`)
    _ffid = int(ffid_slider.value) if ffid_slider is not None else 0
    if remembered.get("_range_ffid") != _ffid:
        remembered.pop("trace_min", None)
        remembered.pop("trace_max", None)
        remembered["_range_ffid"] = _ffid
        zoom_hist.clear()
    _value_of = lambda p: remembered.get(p.key, default_for(p, ctx, _ffid))
    _live_params = [_p for _p in func.params if _p.live]
    live = panel([_p for _p in _live_params if _p.group != "Display"], _value_of, True,
                on_change=lambda v: remembered.update(v), skip_groups=("Shot",))
    live_display = panel([_p for _p in _live_params if _p.group == "Display"], _value_of, True,
                         on_change=lambda v: remembered.update(v), skip_groups=("Display",))
    return live, live_display


@app.cell
def _(WIDGET_INFO, ctx, default_for, func, panel, remembered):
    # ---- FORM parameters (thresholds ...): applied with the Run button ------
    _panel = panel(
        [_p for _p in func.params if not _p.live],
        lambda p: remembered.get(p.key, default_for(p, ctx, 0)),
        False,
    )
    form = (
        _panel.form(
            submit_button_label="▶  Run" if not func.autorun else "▶  Apply",
            submit_button_tooltip=WIDGET_INFO["run_button"],
            bordered=False,
            on_change=lambda v: remembered.update(v) if v else None,
        )
        if _panel is not None
        else None
    )
    return (form,)


@app.cell
def _(html, mo):
    # ---- generic renderers for the Output objects a function returns -------
    def html_table(columns, rows, mono):
        _th = "".join(
            f'<th style="text-align:left;padding:2px 10px;border-bottom:2px solid #8886">{html.escape(c)}</th>'
            for c in columns
        )
        _body = []
        for _r in rows:
            _tds = []
            for _c in columns:
                _style = "padding:1px 10px;border-bottom:1px solid #8883;vertical-align:top;"
                if _c in mono:
                    _style += "font-family:ui-monospace,Menlo,Consolas,monospace;white-space:pre;"
                _tds.append(f'<td style="{_style}">{html.escape(str(_r.get(_c, "")))}</td>')
            _body.append("<tr>" + "".join(_tds) + "</tr>")
        return mo.Html(
            '<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:13px">'
            f"<thead><tr>{_th}</tr></thead><tbody>{''.join(_body)}</tbody></table></div>"
        )

    def render(o):
        if o.kind == "markdown":
            _body = mo.md(o.content)
        elif o.kind == "image":
            _body = mo.image(o.content, style={"width": "100%", "height": "auto"})
        elif o.kind == "flip":
            # flip-flop: the frames in one place, the tabs switch between them (same size, same scaling)
            _body = mo.ui.tabs({_fr["label"]: mo.image(_fr["png"], style={"width": "100%", "height": "auto"})
                                for _fr in o.content})
        elif o.kind == "table" and o.paginate:
            _body = mo.vstack([mo.ui.table(
                [{c: r.get(c, "") for c in o.columns} for r in o.content],
                selection=None, page_size=15, show_column_summaries=False,
            )]).style({"max-width": "100%", "overflow-x": "auto", "min-width": "0"})
        elif o.kind == "table":
            _body = html_table(o.columns, o.content, o.mono)
        else:
            _body = mo.md(str(o.content))
        return mo.vstack([mo.md(f"#### {o.title}") if o.title else mo.md(""), _body], gap=0.3)

    def coerce(params, values):
        _cast = {"int": int, "float": float, "bool": bool, "ffid": int}
        return {
            p.key: _cast.get(p.kind, lambda x: x)(values[p.key])
            for p in params
            if values.get(p.key) is not None          # an emptied number box -> the function's default
        }

    return coerce, render


@app.cell
def _(Path, coerce, ctx, default_for, ffid_slider, form, func, grid_rev, last, live, live_display, loaded, mo, not_loaded, path_input, show_progress,
       pipe_mem, pipe_params, pipe_slots, pipe_steps, pipe_view, plot_tag, render, run_version, start_whole,
       fk_model):
    from functions.noise import polygon_rows
    from functions.steps import FK_MODES
    # ---- RIGHT: run the selected function, show its outputs ----------------
    run_version()  # (runs again after a ▶ of the flow)
    grid_rev  # (runs again when the survey grid of the Data card changes)
    _vals = {}
    if ffid_slider is not None:
        _vals["ffid"] = ffid_slider.value
    if live is not None:
        _vals.update(live.value)
    if live_display is not None:
        _vals.update(live_display.value)

    _form_vals = None
    if form is not None:
        # untouched form: tools that are cheap run at once with the defaults shown
        _form_vals = form.value if form.value is not None else (form.element.value if func.autorun else None)

    last.clear()
    zoom_select, right_top = None, None
    if func.kind == "loader":
        # Load Data: what the last load produced (the work itself is done by the loader cell, with its progress bar)
        if loaded["error"]:
            right = mo.callout(mo.md(loaded["error"]), kind="danger")
        elif not not_loaded and loaded["outs"]:
            right = mo.vstack([render(_o) for _o in loaded["outs"]], gap=1.5).style({"min-width": "0", "max-width": "100%"})
        else:
            right = mo.callout(
                mo.md(f"**{func.label}** — {func.description}\n\n"
                      "Type the SEG-Y file path on the left, then press **▶ Load data**."),
                kind="info",
            )
    elif not_loaded:
        right = mo.callout(
            mo.md("**No data loaded yet.** Open **Load Data** (in the Flow's **＋ Add a function** list, under "
                  "\"open one of these on its own\") and press **▶ Load data**. The file in the *SEG-Y file* box is "
                  "read once, then every tool works on it."),
            kind="warn",
        )
    elif func.kind == "pipeline" and pipe_slots is not None and not pipe_slots.value:
        right = mo.callout(
            mo.md("**The flow is empty.** Click **＋ Add a function** in the Flow card on the left; the functions "
                  "then run from top to bottom on the selected shot."),
            kind="info",
        )
    elif form is not None and _form_vals is None:
        right = mo.callout(
            mo.md(
                f"**{func.label}** — {func.description}\n\n"
                "Set the parameters on the left, then press **▶ Run**."
            ),
            kind="info",
        )
    else:
        _vals.update(_form_vals or {})
        _extra, _whole_msg = {}, None
        if func.kind == "pipeline" and pipe_slots is not None:
            # the flow: one {"step": key, "params": {...}} per place, top to bottom; view = the stage shown
            _extra["steps"] = [
                {"step": pipe_steps[_label].key, "params": dict(pipe_params.value.get(str(_i)) or {})}
                for _i, _label in enumerate(pipe_slots.value)
                if _label in pipe_steps
            ]
            _extra["view"] = list(pipe_view.options).index(pipe_view.value)
            # nothing is computed unless a ▶ was pressed (then run up to that stage); otherwise the stored results show
            _extra["run_to"] = pipe_mem.pop("run_to", None)
            # ...and the same ▶ runs the flow up to there on the whole data, in the background (card bottom right)
            _whole_to = pipe_mem.pop("whole_to", None)
            _whole_msg = start_whole(path_input.value, _extra["steps"], _whole_to) if _whole_to is not None else None
            last["flow_all"] = _extra["steps"]
            # the whole-data run uses the flow as set up, up to the stage on screen - whether its steps ran here or not
            last["flow_steps"] = _extra["steps"][:_extra["view"]]
            last["flow_path"] = path_input.value
        try:
            with show_progress():               # long functions (header read, stacking) show a progress bar here
                _outs = func.run(path_input.value, **coerce(func.params, _vals), **_extra)
            _stem = f"{Path(path_input.value).stem}_{func.key}" + (f"_FFID{_vals['ffid']}" if _vals.get("ffid") else "")
            # what the figures were made with, for the file names: parameters that differ from what the widgets started
            # with (header-derived starting values such as the record length do not count)
            _used = coerce(func.params, _vals)
            _start = {_p.key: default_for(_p, ctx, int(_vals.get("ffid") or 0)) for _p in func.params}
            _tag = plot_tag(_used, _start, flow=_extra.get("steps"), view=_extra.get("view"),
                            default_of=lambda _p: default_for(_p, ctx, 0))
            last.update(outs=_outs, stem=_stem, tag=_tag)
            _blocks = []
            for _o in _outs:
                if _o.kind == "data":                  # processed data: only offered by the Save data button
                    continue
                if _o.kind == "image" and _o.pick == "fk" and _o.figure is not None:
                    # the F-K Filter's F-K domain goes into the polygon editor (fk_editor.py), shown with the gather
                    # filtered by the current zone next to it (layout cell); edits re-run the filter (FK PICK cell)
                    _slot = _extra.get("view", 0) - 1
                    _p = dict((pipe_params.value or {}).get(str(_slot)) or {})
                    _manual = _p.get("fk_mode") == FK_MODES[1]
                    fk_model.show(_o.content, _o.figure, polygon_rows(_p.get("fk_polygon")) if _manual else [],
                                  _p.get("fk_mirror", True))
                    last.update(fk_slot=_slot, fk_show=True,
                                fk_live=next((_q for _q in _outs if _q.kind == "image" and _q.pick == "fk_live"), None))
                    continue
                if _o.kind == "image" and _o.pick == "fk_live":
                    continue                                        # (drawn next to the polygon editor)
                if _o.kind == "image" and _o.zoom and _o.figure is not None and zoom_select is None:
                    # the gather: drag a box on it to zoom (x = trace position, y = time in ms)
                    _o.figure.set_dpi(90)          # on-screen size only; saved figures use their own dpi
                    zoom_select = mo.ui.matplotlib(_o.figure.axes[0], debounce=True)
                    right_top = mo.vstack([mo.md(f"#### {_o.title}"), zoom_select], gap=0.3)
                else:
                    _blocks.append(render(_o))
            last["zoom"] = zoom_select is not None
            if func.kind == "pipeline" and _whole_msg:
                _blocks.insert(0, mo.callout(mo.md(_whole_msg), kind="info"))
            right = mo.vstack(_blocks, gap=1.5).style({"min-width": "0", "max-width": "100%"})
        except Exception as _e:  # show the problem instead of a stack trace
            right = mo.callout(mo.md(f"**{type(_e).__name__}:** {_e}"), kind="danger")
    return right, right_top, zoom_select


@app.cell
def _(mo):
    # ---- the F-K Filter's polygon editor (fk_editor.py): one instance for the session; the RIGHT cell puts each new
    # F-K plot into it (fk_model), the layout cell shows it (fk_editor), the FK PICK cell applies the edits
    from fk_editor import FkEditor
    fk_model = FkEditor()
    fk_editor = mo.ui.anywidget(fk_model)
    return fk_editor, fk_model


@app.cell
def _(fk_editor, last, pipe_mem, set_pipe_version, set_run_version):
    # ---- FK PICK: every edit made in the polygon editor (point added / moved / deleted, polygon removed) becomes the
    # F-K Filter's reject zone and is applied at once (the gather beside the editor shows the result)
    from functions.noise import F_COL as _F_COL, K_COL as _K_COL, Z_COL as _Z_COL
    from functions.steps import FK_MODES as _FK_MODES
    _v = fk_editor.value or {}
    _slot = last.get("fk_slot")
    if _v.get("rev", 0) != pipe_mem.get("fk_rev") and _slot is not None and _slot >= 0:
        pipe_mem["fk_rev"] = _v.get("rev", 0)                  # (a new plot put in by Python does not count)
        _rows = [{_Z_COL: _z, _K_COL: _k, _F_COL: _f}
                 for _z, _poly in enumerate(_v.get("polys") or [], 1) for _k, _f in _poly]
        _key = (_slot, "fk_filter_step")
        pipe_mem["params"][_key] = {**pipe_mem["params"].get(_key, {}), "fk_mode": _FK_MODES[1], "fk_polygon": _rows}
        if _slot < len(pipe_mem["slots"]):
            pipe_mem["run_to"] = _slot + 1
            pipe_mem["view"] = f"{_slot + 1}. {pipe_mem['slots'][_slot]}"
        set_pipe_version(lambda n: n + 1)
        set_run_version(lambda n: n + 1)
    return


@app.cell
def _(remembered, set_zoom_version, zoom_hist, zoom_select):
    # ---- ZOOM: a box (or lasso) dragged on the gather sets the plot ranges ---
    # x = trace position, y = time in ms.  The values go into `remembered` and the range
    # widgets are rebuilt from it, so the number boxes always show the zoomed window.
    _sel = zoom_select.value if zoom_select is not None else None
    if _sel:
        if hasattr(_sel, "x_min"):
            _xs, _ys = (_sel.x_min, _sel.x_max), (_sel.y_min, _sel.y_max)
        else:                                          # lasso: use its bounding box
            _xs = [_v[0] for _v in _sel.vertices]
            _ys = [_v[1] for _v in _sel.vertices]
        _x0, _x1 = min(_xs), max(_xs)
        _y0, _y1 = min(_ys), max(_ys)
        if _x1 - _x0 >= 1 and _y1 - _y0 >= 10:        # ignore a plain click / tiny drag
            zoom_hist.append({_k: remembered.get(_k) for _k in ("trace_min", "trace_max", "t_min_ms", "t_max_ms")})
            _p0 = max(int(round(_x0)), 1)
            remembered.update(
                trace_min=_p0,
                trace_max=max(int(round(_x1)), _p0 + 1),
                t_min_ms=float(max(round(_y0), 0)),
                t_max_ms=float(round(_y1)),
            )
            set_zoom_version(lambda v: v + 1)
    return


@app.cell
def _(WIDGET_INFO, mo, remembered, set_zoom_version, zoom_hist):
    def _reset(_):
        zoom_hist.clear()
        for _k in ("trace_min", "trace_max", "t_min_ms", "t_max_ms"):
            remembered.pop(_k, None)                  # back to the header values: full record, all traces
        set_zoom_version(lambda v: v + 1)

    def _back(_):
        if not zoom_hist:
            return
        for _k, _v in zoom_hist.pop().items():        # the window before the last zoom (None = header value)
            if _v is None:
                remembered.pop(_k, None)
            else:
                remembered[_k] = _v
        set_zoom_version(lambda v: v + 1)

    zoom_reset = mo.ui.button(label="🔍  Reset zoom", kind="neutral", tooltip=WIDGET_INFO["zoom_reset"], on_click=_reset)
    zoom_back = mo.ui.button(label="↩  Previous view", kind="neutral", tooltip=WIDGET_INFO["zoom_back"], on_click=_back)
    return zoom_back, zoom_reset


@app.cell
def _(BatchJob, DEFAULT_DIR, FORMATS, Path, SEGY_FORMATS, WIDGET_INFO, ctx, jobs, last, mo, os, path_input, resolve_output,
       save_outputs, set_job_version, set_save_msg, tip):
    # ---- Save: writes the figures (and optionally tables) currently shown ----
    save_dir = mo.ui.text(value=DEFAULT_DIR, label="Folder for figures, tables and single-shot files " + tip(WIDGET_INFO["save_dir"]),
                          full_width=True)
    save_fmt = mo.ui.dropdown(options=list(FORMATS), value="png", label="Format " + tip(WIDGET_INFO["save_fmt"]))
    save_dpi = mo.ui.number(start=50, stop=600, step=50, value=150, label="PNG dpi " + tip(WIDGET_INFO["save_dpi"]))
    save_tables = mo.ui.checkbox(value=False, label="also tables (CSV) and text " + tip(WIDGET_INFO["save_tables"]))

    def _save(_):
        if not last.get("outs"):
            set_save_msg("Nothing to save yet - run a function first.")
            return
        try:
            _paths = save_outputs(
                last["outs"], save_dir.value, last["stem"],
                fmt=save_fmt.value, dpi=int(save_dpi.value or 150), tables=bool(save_tables.value),
                tag=last.get("tag", ""),
            )
            if not _paths:
                set_save_msg("No figures on screen to save" + ("" if save_tables.value else " (tick 'also tables' for the tables)") + ".")
            else:
                set_save_msg(f"Saved {len(_paths)} file(s):  \n" + "  \n".join(f"`{_p}`" for _p in _paths))
        except Exception as _e:
            set_save_msg(f"**Save failed - {type(_e).__name__}:** {_e}")

    save_button = mo.ui.button(label="💾  Save figures", kind="success", tooltip=WIDGET_INFO["save_button"], on_click=_save)

    # the same flow on EVERY shot of the file (or an FFID range), written as one SEG-Y while it runs
    _lo, _hi = (ctx.ffids[0], ctx.ffids[-1]) if ctx is not None else (0, 0)
    batch_from = mo.ui.number(start=_lo, stop=_hi, step=1, value=_lo, label="From FFID " + tip(WIDGET_INFO["batch_from"]), debounce=True)
    batch_to = mo.ui.number(start=_lo, stop=_hi, step=1, value=_hi, label="To FFID " + tip(WIDGET_INFO["batch_to"]), debounce=True)
    batch_fmt = mo.ui.dropdown(options=list(SEGY_FORMATS), value="ibm", label="Sample format " + tip(WIDGET_INFO["batch_fmt"]))
    # where the processed file goes: a full path like the input path (default: next to the input file)
    _src = Path(str(path_input.value).strip()).expanduser()
    _default_out = str(_src.parent / f"{_src.stem}_processed.sgy")
    batch_out = mo.ui.text(value=_default_out, label="Output SEG-Y file " + tip(WIDGET_INFO["batch_out"]),
                            full_width=True)
    import json as _json
    from functions.cdp_flow import (processed_file as _processed_file, processed_path as _processed_path,
                                    processed_ready as _processed_ready, split_flow as _split_flow)
    from functions.pipeline import get_step as _get_step

    _cdp_keys = ("nmo_correction_step", "cdp_stack_step")
    _cat = lambda _k: _get_step(_k).category
    _label = lambda _k: _get_step(_k).label

    def _mtime(_f):
        try:
            return os.path.getmtime(_f)
        except OSError:
            return None

    _abbr = {"correct_dead": "cd", "geometric_spreading": "gs", "spiking_decon": "decon", "bandpass_filter": "bp",
             "agc_gain": "agc", "top_mute": "mute"}

    # ▶ Run flow at the end of the Flow runs the whole flow on the whole data in one pass (▶ beside a function only
    # previews the selected shot). With NMO Correction / CDP Stack the shot functions' result is kept (output/flows):
    # the CDP steps need it CDP-sorted, and a later run with the same shot functions reuses it (BatchJob).

    def _target(_steps):
        # what a whole-data run of these steps does: the trailing QC / Display functions only mark or show, they
        # leave nothing on the whole data (QC flags are used by the Correct Dead Traces after them)
        _t = list(_steps)
        while _t and _t[-1]["step"] not in _cdp_keys and _cat(_t[-1]["step"]) in ("QC", "Display"):
            _t.pop()
        return _t

    def _out_for(_t, _final):
        # the flow's product: the output box (untouched default = named after the product); a CDP step that is not
        # the last function writes its own product next to it; shot steps that are not the last: kept internally
        _kind = {"nmo_correction_step": "nmo_gathers", "cdp_stack_step": "stack"}.get(_t[-1]["step"])
        if not _final and not _kind:
            return None
        _box = batch_out.value if _final else ""
        if _kind and str(_box).strip() in ("", _default_out):
            _box = str(Path(str(batch_out.value).strip() or _default_out).expanduser().parent / f"{_src.stem}_{_kind}.sgy")
        _a, _b = int(batch_from.value or _lo), int(batch_to.value or _hi)
        _tag = "_".join(_abbr.get(_s["step"], _s["step"]) for _s in _t if _s["step"] not in _cdp_keys)
        return str(resolve_output(_box, str(path_input.value), f"{_src.stem}_{_tag or _kind}_FFID{_a}-{_b}.sgy"))

    def _key(_t, _out):
        return _json.dumps([str(path_input.value), _t, int(batch_from.value or _lo), int(batch_to.value or _hi),
                           batch_fmt.value, _out], sort_keys=True, default=str)

    def _free_name(_out):
        # <name>_2.sgy, <name>_3.sgy ... : the first that does not exist yet
        _p = Path(_out)
        _n = 2
        while (_p.parent / f"{_p.stem}_{_n}{_p.suffix}").exists():
            _n += 1
        return str(_p.parent / f"{_p.stem}_{_n}{_p.suffix}")

    def _launch(_path, _t, _out, _title, overwrite):
        _cur = jobs.get("current")
        if _cur is not None and _cur.running:          # the newest ▶ Run flow wins
            _cur.cancel()
            _cur.join(10)
        try:
            _job = BatchJob(_path, _t, _out, fmt=batch_fmt.value, ffid_from=int(batch_from.value or _lo),
                            ffid_to=int(batch_to.value or _hi), overwrite=overwrite)
        except Exception as _e:
            return f"🌐 **Not run on the whole data - {type(_e).__name__}:** {_e}"
        _job.title = _title
        _job.key = _key(_t, _out) if _out else ""
        jobs["current"] = _job.start()
        jobs["current_kind"] = "save"
        jobs["announced"] = None
        set_job_version(lambda v: v + 1)
        return (f"🌐 Running the flow **{_job.title}** on the whole data (FFID {_job.ffid_from} – {_job.ffid_to}) -> "
                f"`{_out or 'output/flows'}` - progress and Cancel in the card at the bottom right.")

    def start_whole(_path, _steps, _k):
        """Run the flow up to function _k (all of it when _k is its length) on the whole data, in the background.
        Returns a line for the panel (None when there is nothing to say)."""
        if not _steps:
            return None
        _k = len(_steps) if _k is None or _k < 0 else min(int(_k), len(_steps))
        _final = _k == len(_steps)
        _t = _target(_steps[:_k])
        _name = f"{_k}. {_label(_steps[_k - 1]['step'])}" if _k else ""
        if not _t:
            return (f"🌐 **{_name}** marks or shows only - nothing to run on the whole data for it (it is applied there "
                    "with the functions that change the data).")
        _bad = _split_flow(_t)[2]
        if _bad:
            return "🌐 Not run on the whole data: " + "; ".join(_bad)
        try:
            _out = _out_for(_t, _final)
        except Exception as _e:
            return f"🌐 **Not run on the whole data - {type(_e).__name__}:** {_e}"
        _cur = jobs.get("current")
        if _cur is not None and _cur.running:
            # a run that already covers this (same first functions, same settings) goes on
            if _cur.path == str(_path).strip() and _cur.steps[:len(_t)] == _t and (_out is None or _cur.out_path == _out):
                return f"🌐 Already running on the whole data ({_cur.title}) - see the card at the bottom right."
        _a, _b = int(batch_from.value or _lo), int(batch_to.value or _hi)
        _done = jobs.setdefault("products", {})
        if _out is not None and _mtime(_out) is not None and (
                _done.get(_key(_t, _out)) == _mtime(_out)
                or (not _split_flow(_t)[1] and _processed_file(_processed_path(str(_path), _t, _a, _b)) == _out)):
            return f"🌐 Up to **{_name}** was already run on the whole data with these settings: `{_out}`"
        _title = ("whole flow" if _final else f"up to {_name}") + ("" if _out else " (kept for the next functions)")
        if _out is not None and os.path.exists(_out):
            # the file is there (another flow, other settings): the user decides - overwrite it or a new name
            jobs["pending"] = {"path": str(_path), "steps": _t, "out": _out, "title": _title, "new": _free_name(_out)}
            return (f"🌐 The output file already exists: `{_out}`. Choose **Overwrite** or **Save as a new name** in the "
                    "Flow card (under ▶ Run flow) - or type another name under 🌐 Whole data and press ▶ Run flow again.")
        jobs.pop("pending", None)
        return _launch(str(_path), _t, _out, _title, overwrite=False)

    def _resolve_pending(_choice):
        _p = jobs.pop("pending", None)
        if _p is not None and _choice != "cancel":
            _msg = _launch(_p["path"], _p["steps"], _p["out"] if _choice == "overwrite" else _p["new"], _p["title"],
                           overwrite=_choice == "overwrite")
            set_save_msg(_msg.replace("🌐 ", ""))
        set_job_version(lambda v: v + 1)

    exists_overwrite = mo.ui.button(label="⚠  Overwrite", kind="danger", tooltip=WIDGET_INFO["exists_overwrite"],
                                    on_click=lambda _: _resolve_pending("overwrite"))
    exists_rename = mo.ui.button(label="💾  Save as a new name", kind="success", tooltip=WIDGET_INFO["exists_rename"],
                                 on_click=lambda _: _resolve_pending("rename"))
    exists_cancel = mo.ui.button(label="✕  Cancel", tooltip=WIDGET_INFO["exists_cancel"],
                                 on_click=lambda _: _resolve_pending("cancel"))

    def whole_status(_path, _steps, _k):
        """'⏳' running, '✓' done on the whole data with these settings, '' not (yet) - for function _k of the flow."""
        _t = _target(_steps[:_k])
        if not _t:
            return ""
        _cur = jobs.get("current")
        if _cur is not None and _cur.running and _cur.steps[:len(_t)] == _t:
            return "⏳"
        try:
            _out = _out_for(_t, _k == len(_steps))
            if _out is not None and _t[-1]["step"] in _cdp_keys:
                _m = _mtime(_out)
                return "✓" if _m is not None and jobs.get("products", {}).get(_key(_t, _out)) == _m else ""
            _proc = _processed_path(str(_path), _t, int(batch_from.value or _lo), int(batch_to.value or _hi))
            return "✓" if _processed_ready(_proc) else ""
        except Exception:
            return ""

    def _cancel_batch(_):
        _cur = jobs.get("current")
        if _cur is not None and _cur.running:
            _cur.cancel()

    batch_cancel = mo.ui.button(label="✖  Cancel", kind="danger", tooltip=WIDGET_INFO["batch_cancel"], on_click=_cancel_batch)
    return (batch_cancel, batch_fmt, batch_from, batch_out, batch_to, exists_cancel, exists_overwrite, exists_rename,
            start_whole, whole_status, save_button, save_dir, save_dpi, save_fmt, save_tables)


@app.cell
def _(mo):
    # the timer that re-runs the progress card; it must live in its OWN cell (a cell is not re-run by its own elements)
    job_refresh = mo.ui.refresh(options=["2s"], default_interval="2s")
    return (job_refresh,)


@app.cell
def _(WIDGET_INFO, batch_cancel, job_refresh, job_version, jobs, mo, os, set_job_version):
    # ---- whole-data job: a card fixed at the bottom right of the window (polled every 2 s while the job runs) ----
    job_version()
    _job = jobs.get("current")
    if _job is None:
        job_bar = None                       # nothing to show: no empty output that would still take up room
    else:
        _s = _job.snapshot()

        def _fmt_t(sec):
            sec = int(sec or 0)
            return f"{sec // 60}:{sec % 60:02d}"

        def _dismiss(_):
            jobs.pop("current", None)
            set_job_version(lambda v: v + 1)

        _running = _s["status"] == "running"
        if not _running and jobs.get("announced") is not _job:
            # finished: remember its product (a ▶ with the same settings then does not make it again) and redraw the
            # page once, so the Flow card's 🌐 marks show what is done on the whole data
            jobs["announced"] = _job
            if _s["status"] == "done" and getattr(_job, "key", "") and _s["out_path"]:
                try:
                    jobs.setdefault("products", {})[_job.key] = os.path.getmtime(_s["out_path"])
                except OSError:
                    pass
            set_job_version(lambda v: v + 1)
        _pct = 100.0 * _s["done"] / _s["total"] if _s["total"] else 0.0
        _name = _s["out_path"].rsplit("/", 1)[-1] if _s["out_path"] else "kept in output/flows for the next functions"
        _look = {"running": ("#2a78d6", "⏳"), "done": ("#2b9a66", "✅"), "cancelled": ("#c98a1b", "⚠️"), "error": ("#e34948", "❌")}
        _color, _icon = _look.get(_s["status"], ("#6b7280", "•"))
        _r = _s["result"]
        if _s["status"] == "running" and _s.get("unit", "shots") != "shots":     # the CDP part of the flow
            _detail = (f"{_s['phase']}: {_s['done']:,} / {_s['total']:,} {_s['unit']} · elapsed {_fmt_t(_s['elapsed'])}")
        elif _s["status"] == "running":
            _detail = (f"{_s['done']:,} / {_s['total']:,} shots · {_s['traces_in']:,} traces · elapsed {_fmt_t(_s['elapsed'])}"
                       + (f" · about {_fmt_t(_s['eta'])} left" if _s["eta"] else ""))
        elif _s["status"] == "done" and _r is not None and _s.get("product"):
            _what = {"nmo_correction_step": "NMO-corrected CDP gathers", "cdp_stack_step": "stack"}.get(_s["product"], "")
            _detail = (f"{_what}: {_s.get('cdps', 0):,} CDPs · {_r.traces_out:,} traces written · "
                       f"{_r.bytes_out / 1e9:.2f} GB · {_fmt_t(_r.seconds)}"
                       + (f"<br/>report: <code>{_r.report_path}</code>" if _r.report_path else ""))
        elif _s["status"] == "done" and _r is not None:
            _detail = (f"{_r.shots:,} shots · traces {_r.traces_in:,} → {_r.traces_out:,} "
                       f"({_r.traces_in - _r.traces_out:,} removed) · {_r.bytes_out / 1e9:.2f} GB · {_fmt_t(_r.seconds)}"
                       + (f"<br/>report: <code>{_r.report_path}</code>" if _r.report_path else ""))
        else:
            _detail = _s["message"]
        _what = getattr(_job, "title", "") or "the flow"
        _title = {"running": f"Running {_what} on the whole data", "done": f"Whole data: {_what} done",
                  "cancelled": "Cancelled - no file was written", "error": "Whole-data run failed"}.get(_s["status"], "")
        if _s["status"] == "done" and _s["message"] != "finished":
            _detail = _s["message"]
        elif _running and getattr(_job, "resumed", ""):
            _detail += f"<br/>started from {_job.resumed}"
        _card = mo.Html(
            f'<div style="border:1px solid {_color};border-left:6px solid {_color};border-radius:10px;padding:10px 16px">'
            f'<div style="font-weight:700">{_icon} {_title} <span style="opacity:.65;font-weight:400">— {_name}</span></div>'
            f'<progress value="{_pct:.1f}" max="100" style="width:100%;height:10px;margin:6px 0"></progress>'
            f'<div style="font-size:.9rem;overflow-wrap:anywhere">{_detail}</div></div>'
        )
        _dismiss_btn = mo.ui.button(label="✕ Dismiss", tooltip=WIDGET_INFO["job_dismiss"], on_click=_dismiss)
        # while running: Cancel + the timer (shown = ticking); afterwards: Dismiss, and the timer stops
        _inner = mo.vstack([_card] + ([mo.hstack([batch_cancel, job_refresh], justify="start", gap=1)] if _running else [_dismiss_btn]),
                           gap=0.4)
        job_bar = mo.Html(f'<div class="qc-job">{_inner}</div>')
    job_bar
    return (job_bar,)


@app.cell
def _(FUNCTION_INFO, Path, batch_fmt, batch_from, batch_out, batch_to, ctx, ctx_error, job_version, jobs,
       exists_cancel, exists_overwrite, exists_rename, fk_editor,
       whole_status,
       grid_msg, grid_reset, survey_grid,
       estimate_batch, ffid_slider, flow_add, flow_toggle, form, func, last, live, live_display, load_button,
       mo, not_loaded, path_input, pipe_cards, pipe_down, pipe_remove, pipe_run, run_all, pipe_slots, pipe_steps, pipe_up,
       pipe_view, right, right_top, save_button, save_dir, save_dpi,
       save_fmt, save_msg, save_tables, tip, tool_page, topbar, zoom_back, zoom_reset):
    # ---- page layout: header, category toolbar, parameter cards left, figure + results right ----
    _card = {
        "border": "1px solid rgba(128,128,128,.28)", "border-radius": "12px", "padding": "14px 16px",
        "background": "rgba(128,128,128,.05)", "min-width": "0", "box-sizing": "border-box",
    }
    _muted = 'style="opacity:.7"'

    _data = [mo.md("#### 📁 Data"), path_input]
    if ctx_error:
        _data.append(mo.callout(mo.md(ctx_error), kind="danger"))
    elif not_loaded:
        _data.append(mo.md(f"<span {_muted}>Not loaded yet - open <b>Load Data</b> (Flow card, ＋ Add a function) "
                           "and press ▶ Load data.</span>"))
    elif ctx is not None:
        _data.append(mo.md(
            f"<span {_muted}>{len(ctx.index):,} shots · FFID {ctx.ffids[0]} – {ctx.ffids[-1]} · "
            f"{ctx.sgy.ntraces:,} traces · {ctx.record_ms:g} ms records</span>"
        ))
        if survey_grid is not None:
            _data.append(mo.accordion({"🧭  Survey grid (IL / XL corner points)": mo.vstack([
                mo.md(f"<span {_muted}>The data's inline / crossline grid, used by every function that needs IL / XL "
                      "(Acquisition Geometry / Fold, NMO Correction, CDP Stack, the CDP lists and 🛠 Tools). Filled in "
                      "from the table saved for this file or the IL / XL of the headers; change any value (or type "
                      "your own corners, at least 3) and it is saved for this file.</span>"),
                survey_grid, grid_reset, mo.md(grid_msg).style({"font-size": ".85rem"}),
            ], gap=0.5)}))

    # the Flow: a card of its own, right after Data, shown once data is loaded - regardless of which function above
    # is selected, so a flow can be built (or changed) while looking at any other tool.  Every function in it has
    # ▲ ▼ − and its own parameters, then a ＋ at the end to add the next one (clicking it drops the function list
    # down right there; picking one adds it and the ＋ moves to the new end).
    _flow_card = None
    if pipe_slots is not None:
        _flow = [mo.md("#### ⚙ Flow"), mo.md(f"<span {_muted}>Functions chained, run top to bottom: shot steps on the "
                                             "selected shot, then NMO Correction / CDP Stack on CDP gathers of their "
                                             "result.</span>"),
                 mo.accordion({"ⓘ  How the flow works": mo.md(FUNCTION_INFO.get("flow", ""))})]
        if not pipe_slots.value and not flow_add:
            _flow.append(mo.md(f"<span {_muted}>Empty - click ＋ below to add a function.</span>"))
        job_version()                     # (drawn again when a whole-data run starts / ends: the 🌐 marks)
        _all = last.get("flow_all") or []
        _same = len(_all) == len(pipe_slots.value)
        _marks = {"✓": ('<span title="Run on the whole data with these settings - the next functions start from '
                        'this result" style="color:#2b9a66;font-size:.85rem">🌐✓</span>'),
                  "⏳": '<span title="Running on the whole data now" style="font-size:.85rem">🌐⏳</span>'}
        for _i, _label in enumerate(pipe_slots.value):
            _desc = pipe_steps[_label].description if _label in pipe_steps else ""
            _st = whole_status(path_input.value, _all, _i + 1) if _same and ctx is not None else ""
            _flow.append(mo.hstack(
                [mo.md(f"**{_i + 1}. {_label}** {tip(_desc)} {_marks.get(_st, '')}"),
                 mo.hstack([pipe_run[_i], pipe_up[_i], pipe_down[_i], pipe_remove[_i]], justify="end", gap=0.25)],
                justify="space-between", align="center", gap=0.5))
            if str(_i) in pipe_cards:
                _flow.append(pipe_cards[str(_i)])
        _flow.append(mo.hstack([flow_toggle] + ([run_all] if run_all is not None and pipe_slots.value else []),
                               justify="start", gap=0.6))
        _pending = jobs.get("pending")
        if _pending:
            # ▶ Run flow found its output file already there: overwrite it, or write <name>_2.sgy instead
            _flow.append(mo.vstack([
                mo.md(f"⚠ **The output file already exists:** `{_pending['out']}`  \n"
                      f"Overwrite it, or save as `{Path(_pending['new']).name}`? (Or type another name under 🌐 Whole "
                      "data and press ▶ Run flow again.)").style({"overflow-wrap": "anywhere"}),
                mo.hstack([exists_overwrite, exists_rename, exists_cancel], justify="start", gap=0.6, wrap=True),
            ], gap=0.5).style({"border": "1px solid #c98a1b", "border-left": "6px solid #c98a1b", "border-radius": "10px",
                               "padding": "10px 12px", "background": "rgba(201,138,27,.08)"}))
        if pipe_slots.value and ctx is not None:
            # where / on which shots the whole-data runs of the ▶ buttons write
            try:
                _est = estimate_batch(path_input.value, int(batch_from.value or 0), int(batch_to.value or 0))
                _est_txt = f"{_est['shots']:,} shots · {_est['traces']:,} traces · about {_est['gigabytes']:.1f} GB per run"
            except Exception as _e:
                _est_txt = f"<b>{_e}</b>"
            _flow.append(mo.accordion({"🌐  Whole data: output file & FFID range": mo.vstack([
                mo.md(f"<span {_muted}>▶ beside a function previews the selected shot only. <b>▶ Run flow</b> runs "
                      "the whole flow on every shot of the range in one pass - each shot read once, all functions in "
                      "memory, the result written once - in the background (progress and Cancel bottom right), and "
                      "writes the flow's product to the file below: the processed shots, or the NMO-corrected gathers "
                      "/ the stack when the flow ends with NMO Correction / CDP Stack. 🌐✓ = done on the whole data "
                      "with these settings.</span>"),
                batch_out,
                mo.hstack([batch_from, batch_to, batch_fmt], justify="start", align="end", gap=1.2, wrap=True),
                mo.md(f"<span {_muted}>{_est_txt}</span>"),
            ], gap=0.5)}))
        if flow_add is not None:
            _flow.append(flow_add)
        if pipe_view is not None:
            _flow.append(pipe_view)
        _flow_card = mo.vstack(_flow, gap=0.8).style(_card)

    # the currently selected function's own parameters (Flow itself needs no separate header here - the Flow
    # card above already introduces it - just its Shot / Display controls for the stage on screen)
    _params = [] if func.kind == "pipeline" else [mo.md(f"#### {func.label}"), mo.md(f"<span {_muted}>{func.description}</span>")]
    if func.kind == "loader":
        _params.append(load_button)
    if ffid_slider is not None:
        _k = ctx.index.position(int(ffid_slider.value))
        _params += [
            mo.md("**Shot**"),
            ffid_slider,
            mo.md(f"<span {_muted}>{int(ctx.index.count[_k])} traces · "
                  f"file traces {int(ctx.index.first[_k]) + 1}–{int(ctx.index.first[_k] + ctx.index.count[_k])}</span>"),
        ]
    if live is not None:
        _params.append(live)
    if live_display is not None:
        # collapsible, tinted so it stands out from the tool's own parameters (styles: custom.css .qc-display)
        _params.append(mo.Html(f'<div class="qc-display">{mo.accordion({"🖼  Display options": live_display})}</div>'))
    if form is not None:
        _params.append(form)
    elif live is None and live_display is None and ffid_slider is None and func.kind not in ("loader", "pipeline"):
        _params.append(mo.md("_This function has no parameters — it runs when selected._"))

    _left_cards = [mo.vstack(_data, gap=0.5).style(_card)]
    if _flow_card is not None:
        _left_cards.append(_flow_card)
    if _params:
        _left_cards.append(mo.vstack(_params, gap=0.8).style(_card))
    left = mo.vstack(_left_cards, gap=1).style({"min-width": "0"})

    # right column: [zoom bar + gather figure] first, so the figure is on the first screen; then the results, then Save at the bottom
    _parts = []
    if right_top is not None:
        _parts.append(
            mo.vstack(
                [
                    mo.hstack(
                        [zoom_reset, zoom_back,
                         mo.md(f"<span {_muted}>Drag a box on the gather to zoom in (Shift + drag: lasso)</span>")],
                        justify="start", align="center", gap=0.8, wrap=True,
                    ),
                    right_top,
                ],
                gap=0.6,
            ).style(_card)
        )
    _save_card = None                       # 💾 Save figures: shown at the BOTTOM, after the results (the whole data is
    if last.get("outs"):                    # written by ▶ Run flow of the Flow card)
        _save_card = (
            mo.vstack(
                [
                    mo.md("#### 💾 Save figures"),
                    mo.hstack(
                        [save_button, save_fmt, save_dpi, save_tables],
                        justify="start", align="end", gap=1.2, wrap=True,
                    ),
                    save_dir,
                    mo.md(save_msg()).style({"overflow-wrap": "anywhere"}),
                ],
                gap=0.5,
            ).style(_card)
        )
    if last.get("fk_show"):
        # the F-K Filter's polygon editor with the shot gather filtered by the current reject zone beside it
        _live = last.get("fk_live")
        _half = {"flex": "1 1 0", "min-width": "0"}
        _parts.append(mo.hstack(
            [mo.vstack([mo.md("#### F-K domain - click to add points, drag them to change the shape"), fk_editor],
                       gap=0.3).style(_half)]
            + ([mo.vstack([mo.md(f"#### {_live.title}"),
                           mo.image(_live.content, style={"width": "100%", "height": "auto"})], gap=0.3).style(_half)]
               if _live is not None else []),
            gap=1, align="start", widths="equal").style(_card))
    _parts.append(mo.vstack([right], gap=0).style({**_card, "background": "transparent", "border": "none", "padding": "0"}))
    if _save_card is not None:
        _parts.append(_save_card)
    _right = mo.vstack(_parts, gap=1).style({"min-width": "0", "max-width": "100%"})

    # top bar (fixed) + two panels that scroll on their own (styles: custom.css).  The panels sit in a CSS grid,
    # not a row of flex items: the right one may shrink below the width of a wide table, so the page never grows
    # wider than the window.
    # (a tab opened from the 🛠 Tools menu shows only that tool: tool_page, built in the TOOL WINDOW cells)
    tool_page if tool_page is not None else mo.Html(
        '<div class="qc-root">'
        f"{topbar}"
        f'<div class="qc-panels"><div class="qc-scroll">{left}</div><div class="qc-scroll">{_right}</div></div>'
        "</div>"
    )
    return


@app.cell
def _(tool_window):
    # TOOL NOTEBOOK: a tab opened for a notebook tool (kind="notebook", functions/notebook_tools.py) loads that
    # notebook's marimo app; the next cell runs it inside this page, as it is
    tool_notebook_app = None
    if tool_window is not None and tool_window.kind == "notebook":
        import importlib.util as _ilu
        import sys as _sys
        from pathlib import Path as _P

        _nb = _P(tool_window.run())
        if str(_nb.parent) not in _sys.path:              # its own helpers (e.g. sc_scaling_pipeline.py) sit next to it
            _sys.path.insert(0, str(_nb.parent))
        _spec = _ilu.spec_from_file_location(f"_tool_notebook_{tool_window.key}", _nb)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        tool_notebook_app = _mod.app
    return (tool_notebook_app,)


@app.cell
async def _(mo, os, tool_notebook_app, tool_window_file):
    # the embedded notebook: all its own widgets, buttons and state work as in the notebook itself. The main window's
    # file (?segy=) goes in as the notebook's default_segy_path (the value its path box starts with), when it has one
    _defs = {"default_segy_path": tool_window_file} if tool_window_file else None
    tool_notebook = None
    if tool_notebook_app is not None:
        # the notebook appears only once all its cells have run - e.g. Shot Geometry QC reads every trace of the file
        # for its maps, minutes on a large file that is not in memory - so say so instead of showing an empty page
        _size = ""
        try:
            _size = f" ({os.path.getsize(tool_window_file) / 1e9:.1f} GB)" if tool_window_file else ""
        except OSError:
            pass
        with mo.status.spinner(title="Starting the notebook ...",
                               subtitle=f"it first reads the whole file{_size} for its maps - on a large file this "
                                        "takes a few minutes; the notebook appears here when it is done"):
            tool_notebook = await tool_notebook_app.embed(defs=_defs)
    mo.vstack([tool_notebook.output]).style({"padding": "0 10px 16px"}) if tool_notebook is not None else None
    return


@app.cell
def _(mo, tool_card, tool_plots, tool_stem):
    # TOOL PLOT 1: only in a Tools-window tab (tool_plots is empty otherwise -> no output)
    mo.vstack([tool_card(tool_plots[0], tool_stem)]).style({"padding": "0 10px"}) if len(tool_plots) > 0 else None
    return


@app.cell
def _(mo, tool_card, tool_plots, tool_stem):
    # TOOL PLOT 2: only in a Tools-window tab (tool_plots is empty otherwise -> no output)
    mo.vstack([tool_card(tool_plots[1], tool_stem)]).style({"padding": "0 10px"}) if len(tool_plots) > 1 else None
    return


@app.cell
def _(mo, tool_card, tool_plots, tool_stem):
    # TOOL PLOT 3: only in a Tools-window tab (tool_plots is empty otherwise -> no output)
    mo.vstack([tool_card(tool_plots[2], tool_stem)]).style({"padding": "0 10px"}) if len(tool_plots) > 2 else None
    return


@app.cell
def _(mo, tool_card, tool_plots, tool_stem):
    # TOOL PLOT 4: only in a Tools-window tab (tool_plots is empty otherwise -> no output)
    mo.vstack([tool_card(tool_plots[3], tool_stem)]).style({"padding": "0 10px"}) if len(tool_plots) > 3 else None
    return


@app.cell
def _(mo, tool_card, tool_stem, tool_tail):
    # TOOL TABLES: the long tables of a Tools-window run (e.g. every trace), last - after the plots
    mo.vstack([tool_card(_o, tool_stem) for _o in tool_tail], gap=1).style({"padding": "0 10px 16px"}) if tool_tail else None
    return


if __name__ == "__main__":
    app.run()
