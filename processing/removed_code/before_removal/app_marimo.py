import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full", app_title="Processing Tool", css_file="custom.css")


@app.cell
def _():
    import os
    import sys
    import html
    from types import SimpleNamespace
    from pathlib import Path

    import marimo as mo

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from functions import segy_io
    from functions.param_help import PARAM_INFO, WIDGET_INFO, info_text
    from functions import (DEFAULT_DIR, FORMATS, SEGY_FORMATS, BatchJob, default_for, estimate_batch, file_context,
                              get_categories, get_functions, get_steps, plot_tag, resolve_output, save_data, save_outputs)

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
    # the whole-data job (a BatchJob runs in its own thread; the GUI only polls it)
    jobs = {}
    # the flow the user built in the Pipeline tool: step per slot (its length is the number of steps),
    # parameters per (slot, step)
    pipe_mem = {"slots": [], "params": {}, "view": None}
    # the file the Load Data tool has read: its normalised path, the outputs it produced, the Load-button count
    # already handled, and the error of the last attempt
    loaded = {"path": None, "outs": None, "seen": 0, "error": None}
    return (BatchJob, DEFAULT_DIR, DEFAULT_FILE, FORMATS, Path, SEGY_FORMATS, default_for, estimate_batch, file_context,
            functions, get_categories, get_steps, html, info_text, jobs, last, loaded, mo, pipe_mem, plot_tag, remembered,
            resolve_output, save_data, save_outputs, segy_io, SimpleNamespace, tip, zoom_hist, PARAM_INFO, WIDGET_INFO)


@app.cell
def _(functions, mo):
    active, set_active = mo.state(next(iter(functions)), allow_self_loops=True)
    save_msg, set_save_msg = mo.state("")
    zoom_version, set_zoom_version = mo.state(0)     # bumped to rebuild the range widgets after a zoom
    job_version, set_job_version = mo.state(0, allow_self_loops=True)   # bumped when a whole-data job starts / is dismissed
    pipe_version, set_pipe_version = mo.state(0, allow_self_loops=True)  # bumped when a pipe step is added / removed
    load_version, set_load_version = mo.state(0)                         # bumped when Load Data has finished
    open_cat, set_open_cat = mo.state(None, allow_self_loops=True)       # class of functions whose list is dropped down
    flow_ready, set_flow_ready = mo.state(None)                          # the flow the user confirmed as ready (its repr)
    return (active, flow_ready, job_version, load_version, open_cat, pipe_version, save_msg, set_active, set_flow_ready,
            set_job_version, set_load_version, set_open_cat, set_pipe_version, set_save_msg, set_zoom_version,
            zoom_version)


@app.cell
def _(WIDGET_INFO, active, functions, get_categories, get_steps, html, mo, open_cat, pipe_mem, set_active, set_open_cat,
       set_pipe_version):
    # ---- TOP: a title strip, and below it the function panel: one button per CLASS of functions.  A class button
    # drops its functions down; a click on a function opens it and closes the list again.  (Styles: custom.css)
    # class_buttons / func_list are kept as globals on purpose: marimo drops UI elements nothing refers to any more,
    # and a click on such a button would do nothing.
    _look = {"Data": "🗂", "Display": "🖼", "QC": "🔎", "Processing": "⚙️", "Pipeline": "🔗"}
    _all = get_categories()
    _cats = [(_n, _fs) for _n, _fs in _all if _n != "Flow"]               # the flow itself is opened by the + at the path
    _step_label = {_s.key: _s.label for _s in get_steps()}
    _cat_of = {_f.key: _n for _n, _fs in _all for _f in _fs}
    _now, _open = active(), open_cat()

    def _toggle(_c):
        set_open_cat(lambda _cur: None if _cur == _c else _c)

    def _pick(_k):
        _f = functions[_k]
        if _f.step_key:                                   # a processing function: add it to the flow, and show the flow
            if len(pipe_mem["slots"]) < 12:
                pipe_mem["slots"].append(_step_label[_f.step_key])
                pipe_mem["view"] = None                   # a changed flow shows its last stage again
                set_pipe_version(lambda n: n + 1)
            set_active("pipeline")
        else:
            set_active(_k)
        set_open_cat(None)

    def _open_flow(_):
        set_active("pipeline")
        set_open_cat(None)

    # the + at the file path: build a flow (add functions one after another, see the result after any step)
    add_flow_button = mo.ui.button(label="＋", kind="success", tooltip=WIDGET_INFO["add_flow"], on_click=_open_flow)

    class_buttons = [
        mo.ui.button(
            label=f"{_look.get(_n, '▫️')} {_n} {'▴' if _n == _open else '▾'}",
            kind="success" if _n == _cat_of.get(_now) else "neutral",
            tooltip=f"{len(_fs)} function(s): " + ", ".join(_f.label for _f in _fs),
            on_click=lambda _, _c=_n: _toggle(_c),
        )
        for _n, _fs in _cats
    ]
    func_list, _drop = None, ""
    if _open:
        func_list = mo.vstack(
            [
                mo.hstack(
                    [
                        mo.ui.button(label=_f.label, kind="success" if _f.key == _now else "neutral", full_width=True,
                                     tooltip=_f.description + ("\n\nClick: adds this function to the flow." if _f.step_key else ""),
                                 on_click=lambda _, _k=_f.key: _pick(_k)),
                        mo.md(f'<span style="opacity:.7;font-size:.85rem">{html.escape(_f.description)}</span>'),
                    ],
                    justify="start", align="center", gap=0.8, widths=[1, 3],
                )
                for _n, _fs in _cats if _n == _open for _f in _fs
            ],
            gap=0.45,
        )
        _drop = f'<div class="qc-pop">{func_list}</div>'
    _bar = mo.hstack(class_buttons, justify="start", gap=0.4, wrap=True)
    topbar = mo.Html(
        '<div class="qc-top">'
        '<div class="qc-strip"><span class="qc-title">Processing Tool</span></div>'
        f'<div class="qc-funcs">{_bar}'
        f'<span class="qc-active">{html.escape(_cat_of.get(_now, ""))} › <b>{html.escape(functions[_now].label)}</b></span></div>'
        f"{_drop}</div>"
    )
    return add_flow_button, class_buttons, func_list, topbar


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
def _(active, functions):
    func = functions[active()]
    return (func,)


@app.cell
def _(html, info_text, mo, tip):
    # ---- widget factory: one widget per Param, grouped under headings -------
    def panel(params, value_of, live, on_change=None, skip_groups=()):
        _elements, _lines, _group = {}, [], None
        for _p in params:
            if _p.kind == "ffid":
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
                _w = mo.ui.dropdown(options=_p.choices, value=_val, label=_label)
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
def _(get_steps):
    pipe_steps = {_s.label: _s for _s in get_steps()}
    return (pipe_steps,)


@app.cell
def _(SimpleNamespace, WIDGET_INFO, func, mo, pipe_mem, pipe_steps, pipe_version, set_pipe_version):
    # ---- FLOW: the functions clicked in the Processing tab, top to bottom.  ▲ ▼ move one, the red − removes it. ----
    # The list lives in pipe_mem["slots"]; every change bumps pipe_version so this cell rebuilds its buttons.
    # pipe_up / pipe_down / pipe_remove are globals on purpose: marimo drops buttons nothing refers to any more.
    pipe_slots, pipe_up, pipe_down, pipe_remove = None, [], [], []
    if func.kind == "pipeline":
        pipe_version()
        pipe_mem["slots"][:] = [_l for _l in pipe_mem["slots"] if _l in pipe_steps]

        def _changed():
            pipe_mem["view"] = None                      # a changed flow shows its last stage again
            set_pipe_version(lambda n: n + 1)

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
                # each function takes its parameters along
                _swap = {_i: _j, _j: _i}
                _moved = {(_swap.get(_k[0], _k[0]), _k[1]): _v for _k, _v in pipe_mem["params"].items()}
                pipe_mem["params"].clear()
                pipe_mem["params"].update(_moved)
                _changed()

        _n = len(pipe_mem["slots"])
        pipe_slots = SimpleNamespace(value=list(pipe_mem["slots"]))       # the labels of the functions, top to bottom
        pipe_up = [mo.ui.button(label="▲", tooltip=WIDGET_INFO["step_up"], disabled=_i == 0,
                                on_click=lambda _, _i=_i: _move(_i, -1)) for _i in range(_n)]
        pipe_down = [mo.ui.button(label="▼", tooltip=WIDGET_INFO["step_down"], disabled=_i == _n - 1,
                                  on_click=lambda _, _i=_i: _move(_i, 1)) for _i in range(_n)]
        pipe_remove = [mo.ui.button(label="−", kind="danger", tooltip=WIDGET_INFO["step_remove"],
                                    on_click=lambda _, _i=_i: _remove(_i)) for _i in range(_n)]
    return pipe_down, pipe_remove, pipe_slots, pipe_up


@app.cell
def _(ctx, default_for, mo, panel, pipe_mem, pipe_slots, pipe_steps):
    # parameters of every chosen step (values are remembered per slot and step)
    pipe_params, pipe_cards = None, {}
    if pipe_slots is not None:
        _panels = {}
        for _i, _label in enumerate(pipe_slots.value):
            _st = pipe_steps.get(_label)
            if _st is None or not _st.params:
                continue

            def _value(p, _i=_i, _k=_st.key):
                return pipe_mem["params"].get((_i, _k), {}).get(p.key, default_for(p, ctx, 0))

            def _changed(v, _i=_i, _k=_st.key):
                pipe_mem["params"][(_i, _k)] = dict(v)

            _panels[str(_i)] = panel(_st.params, _value, True, on_change=_changed)
        pipe_params = mo.ui.dictionary(_panels)
        # collapsible cards so three steps do not fill the whole left panel.  The widgets must be shown
        # through the dictionary (pipe_params[key]), or edits would not reach pipe_params.value
        pipe_cards = {_k: mo.accordion({"⚙  parameters": pipe_params[_k]}) for _k in _panels}
    return pipe_cards, pipe_params


@app.cell
def _(mo, pipe_mem, pipe_slots, pipe_steps):
    # which stage of the flow to look at: the input or the result after any step
    pipe_view = None
    if pipe_slots is not None:
        _names = ["0. Input (raw)"] + [
            f"{_k}. {_label}" for _k, _label in enumerate([_l for _l in pipe_slots.value if _l in pipe_steps], 1)
        ]
        pipe_view = mo.ui.dropdown(
            options=_names, label="View stage " + tip(WIDGET_INFO["view_stage"]),
            value=pipe_mem["view"] if pipe_mem["view"] in _names else _names[-1],
            on_change=lambda v: pipe_mem.update(view=v),
        )
    return (pipe_view,)


@app.cell
def _(ctx, default_for, ffid_slider, func, panel, remembered, zoom_hist, zoom_version):
    # ---- LIVE parameters: change -> the tool re-runs at once ----------------
    # Their starting values come from the header: last time = ns * dt, last trace
    # = number of traces in the selected shot.  Trace range resets with the shot.
    zoom_version()                            # rebuild after a mouse zoom / reset (new values in `remembered`)
    _ffid = int(ffid_slider.value) if ffid_slider is not None else 0
    if remembered.get("_range_ffid") != _ffid:
        remembered.pop("trace_min", None)
        remembered.pop("trace_max", None)
        remembered["_range_ffid"] = _ffid
        zoom_hist.clear()
    live = panel(
        [_p for _p in func.params if _p.live],
        lambda p: remembered.get(p.key, default_for(p, ctx, _ffid)),
        True,
        on_change=lambda v: remembered.update(v),
        skip_groups=("Shot",),
    )
    return (live,)


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
def _(Path, coerce, ctx, default_for, ffid_slider, form, func, last, live, loaded, mo, not_loaded, path_input, pipe_params,
       pipe_slots, pipe_steps, pipe_view, plot_tag, render):
    # ---- RIGHT: run the selected function, show its outputs ----------------
    _vals = {}
    if ffid_slider is not None:
        _vals["ffid"] = ffid_slider.value
    if live is not None:
        _vals.update(live.value)

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
            mo.md("**No data loaded yet.** Choose **Data → Load Data** in the toolbar and press **▶ Load data**. "
                  "The file in the *SEG-Y file* box is read once, then every tool works on it."),
            kind="warn",
        )
    elif func.kind == "pipeline" and pipe_slots is not None and not pipe_slots.value:
        right = mo.callout(mo.md("**The flow is empty.** Click a function in the **Processing** tab (top bar) to add it."), kind="info")
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
        _extra = {}
        if func.kind == "pipeline" and pipe_slots is not None:
            # the flow: one {"step": key, "params": {...}} per filled slot, in slot order
            _extra["steps"] = [
                {"step": pipe_steps[_label].key, "params": dict(pipe_params.value.get(str(_i)) or {})}
                for _i, _label in enumerate(pipe_slots.value)
                if _label in pipe_steps
            ]
            _extra["view"] = list(pipe_view.options).index(pipe_view.value)
        try:
            with mo.status.spinner(title=f"Running {func.label} ..."):
                _outs = func.run(path_input.value, **coerce(func.params, _vals), **_extra)
            _stem = f"{Path(path_input.value).stem}_{func.key}" + (f"_FFID{_vals['ffid']}" if _vals.get("ffid") else "")
            # what the figures were made with, for the file names: parameters that differ from what the widgets started
            # with (header-derived starting values such as the record length do not count), plus the pipeline steps
            _used = coerce(func.params, _vals)
            _start = {_p.key: default_for(_p, ctx, int(_vals.get("ffid") or 0)) for _p in func.params}
            _tag = plot_tag(_used, _start, flow=_extra.get("steps"), view=_extra.get("view"),
                            default_of=lambda _p: default_for(_p, ctx, 0))
            last.update(outs=_outs, stem=_stem, tag=_tag)
            _blocks = []
            for _o in _outs:
                if _o.kind == "data":                  # processed data: only offered by the Save data button
                    continue
                if _o.kind == "image" and _o.zoom and _o.figure is not None and zoom_select is None:
                    # the gather: drag a box on it to zoom (x = trace position, y = time in ms)
                    _o.figure.set_dpi(90)          # on-screen size only; saved figures use their own dpi
                    zoom_select = mo.ui.matplotlib(_o.figure.axes[0], debounce=True)
                    right_top = mo.vstack([mo.md(f"#### {_o.title}"), zoom_select], gap=0.3)
                else:
                    _blocks.append(render(_o))
            last["zoom"] = zoom_select is not None
            right = mo.vstack(_blocks, gap=1.5).style({"min-width": "0", "max-width": "100%"})
        except Exception as _e:  # show the problem instead of a stack trace
            right = mo.callout(mo.md(f"**{type(_e).__name__}:** {_e}"), kind="danger")
    return right, right_top, zoom_select


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
def _(BatchJob, DEFAULT_DIR, FORMATS, Path, SEGY_FORMATS, WIDGET_INFO, ctx, jobs, last, mo, path_input, resolve_output,
       save_data, save_outputs, set_flow_ready, set_job_version, set_save_msg, tip):
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

    # the processed data itself (the stage on screen) as a SEG-Y file
    save_data_fmt = mo.ui.dropdown(options=list(SEGY_FORMATS), value="ibm", label="Sample format " + tip(WIDGET_INFO["save_data_fmt"]))

    def _save_data(_):
        if not any(_o.kind == "data" for _o in last.get("outs") or []):
            set_save_msg("This function has no processed data to save.")
            return
        try:
            _paths = save_data(last["outs"], save_dir.value, last["stem"], fmt=save_data_fmt.value, tag=last.get("tag", ""))
            set_save_msg("Saved SEG-Y:  \n" + "  \n".join(f"`{_p}`" for _p in _paths))
        except Exception as _e:
            set_save_msg(f"**Save failed - {type(_e).__name__}:** {_e}")

    save_data_button = mo.ui.button(label="💾  Save data (SEG-Y)", kind="success", tooltip=WIDGET_INFO["save_data_button"],
                                    on_click=_save_data)

    # the same flow on EVERY shot of the file (or an FFID range), written as one SEG-Y while it runs
    _lo, _hi = (ctx.ffids[0], ctx.ffids[-1]) if ctx is not None else (0, 0)
    batch_from = mo.ui.number(start=_lo, stop=_hi, step=1, value=_lo, label="From FFID " + tip(WIDGET_INFO["batch_from"]), debounce=True)
    batch_to = mo.ui.number(start=_lo, stop=_hi, step=1, value=_hi, label="To FFID " + tip(WIDGET_INFO["batch_to"]), debounce=True)
    batch_fmt = mo.ui.dropdown(options=list(SEGY_FORMATS), value="ibm", label="Sample format " + tip(WIDGET_INFO["batch_fmt"]))
    # where the processed file goes: a full path like the input path (default: next to the input file)
    _src = Path(str(path_input.value).strip()).expanduser()
    batch_out = mo.ui.text(value=str(_src.parent / f"{_src.stem}_processed.sgy"), label="Output SEG-Y file " + tip(WIDGET_INFO["batch_out"]),
                            full_width=True)
    batch_overwrite = mo.ui.checkbox(value=False, label="overwrite if the file exists " + tip(WIDGET_INFO["batch_overwrite"]))
    _abbr = {"detect_dead": "dd", "detect_bad": "db", "correct_dead": "cd", "spiking_decon": "decon"}

    def _start_batch(_):
        _cur = jobs.get("current")
        if _cur is not None and _cur.running:
            set_save_msg("A whole-data job is already running - cancel it first (see the bar at the top of the page).")
            return
        _shot = next((_o.content for _o in last.get("outs") or [] if _o.kind == "data"), None)
        if _shot is None or not _shot.flow:
            set_save_msg("There is no flow to apply - run a function with processing steps first.")
            return
        _a, _b = int(batch_from.value or _lo), int(batch_to.value or _hi)
        _tag = "_".join(_abbr.get(_s["step"], _s["step"]) for _s in _shot.flow)
        try:
            # a folder or an empty box gets an automatic name inside / next to the input; see resolve_output
            _out = resolve_output(batch_out.value, _shot.path, f"{Path(_shot.path).stem}_{_tag}_FFID{_a}-{_b}.sgy")
            jobs["current"] = BatchJob(
                _shot.path, _shot.flow, str(_out), fmt=batch_fmt.value, ffid_from=_a, ffid_to=_b,
                overwrite=bool(batch_overwrite.value),
            ).start()
        except Exception as _e:
            set_save_msg(f"**Could not start - {type(_e).__name__}:** {_e}")
            return
        set_save_msg(f"Started: `{_out}`  \nProgress and Cancel are in the bar at the top of the page.")
        set_job_version(lambda v: v + 1)

    def _cancel_batch(_):
        _cur = jobs.get("current")
        if _cur is not None and _cur.running:
            _cur.cancel()

    batch_button = mo.ui.button(label="🌐  Apply to whole data & save", kind="success", tooltip=WIDGET_INFO["batch_button"],
                                on_click=_start_batch)

    # "Is the flow ready?": the whole-data settings and button only appear after the user says yes, and the question comes
    # back as soon as the flow changes (the answer is tied to the flow it was given for)
    def _mark_ready(_):
        _shot = next((_o.content for _o in last.get("outs") or [] if _o.kind == "data"), None)
        if _shot is None or not _shot.flow:
            set_save_msg("There is no flow to apply - run a function with processing steps first.")
            return
        set_flow_ready(repr(_shot.flow))

    ready_button = mo.ui.button(label="✅  Yes, the flow is ready", kind="success", tooltip=WIDGET_INFO["ready_button"],
                                on_click=_mark_ready)
    notready_button = mo.ui.button(label="↩  Not ready - keep editing", tooltip=WIDGET_INFO["notready_button"],
                                   on_click=lambda _: set_flow_ready(None))
    batch_cancel = mo.ui.button(label="✖  Cancel", kind="danger", tooltip=WIDGET_INFO["batch_cancel"], on_click=_cancel_batch)
    return (batch_button, batch_cancel, batch_fmt, batch_from, batch_out, batch_overwrite, batch_to, notready_button,
            ready_button, save_button, save_data_button, save_data_fmt, save_dir, save_dpi, save_fmt, save_tables)


@app.cell
def _(mo):
    # the timer that re-runs the progress bar; it must live in its OWN cell (a cell is not re-run by its own elements)
    job_refresh = mo.ui.refresh(options=["2s"], default_interval="2s")
    return (job_refresh,)


@app.cell
def _(WIDGET_INFO, batch_cancel, jobs, job_refresh, job_version, mo, set_job_version):
    # ---- whole-data job: progress bar at the top of the page (polled every 2 s while the job runs) ----
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
        _pct = 100.0 * _s["done"] / _s["total"] if _s["total"] else 0.0
        _name = _s["out_path"].rsplit("/", 1)[-1] if _s["out_path"] else "(dry run)"
        _look = {"running": ("#2a78d6", "⏳"), "done": ("#2b9a66", "✅"), "cancelled": ("#c98a1b", "⚠️"), "error": ("#e34948", "❌")}
        _color, _icon = _look.get(_s["status"], ("#6b7280", "•"))
        _r = _s["result"]
        if _s["status"] == "running":
            _detail = (f"{_s['done']:,} / {_s['total']:,} shots · {_s['traces_in']:,} traces · elapsed {_fmt_t(_s['elapsed'])}"
                       + (f" · about {_fmt_t(_s['eta'])} left" if _s["eta"] else ""))
        elif _s["status"] == "done" and _r is not None:
            _detail = (f"{_r.shots:,} shots · traces {_r.traces_in:,} → {_r.traces_out:,} "
                       f"({_r.traces_in - _r.traces_out:,} removed) · {_r.bytes_out / 1e9:.2f} GB · {_fmt_t(_r.seconds)}"
                       + (f"<br/>report: <code>{_r.report_path}</code>" if _r.report_path else "")
                       + "".join(f"<br/>Save Data checkpoint: <code>{_p}</code>" for _p in _r.extra_outputs))
        else:
            _detail = _s["message"]
        _title = {"running": "Applying the flow to the whole data", "done": "Whole data processed",
                  "cancelled": "Cancelled - no file was written", "error": "Whole-data job failed"}.get(_s["status"], "")
        _card = mo.Html(
            f'<div style="border:1px solid {_color};border-left:6px solid {_color};border-radius:10px;padding:10px 16px;'
            f'background:rgba(128,128,128,.06)"><div style="font-weight:700">{_icon} {_title} '
            f'<span style="opacity:.65;font-weight:400">— {_name}</span></div>'
            f'<progress value="{_pct:.1f}" max="100" style="width:100%;height:10px;margin:6px 0"></progress>'
            f'<div style="font-size:.9rem;overflow-wrap:anywhere">{_detail}</div></div>'
        )
        _dismiss_btn = mo.ui.button(label="✕ Dismiss", tooltip=WIDGET_INFO["job_dismiss"], on_click=_dismiss)
        # while running: Cancel + the timer (shown = ticking); afterwards: Dismiss, and the timer stops
        job_bar = mo.vstack(
            [_card] + ([mo.hstack([batch_cancel, job_refresh], justify="start", gap=1)] if _running else [_dismiss_btn]),
            gap=0.4,
        )
    job_bar
    return (job_bar,)


@app.cell
def _(add_flow_button, batch_button, batch_fmt, batch_from, batch_out, batch_overwrite, batch_to, ctx, ctx_error, estimate_batch, ffid_slider,
       WIDGET_INFO, flow_ready, form, func, last, live, load_button, mo, not_loaded, notready_button, path_input, pipe_cards,
       pipe_down, pipe_remove, pipe_slots, pipe_steps, pipe_up, pipe_view, tip, ready_button, right, right_top, save_button,
       save_data_button, save_data_fmt, save_dir, save_dpi, save_fmt, save_msg, save_tables, topbar, zoom_back, zoom_reset):
    # ---- page layout: header, category toolbar, parameter cards left, figure + results right ----
    _card = {
        "border": "1px solid rgba(128,128,128,.28)", "border-radius": "12px", "padding": "14px 16px",
        "background": "rgba(128,128,128,.05)", "min-width": "0", "box-sizing": "border-box",
    }
    _muted = 'style="opacity:.7"'

    _data = [mo.md("#### 📁 Data"), mo.hstack([path_input, add_flow_button], justify="start", align="end", gap=0.5, widths=[1, 0])]
    if ctx_error:
        _data.append(mo.callout(mo.md(ctx_error), kind="danger"))
    elif not_loaded:
        _data.append(mo.md(f"<span {_muted}>Not loaded yet - open <b>Data → Load Data</b> and press ▶ Load data.</span>"))
    elif ctx is not None:
        _data.append(mo.md(
            f"<span {_muted}>{len(ctx.index):,} shots · FFID {ctx.ffids[0]} – {ctx.ffids[-1]} · "
            f"{ctx.sgy.ntraces:,} traces · {ctx.record_ms:g} ms records</span>"
        ))

    _params = [] if func.kind == "pipeline" else [mo.md(f"#### {func.label}"), mo.md(f"<span {_muted}>{func.description}</span>")]
    if func.kind == "loader":
        _params.append(load_button)
    if pipe_slots is not None:
        for _i, _label in enumerate(pipe_slots.value):
            _desc = pipe_steps[_label].description if _label in pipe_steps else ""
            _params.append(mo.hstack(
                [mo.md(f"**{_i + 1}. {_label}** {tip(_desc)}"),
                 mo.hstack([pipe_up[_i], pipe_down[_i], pipe_remove[_i]], justify="end", gap=0.25)],
                justify="space-between", align="center", gap=0.5))
            if str(_i) in pipe_cards:
                _params.append(pipe_cards[str(_i)])
        _params.append(pipe_view)
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
    if form is not None:
        _params.append(form)
    elif live is None and ffid_slider is None and func.kind != "loader":
        _params.append(mo.md("_This function has no parameters — it runs when selected._"))

    left = mo.vstack(
        [mo.vstack(_data, gap=0.5).style(_card), mo.vstack(_params, gap=0.8).style(_card)], gap=1
    ).style({"min-width": "0"})

    # right column: [zoom bar + gather figure] first, so the figure is on the first screen; then save, then results
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
    _save_card = None                       # 💾 Save this shot + 🌐 Apply to whole data: shown at the BOTTOM, after the results
    if last.get("outs"):
        _shot = next((_o.content for _o in last["outs"] if _o.kind == "data"), None)
        _has_data = _shot is not None
        _whole = []
        if _has_data and _shot.flow and ctx is not None:
            _steps_txt = " → ".join(_s["step"].replace("_", " ") for _s in _shot.flow)
            if flow_ready() != repr(_shot.flow):
                # ask first: the settings and the run button come only after the user confirms this flow
                _whole = [
                    mo.md("#### 🌐 Apply to whole data"),
                    mo.callout(
                        mo.md(f"**Is the flow ready?**  \n{len(_shot.flow)} step(s): {_steps_txt}.  \n"
                              "Check the result on this shot first. When the flow is what you want, confirm - then you choose "
                              "the output file and the FFID range, and it is applied to every shot."),
                        kind="warn",
                    ),
                    ready_button,
                ]
            else:
                try:
                    _est = estimate_batch(_shot.path, int(batch_from.value or 0), int(batch_to.value or 0))
                    _est_txt = (f"{_est['shots']:,} shots · {_est['traces']:,} traces · about {_est['gigabytes']:.1f} GB "
                                "will be written")
                except Exception as _e:
                    _est_txt = f"<b>{_e}</b>"
                _ck = ("<br/>A <b>Save Data</b> step in the flow also writes a checkpoint file with every shot's data at that "
                       "point, next to the final output." if any(_s["step"] == "save_data_step" for _s in _shot.flow) else "")
                _whole = [
                    mo.md("#### 🌐 Apply to whole data"),
                    mo.callout(mo.md(f"✅ **Flow confirmed:** {_steps_txt}"), kind="success"),
                    mo.md(f"<span {_muted}>Repeats the {len(_shot.flow)} step(s) on every shot from FFID to FFID, "
                          "writing one SEG-Y file while it runs. Nothing is kept in memory, so the whole file can be "
                          f"processed.{_ck}</span>"),
                    batch_out,
                    mo.hstack([batch_from, batch_to, batch_fmt, batch_overwrite], justify="start", align="end", gap=1.2, wrap=True),
                    mo.md(f"<span {_muted}>{_est_txt}</span>"),
                    mo.hstack([batch_button, notready_button], justify="start", gap=0.8),
                ]
        _save_card = (
            mo.vstack(
                [
                    mo.md("#### 💾 Save this shot"),
                    mo.hstack(
                        [save_button, save_fmt, save_dpi, save_tables]
                        + ([save_data_button, save_data_fmt] if _has_data else []),
                        justify="start", align="end", gap=1.2, wrap=True,
                    ),
                    save_dir,
                    *_whole,
                    mo.md(save_msg()).style({"overflow-wrap": "anywhere"}),
                ],
                gap=0.5,
            ).style(_card)
        )
    _parts.append(mo.vstack([right], gap=0).style({**_card, "background": "transparent", "border": "none", "padding": "0"}))
    if _save_card is not None:
        _parts.append(_save_card)
    _right = mo.vstack(_parts, gap=1).style({"min-width": "0", "max-width": "100%"})

    # top bar (fixed) + two panels that scroll on their own (styles: custom.css).  The panels sit in a CSS grid,
    # not a row of flex items: the right one may shrink below the width of a wide table, so the page never grows
    # wider than the window.
    mo.Html(
        '<div class="qc-root">'
        f"{topbar}"
        f'<div class="qc-panels"><div class="qc-scroll">{left}</div><div class="qc-scroll">{_right}</div></div>'
        "</div>"
    )
    return


if __name__ == "__main__":
    app.run()
