import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import struct
    import contextlib
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    HEADER_BYTES = 240
    FLDR_OFF, NS_OFF, DT_OFF = 8, 114, 116
    return (
        DT_OFF,
        FLDR_OFF,
        HEADER_BYTES,
        NS_OFF,
        Path,
        contextlib,
        go,
        json,
        mo,
        np,
        struct,
    )


@app.cell
def _(mo):
    mo.md("""
    # Header Editor -- .su / .sgy

    Generic trace-header field editor: read from either a headerless
    **.su** file or a full **.sgy** (SEG-Y, 3200-byte text + 400-byte
    binary reel header) file, queue edits (set / add / multiply) on any
    standard header field for the current shot, a FLDR range, or the
    whole survey, then write the result out as **.su** or **.sgy** --
    independent of what the input format was. Sample data is never
    interpreted (copied through as opaque bytes), so this works
    regardless of sample format.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Input file
    """)
    return


@app.cell
def _(mo):
    input_path = mo.ui.text(
        label="Input .sgy file path",
        value="../../EAST_ATCHAFALAYA-GCSR-UNDEFINED_SDL062470-16.sgy",
        full_width=True,
    )
    input_path
    return (input_path,)


@app.cell
def _():
    # This tool works exclusively with SEG-Y (.sgy) now -- always a
    # 3600-byte (3200-byte text + 400-byte binary) reel header in front
    # of the trace data.
    reel_bytes_in = 3600
    return (reel_bytes_in,)


@app.cell
def _(
    DT_OFF,
    HEADER_BYTES,
    NS_OFF,
    Path,
    input_path,
    mo,
    reel_bytes_in,
    struct,
):
    def _probe_any(path, reel_bytes):
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(reel_bytes)
            first = f.read(HEADER_BYTES)
        if len(first) < HEADER_BYTES:
            raise RuntimeError(
                f"File shorter than the {reel_bytes}-byte reel header + one "
                f"trace header: {path}"
            )
        order = None
        for cand in (">", "<"):
            _ns = struct.unpack_from(cand + "H", first, NS_OFF)[0]
            if 1 <= _ns <= 20000:
                order = cand
                break
        if order is None:
            raise RuntimeError(
                "Could not determine byte order (ns field unreasonable in "
                "both endiannesses) -- check the input format setting above."
            )
        ns = struct.unpack_from(order + "H", first, NS_OFF)[0]
        dt = struct.unpack_from(order + "H", first, DT_OFF)[0]
        trace_bytes = HEADER_BYTES + ns * 4
        data_bytes = size - reel_bytes
        if trace_bytes <= 0 or data_bytes % trace_bytes != 0:
            raise RuntimeError(
                f"File size after the {reel_bytes}-byte reel header "
                f"({data_bytes:,} bytes) is not an exact multiple of trace "
                f"size {trace_bytes:,} bytes (ns={ns}). Wrong input format "
                f"setting is the most likely cause."
            )
        n_traces = data_bytes // trace_bytes
        return {
            "order": order, "ns": ns, "dt_us": dt, "trace_bytes": trace_bytes,
            "n_traces": n_traces, "file_size": size, "reel_bytes": reel_bytes,
        }

    probe_info = _probe_any(Path(input_path.value), reel_bytes_in)
    mo.md(
        f"**{Path(input_path.value).name}**: **{probe_info['n_traces']:,} "
        f"traces**, ns={probe_info['ns']}, dt={probe_info['dt_us']} us, byte "
        f"order `{probe_info['order']!r}`, reel header = "
        f"{probe_info['reel_bytes']} bytes "
        f"(**{'SEG-Y' if probe_info['reel_bytes'] else 'headerless SU'}**)."
    )
    return (probe_info,)


@app.cell
def _(np):
    def build_extended_header_dtype(order, ns):
        """
        Standard 240-byte SEG-Y trace header, plus this survey's own
        custom fields at their documented byte offsets (same layout as
        build_header_scan_dtype in shot_geometry_qc_marimo.py -- RECLN at
        byte 172, RECSTN at byte 180, etc., per the survey's own EBCDIC
        text header). Exposing these as named fields (cdplbls, cdplblx,
        cdpx, cdpy in particular) lets the IL/XL feature below write into
        them directly instead of leaving them as opaque padding.
        """
        header_dtype = np.dtype([
            ('tracl', order + 'i4'), ('tracr', order + 'i4'), ('fldr', order + 'i4'),
            ('tracf', order + 'i4'), ('ep', order + 'i4'), ('cdp', order + 'i4'), ('cdpt', order + 'i4'),
            ('trid', order + 'i2'), ('nvs', order + 'i2'), ('nhs', order + 'i2'), ('duse', order + 'i2'),
            ('offset', order + 'i4'), ('gelev', order + 'i4'), ('selev', order + 'i4'),
            ('sdepth', order + 'i4'), ('gdel', order + 'i4'), ('sdel', order + 'i4'),
            ('swdep', order + 'i4'), ('gwdep', order + 'i4'),
            ('scalel', order + 'i2'), ('scalco', order + 'i2'),
            ('sx', order + 'i4'), ('sy', order + 'i4'), ('gx', order + 'i4'), ('gy', order + 'i4'),
            ('counit', order + 'i2'),
            ('pad1', 'V24'),
            ('ns', order + 'u2'), ('dt', order + 'u2'),
            ('pad_a', 'V50'),
            ('shtln', order + 'i4'), ('recln', order + 'i4'), ('shtstn', order + 'i4'), ('recstn', order + 'i4'),
            ('shtstat', order + 'i4'), ('recstat', order + 'i4'), ('shot', order + 'i4'),
            ('cdplbls', order + 'i4'), ('cdplblx', order + 'i4'), ('cdpx', order + 'i4'), ('cdpy', order + 'i4'),
            ('offset2', order + 'i4'),
            ('pad_b', 'V24'),
        ])
        assert header_dtype.itemsize == 240, header_dtype.itemsize
        full_dtype = np.dtype([('header', header_dtype), ('samples', f'V{ns * 4}')])
        return header_dtype, full_dtype

    return (build_extended_header_dtype,)


@app.cell
def _(build_extended_header_dtype, probe_info):
    order = probe_info["order"]
    ns = probe_info["ns"]
    header_dtype, _ = build_extended_header_dtype(order, ns)
    return header_dtype, ns, order


@app.cell
def _(header_dtype):
    # Exclude the two opaque padding fields -- nothing meaningful to edit
    # there, and they're not even named per-byte fields.
    header_field_names = [n for n in header_dtype.names if header_dtype[n].kind != "V"]
    return (header_field_names,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### Browse every shot in the survey

    Scan once (header-only pass, a few seconds for a 15GB survey), then
    drag the slider to page through every shot and inspect its header
    fields.
    """)
    return


@app.cell
def _(mo):
    scan_button = mo.ui.run_button(label="🔍 Scan whole survey for shot list")
    scan_button
    return (scan_button,)


@app.cell
def _(mo):
    get_shot_list, set_shot_list = mo.state(None)
    return get_shot_list, set_shot_list


@app.cell
def _(
    header_dtype,
    input_path,
    mo,
    np,
    ns,
    probe_info,
    reel_bytes_in,
    scan_button,
    set_shot_list,
):
    if scan_button.value:
        with mo.status.spinner(title="Scanning whole survey for shot (FLDR) list..."):
            _headers_only_dtype = np.dtype([("header", header_dtype), ("samples", f"V{ns * 4}")])
            _n_traces = probe_info["n_traces"]
            _fldrs = np.empty(_n_traces, dtype=np.int64)
            _n_done = 0
            with open(input_path.value, "rb") as _f:
                _f.seek(reel_bytes_in)
                _remaining = _n_traces
                while _remaining > 0:
                    _take = min(5000, _remaining)
                    _chunk = np.fromfile(_f, dtype=_headers_only_dtype, count=_take)
                    _fldrs[_n_done:_n_done + _take] = _chunk["header"]["fldr"]
                    _n_done += _take
                    _remaining -= _take
            set_shot_list(np.unique(_fldrs))
    return


@app.cell
def _(get_shot_list, mo):
    shot_list = get_shot_list()
    if shot_list is None:
        mo.stop(True, mo.md(
            "*Click **Scan whole survey for shot list** above, then use the "
            "slider below to browse.*"
        ))
    mo.md(f"**{len(shot_list):,} unique shots found**, FLDR {int(shot_list[0])}-{int(shot_list[-1])}.")
    return (shot_list,)


@app.cell
def _(mo, shot_list):
    shot_slider = mo.ui.slider(
        start=0, stop=len(shot_list) - 1, value=0, step=1,
        label="Shot index (drag to browse the whole survey)",
        show_value=True, full_width=True,
    )
    shot_slider
    return (shot_slider,)


@app.cell
def _(shot_list, shot_slider):
    current_fldr = int(shot_list[shot_slider.value])
    return (current_fldr,)


@app.cell
def _(FLDR_OFF, struct):
    def _get_fldr_at_index(f, idx, record_size, order, reel_bytes):
        f.seek(reel_bytes + idx * record_size + FLDR_OFF)
        return struct.unpack(order + "i", f.read(4))[0]

    def find_shot_range(path, order, record_size, n_traces, reel_bytes, target_fldr):
        """
        Binary search for the first/last trace index matching target_fldr,
        relying on traces being stored in field-record order (same
        assumption as the geometry QC tool's shot browser).
        """
        with open(path, "rb") as f:
            lo, hi = 0, n_traces - 1
            found = None
            while lo <= hi:
                mid = (lo + hi) // 2
                v = _get_fldr_at_index(f, mid, record_size, order, reel_bytes)
                if v == target_fldr:
                    found = mid
                    break
                elif v < target_fldr:
                    lo = mid + 1
                else:
                    hi = mid - 1
            if found is None:
                return None

            lo, hi = 0, found
            left = found
            while lo <= hi:
                mid = (lo + hi) // 2
                v = _get_fldr_at_index(f, mid, record_size, order, reel_bytes)
                if v == target_fldr:
                    left = mid
                    hi = mid - 1
                else:
                    lo = mid + 1

            lo, hi = found, n_traces - 1
            right = found
            while lo <= hi:
                mid = (lo + hi) // 2
                v = _get_fldr_at_index(f, mid, record_size, order, reel_bytes)
                if v == target_fldr:
                    right = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
        return left, right

    return (find_shot_range,)


@app.cell
def _(
    current_fldr,
    find_shot_range,
    header_dtype,
    input_path,
    mo,
    np,
    ns,
    order,
    probe_info,
    reel_bytes_in,
):
    _record_size = probe_info["trace_bytes"]
    _n_traces = probe_info["n_traces"]
    _shot_range = find_shot_range(input_path.value, order, _record_size, _n_traces, reel_bytes_in, current_fldr)
    if _shot_range is None:
        mo.stop(True, mo.md(f"**No traces found for FLDR {current_fldr}.**"))
    _left, _right = _shot_range

    _headers_only_dtype = np.dtype([("header", header_dtype), ("samples", f"V{ns * 4}")])
    with open(input_path.value, "rb") as _f:
        _f.seek(reel_bytes_in + _left * _record_size)
        _chunk = np.fromfile(_f, dtype=_headers_only_dtype, count=_right - _left + 1)
    # .copy() is essential here, not cosmetic: _chunk["header"] is a
    # strided VIEW into the interleaved header+samples buffer, and a
    # numpy view keeps its whole base array alive via .base -- without
    # copying, shot_headers silently drags the full per-trace sample
    # data (16KB+/trace) along with it, which is what blew past
    # marimo's output-size limit (23MB) even though only header fields
    # were ever displayed.
    shot_headers = _chunk["header"].copy()

    mo.md(f"**FLDR {current_fldr}**: **{len(shot_headers)} traces** loaded (file trace index {_left}-{_right}).")
    return (shot_headers,)


@app.cell
def _(header_field_names, mo, shot_headers):
    _n_show = min(50, len(shot_headers))
    _rows = [
        {f: shot_headers[f][i].item() for f in header_field_names}
        for i in range(_n_show)
    ]
    mo.vstack([
        mo.md(f"Showing first {_n_show} of {len(shot_headers):,} traces in this shot."),
        mo.ui.table(_rows, page_size=10),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### Edit a header field

    Queue as many edits as you like before writing anything out --
    nothing touches the file until **Write edited copy** below is
    clicked.
    """)
    return


@app.cell
def _(header_field_names, mo):
    edit_field = mo.ui.dropdown(
        options=header_field_names, value=header_field_names[0],
        label="Header field to edit",
    )
    edit_field
    return (edit_field,)


@app.cell
def _(edit_field, mo):
    if edit_field.value == "ns":
        _warn = mo.callout(
            "Editing 'ns' only changes this metadata field -- it does NOT "
            "resize the actual number of samples stored per trace (this "
            "tool reads/writes using the sample count it probed from the "
            "file itself, unaffected by this edit). Only use this to "
            "correct a known-wrong header value, never to actually change "
            "trace length.",
            kind="warn",
        )
    elif edit_field.value == "dt":
        _warn = mo.callout(
            "Editing 'dt' (sample interval) changes only this metadata "
            "field -- it does not resample the trace data.",
            kind="info",
        )
    else:
        _warn = None
    _warn
    return


@app.cell
def _(mo):
    edit_op = mo.ui.radio(
        options=["Set to constant", "Add delta", "Multiply by factor", "Divide by factor"],
        value="Set to constant", label="Operation", inline=True,
    )
    edit_op
    return (edit_op,)


@app.cell
def _(mo):
    edit_value = mo.ui.number(label="Value", value=0.0, step=1.0)
    edit_value
    return (edit_value,)


@app.cell
def _(mo):
    edit_scope = mo.ui.radio(
        options=["Current shot only", "FLDR range", "Whole survey"],
        value="Current shot only", label="Apply to", inline=True,
    )
    edit_scope
    return (edit_scope,)


@app.cell
def _(edit_scope, mo):
    edit_fldr_min = mo.ui.number(label="FLDR range: min", value=0, step=1)
    edit_fldr_max = mo.ui.number(label="FLDR range: max", value=0, step=1)
    mo.hstack([edit_fldr_min, edit_fldr_max]) if edit_scope.value == "FLDR range" else mo.md("")
    return edit_fldr_max, edit_fldr_min


@app.cell
def _(mo):
    queue_edit_button = mo.ui.run_button(label="💾 Queue this edit")
    queue_edit_button
    return (queue_edit_button,)


@app.cell
def _(mo):
    get_edits, set_edits = mo.state([])
    return get_edits, set_edits


@app.cell
def _(
    current_fldr,
    edit_field,
    edit_fldr_max,
    edit_fldr_min,
    edit_op,
    edit_scope,
    edit_value,
    get_edits,
    mo,
    queue_edit_button,
    set_edits,
):
    _msg = None
    if queue_edit_button.value:
        if edit_scope.value == "Current shot only":
            _fmin = _fmax = current_fldr
        elif edit_scope.value == "Whole survey":
            _fmin, _fmax = -(2 ** 31), 2 ** 31 - 1
        else:
            _fmin, _fmax = int(edit_fldr_min.value), int(edit_fldr_max.value)
            if _fmin > _fmax:
                _fmin, _fmax = _fmax, _fmin

        _new_edit = {
            "kind": "simple",
            "field": edit_field.value,
            "op": edit_op.value,
            "value": float(edit_value.value),
            "fldr_min": _fmin,
            "fldr_max": _fmax,
            "scope_label": edit_scope.value,
        }
        set_edits(get_edits() + [_new_edit])
        _msg = mo.md(
            f"Queued: **{edit_op.value}** `{edit_value.value}` on "
            f"**{edit_field.value}** for FLDR {_fmin} to {_fmax} "
            f"({edit_scope.value})."
        )
    _msg
    return


@app.cell
def _(get_edits, mo):
    def _edit_row(i, e):
        _fldr_range = "ALL" if e["scope_label"] == "Whole survey" else f"{e['fldr_min']}-{e['fldr_max']}"
        return {
            "#": i,
            "Field": e["field"],
            "Op": e["op"],
            "Value": e["value"],
            "FLDR range": _fldr_range,
            "Scope": e["scope_label"],
        }

    edits = get_edits()
    if not edits:
        edits_table = mo.md("*No edits queued yet.*")
    else:
        _rows = [_edit_row(i, e) for i, e in enumerate(edits)]
        edits_table = mo.ui.table(_rows, page_size=10, selection="multi")
    edits_table
    return edits, edits_table


@app.cell
def _(mo):
    remove_edit_button = mo.ui.run_button(label="🗑 Remove selected edit(s)")
    clear_edits_button = mo.ui.run_button(label="🧹 Clear all edits")
    mo.hstack([remove_edit_button, clear_edits_button])
    return clear_edits_button, remove_edit_button


@app.cell
def _(edits_table, get_edits, mo, remove_edit_button, set_edits):
    _msg = None
    if remove_edit_button.value:
        _selected = getattr(edits_table, "value", None)
        if _selected:
            _remove_idx = {row["#"] for row in _selected}
            set_edits([e for i, e in enumerate(get_edits()) if i not in _remove_idx])
            _msg = mo.md(f"Removed {len(_remove_idx)} edit(s).")
        else:
            _msg = mo.md("*Select one or more rows in the table above first.*")
    _msg
    return


@app.cell
def _(clear_edits_button, set_edits):
    if clear_edits_button.value:
        set_edits([])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### IL/XL geometry map (plot only)

    Off by default. Switch to IL/XL below and this asks for a grid
    corner table on the spot -- nothing is read from or written to any
    header field, it's purely a plotting convenience computed from
    whatever corner points you enter.
    """)
    return


@app.cell
def _(mo):
    ilxl_plot_toggle = mo.ui.radio(
        options=["X/Y (ft)", "IL/XL"],
        value="X/Y (ft)", inline=True,
        label="Geometry map coordinates",
    )
    ilxl_plot_toggle
    return (ilxl_plot_toggle,)


@app.cell
def _(Path, json):
    ILXL_TABLE_PATH = Path(__file__).resolve().parent / "ilxl_corner_table.json"
    _ILXL_BLANK_TABLE = [
        {"IL": 0, "XL": 0, "X": 0.0, "Y": 0.0},
        {"IL": 0, "XL": 0, "X": 0.0, "Y": 0.0},
        {"IL": 0, "XL": 0, "X": 0.0, "Y": 0.0},
        {"IL": 0, "XL": 0, "X": 0.0, "Y": 0.0},
    ]

    def load_ilxl_table():
        if ILXL_TABLE_PATH.exists():
            try:
                return json.loads(ILXL_TABLE_PATH.read_text())
            except Exception:
                pass
        return list(_ILXL_BLANK_TABLE)

    def save_ilxl_table(rows):
        ILXL_TABLE_PATH.write_text(json.dumps(rows, indent=2))

    return ILXL_TABLE_PATH, load_ilxl_table, save_ilxl_table


@app.cell
def _(ilxl_plot_toggle, load_ilxl_table, mo):
    ilxl_corner_table = mo.ui.data_editor(
        data=load_ilxl_table(),
        label="Grid corner table (IL, XL, X, Y) -- enter at least 2 known corner points",
    )
    ilxl_save_button = mo.ui.run_button(label="💾 Save this corner table (auto-loads next time)")
    _out = mo.vstack([
        mo.md("Enter this survey's known IL/XL &harr; X/Y corner points:"),
        ilxl_corner_table,
        ilxl_save_button,
    ]) if ilxl_plot_toggle.value == "IL/XL" else mo.md("")
    _out
    return ilxl_corner_table, ilxl_save_button


@app.cell
def _(ILXL_TABLE_PATH, ilxl_corner_table, ilxl_plot_toggle, ilxl_save_button, mo, save_ilxl_table):
    _msg = None
    if ilxl_save_button.value:
        save_ilxl_table(ilxl_corner_table.value)
        _msg = mo.callout(
            f"Saved to {ILXL_TABLE_PATH.name} -- will auto-load next time IL/XL is opened.",
            kind="success",
        )
    _display = _msg if ilxl_plot_toggle.value == "IL/XL" else None
    _display
    return


@app.cell
def _(ilxl_corner_table, ilxl_plot_toggle, mo, np):
    _rows = ilxl_corner_table.value
    _xl = np.array([r["XL"] for r in _rows], dtype=np.float64)
    _il = np.array([r["IL"] for r in _rows], dtype=np.float64)
    _x = np.array([r["X"] for r in _rows], dtype=np.float64)
    _y = np.array([r["Y"] for r in _rows], dtype=np.float64)

    _valid = len(_rows) >= 2 and _x.max() != _x.min() and _y.max() != _y.min()
    if _valid:
        # Least-squares line fit: XL is linear in X alone, IL linear in
        # Y alone, for an axis-aligned grid.
        _mx, _cx = np.polyfit(_x, _xl, 1)
        _my, _cy = np.polyfit(_y, _il, 1)
        _bin_x = 1.0 / _mx
        _bin_y = 1.0 / _my
        ilxl_x0 = (1001.0 - _cx) * _bin_x
        ilxl_y0 = (1001.0 - _cy) * _bin_y
        ilxl_bin_ft = (_bin_x + _bin_y) / 2.0
        _msg = mo.md(
            f"Derived grid: **X0 = {ilxl_x0:,.1f}**, **Y0 = {ilxl_y0:,.1f}**, "
            f"**bin size = {ilxl_bin_ft:.2f} ft**."
        )
    else:
        ilxl_x0, ilxl_y0, ilxl_bin_ft = 0.0, 0.0, 1.0
        _msg = mo.callout(
            "Enter at least 2 corner rows with distinct X values and "
            "distinct Y values above to compute IL/XL.",
            kind="warn",
        )

    _display = _msg if ilxl_plot_toggle.value == "IL/XL" else None
    _display
    return ilxl_bin_ft, ilxl_x0, ilxl_y0


@app.cell
def _(ilxl_bin_ft, ilxl_plot_toggle, ilxl_x0, ilxl_y0):
    def to_display_xy(x, y):
        if ilxl_plot_toggle.value == "IL/XL":
            return 1001 + (x - ilxl_x0) / ilxl_bin_ft, 1001 + (y - ilxl_y0) / ilxl_bin_ft
        return x, y
    return (to_display_xy,)


@app.cell
def _(mo):
    geom_scan_button = mo.ui.run_button(label="🔍 Scan whole survey for geometry map")
    geom_scan_button
    return (geom_scan_button,)


@app.cell
def _(mo):
    get_geom_scan, set_geom_scan = mo.state(None)
    return get_geom_scan, set_geom_scan


@app.cell
def _(
    geom_scan_button,
    header_dtype,
    input_path,
    mo,
    np,
    ns,
    probe_info,
    reel_bytes_in,
    set_geom_scan,
):
    if geom_scan_button.value:
        with mo.status.spinner(title="Scanning whole survey for SX,SY,GX,GY,FLDR..."):
            _headers_only_dtype = np.dtype([("header", header_dtype), ("samples", f"V{ns * 4}")])
            _n_traces = probe_info["n_traces"]
            _sx = np.empty(_n_traces, dtype=np.int64)
            _sy = np.empty(_n_traces, dtype=np.int64)
            _gx = np.empty(_n_traces, dtype=np.int64)
            _gy = np.empty(_n_traces, dtype=np.int64)
            _fldr = np.empty(_n_traces, dtype=np.int64)
            _n_done = 0
            with open(input_path.value, "rb") as _f:
                _f.seek(reel_bytes_in)
                _remaining = _n_traces
                while _remaining > 0:
                    _take = min(5000, _remaining)
                    _chunk = np.fromfile(_f, dtype=_headers_only_dtype, count=_take)
                    _h = _chunk["header"]
                    _sx[_n_done:_n_done + _take] = _h["sx"]
                    _sy[_n_done:_n_done + _take] = _h["sy"]
                    _gx[_n_done:_n_done + _take] = _h["gx"]
                    _gy[_n_done:_n_done + _take] = _h["gy"]
                    _fldr[_n_done:_n_done + _take] = _h["fldr"]
                    _n_done += _take
                    _remaining -= _take
            set_geom_scan({"sx": _sx, "sy": _sy, "gx": _gx, "gy": _gy, "fldr": _fldr})
    return


@app.cell
def _(get_geom_scan, mo, np):
    geom_scan = get_geom_scan()
    if geom_scan is None:
        mo.stop(True, mo.md(
            "*Click **Scan whole survey for geometry map** above first.*"
        ))

    _fldr = geom_scan["fldr"]
    _order_idx = np.argsort(_fldr, kind="stable")
    _fldr_sorted = _fldr[_order_idx]
    _first_mask = np.concatenate(([True], _fldr_sorted[1:] != _fldr_sorted[:-1]))
    _first_idx = _order_idx[_first_mask]

    geom_shot_fldr = geom_scan["fldr"][_first_idx]
    geom_shot_sx = geom_scan["sx"][_first_idx].astype(np.float64)
    geom_shot_sy = geom_scan["sy"][_first_idx].astype(np.float64)

    _recv_xy = np.unique(np.column_stack([geom_scan["gx"], geom_scan["gy"]]), axis=0)
    geom_recv_gx = _recv_xy[:, 0].astype(np.float64)
    geom_recv_gy = _recv_xy[:, 1].astype(np.float64)

    return geom_recv_gx, geom_recv_gy, geom_shot_fldr, geom_shot_sx, geom_shot_sy


@app.cell
def _(
    geom_recv_gx,
    geom_recv_gy,
    geom_shot_fldr,
    geom_shot_sx,
    geom_shot_sy,
    go,
    ilxl_plot_toggle,
    mo,
    np,
    to_display_xy,
):
    _is_ilxl = ilxl_plot_toggle.value == "IL/XL"
    _x_label, _y_label = ("XL", "IL") if _is_ilxl else ("X", "Y")

    _recv_x, _recv_y = to_display_xy(geom_recv_gx, geom_recv_gy)
    _shot_x, _shot_y = to_display_xy(geom_shot_sx, geom_shot_sy)

    fig_geom_map = go.Figure()
    fig_geom_map.add_trace(go.Scattergl(
        x=_recv_x, y=_recv_y, mode="markers",
        marker=dict(size=3, color="lightgray"),
        name="Receiver stations", hoverinfo="skip",
    ))
    fig_geom_map.add_trace(go.Scattergl(
        x=_shot_x, y=_shot_y, mode="markers",
        marker=dict(size=6, color="steelblue"),
        name="Shot points",
        customdata=geom_shot_fldr,
        hovertemplate=f"FFID %{{customdata}}<br>{_x_label}=%{{x:.1f}}<br>{_y_label}=%{{y:.1f}}<extra></extra>",
    ))

    # Proportional canvas -- size to the actual extent instead of a
    # fixed square.
    _all_x = np.concatenate([_recv_x, _shot_x])
    _all_y = np.concatenate([_recv_y, _shot_y])
    _x_range = _all_x.max() - _all_x.min()
    _y_range = _all_y.max() - _all_y.min()
    _max_dim = 900
    if _x_range >= _y_range:
        _geom_map_width, _geom_map_height = _max_dim, max(300, int(_max_dim * _y_range / _x_range))
    else:
        _geom_map_height, _geom_map_width = _max_dim, max(300, int(_max_dim * _x_range / _y_range))

    fig_geom_map.update_layout(
        title="Acquisition geometry",
        xaxis_title=_x_label, yaxis_title=_y_label,
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=_geom_map_width, height=_geom_map_height,
        legend=dict(
            itemsizing="constant", orientation="h",
            yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        ),
    )

    def _make_geom_map_png():
        return fig_geom_map.to_image(format="png", width=_geom_map_width, height=_geom_map_height, scale=2)

    mo.vstack([
        mo.ui.plotly(
            fig_geom_map,
            config={
                "displayModeBar": True,
                "toImageButtonOptions": {"filename": "geometry_map", "format": "png", "scale": 2},
            },
        ),
        mo.download(
            data=_make_geom_map_png, filename="geometry_map.png",
            mimetype="image/png", label="💾 Save geometry map",
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Preview on the currently loaded shot
    """)
    return


@app.cell
def _(current_fldr, edits, mo, np, shot_headers):
    def _apply_simple(preview, edit):
        _field = edit["field"]
        _cur = preview[_field].astype(np.float64)
        if edit["op"] == "Set to constant":
            _new = np.full_like(_cur, edit["value"])
        elif edit["op"] == "Add delta":
            _new = _cur + edit["value"]
        elif edit["op"] == "Multiply by factor":
            _new = _cur * edit["value"]
        elif edit["value"] != 0:
            _new = _cur / edit["value"]
        else:
            _new = _cur  # divide by zero -- leave untouched rather than produce inf/nan
        preview[_field] = np.round(_new).astype(preview[_field].dtype)

    _applicable = [e for e in edits if e["fldr_min"] <= current_fldr <= e["fldr_max"]]
    if not _applicable:
        _out = mo.md("*No queued edits apply to this shot.*")
    else:
        _preview = shot_headers.copy()
        _touched_fields = []
        for _edit in _applicable:
            _apply_simple(_preview, _edit)
            _touched_fields.append(_edit["field"])
        _touched_fields = list(dict.fromkeys(_touched_fields))  # de-dup, keep order

        _n_show = min(20, len(_preview))
        _rows = []
        for _i in range(_n_show):
            _row = {"trace #": _i}
            for _f in _touched_fields:
                _row[f"{_f} (before)"] = shot_headers[_f][_i].item()
                _row[f"{_f} (after)"] = _preview[_f][_i].item()
            _rows.append(_row)

        _out = mo.vstack([
            mo.md(
                f"**{len(_applicable)} queued edit(s) apply to this shot** "
                f"({len(shot_headers):,} traces) -- preview of first {_n_show}:"
            ),
            mo.ui.table(_rows, page_size=10),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### Write edited copy

    Streams the whole file once, applying every queued edit whose FLDR
    range matches each trace, and copying every other header field and
    all sample bytes through completely unchanged. Output is always
    SEG-Y (.sgy), matching the input -- the original reel header (text
    + binary) is copied through byte-for-byte, unmodified.
    """)
    return


@app.cell
def _(Path, input_path, mo):
    output_path = mo.ui.text(
        label="Output .sgy file path",
        value=str(Path(input_path.value).with_name(
            Path(input_path.value).stem + "_hdr_edited.sgy"
        )),
        full_width=True,
    )
    output_path
    return (output_path,)


@app.cell
def _(mo):
    export_button = mo.ui.run_button(label="🚀 Write edited copy")
    export_button
    return (export_button,)


@app.cell
def _(
    Path,
    contextlib,
    edits,
    export_button,
    header_dtype,
    input_path,
    mo,
    np,
    output_path,
    probe_info,
    reel_bytes_in,
):
    _out = None
    if export_button.value:
        if not edits:
            mo.stop(True, mo.md(
                "**No edits queued -- nothing to export.** Queue at least "
                "one header edit above first."
            ))
        _in_path = Path(input_path.value)
        _out_path = Path(output_path.value)
        if _out_path.resolve() == _in_path.resolve():
            mo.stop(True, mo.md(
                "**Refusing to write.** Output path is the same as the "
                "input file -- this tool never overwrites the original. "
                "Pick a different output path."
            ))

        _ns = probe_info["ns"]
        _n_traces = probe_info["n_traces"]
        _record_size = probe_info["trace_bytes"]

        _headers_only_dtype = np.dtype([("header", header_dtype), ("samples", f"V{_ns * 4}")])

        # Reel header is copied through byte-for-byte from the input
        # file -- not regenerated -- so anything already in the real
        # text/binary header survives the edit pass unchanged.
        with open(_in_path, "rb") as _f:
            _reel_header = _f.read(reel_bytes_in)

        with mo.status.spinner(
            title=f"Writing edited copy to {output_path.value} "
                  f"(full {Path(input_path.value).stat().st_size / 1e9:.1f} GB "
                  f"file copy -- this can take several minutes)..."
        ):
            _n_written = 0
            _n_touched = 0
            with contextlib.ExitStack() as _stack:
                _fin = _stack.enter_context(open(_in_path, "rb"))
                _fin.seek(reel_bytes_in)
                _fout = _stack.enter_context(open(_out_path, "wb"))
                _fout.write(_reel_header)

                _remaining = _n_traces
                while _remaining > 0:
                    _take = min(5000, _remaining)
                    _chunk = np.fromfile(_fin, dtype=_headers_only_dtype, count=_take)
                    _h = _chunk["header"]
                    _fldr_vals = _h["fldr"]

                    for _edit in edits:
                        _mask = (_fldr_vals >= _edit["fldr_min"]) & (_fldr_vals <= _edit["fldr_max"])
                        if not _mask.any():
                            continue

                        _field = _edit["field"]
                        _cur = _h[_field][_mask].astype(np.float64)
                        if _edit["op"] == "Set to constant":
                            _new = np.full_like(_cur, _edit["value"])
                        elif _edit["op"] == "Add delta":
                            _new = _cur + _edit["value"]
                        elif _edit["op"] == "Multiply by factor":
                            _new = _cur * _edit["value"]
                        elif _edit["value"] != 0:
                            _new = _cur / _edit["value"]
                        else:
                            _new = _cur  # divide by zero -- leave untouched rather than produce inf/nan
                        _h[_field][_mask] = np.round(_new).astype(_h[_field].dtype)
                        _n_touched += int(_mask.sum())

                    _chunk.tofile(_fout)
                    _n_written += _take
                    _remaining -= _take

        _in_size = _in_path.stat().st_size
        _out_size = _out_path.stat().st_size
        _expected_size = reel_bytes_in + _n_written * _record_size
        _out = mo.vstack([
            mo.md(
                f"**Done.** Wrote **{_n_written:,} traces** to "
                f"`{output_path.value}` (SEG-Y), applying "
                f"**{len(edits)} queued edit(s)** ({_n_touched:,} "
                f"total header-value changes across all edits). Every "
                f"other header field and all sample bytes were copied "
                f"through unchanged."
            ),
            mo.md(
                f"QC: output size {_out_size / 1e9:.3f} GB vs. expected "
                f"{_expected_size / 1e9:.3f} GB -- "
                + ("**match, as expected**." if _out_size == _expected_size
                   else "**MISMATCH -- something went wrong, do not use this output.**")
            ),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ### Extract particular FFIDs as SEG-Y

    Streams the whole file once and writes out **only** the traces
    whose FFID (field record / FLDR) matches your list -- everything
    else is skipped. Header and sample bytes for the traces that do
    match are copied through byte-for-byte, unmodified; the reel
    header is copied through unchanged too.
    """)
    return


@app.function
def parse_ffid_spec(spec):
    """'100' / '100,105' / '100-110' / '100, 105-108' -> sorted list of ints."""
    result = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            result.update(range(int(a.strip()), int(b.strip()) + 1))
        else:
            result.add(int(part))
    return sorted(result)


@app.cell
def _(mo):
    extract_ffid_input = mo.ui.text(
        label="FFID(s) to extract -- e.g. 100  or  100,105  or  100-110",
        full_width=True,
    )
    extract_ffid_input
    return (extract_ffid_input,)


@app.cell
def _(Path, input_path, mo):
    extract_output_path = mo.ui.text(
        label="Output .sgy file path",
        value=str(Path(input_path.value).with_name(
            Path(input_path.value).stem + "_ffid_extract.sgy"
        )),
        full_width=True,
    )
    extract_output_path
    return (extract_output_path,)


@app.cell
def _(mo):
    extract_button = mo.ui.run_button(label="🎯 Extract FFIDs to SEG-Y")
    extract_button
    return (extract_button,)


@app.cell
def _(
    Path,
    contextlib,
    extract_button,
    extract_ffid_input,
    extract_output_path,
    header_dtype,
    input_path,
    mo,
    np,
    probe_info,
    reel_bytes_in,
):
    _out = None
    if extract_button.value:
        try:
            _target_ffids = parse_ffid_spec(extract_ffid_input.value)
        except ValueError as _e:
            mo.stop(True, mo.callout(f"Couldn't parse FFID list: {_e}", kind="danger"))
        if not _target_ffids:
            mo.stop(True, mo.callout(
                "Enter at least one FFID (e.g. 100 or 100-110).", kind="warn",
            ))

        _in_path = Path(input_path.value)
        _out_path = Path(extract_output_path.value)
        if _out_path.resolve() == _in_path.resolve():
            mo.stop(True, mo.callout(
                "Refusing to write -- output path is the same as the input "
                "file. Pick a different output path.", kind="danger",
            ))

        _ns = probe_info["ns"]
        _n_traces = probe_info["n_traces"]
        _record_size = probe_info["trace_bytes"]
        _headers_only_dtype = np.dtype([("header", header_dtype), ("samples", f"V{_ns * 4}")])

        with open(_in_path, "rb") as _f:
            _reel_header = _f.read(reel_bytes_in)

        with mo.status.spinner(
            title=f"Scanning {probe_info['n_traces']:,} traces for FFID(s) "
                  f"{_target_ffids} and writing matches to {extract_output_path.value}..."
        ):
            _n_scanned = 0
            _n_written = 0
            _found_ffids = set()
            with contextlib.ExitStack() as _stack:
                _fin = _stack.enter_context(open(_in_path, "rb"))
                _fin.seek(reel_bytes_in)
                _fout = _stack.enter_context(open(_out_path, "wb"))
                _fout.write(_reel_header)

                _remaining = _n_traces
                while _remaining > 0:
                    _take = min(5000, _remaining)
                    _chunk = np.fromfile(_fin, dtype=_headers_only_dtype, count=_take)
                    _fldr_vals = _chunk["header"]["fldr"]
                    _mask = np.isin(_fldr_vals, _target_ffids)
                    if _mask.any():
                        _chunk[_mask].tofile(_fout)
                        _n_written += int(_mask.sum())
                        _found_ffids.update(np.unique(_fldr_vals[_mask]).tolist())
                    _n_scanned += _take
                    _remaining -= _take

        if _n_written == 0:
            _out_path.unlink(missing_ok=True)
            _out = mo.callout(
                f"No traces matched FFID(s) {_target_ffids} -- nothing written.",
                kind="warn",
            )
        else:
            _out_size = _out_path.stat().st_size
            _expected_size = reel_bytes_in + _n_written * _record_size
            _missing = sorted(set(_target_ffids) - _found_ffids)
            _msgs = [
                mo.md(
                    f"**Done.** Extracted **{_n_written:,} traces** "
                    f"(FFID(s) found: {sorted(_found_ffids)}) out of "
                    f"{_n_scanned:,} scanned, to `{extract_output_path.value}`."
                ),
                mo.md(
                    f"QC: output size {_out_size / 1e9:.4f} GB vs. expected "
                    f"{_expected_size / 1e9:.4f} GB -- "
                    + ("**match, as expected**." if _out_size == _expected_size
                       else "**MISMATCH -- something went wrong, do not use this output.**")
                ),
            ]
            if _missing:
                _msgs.append(mo.callout(
                    f"Requested but not found in the survey: {_missing}.", kind="warn",
                ))
            _out = mo.vstack(_msgs)
    _out
    return


if __name__ == "__main__":
    app.run()
