import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # CDP stacking -- step-by-step build

    **1.** Load data &nbsp;→&nbsp; **2.** CDP sorting &nbsp;→&nbsp;
    **3.** Corner points (optional, saved for next time) &nbsp;→&nbsp;
    **4.** CDP heat map &nbsp;→&nbsp; **5.** Velocity model + geometry
    overlay. NMO/stacking itself isn't built yet -- this is just the
    foundation.
    """)
    return


@app.cell
def _():
    import struct
    import json
    import time
    import datetime
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    from scipy.signal import butter, filtfilt

    qc_plot_dir = Path(__file__).resolve().parent / "qc_plots_stack"
    qc_plot_dir.mkdir(parents=True, exist_ok=True)
    return Path, butter, datetime, filtfilt, go, json, mo, np, plt, qc_plot_dir, struct, time


@app.cell
def _(np, struct):
    # Same SEG-Y layout established in shot_geometry_qc_marimo.py: a
    # fixed 3600-byte (3200 text + 400 binary) reel header, then
    # fixed-length 240-byte trace headers.
    REEL_BYTES = 3600
    HEADER_BYTES = 240
    NS_OFF, DT_OFF = 114, 116

    def probe_sgy(path):
        import os
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(REEL_BYTES)
            first = f.read(HEADER_BYTES)
            f.seek(REEL_BYTES - 400 + 24)
            _fmt_bytes = f.read(2)
        order = None
        for cand in ("<", ">"):
            _ns = struct.unpack_from(cand + "H", first, NS_OFF)[0]
            if 1 <= _ns <= 20000:
                order = cand
                break
        if order is None:
            raise RuntimeError("Could not determine byte order -- is this really SEG-Y?")
        ns = struct.unpack_from(order + "H", first, NS_OFF)[0]
        dt_us = struct.unpack_from(order + "H", first, DT_OFF)[0]
        format_code = struct.unpack_from(order + "h", _fmt_bytes)[0]
        trace_bytes = HEADER_BYTES + ns * 4
        data_bytes = size - REEL_BYTES
        if trace_bytes <= 0 or data_bytes % trace_bytes != 0:
            raise RuntimeError(
                f"File size after the reel header ({data_bytes:,} bytes) isn't "
                f"an exact multiple of trace size {trace_bytes:,} bytes (ns={ns})."
            )
        n_traces = data_bytes // trace_bytes
        return {
            "order": order, "ns": ns, "dt_us": dt_us, "trace_bytes": trace_bytes,
            "n_traces": n_traces, "file_size": size, "format_code": format_code,
        }

    def ibm_bits_to_float(bits):
        """Vectorized IBM System/360 float -> IEEE float32 (see shot_geometry_qc_marimo.py)."""
        bits = bits.astype(np.int64)
        sign = np.where((bits >> 31) & 1, -1.0, 1.0)
        exponent = ((bits >> 24) & 0x7F).astype(np.float64)
        mantissa = (bits & 0x00FFFFFF).astype(np.float64)
        out = sign * (mantissa / 16777216.0) * np.power(16.0, exponent - 64.0)
        out = np.where(mantissa == 0, 0.0, out)
        return out.astype(np.float32)

    def decode_samples(raw_bytes, order, format_code):
        """Format 1 = IBM float, else assume IEEE float32 -- decided by each
        file's own binary-header format code, not assumed."""
        if format_code == 1:
            _bits = np.frombuffer(raw_bytes, dtype=order + "u4")
            return ibm_bits_to_float(_bits)
        return np.frombuffer(raw_bytes, dtype=order + "f4").astype(np.float32)

    def build_cdp_scan_dtype(order, ns):
        # This survey's own (previously verified) 240-byte trace-header
        # layout. cdplbls/cdplblx are this survey's own CDP-processing
        # grid labels; cdpx/cdpy are real per-trace midpoint coordinates
        # (verified this session to match true (sx+gx)/2, (sy+gy)/2 to
        # 0.9999 correlation) -- used to derive IL/XL from the
        # user-entered corner-point table instead of trusting
        # cdplbls/cdplblx directly.
        header_dtype = np.dtype([
            ('tracl', order + 'i4'), ('tracr', order + 'i4'), ('fldr', order + 'i4'),
            ('tracf', order + 'i4'), ('ep', order + 'i4'), ('cdp', order + 'i4'), ('cdpt', order + 'i4'),
            ('trid', order + 'i2'), ('nvs', order + 'i2'), ('nhs', order + 'i2'), ('duse', order + 'i2'),
            ('offset', order + 'i4'), ('pad1', 'V74'),
            ('ns', order + 'u2'), ('dt', order + 'u2'),
            ('pad2a', 'V78'),
            ('cdplbls', order + 'i4'), ('cdplblx', order + 'i4'),
            ('cdpx', order + 'i4'), ('cdpy', order + 'i4'),
            ('pad2b', 'V28'),
        ])
        assert header_dtype.itemsize == 240, header_dtype.itemsize
        full_dtype = np.dtype([('header', header_dtype), ('samples', f'V{ns * 4}')])
        return header_dtype, full_dtype

    def build_model_scan_dtype(order, ns):
        # Assumes the velocity model SEG-Y follows the SEG-Y rev1
        # standard 3D-grid fields: INLINE_3D at bytes 189-192,
        # CROSSLINE_3D at bytes 193-196 (0-indexed offsets 188/192) --
        # a documented assumption, not verified against a real model
        # file. Tell me the real byte offsets if this is wrong.
        header_dtype = np.dtype([
            ('pad_a', 'V188'),
            ('inline3d', order + 'i4'), ('crossline3d', order + 'i4'),
            ('pad_b', 'V44'),
        ])
        assert header_dtype.itemsize == 240, header_dtype.itemsize
        full_dtype = np.dtype([('header', header_dtype), ('samples', f'V{ns * 4}')])
        return header_dtype, full_dtype

    return (
        HEADER_BYTES,
        REEL_BYTES,
        build_cdp_scan_dtype,
        build_model_scan_dtype,
        decode_samples,
        probe_sgy,
    )


@app.cell
def _(butter, filtfilt, np):
    def bandpass_filter(arr, dt_s, low_hz, high_hz, order=4):
        """Zero-phase Butterworth bandpass along the last (time) axis.
        low_hz<=0 becomes a low-pass; high_hz at/above Nyquist becomes a
        high-pass; a band covering the whole spectrum is a no-op --
        so one control handles low-pass/high-pass/bandpass/off."""
        _nyq = 0.5 / dt_s
        _lo = max(low_hz, 0.0) / _nyq
        _hi = min(high_hz, _nyq * 0.99) / _nyq
        if _lo <= 0.0 and _hi >= 0.99:
            return arr
        if _lo <= 0.0:
            _b, _a = butter(order, _hi, btype="low")
        elif _hi >= 0.99:
            _b, _a = butter(order, _lo, btype="high")
        else:
            _b, _a = butter(order, [_lo, _hi], btype="band")
        return filtfilt(_b, _a, arr, axis=-1)

    return (bandpass_filter,)


@app.cell
def _(datetime, json, qc_plot_dir):
    def log_job(record):
        """Append one job's full parameters to a persistent audit log --
        every heavy 'job' button (full-survey stack, SEG-Y export) calls
        this so past QC runs stay reconstructable later."""
        record = dict(record)
        record["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
        _log_path = qc_plot_dir / "job_log.jsonl"
        with open(_log_path, "a") as _f:
            _f.write(json.dumps(record) + "\n")
        return _log_path

    return (log_job,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 1. Load data
    """)
    return


@app.cell
def _(mo):
    import os as _os

    # $CDP_INPUT_FILE (set by launch_cdp.sh / launch_stack.sh) wins over the default path
    sgy_path = mo.ui.text(
        label="Input .sgy file path",
        value=_os.environ.get("CDP_INPUT_FILE", "").strip() or "../../EAST_ATCHAFALAYA-GCSR-UNDEFINED_SDL062470-16.sgy",
        full_width=True,
    )
    sgy_path
    return (sgy_path,)


@app.cell
def _(Path, mo, probe_sgy, sgy_path):
    mo.stop(
        not Path(sgy_path.value).exists(),
        mo.callout(f"File not found: {sgy_path.value}", kind="danger"),
    )
    probe_info = probe_sgy(sgy_path.value)
    mo.md(
        f"**{probe_info['n_traces']:,} traces**, ns={probe_info['ns']}, "
        f"dt={probe_info['dt_us']}us, format_code={probe_info['format_code']} "
        f"({'IBM float' if probe_info['format_code'] == 1 else 'IEEE float'}), "
        f"byte order `{probe_info['order']}`."
    )
    return (probe_info,)


@app.cell
def _(np, probe_info):
    ns = probe_info["ns"]
    dt_s = probe_info["dt_us"] / 1e6
    full_t = np.arange(ns) * dt_s
    return full_t, ns


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2. CDP sorting

    Scans the **whole survey** (a header-only pass, in chunks so memory
    stays bounded regardless of file size) and sorts every trace by CDP
    number, then by offset within each CDP.
    """)
    return


@app.cell
def _(mo):
    cdp_scan_button = mo.ui.run_button(label="🔍 Scan & sort whole survey by CDP")
    cdp_scan_button
    return (cdp_scan_button,)


@app.cell
def _(mo):
    get_cdp_scan, set_cdp_scan = mo.state(None)
    return get_cdp_scan, set_cdp_scan


@app.cell
def _(
    build_cdp_scan_dtype,
    cdp_scan_button,
    mo,
    np,
    probe_info,
    set_cdp_scan,
    sgy_path,
):
    if cdp_scan_button.value:
        with mo.status.spinner(title="Scanning whole survey (this reads the full file once)..."):
            _order = probe_info["order"]
            _ns = probe_info["ns"]
            _trace_bytes = probe_info["trace_bytes"]
            _n_traces = probe_info["n_traces"]
            _, _scan_dtype = build_cdp_scan_dtype(_order, _ns)

            _cdp = np.empty(_n_traces, dtype=np.int64)
            _offset = np.empty(_n_traces, dtype=np.int64)
            _cdplbls = np.empty(_n_traces, dtype=np.int64)
            _cdplblx = np.empty(_n_traces, dtype=np.int64)
            _x = np.empty(_n_traces, dtype=np.float64)
            _y = np.empty(_n_traces, dtype=np.float64)

            _chunk_size = 20000
            _n_done = 0
            with open(sgy_path.value, "rb") as _f:
                _f.seek(3600)
                _remaining = _n_traces
                while _remaining > 0:
                    _take = min(_chunk_size, _remaining)
                    _chunk = np.fromfile(_f, dtype=_scan_dtype, count=_take)
                    _h = _chunk["header"]
                    _cdp[_n_done:_n_done + _take] = _h["cdp"]
                    _offset[_n_done:_n_done + _take] = _h["offset"]
                    _cdplbls[_n_done:_n_done + _take] = _h["cdplbls"]
                    _cdplblx[_n_done:_n_done + _take] = _h["cdplblx"]
                    _x[_n_done:_n_done + _take] = _h["cdpx"]
                    _y[_n_done:_n_done + _take] = _h["cdpy"]
                    _n_done += _take
                    _remaining -= _take

            # Sort by CDP first, then by offset within each CDP (lexsort's
            # last key is primary) -- a plain argsort on cdp alone leaves
            # same-CDP traces in on-disk (acquisition) order, not the
            # near-to-far order a gather needs.
            _sort_order = np.lexsort((_offset, _cdp))
            set_cdp_scan({
                "cdp": _cdp[_sort_order],
                "offset": _offset[_sort_order],
                # Raw header CDP-processing grid labels -- kept for
                # reference, but NOT used for plotting anymore (see
                # cdp_il/cdp_xl below, computed from the corner-point
                # table instead).
                "cdplbls": _cdplbls[_sort_order],
                "cdplblx": _cdplblx[_sort_order],
                "x": _x[_sort_order],
                "y": _y[_sort_order],
                "trace_idx": np.arange(_n_traces, dtype=np.int64)[_sort_order],
            })
    return


@app.cell
def _(get_cdp_scan, mo):
    cdp_scan = get_cdp_scan()
    mo.stop(cdp_scan is None, mo.md("*Click **Scan & sort whole survey by CDP** above first.*"))
    return (cdp_scan,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3. Corner points (optional)

    If you know this survey's IL/XL &harr; real X/Y corner points, enter
    them below to enable IL/XL-registered plotting. They're saved to
    disk and auto-load next time (shared with the other tools in this
    folder, since it's the same survey grid).
    """)
    return


@app.cell
def _(Path, json):
    ILXL_TABLE_PATH = Path(__file__).resolve().parent / "ilxl_corner_table.json"
    _BLANK_TABLE = [
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
        return list(_BLANK_TABLE)

    def save_ilxl_table(rows):
        ILXL_TABLE_PATH.write_text(json.dumps(rows, indent=2))

    def has_real_corners(rows):
        return any(r["IL"] != 0 or r["XL"] != 0 or r["X"] != 0.0 or r["Y"] != 0.0 for r in rows)

    return ILXL_TABLE_PATH, has_real_corners, load_ilxl_table, save_ilxl_table


@app.cell
def _(has_real_corners, load_ilxl_table, mo):
    _saved = load_ilxl_table()
    have_corners_toggle = mo.ui.radio(
        options=["No corner points (skip IL/XL registration)", "Yes, I have corner points"],
        value="Yes, I have corner points" if has_real_corners(_saved) else "No corner points (skip IL/XL registration)",
        label="Do you have known corner points for this survey?",
    )
    have_corners_toggle
    return (have_corners_toggle,)


@app.cell
def _(have_corners_toggle, load_ilxl_table, mo):
    data_corner_table = mo.ui.data_editor(
        data=load_ilxl_table(),
        label="Data grid corner points (IL, XL <-> real X, Y, whatever grid convention you use) -- enter at least 2 known corners",
    )
    data_save_button = mo.ui.run_button(label="💾 Save corner points (auto-loads next time)")
    _out = (
        mo.vstack([data_corner_table, data_save_button])
        if have_corners_toggle.value == "Yes, I have corner points"
        else mo.md("")
    )
    _out
    return data_corner_table, data_save_button


@app.cell
def _(
    ILXL_TABLE_PATH,
    data_corner_table,
    data_save_button,
    have_corners_toggle,
    mo,
    save_ilxl_table,
):
    _msg = None
    if data_save_button.value:
        save_ilxl_table(data_corner_table.value)
        _msg = mo.callout(f"Saved to {ILXL_TABLE_PATH.name} -- will auto-load next time.", kind="success")
    _display = _msg if have_corners_toggle.value == "Yes, I have corner points" else None
    _display
    return


@app.cell
def _(np):
    def fit_corners(rows):
        """(IL, XL) -> (X, Y) forward transform from a typed corner table."""
        _il = np.array([r["IL"] for r in rows], dtype=np.float64)
        _xl = np.array([r["XL"] for r in rows], dtype=np.float64)
        _x = np.array([r["X"] for r in rows], dtype=np.float64)
        _y = np.array([r["Y"] for r in rows], dtype=np.float64)
        _valid = len(rows) >= 2 and _xl.max() != _xl.min() and _il.max() != _il.min()
        if not _valid:
            return None
        _mx, _cx = np.polyfit(_xl, _x, 1)
        _my, _cy = np.polyfit(_il, _y, 1)
        return {"mx": _mx, "cx": _cx, "my": _my, "cy": _cy}

    return (fit_corners,)


@app.cell
def _(data_corner_table, fit_corners, have_corners_toggle, mo):
    if have_corners_toggle.value == "Yes, I have corner points":
        data_transform = fit_corners(data_corner_table.value)
        _msg = (
            mo.md("Data grid registered.") if data_transform is not None
            else mo.callout(
                "Enter at least 2 corner rows with distinct IL values and "
                "distinct XL values to register the grid.",
                kind="warn",
            )
        )
    else:
        data_transform = None
        _msg = mo.md("*Skipping IL/XL registration -- plots below will use IL/XL directly, not real X/Y.*")
    _msg
    return (data_transform,)


@app.cell
def _(cdp_scan, data_transform, mo, np):
    if data_transform is not None:
        # IL/XL computed from the corner-point table's registration
        # applied to each trace's real (cdpx, cdpy) position -- NOT the
        # raw cdplbls/cdplblx header fields, which use a different grid
        # (different origin/bin size) that doesn't match whatever
        # convention the corner points were entered in.
        cdp_il = np.round((cdp_scan["y"] - data_transform["cy"]) / data_transform["my"]).astype(np.int64)
        cdp_xl = np.round((cdp_scan["x"] - data_transform["cx"]) / data_transform["mx"]).astype(np.int64)
        _msg = None
    else:
        cdp_il = cdp_scan["cdplbls"]
        cdp_xl = cdp_scan["cdplblx"]
        _msg = mo.callout(
            "No corner points entered above -- falling back to the raw "
            "cdplbls/cdplblx header fields for IL/XL below (a different "
            "grid than corner-point-derived IL/XL would give).",
            kind="warn",
        )
    _msg
    return cdp_il, cdp_xl


@app.cell
def _(cdp_il, cdp_scan, cdp_xl, mo, np):
    _u, _idx_first, _counts = np.unique(cdp_scan["cdp"], return_index=True, return_counts=True)
    _n_unique = len(_u)
    # The full sorted arrays (all _n_unique CDPs) are what everything
    # downstream uses -- this table is just a preview, capped so marimo
    # doesn't have to ship tens of thousands of rows to the browser on
    # every run (that's what blew past its output-size limit here).
    _n_show = min(500, _n_unique)
    cdp_sorted_table = [
        {"CDP": int(_u[i]), "IL": int(cdp_il[_idx_first[i]]), "XL": int(cdp_xl[_idx_first[i]]), "fold": int(_counts[i])}
        for i in range(_n_show)
    ]
    mo.vstack([
        mo.md(
            f"**{_n_unique:,} unique CDPs** across the whole survey, sorted ascending by "
            f"CDP number -- showing the first {_n_show:,}:"
        ),
        mo.ui.table(cdp_sorted_table, page_size=15, selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4. CDP heat map

    IL/XL here come from the corner-point table above -- each trace's
    real `cdpx`/`cdpy` position run through the corner-point
    registration -- not the raw `cdplbls`/`cdplblx` header fields,
    which use this survey's own CDP-processing bin grid and won't
    match whatever grid convention the corner points were entered in.
    (Falls back to the raw header fields, with a warning, if no corner
    points are entered.)
    """)
    return


@app.cell
def _(cdp_il, cdp_scan, cdp_xl, go, mo, np):
    # Fold per CDP computed from the exact CDP number (cdp_scan["cdp"]),
    # never from rounding real position into an integer IL/XL bin --
    # the data's true CDP-processing grid (~110 ft/bin) and whatever
    # grid the corner points were entered in (e.g. a 150 ft/bin
    # acquisition grid) don't divide evenly into each other, so
    # binning by rounded IL/XL merges distinct CDPs into the same cell
    # and inflates fold (this previously showed 34 instead of the true
    # 28 max). One marker per unique CDP at its own precise computed
    # position sidesteps that entirely -- same Scattergl style as the
    # acquisition geometry map in shot_geometry_qc_marimo.py, just
    # colored by fold instead of a binned heatmap.
    _u_cdp, _idx_first, _counts = np.unique(cdp_scan["cdp"], return_index=True, return_counts=True)
    _fold_il = cdp_il[_idx_first]
    _fold_xl = cdp_xl[_idx_first]

    fig_fold = go.Figure(go.Scattergl(
        x=_fold_xl, y=_fold_il, mode="markers",
        marker=dict(
            size=4, color=_counts, colorscale="Viridis",
            colorbar=dict(
                title=dict(text="Fold", font=dict(size=11)),
                x=0.98, xanchor="right", y=0.98, yanchor="top",
                len=0.5, thickness=14, outlinewidth=0,
                bgcolor="rgba(255,255,255,0.65)",
            ),
        ),
        hovertemplate="XL=%{x}<br>IL=%{y}<br>fold=%{marker.color}<extra></extra>",
    ))
    # Proportional canvas sizing from the IL/XL extent -- same approach
    # as the acquisition geometry map in shot_geometry_qc_marimo.py --
    # so the map isn't stretched to a fixed aspect ratio regardless of
    # the survey's actual shape.
    _xl_range = _fold_xl.max() - _fold_xl.min()
    _il_range = _fold_il.max() - _fold_il.min()
    _max_dim = 900
    if _xl_range >= _il_range:
        _fold_width, _fold_height = _max_dim, max(300, int(_max_dim * _il_range / max(_xl_range, 1)))
    else:
        _fold_height, _fold_width = _max_dim, max(300, int(_max_dim * _xl_range / max(_il_range, 1)))

    fig_fold.update_layout(
        title=f"CDP fold map ({int(_counts.max())} max)",
        xaxis_title="XL", yaxis_title="IL",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=_fold_width, height=_fold_height,
        margin=dict(l=70, r=40, t=70, b=60),
    )

    # Lazy PNG export (kaleido) -- deferred until the download button is
    # actually clicked, same reasoning as the geometry map: this cell
    # can re-run often (e.g. after a rescan), and building the PNG
    # eagerly every time would make that noticeably slower.
    def _make_fold_png():
        return fig_fold.to_image(format="png", width=_fold_width, height=_fold_height, scale=2)

    mo.vstack([
        mo.ui.plotly(fig_fold),
        mo.download(data=_make_fold_png, filename="cdp_fold_map.png", mimetype="image/png", label="💾 Save fold map"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 5. Velocity model
    """)
    return


@app.cell
def _(mo):
    vel_model_path = mo.ui.text(
        label="Velocity model .sgy file path",
        value="",
        full_width=True,
    )
    vel_model_path
    return (vel_model_path,)


@app.cell
def _(Path, mo, probe_sgy, vel_model_path):
    # None (not a hard stop) when no model is loaded -- lets the "stack
    # without NMO" toggle below work without ever needing a velocity
    # model, instead of this cell halting the whole rest of the notebook.
    if not vel_model_path.value:
        model_probe_info = None
        _msg = mo.md("*Enter a velocity model file path above if you want NMO correction.*")
    elif not Path(vel_model_path.value).exists():
        model_probe_info = None
        _msg = mo.callout(f"File not found: {vel_model_path.value}", kind="danger")
    else:
        model_probe_info = probe_sgy(vel_model_path.value)
        _msg = mo.md(
            f"**{model_probe_info['n_traces']:,} traces**, ns={model_probe_info['ns']}, "
            f"dt={model_probe_info['dt_us']}us, format_code={model_probe_info['format_code']}, "
            f"byte order `{model_probe_info['order']}`."
        )
    _msg
    return (model_probe_info,)


@app.cell
def _(mo):
    velocity_model_type = mo.ui.radio(
        options=["RMS velocity", "Interval velocity"],
        value="RMS velocity", inline=True,
        label="1. This model's samples are",
    )
    velocity_model_type
    return (velocity_model_type,)


@app.cell
def _(mo):
    model_velocity_units = mo.ui.radio(
        options=["ft/s", "m/s"],
        value="ft/s", inline=True,
        label="2. This model's input velocity values are in",
    )
    model_velocity_units
    return (model_velocity_units,)


@app.cell
def _(mo):
    model_domain_toggle = mo.ui.radio(
        options=["TWTT (time)", "Depth"],
        value="TWTT (time)", inline=True,
        label="3. Model's vertical axis is",
    )
    model_domain_toggle
    return (model_domain_toggle,)


@app.cell
def _(mo, model_domain_toggle, model_probe_info, model_velocity_units):
    # Depth is expressed in whatever length unit matches the input
    # velocity unit (ft/s -> ft, m/s -> m) -- this is just describing
    # the model file's own native sampling, independent of whatever
    # output unit gets picked below.
    _length_unit = "ft" if model_velocity_units.value == "ft/s" else "m"
    _is_depth = model_domain_toggle.value == "Depth"
    # Time domain defaults to the model FILE's own declared dt (not a
    # hardcoded guess) -- a mismatched sample interval silently
    # mis-times every velocity value against depth, which shows up
    # exactly as NMO not flattening events (using the wrong velocity
    # at a given t0), so getting this default right matters a lot.
    # (Falls back to a plain guess if no model is loaded yet.)
    _model_dt_us = model_probe_info["dt_us"] if model_probe_info is not None else 2000
    _default_value = 2.0 if _is_depth else round(_model_dt_us / 1000.0, 3)
    model_sample_interval = mo.ui.number(
        value=_default_value, start=0.001, step=0.5,
        label=(
            f"4. Depth grid size ({_length_unit}, matching the input velocity unit above)"
            if _is_depth
            else f"4. Time sample interval (ms) -- defaulted from this file's own header dt ({_model_dt_us}us)"
        ),
    )
    model_sample_interval
    return (model_sample_interval,)


@app.cell
def _(mo):
    output_velocity_units = mo.ui.radio(
        options=["ft/s", "m/s"],
        value="ft/s", inline=True,
        label="5. Output velocity (RMS, vs TWTT -- always converted to this before NMO) should be in",
    )
    output_velocity_units
    return (output_velocity_units,)


@app.cell
def _(mo):
    model_scan_button = mo.ui.run_button(label="🔍 Scan velocity model grid (whole file)")
    model_scan_button
    return (model_scan_button,)


@app.cell
def _(mo):
    get_model_scan, set_model_scan = mo.state(None)
    return get_model_scan, set_model_scan


@app.cell
def _(
    build_model_scan_dtype,
    mo,
    model_probe_info,
    model_scan_button,
    np,
    set_model_scan,
    vel_model_path,
):
    if model_scan_button.value and model_probe_info is not None:
        with mo.status.spinner(title="Scanning velocity model grid..."):
            _order = model_probe_info["order"]
            _ns = model_probe_info["ns"]
            _n_traces = model_probe_info["n_traces"]
            _, _scan_dtype = build_model_scan_dtype(_order, _ns)
            with open(vel_model_path.value, "rb") as _f:
                _f.seek(3600)
                _chunk = np.fromfile(_f, dtype=_scan_dtype, count=_n_traces)
            _h = _chunk["header"]
            set_model_scan({
                "inline3d": _h["inline3d"].astype(np.int64),
                "crossline3d": _h["crossline3d"].astype(np.int64),
                "trace_idx": np.arange(_n_traces, dtype=np.int64),
            })
    return


@app.cell
def _(get_model_scan, mo):
    # None (not a hard stop) when not scanned yet -- keeps this
    # optional so the "stack without NMO" toggle works without ever
    # needing a velocity model.
    model_scan = get_model_scan()
    mo.md(
        f"Model grid scanned: **{len(model_scan['trace_idx']):,} traces**."
        if model_scan is not None
        else "*Click **Scan velocity model grid** above if you want NMO correction.*"
    )
    return (model_scan,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Corner points for the velocity model
    """)
    return


@app.cell
def _(mo):
    same_corners_toggle = mo.ui.radio(
        options=["Same corner points as the data", "Different corner points (enter separately)"],
        value="Same corner points as the data", inline=True,
        label="Are the model's corner points the same as the data's?",
    )
    same_corners_toggle
    return (same_corners_toggle,)


@app.cell
def _(mo, same_corners_toggle):
    model_corner_table = mo.ui.data_editor(
        data=[
            {"IL": 0, "XL": 0, "X": 0.0, "Y": 0.0},
            {"IL": 0, "XL": 0, "X": 0.0, "Y": 0.0},
            {"IL": 0, "XL": 0, "X": 0.0, "Y": 0.0},
            {"IL": 0, "XL": 0, "X": 0.0, "Y": 0.0},
        ],
        label="Model grid corner points (INLINE_3D=IL, CROSSLINE_3D=XL <-> real X, Y)",
    )
    _out = (
        model_corner_table
        if same_corners_toggle.value == "Different corner points (enter separately)"
        else mo.md("*Using the data's corner points for the model too.*")
    )
    _out
    return (model_corner_table,)


@app.cell
def _(
    data_transform,
    fit_corners,
    mo,
    model_corner_table,
    same_corners_toggle,
):
    if same_corners_toggle.value == "Different corner points (enter separately)":
        model_transform = fit_corners(model_corner_table.value)
        _msg = (
            mo.md("Model grid registered.") if model_transform is not None
            else mo.callout(
                "Enter at least 2 model corner rows with distinct IL values and "
                "distinct XL values to register the model grid.",
                kind="warn",
            )
        )
    else:
        model_transform = data_transform
        _msg = mo.md("Model grid uses the data's corner-point registration.")
    _msg
    return (model_transform,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Geometry overlay: shots (data) vs. velocity model, in data IL/XL terms
    """)
    return


@app.cell
def _(cdp_il, cdp_xl, data_transform, go, mo, model_scan, model_transform, np):
    mo.stop(
        model_scan is None,
        mo.md("*Load and scan a velocity model above to see this overlay (not required for a raw stack without NMO).*"),
    )
    mo.stop(
        data_transform is None or model_transform is None,
        mo.md("*Enter corner points above (for the data, and the model if different) to see this overlay.*"),
    )

    # Dedup the model's own grid to unique (INLINE_3D, CROSSLINE_3D)
    # positions first -- a velocity cube can have far more traces than
    # unique positions (repeats along a third axis, e.g. angle/offset
    # stacks) -- then convert into real X/Y, then invert the DATA's
    # transform to express those same points as equivalent data IL/XL.
    _model_xy_native = np.unique(
        np.column_stack([model_scan["crossline3d"], model_scan["inline3d"]]), axis=0
    )
    _model_x = model_transform["mx"] * _model_xy_native[:, 0].astype(np.float64) + model_transform["cx"]
    _model_y = model_transform["my"] * _model_xy_native[:, 1].astype(np.float64) + model_transform["cy"]
    _model_as_data_xl = (_model_x - data_transform["cx"]) / data_transform["mx"]
    _model_as_data_il = (_model_y - data_transform["cy"]) / data_transform["my"]

    # Dedup to unique (IL, XL) bin positions before plotting -- same
    # pattern as the "unique receiver stations" dedup in
    # shot_geometry_qc_marimo.py's geometry map. Plotting one marker per
    # trace (950K+) instead of per unique bin is what blew past
    # marimo's output-size limit.
    _data_xy = np.unique(np.column_stack([cdp_xl, cdp_il]), axis=0)
    _data_xl_unique = _data_xy[:, 0]
    _data_il_unique = _data_xy[:, 1]

    # Extra safety cap regardless of dedup -- a dense 3D velocity cube
    # or a very fine data grid can still have far more unique positions
    # than are useful to actually see on a scatter plot. Decimate
    # evenly rather than truncate, so the plotted subset still spans
    # the full extent.
    _max_points = 30000

    def _decimate(x, y):
        if len(x) <= _max_points:
            return x, y
        _step = int(np.ceil(len(x) / _max_points))
        return x[::_step], y[::_step]

    _data_xl_unique, _data_il_unique = _decimate(_data_xl_unique, _data_il_unique)
    _model_as_data_xl, _model_as_data_il = _decimate(_model_as_data_xl, _model_as_data_il)

    # Model drawn first (background context), data drawn second so it
    # renders ON TOP -- previously the model's markers were added last
    # and, being bigger/opaque, shadowed the data wherever the two
    # overlapped.
    fig_geom = go.Figure()
    fig_geom.add_trace(go.Scattergl(
        x=_model_as_data_xl, y=_model_as_data_il, mode="markers",
        marker=dict(size=5, color="lightgray", symbol="square"),
        name="Model grid (converted to data IL/XL)", hoverinfo="skip",
    ))
    fig_geom.add_trace(go.Scattergl(
        x=_data_xl_unique, y=_data_il_unique, mode="markers",
        marker=dict(size=3, color="steelblue"),
        name="Data (unique IL/XL bins)",
    ))

    # Proportional canvas sizing from the combined data+model extent --
    # same approach as the acquisition geometry map in
    # shot_geometry_qc_marimo.py.
    _all_x = np.concatenate([_data_xl_unique, _model_as_data_xl])
    _all_y = np.concatenate([_data_il_unique, _model_as_data_il])
    _x_range = _all_x.max() - _all_x.min()
    _y_range = _all_y.max() - _all_y.min()
    _max_dim = 900
    if _x_range >= _y_range:
        _geom_width, _geom_height = _max_dim, max(300, int(_max_dim * _y_range / max(_x_range, 1)))
    else:
        _geom_height, _geom_width = _max_dim, max(300, int(_max_dim * _x_range / max(_y_range, 1)))

    fig_geom.update_layout(
        title="Data vs. model grid coverage",
        xaxis_title="Data XL", yaxis_title="Data IL",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=_geom_width, height=_geom_height,
        # Legend outside the plot area, horizontal along the top --
        # same placement as the acquisition geometry map.
        legend=dict(
            itemsizing="constant", orientation="h",
            yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        ),
    )

    def _make_geom_png():
        return fig_geom.to_image(format="png", width=_geom_width, height=_geom_height, scale=2)

    mo.vstack([
        mo.ui.plotly(fig_geom),
        mo.download(data=_make_geom_png, filename="geometry_overlay.png", mimetype="image/png", label="💾 Save geometry overlay"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 6. NMO correction & stacking

    Pick a CDP below to stack its gather -- with NMO correction, the
    velocity comes from the model, matched by real position (not CDP
    number), exactly the same lookup as the geometry overlay above.
    """)
    return


@app.cell
def _(mo):
    apply_nmo_toggle = mo.ui.radio(
        options=["Apply NMO correction", "Stack without NMO (raw, no velocity model needed)"],
        value="Apply NMO correction",
        label="Correction",
    )
    apply_nmo_toggle
    return (apply_nmo_toggle,)


@app.cell
def _(mo):
    stretch_mute_pct = mo.ui.slider(0, 100, value=30, step=5, label="NMO stretch mute (%, 0 = off, ignored without NMO)")
    clip_percentile = mo.ui.slider(80, 100, value=98, step=0.5, label="Amplitude clip percentile (all image plots)")
    mo.hstack([stretch_mute_pct, clip_percentile])
    return clip_percentile, stretch_mute_pct


@app.cell
def _(mo):
    configure_filter_button = mo.ui.run_button(label="⚙️ Configure filter")
    close_filter_panel_button = mo.ui.run_button(label="✅ Done")
    return close_filter_panel_button, configure_filter_button


@app.cell
def _(mo):
    get_filter_panel_open, set_filter_panel_open = mo.state(False)
    return get_filter_panel_open, set_filter_panel_open


@app.cell
def _(close_filter_panel_button, configure_filter_button, set_filter_panel_open):
    if configure_filter_button.value:
        set_filter_panel_open(True)
    if close_filter_panel_button.value:
        set_filter_panel_open(False)
    return


@app.cell
def _(mo):
    # Always constructed (not just when the panel is open) -- their
    # .value must stay usable downstream even while the panel is
    # closed, same pattern as the corner-point tables above.
    filter_type = mo.ui.radio(
        options=["Low-pass", "High-pass", "Bandpass"],
        value="Bandpass", inline=True,
        label="Filter type",
    )
    filter_cutoff_hz = mo.ui.number(value=60.0, start=0.1, step=1.0, label="Cutoff frequency (Hz)")
    filter_low_hz = mo.ui.number(value=5.0, start=0.0, step=1.0, label="Low-cut (Hz)")
    filter_high_hz = mo.ui.number(value=80.0, start=1.0, step=1.0, label="High-cut (Hz)")
    return filter_cutoff_hz, filter_high_hz, filter_low_hz, filter_type


@app.cell
def _(
    filter_cutoff_hz,
    filter_high_hz,
    filter_low_hz,
    filter_type,
):
    # Resolve whatever the panel is set to into the (low_hz, high_hz)
    # pair bandpass_filter() already understands -- low<=0 is a
    # low-pass, high>=Nyquist is a high-pass, anything else is a band.
    if filter_type.value == "Low-pass":
        active_filter_low_hz, active_filter_high_hz = 0.0, filter_cutoff_hz.value
    elif filter_type.value == "High-pass":
        active_filter_low_hz, active_filter_high_hz = filter_cutoff_hz.value, 1.0e9
    else:
        active_filter_low_hz, active_filter_high_hz = filter_low_hz.value, filter_high_hz.value
    return active_filter_high_hz, active_filter_low_hz


@app.cell
def _(
    close_filter_panel_button,
    configure_filter_button,
    filter_cutoff_hz,
    filter_high_hz,
    filter_low_hz,
    filter_type,
    get_filter_panel_open,
    mo,
):
    if get_filter_panel_open():
        _param_row = (
            mo.hstack([filter_low_hz, filter_high_hz])
            if filter_type.value == "Bandpass"
            else filter_cutoff_hz
        )
        _out = mo.vstack([
            configure_filter_button,
            mo.callout(
                mo.vstack([mo.md("**Filter settings**"), filter_type, _param_row, close_filter_panel_button]),
                kind="neutral",
            ),
        ])
    else:
        _out = configure_filter_button
    _out
    return


@app.cell
def _(mo):
    filter_at_raw = mo.ui.checkbox(value=False, label="Apply at: raw gather (before NMO)")
    filter_at_postnmo = mo.ui.checkbox(value=False, label="Apply at: after NMO (before stack)")
    filter_at_stack = mo.ui.checkbox(value=False, label="Apply at: stacked trace/section (after stack)")
    mo.vstack([
        mo.md("Turn the filter above on at whichever stage(s) you want it -- any combination:"),
        filter_at_raw, filter_at_postnmo, filter_at_stack,
    ])
    return filter_at_postnmo, filter_at_raw, filter_at_stack


@app.cell
def _(cdp_il, cdp_scan, cdp_xl, mo, np):
    _u_cdp, _idx_first, _counts = np.unique(cdp_scan["cdp"], return_index=True, return_counts=True)
    _order_idx = np.argsort(-_counts)[:200]
    nmo_cdp_options = {
        f"CDP {int(_u_cdp[i])}  (IL {int(cdp_il[_idx_first[i]])}, XL {int(cdp_xl[_idx_first[i]])}, fold {int(_counts[i])})": int(_u_cdp[i])
        for i in _order_idx
    }
    nmo_cdp_picker = mo.ui.dropdown(
        options=nmo_cdp_options,
        value=next(iter(nmo_cdp_options)),
        label="CDP to NMO-correct & stack (top 200 by fold)",
    )
    nmo_cdp_picker
    return (nmo_cdp_picker,)


@app.cell
def _(
    HEADER_BYTES,
    REEL_BYTES,
    cdp_scan,
    decode_samples,
    np,
    output_velocity_units,
    probe_info,
    sgy_path,
):
    def load_gather_for_cdp(target_cdp):
        """Targeted read of one CDP's traces (sorted by offset) plus its
        real (x, y) position, using the already-scanned cdp_scan index."""
        _mask = cdp_scan["cdp"] == target_cdp
        _idxs = cdp_scan["trace_idx"][_mask]
        # Offsets are stored in feet (this survey's own header
        # convention) -- convert to the chosen output unit system so
        # they stay consistent with model_velocity_for's velocity units.
        _offsets_raw = cdp_scan["offset"][_mask].astype(np.float64)
        if output_velocity_units.value == "m/s":
            _offsets_raw = _offsets_raw / 3.280839895
        _x = float(cdp_scan["x"][_mask][0])
        _y = float(cdp_scan["y"][_mask][0])

        _order = probe_info["order"]
        _ns = probe_info["ns"]
        _fmt = probe_info["format_code"]
        _trace_bytes = probe_info["trace_bytes"]

        _traces = np.empty((len(_idxs), _ns), dtype=np.float32)
        with open(sgy_path.value, "rb") as _f:
            for _row, _idx in enumerate(_idxs):
                _f.seek(REEL_BYTES + int(_idx) * _trace_bytes + HEADER_BYTES)
                _raw = _f.read(_ns * 4)
                _traces[_row] = decode_samples(_raw, _order, _fmt)

        _sort_order = np.argsort(_offsets_raw)
        return _traces[_sort_order].astype(np.float64), _offsets_raw[_sort_order], _x, _y

    return (load_gather_for_cdp,)


@app.cell
def _(
    HEADER_BYTES,
    REEL_BYTES,
    decode_samples,
    full_t,
    model_domain_toggle,
    model_probe_info,
    model_sample_interval,
    model_scan,
    model_transform,
    model_velocity_units,
    np,
    output_velocity_units,
    vel_model_path,
    velocity_model_type,
):
    def model_velocity_for(x, y):
        """Real-position-matched (not CDP-matched) velocity lookup: find
        the model trace nearest this (x, y), decode it, convert to a
        time-domain RMS velocity, and interpolate onto the data's own
        time axis. Returns (vel_of_t0, match_info)."""
        _model_x = model_transform["mx"] * model_scan["crossline3d"].astype(np.float64) + model_transform["cx"]
        _model_y = model_transform["my"] * model_scan["inline3d"].astype(np.float64) + model_transform["cy"]
        _dist2 = (_model_x - x) ** 2 + (_model_y - y) ** 2
        _best = int(np.argmin(_dist2))
        _trace_idx = int(model_scan["trace_idx"][_best])
        _match_info = {
            "inline3d": int(model_scan["inline3d"][_best]),
            "crossline3d": int(model_scan["crossline3d"][_best]),
            "distance_ft": float(np.sqrt(_dist2[_best])),
        }

        _order = model_probe_info["order"]
        _ns = model_probe_info["ns"]
        _fmt = model_probe_info["format_code"]
        _trace_bytes = model_probe_info["trace_bytes"]
        with open(vel_model_path.value, "rb") as _f:
            _f.seek(REEL_BYTES + _trace_idx * _trace_bytes + HEADER_BYTES)
            _raw = _f.read(_ns * 4)
        _native = decode_samples(_raw, _order, _fmt).astype("float64")

        # Convert the input velocity (whatever units it's declared in)
        # into the chosen OUTPUT unit BEFORE any Dix conversion, so
        # interval->RMS happens in one consistent system rather than
        # mixing units mid-calculation. The data's offsets get the
        # same treatment in load_gather_for_cdp, so NMO always sees
        # offset and velocity in matching units.
        _ft_per_m = 3.280839895
        if model_velocity_units.value == "m/s" and output_velocity_units.value == "ft/s":
            _native = _native * _ft_per_m
        elif model_velocity_units.value == "ft/s" and output_velocity_units.value == "m/s":
            _native = _native / _ft_per_m

        if model_domain_toggle.value == "Depth":
            # Depth-domain models are always treated as interval
            # velocity vs depth -- integrate to get two-way time per
            # depth step, then Dix-convert on that (irregular) time axis.
            _dz = model_sample_interval.value
            _vint = _native
            _dt_i = np.where(_vint > 0, 2.0 * _dz / np.where(_vint > 0, _vint, 1.0), 0.0)
            _model_t = np.cumsum(_dt_i)
            _cum_v2dt = np.cumsum(_vint ** 2 * _dt_i)
            _vrms_native = np.sqrt(
                np.divide(_cum_v2dt, _model_t, out=np.full_like(_cum_v2dt, _vint[0] ** 2), where=_model_t > 0)
            )
            _vint_on_t = _vint
            _native_axis = np.arange(_ns) * _dz
            _native_axis_kind = "Depth"
        else:
            _model_dt_s = model_sample_interval.value / 1000.0
            _model_t = np.arange(_ns) * _model_dt_s
            if velocity_model_type.value == "Interval velocity":
                _dt_arr = np.diff(_model_t, prepend=0.0)
                _cum = np.cumsum(_native ** 2 * _dt_arr)
                _vrms_native = np.sqrt(
                    np.divide(_cum, _model_t, out=np.full_like(_cum, _native[0] ** 2), where=_model_t > 0)
                )
                _vint_on_t = _native
            else:
                _vrms_native = _native
                # Native data is RMS-only -- invert Dix to get the
                # equivalent interval velocity, for the sections/overlay
                # plots' "Interval" display option.
                _dt_arr = np.diff(_model_t, prepend=0.0)
                _cum = _model_t * _vrms_native ** 2
                _prev_cum = np.concatenate(([0.0], _cum[:-1]))
                _vint2 = np.divide(
                    _cum - _prev_cum, _dt_arr,
                    out=np.full_like(_cum, _vrms_native[0] ** 2), where=_dt_arr > 0,
                )
                _vint_on_t = np.sqrt(np.maximum(_vint2, 0.0))
            _native_axis = _model_t
            _native_axis_kind = "Time"

        # Exposed for the input-vs-output velocity comparison plot --
        # not needed for the interpolated result used in NMO itself.
        _match_info["native_velocity"] = _native
        _match_info["native_axis"] = _native_axis
        _match_info["native_axis_kind"] = _native_axis_kind
        _match_info["vrms_native"] = _vrms_native
        _match_info["model_t"] = _model_t

        _vel_of_t0 = np.interp(full_t, _model_t, _vrms_native, left=_vrms_native[0], right=_vrms_native[-1])
        _match_info["interval_of_t0"] = np.interp(
            full_t, _model_t, _vint_on_t, left=_vint_on_t[0], right=_vint_on_t[-1]
        )
        return _vel_of_t0, _match_info

    # None (not a hard stop) when the model isn't fully set up -- keeps
    # this optional so "stack without NMO" works without ever needing
    # a velocity model loaded.
    if model_probe_info is None or model_scan is None or model_transform is None:
        model_velocity_for = None
    return (model_velocity_for,)


@app.cell
def _(np):
    def nmo_stack(traces, offsets, full_t, vel_of_t0, mute_frac, post_nmo_filter=None):
        _n_traces, _ns = traces.shape
        _nmo = np.full((_n_traces, _ns), np.nan)
        for _i in range(_n_traces):
            _x = offsets[_i]
            _t_moveout = np.sqrt(full_t ** 2 + (_x / vel_of_t0) ** 2)
            _amp = np.interp(_t_moveout, full_t, traces[_i], left=0.0, right=0.0)
            if mute_frac > 0:
                with np.errstate(divide="ignore", invalid="ignore"):
                    _stretch = np.where(full_t > 0, (_t_moveout - full_t) / full_t, 0.0)
                _amp = np.where(np.abs(_stretch) > mute_frac, np.nan, _amp)
            _nmo[_i] = _amp
        _live_count = np.sum(~np.isnan(_nmo), axis=0)
        _nmo_filled = np.nan_to_num(_nmo, nan=0.0)
        if post_nmo_filter is not None:
            # Filtering after the mute zeroes out muted samples smears a
            # little energy across that boundary -- an expected real
            # effect of filtering a muted gather, not a bug.
            _nmo_filled = post_nmo_filter(_nmo_filled)
        _stack = np.divide(
            np.sum(_nmo_filled, axis=0), _live_count,
            out=np.zeros(_ns), where=_live_count > 0,
        )
        return _nmo_filled, _stack

    return (nmo_stack,)


@app.cell
def _(
    active_filter_high_hz,
    active_filter_low_hz,
    apply_nmo_toggle,
    bandpass_filter,
    filter_at_postnmo,
    filter_at_raw,
    filter_at_stack,
    full_t,
    load_gather_for_cdp,
    mo,
    model_velocity_for,
    nmo_cdp_picker,
    nmo_stack,
    np,
    output_velocity_units,
    stretch_mute_pct,
):
    nmo_gather_traces, _offsets, _x, _y = load_gather_for_cdp(nmo_cdp_picker.value)

    _dt_s = full_t[1] - full_t[0]
    _filt = lambda _arr: bandpass_filter(_arr, _dt_s, active_filter_low_hz, active_filter_high_hz)

    _want_nmo = apply_nmo_toggle.value == "Apply NMO correction"
    _warnings = []
    if _want_nmo and model_velocity_for is None:
        _warnings.append(
            "No velocity model loaded/scanned above -- stacking without NMO "
            "instead. Load a model to apply NMO, or switch the toggle to "
            "'Stack without NMO' to silence this."
        )
        _want_nmo = False

    # Independent on/off toggles at each stage. Without NMO there's no
    # separate correction step, so "after NMO" and "raw gather" both
    # just mean "before averaging" -- either one filters there once.
    if filter_at_raw.value or (not _want_nmo and filter_at_postnmo.value):
        nmo_gather_traces = _filt(nmo_gather_traces)

    if _want_nmo:
        nmo_vel_of_t0, nmo_match_info = model_velocity_for(_x, _y)
        _post_nmo_filter = _filt if filter_at_postnmo.value else None
        nmo_traces_display, nmo_stacked_trace = nmo_stack(
            nmo_gather_traces, _offsets, full_t, nmo_vel_of_t0, stretch_mute_pct.value / 100.0,
            post_nmo_filter=_post_nmo_filter,
        )
        if filter_at_stack.value:
            nmo_stacked_trace = _filt(nmo_stacked_trace)
        _vel_unit = output_velocity_units.value

        # Sanity checks -- NMO not flattening events is almost always
        # either (a) the matched model trace being nowhere near this CDP
        # (bad corner-point registration for the model) or (b) a velocity
        # that's implausible for seismic data (wrong units, or RMS/interval
        # mixed up). Neither is a crash, so they'd otherwise show up only
        # as a subtly wrong-looking stack.
        if nmo_match_info["distance_ft"] > 5000:
            _warnings.append(
                f"Matched model trace is {nmo_match_info['distance_ft']:,.0f} ft away -- "
                "that's far for a velocity lookup. Check the model's corner points "
                "(same-as-data is only correct if the model truly uses the data's own "
                "grid numbering) and the INLINE_3D/CROSSLINE_3D byte-offset assumption."
            )
        _vel_min_ok, _vel_max_ok = (5000, 25000) if _vel_unit == "ft/s" else (1500, 7600)
        if nmo_vel_of_t0.min() < _vel_min_ok or nmo_vel_of_t0.max() > _vel_max_ok:
            _warnings.append(
                f"Velocity range {nmo_vel_of_t0.min():,.0f}-{nmo_vel_of_t0.max():,.0f} {_vel_unit} "
                f"is outside the typical seismic range ({_vel_min_ok:,}-{_vel_max_ok:,} {_vel_unit}) -- "
                "check the input velocity units and RMS/Interval selection above."
            )

        _info_md = mo.md(
            f"CDP {nmo_cdp_picker.value}: real position X={_x:,.1f}, Y={_y:,.1f} -- "
            f"matched model trace INLINE_3D={nmo_match_info['inline3d']}, "
            f"CROSSLINE_3D={nmo_match_info['crossline3d']} "
            f"({nmo_match_info['distance_ft']:,.1f} ft away). "
            f"Output velocity range (RMS, TWTT): "
            f"{nmo_vel_of_t0.min():,.0f} - {nmo_vel_of_t0.max():,.0f} {_vel_unit}."
        )
    else:
        # Raw stack -- no moveout correction, no velocity needed at all.
        nmo_vel_of_t0 = None
        nmo_match_info = None
        nmo_traces_display = nmo_gather_traces
        nmo_stacked_trace = np.mean(nmo_gather_traces, axis=0)
        if filter_at_stack.value:
            nmo_stacked_trace = _filt(nmo_stacked_trace)
        _info_md = mo.md(
            f"CDP {nmo_cdp_picker.value}: raw stack (no NMO), "
            f"{nmo_gather_traces.shape[0]} traces."
        )

    mo.vstack([_info_md, *[mo.callout(_w, kind="warn") for _w in _warnings]])
    return (
        nmo_gather_traces,
        nmo_match_info,
        nmo_stacked_trace,
        nmo_traces_display,
        nmo_vel_of_t0,
    )


@app.cell
def _(
    clip_percentile,
    full_t,
    mo,
    nmo_gather_traces,
    nmo_match_info,
    nmo_stacked_trace,
    nmo_traces_display,
    nmo_vel_of_t0,
    np,
    output_velocity_units,
    plt,
    qc_plot_dir,
):
    _y_max_ms = full_t[-1] * 1000.0
    _t_ms = full_t * 1000.0

    _vclip_raw = np.percentile(np.abs(nmo_gather_traces), clip_percentile.value) or 1.0
    _vclip_nmo = np.percentile(np.abs(nmo_traces_display), clip_percentile.value) or 1.0

    _fig, _axes = plt.subplots(1, 4, figsize=(13.5, 6), gridspec_kw={"width_ratios": [1, 1, 0.4, 0.6]})
    _axes[0].imshow(
        nmo_gather_traces.T, aspect="auto", cmap="gray",
        vmin=-_vclip_raw, vmax=_vclip_raw,
        extent=[0, nmo_gather_traces.shape[0], _y_max_ms, 0],
    )
    _axes[0].set_ylim(_y_max_ms, 0)
    _axes[0].set_title("CDP gather (offset-sorted)")
    _axes[0].set_xlabel("Trace #")
    _axes[0].set_ylabel("Time (ms)")

    _axes[1].imshow(
        nmo_traces_display.T, aspect="auto", cmap="gray",
        vmin=-_vclip_nmo, vmax=_vclip_nmo,
        extent=[0, nmo_traces_display.shape[0], _y_max_ms, 0],
    )
    _axes[1].set_ylim(_y_max_ms, 0)
    _axes[1].set_title("NMO stretch section")
    _axes[1].set_xlabel("Trace #")

    _vclip_stack = np.percentile(np.abs(nmo_stacked_trace), clip_percentile.value) or 1e-9
    _axes[2].plot(nmo_stacked_trace, _t_ms, color="black", linewidth=0.7)
    _axes[2].set_xlim(-_vclip_stack, _vclip_stack)
    _axes[2].set_ylim(_y_max_ms, 0)
    _axes[2].set_title("Stack")
    _axes[2].set_xlabel("Amp")

    if nmo_vel_of_t0 is not None:
        _line_out, = _axes[3].plot(
            nmo_vel_of_t0, _t_ms, color="tab:blue", linewidth=1.2, label="Output (RMS)",
        )
        _axes[3].set_title("Velocity used")
        _axes[3].set_xlabel(output_velocity_units.value)

        if nmo_match_info is not None:
            # Input (as-loaded) velocity overlaid on the same panel --
            # replaces the separate "Convert to RMS" comparison plot.
            # Own y-axis (twinx) since a depth-domain model's native
            # axis is depth, not time.
            _input_axis_kind = nmo_match_info["native_axis_kind"]
            _input_axis = nmo_match_info["native_axis"]
            if _input_axis_kind == "Time":
                _input_axis = _input_axis * 1000.0
            _length_unit = "ft" if output_velocity_units.value == "ft/s" else "m"
            _input_unit_label = f"{_input_axis_kind} ({'ms' if _input_axis_kind == 'Time' else _length_unit})"

            _ax_vel_in = _axes[3].twinx()
            _line_in, = _ax_vel_in.plot(
                nmo_match_info["native_velocity"], _input_axis, color="tab:red", linewidth=1.2,
                label=f"Input ({_input_axis_kind.lower()})",
            )
            _ax_vel_in.set_ylabel(_input_unit_label, color="tab:red", fontsize=7)
            _ax_vel_in.tick_params(axis="y", labelcolor="tab:red", labelsize=6)

            # Sync the twin axis to the same real TWT range as the main
            # panel (converted into the twin's own units for a depth
            # model) -- otherwise each axis auto-scales to its own data
            # range and a shared pixel height stops meaning the same
            # time on both curves.
            if _input_axis_kind == "Time":
                _ax_vel_in.set_ylim(_y_max_ms, 0)
                _model_max_ms = nmo_match_info["model_t"][-1] * 1000.0
            else:
                _model_t_ms = nmo_match_info["model_t"] * 1000.0
                _depth_at_0 = np.interp(0.0, _model_t_ms, _input_axis)
                _depth_at_max = np.interp(_y_max_ms, _model_t_ms, _input_axis, right=_input_axis[-1])
                _ax_vel_in.set_ylim(_depth_at_max, _depth_at_0)
                _model_max_ms = _model_t_ms[-1]

            # Model's own recorded range can be shorter than the full
            # data window -- beyond it, the output curve just holds
            # flat at the last known value.
            if _model_max_ms < _y_max_ms * 0.999:
                _axes[3].axhline(_model_max_ms, color="gray", linestyle="--", linewidth=0.7)

            _axes[3].legend(handles=[_line_out, _line_in], loc="lower right", fontsize=6)
    else:
        _axes[3].text(0.5, 0.5, "No NMO\n(raw stack)", ha="center", va="center", transform=_axes[3].transAxes)
        _axes[3].set_title("Velocity used")
    _axes[3].set_ylim(_y_max_ms, 0)

    _fig.tight_layout()
    _p = qc_plot_dir / "nmo_preview.png"
    _fig.savefig(_p, dpi=130)
    plt.close(_fig)
    mo.vstack([
        mo.image(src=str(_p)),
        mo.download(data=_p.read_bytes(), filename="nmo_preview.png", mimetype="image/png", label="💾 Save this figure"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 7. Stack the full data

    NMO-corrects and stacks **every CDP** in the scanned survey (each
    using its own coordinate-matched model velocity, same as above) --
    this is the expensive step, gated behind a button. Once it's done,
    picking an inline/crossline slice below is instant, since it's just
    reading back already-computed stacks.
    """)
    return


@app.cell
def _(mo):
    full_stack_button = mo.ui.run_button(label="📚 Stack full data (all CDPs)")
    full_stack_button
    return (full_stack_button,)


@app.cell
def _(mo):
    get_full_stack, set_full_stack = mo.state(None)
    return get_full_stack, set_full_stack


@app.cell
def _(
    active_filter_high_hz,
    active_filter_low_hz,
    apply_nmo_toggle,
    bandpass_filter,
    cdp_il,
    cdp_scan,
    cdp_xl,
    clip_percentile,
    filter_at_postnmo,
    filter_at_raw,
    filter_at_stack,
    filter_type,
    full_stack_button,
    full_t,
    load_gather_for_cdp,
    log_job,
    mo,
    model_domain_toggle,
    model_sample_interval,
    model_velocity_for,
    model_velocity_units,
    nmo_stack,
    np,
    ns,
    output_velocity_units,
    set_full_stack,
    sgy_path,
    stretch_mute_pct,
    time,
    vel_model_path,
    velocity_model_type,
):
    if full_stack_button.value:
        _t_start = time.time()
        _u_cdp, _idx_first = np.unique(cdp_scan["cdp"], return_index=True)
        _n_cdp = len(_u_cdp)
        _mute_frac = stretch_mute_pct.value / 100.0
        _want_nmo = apply_nmo_toggle.value == "Apply NMO correction" and model_velocity_for is not None
        _dt_s = full_t[1] - full_t[0]
        _filt = lambda _arr: bandpass_filter(_arr, _dt_s, active_filter_low_hz, active_filter_high_hz)
        _filter_raw = filter_at_raw.value or (not _want_nmo and filter_at_postnmo.value)
        _filter_post_nmo = _want_nmo and filter_at_postnmo.value
        _filter_stack = filter_at_stack.value

        _stack_out = np.empty((_n_cdp, ns), dtype=np.float64)
        _il_out = cdp_il[_idx_first]
        _xl_out = cdp_xl[_idx_first]
        _x_out = np.empty(_n_cdp, dtype=np.float64)
        _y_out = np.empty(_n_cdp, dtype=np.float64)

        with mo.status.progress_bar(
            total=_n_cdp, title="Stacking full data..." if _want_nmo else "Stacking full data (raw, no NMO)...",
            subtitle=f"{_n_cdp:,} CDPs -- this can take a while",
        ) as _bar:
            for _i, _cdp in enumerate(_u_cdp):
                _traces, _offsets, _x, _y = load_gather_for_cdp(int(_cdp))
                _x_out[_i] = _x
                _y_out[_i] = _y
                if _filter_raw:
                    _traces = _filt(_traces)
                if _want_nmo:
                    _vel, _ = model_velocity_for(_x, _y)
                    _, _stack = nmo_stack(
                        _traces, _offsets, full_t, _vel, _mute_frac,
                        post_nmo_filter=_filt if _filter_post_nmo else None,
                    )
                else:
                    _stack = np.mean(_traces, axis=0)
                if _filter_stack:
                    _stack = _filt(_stack)
                _stack_out[_i] = _stack
                _bar.update()

        set_full_stack({
            "cdp": _u_cdp, "il": _il_out, "xl": _xl_out,
            "x": _x_out, "y": _y_out, "stack": _stack_out,
        })

        log_job({
            "job": "full_stack",
            "sgy_path": sgy_path.value,
            "n_cdp": int(_n_cdp),
            "elapsed_s": round(time.time() - _t_start, 2),
            "apply_nmo": bool(_want_nmo),
            "stretch_mute_pct": stretch_mute_pct.value,
            "clip_percentile": clip_percentile.value,
            "filter": {
                "type": filter_type.value,
                "low_hz": active_filter_low_hz,
                "high_hz": active_filter_high_hz,
                "stage_raw": filter_at_raw.value,
                "stage_post_nmo": filter_at_postnmo.value,
                "stage_stack": filter_at_stack.value,
            },
            "velocity_model": {
                "path": vel_model_path.value,
                "native_type": velocity_model_type.value,
                "units_in": model_velocity_units.value,
                "domain": model_domain_toggle.value,
                "sample_interval": model_sample_interval.value,
                "units_out": output_velocity_units.value,
            } if _want_nmo else None,
        })
    return


@app.cell
def _(get_full_stack, mo):
    full_stack = get_full_stack()
    mo.stop(
        full_stack is None,
        mo.md("*Click **Stack full data** above -- can take a while for the whole survey.*"),
    )
    mo.md(f"Full stack ready: **{len(full_stack['cdp']):,} CDPs** stacked.")
    return (full_stack,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Stacked sections (IL section left, XL section right)

    Enter a fixed IL (for the inline section, varying XL) and a fixed
    XL (for the crossline section, varying IL) -- both are shown side
    by side, sharing the same time axis.
    """)
    return


@app.cell
def _(cdp_il, cdp_xl, mo, np):
    # Default to the most common IL/XL rather than 0 -- 0 is never a
    # real IL/XL value, so it would always show "no CDPs found" until
    # the user picks something themselves.
    _u_il, _il_counts = np.unique(cdp_il, return_counts=True)
    _default_il = int(_u_il[np.argmax(_il_counts)])
    _u_xl, _xl_counts = np.unique(cdp_xl, return_counts=True)
    _default_xl = int(_u_xl[np.argmax(_xl_counts)])

    section_il_value = mo.ui.number(value=_default_il, label="Fixed IL (inline section)")
    section_xl_value = mo.ui.number(value=_default_xl, label="Fixed XL (crossline section)")
    mo.hstack([section_il_value, section_xl_value])
    return section_il_value, section_xl_value


@app.cell
def _(mo):
    section_velocity_type = mo.ui.radio(
        options=["RMS", "Interval"],
        value="RMS", inline=True,
        label="Velocity model sections/overlay below should plot",
    )
    section_velocity_type
    return (section_velocity_type,)


@app.cell
def _(
    clip_percentile,
    full_stack,
    full_t,
    mo,
    np,
    plt,
    qc_plot_dir,
    section_il_value,
    section_xl_value,
):
    mo.stop(
        full_stack is None,
        mo.md("*Click **Stack full data** above -- can take a while for the whole survey.*"),
    )

    def _extract(fixed_arr, vary_arr, fixed_value):
        _mask = fixed_arr == int(fixed_value)
        if not _mask.any():
            return None
        _vary_vals = vary_arr[_mask]
        _section = full_stack["stack"][_mask]
        _order_idx = np.argsort(_vary_vals)
        return _vary_vals[_order_idx], _section[_order_idx]

    _il_result = _extract(full_stack["il"], full_stack["xl"], section_il_value.value)
    _xl_result = _extract(full_stack["xl"], full_stack["il"], section_xl_value.value)

    mo.stop(
        _il_result is None and _xl_result is None,
        mo.callout(
            f"No stacked CDPs found with IL={int(section_il_value.value)} "
            f"or XL={int(section_xl_value.value)}.",
            kind="warn",
        ),
    )

    _y_max_ms = full_t[-1] * 1000.0
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 6.5), gridspec_kw={"width_ratios": [60, 40]}, sharey=True)

    for _ax, _result, _fixed_label, _fixed_val, _vary_label in (
        (_axes[0], _il_result, "IL", section_il_value.value, "XL"),
        (_axes[1], _xl_result, "XL", section_xl_value.value, "IL"),
    ):
        if _result is None:
            _ax.text(0.5, 0.5, f"No CDPs at {_fixed_label}={int(_fixed_val)}", ha="center", va="center", transform=_ax.transAxes)
            _ax.set_title(f"{_fixed_label}={int(_fixed_val)} (empty)")
            continue
        _vary_vals, _section = _result
        _vclip = np.percentile(np.abs(_section), clip_percentile.value) or 1.0
        _ax.imshow(
            _section.T, aspect="auto", cmap="gray",
            vmin=-_vclip, vmax=_vclip,
            extent=[_vary_vals[0], _vary_vals[-1], _y_max_ms, 0],
        )
        _ax.set_title(f"{_fixed_label}={int(_fixed_val)}, {len(_vary_vals)} CDPs")
        _ax.set_xlabel(_vary_label)

    _axes[0].set_ylim(_y_max_ms, 0)
    _axes[0].set_ylabel("Time (ms)")

    _fig.tight_layout()
    _p = qc_plot_dir / "stacked_sections.png"
    _fig.savefig(_p, dpi=130)
    plt.close(_fig)
    mo.vstack([
        mo.image(src=str(_p)),
        mo.download(data=_p.read_bytes(), filename="stacked_sections.png", mimetype="image/png", label="💾 Save this figure"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Velocity model sections (same IL/XL, same time axis)

    The model's own velocity along the same two lines, for a direct
    comparison against the stacked sections above. Gray = beyond the
    model's own recorded range (no data there, not zero velocity).
    """)
    return


@app.cell
def _(
    full_stack,
    full_t,
    mo,
    model_velocity_for,
    np,
    plt,
    qc_plot_dir,
    section_il_value,
    section_velocity_type,
    section_xl_value,
):
    mo.stop(full_stack is None, mo.md(""))
    mo.stop(
        model_velocity_for is None,
        mo.callout("No velocity model loaded -- nothing to show here.", kind="info"),
    )

    def _extract_positions(fixed_arr, vary_arr, fixed_value):
        _mask = fixed_arr == int(fixed_value)
        if not _mask.any():
            return None
        _vary_vals = vary_arr[_mask]
        _x_vals = full_stack["x"][_mask]
        _y_vals = full_stack["y"][_mask]
        _order_idx = np.argsort(_vary_vals)
        return _vary_vals[_order_idx], _x_vals[_order_idx], _y_vals[_order_idx]

    def _build_model_section(positions):
        if positions is None:
            return None
        _vary_vals, _x_vals, _y_vals = positions
        _section = np.empty((len(_vary_vals), len(full_t)), dtype=np.float64)
        for _i in range(len(_vary_vals)):
            _vel, _match = model_velocity_for(_x_vals[_i], _y_vals[_i])
            _section[_i] = _vel if section_velocity_type.value == "RMS" else _match["interval_of_t0"]
            # Beyond the model's own recorded range, model_velocity_for
            # just holds the last known value flat -- mark that as
            # missing (gray) rather than showing it as real data.
            _section[_i][full_t > _match["model_t"][-1]] = np.nan
        return _vary_vals, _section

    _il_positions = _extract_positions(full_stack["il"], full_stack["xl"], section_il_value.value)
    _xl_positions = _extract_positions(full_stack["xl"], full_stack["il"], section_xl_value.value)
    _il_model = _build_model_section(_il_positions)
    _xl_model = _build_model_section(_xl_positions)

    mo.stop(
        _il_model is None and _xl_model is None,
        mo.callout(
            f"No stacked CDPs found with IL={int(section_il_value.value)} "
            f"or XL={int(section_xl_value.value)}.",
            kind="warn",
        ),
    )

    # Same time axis as the stacked sections above, so the two figures
    # line up exactly when compared side by side.
    _y_max_ms = full_t[-1] * 1000.0
    _cmap = plt.get_cmap("viridis").copy()
    _cmap.set_bad(color="lightgray")

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 6.5), gridspec_kw={"width_ratios": [60, 40]}, sharey=True)
    _im = None
    for _ax, _result, _fixed_label, _fixed_val, _vary_label in (
        (_axes[0], _il_model, "IL", section_il_value.value, "XL"),
        (_axes[1], _xl_model, "XL", section_xl_value.value, "IL"),
    ):
        if _result is None:
            _ax.text(0.5, 0.5, f"No CDPs at {_fixed_label}={int(_fixed_val)}", ha="center", va="center", transform=_ax.transAxes)
            _ax.set_title(f"{_fixed_label}={int(_fixed_val)} (empty)")
            continue
        _vary_vals, _section = _result
        _im = _ax.imshow(
            _section.T, aspect="auto", cmap=_cmap,
            extent=[_vary_vals[0], _vary_vals[-1], _y_max_ms, 0],
        )
        _ax.set_title(f"{_fixed_label}={int(_fixed_val)} {section_velocity_type.value.lower()} velocity")
        _ax.set_xlabel(_vary_label)

    _axes[0].set_ylim(_y_max_ms, 0)
    _axes[0].set_ylabel("Time (ms)")

    # Small horizontal colorbar inset INSIDE the IL plot itself
    # (bottom-right corner, axes-relative coordinates) -- doesn't
    # consume any extra figure space, so this figure's size stays
    # identical to the stacked-sections figure above for a direct
    # side-by-side comparison. No legend/label text, just the value
    # ticks.
    if _im is not None:
        _cax = _axes[0].inset_axes([0.55, 0.06, 0.4, 0.035])
        _cb = _fig.colorbar(_im, cax=_cax, orientation="horizontal")
        _cb.ax.tick_params(labelsize=7)
        _cb.outline.set_linewidth(0.5)

    _fig.tight_layout()
    _p = qc_plot_dir / "model_sections.png"
    _fig.savefig(_p, dpi=130)
    plt.close(_fig)
    mo.vstack([
        mo.image(src=str(_p)),
        mo.download(data=_p.read_bytes(), filename="model_sections.png", mimetype="image/png", label="💾 Save this figure"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    #### Overlay: stacked section + velocity model

    The velocity model in color, laid over the grayscale stack, so
    velocity trends can be checked directly against the reflectors
    they should track. Where the model has no data, the stack shows
    through with no color on top.
    """)
    return


@app.cell
def _(
    clip_percentile,
    full_stack,
    full_t,
    mo,
    model_velocity_for,
    np,
    plt,
    qc_plot_dir,
    section_il_value,
    section_velocity_type,
    section_xl_value,
):
    mo.stop(full_stack is None, mo.md(""))
    mo.stop(
        model_velocity_for is None,
        mo.callout("No velocity model loaded -- nothing to overlay.", kind="info"),
    )

    def _extract(fixed_arr, vary_arr, fixed_value):
        _mask = fixed_arr == int(fixed_value)
        if not _mask.any():
            return None
        _vary_vals = vary_arr[_mask]
        _stack_section = full_stack["stack"][_mask]
        _x_vals = full_stack["x"][_mask]
        _y_vals = full_stack["y"][_mask]
        _order_idx = np.argsort(_vary_vals)
        return (
            _vary_vals[_order_idx],
            _stack_section[_order_idx],
            _x_vals[_order_idx],
            _y_vals[_order_idx],
        )

    def _build_overlay(result):
        if result is None:
            return None
        _vary_vals, _stack_section, _x_vals, _y_vals = result
        _vel_section = np.empty((len(_vary_vals), len(full_t)), dtype=np.float64)
        for _i in range(len(_vary_vals)):
            _vel, _match = model_velocity_for(_x_vals[_i], _y_vals[_i])
            _vel = _vel if section_velocity_type.value == "RMS" else _match["interval_of_t0"]
            _vel_section[_i] = _vel
            # Beyond the model's own recorded range -- make it NaN so
            # the color layer goes fully transparent there instead of
            # painting a flat, made-up value on top of the stack.
            _vel_section[_i][full_t > _match["model_t"][-1]] = np.nan
        return _vary_vals, _stack_section, _vel_section

    _il_result = _extract(full_stack["il"], full_stack["xl"], section_il_value.value)
    _xl_result = _extract(full_stack["xl"], full_stack["il"], section_xl_value.value)
    _il_overlay = _build_overlay(_il_result)
    _xl_overlay = _build_overlay(_xl_result)

    mo.stop(
        _il_overlay is None and _xl_overlay is None,
        mo.callout(
            f"No stacked CDPs found with IL={int(section_il_value.value)} "
            f"or XL={int(section_xl_value.value)}.",
            kind="warn",
        ),
    )

    # Same time axis and 60:40 layout as the two figures above, so all
    # three line up for a direct comparison.
    _y_max_ms = full_t[-1] * 1000.0
    _cmap = plt.get_cmap("viridis").copy()
    _cmap.set_bad(alpha=0.0)

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 6.5), gridspec_kw={"width_ratios": [60, 40]}, sharey=True)
    _im = None
    for _ax, _result, _fixed_label, _fixed_val, _vary_label in (
        (_axes[0], _il_overlay, "IL", section_il_value.value, "XL"),
        (_axes[1], _xl_overlay, "XL", section_xl_value.value, "IL"),
    ):
        if _result is None:
            _ax.text(0.5, 0.5, f"No CDPs at {_fixed_label}={int(_fixed_val)}", ha="center", va="center", transform=_ax.transAxes)
            _ax.set_title(f"{_fixed_label}={int(_fixed_val)} (empty)")
            continue
        _vary_vals, _stack_section, _vel_section = _result
        _extent = [_vary_vals[0], _vary_vals[-1], _y_max_ms, 0]
        _vclip = np.percentile(np.abs(_stack_section), clip_percentile.value) or 1.0
        _ax.imshow(
            _stack_section.T, aspect="auto", cmap="gray",
            vmin=-_vclip, vmax=_vclip, extent=_extent,
        )
        _im = _ax.imshow(
            _vel_section.T, aspect="auto", cmap=_cmap, alpha=0.5,
            extent=_extent,
        )
        _ax.set_title(f"{_fixed_label}={int(_fixed_val)} + {section_velocity_type.value.lower()} vel, {len(_vary_vals)} CDPs")
        _ax.set_xlabel(_vary_label)

    _axes[0].set_ylim(_y_max_ms, 0)
    _axes[0].set_ylabel("Time (ms)")

    # Same inset colorbar placement as the velocity model figure above
    # -- inside the IL plot, bottom-right, no legend text.
    if _im is not None:
        _cax = _axes[0].inset_axes([0.55, 0.06, 0.4, 0.035])
        _cb = _fig.colorbar(_im, cax=_cax, orientation="horizontal")
        _cb.ax.tick_params(labelsize=7)
        _cb.outline.set_linewidth(0.5)

    _fig.tight_layout()
    _p = qc_plot_dir / "overlay_sections.png"
    _fig.savefig(_p, dpi=130)
    plt.close(_fig)
    mo.vstack([
        mo.image(src=str(_p)),
        mo.download(data=_p.read_bytes(), filename="overlay_sections.png", mimetype="image/png", label="💾 Save this figure"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 8. Export the full stack

    Writes the full stacked volume from step 7 (one zero-offset trace
    per CDP) to a new SEG-Y file -- IEEE float32 samples, CDP number
    and INLINE_3D/CROSSLINE_3D (bytes 189-196, same convention this
    tool itself reads for the velocity model) in each trace header.
    """)
    return


@app.cell
def _(mo):
    export_stack_button = mo.ui.run_button(label="💾 Export full stack to SEG-Y")
    export_stack_button
    return (export_stack_button,)


@app.cell
def _(
    export_stack_button,
    full_stack,
    log_job,
    mo,
    probe_info,
    qc_plot_dir,
    sgy_path,
    struct,
    time,
):
    mo.stop(full_stack is None, mo.md("*Stack the full data (step 7) before exporting.*"))
    mo.stop(not export_stack_button.value, mo.md("*Click **Export full stack to SEG-Y** above.*"))

    _t_start = time.time()
    _order = probe_info["order"]
    _ns = full_stack["stack"].shape[1]
    _dt_us = probe_info["dt_us"]
    _n_cdp = len(full_stack["cdp"])

    _p = qc_plot_dir / "full_stack.sgy"
    with mo.status.progress_bar(
        total=_n_cdp, title="Writing SEG-Y...", subtitle=str(_p),
    ) as _bar, open(_p, "wb") as _f:
        _f.write(b"\x00" * 3200)
        _bin_hdr = bytearray(400)
        struct.pack_into(_order + "H", _bin_hdr, 16, _dt_us)
        struct.pack_into(_order + "H", _bin_hdr, 20, _ns)
        struct.pack_into(_order + "h", _bin_hdr, 24, 5)  # format 5 = IEEE float32
        _f.write(bytes(_bin_hdr))
        for _i in range(_n_cdp):
            _hdr = bytearray(240)
            struct.pack_into(_order + "i", _hdr, 0, _i + 1)  # tracl
            struct.pack_into(_order + "i", _hdr, 20, int(full_stack["cdp"][_i]))
            struct.pack_into(_order + "i", _hdr, 36, 0)  # offset -- stacked, zero-offset
            struct.pack_into(_order + "H", _hdr, 114, _ns)
            struct.pack_into(_order + "H", _hdr, 116, _dt_us)
            struct.pack_into(_order + "i", _hdr, 188, int(full_stack["il"][_i]))
            struct.pack_into(_order + "i", _hdr, 192, int(full_stack["xl"][_i]))
            struct.pack_into(_order + "i", _hdr, 204, int(round(full_stack["x"][_i] * 100)))
            struct.pack_into(_order + "i", _hdr, 208, int(round(full_stack["y"][_i] * 100)))
            _f.write(bytes(_hdr))
            _f.write(full_stack["stack"][_i].astype(_order + "f4").tobytes())
            _bar.update()

    _size_mb = _p.stat().st_size / 1e6
    log_job({
        "job": "export_sgy",
        "sgy_path": sgy_path.value,
        "output_path": str(_p),
        "n_cdp": int(_n_cdp),
        "elapsed_s": round(time.time() - _t_start, 2),
        "file_size_mb": round(_size_mb, 1),
    })
    _msgs = [mo.md(f"Wrote **{_p}** -- {_n_cdp:,} traces, {_size_mb:,.1f} MB.")]
    if _size_mb <= 300:
        _msgs.append(
            mo.download(
                data=_p.read_bytes(), filename="full_stack.sgy",
                mimetype="application/octet-stream", label="⬇️ Download full_stack.sgy",
            )
        )
    else:
        _msgs.append(
            mo.callout(
                f"File is {_size_mb:,.0f} MB -- too large to also offer as a browser "
                "download; use the saved path above instead.",
                kind="info",
            )
        )
    mo.vstack(_msgs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 9. Job log

    Every **Stack full data** and **Export full stack to SEG-Y** run is
    recorded here -- input file, filter settings, velocity model, and
    timing -- so past QC runs stay auditable.
    """)
    return


@app.cell
def _(json, mo, qc_plot_dir):
    _log_path = qc_plot_dir / "job_log.jsonl"
    if _log_path.exists():
        _rows = [json.loads(_line) for _line in _log_path.read_text().splitlines() if _line.strip()]
        _rows = _rows[-20:][::-1]
        _display_rows = [
            {
                "timestamp": _r.get("timestamp"),
                "job": _r.get("job"),
                "sgy_path": _r.get("sgy_path"),
                "n_cdp": _r.get("n_cdp"),
                "elapsed_s": _r.get("elapsed_s"),
                "apply_nmo": _r.get("apply_nmo"),
                "stretch_mute_pct": _r.get("stretch_mute_pct"),
                "clip_percentile": _r.get("clip_percentile"),
                "filter": json.dumps(_r["filter"]) if _r.get("filter") else None,
                "velocity_model": json.dumps(_r["velocity_model"]) if _r.get("velocity_model") else None,
                "output_path": _r.get("output_path"),
                "file_size_mb": _r.get("file_size_mb"),
            }
            for _r in _rows
        ]
        _out = mo.ui.table(_display_rows, selection=None) if _display_rows else mo.md("*Log is empty.*")
    else:
        _out = mo.md(
            "*No jobs run yet -- log will appear here after the first "
            "**Stack full data** or **Export full stack to SEG-Y** run.*"
        )
    _out
    return


if __name__ == "__main__":
    app.run()
