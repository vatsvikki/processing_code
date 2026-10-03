import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sc_scaling_pipeline import build_dtypes, HEADER_BYTES, NS_OFF, read_header_field

    return (
        HEADER_BYTES,
        NS_OFF,
        Path,
        build_dtypes,
        go,
        json,
        mo,
        np,
        pd,
        read_header_field,
    )


@app.cell
def _(Path, json):
    # Inputs saved with the "Save these inputs" button (grid, file format, header IL/XL bytes), one entry per input
    # file, in a JSON file next to this notebook. The last file saved is opened first next time, and a file with no
    # entry of its own starts from the last file's grid.
    SETTINGS_FILE = Path(__file__).resolve().parent / "cdp_from_header_settings.json"
    DEFAULT_FILE = "/Users/vikas/SCube/harrison/data/raw_data/EAST_ATCHAFALAYA-GCSR-UNDEFINED_SDL062470-16.su"
    BUILTIN_GRID = {"il_min": 951, "il_max": 1404, "xl_min": 951, "xl_max": 1686,
                    "x0": 1971388.1, "y0": 428622.0, "il_bin": 150.0, "xl_bin": 150.0}

    def _key(path):
        return str(Path(path).expanduser().resolve()) if str(path).strip() else ""

    def read_settings():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except (OSError, ValueError):
            return {}

    def settings_for(path):
        """(saved entry for this file, where the values shown come from)."""
        _s = read_settings()
        _files = _s.get("files", {})
        if _key(path) in _files:
            return _files[_key(path)], "saved for this file"
        _last = _s.get("last_file", "")
        if _last in _files:
            return _files[_last], f"nothing saved for this file yet - starting from the inputs saved for `{Path(_last).name}`"
        return {}, "nothing saved yet - built-in defaults"

    def save_settings(path, entry):
        _s = read_settings()
        _s.setdefault("files", {})[_key(path)] = entry
        _s["last_file"] = _key(path)
        _tmp = SETTINGS_FILE.with_suffix(".tmp")
        _tmp.write_text(json.dumps(_s, indent=2))
        _tmp.replace(SETTINGS_FILE)

    return BUILTIN_GRID, DEFAULT_FILE, SETTINGS_FILE, read_settings, save_settings, settings_for


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # CDP from header

    Four plots, all the same size, colour bar drawn inside each figure:

    1. **CDP from header** — scatter, positioned by header coordinates, coloured by header CDP.
    2. **CDP from header** — map, positioned by IL/XL *calculated from the corner points* (the header's own
       IL/XL usually isn't populated, so this is what makes a gridded header-CDP map possible at all).
    3. **CDP recalculated** — scatter, positioned by `CalcCDPX/CalcCDPY`, coloured by the recalculated CDP.
    4. **CDP recalculated** — map, positioned by IL/XL calculated from `CalcCDPX/CalcCDPY`, coloured by the
       recalculated CDP.

    Accepts either **SU** (`.su`, no file header — trace 1 starts at byte 0) or **SEG-Y** (`.sgy`/`.segy`, a
    3200-byte textual + 400-byte binary file header — 3600 bytes — before trace 1). Set the path and format
    below, then click **Load data**.
    """)
    return


@app.cell
def _(DEFAULT_FILE, mo, read_settings):
    import os

    # $CDP_INPUT_FILE (set by launch_cdp.sh / launch_stack.sh) wins, then the last file saved on this machine
    su_path = mo.ui.text(
        label="Input file path (.su or .sgy/.segy)",
        value=os.environ.get("CDP_INPUT_FILE", "").strip() or read_settings().get("last_file") or DEFAULT_FILE,
        full_width=True,
    )
    su_path
    return (su_path,)


@app.cell
def _(settings_for, su_path):
    # what was saved for the file in the box (re-read whenever the path changes)
    saved, saved_note = settings_for(su_path.value)
    return saved, saved_note


@app.cell
def _(mo, saved):
    _formats = ["Auto (by file extension)", "SU — no file header", "SEG-Y — 3600-byte file header"]
    file_format = mo.ui.dropdown(
        options=_formats,
        value=saved.get("file_format") if saved.get("file_format") in _formats else _formats[0],
        label="File format",
    )
    load_button = mo.ui.run_button(label="📂 Load data", kind="success")
    mo.hstack([file_format, load_button], justify="start", gap=1)
    return file_format, load_button


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Where the header IL / XL come from.** There is no single universal byte position for these words —
    SEG-Y rev1 puts them at bytes **189** (inline) / **193** (crossline), 1-based, 4-byte integers each. Some
    contractors' own header-schema documents place them elsewhere. Set the byte position below to whatever
    your file's own documentation says (used only for the hover / table columns — plots 2 and 4 always use
    the corner-point-calculated IL/XL instead, since header IL/XL is often not populated at all).
    """)
    return


@app.cell
def _(mo, saved):
    def _byte(name, default):
        return min(237, max(119, int(saved.get(name, default))))

    il_byte = mo.ui.number(start=119, stop=237, step=1, value=_byte("il_byte", 189), label="Header INLINE byte (1-based)")
    xl_byte = mo.ui.number(start=119, stop=237, step=1, value=_byte("xl_byte", 193), label="Header CROSSLINE byte (1-based)")
    mo.hstack([il_byte, xl_byte])
    return il_byte, xl_byte


@app.cell
def _(
    HEADER_BYTES,
    NS_OFF,
    Path,
    build_dtypes,
    file_format,
    il_byte,
    load_button,
    mo,
    np,
    read_header_field,
    su_path,
    xl_byte,
):
    mo.stop(not load_button.value, mo.md("👆 Set the path and format above, then click **Load data**."))

    _path = Path(su_path.value)
    if not _path.exists():
        raise FileNotFoundError(f"File not found: {_path}")

    # SU has no file header (trace 1 starts at byte 0). SEG-Y has a 3200-byte textual + 400-byte binary file
    # header (3600 bytes) before trace 1 - everything after that follows the same per-trace layout build_dtypes()
    # already describes, so skipping 3600 bytes is all SEG-Y support needs here.
    if file_format.value.startswith("SU"):
        _header_offset = 0
    elif file_format.value.startswith("SEG-Y"):
        _header_offset = 3600
    else:
        _header_offset = 3600 if _path.suffix.lower() in (".sgy", ".segy") else 0

    with mo.status.spinner(title=f"Scanning whole survey headers ({'SEG-Y' if _header_offset else 'SU'} format)..."):
        _size = _path.stat().st_size
        with open(_path, "rb") as _f:
            _f.seek(_header_offset)
            _first = _f.read(HEADER_BYTES)
            if len(_first) < HEADER_BYTES:
                raise RuntimeError(
                    f"File shorter than one trace header after skipping the {_header_offset}-byte file header. "
                    "Wrong file format selected above?"
                )
            _order = None
            for _cand in (">", "<"):
                _ns_try = read_header_field(_first, _cand, "H", NS_OFF)
                if 1 <= _ns_try <= 20000:
                    _order = _cand
                    break
            if _order is None:
                raise RuntimeError(
                    "Could not determine byte order (ns field unreasonable in both endiannesses) - try switching "
                    "the File format above."
                )
            _ns = read_header_field(_first, _order, "H", NS_OFF)

        _, _full_dtype = build_dtypes(_order, _ns)
        _trace_bytes = HEADER_BYTES + _ns * 4
        _data_size = _size - _header_offset
        if _data_size <= 0 or _data_size % _trace_bytes != 0:
            raise RuntimeError(
                f"File size after the {_header_offset}-byte file header ({_data_size:,} bytes) is not an exact "
                f"multiple of the trace size ({_trace_bytes:,} bytes, ns={_ns}). Wrong file format selected above?"
            )
        _n = _data_size // _trace_bytes

        _sx = np.empty(_n, dtype=np.int64)
        _sy = np.empty(_n, dtype=np.int64)
        _gx = np.empty(_n, dtype=np.int64)
        _gy = np.empty(_n, dtype=np.int64)
        _cdp = np.empty(_n, dtype=np.int64)
        _il_hdr = np.empty(_n, dtype=np.int64)
        _xl_hdr = np.empty(_n, dtype=np.int64)

        # build_dtypes() only extracts SX/SY/GX/GY/FLDR/CDP/etc.; everything from byte 119 (1-based) onward is
        # an opaque "pad2" void field. Re-interpret that same already-read block to pull out the header's own
        # inline / crossline words at the chosen byte position: pad2 starts at byte 119 (1-based), so a 4-byte
        # field starting at 1-based byte B sits at relative offset (B - 1) - 118 within it.
        _il_rel = (il_byte.value - 1) - 118
        _xl_rel = (xl_byte.value - 1) - 118
        _pad2_dtype = np.dtype({"names": ["il", "xl"], "formats": [_order + "i4", _order + "i4"],
                                "offsets": [_il_rel, _xl_rel], "itemsize": 122})
        _done = 0
        with open(_path, "rb") as _f:
            _f.seek(_header_offset)
            _remaining = _n
            while _remaining > 0:
                _take = min(20000, _remaining)
                _chunk = np.fromfile(_f, dtype=_full_dtype, count=_take)
                _h = _chunk["header"]
                _sx[_done:_done + _take] = _h["sx"]
                _sy[_done:_done + _take] = _h["sy"]
                _gx[_done:_done + _take] = _h["gx"]
                _gy[_done:_done + _take] = _h["gy"]
                _cdp[_done:_done + _take] = _h["cdp"]
                _il_xl = np.frombuffer(_h["pad2"].tobytes(), dtype=_pad2_dtype)
                _il_hdr[_done:_done + _take] = _il_xl["il"]
                _xl_hdr[_done:_done + _take] = _il_xl["xl"]
                _done += _take
                _remaining -= _take

    header_scan = {"sx": _sx, "sy": _sy, "gx": _gx, "gy": _gy, "cdp": _cdp, "il_hdr": _il_hdr, "xl_hdr": _xl_hdr}
    file_info = {"path": _path, "header_offset": _header_offset, "order": _order, "ns": _ns,
                "full_dtype": _full_dtype, "trace_bytes": _trace_bytes, "n_traces": _n}
    mo.md(
        f"**{_n:,} traces scanned** ({'SEG-Y, 3600-byte file header skipped' if _header_offset else 'SU, no file header'}; "
        f"header INLINE from byte {il_byte.value}, CROSSLINE from byte {xl_byte.value})."
    )
    return file_info, header_scan


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Grid definition & corner points

    Enter the survey's IL / XL grid (axis-aligned, no rotation): the IL / XL range, the X / Y of the first
    corner (min IL, min XL) and the bin size. IL runs along Y, XL along X. **💾 Save these inputs** keeps them —
    together with the file format and header IL / XL bytes above — for this input file, and they are filled in
    again the next time the notebook is opened.
    """)
    return


@app.cell
def _(BUILTIN_GRID, mo, saved):
    _g = {**BUILTIN_GRID, **saved.get("grid", {})}
    grid_form = mo.ui.dictionary({
        "il_min": mo.ui.number(value=int(_g["il_min"]), step=1, label="IL min"),
        "il_max": mo.ui.number(value=int(_g["il_max"]), step=1, label="IL max"),
        "xl_min": mo.ui.number(value=int(_g["xl_min"]), step=1, label="XL min"),
        "xl_max": mo.ui.number(value=int(_g["xl_max"]), step=1, label="XL max"),
        "x0": mo.ui.number(value=float(_g["x0"]), step=0.1, label="X at (IL min, XL min)"),
        "y0": mo.ui.number(value=float(_g["y0"]), step=0.1, label="Y at (IL min, XL min)"),
        "il_bin": mo.ui.number(value=float(_g["il_bin"]), step=0.01, label="IL bin size (along Y)"),
        "xl_bin": mo.ui.number(value=float(_g["xl_bin"]), step=0.01, label="XL bin size (along X)"),
    })
    save_button = mo.ui.run_button(label="💾 Save these inputs", kind="success")
    return grid_form, save_button


@app.cell
def _(grid_form, mo, save_button, saved_note):
    mo.vstack([
        mo.hstack([grid_form["il_min"], grid_form["il_max"], grid_form["xl_min"], grid_form["xl_max"]], justify="start"),
        mo.hstack([grid_form["x0"], grid_form["y0"], grid_form["il_bin"], grid_form["xl_bin"]], justify="start"),
        mo.hstack([save_button, mo.md(f"<small>Values shown: {saved_note}.</small>")], justify="start", align="center"),
    ])
    return


@app.cell
def _(grid_form, mo, pd):
    _v = grid_form.value
    _bad = [_m for _ok, _m in [
        (_v["il_max"] >= _v["il_min"], "IL max must be ≥ IL min"),
        (_v["xl_max"] >= _v["xl_min"], "XL max must be ≥ XL min"),
        (_v["il_bin"] > 0 and _v["xl_bin"] > 0, "bin sizes must be > 0"),
    ] if not _ok]
    mo.stop(bool(_bad), mo.callout(mo.md("**Grid not valid:** " + "; ".join(_bad) + "."), kind="danger"))

    grid = {"il_min": int(_v["il_min"]), "il_max": int(_v["il_max"]), "xl_min": int(_v["xl_min"]),
            "xl_max": int(_v["xl_max"]), "x0": float(_v["x0"]), "y0": float(_v["y0"]),
            "il_bin": float(_v["il_bin"]), "xl_bin": float(_v["xl_bin"])}
    grid["n_xl"] = grid["xl_max"] - grid["xl_min"] + 1
    _x1 = grid["x0"] + (grid["xl_max"] - grid["xl_min"]) * grid["xl_bin"]
    _y1 = grid["y0"] + (grid["il_max"] - grid["il_min"]) * grid["il_bin"]

    # the equations with the entered numbers - also quoted in the plot descriptions below
    grid_eq = (
        "CalcCDPX = (SrcX + RecX) / 2 = (SX + GX) / 2\n"
        "CalcCDPY = (SrcY + RecY) / 2 = (SY + GY) / 2\n"
        f"IL       = round({grid['il_min']} + (CalcCDPY - {grid['y0']:.10g}) / {grid['il_bin']:.10g})\n"
        f"XL       = round({grid['xl_min']} + (CalcCDPX - {grid['x0']:.10g}) / {grid['xl_bin']:.10g})\n"
        f"N_XL     = {grid['xl_max']} - {grid['xl_min']} + 1 = {grid['n_xl']}\n"
        "CDP      = IL * N_XL + XL"
    )
    # the four corners of the entered grid, going round: (min IL, min XL) -> (min IL, max XL) -> (max, max) -> (max, min)
    corner_points_df = pd.DataFrame({
        "Corner": [1, 2, 3, 4],
        "IL": [grid["il_min"], grid["il_min"], grid["il_max"], grid["il_max"]],
        "XL": [grid["xl_min"], grid["xl_max"], grid["xl_max"], grid["xl_min"]],
        "X": [round(_v, 2) for _v in (grid["x0"], _x1, _x1, grid["x0"])],
        "Y": [round(_v, 2) for _v in (grid["y0"], grid["y0"], _y1, _y1)],
    })

    def _make_corners_csv():
        return corner_points_df.to_csv(index=False).encode("utf-8")

    mo.vstack([
        mo.md(f"### Corner points\n{grid['il_max'] - grid['il_min'] + 1:,} inlines × {grid['n_xl']:,} crosslines, "
              f"bins {grid['il_bin']:.10g} (IL) × {grid['xl_bin']:.10g} (XL)"),
        mo.ui.table(corner_points_df, selection=None, pagination=False, show_column_summaries=False,
                    format_mapping={"X": "{:,.2f}", "Y": "{:,.2f}"}),
        mo.download(data=_make_corners_csv, filename="corner_points.csv", mimetype="text/csv",
                    label="💾 Save corner points as CSV"),
        mo.md(f"```\n{grid_eq}\n```"),
    ])
    return corner_points_df, grid, grid_eq


@app.cell
def _(
    SETTINGS_FILE,
    file_format,
    grid,
    il_byte,
    mo,
    save_button,
    save_settings,
    su_path,
    xl_byte,
):
    mo.stop(not save_button.value)
    save_settings(su_path.value, {
        "file_format": file_format.value, "il_byte": int(il_byte.value), "xl_byte": int(xl_byte.value),
        "grid": {_k: _v for _k, _v in grid.items() if _k != "n_xl"},
    })
    mo.md(f"✅ Saved the grid, file format and header IL / XL bytes for `{su_path.value}` in `{SETTINGS_FILE}`.")
    return


@app.cell
def _(grid, header_scan, mo, np):
    # Cheap vectorised math (no file I/O) - used both to position plot 2 (header CDP on a working grid) and to
    # produce the fully recalculated CDP for plots 3 and 4. The grid comes from the inputs above.
    calc_cdp_x = (header_scan["sx"].astype(np.float64) + header_scan["gx"].astype(np.float64)) / 2.0
    calc_cdp_y = (header_scan["sy"].astype(np.float64) + header_scan["gy"].astype(np.float64)) / 2.0
    calc_il = np.round(grid["il_min"] + (calc_cdp_y - grid["y0"]) / grid["il_bin"]).astype(np.int64)
    calc_xl = np.round(grid["xl_min"] + (calc_cdp_x - grid["x0"]) / grid["xl_bin"]).astype(np.int64)
    cdp_calculated = calc_il * grid["n_xl"] + calc_xl

    _outside = int(np.count_nonzero((calc_il < grid["il_min"]) | (calc_il > grid["il_max"])
                                    | (calc_xl < grid["xl_min"]) | (calc_xl > grid["xl_max"])))
    mo.callout(
        mo.md(f"**{_outside:,} of {len(calc_il):,} traces** fall outside IL {grid['il_min']}–{grid['il_max']} / "
              f"XL {grid['xl_min']}–{grid['xl_max']} - check the grid inputs."),
        kind="warn",
    ) if _outside else mo.md(f"All {len(calc_il):,} traces fall inside the grid.")
    return calc_cdp_x, calc_cdp_y, calc_il, calc_xl, cdp_calculated


@app.cell
def _():
    # Shared figure sizing / colour-bar style so all four plots below come out the same size - the colour bar
    # is drawn inside the plot area (inset), not appended to the right of it, so it never changes the figure's
    # overall width the way plotly's default colorbar placement would.
    FIG_WIDTH, FIG_HEIGHT = 900, 600

    def inset_colorbar():
        return dict(title=dict(text="CDP", font=dict(size=11)), x=0.98, xanchor="right", y=0.98, yanchor="top",
                    len=0.4, thickness=14, outlinewidth=0, bgcolor="rgba(255,255,255,0.65)")

    return FIG_HEIGHT, FIG_WIDTH, inset_colorbar


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 1. CDP from header — scatter, positioned by header coordinates

    **What this shows.** One point per trace. Every point is placed at that trace's source-receiver midpoint,
    computed straight from the raw coordinate bytes in the header (`SX`, `SY`, `GX`, `GY` — SEG-Y bytes 73-76,
    77-80, 81-84, 85-88). The point's colour is the CDP number already stored in that same trace's header
    (SEG-Y bytes 21-24) — **nothing on this plot is calculated**, every value plotted is read directly from
    the file. Hover any point to see its header IL, XL, CDP and raw SX/SY/GX/GY.

    **Equation used:**
    ```
    X (plot position) = (SrcX + RecX) / 2 = (SX + GX) / 2
    Y (plot position) = (SrcY + RecY) / 2 = (SY + GY) / 2
    Colour             = CDP        (read directly from the header, byte 21-24)
    ```
    """)
    return


@app.cell
def _(FIG_HEIGHT, FIG_WIDTH, go, header_scan, inset_colorbar, mo, np):
    # Full resolution (every trace) - used only for the PNG export, never sent to the browser as interactive
    # figure data (~950k points x 7 customdata columns is too large a payload for marimo's live output).
    _x_full = (header_scan["sx"].astype(np.float64) + header_scan["gx"].astype(np.float64)) / 2.0
    _y_full = (header_scan["sy"].astype(np.float64) + header_scan["gy"].astype(np.float64)) / 2.0
    _custom_full = np.stack([header_scan["il_hdr"], header_scan["xl_hdr"], header_scan["cdp"],
                             header_scan["sx"], header_scan["sy"], header_scan["gx"], header_scan["gy"]], axis=1)

    def _make_full_fig():
        _fig = go.Figure(go.Scattergl(
            x=_x_full, y=_y_full, mode="markers",
            marker=dict(size=3, color=header_scan["cdp"], colorscale="Viridis", colorbar=inset_colorbar()),
            customdata=_custom_full,
            hovertemplate=(
                "X=%{x:,.1f}  Y=%{y:,.1f}<br>"
                "header IL=%{customdata[0]}  header XL=%{customdata[1]}<br>"
                "header CDP=%{customdata[2]}<br>"
                "SX=%{customdata[3]:,}  SY=%{customdata[4]:,}<br>"
                "GX=%{customdata[5]:,}  GY=%{customdata[6]:,}<extra></extra>"
            ),
        ))
        _fig.update_layout(
            title="1. CDP from header — positioned by header coordinates",
            xaxis_title="X (SX/GX midpoint)", yaxis_title="Y (SY/GY midpoint)",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            width=FIG_WIDTH, height=FIG_HEIGHT,
            margin=dict(l=70, r=40, t=70, b=60),
        )
        return _fig

    def _make_scatter_png():
        return _make_full_fig().to_image(format="png", width=FIG_WIDTH, height=FIG_HEIGHT, scale=2)

    # Decimated version - what's actually shown interactively. Systematic stride (not random) so the CDP
    # gradient / survey outline still look right, capped at 20k points so the live output stays well under
    # marimo's default output_max_bytes.
    _n = len(header_scan["cdp"])
    _max_points = 20_000
    _stride = max(1, _n // _max_points)
    _idx = np.arange(0, _n, _stride)

    fig_cdp_scatter = go.Figure(go.Scattergl(
        x=_x_full[_idx], y=_y_full[_idx], mode="markers",
        marker=dict(size=3, color=header_scan["cdp"][_idx], colorscale="Viridis", colorbar=inset_colorbar()),
        customdata=_custom_full[_idx],
        hovertemplate=(
            "X=%{x:,.1f}  Y=%{y:,.1f}<br>"
            "header IL=%{customdata[0]}  header XL=%{customdata[1]}<br>"
            "header CDP=%{customdata[2]}<br>"
            "SX=%{customdata[3]:,}  SY=%{customdata[4]:,}<br>"
            "GX=%{customdata[5]:,}  GY=%{customdata[6]:,}<extra></extra>"
        ),
    ))
    fig_cdp_scatter.update_layout(
        title=f"1. CDP from header (showing {len(_idx):,} of {_n:,} traces, every {_stride}th)",
        xaxis_title="X (SX/GX midpoint)", yaxis_title="Y (SY/GY midpoint)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=FIG_WIDTH, height=FIG_HEIGHT,
        margin=dict(l=70, r=40, t=70, b=60),
    )

    mo.vstack([
        mo.md(f"*Showing {len(_idx):,} of {_n:,} traces (every {_stride}th) for interactive display — the "
              "downloaded PNG below uses every trace.*") if _stride > 1 else mo.md(""),
        mo.ui.plotly(
            fig_cdp_scatter,
            config={"displayModeBar": True,
                   "toImageButtonOptions": {"filename": "cdp_from_header_scatter", "format": "png", "scale": 2}},
        ),
        mo.download(data=_make_scatter_png, filename="cdp_from_header_scatter.png", mimetype="image/png",
                   label="💾 Save full-resolution plot 1"),
    ])
    return


@app.cell(hide_code=True)
def _(grid_eq, mo):
    mo.vstack([
        mo.md("""
    ---
    ## 2. CDP from header — map, positioned by calculated IL / XL

    **What this shows.** The same header CDP values as plot 1, but laid out on a proper IL x XL grid instead
    of raw X/Y coordinates — which plot 1 can't do, since most files don't actually populate the header's own
    inline/crossline words (see the byte-position note above). Every trace's midpoint is converted to an IL,
    XL cell using the survey's documented corner-point grid; all traces landing in the same cell are averaged.
    Each cell is coloured by the **mean of the header's own CDP** for the traces in that cell — the position
    is calculated, the colour is not.

    **Equation used** (same corner-point grid as the "Grid definition" section above):
    """),
        mo.md(f"```\n{grid_eq}\nColour   = mean(CDP)   for all traces sharing that (IL, XL) cell   [CDP read directly from the header]\n```"),
    ], gap=0)
    return


@app.cell
def _(
    FIG_HEIGHT,
    FIG_WIDTH,
    calc_il,
    calc_xl,
    go,
    header_scan,
    inset_colorbar,
    mo,
    np,
):
    _il, _xl, _cdp = calc_il, calc_xl, header_scan["cdp"]
    _il_min, _il_max = int(_il.min()), int(_il.max())
    _xl_min, _xl_max = int(_xl.min()), int(_xl.max())
    _n_i, _n_j = _il_max - _il_min + 1, _xl_max - _xl_min + 1
    _key = (_il - _il_min) * _n_j + (_xl - _xl_min)
    _n_bins = _n_i * _n_j
    _counts = np.bincount(_key, minlength=_n_bins).astype(np.float64)
    _safe = np.where(_counts > 0, _counts, 1.0)
    _sums = np.bincount(_key, weights=_cdp.astype(np.float64), minlength=_n_bins)
    _grid = np.where(_counts > 0, _sums / _safe, np.nan).reshape(_n_i, _n_j)
    _il_axis = np.arange(_il_min, _il_max + 1)
    _xl_axis = np.arange(_xl_min, _xl_max + 1)

    fig_header_cdp_calc_grid = go.Figure(go.Heatmap(
        x=_xl_axis, y=_il_axis, z=_grid, colorscale="Viridis", colorbar=inset_colorbar(),
        hovertemplate="XL=%{x}<br>IL=%{y}<br>header CDP=%{z:.0f}<extra></extra>",
    ))
    fig_header_cdp_calc_grid.update_layout(
        title="2. CDP from header — positioned by calculated IL/XL",
        xaxis_title="XL (calculated)", yaxis_title="IL (calculated)",
        xaxis=dict(range=[int(_xl_axis.min()) - 2, int(_xl_axis.max()) + 2]),
        yaxis=dict(scaleanchor="x", scaleratio=1, range=[int(_il_axis.min()) - 2, int(_il_axis.max()) + 2]),
        width=FIG_WIDTH, height=FIG_HEIGHT,
        margin=dict(l=70, r=40, t=70, b=60),
    )

    def _make_header_cdp_calc_grid_png():
        return fig_header_cdp_calc_grid.to_image(format="png", width=FIG_WIDTH, height=FIG_HEIGHT, scale=2)

    mo.vstack([
        mo.ui.plotly(fig_header_cdp_calc_grid, config={"displayModeBar": True,
                     "toImageButtonOptions": {"filename": "cdp_header_by_calc_ilxl", "format": "png", "scale": 2}}),
        mo.download(data=_make_header_cdp_calc_grid_png, filename="cdp_header_by_calc_ilxl.png", mimetype="image/png",
                   label="💾 Save plot 2"),
    ])
    return


@app.cell(hide_code=True)
def _(grid_eq, mo):
    mo.vstack([
        mo.md("""
    ---
    ## 3. CDP recalculated — scatter, positioned by CalcCDPX / CalcCDPY

    **What this shows.** One point per trace, positioned the same way as plot 1 (the source-receiver
    midpoint), but now the **colour is a brand-new CDP number that this notebook computes from scratch** —
    the header's own CDP is not used anywhere on this plot. The midpoint is converted to an IL, XL cell via
    the corner-point grid, and CDP is rebuilt from that IL/XL with a simple row-major formula. Hover any
    point to see its calculated IL, XL and CDP.

    **Equation used** (CDP, the last line, is the colour plotted):
    """),
        mo.md(f"```\n{grid_eq}\n```"),
    ], gap=0)
    return


@app.cell
def _(
    FIG_HEIGHT,
    FIG_WIDTH,
    calc_cdp_x,
    calc_cdp_y,
    calc_il,
    calc_xl,
    cdp_calculated,
    go,
    inset_colorbar,
    mo,
    np,
):
    _custom_full = np.stack([calc_il, calc_xl, cdp_calculated], axis=1)

    def _make_calc_full_fig():
        _fig = go.Figure(go.Scattergl(
            x=calc_cdp_x, y=calc_cdp_y, mode="markers",
            marker=dict(size=3, color=cdp_calculated, colorscale="Viridis", colorbar=inset_colorbar()),
            customdata=_custom_full,
            hovertemplate=(
                "X=%{x:,.1f}  Y=%{y:,.1f}<br>calc IL=%{customdata[0]}  calc XL=%{customdata[1]}<br>"
                "calc CDP=%{customdata[2]}<extra></extra>"
            ),
        ))
        _fig.update_layout(
            title="3. CDP recalculated — positioned by CalcCDPX/CalcCDPY",
            xaxis_title="CalcCDPX", yaxis_title="CalcCDPY",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            width=FIG_WIDTH, height=FIG_HEIGHT, margin=dict(l=70, r=40, t=70, b=60),
        )
        return _fig

    def _make_calc_scatter_png():
        return _make_calc_full_fig().to_image(format="png", width=FIG_WIDTH, height=FIG_HEIGHT, scale=2)

    # Decimated for interactive display, same reasoning as plot 1.
    _n = len(cdp_calculated)
    _max_points = 20_000
    _stride = max(1, _n // _max_points)
    _idx = np.arange(0, _n, _stride)

    fig_cdp_calc_scatter = go.Figure(go.Scattergl(
        x=calc_cdp_x[_idx], y=calc_cdp_y[_idx], mode="markers",
        marker=dict(size=3, color=cdp_calculated[_idx], colorscale="Viridis", colorbar=inset_colorbar()),
        customdata=_custom_full[_idx],
        hovertemplate=(
            "X=%{x:,.1f}  Y=%{y:,.1f}<br>calc IL=%{customdata[0]}  calc XL=%{customdata[1]}<br>"
            "calc CDP=%{customdata[2]}<extra></extra>"
        ),
    ))
    fig_cdp_calc_scatter.update_layout(
        title=f"3. CDP recalculated (showing {len(_idx):,} of {_n:,} traces, every {_stride}th)",
        xaxis_title="CalcCDPX", yaxis_title="CalcCDPY",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        width=FIG_WIDTH, height=FIG_HEIGHT, margin=dict(l=70, r=40, t=70, b=60),
    )

    mo.vstack([
        mo.md(f"*Showing {len(_idx):,} of {_n:,} traces (every {_stride}th) for interactive display — the "
              "downloaded PNG below uses every trace.*") if _stride > 1 else mo.md(""),
        mo.ui.plotly(
            fig_cdp_calc_scatter,
            config={"displayModeBar": True,
                   "toImageButtonOptions": {"filename": "cdp_recalculated_scatter", "format": "png", "scale": 2}},
        ),
        mo.download(data=_make_calc_scatter_png, filename="cdp_recalculated_scatter.png", mimetype="image/png",
                   label="💾 Save full-resolution plot 3"),
    ])
    return


@app.cell(hide_code=True)
def _(grid_eq, mo):
    mo.vstack([
        mo.md("""
    ---
    ## 4. CDP recalculated — map, positioned by calculated IL / XL

    **What this shows.** The same calculated IL x XL grid as plot 2, but here **both** the position **and**
    the colour come from the recalculated CDP — nothing from the header is used on this plot at all (not even
    to colour it, unlike plot 2). Each grid cell is coloured by the mean of the recalculated CDP for every
    trace whose calculated IL/XL lands in that cell. This is the "purely recalculated" counterpart to plot 2's
    "header CDP on a working grid."

    **Equation used** (identical formula to plot 3, just binned into a grid instead of plotted as loose points):
    """),
        mo.md(f"```\n{grid_eq}\nColour   = mean(CDP)   for all traces sharing that (IL, XL) cell\n```"),
    ], gap=0)
    return


@app.cell
def _(
    FIG_HEIGHT,
    FIG_WIDTH,
    calc_il,
    calc_xl,
    cdp_calculated,
    go,
    inset_colorbar,
    mo,
    np,
):
    _il_min, _il_max = int(calc_il.min()), int(calc_il.max())
    _xl_min, _xl_max = int(calc_xl.min()), int(calc_xl.max())
    _n_i, _n_j = _il_max - _il_min + 1, _xl_max - _xl_min + 1
    _key = (calc_il - _il_min) * _n_j + (calc_xl - _xl_min)
    _n_bins = _n_i * _n_j
    _counts = np.bincount(_key, minlength=_n_bins).astype(np.float64)
    _safe = np.where(_counts > 0, _counts, 1.0)
    _sums = np.bincount(_key, weights=cdp_calculated.astype(np.float64), minlength=_n_bins)
    _grid = np.where(_counts > 0, _sums / _safe, np.nan).reshape(_n_i, _n_j)
    _il_axis = np.arange(_il_min, _il_max + 1)
    _xl_axis = np.arange(_xl_min, _xl_max + 1)

    fig_cdp_calc_grid = go.Figure(go.Heatmap(
        x=_xl_axis, y=_il_axis, z=_grid, colorscale="Viridis", colorbar=inset_colorbar(),
        hovertemplate="XL=%{x}<br>IL=%{y}<br>CDP=%{z:.0f}<extra></extra>",
    ))
    fig_cdp_calc_grid.update_layout(
        title="4. CDP recalculated — positioned by calculated IL/XL",
        xaxis_title="XL (calculated)", yaxis_title="IL (calculated)",
        xaxis=dict(range=[int(_xl_axis.min()) - 2, int(_xl_axis.max()) + 2]),
        yaxis=dict(scaleanchor="x", scaleratio=1, range=[int(_il_axis.min()) - 2, int(_il_axis.max()) + 2]),
        width=FIG_WIDTH, height=FIG_HEIGHT, margin=dict(l=70, r=40, t=70, b=60),
    )

    def _make_calc_grid_png():
        return fig_cdp_calc_grid.to_image(format="png", width=FIG_WIDTH, height=FIG_HEIGHT, scale=2)

    mo.vstack([
        mo.ui.plotly(fig_cdp_calc_grid, config={"displayModeBar": True,
                     "toImageButtonOptions": {"filename": "cdp_recalculated_grid", "format": "png", "scale": 2}}),
        mo.download(data=_make_calc_grid_png, filename="cdp_recalculated_grid.png", mimetype="image/png",
                   label="💾 Save plot 4"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Every trace — IL, XL, CDP and coordinates

    The header's own IL, XL and CDP for every trace, alongside its raw coordinates (SX, SY, GX, GY), the
    midpoint (X, Y) used in plot 1, and the recalculated CDP (`calc_cdp`, from `CalcCDPX = (SrcX+RecX)/2`,
    `CalcCDPY = (SrcY+RecY)/2` and the corner-point grid, used in plots 2-4).
    """)
    return


@app.cell
def _(
    calc_cdp_x,
    calc_cdp_y,
    calc_il,
    calc_xl,
    cdp_calculated,
    header_scan,
    mo,
    np,
    pd,
):
    cdp_header_table_df = pd.DataFrame({
        "trace": np.arange(1, len(header_scan["cdp"]) + 1),
        "il_hdr": header_scan["il_hdr"],
        "xl_hdr": header_scan["xl_hdr"],
        "cdp": header_scan["cdp"],
        "sx": header_scan["sx"],
        "sy": header_scan["sy"],
        "gx": header_scan["gx"],
        "gy": header_scan["gy"],
        "x": np.round((header_scan["sx"].astype(np.float64) + header_scan["gx"].astype(np.float64)) / 2.0, 1),
        "y": np.round((header_scan["sy"].astype(np.float64) + header_scan["gy"].astype(np.float64)) / 2.0, 1),
        "calc_cdp_x": np.round(calc_cdp_x, 1),
        "calc_cdp_y": np.round(calc_cdp_y, 1),
        "calc_il": calc_il,
        "calc_xl": calc_xl,
        "calc_cdp": cdp_calculated,
    })

    cdp_header_table = mo.ui.table(cdp_header_table_df, selection=None, page_size=15, show_column_summaries=False,
                                   label="Every trace — header IL / XL / CDP and coordinates")
    cdp_header_table
    return (cdp_header_table_df,)


@app.cell
def _(cdp_header_table_df, mo):
    def _make_header_table_csv():
        return cdp_header_table_df.to_csv(index=False).encode("utf-8")

    mo.download(data=_make_header_table_csv, filename="cdp_from_header_table.csv", mimetype="text/csv",
               label="💾 Save table as CSV")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## Extract: write the recalculated CDP into a new file

    Writes a full copy of the loaded file — **byte-identical to the original** — except the CDP header word
    (SEG-Y bytes 21-24) is replaced with the **recalculated CDP** (`CDP = IL * N_XL + XL`, same value as plots
    3/4) for every trace. Every other header field, all sample data, and (for SEG-Y) the 3600-byte file
    header are copied straight through, unchanged.

    The output is written directly to disk at the path below — for a file this size, a browser download isn't
    practical, so there's no download button here; use the path to feed the file into whatever reads it next.
    """)
    return


@app.cell
def _(Path, mo, su_path):
    _p = Path(su_path.value)
    _default_out = _p.parent / f"{_p.stem}_recalc_cdp{_p.suffix}"
    output_path = mo.ui.text(label="Output file path", value=str(_default_out), full_width=True)
    output_path
    return (output_path,)


@app.cell
def _(mo):
    write_button = mo.ui.run_button(label="💾 Write file with recalculated CDP", kind="warn")
    write_button
    return (write_button,)


@app.cell
def _(Path, cdp_calculated, file_info, mo, np, output_path, write_button):
    mo.stop(not write_button.value, mo.md("👆 Set the output path above, then click **Write file with recalculated CDP**."))

    _out_path = Path(output_path.value)
    _src_path = file_info["path"]
    _header_offset = file_info["header_offset"]
    _full_dtype = file_info["full_dtype"]
    _n = file_info["n_traces"]

    if _out_path.resolve() == _src_path.resolve():
        raise ValueError("Output path must be different from the input path - refusing to overwrite the source file.")

    with mo.status.spinner(title=f"Writing {_n:,} traces ({_out_path.name})..."):
        with open(_src_path, "rb") as _fin, open(_out_path, "wb") as _fout:
            if _header_offset:
                _fout.write(_fin.read(_header_offset))  # copy the SEG-Y textual+binary file header verbatim
            _done = 0
            while _done < _n:
                _take = min(20000, _n - _done)
                _chunk = np.fromfile(_fin, dtype=_full_dtype, count=_take)
                if _chunk.shape[0] != _take:
                    raise RuntimeError("Unexpected short read while writing the output file.")
                _chunk["header"]["cdp"] = cdp_calculated[_done:_done + _take].astype(np.int32)
                _chunk.tofile(_fout)
                _done += _take

    _out_size = _out_path.stat().st_size
    mo.md(
        f"✅ **Wrote {_n:,} traces** ({_out_size / 1e9:.2f} GB) to `{_out_path}` — CDP field replaced with the "
        "recalculated value, everything else unchanged."
    )
    return


if __name__ == "__main__":
    app.run()
