import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import struct
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _SC_SCALING_DIR = Path(__file__).resolve().parent.parent / "SC_scaling"
    if str(_SC_SCALING_DIR) not in sys.path:
        sys.path.insert(0, str(_SC_SCALING_DIR))
    from sc_scaling_pipeline import probe_file, build_dtypes

    # Saved-to-disk + mo.image(src=...) instead of embedding the figure
    # inline -- a full-resolution shot image easily exceeds marimo's
    # ~10MB output-size limit (hit twice already on this survey with
    # Plotly; a saved PNG referenced by path has no such limit).
    plot_dir = Path(__file__).resolve().parent / "suximage_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    return Path, build_dtypes, mo, np, plot_dir, plt, probe_file, struct


@app.cell
def _(mo):
    mo.md("""
    # Shot display — SU `suximage`-style

    Reproduces Seismic Unix's `suximage`: a grayscale raster
    ("density") image of one shot gather, amplitude vs. time,
    traces plotted **in as-recorded order** (trace sequence number
    within the field record — the same axis `suximage` uses by
    default, i.e. *not* sorted by offset), with a percentile-based
    clip exactly like `suximage`'s `perc=` parameter.

    This is a plain single-shot viewer — no geometry toggles, no
    reference-velocity overlay. See `shot_geometry_qc_marimo.py` in
    this folder for the geometry-QC version.
    """)
    return


@app.cell
def _(mo):
    su_path = mo.ui.text(
        label="Input .su file path",
        value="../../EAST_ATCHAFALAYA-GCSR-UNDEFINED_SDL062470-16.su",
        full_width=True,
    )
    su_path
    return (su_path,)


@app.cell
def _(struct):
    HEADER_BYTES = 240
    FLDR_OFF = 8
    NS_OFF, DT_OFF = 114, 116

    def detect_endianness(path):
        with open(path, "rb") as f:
            buf = f.read(HEADER_BYTES)
        for order in ("<", ">"):
            ns = struct.unpack_from(order + "H", buf, NS_OFF)[0]
            if 1 <= ns <= 20000:
                return order
        raise ValueError("Could not determine byte order.")

    def get_fldr_at_index(f, idx, record_size, order):
        f.seek(idx * record_size + FLDR_OFF)
        return struct.unpack(order + "i", f.read(4))[0]

    def find_shot_range(path, order, record_size, n_traces, target_fldr):
        """
        Binary search for the shot's contiguous trace block, relying
        on this survey's own EBCDIC header claim that traces are
        stored in field-record order -- turns a lookup that could
        touch ~1,000,000 traces into ~20 seeks total.
        """
        with open(path, "rb") as f:
            lo, hi = 0, n_traces - 1
            found = None
            while lo <= hi:
                mid = (lo + hi) // 2
                v = get_fldr_at_index(f, mid, record_size, order)
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
                v = get_fldr_at_index(f, mid, record_size, order)
                if v == target_fldr:
                    left = mid
                    hi = mid - 1
                else:
                    lo = mid + 1

            lo, hi = found, n_traces - 1
            right = found
            while lo <= hi:
                mid = (lo + hi) // 2
                v = get_fldr_at_index(f, mid, record_size, order)
                if v == target_fldr:
                    right = mid
                    lo = mid + 1
                else:
                    hi = mid - 1

        return left, right

    def read_shot(path, order, target_fldr):
        """
        Read one shot's traces in as-stored order (no sorting) --
        matches suximage's default trace axis, which is just the
        column position within the file.
        """
        import os
        import numpy as np

        with open(path, "rb") as f:
            first_header = f.read(HEADER_BYTES)
            ns = struct.unpack_from(order + "H", first_header, NS_OFF)[0]

        record_size = HEADER_BYTES + ns * 4
        file_size = os.path.getsize(path)
        if file_size % record_size != 0:
            raise ValueError(
                "File size doesn't divide evenly by record size -- ns may "
                "not be constant across the file, so binary search isn't "
                "safe here."
            )
        n_traces = file_size // record_size

        shot_range = find_shot_range(path, order, record_size, n_traces, target_fldr)
        if shot_range is None:
            return None
        left, right = shot_range

        traces = []
        dt_val = None
        with open(path, "rb") as f:
            f.seek(left * record_size)
            for _ in range(left, right + 1):
                header = f.read(HEADER_BYTES)
                dt_us = struct.unpack_from(order + "H", header, DT_OFF)[0]
                raw = f.read(ns * 4)
                samples = np.frombuffer(raw, dtype=order + "f4").astype(np.float32)
                traces.append(samples)
                dt_val = dt_us

        return {
            "traces": np.column_stack(traces),
            "ns": ns,
            "dt_us": dt_val,
        }

    return detect_endianness, read_shot


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Browse every shot in the survey

    Scan once (a few seconds for the whole file), then page through
    every shot with the slider.
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
    Path,
    build_dtypes,
    mo,
    np,
    probe_file,
    scan_button,
    set_shot_list,
    su_path,
):
    if scan_button.value:
        with mo.status.spinner(title="Scanning whole survey for shot (FLDR) list..."):
            _info = probe_file(Path(su_path.value))
            _, _full_dtype = build_dtypes(_info["order"], _info["ns"])
            _fldrs = np.empty(_info["n_traces"], dtype=np.int64)
            _n_done = 0
            with open(su_path.value, "rb") as _f:
                _remaining = _info["n_traces"]
                while _remaining > 0:
                    _take = min(5000, _remaining)
                    _chunk = np.fromfile(_f, dtype=_full_dtype, count=_take)
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
            "*Click **Scan whole survey for shot list** above, then use "
            "the slider below to browse every shot.*"
        ))
    mo.md(f"**{len(shot_list):,} unique shots found**, "
          f"FFID {int(shot_list[0])}-{int(shot_list[-1])}.")
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
def _(current_fldr, detect_endianness, mo, read_shot, su_path):
    order = detect_endianness(su_path.value)
    shot = read_shot(su_path.value, order, current_fldr)
    if shot is None:
        mo.stop(True, mo.md(f"**No traces found for FFID {current_fldr}.**"))
    mo.md(
        f"**FFID {current_fldr}** -- **{shot['traces'].shape[1]} traces**, "
        f"ns={shot['ns']}, dt={shot['dt_us']} us."
    )
    return (shot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `suximage`-style display controls

    - **perc** — percentile clip, same meaning as SU's `perc=`: the
      display is clipped at the amplitude level below which `perc`%
      of all |samples| fall. `perc=100` means no clipping (full
      dynamic range, usually dominated by a few big spikes).
    - **Manual clip** — like SU's `clip=`: an absolute amplitude
      value instead of a percentile.
    - **legend** — like SU's `legend=1`: show a colorbar.
    - **polarity** — like SU's `polarity=-1`: flip which way
      positive amplitude shades.
    """)
    return


@app.cell
def _(mo):
    clip_mode = mo.ui.radio(
        options=["Percentile (perc)", "Manual clip value"],
        value="Percentile (perc)",
        label="Clip mode",
    )
    clip_mode
    return (clip_mode,)


@app.cell
def _(clip_mode, mo):
    perc = mo.ui.slider(50, 100, value=99, step=0.5, label="perc (clip percentile)")
    manual_clip = mo.ui.number(value=1.0, label="Manual clip value")
    mo.hstack([perc, manual_clip]) if clip_mode.value == "Manual clip value" else perc
    return manual_clip, perc


@app.cell
def _(mo):
    cmap_select = mo.ui.dropdown(
        options=["gray", "gray_r", "seismic", "RdBu_r"],
        value="gray",
        label="Colormap",
    )
    legend = mo.ui.checkbox(label="Show colorbar (legend)", value=True)
    polarity = mo.ui.checkbox(label="Reverse polarity")
    full_resolution = mo.ui.checkbox(
        label="Full time resolution (slower to render)"
    )
    mo.hstack([cmap_select, legend, polarity, full_resolution])
    return cmap_select, full_resolution, legend, polarity


@app.cell
def _(np):
    def decimate_for_display(data, time_axis, target_rows=800):
        """
        Max-abs block decimation along the time axis so rendering
        stays fast regardless of ns -- a bin's amplitude only drops
        if every sample inside it is small, so the arrival shape is
        preserved. All clip/percentile stats are computed on the
        full-resolution data before this runs; only the plotted grid
        is decimated.
        """
        n_samples = data.shape[0]
        if n_samples <= target_rows:
            return data, time_axis

        factor = int(np.ceil(n_samples / target_rows))
        n_bins = -(-n_samples // factor)
        pad = n_bins * factor - n_samples
        if pad:
            data = np.pad(data, ((0, pad), (0, 0)), mode="edge")

        reshaped = data.reshape(n_bins, factor, data.shape[1])
        idx = np.argmax(np.abs(reshaped), axis=1)
        binned = np.take_along_axis(reshaped, idx[:, None, :], axis=1)[:, 0, :]
        binned_time = time_axis[::factor][:n_bins]
        return binned, binned_time

    return (decimate_for_display,)


@app.cell
def _(
    clip_mode,
    cmap_select,
    current_fldr,
    decimate_for_display,
    full_resolution,
    legend,
    manual_clip,
    mo,
    np,
    perc,
    plot_dir,
    plt,
    polarity,
    shot,
):
    if shot is None:
        mo.stop(True, mo.md("*Load a shot above to begin.*"))

    data = shot["traces"].astype(np.float64)
    if polarity.value:
        data = -data

    n_traces = data.shape[1]
    dt_ms = shot["dt_us"] / 1000.0
    time_ms = np.arange(shot["ns"]) * dt_ms
    trace_num = np.arange(1, n_traces + 1)

    if clip_mode.value == "Manual clip value":
        clip = float(manual_clip.value) or 1.0
    else:
        clip = np.percentile(np.abs(data), perc.value) or 1.0

    if full_resolution.value:
        data_disp, time_disp = data, time_ms
    else:
        data_disp, time_disp = decimate_for_display(data, time_ms)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(
        data_disp, aspect="auto", cmap=cmap_select.value, vmin=-clip, vmax=clip,
        extent=[trace_num[0], trace_num[-1], time_disp[-1], time_disp[0]],
        interpolation="nearest",
    )
    ax.set_xlabel("Trace number (as recorded)")
    ax.set_ylabel("Time (ms)")
    ax.set_title(
        f"FFID {current_fldr} -- {n_traces} traces -- "
        f"clip={clip:.4g}"
        + (f" (perc={perc.value})" if clip_mode.value != "Manual clip value" else " (manual)"),
        fontsize=10,
    )
    if legend.value:
        fig.colorbar(im, ax=ax, shrink=0.8, label="Amplitude")
    fig.tight_layout()

    _p = plot_dir / f"fldr{current_fldr}_suximage.png"
    fig.savefig(_p, dpi=130)
    plt.close(fig)

    mo.image(src=str(_p), width=800)
    return


if __name__ == "__main__":
    app.run()
