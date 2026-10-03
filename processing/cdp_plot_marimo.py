import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full", app_title="CDP / IL / XL Plot")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from functions import segy_io

    default_path = "/Users/vikas/SCube/harrison/data/raw_data/EAST_ATCHAFALAYA-GCSR-UNDEFINED_SDL062470-16.sgy"
    return Path, default_path, mo, np, plt, segy_io


@app.cell

def _(mo):
    mo.md(
        "# CDP plot\n\n"
        "Load a SEG-Y file, read CDP / ensemble values from its trace headers, "
        "and plot them against inline and crossline values. Header byte numbers "
        "are 1-based, as in the SEG-Y specification."
    )


@app.cell

def _(default_path, mo):
    file_path = mo.ui.text(value=default_path, label="SEG-Y file path", full_width=True)
    load = mo.ui.run_button(label="Load header")
    return file_path, load


@app.cell

def _(file_path, load, mo, np, segy_io):
    if not load.value:
        info = mo.md("Press **Load header** to read the SEG-Y headers.")
        return info, None, None, None

    try:
        path = file_path.value.strip()
        sgy = segy_io.open_segy(path)
        sample_n = min(sgy.ntraces, 20_000)
        sample_idx = np.linspace(0, sgy.ntraces - 1, sample_n).round().astype(int)
        sample_raw = sgy.read_headers_at(sample_idx)
        info = mo.md(
            f"Loaded `{path}`: **{sgy.ntraces:,} traces**, byte order `{sgy.order}`, "
            f"{sgy.ns:,} samples/trace."
        )
        return info, path, sgy, sample_raw
    except Exception as exc:
        info = mo.md(f"**Could not load file:** `{type(exc).__name__}: {exc}`")
        return info, None, None, None


@app.cell

def _(mo, np, sample_raw, segy_io, sgy):
    if sgy is None:
        detected_message, options = mo.md(""), {}
        return detected_message, options

    def values(byte):
        return segy_io.header_word(sample_raw, int(byte), "i4", sgy.order)

    def usable(byte):
        if not 1 <= int(byte) <= 237:
            return False
        value = values(byte)
        nonzero = value[value != 0]
        return len(nonzero) >= 3 and len(np.unique(nonzero)) >= 2

    known_pairs = [(189, 193, "standard inline / crossline"), (173, 181, "receiver line / station")]
    detected = [(il, xl, label) for il, xl, label in known_pairs if usable(il) and usable(xl)]
    options = {f"IL {il}, XL {xl} - {label}": (il, xl) for il, xl, label in detected}
    message = (
        "Detected usable header pairs: " + ", ".join(options)
        if options else "No standard IL / XL header pair was detected. Enter the byte locations manually."
    )
    detected_message = mo.md(message)
    return detected_message, options


@app.cell

def _(mo, options, sgy):
    if sgy is None:
        return None, None, None, None, None, None

    mode = mo.ui.radio(
        options=["Use detected bytes", "Enter bytes manually"],
        value="Use detected bytes",
        label="IL / XL source",
    )
    detected = mo.ui.dropdown(
        options=list(options) or ["No detected pair"],
        value=next(iter(options), "No detected pair"),
        label="Detected pair",
    )
    il_byte = mo.ui.number(value=189, start=1, stop=237, step=1, label="Inline (IL) byte")
    xl_byte = mo.ui.number(value=193, start=1, stop=237, step=1, label="Crossline (XL) byte")
    cdp_byte = mo.ui.number(value=21, start=1, stop=237, step=1, label="CDP / ensemble byte")
    controls = mo.vstack([mode, detected, mo.hstack([il_byte, xl_byte, cdp_byte])])
    return controls, mode, detected, il_byte, xl_byte, cdp_byte


@app.cell

def _(cdp_byte, detected, il_byte, mode, options, sgy, xl_byte):
    if sgy is None:
        return None, None, None
    if mode.value == "Use detected bytes" and detected.value in options:
        il, xl = options[detected.value]
    else:
        il, xl = int(il_byte.value), int(xl_byte.value)
    cdp = int(cdp_byte.value)
    return il, xl, cdp


@app.cell

def _(Path, cdp, il, mo, np, path, plt, segy_io, sgy, xl):
    if sgy is None:
        summary, plot_result = mo.md(""), None
        return summary, plot_result

    try:
        raw = sgy.read_headers(0, sgy.ntraces)
        il_values = segy_io.header_word(raw, il, "i4", sgy.order)
        xl_values = segy_io.header_word(raw, xl, "i4", sgy.order)
        cdp_values = segy_io.header_word(raw, cdp, "i4", sgy.order)
        valid = (il_values != 0) & (xl_values != 0)
        if valid.sum() < 2:
            summary, plot_result = mo.md(
                "**The selected IL / XL bytes do not contain enough non-zero values.**"
            ), None
            return summary, plot_result

        il_values, xl_values, cdp_values = il_values[valid], xl_values[valid], cdp_values[valid]
        pairs = np.column_stack([il_values, xl_values])
        unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
        median_cdp = np.array([np.median(cdp_values[inverse == i]) for i in range(len(unique_pairs))])

        fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
        plot = ax.scatter(unique_pairs[:, 1], unique_pairs[:, 0], c=median_cdp, s=9, cmap="viridis", alpha=0.85)
        ax.set_xlabel("Crossline (XL)")
        ax.set_ylabel("Inline (IL)")
        ax.set_title(f"CDP / IL / XL map: {Path(path).name}")
        ax.grid(True, alpha=0.25)
        fig.colorbar(plot, ax=ax, label="Median CDP / ensemble")
        summary = mo.md(
            f"Using IL byte **{il}**, XL byte **{xl}**, CDP byte **{cdp}**. "
            f"Plotted **{len(unique_pairs):,}** IL/XL bins from **{len(il_values):,}** traces."
        )
        plot_result = fig
        return summary, plot_result
    except Exception as exc:
        summary = mo.md(f"**Could not plot headers:** `{type(exc).__name__}: {exc}`")
        plot_result = None
        return summary, plot_result


@app.cell

def _(controls, detected_message, info, mo, plot_result, summary):
    mo.vstack([info, detected_message, controls, summary, plot_result])


if __name__ == "__main__":
    app.run()
