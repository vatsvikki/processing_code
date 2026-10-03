"""Full explanations for every parameter, input and button - shown in the GUI as an (i) hover (no UI code here).

`info_text(param)` gives the text of a tool / step parameter; WIDGET_INFO the text of the inputs and buttons of the
page itself (file path, Save bar, whole-data bar, ...).  Where two parameters share a key but mean different things the
key is written "key:kind" (e.g. "velocity:table" is the NMO velocity function, "velocity:float" the mute velocity).
"""
from __future__ import annotations

from .registry import Param

PARAM_INFO: dict[str, str] = {
    "velocity_model_type": "What the model's samples are: RMS velocity (used as is) or interval velocity (turned into RMS "
                           "with Dix: Vrms^2(t) = (1/t) * sum(Vint^2 * dt)). A depth model is always taken as interval "
                           "velocity vs depth.",
    "model_velocity_units": "The unit of the model's velocity values (ft/s or m/s). They are converted to the output unit "
                            "(5.) before anything else.",
    "stretch_mute_pct": "NMO stretches the wavelet, most at shallow times and far offsets: the stretch of a sample is "
                        "(t(x) - t0) / t0. Samples stretched by more than this percentage are muted (set to 0 and left out "
                        "of the stack). 0 = no mute; 30 % is the brute-stack notebook's default.",
    "nmo_clip_pct": "Amplitude clip percentile of the image plots: the grey scale is set so that this percentile of "
                    "|amplitude| is full black / white (98 = the strongest 2 % saturate).",
    "section_il": "The inline of the IL section of the stack (the CDPs with this data IL, by XL). 0 = the IL with the most "
                  "CDPs.",
    "section_xl": "The crossline of the XL section of the stack (the CDPs with this data XL, by IL). 0 = the XL with the "
                  "most CDPs.",
    "section_velocity_type": "What the velocity sections and the overlay show: the RMS velocity used for NMO, or the "
                             "equivalent interval velocity (gray beyond the model's own range).",
    # ---- shot selection and display -------------------------------------------------------------------------------
    "ffid": "FFID = field file ID, the shot number.\nThe slider only offers FFIDs that really exist in the file (read from "
            "the trace headers), so a missing shot can never be picked. Drag it, or type a number in the box next to it: the "
            "tool re-runs for that shot at once.\nThe trace-range and time-range boxes go back to the full shot when you "
            "change shot.",
    "trace": "Trace in shot: whose 240-byte trace header is decoded in the first table (1 = the first trace of the selected "
             "shot, in the order recorded).\nThe second table shows, for every header word, the minimum, maximum and number "
             "of distinct values over the whole shot. A number larger than the shot's trace count uses the last trace.",
    "sort_by": "Order in which the traces are DRAWN: file = as recorded (channel order); offset = by absolute source-receiver "
               "offset; channel = trace number within the field record; receiver = receiver line, then station; cdp = by the "
               "CDP (ensemble) number of the header; cdp_offset = by CDP, then by offset.\nThe second label row under the "
               "gather shows the value it is sorted by. Only the display changes, the data are not modified.",
    "clip_pct": "Amplitude clip percentile: the grey scale is set so that this percentile of |amplitude| is full black / "
                "white.\n98 (default) lets the strongest 2 % of the samples saturate so that weak events stay visible; "
                "100 = no clipping (one strong sample can wash the whole picture out); 90-95 gives more contrast. Display "
                "only; saved figures use the same value.",
    "agc_ms": "AGC window for the DISPLAY, ms (0 = off): every sample is divided by the RMS amplitude of a window this long "
              "centred on it, so weak deep events become visible next to strong shallow ones.\nTypical values 300-1000 ms. "
              "It does not change the data (use the AGC processing step for that).",
    "t_min_ms": "Start of the time axis shown, ms (0 = the top of the record).\nDragging a zoom box on the gather sets this "
                "and the other range boxes; 'Reset zoom' returns to the full record.",
    "t_max_ms": "End of the time axis shown, ms. It starts at the record length taken from the header (number of samples x "
                "sample interval); lower it to look at the shallow part only.\nA zoom box sets it too.",
    "fig_height": "Height of the figure in inches (default 7). The width is fixed at 14 inches (1540 px) for every figure so "
                  "that gathers, maps and stacks line up.\nLower it to fit a small screen, raise it for more vertical detail. "
                  "Saved figures have the same size.",
    "trace_min": "First trace position drawn (1-based, counted in the display order set by 'Trace order in gather').\nA zoom "
                 "box sets it; 'Reset zoom' goes back to 1.",
    "trace_max": "Last trace position drawn. It starts at the number of traces of the selected shot (0 would also mean "
                 "'to the end').\nA zoom box sets it; lower it to look at the first traces only.",
    # ---- map figures -------------------------------------------------------------------------------------------------
    "show_sources": "Draw the source positions: one marker for every shot (one per FFID, taken from the source coordinates "
                    "in the header of its first trace).",
    "show_receivers": "Draw the receiver positions: the distinct receiver coordinates found in the shots that were read "
                      "(see 'Shots read for the receiver positions').",
    "show_selected": "Highlight the shot chosen with the FFID slider: its source as a star and the receivers that recorded it "
                     "in green, plus a second figure with the receiver spread relative to the source.",
    "show_grid": "Draw the inline / crossline grid: faint lines at round IL and XL numbers with their labels, the dashed survey "
                 "outline and its labelled corners.\nNeeds an IL / XL grid: from the corner-point table or from the headers.",
    "ilxl_ticks": "Write IL and XL numbers on the axes instead of X and Y: ticks at round IL / XL values placed at their true "
                  "positions.\nNeeds an IL / XL grid. If the grid is turned against the plot axes the axis label says so and "
                  "the values are read along the middle of the other axis.",
    "rec_shots": "How many evenly spaced shots are read to find the receiver positions (the sources are always taken from "
                 "every shot).\n60 (default) is enough when the receiver layout is fixed and takes about 3 s; 0 = read every "
                 "trace header of the file (about 20 s for 950 000 traces) - use it when the receivers move between shots.",
    "cmap": "Colour map of the value shown (fold, or CDP number): viridis (default, dark = low value), plasma, turbo, "
            "cividis (colour-blind friendly) or magma.",
    "fold_max": "Top of the colour scale (a fold value). 0 = automatic: the 99.5th percentile of the fold, so a few very high "
                "bins do not wash out the rest.\nType a number to plot several maps on the same scale.",
    "bin_align": "Where the CMP bins sit relative to whole IL / XL numbers.\nauto (default) tries the four half-bin shifts and "
                 "keeps the smoothest fold, i.e. the one that puts the source-receiver midpoints at the bin CENTRES. Midpoints "
                 "lying exactly on a bin edge alias into stripes.\n'bin centres on whole IL / XL' and 'bin edges on whole IL / "
                 "XL' force one alignment.",
    # ---- QC analysis and criteria ------------------------------------------------------------------------------------
    "ref_window": "Neighbour window for the reference RMS, in traces (odd number, default 21): every trace is compared with "
                  "the running median RMS of this many neighbouring live traces.\nLarger = smoother reference; 1 = one global "
                  "median for the whole shot.",
    "ref_sort": "Order used to pick the neighbours for the reference RMS: offset (default; the steadiest, amplitude varies "
                "smoothly with offset), receiver (line, then station), channel or file order.\nIndependent of how the gather "
                "is drawn.",
    "t_start_ms": "Start of the analysis window, ms: the QC measurements (RMS, peak-to-peak, zero fraction, crest factor, DC) "
                  "use only this part of each trace.\nUse it to leave out a muted top.",
    "t_end_ms": "End of the analysis window, ms. Starts at the record length (from the header).\nSet it lower to leave out a "
                "noisy tail.",
    "dead_p2p_max": "A trace is dead if its peak-to-peak amplitude in the analysis window is <= this value.\n0 (default) = "
                    "only exactly flat / all-zero traces. Raise it slightly to also catch traces that carry only a tiny noise "
                    "floor.",
    "dead_zero_frac": "Dead if the fraction of exactly-zero samples is >= this (0 to 1, default 0.9 = 90 %). 0 = test off.\n"
                      "Catches traces that are mostly zero, e.g. muted or truncated ones.",
    "dead_rel_rms": "Near-dead test: dead if the trace RMS < this x the reference RMS of its neighbours (default 0.01 = 40 dB "
                    "below them). 0 = off.\nFinds extremely weak traces that are not exactly flat.",
    "noisy_ratio": "Bad (noisy) if the RMS > this x the reference RMS of the neighbours (default 5). 0 = off.\nCatches traces "
                   "swamped by noise.",
    "weak_ratio": "Bad (weak) if the RMS < this x the reference RMS of the neighbours (default 0.1). 0 = off.\nCatches traces "
                  "that are much weaker than their surroundings but not dead.",
    "spike_crest": "Bad (spike) if max|amplitude| / RMS > this (the crest factor; default 30, normal traces are 8-15 in this "
                   "survey). 0 = off.\nCatches traces with isolated large spikes.",
    "dc_ratio_max": "Bad (DC offset) if |mean| / standard deviation > this (default 1). 0 = off.\nCatches traces with a "
                    "constant offset from zero.",
    "corr_ratio_min": "Low-correlation test: every live trace is cross-correlated with its nearest neighbours on the same "
                      "receiver line (best positive correlation within the lag limit, median over the neighbours). It is bad "
                      "if this score < this x the median score of traces at similar offsets (default 0.4). 0 = off.\nFinds "
                      "noisy, spiky or scrambled traces that have a normal amplitude, and hints when a polarity reversal "
                      "would fit. It makes the whole-survey scan about 4x slower.",
    "corr_neighbours": "Neighbours on each side used for the correlation test (default 2, range 1-6). More = a steadier score, "
                       "slower.",
    "corr_lag_ms": "Largest time shift searched when correlating neighbouring traces (default 40 ms): it allows for the "
                   "moveout between adjacent traces.",
    # ---- dead-trace correction, decon -------------------------------------------------------------------------------
    "mode": "How the flagged dead traces are corrected: interpolate = replace each from its live neighbours on the same "
            "receiver line (trace count unchanged); remove = delete them from the gather (fewer traces, and the later steps "
            "and the saved file have fewer traces).",
    "power": "Time power of the geometric-spreading gain: every sample is multiplied by (t / 1 s)^power.\n0 = no gain, 1-2 "
             "is usual; a larger power boosts the late times more.",
    "operator_ms": "Operator length of the prediction-error filter, ms (default 160).\nA longer operator removes longer "
                   "reverberations but risks damaging real events.",
    "prewhite_pct": "Pre-whitening, %: white noise added to the zero lag of the autocorrelation to stabilise the filter.\n"
                    "0.1-1 % is usual (default 0.1); more = a more stable but less 'spiking' result.",
    "design_start_ms": "Start of the design window, ms: the filter is computed from this part of each trace and applied to the "
                       "whole trace.",
    "design_end_ms": "End of the design window, ms; starts at the record length. A shorter window designs the filter from "
                     "the better-quality shallow part only.",
    "balance": "Restore the input RMS of every trace after the deconvolution (on by default). Off = the raw filter output, "
               "whose amplitudes are lower than the input.",
    # ---- processing steps ---------------------------------------------------------------------------------------------
    "f1": "Band-pass LOW CUT, Hz: below this frequency the response is 0. Together with low pass / high pass / high cut it "
          "forms a zero-phase trapezoid (Ormsby). Must satisfy 0 <= low cut < low pass < high pass < high cut <= Nyquist "
          "(250 Hz for 2 ms samples).",
    "f2": "Band-pass LOW PASS, Hz: between the low cut and this frequency the response ramps up from 0 to 1; above it the "
          "signal passes unchanged.",
    "f3": "Band-pass HIGH PASS, Hz: up to this frequency the response is 1; between it and the high cut it ramps down to 0.",
    "f4": "Band-pass HIGH CUT, Hz: above this frequency the response is 0. Must stay below the Nyquist frequency (250 Hz for "
          "2 ms samples).",
    "window_ms": "AGC window, ms (default 500): every sample is divided by the RMS amplitude of a window this long around it.\n"
                 "Shorter = flatter amplitudes but more distortion of real amplitude changes; longer = gentler.",
    "velocity:float": "Mute velocity (length unit of the file per second; default 12000 ft/s or 3500 m/s): the mute line is "
                      "t = t0 + |offset| / velocity and everything ABOVE it is zeroed.\nPick a velocity slightly faster than "
                      "the first arrivals so that only the noise ahead of them is removed.",
    "t0_ms": "Time of the mute line at zero offset, ms (default 0): the mute line is t = t0 + |offset| / velocity. Raise it to also mute a short time window at the top of every trace, e.g. to remove the direct wave near the source.",
    "taper_ms": "Length of the linear taper below the mute line, ms (default 100): it avoids an abrupt cut.",
}

# The page's own inputs and buttons (not tool parameters)
WIDGET_INFO: dict[str, str] = {
    "segy_file": "Full path of the SEG-Y file, on the machine that runs the app (~ is expanded).\nTyping a new path only "
                 "changes the box: press Load Data (Data class) to read it. A file must be in field-record order (FFIDs "
                 "ascending) for the shot lookup.",
    "load_button": "Reads the file: EBCDIC and binary headers, the trace count, and an index of every shot (about 2 s for "
                   "950 000 traces), with a progress bar at the top. Every other tool waits for it. Pressing it again reads "
                   "the file from disk again.",
    "run_button": "Runs the tool with the values in this form (the 'Apply' / 'Run' button). Options that are drawn above the "
                  "form re-run the tool as soon as you change them; the values in the form only take effect when you press "
                  "this button.",
    "class_button": "A class of functions: click to drop down its functions, click a function to open it (the list closes).",
    "function_button": "Open this function.",
    "zoom_reset": "Back to the full record: the full time range and all traces of the shot.",
    "zoom_back": "Back to the previous zoom window (one step per press).",
    "flow_add": "Opens the list of every function, Load Data first. A processing / QC one is added to the end of the "
                "flow (up to 12; the same function may be used more than once) and the ＋ then appears again after it, "
                "ready to add the next one - the flow runs from top to bottom on the selected shot. Load Data, EBCDIC & "
                "Headers, Plot Shot Gather, Acquisition Geometry and Acquisition Fold open on their own instead: they "
                "read / report on the shot rather than changing it, so they cannot be chained into a flow.",
    "step_up": "Move this function one place up in the flow (it then runs earlier). Its parameters move with it.",
    "step_down": "Move this function one place down in the flow (it then runs later). Its parameters move with it.",
    "step_remove": "Remove this function from the flow. The functions below move up and keep their parameters.",
    "view_stage": "Which stage of the flow is shown and saved: 0 = the untouched input, k = the result after step k. The "
                  "spectrum / autocorrelation figure compares that stage with the input.",
    "batch_out": "Full path of the output SEG-Y file, like the input path. A folder or an empty box gets an automatic name; a "
                 "name without extension gets .sgy. The job never writes onto the input file and refuses to overwrite unless "
                 "'overwrite' is ticked. It writes <name>.part and renames it only when every shot is done.",
    "batch_from": "First FFID of the range to process (default: the first shot of the file). The whole-data results are kept per "
                  "range: another range runs again from the first function.",
    "batch_to": "Last FFID of the range to process (default: the last shot of the file). The whole-data results are kept per "
                "range: another range runs again from the first function.",
    "batch_fmt": "Sample format of the output file: ibm = 4-byte IBM float, ieee = 4-byte IEEE float.",
    "batch_overwrite": "Allow replacing an output file that already exists. Off = the job stops instead of overwriting.",
    "batch_button": "Runs the flow on screen on EVERY shot of the chosen range and writes its product as one SEG-Y file (a "
                    "15 GB file is never held in memory): the processed shots - or, when the flow has CDP steps, the "
                    "CDP-sorted gathers, the NMO-corrected CDP gathers or the stack (the last CDP step decides). With CDP "
                    "steps the shot steps' result is kept in the output folder, so CDP Sort / NMO / Stack show every CDP "
                    "of it at once, and a later run with the same shot steps does not process the shots again. A "
                    "per-shot report CSV is written next to the output; a cancelled or failed run leaves no partial file.",
    "step_run": "Run the flow up to this function now, on the selected shot, and show its result. The functions above it "
                "that already ran with the same settings are reused (their result is kept), so only what changed is "
                "computed. Changing a parameter never runs anything by itself - it marks this function and the ones "
                "below it 'not run yet' until you press ▶. Selected shot only - ▶ Run flow runs the whole data.",
    "run_all": "Run every function of the flow on the selected shot, and the whole flow on the whole data (every shot of "
               "the FFID range) in one pass - each shot read once, the result written once to the output file.",
    "batch_cancel": "Stops the whole-data run (press Run again to restart it). No file is written by a cancelled run (a "
                    "partial '.part' file is removed).",
    "job_dismiss": "Remove this message from the top of the page.",
    "job_refresh": "The bar re-checks the job every 2 s while it runs; this timer keeps it ticking.",
    "save_dir": "Folder for the saved figures, tables and single-shot SEG-Y files (created if needed). Default: the 'output' "
                "folder of the app.",
    "save_fmt": "File format of the saved figures: png (raster, resolution set by 'PNG dpi'), pdf or svg (vector graphics; "
                "need a figure the tool kept).",
    "save_dpi": "Resolution of PNG files in dots per inch (50-600, default 150). A 14-inch-wide figure at 150 dpi is 2100 "
                "pixels wide. Not used for pdf / svg.",
    "save_tables": "Also write the tables of the result as CSV files and the text results as Markdown, next to the figures.",
    "save_button": "Writes the figures on screen to the chosen folder. File names are <file>_<tool>_FFID<n>_<figure title>_"
                   "<parameters>.<ext>, where <parameters> lists the settings that differ from their defaults (or 'defaults'), "
                   "so saving with other settings never overwrites the earlier image.",
    "save_data_fmt": "Sample format of the SEG-Y file written for the shot on screen: ibm = 4-byte IBM float (as the input "
                     "usually is), ieee = 4-byte IEEE float.",
    "exists_overwrite": "Replace the existing output file with this run's result (the old file is gone once the run "
                        "finishes; a cancelled run leaves it as it was).",
    "exists_rename": "Keep the existing file and write this run's result next to it as <name>_2.sgy (or _3, _4 ... - the "
                     "first name not taken).",
    "exists_cancel": "Do not run - e.g. to type another name under 🌐 Whole data first.",
}


def info_text(p: Param) -> str:
    """The (i) text of a parameter: its full explanation, else its short help."""
    return getattr(p, "info", "") or PARAM_INFO.get(f"{p.key}:{p.kind}") or PARAM_INFO.get(p.key) or p.help or ""


# ---------------------------------------------------------------------------------------------------------------------
# FUNCTION_INFO: the full write-up of a Processing (Flow) function - what it computes, in the same terms as the code
# (functions/correction.py, decon.py, filters.py, steps.py), and how each of its own parameters changes the result.
# Shown behind a clickable "ⓘ Full explanation" in the Flow's "Pick a function to add" list and on an added step.
# Keyed by the step's key (pipeline.Step.key), not its label.
# ---------------------------------------------------------------------------------------------------------------------
FUNCTION_INFO: dict[str, str] = {
    "correct_dead": r"""
**What it does.** No frequency-domain math - it works trace by trace, across the gather, not across time.

*Interpolate* (default): for a dead trace at position $i$ on a receiver line, it finds the nearest **live** trace to
its left ($i_L$) and right ($i_R$) *on the same receiver line* (file / channel order) and linearly interpolates:

$$
w = \frac{i - i_L}{i_R - i_L}, \qquad \text{data}[i] = (1-w)\,\text{data}[i_L] + w\,\text{data}[i_R]
$$

At the end of a line, or if the line has no live trace at all, the nearest live trace (from the whole gather) is
copied instead of interpolated. Because this only mixes whole traces, it assumes the neighbouring traces are already
reasonably aligned in time (no NMO has been applied yet) - it will not "fix" a dead trace into the correct moveout,
only give it a plausible amplitude and waveform.

*Remove*: dead traces are deleted outright and the gather gets shorter (trace count drops); nothing is interpolated,
so a later step sees one fewer trace and the geometry (fold, offsets) of the shot changes.

**Which traces are dead.** If a *Detect Dead Traces* step has not run earlier in the flow, this step finds them
itself with the default thresholds (peak-to-peak $\le 0$, zero-sample fraction $\ge 0.9$, RMS $< 0.01\times$ the
neighbour reference RMS - the same tests as the *Dead Traces* tool). Running *Detect Dead Traces* first, with your
own thresholds, lets you control exactly which traces this step treats as dead.

**Parameter**
- **Correct dead traces by** (*interpolate* / *remove*) - chooses between the two behaviours above. Interpolate keeps
  the trace count and offsets intact (safer for anything downstream that expects one trace per receiver); remove is
  appropriate when a placeholder trace would be worse than a gap (e.g. before a stack).
""",
    "geometric_spreading": r"""
**What it does.** A heuristic time-power gain - not tied to a velocity model or the true geometrical-spreading loss
of this survey, just a smooth increasing gain applied to every sample:

$$
\text{data}[n] \;\leftarrow\; \text{data}[n] \times \left(\frac{t_n}{1\,\text{s}}\right)^{\text{power}}
$$

$$
t_n = \max(n \cdot dt,\; dt)
$$

($t_n$ is the sample's two-way time in seconds; it is floored at one sample so $t=0$ is never raised to a negative
power / does not zero the first sample.) The gain is the same for every trace of the shot - it does not use offset,
so it is not an NMO- or offset-dependent correction, only a time-dependent one.

**Parameter**
- **Time power** - the exponent above. $0$ = no gain (pass-through). $1$ gives a gain that grows linearly with time
  (a mild, commonly-used approximation for spherical amplitude decay $\propto 1/t$ in a roughly constant-velocity
  medium). $2$ (the default) grows with the square of time - a stronger boost of late/weak arrivals that in practice
  also compensates some of the intrinsic (absorption) attenuation a $1/t$ gain leaves behind, at the cost of raising
  late-time noise more. Larger values raise the far/late part of the trace more aggressively; watch the bottom of the
  gather for noise blow-up when raising it.
""",
    "spiking_decon": r"""
**What it does.** Wiener-Levinson **spiking** (zero-delay, unit prediction-distance) deconvolution, solved
independently for every trace:

1. The autocorrelation $r[0..L]$ of the trace is computed inside the **design window**
   ($L$ = the operator length in samples).
2. **Pre-whitening**: $r[0] \leftarrow r[0]\,(1 + \text{prewhite\_pct}/100)$ - a small amount is added to the zero
   lag before solving, which regularises the system (equivalent to adding white noise to the trace for the purpose
   of the solve only).
3. The **Levinson-Durbin recursion** solves for the prediction-error filter $a = [1, a_1, \dots, a_L]$ that
   minimises the mean-square one-step prediction error, i.e. that best predicts (and so can remove) the
   trace's own reverberant / repeating structure from its autocorrelation.
4. The trace is convolved with $a$ over its **whole length** (not just the design window): this both compresses the
   source wavelet towards a spike and flattens (whitens) the amplitude spectrum, which is what removes short-period
   multiples / reverberation and sharpens the wavelet.
5. If **RMS restore** is on, each output trace is rescaled so its RMS matches its input trace's RMS (a pure
   level correction after the fact - it does not change the shape of the deconvolved waveform).

**Parameters**
- **Operator length, ms** - $L$ in samples $= \text{round}(\text{operator\_ms}/dt)$. Longer removes longer-period
  reverberations / multiples, but estimates more filter coefficients from the same design window (more risk of
  overfitting the noise and distorting closely-spaced real events); shorter is more stable but leaves longer-period
  ringing untouched.
- **Pre-whitening, %** - the regularisation above. Higher = a more stable, gentler filter that behaves closer to
  a pass-through (less spiking, but also less risk of an unstable / ringy operator on a noisy or dead-ish trace);
  lower = a sharper, more aggressive deconvolution that can become unstable on low-fold or noisy data. 0.1-1 % is
  the usual range.
- **Design window start / end, ms** - where the autocorrelation (the trace statistics the filter is built from) is
  measured; the filter itself is still applied to the whole trace. Pick a window with representative reflectivity
  (avoid a window dominated by a single huge direct arrival, or by dead/muted samples) - fewer than 8 samples in the
  window is refused outright.
- **Restore input RMS of every trace** - on: re-levels each trace to its own input RMS after deconvolution (usual
  choice, keeps the display comparable to the input). Off: the raw prediction-error-filter output is shown, whose
  amplitude can be much lower than the input (decon whitens the spectrum, which usually lowers total energy).
""",
    "fk_filter_step": r"""
**What it does.** Removes ground roll and other slow, linear noise with a velocity fan in the frequency-wavenumber
(F-K) domain. Each receiver line of the shot is filtered on its own: its traces are put at their position along the
line (from the receiver X / Y, scaled to the header offset) on a regular grid (gaps stay empty), and

$$
D(f, k) = \iint d(t, x)\, e^{-2\pi i (f t - k x)}\, dt\, dx, \qquad v_{app} = \left|\frac{f}{k}\right|
$$

An event that moves out with velocity $v$ maps to the line $f = v\,|k|$. The pass weight is 0 below the reject
velocity, 1 above the pass velocity and a cosine taper between (no sharp edge, no ringing); frequencies above
*Filter only below* are left untouched. The filtered line is transformed back and every trace taken from its grid
column.

**Parameters.** *Reject below* - set just above the fastest noise (ground roll is typically 1000-3000 ft/s). *Pass above*
- everything faster is fully kept; reflections have far higher apparent velocities. *Output: Removed noise* shows
what is taken away: it should hold no reflections.

**Limits.** F-K needs the noise to be sampled finely enough: slow ground roll is aliased above
$f = v / (2\,\Delta x)$ (e.g. 1500 ft/s on 278 ft spacing: above 2.7 Hz) and the aliased part folds into the pass
zone. Then use **Radon Filter (Linear)**, which copes better with aliasing. The F-K spectrum (before / after, with the
two velocity lines) of the longest line is shown below the gather as a **flip-flop** - click *Before* / *After* to
switch in place (same colour scale) - and so is the shot gather: *Before* / *After* / *Removed noise*, all at the
input's clip, so what changes between them is the data, not the display.
""",
    "radon_step": r"""
**What it does.** A least-squares (Hampson) Radon transform models the data as a sum of events along curves
$t = \tau + p\,\phi(x)$; the noise part of that model is turned back into traces and **subtracted**, so the rest of
the data is unchanged. For every frequency the model is the damped least-squares solution

$$
m = (L^H L + \mu I)^{-1} L^H d, \qquad L_{x,p} = e^{-2\pi i f p\, \phi(x)}
$$

with $\mu$ = *Damping* % of the mean diagonal of $L^H L$.

- **Linear (ground roll):** $\phi(x) = |x|$ (offset), $p$ = slowness from 0 to $1 / v_{min}$. Events slower than
  *Remove events slower than* ($p > 1 / v_{cut}$) are removed, below *Highest frequency* only. Works in offset, so it
  needs no regular trace spacing and handles aliased ground roll much better than F-K.
- **Parabolic (multiples):** the gather is first NMO-corrected with the *primary velocity function*; primaries become
  flat, multiples (slower) keep a residual moveout $q$ at the far offset: $\phi(x) = (x / x_{max})^2$. Moveouts above
  *Remove moveout above* are modelled, taken back to recording time (inverse NMO) and subtracted.

**Parameters.** *Number of slownesses / moveouts* - finer separation, but slower. *Damping* - more = smoother panel,
less sharp separation. *Output: Removed noise* shows the subtracted noise - it should hold no primaries.

**Notes.** On a shot gather the parabolic version assumes roughly flat layers (a shot gather then looks like a CMP
gather); multiples close to the primary moveout (small $q$) cannot be separated. A linear Radon of a long record is
padded in time by the largest shift ($x_{max} / v_{min}$), so a lower *Slowest velocity* costs time (about 2 s per
shot on this survey). Below the gather: a flip-flop of the shot gather (*Before* / *After* / *Removed noise*, same
clip) and the Radon panel with the cut line.
""",
    "bandpass_filter": r"""
**What it does.** A zero-phase trapezoidal (Ormsby) band-pass, applied in the frequency domain so it introduces no
time shift. The frequency response is built from the four corner frequencies:

$$
H(f) = \begin{cases}
0 & f < f_1 \\
\dfrac{f - f_1}{f_2 - f_1} & f_1 \le f < f_2 \\
1 & f_2 \le f \le f_3 \\
\dfrac{f_4 - f}{f_4 - f_3} & f_3 < f \le f_4 \\
0 & f > f_4
\end{cases}
$$

Every trace is Fourier-transformed, multiplied by $H(f)$, and inverse-transformed. Because $H(f)$ is real (no
imaginary / phase part), the filter does not shift events in time - only their relative amplitude by frequency
changes, and the wavelet may ring a little more or less depending on how steep the ramps are (see below).

**Parameters** (all in Hz; must satisfy $0 \le f_1 < f_2 < f_3 < f_4 \le$ Nyquist $= 500/dt$)
- **Low cut ($f_1$)** - everything below this frequency is removed completely. Raise it to cut more low-frequency
  noise (ground roll footprint, DC drift, cable/wind noise).
- **Low pass ($f_2$)** - above this the low-frequency ramp reaches full amplitude (1). The gap $f_2 - f_1$ is the
  low-end ramp width: narrower = a sharper low-cut (more ringing / Gibbs artifacts in time), wider = a gentler
  roll-off (less ringing, less sharp separation from the rejected band).
- **High pass ($f_3$)** - up to here the pass band stays fully open (1); above this the high-frequency ramp begins.
- **High cut ($f_4$)** - full attenuation is reached here and above (and always by the Nyquist frequency). The gap
  $f_4 - f_3$ is the high-end ramp width, with the same trade-off as the low end: narrower = sharper cut, more
  ringing; wider = smoother, less ringing.
""",
    "agc_gain": r"""
**What it does.** A sliding-window RMS automatic gain control - every sample is divided by the RMS amplitude of a
window of length **AGC window, ms** centred on it:

$$
\text{data}[n] \;\leftarrow\; \frac{\text{data}[n]}{\sqrt{\dfrac{1}{N}\sum_{k=n-N/2}^{n+N/2} \text{data}[k]^2}}
$$

where $N = \text{round}(\text{window\_ms}/dt)$, forced odd and at least 3 samples, computed with a running
cumulative-sum (so it costs the same whether the window is short or long). This equalises amplitude along the trace
- it does **not** preserve true relative amplitude between near and far offsets, or between strong and weak
reflections, so it is a display / picking aid more than a preserved-amplitude correction.

**Parameter**
- **AGC window, ms** - the window length $N$ above. Short window: fast-acting, aggressively flattens amplitude
  (strong events get suppressed quickly, but weak zones - e.g. ahead of the first break, or a real gap in the data -
  get boosted along with their noise). Long window: smoother, slower-reacting gain that changes less from sample to
  sample and preserves more of the true amplitude structure, at the cost of equalising far/near or strong/weak zones
  less completely.
""",
    "top_mute": r"""
**What it does.** Zeroes everything **above** (earlier than) a straight mute line and tapers the transition. For
every trace, from its offset header:

$$
t_{\text{line}} = t_0 + \frac{|\text{offset}|}{\text{velocity}} \times 1000 \;\;(\text{ms})
$$

i.e. a straight-ray line through $(0, t_0)$ with the given velocity - meant to sit just ahead of (slightly faster
than) the true first-break moveout, so only the noise cone ahead of the first arrivals is removed. The per-sample
weight is a linear ramp:

$$
w(t) = \operatorname{clip}\!\left(\frac{t - t_{\text{line}}}{\text{taper\_ms}},\; 0,\; 1\right)
$$

$$
\text{data}[t] \;\leftarrow\; \text{data}[t] \times w(t)
$$

so samples at or before $t_{\text{line}}$ are fully zeroed, samples at or after $t_{\text{line}} + \text{taper\_ms}$
are untouched, and the taper interval between them ramps linearly from 0 to 1.

**Parameters**
- **Mute velocity** - the slope of the line, in the survey's length unit per second (its "auto" starting value is a
  per-survey guess, slightly faster than the true first-break velocity). Too low mutes into real early reflection
  energy at far offsets; too high leaves first-break noise unmuted near the line.
- **Time at zero offset, ms** ($t_0$) - raises or lowers the whole mute line uniformly, independent of offset.
- **Taper length, ms** - width of the linear ramp. 0 = a hard on/off cut (can ring when a later frequency filter is
  applied to the sharp edge); larger = a softer transition, at the cost of leaving in more of the pre-mute noise
  close to the line.
""",
    # ---- Data functions (functions/tools.py) - not a flow step: open on their own from the ＋ list ------------------
    "load_data": r"""
**What it does.** Reads the SEG-Y file named in the *SEG-Y file* box: the 3200-byte EBCDIC text header and 400-byte
binary header (format, sample count, sample interval), then indexes every shot by scanning the FFID word of every
trace header (not the samples) to find where each field record starts and how many traces it has - that index is
what lets any tool jump straight to one shot later (binary search) instead of reading the file from the start.

Nothing else works until this has run once: every other function opens the file *this* run read (the path box alone
changes nothing by itself). Press it again after editing the path box to open a different file, or to re-read the
same file if it changed on disk.

No parameters - it always re-reads the file named in the *SEG-Y file* box.
""",
    "file_headers": r"""
**What it does.** A read-only look at the raw file, three levels at once, for the shot / trace chosen on the left:

- the **EBCDIC text header** - the 40 lines of free-text survey information at the start of the file
- the **binary header** - the fixed-format block that gives the sample count, sample interval, data format code and
  more, once for the whole file
- every one of the **88 trace-header words** of the one chosen trace (240 bytes), plus, for that same word, the
  **minimum, maximum and number of distinct values seen over the whole shot** - a fast way to notice a header word
  that is constant when it should vary (or the reverse) without opening every trace by hand

**Parameters**
- **Shot FFID** - which shot's trace headers are summarised (min / max / distinct count) and whose one trace is shown
  in full.
- **Trace in shot** - which trace of that shot (1 = first) has its full 88-word header listed.
""",
    # ---- "Display" functions of the Flow (functions/steps.py) - a map / gather for this stage; they do not change
    # the gather itself (see state.view_extra), so a later step in the same flow still gets the unchanged input.
    "plot_shot_gather_step": r"""
**What it does.** No maths - the plain shot gather image, the same one the flow already shows for any stage (marking
whatever dead / bad traces earlier *Detect* steps in this flow have flagged, same as always). Adding it as its own
step changes nothing about the data; it is only a labelled checkpoint - useful to give one particular point of the
flow an explicit name in the step list, e.g. before and after a filter, without needing *View stage* to remember which
number that was.
""",
    "acquisition_geometry_step": r"""
**What it does.** A plan-view (map) of the whole survey: every source position, the receiver positions, and this
shot's own spread highlighted - its source as a star, the receivers that recorded it joined to it - plus a second
figure of that spread relative to the source. Draws the **IL / XL grid** (see below) over the map when one is known,
and lists the survey's coordinate ranges. Does not touch the gather - a step after this one in the flow still gets
whatever came into this step unchanged.

**Where the IL / XL grid comes from** - the **Survey grid** of the 📁 Data card (asked once, after Load data, for the
whole app): the corner table you saved for this file, else the **receiver headers** (receiver line = IL, receiver
station = XL, fitted against the receiver X / Y coordinates), else the **inline / crossline header words**, fitted
against the trace midpoints. The same grid is used by *Acquisition Fold*, *NMO Correction*, *CDP Stack* and the CDP lists.

**Parameters**
- **Show source / receiver positions**, **Highlight the selected shot** - what is drawn on the map.
- **Show the IL / XL grid**, **Tick labels in IL / XL** - draw the grid lines / label the axes in IL / XL instead of
  X / Y (needs a grid).
- **Shots read for the receiver positions** - how many shots' receiver headers are scanned to find the distinct
  receiver positions (0 = every shot; use this when the receivers move during the survey).
""",
    "acquisition_fold_step": r"""
**What it does.** A fold map - traces per bin - counted from the **source-receiver midpoint** of every trace in the
file (not just this shot), on the same canvas and with the same IL / XL grid as *Acquisition Geometry*. Bins are one
IL x one XL cell of the grid; without a grid, square bins half the receiver spacing wide are used instead. The **first
run reads every trace header** (~20 s for 950 000 traces); after that, changing a display setting is instant. Does
not touch the gather.

**Parameters**
- **Bin alignment** - where the bins sit relative to whole IL / XL numbers; *auto* tries the four possible half-bin
  shifts and keeps the one with the smoothest fold (real midpoints almost never land exactly on a bin edge; the
  shift that avoids that gives the least striped-looking map).
- **Colour map**, **Colour scale maximum** (0 = automatic: the 99.5th percentile, so a few very high bins do not
  wash out the rest) - how the fold is coloured.
- **Show source / receiver positions**, **Show the IL / XL grid**, **Tick labels in IL / XL** - as *Acquisition
  Geometry*; the grid is the **Survey grid** of the 📁 Data card.
""",
    # ---- QC steps (functions/steps.py) - flag traces for the steps after them (esp. Correct Dead Traces) and mark
    # them on the flow's gather plot; a note (count + which traces), not a separate table / threshold plot.
    "detect_dead": r"""
**What it does.** Every trace is measured inside the analysis window (RMS, peak-to-peak, fraction of exactly-zero
samples) and compared against a **reference RMS**: the running median RMS of its neighbouring *live* traces (in
offset order by default, independent of how the gather is displayed). A trace is flagged **dead** if any one test
fires (a threshold of 0 switches that test off):

- flat / all-zero: peak-to-peak $\le$ **Dead if peak-to-peak amplitude**
- mostly zeros: zero-sample fraction $\ge$ **Dead if fraction of zero samples**
- near-dead: $\text{RMS} < \text{reference} \times$ **Dead if RMS < this x reference**

Its only effect on the flow is the flag itself and the marks (blue) on the flow's gather plot - *Correct Dead
Traces*, later in the flow, uses it (and finds the same dead traces itself, with the default thresholds, if this
step is not run first). Add this step before *Correct Dead Traces* when you want to set your own thresholds instead
of the defaults.
""",
    "detect_bad": r"""
**What it does.** The same reference-RMS idea as *Detect Dead Traces*, but for traces that have signal, just not
trustworthy signal - a trace is flagged **bad** if any one test fires:

- noisy: $\text{RMS} >$ **Noisy if RMS > this x reference**
- weak: $\text{RMS} <$ **Weak if RMS < this x reference**
- spike: $\max|\text{amp}| / \text{RMS} >$ **Spike if max|amp| / RMS >** (crest factor; normal traces are usually 8-15)
- DC offset: $|\text{mean}| / \text{std} >$ **DC offset if |mean| / std >**
- low correlation: every live trace is cross-correlated with its nearest live neighbours (on the same receiver line,
  searching a lag range for the best moveout-shifted match); its score is compared with the running median score of
  traces at similar offsets, so steep near-source arrivals and line ends are not flagged just for being less
  coherent. Catches noisy / spiky / scrambled traces that have a normal amplitude, and hints "polarity reversed?"
  when the flipped trace would correlate well. This test alone makes the step markedly slower.

A trace already flagged dead is not also flagged bad. It marks the flagged traces (red) on the flow's gather plot.
Nothing downstream in this flow currently corrects bad traces automatically - this step is for visibility (and for a
later Save / export step to carry the flags), not a fix.
""",
    "nmo_correction_step": r"""
**What it does.** The NMO correction of the brute-stack notebook (section 6), on one CDP gather:

1. **The CDP gather** - the chosen CDP's traces (top 200 by fold, or any CDP number), nearest offset first, **as the
   steps above produced them** (decon, filters ...): from the whole-data run's result when it has been made, else the
   shots this CDP's traces come from are processed on the spot. Its real position is the mean source-receiver midpoint.
2. **Its velocity** - *Velocity for NMO* chooses where it comes from: a **velocity function** (one table of
   zero-offset time / RMS velocity, linear between rows, held flat beyond them, the same for every CDP) or the
   **velocity-model SEG-Y**: the model's traces are placed in X / Y from their INLINE_3D /
   CROSSLINE_3D words (bytes 189 / 193) through the corner points ($X = m_x \cdot XL + c_x$, $Y = m_y \cdot IL + c_y$,
   least squares), and the trace **nearest the CDP's position** is used. Its values are converted to the output unit,
   then to an RMS velocity against two-way time: RMS used as is; interval via Dix,
   $V_{rms}^2(t) = \frac{1}{t}\sum V_{int}^2\,\Delta t$; a depth model (interval velocity vs depth) is first turned
   into time, $\Delta t = 2\,\Delta z / V_{int}$. Then interpolated onto the data's time axis (held flat beyond the
   model's end).
3. **NMO** - every sample at zero-offset time $t_0$ is read from $t(x) = \sqrt{t_0^2 + (x / V_{rms}(t_0))^2}$, $x$ = |offset|
   (in the output length unit). The **stretch** of a sample is $\frac{t(x) - t_0}{t_0}$; samples stretched by more
   than the mute percentage are muted (0 % = off).

**Settings** - 1 RMS or interval samples; 2 the model's velocity unit; 3 time (TWTT) or depth axis; 4 the model's
sample interval (0 = its own header word: ms for a time model, the depth step for a depth model, e.g. 10000 -> 10 ft -
a wrong value mis-times every velocity, which shows as NMO not flattening events); 5 the output velocity unit (offsets are converted too).
**First the choice: Velocity for NMO** - *Velocity model SEG-Y* asks for the model's file, settings 1-4 and the
**model's corner points** (its INLINE_3D / CROSSLINE_3D <-> X / Y - left empty, they are read from the model's text
header when it has a GRID CORNERS block); *Velocity function* asks only for its table, used for every CDP. Setting 5
(the output unit) is used by both. The **data's** corner points are not asked here: they are the **Survey grid** of
the 📁 Data card, the same for every function (they give each CDP its data IL / XL and place the model grid in the
overlay).

The CDP (its real midpoint X / Y) and the model's traces (through the model's corner points) meet in real X / Y, where
the nearest model trace is found.

**Shown:** the CDP gather, the NMO stretch section and the velocity used (output RMS, with the model's own input trace
on its own axis), and the **data vs. model grid coverage** overlay in the data's IL / XL: the model grid (grey), the
data's CDP bins (blue), the CDP being corrected (red star) and the model trace its velocity came from (orange cross).
Warnings as in the notebook: a matched model trace more than 5,000 ft away, or velocities outside 5,000-25,000 ft/s
(1,500-7,600 m/s).

**In the flow** - it comes after the processing steps (a processing step below it is refused). A **CDP Stack** step
below it stacks its NMO-corrected gathers (with this velocity and mute - NMO is not done again). **Run flow on whole
data** with NMO Correction as the last step writes the NMO-corrected CDP gathers of the whole survey (CDP order,
nearest offset first, original headers).
""",
    "cdp_recalculate": r"""
**What it does.** Rebuilds the CDP number of every trace from its source-receiver midpoint and an IL / XL grid you
enter (as in *plot_cdp_from_header_marimo.py*), and compares it with the CDP in the headers.

$$X_{mid} = \frac{SX + GX}{2},\quad Y_{mid} = \frac{SY + GY}{2}\qquad\text{(coordinate scalar of bytes 71-72 applied)}$$
$$IL = \text{round}\left(IL_{min} + \frac{Y_{mid} - Y_0}{\Delta_{IL}}\right),\quad
  XL = \text{round}\left(XL_{min} + \frac{X_{mid} - X_0}{\Delta_{XL}}\right)$$
$$CDP = IL \cdot N_{XL} + XL,\qquad N_{XL} = XL_{max} - XL_{min} + 1$$

**Inputs** - the grid: IL / XL range, X / Y of the corner (IL min, XL min), bin sizes (IL along Y, XL along X); it is
axis-aligned (no rotation). The grid of every run is saved and filled in again next time (`~/.trace_qc/cdp_recalc_grids.json`).

**Shown** - a summary (traces, grid size, CDP range, traces outside the grid, how many header CDPs already equal the
recalculated ones), the equations with your numbers, the four **corner points** (CSV), the notebook's four plots (header
CDP and recalculated CDP, each by midpoint and on the IL / XL grid; full-resolution PNGs) and the first 5,000 traces
(header vs recalculated). **Write a SEG-Y with the recalculated CDP** copies the file with only bytes 21-24 replaced.
""",
    "shot_geometry_qc": r"""
**What it does.** Runs the notebook *shot_geometry_qc_marimo.py* as it is, in its own tab: a check that each shot's
geometry is right, from its first arrivals.

- **Linear-velocity overlay** - a straight line $t = |offset| / V$ is drawn on the shot record (raw offsets and offsets
  recomputed from the coordinates, $\sqrt{(GX - SX)^2 + (GY - SY)^2}$): if it follows the first arrivals, the
  geometry is consistent; if it diverges, the coordinates / offsets of that shot are suspect.
- **Fix toggles** - negate SX / SY / GX / GY, swap X / Y of the source or the receivers, or shift any of them by a
  constant, live on the recomputed panel; save a correction per FFID.
- **Hilbert first-break picks** (STA / LTA on the envelope energy) and the **automatic search** - the misfit between the
  picks and the line for every toggle combination and for shifts of each coordinate, best first.
- **Whole-survey audit** - the mean pick-vs-line misfit of every shot, the shots above a threshold flagged, on a map.
- **Write corrected copy** - the queued corrections applied (and the shots queued to drop removed) in a new SEG-Y.

Opening it reads the whole file for its maps, and *Scan* and the audit read it again - each with a progress bar.
""",
    "flow": r"""
**What it is.** A chain of functions run top to bottom. Add them with **＋ Add a function**, order them with ▲ ▼,
remove them with −; each has its parameters and this explanation in its card.

**Nothing runs by itself.** Changing a parameter only records it: that function and the ones below it are "not run yet"
(the panel says so and shows the last result still valid). **▶** beside a function runs the flow up to it on the
selected shot and shows its result; **▶ Run flow** (at the end) runs every function. The result of every function is kept, so a ▶
reuses the functions above that already ran with the same settings and computes only what changed (e.g. after
Spiking Decon has run, ▶ on NMO Correction starts from the deconvolved data). Display settings (clip, trace order,
zoom, figure size) only redraw. Another FFID shows its raw shot until ▶ is pressed for it.

- **Shot steps** (QC, Processing, Display) work on the shot chosen with the FFID slider - instant, for tuning.
- **CDP steps** - *NMO Correction* and *CDP Stack* - work on CDP gathers of the data the shot steps above them produced.
  They must come after the processing steps.
- **View stage** shows the gather after any function, with its spectrum / autocorrelation against the input.

**Whole data.** **▶ Run flow** (at the end of the Flow card) also runs the whole flow on every shot of the FFID
range, in the background (progress and Cancel in the card bottom right), in one pass: each shot is read once, all
functions run on it in memory, and the result is written once - to the output file set under 🌐 *Whole data* in the
Flow card: the processed shots, or - when it ends with NMO Correction / CDP Stack - the NMO-corrected CDP gathers or
the stack. ▶ beside a function only previews the selected shot. With NMO Correction / CDP Stack the shot functions'
result is written once more (output/flows) - those steps need the data sorted by CDP; a later run with the same shot
functions (e.g. only the stack settings changed) reuses it. 🌐✓ beside a function = done on the whole data with these
settings. Results of earlier settings that the flow can no longer use are deleted (each is as big as the input). Tools that work on the whole file on their own (CDP Sort, CDP Recalculate, Shot Geometry QC) are in 🛠 Tools.
""",
    "cdp_stack_step": r"""
**What it does.** Stacks the CDP gathers the steps above it produced - it has no velocity, NMO, sorting or filter of
its own:

- **with NMO Correction above** - the stack of that step's NMO-corrected gathers: at every time, the mean of the live
  (not stretch-muted) samples, $\text{stack}(t_0) = \frac{\sum_i a_i(t_0)}{\#\,\text{live}(t_0)}$ - with that step's
  velocity and mute (NMO is done once, as that step says);
- **without it** (e.g. after CDP Sort) - the plain mean of the CDP gathers.

The preview stacks the CDP you pick and shows the notebook's panels: its input gathers, their stack and the velocity
used. On the whole data (▶ / ▶ Run flow) it stacks every CDP (the flow's product is then the stack: one zero-offset trace per CDP,
with the CDP number, data IL / XL in INLINE_3D / CROSSLINE_3D and X / Y x 100 in the header); the stack is kept in the
output folder and its sections are shown here:

- **Stacked sections** - a fixed IL (varying XL) left and a fixed XL (varying IL) right, same time axis; IL / XL of each
  CDP from its position and the *data* corner points (0 = the most common IL / XL).
- **Velocity sections** (RMS or interval) of the NMO step's velocity along the same lines - gray beyond the model's own
  range - and their **overlay** on the stack (with NMO Correction above).
""",
    "cdp_sort_tool": r"""
**What it does.** The CDP sort of the brute-stack notebook, over the **whole survey** of the file in the path box:
every trace header is read once (cached for the file, with a progress bar), and the traces are put in order by CDP
number (bytes 21-24), then by $|\text{offset}|$ within each CDP - `lexsort((|offset|, cdp))`, so each CDP's traces run
near to far instead of in recorded order.

- **Summary** - traces, number of CDPs, CDP range and fold (min / median / max).
- **The gather of one CDP** - *CDP to show* (0 = the highest fold; a number not in the file shows the nearest CDP),
  read from the file nearest offset first, its traces labelled by offset.
- **All CDPs** - a table of every CDP: fold, offset range, mean source-receiver midpoint
  ($\frac{SX+GX}{2}, \frac{SY+GY}{2}$, coordinate scalar applied) and IL / XL when a grid is known (the corner table saved
  for this file, or the headers' IL / XL).
- **Write a CDP-sorted copy of the file** (optional) - every trace in that order, headers and samples unchanged (written
  to `<name>.part` and renamed when complete).

It sorts the **raw** file and is independent of the Flow: the Flow's *NMO Correction* and *CDP Stack* make their own
CDP gathers from the data the steps above them produced, so the Flow needs no sort step.
""",
}
