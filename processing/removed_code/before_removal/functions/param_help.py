"""Full explanations for every parameter, input and button - shown in the GUI as an (i) hover (no UI code here).

`info_text(param)` gives the text of a tool / step parameter; WIDGET_INFO the text of the inputs and buttons of the
page itself (file path, Save bar, whole-data bar, ...).  Where two parameters share a key but mean different things the
key is written "key:kind" (e.g. "velocity:table" is the NMO velocity function, "velocity:float" the mute velocity).
"""
from __future__ import annotations

from .registry import Param

PARAM_INFO: dict[str, str] = {
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
               "gather shows the value it is sorted by. Only the display changes, the data are not modified (the Sort Traces "
               "step reorders the data itself).",
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
    "corners": "Survey corner points as a table, one corner per row, columns IL, XL, X, Y (coordinates in the file's units).\n"
               "The table starts filled with the IL / XL grid the headers describe (receiver line / station, or the inline / "
               "crossline words), or with the table saved for this file earlier.\nChange any value - or type your own corners "
               "(3 or 4) - and press Apply / Run: your table then REPLACES the header IL / XL, is saved for this file and "
               "comes back next time. X, Y may be left empty when the headers give a grid; empty rows are ignored; '+ New row' "
               "adds a row. An untouched table means 'use the headers' and saves nothing. Corners that do not fit a regular "
               "grid give a warning.",
    "forget_saved": "Tick and press Apply / Run to delete the corner table saved for this file and go back to the IL / XL of "
                    "the headers.\nThe table on screen may still show your old numbers until you switch tool.",
    "rec_shots": "How many evenly spaced shots are read to find the receiver positions (the sources are always taken from "
                 "every shot).\n60 (default) is enough when the receiver layout is fixed and takes about 3 s; 0 = read every "
                 "trace header of the file (about 20 s for 950 000 traces) - use it when the receivers move between shots.",
    "cmap": "Colour map of the fold: viridis (default, dark = low fold), plasma, turbo, cividis (colour-blind friendly) or "
            "magma.",
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
    "correction": "How the dead traces are corrected: interpolate = replace each from its live neighbours on the same receiver "
                  "line (the trace count stays the same); remove = delete them from the gather (fewer traces).",
    "mode": "How the flagged dead traces are corrected: interpolate = replace each from its live neighbours on the same "
            "receiver line (trace count unchanged); remove = delete them from the gather (fewer traces, and the later steps "
            "and the saved file have fewer traces).",
    "show": "Plot: raw = the data as recorded, with the dead traces marked; corrected = the gather after the correction chosen "
            "above.",
    "marks": "Mark the dead traces on the plot. Raw view: blue vertical lines and diamonds on the top edge. Corrected view: "
             "green lines and triangles at the interpolated traces (or at the removal points).\nOnly shots that have dead "
             "traces show marks (try FFID 559 or 1288); a shot without dead traces has nothing to mark.",
    "power": "Time power of the geometric-spreading gain: every sample is multiplied by (t / 1 s)^power.\n0 = no gain, 1-2 "
             "is usual; a larger power boosts the late times more.",
    "view": "Plot: 'after decon' = the deconvolved gather; 'input' = the gather before the deconvolution.",
    "dead_first": "Dead traces before decon: first run Detect + Correct Dead Traces with the default thresholds (the same "
                  "steps as in the Pipeline) - interpolate or remove - so that dead traces do not disturb the filter "
                  "design.\nnone = deconvolve the gather exactly as recorded.",
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
    "by": "Sort by: offset (absolute source-receiver offset), channel, receiver (line, then station), cdp (the CDP number of "
          "the trace header), 'cdp, offset' (by CDP, and by offset within a CDP).\nThe traces of the gather are reordered "
          "(headers, dead / bad flags and the link to the source traces follow), so later steps and the saved file see the "
          "new order.",
    "descending": "Reverse the order: the largest value comes first (e.g. the farthest offset, the highest CDP). Off = smallest first.",
    "method": "How the traces are stacked into one: mean = sum / number of traces, counting only samples that are not zero "
              "(so a mute does not lower the amplitude); sum = plain sum.",
    # ---- whole-survey scan --------------------------------------------------------------------------------------------
    "ffid_from": "First FFID of the range scanned. It starts at the first FFID of the file.",
    "ffid_to": "Last FFID of the range scanned. It starts at the last FFID of the file, i.e. the whole dataset.",
    "workers": "Number of parallel threads used for the scan (default 8). More threads = faster on a machine with more cores.",
    # ---- CMP / NMO / stack ---------------------------------------------------------------------------------------------
    "cmp_sort_by": "How the traces are sorted.\n'CMP bin, then offset' (default) is the CMP sort of the brute stack: the traces "
                   "are grouped by the CMP bin (CDP gather) of the IL / XL grid their source-receiver midpoint falls in, and "
                   "ordered by offset inside a bin.\n'CMP bin (cdp)' only groups by bin - inside a bin the traces stay in file "
                   "order (the gather is then labelled by the CDP number of the header).\n'offset' orders all traces by offset "
                   "only.\nThe sort follows the bins, not the CDP number of the header (the header CDP can follow another bin "
                   "layout; the QC reports how many bins mix header CDP numbers). Only an index is sorted - no samples are "
                   "copied.",
    "low_fold_pct": "QC: a bin counts as low-fold if its fold is below this percentage of the median fold (default 25 %). "
                    "The QC table reports how many bins are below it; more than 15 % of the bins is flagged.",
    "offset_tol_pct": "QC: tolerance, in %, between the offset header word and the offset computed from the source and "
                      "receiver coordinates (default 2 %). Traces that differ by more are counted; more than 0.5 % of the "
                      "traces is flagged, because a wrong offset breaks the offset sort.",
    "il": "IL number of the CMP bin to show (bin centre). 0 together with XL 0 = the bin with the highest fold.\nThe bin that "
          "contains the numbers is used; with shifted bins the centres are at x.5.",
    "xl": "XL number of the CMP bin to show (bin centre). 0 together with IL 0 = the bin with the highest fold.",
    "velocity_source": "Where the NMO velocity comes from.\n'1-D guess function' = the table below: one function v(t) used for "
                       "EVERY bin (a rough starting guess is fine for a brute stack).\n'velocity model file' = a file with "
                       "picks at several locations; each bin gets its own function, placed by the coordinates, IL / XL or CDP "
                       "numbers found in the file - whichever match the data best.",
    "velocity:table": "1-D velocity function as a table: time in ms (first column) and velocity in the length unit of the file "
                      "per second (second column: ft/s here), one pick per row, linear between the picks and constant outside "
                      "them.\nIt starts as a plausible guess for the unit of the file - replace it with your own picks; "
                      "'+ New row' adds a pick. Used when 'Velocity input' is the 1-D guess function. Velocity should "
                      "normally increase with time.",
    "velocity_file": "Path of a velocity model file (used when 'Velocity input' is 'velocity model file').\nText or CSV with "
                     "a header row naming the columns: the location as IL + XL, or X + Y, or CDP; then TIME (ms) and VELOCITY; "
                     "one pick per row. Several picks per location make its function; between locations the four nearest "
                     "are combined by inverse distance. A file with only TIME and VELOCITY is a 1-D function. If the file's "
                     "locations do not match the data (IL/XL, X/Y and CDP are all tested) an error tells you the match "
                     "percentages.",
    "stretch": "NMO stretch mute, % (default 30; 0 = off): the moveout correction stretches the far-offset shallow samples; "
               "samples stretched by more than this are muted (zeroed) so that they do not smear the stack.\nLower = "
               "stronger mute (cleaner shallow stack, less far-offset data).",
    "line_kind": "Which line to stack: an inline (IL fixed, XL along the section) or a crossline (XL fixed, IL along).",
    "line_no": "IL (or XL) number of the line to stack. 0 = the line with the most traces.\nWith shifted bins the numbers "
               "are x.5.",
    "out_dir": "Folder where the files are written (created if needed). Default: the app's output folder.",
    "sample_format": "SEG-Y sample format of the written stack: ibm = 4-byte IBM floating point (the SEG-Y standard, what "
                     "the source file usually has); ieee = 4-byte IEEE floating point.",
    # ---- Save Data step -------------------------------------------------------------------------------------------------
    "folder": "Folder where the SEG-Y file is written (created if needed). Default: the app's output folder. In a whole-data run the checkpoint file goes to this folder too.",
    "name": "File name (without .sgy); empty = the source file's name. A code of the steps before Save Data (and the FFID "
            "for a single shot) is added, so different processing never overwrites an earlier file.",
    "fmt": "Sample format of the file: ibm = 4-byte IBM float (SEG-Y standard), ieee = 4-byte IEEE float.",
    "write": "Write the file. Off by default in the Pipeline: the pipe re-runs at every change of a setting and every new "
             "shot, and each run would write a file - tick it when you want this shot saved. In 'Apply to whole data' a "
             "checkpoint file with every shot's data at this point is always written.",
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
    "add_flow": "Open the flow. You build it by clicking functions in the Processing tab (top bar): each click adds that function at "
                "the end of the flow. Here every function has ▲ ▼ to move it up or down and a red - to remove it, plus its own "
                "parameters. The flow runs from top to bottom on the selected shot; 'View stage' shows the result after any "
                "function, and when the flow is right you can apply it to the whole data. Click a class on the top bar to go "
                "back to a single function.",
    "class_button": "A class of functions: click to drop down its functions, click a function to open it (the list closes).",
    "function_button": "Open this function.",
    "step_up": "Move this function one place up in the flow (it then runs earlier). Its parameters move with it.",
    "step_down": "Move this function one place down in the flow (it then runs later). Its parameters move with it.",
    "step_remove": "Remove this function from the flow. The functions below move up and keep their parameters.",
    "view_stage": "Which stage of the flow is shown and saved: 0 = the untouched input, k = the result after step k. The "
                  "spectrum / autocorrelation figure compares that stage with the input.",
    "zoom_reset": "Back to the full record: the full time range and all traces of the shot.",
    "zoom_back": "Back to the previous zoom window (one step per press).",
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
    "save_data_button": "Writes the processed shot on screen (for the Pipeline: the stage chosen in 'View stage') as a SEG-Y "
                        "file: the source's EBCDIC / binary headers, and the source 240-byte trace header of every surviving "
                        "trace with the processed samples. The processing steps are noted on text-header cards C37-C40.",
    "ready_button": "Confirms that the steps on screen are the flow you want. The output file, FFID range and the run button "
                    "then appear. Check the result on a few shots first: running the whole file takes a while.",
    "notready_button": "Take the whole-data settings away again and keep editing the flow.",
    "batch_out": "Full path of the output SEG-Y file, like the input path. A folder or an empty box gets an automatic name; a "
                 "name without extension gets .sgy. The job never writes onto the input file and refuses to overwrite unless "
                 "'overwrite' is ticked. It writes <name>.part and renames it only when every shot is done.",
    "batch_from": "First FFID of the range to process (default: the first shot of the file).",
    "batch_to": "Last FFID of the range to process (default: the last shot of the file).",
    "batch_fmt": "Sample format of the output file: ibm = 4-byte IBM float, ieee = 4-byte IEEE float.",
    "batch_overwrite": "Allow replacing an output file (and Save Data checkpoint files) that already exists. Off = the job "
                       "stops instead of overwriting.",
    "batch_button": "Applies the flow on screen to EVERY shot of the chosen range and writes one SEG-Y file while it runs "
                    "(a 15 GB file is never held in memory). Progress and a Cancel button are in the bar at the top of the "
                    "page; a per-shot report CSV is written next to the output. A cancelled or failed job leaves no partial "
                    "file.",
    "batch_cancel": "Stops the whole-data job after the shots in progress; no file is written (the .part file is removed).",
    "job_dismiss": "Remove this message from the top of the page.",
    "job_refresh": "The bar re-checks the job every 2 s while it runs; this timer keeps it ticking.",
}


def info_text(p: Param) -> str:
    """The (i) text of a parameter: its full explanation, else its short help."""
    return getattr(p, "info", "") or PARAM_INFO.get(f"{p.key}:{p.kind}") or PARAM_INFO.get(p.key) or p.help or ""
