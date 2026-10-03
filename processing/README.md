# Shot-gather trace QC (dead / bad traces)

marimo GUI + a separate, GUI-independent function package.

```
processing/
├── app_marimo.py              UI ONLY (marimo) – function panel, parameter form, output panel
├── custom.css                 page layout: full width, fixed top, two scroll panels
├── run_app.sh                 launcher: finds / builds a Python with the packages, HOST / PORT / token options
├── requirements.txt           marimo, numpy, matplotlib
├── .env.example               per-machine settings (SEGY_FILE, PYTHON, HOST, PORT, TOKEN_PASSWORD) -> copy to .env
├── output/                    default folder for saved figures / tables / single-shot SEG-Y
└── functions/                 all logic, no UI imports
    ├── segy_io.py             SEG-Y reader: EBCDIC, binary header, shot lookup by FFID, IBM->IEEE
    ├── geometry.py            source / receiver positions from the trace headers (headers only, memory-mapped)
    ├── grid.py                IL / XL grid: corner points or header words, affine map, saved corner points
    ├── ebcdic.py              EBCDIC / binary / trace header -> table rows (all 88 trace-header words)
    ├── context.py             file facts for widgets: real FFIDs, shot sizes, record length
    ├── detection.py           dead / bad trace tests (numpy)
    ├── correction.py          dead traces: interpolate or remove
    ├── decon.py               spiking (Wiener-Levinson) deconvolution + spectrum / autocorrelation statistics
    ├── pipeline.py            engine of the Processing tools and the Flow: @step registry, run_steps(), PipeState (gather + dead / bad flags)
    ├── steps.py               the flow's functions: QC (detect dead / bad), Processing (correct dead, geometric spreading, spiking decon, bandpass, AGC, top mute, sort) and Display (plot gather, acquisition geometry / fold)
    ├── params.py              parameter groups shared by tools and steps
    ├── saving.py              figures -> PNG / PDF / SVG, tables -> CSV, processed data -> SEG-Y
    ├── segy_write.py          writes a processed shot as SEG-Y (original header bytes, patched with any named header a step changed in memory - IBM or IEEE samples)
    ├── batch.py               applies a flow to every shot of the file and writes one SEG-Y (background job)
    ├── plotting.py            matplotlib figures (return Figure / PNG bytes)
    ├── registry.py            @register + Param/Output: how a function describes itself to a GUI
    └── tools.py               Load Data, EBCDIC & Headers, and the Flow itself (the flow's own functions live in steps.py)
removed_code/                      archived code: the earlier Sort / NMO / Stack / CMP Sort / Save Data / pipeline-UI code, and
                                    the standalone QC tools (Dead Traces, Bad Traces, Dead + Bad QC, Whole-Survey QC) - not
                                    imported by the app; delete when not needed
```

## Run

```bash
./run_app.sh            # http://localhost:2718   (PORT=xxxx ./run_app.sh to change; HOST=0.0.0.0 to share)
./run_app.sh edit       # same, as an editable notebook
```

### Run it on another laptop / desktop / server

1. **Copy the folder** (`processing/`: `app_marimo.py`, `functions/`, `run_app.sh`, `requirements.txt`) to the machine,
   e.g. `rsync -av processing/ user@host:processing/`. Nothing in the code is tied to this Mac.
2. **Run `./run_app.sh`.** It uses the first Python that already has marimo + numpy + matplotlib (`$PYTHON`, `.venv`,
   `python3`, `python`); if none has them it creates `.venv` and installs `requirements.txt` (internet needed once).
3. **Tell it where the data is** - the SEG-Y file is read on the machine that runs the app, so use a path valid there:
   `SEGY_FILE=/data/survey.sgy ./run_app.sh`, or type the path into the *SEG-Y file* box. Output goes to `output/`
   next to the app (changeable in the Save bar).

**One command from this machine:** `./launch_remote.sh [ssh-host] [edit]` (default host `seismic-unix`) copies the code
to `~/processing` on the host, starts the app there, tunnels the port and opens `http://localhost:2718`; Ctrl+C stops
it. Give the data as a path *on the remote*: `SEGY_FILE=/data/survey.sgy ./launch_remote.sh`. Other knobs: `REMOTE_DIR`,
`PORT`, `LOCAL_PORT`, `NO_SYNC=1`.

Settings can be kept in a `.env` file beside `run_app.sh` (copy `.env.example`): `SEGY_FILE`, `PYTHON`, `HOST`, `PORT`,
`TOKEN_PASSWORD`. Variables set in the shell win over `.env`.

**Using the app from a different machine than the one it runs on** - pick one:

| Way | Command | Notes |
|---|---|---|
| **SSH tunnel (safest)** | on the server `./run_app.sh`; on your laptop `ssh -L 2718:localhost:2718 user@server`, then open `http://localhost:2718` | nothing is exposed to the network, no firewall change |
| **Direct on the LAN** | `HOST=0.0.0.0 ./run_app.sh`, open `http://<server-ip>:2718/?access_token=...` | a login token is switched on automatically and the URL with the token is printed; set `TOKEN_PASSWORD=...` for a fixed password; allow the port in the firewall |
| **Keep running after logout** | `HOST=0.0.0.0 nohup ./run_app.sh > app.log 2>&1 &` (or inside `tmux` / `screen`) | |

`./run_app.sh edit` on a non-local `HOST` gives whoever has the token full code access to the machine - keep the token
secret, and never expose the port to the open internet (use the SSH tunnel or a VPN instead).

**First press Load Data → ▶ Load data** (Flow card, left panel, **＋ Add a function** - **Load Data** is the first entry
of the list). The file in the *SEG-Y file* box is read only then: a progress bar at the top of the page shows the
headers being read and the shots being indexed (about 2 s for 950 000 traces), and the page then lists the file summary
and every shot (FFID, first trace, traces). All other functions wait for it (they say so) and work on the loaded file;
a new path in the box needs a new Load. Pressing Load again re-reads the file from disk.

There is no top-bar menu at all - **every function of the app**, whatever it does, is picked from the same **Flow card**
in the left panel: click **＋ Add a function** to drop one flat list down, Load Data first. Almost every function
**adds to the flow** when clicked (see "Flow" below) - all the Processing and QC functions, and also the "display"
ones, **Plot Shot Gather**, **Acquisition Geometry** and **Acquisition Fold**: they show a gather / map for that
point of the flow but do not change the data, so a step after them still gets whatever came in unchanged.
Only **Load Data** and **EBCDIC & Headers** (the 40-line EBCDIC text header, the full binary header, and all 88 words
of the 240-byte trace header of the chosen *Shot FFID* + *Trace in shot*, followed by min / max / number of distinct
values of every trace-header word over that shot) **open on their own** instead, on the right - they read / report on
the file rather than on one flow stage, so they do not fit into a flow.

There is no standalone QC tool any more (no Dead Traces / Bad Traces / Dead + Bad QC / Whole-Survey QC function) - add
*Detect Dead Traces* and/or *Detect Bad Traces* to the flow instead, which mark the flagged traces on the flow's own
gather plot.

**Inline / crossline (IL / XL) grid - the Survey grid of the 📁 Data card.** Asked once, after *Load data*, and used by
every function that needs the data's IL / XL (Acquisition Geometry / Fold, NMO Correction, CDP Stack, the CDP lists and
the 🛠 Tools). It comes from, in this order:
1. **the corner table** of the Data card (rows of IL, XL, X, Y; 3-4 corners; *+ New row* for more): whatever you change
   in it replaces the IL / XL of the headers. `IL, XL` alone (X, Y empty) works when the headers give a grid to place
   them;
2. the **receiver headers**: IL = receiver line (RECLN, byte 173), XL = receiver station (RECSTN, byte 181), fitted
   against the receiver X / Y (a few mis-positioned receivers are ignored and counted);
3. the inline / crossline words (bytes 189 / 193) fitted against the trace midpoints.

The table starts filled in: with the table saved for that file, else with the corners of the grid the headers describe.
Edit any value - what is on screen runs again at once - and the table is saved for that SEG-Y file in
`~/.trace_qc/geometry_grids.json` (`$TRACE_QC_HOME` to change the folder) and comes back next time. *Use the header IL /
XL* (or editing the table back to the header values) removes the saved table. Typed corners that do not fit a regular
grid get a warning. (For this survey the receiver words give a 150 ft x 150 ft grid: lines 128-408, stations 212-630.)
The velocity model's own corner points are asked only in *NMO Correction*, when a velocity model is used.

The corner table as it is usually supplied is kept as the example in the code (`grid.CORNER_EXAMPLE`, this survey's bin
grid, 150 ft x 150 ft); `parse_corners` also accepts it as pasted text (tabs, heading rows, closing row):

```
IL      XL      coords (ft)
                x-coords    y-coords
1001    1001    1971388.1   428622
1001    1686    2074138.1   428622
1404    1686    2074138.1   489072
1404    1001    1971388.1   489072
1001    1001    1971388.1   428622
```


* **Zoom:** drag a box on the gather (Shift + drag = lasso, its bounding box is used). The time and trace range boxes follow.
  **↩ Previous view** steps back one zoom, **🔍 Reset zoom** returns to the full record.
* **FFID** is a scroll bar over the FFIDs that really exist in the file (read from the trace headers).
* Plot ranges start from the header: time 0 → `ns × dt`, traces 1 → number of traces in the selected shot.
* Display settings, FFID and the correction toggle are *live* (the tool re-runs when they change); thresholds are applied
  with the **▶ Apply / ▶ Run** button.
* The gather is shown in common-shot order (as recorded); *Trace order in gather* re-sorts the display by offset, channel,
  receiver, cdp or cdp, then offset (the data are not changed).
* **💾 Save figures** (the *Save this shot* card at the bottom of the results) writes the figures on screen to the chosen
  folder (default `output/`) as **PNG** (choose dpi), **PDF** or **SVG**; tick *also tables* to add the tables as CSV. Files are
  named `<file>_<tool>_FFID<n>_<figure title>_<parameters>.<ext>`, where `<parameters>` lists the settings that differ from their
  defaults as `key-value` (e.g. `clip_pct-95_agc_ms-200_trace_max-300`; `defaults` when nothing was changed; a table such as the
  corner table shows as a short code), so saving the same
  shot with other settings - or a different zoom - never overwrites the earlier image.
* **💾 Save data (SEG-Y)** (the Processing tools) writes the processed shot on screen as a SEG-Y file with IBM (as the input) or
  IEEE samples. The EBCDIC text header, binary header and the full 240-byte trace header of every surviving trace are copied from
  the source file; removed traces are simply not written, and the processing done is noted in the text header lines C37-C40.
  This saves the one shot on screen.
* **🌐 Whole data** (Flow tool, below the Save bar): **▶ Run flow on whole data** runs the flow on every shot from FFID
  to FFID and writes its product as ONE SEG-Y file while it runs (a 15 GB file is never held in memory), in the
  background, with progress and Cancel in the card at the bottom right of the page:
  - a flow of shot steps only -> the processed shots (as recorded, each trace with its original header);
  - a flow with **NMO Correction / CDP Stack** -> the shot steps are applied to every shot first (their
    result is kept in `output/flows/`, so the CDP steps show every CDP of it at once and a later run with the same
    shot steps does not process the shots again), then the LAST CDP step decides the product: the NMO-corrected
    CDP gathers, or the stack (one zero-offset trace per CDP, with the velocity of an NMO Correction
    step above it). Processing steps must come before the first CDP step.
  - Type the output as a full **Output SEG-Y file** path, like the input path (default: next to the input, `<name>_processed.sgy`;
    a folder or an empty box gets an automatic name; a name without extension gets `.sgy`). The job refuses to write onto
    the input file, refuses to overwrite unless you tick *overwrite*, and checks the free disk space first. It writes
    `<out>.part` and renames it only when every shot is done, so a cancelled or failed job leaves no half-written file;
    `<out>_report.csv` lists, per shot, the traces in / out and each function's note.

  The **FFID range** limits the shots used, and the card at the bottom right of the page shows progress / ETA with
  **✖ Cancel**.

Layout (full window width, styles in `custom.css`): a blue **Processing Tool** strip - just the title and the currently
open function's label on the right (its class, e.g. "Display", from `@register(..., category=...)` - the Flow itself
has none) - with no menu of its own; every function is reached from the **Flow** card instead (see above and "Flow"
below). Below the strip: **cards on the left** - the file, then the **Flow** card (always present, regardless of which
function is open - it is where **＋ Add a function** lives), then the open function's own parameters (the *Display*
group of these is its own collapsible, tinted card, set apart from the rest) - and on the right the **gather figure
first** (with the zoom bar, for a function that has one), then the summary / other figures / tables, and at the
**bottom** the *Save this shot* card (with *Whole data* in the Flow). The strip stays put: **only the lower
left and right panels scroll, each with its own scrollbar** (the page itself does not scroll). Wide tables scroll
inside their card, and the figure follows the column width. *Figure height* (Display) makes the gather shorter for
small screens.

**(i) explanations.** Every input has an **ⓘ** next to its label (hover it for the full explanation: what it does, units, default,
effect, when to change it) and every button has a hover tooltip. The texts live in `functions/param_help.py` (`PARAM_INFO` for the
tool / step parameters, `WIDGET_INFO` for the page's own inputs and buttons, `FUNCTION_INFO` for a Flow function's own **ⓘ Full
explanation** - the maths it uses and how each of its parameters changes the result, click to expand).

## Flow

The **Flow** is the only way to run a processing or QC step; there is no standalone tool for any of them, and no
Processing / QC entry on the top bar. Every function of it is a step of `functions/steps.py`:

| Function | Category | What it does |
|---|---|---|
| Detect Dead Traces | QC | flag flat / zero / near-dead traces (used by Correct Dead Traces, marked on the plot) |
| Detect Bad Traces | QC | flag noisy / weak / spiky / DC-offset / low-correlation traces (marked red on the plot) |
| Correct Dead Traces | Processing | interpolate dead traces from live neighbours on the same receiver line, or remove them (found with the default thresholds if Detect Dead Traces did not run first) |
| Geometric Spreading | Processing | gain (t / 1 s)^power that compensates the amplitude decay |
| Spiking Decon | Processing | Wiener-Levinson spiking deconvolution: operator length, pre-whitening, design window, RMS restore |
| Bandpass Filter | Processing | zero-phase trapezoidal band-pass (low cut / low pass / high pass / high cut, Hz) |
| AGC | Processing | automatic gain control over a sliding window |
| Top Mute | Processing | zero everything above t = t0 + \|offset\| / velocity, with a taper |
| Plot Shot Gather | Display | the plain gather at this point of the flow (no maths) - a labelled checkpoint, changes nothing |
| Acquisition Geometry | Display | plan view of every source / receiver and this shot's spread, with the IL / XL grid - changes nothing |
| Acquisition Fold | Display | fold map (traces per bin, whole file) on the same canvas as Acquisition Geometry - changes nothing |
| NMO Correction | Display | NMO of one CDP gather with a velocity-model SEG-Y (nearest model trace by position, RMS / interval, time / depth, units, the model's corner points) or a velocity function (time / Vrms table), and an NMO-stretch mute: CDP gather, NMO stretch section, velocity used, and the data-over-model grid overlay - as in the brute-stack notebook; changes nothing |
| CDP Stack | Display | stacks what the steps above produced, nothing more: the NMO-corrected gathers of an NMO Correction step above (mean of the live samples, its velocity and mute - NMO is not done again) or the plain mean of the CDP gathers; preview of one CDP, and after *Run flow on whole data* the IL / XL stacked sections, velocity sections and overlay |

The two QC functions only flag traces (for the functions after them, and for the flow's own gather plot); the five
Display functions only show a gather / map / table for that point in the flow (`state.view_extra`, see "Add a step"
below) - neither changes the data, so whatever comes after either kind of function still gets the same input the
function itself got. To see the traces in another order use the *Trace order in gather* display setting (it changes
only the plot).

## 🛠 Tools

Whole-file utilities, independent of the Flow, in the **🛠 Tools** menu of the top bar. Each opens in its own browser
tab with the main window's file (and its own ⓘ full explanation):

| Tool | What it does |
|---|---|
| CDP Sort | the brute-stack notebook's CDP sort of the raw file: every trace by CDP, then offset; any CDP's gather, the table of all CDPs (fold, offsets, midpoint, IL / XL), and optionally a CDP-sorted copy of the file |
| CDP Recalculate | the CDP of every trace rebuilt from its midpoint and an IL / XL grid you enter (saved for next time): corner points, the four CDP plots, and optionally a copy of the file with the new CDP |
| Shot Geometry QC | the notebook *shot_geometry_qc_marimo.py* as it is: linear-velocity overlay per shot, fix toggles / shifts, first-break picks, whole-survey audit, maps, corrected copy (served on the app's port + 1, with progress bars) |

It is built in the **Flow** card, in the left panel, below the file card - shown once a file is loaded, and the same
regardless of which function above is currently open, so a flow can be built while looking at another one:

* **Flow (top to bottom)** - every function already in the flow, with **▲ ▼** (move it one place up / down, its
  parameters move with it), a red **−** (remove it; the functions below move up and keep their parameters) and its own
  collapsible parameters (with, in the same accordion, an **ⓘ full explanation** of the maths it uses and how each
  parameter changes the result).
* **＋ Add a function** - at the end of the list; click it to drop down the full list of functions right there, each
  with its description and its own **ⓘ Full explanation (maths & parameters)**. Click a function's button (not its ⓘ)
  to add it at the end of the flow and close the list; the ＋ then reappears after the new function, ready for the next
  one. A function may be used more than once; the flow holds up to 12.
* **▶ / ▶ Run all** - nothing runs by itself: changing a parameter only records it (that function and the ones below are
  shown as "not run yet"). **▶** beside a function runs the flow up to it on the selected shot; **▶ Run all** runs it
  all. Results are kept per shot and reused while the settings above are unchanged, so only what changed is computed;
  display settings only redraw.
* **View stage** - the gather after any function (0 = the untouched input) with the spectrum / autocorrelation comparison
  against the input. *Save figures*, *Save data (SEG-Y)* and *Run flow on whole data* all work on the stage on screen
  (so *View stage* also decides how many of the flow's functions they apply - a change to the flow shows its last
  stage again, so normally the whole flow).

Interacting with the flow (adding, moving, removing, changing a parameter or the view stage) always switches the right
panel to the flow's own result. The functions run top to bottom on the selected shot.

A flow is plain data, so it works without the GUI (and `BatchJob` applies it to a whole file):

```python
from functions import segy_io, run_steps
g = segy_io.read_shot("file.sgy", 398)
stages = run_steps(g, [{"step": "detect_dead"},
                       {"step": "correct_dead", "params": {"mode": "interpolate"}},
                       {"step": "spiking_decon", "params": {"operator_ms": 200, "prewhite_pct": 0.5}}])
stages[-1].state.gather.data      # result; stages[0] is the input

from functions import BatchJob
BatchJob("file.sgy", flow, "out.sgy", ffid_from=398, ffid_to=1982).start()     # whole file, background thread
```

Add a step in `functions/steps.py` (`category="Processing"`, `"QC"` or `"Display"`) and it is offered in the Flow's ＋
list at once - give it an entry in `functions/param_help.FUNCTION_INFO` (keyed by its step key) for the ⓘ full
explanation:

```python
@step("My Step", "what it does", params=[Param("gain", "Gain", "float", 2.0)], order=50)
def my_step(state, gain=2.0):
    g = state.gather
    new = ShotGather(g.ffid, g.i0, g.data * gain, g.headers, g.dt_ms)
    return replace(state, gather=new), f"scaled by {gain:g}"
```

A `category="Display"` step shows something other than the processed gather (a map, typically) instead of changing
it: leave `state.gather` as it is and set `state.view_extra` to the `Output`s to show for this stage (`markdown(...)`,
`table(...)`, `figure(...)` - see `acquisition_fold_step` for a worked example); `run_steps()` clears `view_extra`
before every step runs, so it only applies to the one stage that set it.

## Detection logic

Every trace is measured inside an analysis window (RMS, peak-to-peak, zero-sample fraction,
crest factor, DC ratio) and compared with a **reference RMS** = running median RMS of its
neighbouring live traces (neighbours taken in offset order by default, independent of how the
gather is displayed; the gather itself is shown in common-shot order as recorded).
A threshold of `0` switches that single test off.

| Class | Test | Parameter |
|---|---|---|
| **Dead** (blue ◆) | flat / all-zero trace | peak-to-peak ≤ x |
| | mostly zeros | zero-sample fraction ≥ x |
| | near-dead | RMS < x · reference |
| **Bad** (red ▼) | noisy | RMS > x · reference |
| | weak | RMS < x · reference |
| | spike | max\|amp\| / RMS > x |
| | DC offset | \|mean\| / std > x |
| | low correlation | neighbour correlation < x · local reference (0.4) |

**Low correlation:** every live trace is cross-correlated with its nearest live neighbours (2 on each side) on the
same receiver line, searching ±40 ms of moveout shift; its score is the median of the best positive correlations.
The score is compared with the running median score of traces at similar offsets, so steep near-source arrivals and
receiver-line ends are not flagged just for being less coherent. It catches noisy / spiky / scrambled traces that
have a normal amplitude, and hints "(polarity reversed?)" when the flipped trace would correlate well. On this
survey ratio 0.4 flags ~1 % of the traces (0.3 ~0.4 %, 0.5 ~2.4 %); roughly half of the extra traces it finds look
genuinely noisy on inspection, the rest are mostly line-edge traces. It makes *Detect Bad Traces* markedly slower.

A dead trace is reported as dead only. Defaults were tuned on FFIDs 398–1982 of this survey
(normal crest factor 8–15, so spike limit 30) - tune them by adding *Detect Dead Traces* / *Detect Bad Traces* to the
flow and watching what they mark on the gather.

## Add a new tool (Data class, no UI change needed)

A processing, QC or "display" (map / gather) function belongs in `functions/steps.py` as a flow step (see "Flow"
above, including for one that reads the whole file rather than one shot, like Acquisition Fold). This - a tool of its
own, outside the flow, opening on its own the way Load Data and EBCDIC & Headers do - is only for something that does
not fit a flow stage at all, in `functions/tools.py`:

```python
@register("My Tool", "what it does", params=[Param("thr", "Threshold", "float", 3.0, min=0)], order=60)
def my_tool(path: str, thr: float = 3.0):
    g = segy_io.read_shot(path, 0)
    return [markdown("Result", f"{g.ntr} traces"), image("Plot", png_bytes)]
```

It appears as a new button with an auto-generated parameter form. Return `markdown(...)`,
`table(...)` and/or `image(...)` outputs. To use another GUI (Qt, Streamlit, ...), loop over
`functions.get_functions()`: each has `.label`, `.description`, `.params`, `.run(path, **values)`.

## Use without a GUI

```python
from functions import segy_io, detection, plotting
g   = segy_io.read_shot("file.sgy", ffid=398)
res = detection.run_qc(g, noisy_ratio=5, weak_ratio=0.1)
plotting.plot_gather(g, res).savefig("gather.png")
```

Notes: shots are found by binary search on FFID (file is in field-record order), so opening a shot
in a 15 GB file takes < 1 s. Trace-header bytes follow this survey's EBCDIC header (FFID 9, OFFSET 37,
RECLN 173, RECSTN 181).
