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
    ├── pipeline.py            flow machinery: @step registry, run_steps(), PipeState (gather + dead / bad flags)
    ├── steps.py               the pipeline steps (detect dead / bad, correct dead, spiking decon)
    ├── params.py              parameter groups shared by tools and steps
    ├── survey.py              dead / bad scan of every shot in the file
    ├── saving.py              figures -> PNG / PDF / SVG, tables -> CSV, processed data -> SEG-Y
    ├── segy_write.py          writes a processed shot as SEG-Y (original headers kept, IBM or IEEE samples)
    ├── batch.py               applies a flow to every shot of the file and writes one SEG-Y (background job)
    ├── plotting.py            matplotlib figures (return Figure / PNG bytes)
    ├── registry.py            @register + Param/Output: how a function describes itself to a GUI
    └── tools.py               the functions shown as buttons
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

**First press Data → Load Data → ▶ Load data.** The file in the *SEG-Y file* box is read only then: a progress bar at the
top of the page shows the headers being read and the shots being indexed (about 2 s for 950 000 traces), and the page
then lists the file summary and every shot (FFID, first trace, traces). All other tools wait for it (they say so) and
work on the loaded file; a new path in the box needs a new Load. Pressing Load again re-reads the file from disk.

Tools: **Load Data** · **EBCDIC & Headers** (the 40-line EBCDIC text header, the full binary header, and all 88 words of
the 240-byte trace header of the chosen *Shot FFID* + *Trace in shot*, followed by min / max / number of distinct values of
every trace-header word over that shot) · **Acquisition Fold** (fold map = traces per bin, counted from the source-receiver midpoints of every trace; same figure
size and axes as the geometry plot, with the colour bar inside the figure; bins are the IL / XL grid cells, *Bin
alignment* "auto" puts the midpoints at the bin centres; the first Run reads all trace headers, ~20 s for 950 000 traces,
then everything is instant; uses the same corner table as the geometry tool) · **Acquisition Geometry** (plan view of every source and the receiver positions, in the same style as the gather
figures; the selected shot's source and spread are highlighted, plus a second figure with its receiver spread relative to the source;
uses the *Figure height* and Save options like the other plots) · **Plot Shot Gather** · **Dead Traces** · **Bad Traces** · **Dead + Bad QC** · **Whole-Survey QC** (dead and bad
trace counts for every shot in the file, ~15 s for 950 000 traces; press Run) · the **Processing** tab: **CMP Sort** (own
tool, see below) and the processing functions **Correct Dead Traces**, **Geometric Spreading**, **Spiking Decon**, **Bandpass
Filter**, **AGC**, **Top Mute**, **Sort Traces**, **NMO Correction**, **Stack Traces** and **Save Data** - clicking one of these
**adds it to the flow** (see "Flow" below) 

**Inline / crossline (IL / XL) grid in Acquisition Geometry.** The grid comes from, in this order:
1. **the corner-point table** (rows of IL, XL, X, Y; 3-4 corners; use *+ New row* for more): whatever you change in it
   replaces the IL / XL of the headers, even when the headers have IL / XL. `IL, XL` alone (X, Y empty) works when the
   headers give a grid to place them;
2. the **receiver headers**: IL = receiver line (RECLN, byte 173), XL = receiver station (RECSTN, byte 181), fitted
   against the receiver X / Y (a few mis-positioned receivers are ignored and counted);
3. the inline / crossline words (bytes 189 / 193) fitted against the trace midpoints.

The table starts filled in: with the table saved for that file, else with the corners of the grid the headers describe
(an untouched table means "use the headers" and saves nothing). Edit any value and press Apply: the table is saved for
that SEG-Y file in `~/.trace_qc/geometry_grids.json` (`$TRACE_QC_HOME` to change the folder) and comes back filled in next
time. *Back to the header IL / XL* (or editing the table back to the header values) removes the saved table. If the
headers form no grid, the tool says why and asks for the table. Typed corners that do not fit a regular grid get a
warning. With a grid it draws the IL / XL lines and outline, offers IL / XL tick labels, and lists the ranges, spacing,
azimuth and the IL / XL of the selected shot's source and receivers. (For this survey the receiver words give a 150 ft x
150 ft grid: lines 128-408, stations 212-630.)

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
  defaults as `key-value` (e.g. `clip_pct-95_agc_ms-200_trace_max-300`; `defaults` when nothing was changed; for the Pipeline
  e.g. `steps-agc_gain+top_mute_view-2`; a table such as the velocity or corner table shows as a short code), so saving the same
  shot with other settings - or a different zoom - never overwrites the earlier image.
* **💾 Save data (SEG-Y)** (Pipeline and the processing tools) writes the processed shot on screen - for the Pipeline the stage
  chosen in *View stage* - as a SEG-Y file with IBM (as the input) or IEEE samples. The EBCDIC text header, binary header and the
  full 240-byte trace header of every surviving trace are copied from the source file; removed traces are simply not written, and
  the processing steps are noted in the text header lines C37-C40. This button saves the one shot on screen.
* **🌐 Apply to whole data** (below the Save bar) first asks **"Is the flow ready?"** with the steps listed: check the result
  on the shot, answer *Yes, the flow is ready*, and only then the output, FFID range and the run button appear (*Not ready - keep
  editing* takes them away again, and changing the flow asks the question again). It then repeats the steps on screen
  (Pipeline: the stages up to *View stage*; Spiking Decon; Correct Dead Traces ...) on **every shot from FFID to FFID** and
  writes ONE SEG-Y file while it runs (a 15 GB file is never held in memory). Type the output as a full **Output SEG-Y file**
  path, like the input path (default: next to the input, `<name>_processed.sgy`; a folder or an empty box gets an automatic name;
  a name without extension gets `.sgy`). The job refuses to write onto the input file, refuses to overwrite unless you tick
  *overwrite*, and checks the free disk space first. It writes `<out>.part` and renames it only when every shot is done, so a
  cancelled or failed job leaves no half-written file. A bar at the top of the page shows progress and ETA with **✖ Cancel**;
  `<out>_report.csv` lists, per shot, the traces in / out and each step's note. A *Save Data* step in the flow also writes a
  checkpoint file with every shot's data at that point. Timing on this survey: ~37 s of computing for the 3-step flow over all
  951,938 traces (8 threads), plus the disk write.

Layout (full window width, styles in `custom.css`): a blue **Processing Tool** strip, and below it the **function panel** with one
button per class of functions (Data · Display · QC · Processing; a function joins a class with `@register(..., category="QC")`).
Click a class to drop down its functions, click a function to open it (the list closes); the bar shows the open function on the
right. The flow builder (Pipeline) is opened by the **＋ at the file path**. Under the function panel: **cards on the left** for the
file and the selected function's parameters, and on the right the **gather figure first** (with the zoom bar), then the summary /
other figures / tables, and at the **bottom** the *Save this shot* and *Apply to whole data* cards. The strip and the function panel
stay put: **only the lower left and right panels scroll, each with its own scrollbar** (the page itself does not scroll). Wide
tables scroll inside their card, and the figure follows the column width. *Figure height* (Display) makes the gather shorter for
small screens.

**(i) explanations.** Every input has an **ⓘ** next to its label (hover it for the full explanation: what it does, units, default,
effect, when to change it) and every button has a hover tooltip. The texts live in `functions/param_help.py` (`PARAM_INFO` for the
tool / step parameters, `WIDGET_INFO` for the page's own inputs and buttons).

## CMP Sort (Processing class)

The CMP sort of the brute stack: every trace goes to the **CMP bin** (CDP gather) of the IL / XL grid its source-receiver midpoint
falls in - the bins of the fold map, so it takes the same **corner-points table** and *bin alignment* - and inside a bin the
traces are ordered by offset. *Sort traces by*: **CMP bin, then offset** (the CMP sort, default), **CMP bin (cdp)** (grouped by bin,
file order inside) or **offset** only. The sort follows the bins, **not** the CDP number of the trace header (that can follow
another bin layout - in this survey 22 % of the bins mix header CDP numbers - which is why a sort by the header number looked
wrong); the header CDP is shown next to the bins. Only an index is sorted, no samples are copied.

It shows the CMP gather of one bin (IL / XL, 0 = the highest fold) in that order, the table of the bin's traces, the sorted order
of the whole file from that bin on, and the **QC of a proper sort**: a figure of four panels and a table of checks, each marked
✅ / ⚠ / ℹ with what it means:

| Check | What it looks for |
|---|---|
| Traces outside the bins | midpoints outside the corner points (corner points too small) |
| Fold, low-fold bins, holes in the coverage | bins that would stack poorly; empty bins between filled bins of a line |
| Near / far offset coverage | bins with no near traces or no far traces |
| **Bins in order, offsets ascending inside every bin** | the sorted file: the bin number and, inside a bin, the offset never decrease |
| Duplicate traces | the same FFID + channel twice |
| Offset header vs coordinates | the offset word against the distance between source and receiver (tolerance is a parameter) |
| Zero coordinates | traces with source or receiver coordinates all zero |
| Header CDP inside a bin | whether the CDP header follows these bins (information only) |
| **Neighbouring offsets look alike (data)** | in 40 sampled CMP gathers, traces that neighbour in offset correlate far more (here 0.45) than random pairs (0.22): a wrong sort or mixed gathers would not |
| Sort order chosen | the key never decreases along the sorted order |

The first run reads the headers of every trace (~20 s for 950 000 traces, then kept); the QC then takes ~2 s. The per-shot
**Sort Traces** step labels the second row of its gather with the key it sorted by (offset, cdp ...), so a sorted gather shows
ascending values instead of channel numbers.

## Flow (the processing functions in a chain)

The flow is a plain list, opened by the **＋ at the file path** and built by **clicking a function in the Processing tab**: each click
adds that function at the end of the flow (up to 12). Each function in the list has **▲ ▼** (move it one place up / down; its
parameters move with it), a red **−** (remove it; the functions below move up and keep their parameters) and its own
(collapsible) parameters. There is no separate add button, dropdown or picker. The functions run from top to bottom on the
selected shot; *View stage* shows the gather after any of them (0 = untouched input) with a spectrum / autocorrelation comparison
against the input, and *Save data (SEG-Y)* / *Apply to whole data* work on it. The two QC steps *Detect Dead / Bad Traces* are not
offered (Correct Dead Traces finds the dead traces itself with the default thresholds).

| Step | What it does |
|---|---|
| Detect Dead Traces | flags flat / zero / near-dead traces (marked on the plot) |
| Detect Bad Traces | flags noisy / weak / spiky / DC-offset traces |
| Correct Dead Traces | interpolate them from live neighbours on the same receiver line, or remove them |
| Spiking Decon | Wiener-Levinson spiking deconvolution: operator length, pre-whitening, design window, RMS restore |
| Geometric Spreading | gain (t / 1 s)^power that compensates the amplitude decay |
| Bandpass Filter | zero-phase trapezoidal band-pass (low cut / low pass / high pass / high cut, Hz) |
| AGC | automatic gain control over a sliding window |
| Top Mute | zero everything above t = t0 + \|offset\| / velocity, with a taper |
| Sort Traces | reorder the traces by offset, channel, receiver line / station, **cdp** (the CDP number of the header) or **cdp, offset** (by CDP, and by offset within a CDP); ascending or descending |
| NMO Correction | normal-moveout correction with a 1-D velocity table (+ stretch mute) |
| Stack Traces | stack (mean or sum) all traces of the gather into one trace, normally after NMO |
| Save Data | write the data **as it is at this point of the flow** to a SEG-Y file (source trace headers kept; folder, name, sample format). Off by default (*Write file*), because the pipe re-runs at every change - tick it to save this shot. In *Apply to whole data* it always writes a checkpoint file with EVERY shot's data at that point, next to the final output. The name carries a code of the steps before it, so another flow never overwrites it. Also a tool in the Processing class |

The steps after Detect Bad Traces are all processing steps (no QC); the two Detect steps are only there to feed
*Correct Dead Traces*. A step may change the trace count (Correct Dead Traces "remove", Stack Traces) or the order (Sort Traces):
*Save data (SEG-Y)* and *Apply to whole data* follow it, writing each output trace with the header of the source trace it
came from (a stacked shot gets the header of its middle trace).

A flow is plain data, so it works without the GUI:

```python
from functions import segy_io, run_steps
g = segy_io.read_shot("file.sgy", 398)
stages = run_steps(g, [{"step": "detect_dead"},
                       {"step": "correct_dead", "params": {"mode": "interpolate"}},
                       {"step": "spiking_decon", "params": {"operator_ms": 200, "prewhite_pct": 0.5}}])
stages[-1].state.gather.data      # result; stages[0] is the input
```

Add a step (no UI change needed) in `functions/steps.py`:

```python
@step("My Step", "what it does", params=[Param("gain", "Gain", "float", 2.0)], order=50)
def my_step(state, gain=2.0):
    g = state.gather
    new = ShotGather(g.ffid, g.i0, g.data * gain, g.headers, g.dt_ms)
    return replace(state, gather=new), f"scaled by {gain:g}"
```

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
genuinely noisy on inspection, the rest are mostly line-edge traces. It makes the whole-survey scan ~4x slower.

A dead trace is reported as dead only. Defaults were tuned on FFIDs 398–1982 of this survey
(normal crest factor 8–15, so spike limit 30) – tune them with the RMS-vs-threshold plot.

## Add a new function (no UI change needed)

In `functions/tools.py`:

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
