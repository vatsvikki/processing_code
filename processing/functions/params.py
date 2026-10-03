"""Parameter groups shared by the tools and the pipeline steps (no logic, no UI)."""
from __future__ import annotations

from .registry import Param

SHOT = [
    Param("ffid", "Shot FFID", "ffid", 0, live=True, group="Shot",
          help="Scroll through the FFIDs that really exist in the file (read from the trace headers)"),
    Param("sort_by", "Trace order in gather", "choice", "file", live=True,
          choices=["file", "offset", "channel", "receiver", "cdp", "cdp_offset"],
          help="file = common-shot order as recorded (channel order). The others re-sort the display only "
               "(cdp_offset = by CDP, then offset)",
          group="Shot"),
]
DISPLAY = [
    Param("clip_pct", "Amplitude clip percentile", "float", 98.0, min=50, max=100, step=0.5, live=True, group="Display"),
    Param("agc_ms", "AGC window, ms  (0 = off)", "float", 0.0, min=0, step=100, live=True, group="Display"),
    Param("t_min_ms", "Plot from time, ms", "float", 0.0, min=0, step=100, live=True, group="Display"),
    Param("t_max_ms", "Plot to time, ms", "float", 0.0, min=0, step=100, live=True, auto="record_ms",
          help="Starts at the record length in the header (samples x sample interval)", group="Display"),
    Param("fig_height", "Figure height, inches", "float", 7.0, min=3, max=14, step=0.5, live=True, group="Display",
          help="Shorter = the whole gather fits on a small screen (the width follows the page)"),
    Param("trace_min", "First trace position", "int", 1, min=1, step=1, live=True, group="Display"),
    Param("trace_max", "Last trace position", "int", 0, min=1, step=1, live=True, auto="last_trace",
          help="Starts at the number of traces in the selected shot", group="Display"),
]
ANALYSIS = [
    Param("ref_window", "Neighbour window for reference RMS, traces", "int", 21, min=1, step=2,
          help="Each trace is compared with the running median RMS of this many neighbouring live traces "
               "(1 = one global median for the whole shot)", group="Analysis"),
    Param("ref_sort", "Neighbour order for reference RMS", "choice", "offset",
          choices=["offset", "channel", "receiver", "file"],
          help="Independent of the gather display. Offset gives the steadiest reference; "
               "file / channel order mixes near and far traces and flags more", group="Analysis"),
    Param("t_start_ms", "Analysis window start, ms", "float", 0.0, min=0, step=100, group="Analysis"),
    Param("t_end_ms", "Analysis window end, ms", "float", 0.0, min=0, step=100, auto="record_ms", group="Analysis"),
]
DEAD = [
    Param("dead_p2p_max", "Dead if peak-to-peak amplitude <=", "float", 0.0, min=0,
          help="0 = exactly flat / all-zero trace. Raise for a small absolute floor.", group="Dead-trace criteria"),
    Param("dead_zero_frac", "Dead if fraction of zero samples >=  (0 = off)", "float", 0.9, min=0, max=1, step=0.05,
          group="Dead-trace criteria"),
    Param("dead_rel_rms", "Dead if RMS < this x reference  (0 = off)", "float", 0.01, min=0, step=0.005,
          help="Near-dead traces: 0.01 means 40 dB below the neighbours", group="Dead-trace criteria"),
]
BAD = [
    Param("noisy_ratio", "Noisy if RMS > this x reference  (0 = off)", "float", 5.0, min=0, step=0.5,
          group="Bad-trace criteria"),
    Param("weak_ratio", "Weak if RMS < this x reference  (0 = off)", "float", 0.1, min=0, step=0.01,
          group="Bad-trace criteria"),
    Param("spike_crest", "Spike if max|amp| / RMS >  (0 = off)", "float", 30.0, min=0, step=1,
          help="Crest factor; normal traces are ~8-15 in this survey", group="Bad-trace criteria"),
    Param("dc_ratio_max", "DC offset if |mean| / std >  (0 = off)", "float", 1.0, min=0, step=0.1,
          group="Bad-trace criteria"),
    Param("corr_ratio_min", "Low correlation if score < this x local reference  (0 = off)", "float", 0.4, min=0, step=0.05,
          help="Every trace is cross-correlated with its neighbours on the same receiver line (best positive "
               "correlation within the lag range, median over the neighbours). The score is compared with the median "
               "score of traces at similar offsets, so steep near-source arrivals and line ends are not flagged. "
               "0.4 = clearly less coherent than the traces around it", group="Bad-trace criteria"),
    Param("corr_neighbours", "Neighbours on each side for the correlation", "int", 2, min=1, max=6, step=1,
          group="Bad-trace criteria"),
    Param("corr_lag_ms", "Correlation max lag, ms", "float", 40.0, min=0, step=10,
          help="Time shift searched between neighbours: allows for the moveout between adjacent traces",
          group="Bad-trace criteria"),
]

_DISPLAY_KEYS = {p.key for p in DISPLAY} | {"sort_by"}

DECON = [
    Param("operator_ms", "Operator length, ms", "float", 160.0, min=0, step=20, live=True, group="Spiking deconvolution",
          help="Length of the prediction-error filter. Longer = shorter reverberations removed, more risk to real events"),
    Param("prewhite_pct", "Pre-whitening, %", "float", 0.1, min=0, step=0.05, live=True, group="Spiking deconvolution",
          help="White noise added to the autocorrelation zero lag; stabilises the filter (0.1 - 1 % is usual)"),
    Param("design_start_ms", "Design window start, ms", "float", 0.0, min=0, step=100, live=True,
          group="Spiking deconvolution", help="The filter is designed in this window and applied to the whole trace"),
    Param("design_end_ms", "Design window end, ms", "float", 0.0, min=0, step=100, live=True, auto="record_ms",
          group="Spiking deconvolution"),
    Param("balance", "Restore input RMS of every trace", "bool", True, live=True, group="Spiking deconvolution",
          help="Off: raw prediction-error-filter output, amplitudes are lower than the input"),
]
