Snapshot of the code removed when the standalone QC tools (Dead Traces, Bad Traces, Dead + Bad QC,
Whole-Survey QC) were dropped from the app: the user asked for the QC category to be folded into the
Flow's "pick a function" list, then asked to remove that "open a QC report on its own" section entirely.

  survey.py    whole-file dead/bad scan (functions.survey.scan_survey), used only by the removed
               "Whole-Survey QC" tool; not imported anywhere else, so it is not part of the package
               any more (see functions/__init__.py, functions/tools.py).

The QC *steps* Detect Dead Traces / Detect Bad Traces (functions/steps.py) are unaffected - they still
work as functions of the Flow, same as before.  The math these tools used (functions/detection.py:
run_qc, flagged_table) is unchanged and still used by those two steps and by Correct Dead Traces.

The tool functions themselves (detect_dead_traces, detect_bad_traces, trace_qc, survey_qc) and their
`_qc()` helper were simply deleted from functions/tools.py (they were short and fully reconstructable
from functions/detection.py + functions/plotting.py's plot_qc_metrics / plot_correlation_metrics /
plot_survey_counts, which were left in place since removing them isn't necessary and they cost nothing
sitting unused).  Delete this folder when you no longer need it.
