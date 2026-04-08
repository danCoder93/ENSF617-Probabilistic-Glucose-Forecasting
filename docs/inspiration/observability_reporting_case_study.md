# Observability Reporting Case Study Notes

Role: Preserved inspiration note from an earlier narrative analysis pass.
Audience: Human authors preparing reports, slides, or narrative writeups.
Owns: Historical case-study framing and descriptive commentary on run artifacts.
Related docs: [`../artifact_diagnosis.md`](../artifact_diagnosis.md),
[`paper_material_notes.md`](paper_material_notes.md),
[`../current_architecture.md`](../current_architecture.md).

## OVERVIEW

We took the workflow from something that trained and dumped a bunch of outputs into a folder to something that now leaves behind a much clearer trail of what data it used, how it performed, where it struggled, and which artifacts actually matter for reporting. We added cleaner run-level observability, structured metric summaries, grouped evaluation tables, automatic visual reports, threshold-based accuracy analysis, and event-aware analysis tied back to the processed data context. That gives us a much better handle on both debugging and real model interpretation, and it means the README, final report, and presentation can all build off the same artifact trail instead of us stitching the story together afterward

Observability, Reporting, and Analysis Additions

One of the big goals for this project was to move past a workflow where the model just trains and leaves a pile of outputs behind without much structure. We wanted each run to leave a much clearer trail of what data it used, how the run behaved, what results came out of it, and which artifacts actually matter for debugging, reporting, and final interpretation.

That is what this work added.

At this point, the workflow does a much better job of turning a run into something we can actually inspect, analyze, and reuse in the README, final report, and slides.

1. Core run-level observability
artifacts/main_run/data_summary.json

This file captures dataset-level observability information. Its job is to tell us what data the run was actually built on before we even get into model performance.

It includes things like:

row count
subject count
timestamp coverage
missing-value counts
rows per subject
continuous-column summaries
categorical-column summaries
split sizes
basic data config values

This is important because it gives us a clean, report-ready summary of the dataset and preprocessing footprint instead of forcing us to reverse-engineer that story from raw code or intermediate tables.

From the 3-epoch local run, this summary showed:

300,607 rows
25 subjects
time coverage from 2023-12-08 to 2024-04-07
0 missing values in the modeled columns
143,237 / 27,221 / 28,234 train / validation / test windows.
artifacts/main_run/metrics_summary.json

This is the compact metric artifact for a run. Its job is to give a readable summary of the main evaluation results without needing to dig through logs or the full run summary.

It includes:

the checkpoint used for evaluation
scalar test metrics
structured evaluation output
paths to related artifacts such as prediction exports and reports

From the 3-epoch local run, the main test results were:

test MAE = 14.08 mg/dL
test RMSE = 20.97 mg/dL
prediction interval width = 51.59 mg/dL
predicted and target means were closely aligned, which suggests limited aggregate mean bias.
artifacts/main_run/run_summary.json

This is the high-level run summary. It acts as the main snapshot of the run setup and records the major artifact locations and important runtime choices.

It includes:

resolved runtime configuration
device profile information
optimizer settings
data summary linkage
metrics summary linkage
prediction artifact linkage
report artifact linkage

This is the best place to look if someone wants one compact summary of the run.

artifacts/main_run/report_index.json

This file acts as the artifact map for the run. It points to the major outputs produced during execution so that someone opening the folder does not have to guess where everything lives.

It links to:

run_summary.json
data_summary.json
metrics_summary.json
grouped metric CSVs
prediction artifacts
generated reports
logging outputs

This made the artifact folder much easier to navigate and gave us a cleaner entry point for reporting.

1. Prediction export improvements
artifacts/main_run/test_predictions.pt

This is the raw tensor export of the model predictions. It preserves the direct model outputs and is most useful for debugging or lower-level analysis in Python.

artifacts/main_run/test_predictions.csv

This is the flat, analysis-friendly prediction export. It turns the raw prediction tensors into a row-per-horizon table that is much easier to inspect, plot, and reuse in reporting.

The table includes:

batch and sample indices
subject_id
decoder start and end
timestamps
horizon index
target
quantile predictions
median prediction
residual
prediction interval width

We also updated the export logic so it can include the last observed glucose from the encoder/history window going forward. That change matters because it unlocks clean persistence-baseline benchmarking later without changing training logic.

This was an important step because it made the export more useful for downstream analysis rather than just acting as a thin dump of forecast values.

1. Grouped evaluation artifacts

To make the results easier to interpret, we added grouped metric exports so the run leaves behind more than one average score.

artifacts/main_run/metrics_by_horizon.csv

This file breaks evaluation down by forecast horizon.

Its purpose is to show:

how MAE changes as we predict farther into the future
how RMSE changes with horizon
how interval width changes with horizon
whether forecast reliability drops smoothly or suddenly

This turned out to be one of the most useful artifacts. It showed a very clear error increase with horizon, which gave us a much better understanding of where the model is actually reliable.

artifacts/main_run/metrics_by_subject.csv

This file breaks evaluation down by subject.

Its purpose is to show:

how consistent performance is across individuals
whether the model is equally reliable for everyone
which subjects are easier or harder

This was important because the analysis showed meaningful subject-to-subject variability, which is a much more realistic result than pretending one overall metric tells the whole story.

artifacts/main_run/metrics_by_glucose_range.csv

This file breaks evaluation down by glucose range.

Its purpose is to show whether the model behaves differently in:

low glucose
in-range glucose
high glucose

This ended up being one of the strongest findings in the whole analysis. The model was clearly strongest in the 70–180 mg/dL range and much weaker at the extremes, especially in hypoglycemia. That matters much more than a single overall MAE when thinking about real-world usefulness.

1. Automatic report generation
artifacts/main_run/reports/residual_histogram.html

This report shows the residual distribution. Its purpose is to help us see:

whether errors are centered near zero
whether the model has long tails
whether there are occasional large misses
artifacts/main_run/reports/horizon_metrics.html

This report visualizes how key metrics change by forecast horizon.

Its purpose is to make it immediately obvious that:

short-horizon predictions are much stronger
long-horizon predictions are harder
uncertainty tends to widen as the horizon extends
artifacts/main_run/reports/forecast_overview.html

This report overlays target values, predicted medians, and intervals over time.

Its purpose is to help us qualitatively assess:

whether forecasts track the general trend
whether they lag rapid changes
whether uncertainty bands are informative or just very wide

We also made the report generation path safer by only allowing Plotly report generation when the prediction table actually exists. That way the workflow is more robust and does not assume artifacts were created when they were not.

1. Analysis outputs built on top of the run artifacts

We then took the core run artifacts and built a cleaner analysis layer on top of them.

All of the files below live under:

artifacts/main_run/analysis_outputs/

This folder is where the stronger reporting and interpretation artifacts now live.

1. Metrics analysis workbook and executive summary
artifacts/main_run/analysis_outputs/metrics_analysis_workbook.xlsx

This is the main analysis workbook.

Its purpose is to gather the key result tables and visuals into one place that is easy to inspect and easy to reuse when writing the report.

It includes:

overall headline metrics
grouped metrics by horizon
grouped metrics by subject
grouped metrics by glucose range
derived insights
embedded plots

This made the analysis much easier to share and much easier to translate into the writeup.

artifacts/main_run/analysis_outputs/executive_summary_metrics_analysis.md

This is the short written summary of the run.

Its purpose is to give a quick, readable overview of:

what looks encouraging
what still needs work
what the run means
what the next experiments should be

This was useful because it forced the raw metrics into a cleaner narrative instead of just leaving them as numbers.

1. Core analysis plots
artifacts/main_run/analysis_outputs/horizon_error_plot.png

This plot shows how error grows with forecast horizon.

Its purpose is to make it visually clear that the model is much stronger at short horizons than long ones.

artifacts/main_run/analysis_outputs/glucose_range_error_plot.png

This plot compares performance across glucose ranges.

Its purpose is to show that the model is strongest in the 70–180 mg/dL range and much weaker in low and high glucose regimes.

artifacts/main_run/analysis_outputs/subject_mae_plot.png

This plot shows MAE by subject.

Its purpose is to show subject-to-subject variability and make heterogeneity obvious instead of hiding it inside a single overall average.

1. Threshold-based accuracy analysis

One of the most useful additions was moving beyond only MAE and RMSE and asking a much more practical question:

How often are the predictions actually close enough?

To answer that, we built threshold-based accuracy artifacts.

artifacts/main_run/analysis_outputs/threshold_accuracy_summary.csv

This gives the overall threshold-based summary.

It reports:

overall MAE
percent of predictions within 10 mg/dL
percent within 20 mg/dL
percent within 30 mg/dL

From the 3-epoch run, the results were:

54.17% within 10 mg/dL
76.39% within 20 mg/dL
87.09% within 30 mg/dL

That gave us a much more intuitive way to talk about usefulness than average error alone.

artifacts/main_run/analysis_outputs/threshold_accuracy_by_horizon.csv

This breaks threshold accuracy down by forecast horizon.

Its purpose is to show how “close enough” performance decays as the model predicts farther ahead.

This confirmed that:

short-horizon forecasting is very strong
longer-horizon forecasting is still useful, but much less reliable
artifacts/main_run/analysis_outputs/threshold_accuracy_by_glucose_range.csv

This breaks threshold accuracy down by glucose range.

Its purpose is to show how often predictions stay within practical error bands in:

low glucose
in-range glucose
high glucose

This reinforced one of the clearest findings in the analysis:

the model is strongest in the 70–180 mg/dL range
the model is weakest in low glucose
artifacts/main_run/analysis_outputs/threshold_accuracy_by_horizon.png

This plot shows threshold accuracy curves by horizon.

artifacts/main_run/analysis_outputs/threshold_accuracy_by_glucose_range.png

This plot shows threshold accuracy by glucose range.

artifacts/main_run/analysis_outputs/threshold_accuracy_summary.md

This is the written interpretation of the threshold-based analysis.

Its purpose is to translate the grouped threshold numbers into plain language and make the results easier to explain in the report.

This was a major step because it turned the evaluation into something much easier to understand from a real-use perspective.

1. Event-aware analysis

After that, we added event-aware analysis so we could move past general accuracy and start asking:

when does the model struggle?
are meals harder?
are insulin events harder?
does context matter?

To do that, we joined prediction outputs back to processed data context using:

subject_id
timestamp

and used the processed dataset columns such as:

carbs_g
bolus_insulin_u
correction_insulin_u
meal_insulin_u
device_mode
artifacts/main_run/analysis_outputs/predictions_with_event_context.csv

This is the merged analysis table that combines predictions with event/context fields from the processed dataset.

Its purpose is to support downstream event-aware slicing and future follow-up analyses.

artifacts/main_run/analysis_outputs/metrics_by_device_mode.csv

This groups performance by device mode.

Its purpose is to see whether the model behaves differently in:

regular
sleep
exercise
and any unmatched/missing join cases

This still needs a bit of cleanup before it should be reported too strongly, because one label looked suspicious and likely reflects a join or encoding artifact.

artifacts/main_run/analysis_outputs/metrics_by_meal_event.csv

This groups performance into:

meal-event windows
non-meal windows

Its purpose is to quantify whether the model performs worse when meals are happening.

The result was clear:

meal windows were harder than non-meal windows
artifacts/main_run/analysis_outputs/metrics_by_insulin_event.csv

This groups performance into:

insulin-event windows
non-insulin windows

Its purpose is to quantify whether the model struggles more around insulin-related events.

This turned out to be one of the strongest diagnostic findings:

insulin-event windows were substantially harder than non-insulin windows
artifacts/main_run/analysis_outputs/event_aware_summary.md

This is the written interpretation of the event-aware analysis.

Its purpose is to summarize the context-specific findings in plain language.

artifacts/main_run/analysis_outputs/mae_by_device_mode.png

This plot shows MAE by device mode.

artifacts/main_run/analysis_outputs/mae_by_event_group.png

This plot shows MAE for:

meal vs non-meal windows
insulin-event vs non-insulin windows

This layer was important because it moved the analysis much closer to real operating conditions. Instead of only knowing that the model has a certain average error, we now know that it is clearly weaker around meals and even more so around insulin events.

That gives us much better guidance for improvement.

1. What the current results say

From the 3-epoch local Apple Silicon run, the overall picture is:

the pipeline works end to end
the artifact trail is much clearer
the model is learning meaningful signal
the model is strongest at short horizons
the model is strongest in the 70–180 mg/dL range
the model is weakest in low glucose
the model is weaker around meals
the model is even weaker around insulin events
there is meaningful subject-to-subject variability
there is at least one suspicious horizon-level coverage value that still needs follow-up

So the honest interpretation is that this is a strong engineering and analysis milestone, not a final-performance claim.

The run confirms that:

the workflow is working
the model is learning
the grouped metrics are useful
the artifact pipeline is strong enough to support real reporting
the next improvements should focus on difficult regimes rather than just overall averages
11. What this work achieved

In plain terms, this work took the project from a place where the model could run and log outputs into a folder, to a place where each run now leaves behind a much clearer record of:

what data it used
how the data was distributed
how the model performed overall
how performance changed by horizon
how performance changed by glucose range
how performance changed by subject
how often predictions were actually close enough to be useful
how performance changed during meals and insulin events
which files matter for debugging
which files matter for reporting
