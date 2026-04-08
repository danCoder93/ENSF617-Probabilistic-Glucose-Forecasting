# Results And Discussion

Role: Canonical results-and-discussion section for the research companion.
Audience: Researchers and collaborators translating current artifact evidence
into manuscript-ready findings later.
Owns: Metric interpretation, uncertainty interpretation, strengths, risks, and
discussion framing for the present implementation.
Related docs: [`methods.md`](methods.md), [`conclusions.md`](conclusions.md),
[`../artifact_diagnosis.md`](../artifact_diagnosis.md),
[`../inspiration/paper_material_notes.md`](../inspiration/paper_material_notes.md).

## Status

This section is still placeholder-friendly in the paper-writing sense. The
repository already produces meaningful evidence, but the final manuscript
discussion should still be written as a human argument rather than as a
mechanical inventory of outputs.

## Where The Current Evidence Lives

The strongest repository-native evidence currently lives in:

- `run_summary.json`
- `report_index.json`
- `metrics_summary.json`
- grouped evaluation tables
- exported prediction artifacts
- the case-study interpretation in [`../artifact_diagnosis.md`](../artifact_diagnosis.md)

Those artifacts already support a serious discussion. What they do not yet
provide is the final paper-shaped narrative that would connect those outputs to
a specific empirical claim.

## How To Read A Quantile Forecast

Suppose that for one future timestep the model predicts:

- \(q_{0.1} = 104\)
- \(q_{0.5} = 120\)
- \(q_{0.9} = 142\)

The most useful way to read that forecast is not to treat the three numbers as
three different targets. They are three summaries of the same conditional
future-glucose distribution.

From a regression point of view, the model has produced a lower estimate, a
central estimate, and an upper estimate for the same future glucose value. From
a distributional point of view, the conditional CDF reaches 0.1 at 104, 0.5 at
120, and 0.9 at 142. From an operational forecasting point of view, the model
has produced a central 80% interval \([104, 142]\) with a median forecast of
120.

Those readings are compatible, not competing. If the realized glucose value is
118, the forecast interval contains the truth and the median forecast error is
small. If the realized value is 155, the upper quantile was too low for that
case and repeated errors of that type would show up later as undercoverage.

## What The Reported Metrics Actually Mean

The repository's metric surface makes the most sense when it is read as a split
between point-forecast summaries and probabilistic summaries.

MAE and RMSE are point metrics. They are computed on the representative point
forecast, which in the current configuration is the median quantile because the
tuple explicitly contains 0.5. MAE captures average absolute error. RMSE gives
larger misses more weight. Mean bias captures signed over- or underprediction.

Pinball loss is different. It evaluates the full quantile tensor and is the
metric that is most faithful to the actual training objective.

The interval statistics fill in the rest of the picture. Prediction-interval
width measures sharpness: narrow intervals are more informative, provided they
are not simply overconfident. Empirical interval coverage measures how often
the truth falls inside the predicted outer interval. With the current
\((0.1, 0.5, 0.9)\) setup, that interval is nominally a central 80% interval,
so coverage near 0.8 is the natural target.

Taken together, those metrics support a much more honest reading of the model
than any single scalar could. The point metrics say how far off the median-like
forecast is. The probabilistic metrics say whether the forecast distribution is
sharp, aligned, and plausibly calibrated.

## What Is Strong About The Current Design

Several parts of the current implementation are already quite strong.

The repository is cleanly layered. Downloading, preprocessing, splitting,
indexing, dataset assembly, model definition, training orchestration,
evaluation, observability, and reporting are all separated instead of being
folded into one training script. That matters because it makes the system
inspectable as research software rather than merely runnable.

The semantic feature contract is another real strength. The distinction between
static, known, observed, and target variables is not just described in prose;
it is reflected consistently in the data layer and the model interface. That
consistency makes the method easier to reason about and reduces accidental
future leakage.

The late-fusion design is also intellectually cleaner than many hybrid models.
The current branch does not pretend that the TCN is merely a preprocessor for
the TFT. It treats TCN and TFT as parallel representational branches and fuses
them in latent space. Whether that design is optimal is still an empirical
question, but as a methodological claim it is coherent.

Finally, probabilistic output is the right default for this task. Glucose
forecasting is uncertain in a way that point-only forecasts flatten too
aggressively. The repository's emphasis on quantiles, interval width, and
coverage makes the method look more mature than a typical "train and report one
error metric" setup.

## What Still Limits The Current Claims

The current branch is disciplined, but its strongest claims should still be
framed carefully.

One limitation is that the code is still transitional in a few places. The
DataModule can still synthesize fallback feature specifications when the full
feature declaration is not populated explicitly. Some architecture code also
retains visible upstream TFT lineage. Neither issue is fatal, but both remind
the reader that the implementation is still consolidating.

Another limitation is that the default split policy answers one scientific
question more strongly than another. Within-subject chronological splitting is
reasonable for personalized forecasting, but it does not establish
generalization to unseen patients. The distinction becomes even more important
because subject identity appears as a static categorical feature in the default
contract. That can help personalized forecasting, but it also changes the
meaning of reported performance.

The current sample-generation strategy raises a similar issue. With stride-1
windows, the effective training set contains many highly overlapping examples.
That is common in forecasting and can be useful, but it means the apparent
sample count can overstate the diversity of the optimization signal.

On the preprocessing side, the branch makes reasonable but nontrivial
assumptions. Basal insulin is forward- and backward-filled as though it were a
stable state on the shared time grid. Event variables are zero-filled as though
the absence of a record means the absence of an event. Those choices are
coherent, but they are still assumptions about the data-generating process.

The probabilistic evaluation layer is also promising but not yet exhaustive.
Interval width and empirical coverage are meaningful first-pass uncertainty
metrics, but they are not the whole calibration story. Coverage by horizon,
quantile reliability plots, calibration curves, and CRPS-style summaries would
all make the uncertainty claims stronger.

Finally, the hybrid architecture itself is plausible rather than conclusively
validated. It is easy to tell a good story about why TCN and TFT should work
well together. It is harder, and more important, to show in ablations that the
extra complexity pays for itself against strong TFT-only or TCN-only baselines.

## How The Discussion Should Eventually Read

The final paper-style discussion should connect four things clearly:

- the headline quantitative results
- how behavior changes by horizon, subject, or glucose range
- what the uncertainty estimates are actually saying
- where the present method is still limited

The current artifact layer is already capable of supporting that discussion.
What remains is less a tooling gap than a prose-and-claim gap.

## Practical Follow-On Questions

The most valuable next empirical questions are fairly concrete:

- How much of the current performance depends on within-subject forecasting
  rather than across-subject generalization?
- How much does subject identity contribute?
- Does the fused model outperform TFT-only and TCN-only baselines cleanly?
- Are the current intervals sharp and well calibrated across the full horizon,
  or only in aggregate?
- How sensitive are the results to preprocessing assumptions such as basal
  filling and event zero-filling?

Those are the questions most likely to determine whether the repository's
current methodological story becomes a strong paper claim or remains mainly an
architectural contribution.

## Interim Reading Path

Until the final manuscript narrative is written, the best supporting path is:

1. [`methods.md`](methods.md)
2. [`../artifact_diagnosis.md`](../artifact_diagnosis.md)
3. [`conclusions.md`](conclusions.md)
4. [`../inspiration/paper_material_notes.md`](../inspiration/paper_material_notes.md)
