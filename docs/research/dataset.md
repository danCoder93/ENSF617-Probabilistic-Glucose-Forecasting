# Dataset

Role: Canonical research-side reference for dataset context, preprocessing, and
the data contract.
Audience: Researchers and collaborators who want the full dataset and data
handling story without leaving the research folder.
Owns: Dataset provenance, canonical table semantics, preprocessing logic,
feature roles, sequence construction, and the batch interface emitted to the
model.
Related docs: [`introduction.md`](introduction.md), [`methodology.md`](methodology.md),
[`results_and_discussion.md`](results_and_discussion.md),
[`../primer/data_pipeline_walkthrough.md`](../primer/data_pipeline_walkthrough.md),
[`references.md`](references.md).

## Dataset Provenance And Research Context

The current branch is built around the **AZT1D** dataset. The default data URL
in `DataConfig` points to the public Mendeley release, and the surrounding
docs and code treat AZT1D as the present working dataset for the repository.
The release describes a real-world type 1 diabetes cohort with data from
25 individuals using automated insulin delivery systems. The available signals
include glucose values, insulin delivery, carbohydrate intake, and device-mode
information, with glucose sampled on a 5-minute cadence. That aligns with the
repository's default `sampling_interval_minutes = 5`.

This dataset matters for more than convenience. It is rich enough to support a
serious multimodal forecasting problem, because it carries not only glucose and
meal information but also basal insulin, total bolus insulin,
correction-specific insulin, meal-related insulin, device mode, and bolus type.
At the same time, it is clearly a real clinical-control dataset rather than a
machine-learning toy. That means duplicated timestamps, inconsistent strings,
persisting state variables, and event variables that should not be propagated
forward are all part of the raw reality. Much of the data layer exists to turn
that raw timeline into a contract the model can learn from without erasing the
meaning of the original signals.

## From Raw Export To Canonical Table

The first job of the data pipeline is not modeling. It is normalization. The
`AZT1DPreprocessor` takes vendor-shaped exports and rewrites them into one
canonical processed CSV.

It deliberately does not split the data, build windows, or assemble tensors.
Its job is narrower:

> convert raw exports into one stable processed table with canonical names

The canonical columns are:

- `subject_id`
- `timestamp`
- `glucose_mg_dl`
- `basal_insulin_u`
- `bolus_insulin_u`
- `correction_insulin_u`
- `meal_insulin_u`
- `carbs_g`
- `device_mode`
- `bolus_type`
- `source_file`

The preprocessor maps raw fields such as `EventDateTime`, `Basal`,
`TotalBolusInsulinDelivered`, `CorrectionDelivered`, `FoodDelivered`, and
`CarbSize` into that vocabulary. It also accepts multiple raw glucose spellings,
including `CGM` and `Readings (CGM / BGM)`.

This is one of the quiet strengths of the repository. Without canonicalization,
every downstream stage would need to reason repeatedly about raw naming,
raw-file structure, and dataset-specific quirks. The canonical table lets the
rest of the pipeline speak one internal language.

## What The Main Columns Mean

The cleaned table is best read as a harmonized patient timeline, where each row
represents one standardized time point on the shared 5-minute grid.

### Identity, time, and provenance

`subject_id` identifies the patient. In the raw archive, that identity is
inferred from folder structure; once extracted, it becomes a regular column.

`timestamp` is the temporal anchor for the row. It is later converted to a
datetime value, rounded to the nearest minute, sorted, and used for continuity
checks.

`source_file` is not a predictive physiological feature. It is there so that
the canonical table retains provenance and can still be debugged against the
original export.

### Glucose as the target signal

`glucose_mg_dl` is the supervised target variable. It is obtained from
continuous glucose monitoring systems, and that detail matters. CGM glucose is
not a direct blood measurement. It is a high-frequency time series sampled from
interstitial fluid, which means it is a delayed and smoothed proxy of blood
glucose rather than an instantaneous reading. The lag is often modest, but it
becomes especially visible during rapid glycemic change.

That physiological fact shapes the forecasting problem. From the model's point
of view, glucose is not just a number to regress against. It is a lagged
physiological signal whose recent history carries important information about
where the trajectory is going next. The AZT1D materials also frame CGM as the
central signal for variability analysis, trend analysis, and downstream
prediction, which matches its role here.

### Insulin signals

The insulin columns are easier to understand together than separately because
they describe different control actions in the same closed-loop setting.

`basal_insulin_u` is the continuous background insulin signal. Clinically, it
exists to stabilize glucose during fasting conditions and to counter endogenous
glucose production such as hepatic glucose output. In the data, basal insulin
behaves like a state variable rather than like an event. Hourly basal values
are repeated across the corresponding 5-minute intervals, producing a
piecewise-constant signal. From a modeling standpoint, this is the column most
closely associated with slow background drift and longer-horizon stability.

`bolus_insulin_u` is different. It is the total event-level insulin delivered
at specific times, usually either to cover meals or to correct elevated
glucose. In the aligned time series it becomes a sparse spike-like signal, with
zero values at most timesteps and non-zero values only where a dose was
delivered.

The repository then splits total bolus into two more specific columns.
`correction_insulin_u` captures the reactive part of bolus delivery, meaning
the insulin given in response to high glucose. `meal_insulin_u` captures the
anticipatory part, meaning insulin delivered to cover carbohydrate intake. That
distinction is not bookkeeping for its own sake. It matters because
meal-related insulin and correction insulin play different causal roles in the
future trajectory. One is a feedforward action tied to expected disturbance.
The other is a feedback action tied to observed deviation.

On the data side, both `correction_insulin_u` and `meal_insulin_u` are treated
as sparse event features. The values are aligned to the 5-minute grid and
zero-filled when no event exists at a timestep. In practice, that means most
sample windows contain long runs of zeros punctuated by localized spikes.

### Carbohydrate input

`carbs_g` represents carbohydrate intake in grams and is one of the most
causally important exogenous signals in the dataset. Physiologically, it is the
main driver of postprandial glucose excursions. Its effect is delayed because
digestion and absorption take time, and it is not perfectly linear because the
glycemic response depends on context.

In the table, `carbs_g` is treated as an event variable rather than a
persistent state. Missing values are interpreted as no recorded event at that
timestep and are therefore filled with zero.

### Device context and bolus intent

`device_mode` records the operating mode of the insulin-delivery system,
typically values such as `regular`, `sleep`, or `exercise`. This is more than
metadata. It is a contextual regime variable that can change how the same
glucose, meal, and insulin signals should be interpreted. The repository
therefore treats it as a state variable: it is aligned to the common grid and
forward-filled until the next mode change.

`bolus_type` records the intent and mechanism of bolus delivery through
categories such as `standard`, `correction`, and `automatic`. It helps the
model distinguish why insulin was delivered, not just how much. Unlike
`device_mode`, it is event-local and remains sparse in the aligned timeline.

## Cleaning And Table-Wide Normalization

After raw canonicalization, the repository applies dataframe-level
normalization through `load_processed_frame(...)`. This is where the branch's
practical scientific assumptions become visible.

Rows missing subject identity, timestamp, or the glucose target are dropped,
because they cannot participate in legal supervised forecasting samples.
Timestamps are converted to datetime and rounded to the nearest minute so that
minor formatting inconsistencies do not leak into continuity logic. Rows are
sorted by subject and time, exact duplicates are removed, and duplicate
timestamps are collapsed so that the later indexing layer sees one coherent row
per nominal time point.

The more interesting choices involve imputation and persistence. Basal insulin
is converted to numeric, then forward-filled within each subject, then
backward-filled for leading gaps, and only then zero-filled if anything still
remains missing. That is a strong statement that basal behaves like a
background state on the unified time grid rather than like an isolated event.
It is a defensible choice, but still an assumption.

Event-style variables are handled differently. `bolus_insulin_u`,
`correction_insulin_u`, `meal_insulin_u`, and `carbs_g` are zero-filled once
they have been aligned to the common 5-minute grid. That is the right behavior
for sparse event signals. Forward-filling those values would distort the data
severely.

`device_mode` is normalized into a controlled vocabulary, with raw placeholders
such as `"0"` mapped to `"regular"`, and then forward-filled within subject.
That again reflects a state interpretation. `bolus_type`, by contrast, is not
forward-filled because it should remain tied to individual bolus events.

Finally, the transform layer adds known-ahead calendar features:

- `minute_of_day_sin`
- `minute_of_day_cos`
- `day_of_week_sin`
- `day_of_week_cos`
- `is_weekend`

and normalizes declared categorical vocabularies into stable category orders so
that later encoding and embedding cardinalities remain stable.

## Why These Preprocessing Choices Matter

Preprocessing is not just cleaning. It defines what kind of forecasting problem
the model is allowed to see.

The most important distinction is between persistent states and sparse events.
Basal insulin and device mode are treated as state-like signals. Bolus insulin,
correction insulin, meal insulin, and carbohydrates are treated as event-like
signals. That is medically sensible and algorithmically important.

The second important distinction is continuity. The repository later refuses to
build windows across gaps in the expected 5-minute cadence. So the data layer
is not pretending the timeline is uniformly sampled when it is not.

The third is imputation. Forward-filling device mode assumes persistence.
Forward- and backward-filling basal insulin assumes a stable underlying basal
state. Zero-filling event variables assumes that the absence of a record means
the absence of an event. These assumptions are coherent, but they are still
assumptions, and the docs should say so plainly.

## Semantic Feature Roles

The repository adopts one of the most useful ideas from the TFT literature:
not every variable plays the same forecasting role.

Static variables are time-invariant within a sample. In the current branch,
subject identity is the main static variable.

Known variables are time-varying but legitimately available ahead of forecast
time. The default calendar features are the clearest example.

Observed variables are available historically but not known prospectively. Meal
events, insulin events, device mode, and bolus type all belong here under the
current contract.

The target variable is the glucose series itself. Historical glucose belongs in
the encoder; future glucose is what the model must predict.

This partition is one of the main reasons the current data contract feels
disciplined. It is the line that separates future-known context from future
leakage.

## The Current Feature Contract

The default grouping is:

### Static categorical

- `subject_id`

### Static continuous

- none by default

### Known continuous

- `minute_of_day_sin`
- `minute_of_day_cos`
- `day_of_week_sin`
- `day_of_week_cos`
- `is_weekend`

### Known categorical

- none by default

### Observed continuous

- `basal_insulin_u`
- `bolus_insulin_u`
- `correction_insulin_u`
- `meal_insulin_u`
- `carbs_g`

### Observed categorical

- `device_mode`
- `bolus_type`

### Target

- `glucose_mg_dl`

This is a conservative contract, but a sensible one. It respects the causal
availability of information, even if it likely leaves some future-known
structure unused in more ambitious experimental variants.

## Sequence Construction And Split Policy

Once the cleaned dataframe exists, the repository still does not go directly to
tensors. It first constructs a list of legal windows through
`build_sequence_index(...)`, which produces `SampleIndexEntry` objects storing
the encoder and decoder boundaries over the split dataframe.

That separation is one of the cleaner design choices in the pipeline. Indexing
answers one question: which windows are legal? Dataset assembly answers another:
how should one legal window be packed into tensors? Those are not the same
responsibility, and they fail for different reasons.

For a window to be legal, it must stay entirely inside a contiguous segment
whose timestep gaps match the expected sampling interval. If the subject
timeline contains a discontinuity, the code breaks it into separate segments
and no sample may cross the gap.

A legal segment must also be long enough:

\[
L_{required} = L_e + L_d
\]

With the default settings, that is \(168 + 12 = 180\) rows. Segments shorter
than that produce no default sample.

The repository uses `window_stride = 1` by default, so consecutive windows
overlap heavily. That is a common choice in forecasting, but it does mean the
effective sample set contains many near-neighbors.

The default split policy is also worth reading carefully. The branch defaults to
within-subject chronological splitting rather than subject-held-out splitting.
That makes sense for personalized forecasting, because it asks whether a
subject's earlier history can forecast that same subject's later future. It is
not the same as testing generalization to unseen patients, and the docs should
keep making that distinction explicit.

## The Batch Contract Emitted To The Model

For each indexed window, `AZT1DSequenceDataset.__getitem__` returns a
dictionary with these keys:

- `static_categorical`
- `static_continuous`
- `encoder_continuous`
- `encoder_categorical`
- `decoder_known_continuous`
- `decoder_known_categorical`
- `target`
- `metadata`

This is the most important interface on the data side.

`static_categorical` is a one-row categorical vector anchored at the encoder
start, usually just subject identity. `static_continuous` is a one-row
continuous vector and is empty by default. `encoder_continuous` stores

\[
[\text{known continuous} \; | \; \text{observed continuous} \; | \; \text{target history}]
\]

in that order across the encoder window. `encoder_categorical` stores

\[
[\text{known categorical} \; | \; \text{observed categorical}]
\]

across the encoder. `decoder_known_continuous` and
`decoder_known_categorical` hold future-known decoder features. `target` holds
the future glucose sequence. `metadata` exists for debugging and later
reporting, not as a model input.

The dataset deliberately emits zero-width tensors instead of `None` for empty
groups. That keeps batching predictable and lets later code reason about widths
without constantly checking for missing keys.

## Default Shapes And Timeline Meaning

Let:

- \(B\) = batch size
- \(L_e\) = encoder length
- \(L_d\) = decoder length
- \(F_{sc}\) = number of static categorical features
- \(F_{s\ell}\) = number of static continuous features
- \(F_{kc}\) = number of known categorical features
- \(F_{k\ell}\) = number of known continuous features
- \(F_{oc}\) = number of observed categorical features
- \(F_{o\ell}\) = number of observed continuous features
- \(F_t\) = number of target channels

In the default branch:

- \(F_{sc} = 1\)
- \(F_{s\ell} = 0\)
- \(F_{kc} = 0\)
- \(F_{k\ell} = 5\)
- \(F_{oc} = 2\)
- \(F_{o\ell} = 5\)
- \(F_t = 1\)

So the default shapes are:

- `static_categorical`: \([B, 1]\)
- `static_continuous`: \([B, 0]\)
- `encoder_continuous`: \([B, 168, 11]\)
- `encoder_categorical`: \([B, 168, 2]\)
- `decoder_known_continuous`: \([B, 12, 5]\)
- `decoder_known_categorical`: \([B, 12, 0]\)
- `target`: \([B, 12]\)

The cleanest way to interpret these is on the original patient timeline.
Suppose rows 1000 through 1167 form the encoder and rows 1168 through 1179
form the decoder. Then the model sees 14 hours of history and is asked to
forecast the next hour. `encoder_continuous` carries the historical trajectory
of calendar features, observed insulin and carbohydrate signals, and historical
glucose. `encoder_categorical` carries device mode and bolus type history.
`decoder_known_continuous` carries the next hour's known-ahead time features.
`target` carries the true future glucose sequence.

That is the actual information boundary of the forecasting problem.

## Why The DataModule, Dataset, And Indexer Are Separate

The top-level data design is intentionally layered.

`AZT1DDataModule` owns the full data lifecycle: downloading and preparing
processed data on disk, loading the cleaned dataframe, building category maps,
splitting the data, indexing legal windows, creating dataset instances, and
constructing DataLoaders. It also caches categorical cardinalities, because the
model's embedding tables depend on them. That makes the DataModule the place
where runtime-discovered data facts are bound into model configuration.

The dataset owns only sample assembly. It does not decide which windows are
legal, and it does not own the whole data lifecycle.

The indexing layer owns legality. It answers a different question from the
dataset, and that difference is exactly why the separation is useful.

## Best Next Reads

- Model, loss, training, and runtime story: [`methodology.md`](methodology.md)
- Metric interpretation and design risks:
  [`results_and_discussion.md`](results_and_discussion.md)
- Dataset source trail and provenance notes: [`references.md`](references.md)
- Guided engineering walkthrough:
  [`../primer/data_pipeline_walkthrough.md`](../primer/data_pipeline_walkthrough.md)
