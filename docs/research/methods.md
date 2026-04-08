# Methods

Role: Canonical methods section for the research companion.
Audience: Researchers who want the model, loss, and training story without
leaving the research folder.
Owns: Method framing, architecture, probabilistic supervision, runtime
binding, and the code-to-algorithm crosswalk.
Related docs: [`dataset.md`](dataset.md), [`introduction.md`](introduction.md),
[`results_and_discussion.md`](results_and_discussion.md),
[`../current_architecture.md`](../current_architecture.md),
[`../repository_primer.md`](../repository_primer.md).

## Method Summary

The current branch implements a probabilistic glucose-forecasting pipeline
built around a late-fused TCN + TFT architecture. The method is easier to
understand if it is read as one end-to-end chain rather than as a list of
independent components:

1. a cleaned dataframe is converted into grouped tensors with explicit semantic
   roles
2. a history-focused TCN path and a feature-aware TFT path process that same
   sample in parallel
3. the resulting horizon-aligned latent representations are fused
4. the fused representation is projected into quantile forecasts
5. training uses pinball loss, while reporting still exposes familiar
   point-error metrics through the median forecast

That combination is what gives the repository its current identity. It is not
just using a Temporal Fusion Transformer with a few extra features attached. It
is making a more specific architectural claim: short-range temporal pattern
extraction and feature-role-aware multi-horizon reasoning are both useful, and
they should meet in representation space before the final prediction head.

## The Current Experimental Method

At the highest level, the method has five stable parts.

The data side converts a cleaned dataframe into semantically grouped tensors
rather than one anonymous feature matrix. The model side uses three TCN
branches, one TFT branch, one fusion GRN, and one final head that emits
quantiles. The supervision side uses pinball loss rather than point-only
regression loss. The evaluation side still reports MAE and RMSE by extracting a
representative point forecast, usually the median quantile. The runtime side
delegates the outer training loop to PyTorch Lightning while keeping
model-building, data-binding, and observability policy inside the repository's
own orchestration layer.

That is an important design choice. It means the repository is treating method,
runtime, and evaluation as related but separable concerns instead of blending
them into one script.

## Why The Method Is Probabilistic

The repository does not produce a full parametric density such as a Gaussian
with learned mean and variance. Instead, it predicts a small set of quantiles.
In practical terms, that means the model is trying to estimate selected cut
points of the conditional distribution of future glucose.

If the future glucose value is written as \(Y\) and the available model context
as \(X\), then the conditional cumulative distribution function is

\[
F_{Y|X}(y) = P(Y \le y \mid X)
\]

and the conditional quantile at level \(q\) is

\[
Q_q(X) = F^{-1}_{Y|X}(q)
\]

The current model predicts the tuple \((0.1, 0.5, 0.9)\). So for each future
timestep it emits:

- a 10th-percentile cutoff
- a median
- a 90th-percentile cutoff

That matters because glucose uncertainty is not constant across contexts. Some
forecast windows are comparatively easy. Others are harder because meals are
poorly logged, control actions interact, or the recent trajectory is unstable.
A point forecast hides that variation. A quantile forecast exposes it.

## Three Ways To Read The Outputs

The probabilistic outputs can be understood through three complementary lenses.

### Classical regression lens

The model still has one target: future glucose. What changes is that it no
longer emits only one central answer. It emits several distributional cutoffs
for that same target.

### Conditional-distribution lens

The output channels are inverse-CDF cut points of the conditional future
glucose distribution.

### Forecast-interval lens

The lower and upper quantiles define an uncertainty interval at each future
timestep, while the middle quantile acts as the median forecast.

These lenses are useful precisely because none of them is complete on its own.
The first is intuitive, the second is mathematically clean, and the third is
the most operational when reading forecasts.

## The Model In One Pass

The repository's core model is `FusedModel`, a LightningModule that wraps the
entire hybrid architecture.

Conceptually, the forward pass does the following:

1. unpack the grouped encoder tensors into semantically meaningful pieces
2. build a TCN input from observed continuous history plus target history
3. build grouped TFT inputs from static, known, observed, and target-history
   families
4. run three TCN branches with kernel sizes 3, 5, and 7
5. run the TFT branch
6. concatenate the resulting horizon-aligned latent tensors
7. pass that concatenated representation through a GRN fusion layer
8. project the fused representation into one output channel per quantile

The important architectural point is where the fusion happens. The current
branch does not fuse already-decoded branch forecasts. It fuses latent
representations and leaves the final quantile mapping to the shared head.

## Why Fuse TCN And TFT

The TCN and TFT branches are doing different jobs.

The TCN side is the history specialist. It is good at causal local-to-mid-range
pattern extraction and is well suited to motifs such as post-meal rises,
correction-driven declines, short-term oscillations, and local rate-of-change
structure. The repository reinforces that role by giving the TCN only observed
continuous history plus target history. It does not ask the TCN to reason about
future-known decoder covariates directly.

The TFT side is the feature-role-aware branch. It is designed for settings with
static covariates, known future inputs, observed historical variables, and
target history. That matches this repository closely. It is the branch best
equipped to use static subject context, future-known time features, and the
semantic partition of variables introduced on the data side.

The fusion layer exists because neither branch captures the whole problem by
itself. The TCN path contributes efficient multi-scale history encoding. The
TFT path contributes richer structured reasoning over mixed feature families and
the forecast horizon. Late fusion lets the repository combine those inductive
biases without pretending that one branch is only a helper for the other.

## The TCN Path

The TCN implementation follows standard causal temporal-convolution ideas, but
it is adapted to the repository's forecasting contract.

`CausalConv1d` pads only on the left, which ensures that the representation at
time \(t\) depends only on times \(\le t\). For forecasting, that is
non-negotiable. If the convolution could peek into the future, evaluation would
be invalid.

Each `TemporalBlock` contains two causal convolutions separated by normalization,
activation, and dropout, together with a residual connection. The residual path
matters because it lets each block refine the incoming representation instead of
rebuilding it from scratch. The branch also standardizes on layer
normalization rather than batch normalization, which is sensible for
heterogeneous medical time series where batch composition can vary sharply by
subject and condition.

The fused model instantiates three TCN branches with kernel sizes 3, 5, and 7.
All three consume the same history tensor, but each brings a different local
receptive-field bias. That is a reasonable design for glucose forecasting,
where short-lag sensor behavior, meal response curves, insulin action, and
slower drift all operate on different scales.

## The TFT Path

The TFT branch is based on the architecture introduced by Lim et al., with
project-specific adaptations and an implementation lineage closer to NVIDIA's
DeepLearningExamples code.

The core TFT idea preserved here is that variables are embedded separately
rather than flattened into one generic dense vector immediately. Categorical
variables get embedding tables. Continuous variables get learned linear
embeddings. That preserves variable identity, which in turn makes context-aware
variable selection meaningful.

The implementation assumes seven input families:

1. static categorical
2. static continuous
3. temporal known categorical
4. temporal known continuous
5. temporal observed categorical
6. temporal observed continuous
7. temporal target history

That list matters because the grouped tensor contract on the data side exists
largely to feed this worldview. The TFT branch is where the repository most
explicitly reasons about what is known ahead of time, what is only observed
historically, and what must still be forecast.

If a continuous group tensor has shape

\[
[B, T, F]
\]

then the learned embedding matrix has shape

\[
[F, H]
\]

so the embedded result becomes

\[
[B, T, F, H].
\]

In words: each scalar variable at each timestep gets its own hidden vector. The
branch can then apply variable selection and temporal modeling without erasing
which variable is which too early.

## Fusion, Head, And Tensor Shapes

The GRN appears both inside the TFT stack and again at the final fusion stage.
Conceptually, a Gated Residual Network projects an input into a hidden space,
applies nonlinear transformation, uses a gate to decide how much transformed
content should pass, and then adds a residual shortcut. Reusing the same style
of block at fusion keeps the architecture numerically and stylistically aligned
instead of bolting on a completely different fusion mechanism.

With the current defaults:

- encoder length = 168
- prediction length = 12
- TFT hidden size = 128
- each TCN branch hidden size = 128

the main tensor flow looks like this:

- TCN input: \([B, 168, \text{observed\_cont} + \text{target}]\)
- each TCN branch output: \([B, 12, 128]\)
- TFT latent decoder features: \([B, 12, 128]\)
- concatenated fusion tensor: \([B, 12, 512]\)
- post-fusion GRN output: \([B, 12, 128]\)
- final quantile output: \([B, 12, 3]\)

The head itself does not encode quantile theory. It simply produces one output
channel per configured quantile. The loss function gives those channels their
meaning.

## Training Objective

The central training objective is pinball loss, also called quantile loss. For
one quantile \(q \in (0, 1)\), target \(y\), and prediction \(\hat{y}_q\), the
loss is

\[
\mathcal{L}_q(y, \hat{y}_q) =
\max((q - 1)(y - \hat{y}_q), q(y - \hat{y}_q)).
\]

Equivalent piecewise form:

\[
\mathcal{L}_q(y, \hat{y}_q) =
\begin{cases}
q(y - \hat{y}_q), & y \ge \hat{y}_q \\
(1 - q)(\hat{y}_q - y), & y < \hat{y}_q
\end{cases}
\]

This asymmetry is what makes each output channel settle into its role. When
\(q = 0.1\), overprediction is penalized much more heavily than underprediction,
so the model is pushed toward a lower conditional cutoff. When \(q = 0.9\), the
asymmetry reverses and the channel is pushed toward an upper cutoff. When
\(q = 0.5\), the asymmetry disappears and the channel behaves like a median
estimator.

That is the key intuition to preserve: the model does not explicitly count how
many values fall below each forecast during inference. The quantile semantics
emerge from the loss geometry over many optimization steps.

## Point Metrics Still Matter

Although the model is trained probabilistically, the repository still reports
MAE and RMSE. That does not contradict the method. It reflects a practical
choice about how forecasts are read.

The evaluation code extracts a representative point forecast by selecting the
configured quantile closest to 0.5. Because the current quantile tuple
contains 0.5 exactly, the median forecast becomes the point forecast used for
familiar scalar error metrics. So the loss uses the full quantile tensor, while
the reporting layer still exposes a point summary for readers who want one.

## Runtime Binding And Trainer Role

The branch uses dataclass-based configuration objects to keep the method
coherent across data preparation, model construction, and runtime policy.

At the top level:

- `DataConfig` owns data paths, split policy, sequence lengths, loader
  behavior, and dataset facts
- `TFTConfig` owns the TFT branch
- `TCNConfig` owns the TCN branch
- `Config` groups the data and model configuration
- `TrainConfig`, `SnapshotConfig`, and `ObservabilityConfig` govern runtime
  policy

That separation matters because some model facts are not fully known until the
data has been prepared. Categorical cardinalities are the clearest example. The
pipeline therefore starts from a declarative config, prepares the data, builds
category maps, binds those discovered facts back into the model config, and
only then instantiates the model.

`FusedModelTrainer` is the orchestration layer that holds those pieces
together. It prepares the DataModule, binds runtime metadata into the model,
optionally compiles the model, builds callbacks and the Lightning trainer, and
launches `fit`, `test`, or prediction flows. PyTorch Lightning still owns the
outer epoch loop, checkpointing, early stopping, and device mechanics, but the
repository owns the higher-level method assembly.

## Shape Walkthrough With Default Settings

Assume:

- batch size \(B = 64\)
- encoder length \(L_e = 168\)
- decoder length \(L_d = 12\)

Then the main grouped tensors and outputs are:

- `encoder_continuous`: \([64, 168, 11]\)
- `encoder_categorical`: \([64, 168, 2]\)
- `decoder_known_continuous`: \([64, 12, 5]\)
- `target`: \([64, 12]\)
- model output: \([64, 12, 3]\)

That means each sample carries 14 hours of encoder history and 1 hour of future
forecast, with three quantile channels for every 5-minute step in the decoder
horizon.

## Code-To-Algorithm Crosswalk

This section maps major files to algorithmic roles.

### `defaults.py`

Defines baseline research defaults for data lengths, split ratios, model sizes,
quantiles, and runtime policy.

### `src/data/preprocessor.py`

Normalizes raw vendor exports into the canonical CSV.

### `src/data/transforms.py`

Performs dataframe-wide cleanup, imputation, normalization, time-feature
generation, categorical normalization, and vocabulary fitting.

### `src/data/schema.py`

Declares canonical columns, feature groups, categorical vocabularies, and
semantic feature meaning.

### `src/data/indexing.py`

Determines which windows are valid and how data are split.

### `src/data/dataset.py`

Converts one valid window into one grouped tensor sample.

### `src/data/datamodule.py`

Coordinates the whole data lifecycle and binds runtime-discovered metadata into
model configuration.

### `src/config/data.py`

Owns the data contract.

### `src/config/model.py`

Owns the TCN, TFT, and top-level model config contracts.

### `src/models/tcn.py`

Owns the project-specific temporal convolution branch.

### `src/models/tft.py`

Owns the TFT implementation and embedding front-end.

### `src/models/grn.py`

Owns the reusable gated residual network block.

### `src/models/fused_model.py`

Owns the main LightningModule, including forward logic, loss logic, and
optimizer behavior.

### `src/evaluation/metrics.py`

Owns primitive error metrics and probabilistic interval summaries.

### `src/train.py`

Owns the reusable Lightning orchestration wrapper around fit, test, and predict
flow.

### `src/workflows/training.py`

Owns the higher-level workflow that packages fit, evaluation, reports, and run
summary generation.

## Important Equations

### Required window length

\[
L_{required} = L_e + L_d
\]

### Point metrics

\[
MAE = \frac{1}{N} \sum_i |\hat{y}_i - y_i|
\]

\[
RMSE = \sqrt{\frac{1}{N} \sum_i (\hat{y}_i - y_i)^2}
\]

### Mean bias

\[
Bias = \frac{1}{N} \sum_i (\hat{y}_i - y_i)
\]

### Quantile pinball loss

\[
\mathcal{L}_q(y, \hat{y}_q) =
\max((q - 1)(y - \hat{y}_q), q(y - \hat{y}_q))
\]

### Mean interval width

For the outermost quantiles:

\[
MPIW = \frac{1}{N} \sum_i (\hat{y}_{i, q_{high}} - \hat{y}_{i, q_{low}})
\]

### Empirical coverage

\[
Coverage =
\frac{1}{N} \sum_i \mathbf{1}\left[\hat{y}_{i, q_{low}} \le y_i \le \hat{y}_{i, q_{high}}\right]
\]

## Best Next Reads

- Data provenance and the input contract: [`dataset.md`](dataset.md)
- Metric interpretation and model limitations:
  [`results_and_discussion.md`](results_and_discussion.md)
- Closing synthesis and future work: [`conclusions.md`](conclusions.md)
- Provenance and external sources: [`references.md`](references.md)
