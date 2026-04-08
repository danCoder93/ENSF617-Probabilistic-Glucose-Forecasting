# Understanding the Repository's Probabilistic Glucose Forecasting Model

Role: Conceptual lens focused on quantile forecasting and uncertainty
interpretation.
Audience: Readers who want probabilistic intuition without traversing the full
codebase architecture first.
Owns: Quantile semantics, pinball-loss intuition, and interval interpretation.
Related docs: [`methods_exposition.md`](methods_exposition.md),
[`../repository_primer.md`](../repository_primer.md),
[`../current_architecture.md`](../current_architecture.md).

## Theoretical Foundations and Architectural Interpretation of Quantile-Based Probabilistic Forecasting

**Purpose.** This document turns the repository's current model-facing design and the conceptual discussion around it into one coherent learning note. Its goal is not merely to list facts about the code, but to help a reader move smoothly from familiar regression thinking to the probabilistic forecasting view used by the repository's hybrid **TCN + TFT** model.

**Scope.** The note is intentionally limited to the parts of the repository that matter for understanding:

- how data enters the model
- how the model organizes and processes that data
- what the model is trained to predict
- how quantile outputs should be interpreted
- how training and evaluation metrics relate to those outputs

It does **not** discuss runtime environment logic, observability internals, workflow orchestration, or deployment concerns.

---

## 1. Classical Regression Foundations

A useful way to begin is with the most classical machine learning picture:

- a set of inputs \(X\)
- one target \(Y\)
- a model that learns a function \(f(X)\) to predict \(Y\)

In standard tabular regression, this often means:

- many input columns
- one scalar output
- one central prediction, often interpreted as a mean-like prediction

This mental model is important because the present repository does **not** abandon it. Instead, it extends it.

At a high level, the repository is still solving a supervised learning problem in which inputs are used to predict a target. The main difference is that the target is not a one-shot scalar attached to a single row. Rather, the repository solves a **time-series forecasting** problem, where a model uses a **history window** of patient data to predict a **future sequence** of glucose values.

So the classical supervised view remains valid, but it must be updated in two ways:

1. the input is now a structured sequence rather than a single flat row
2. the output is no longer only one central forecast, but a probabilistic summary of possible future outcomes

That shift motivates everything that follows.

---

## 2. Transition from Point Estimation to Probabilistic Forecasting

If one begins from ordinary regression, the natural first question is:

> what exactly changes when the model becomes probabilistic?

The easiest answer is that ordinary regression usually aims to produce **one best central answer**, whereas probabilistic forecasting aims to describe **both the central forecast and the uncertainty around it**.

For example, a standard regression model might say:

- future glucose = 120

A quantile-based probabilistic model instead might say:

- \(q_{0.1} = 100\)
- \(q_{0.5} = 120\)
- \(q_{0.9} = 145\)

This is still a forecast of **future glucose**. The target has not changed. What has changed is the **kind of answer**:

- one lower conditional cutoff
- one middle conditional cutoff
- one upper conditional cutoff

This is the first key conceptual transition:

> quantile forecasting does not introduce multiple new targets; it gives multiple distributional views of the same target.

This makes the probabilistic model richer than a single point forecaster while still remaining close to the classical regression framework.

---

## 3. Quantile Regression as a Distributional Learning Framework

Once the need for probabilistic forecasting is established, the next academic step is to clarify what kind of probabilistic object the model is producing.

The repository does not produce a full parametric density such as a Gaussian with mean and variance. Instead, it predicts a **small set of quantiles**. This means the model learns selected cutoff points of the conditional distribution of future glucose.

### 3.1 Mean regression versus quantile regression

In ordinary regression, the model often estimates the **conditional mean** of the target given the inputs.

In quantile regression, the model estimates a chosen **conditional quantile** of the target given the inputs.

If the future glucose value is written as \(Y\) and all available model information as \(X\), then the conditional cumulative distribution function is

\[
F_{Y|X}(y) = P(Y \le y \mid X)
\]

A conditional quantile at level \(q\) is the value \(Q_q(X)\) such that

\[
Q_q(X) = F^{-1}_{Y|X}(q)
\]

So:

- \(q_{0.1}\) is the estimated value where the conditional CDF reaches 0.1
- \(q_{0.5}\) is the estimated median
- \(q_{0.9}\) is the estimated value where the conditional CDF reaches 0.9

This mathematical view provides a more precise version of the earlier intuition. The model is not merely giving a "low guess" and a "high guess." It is estimating **specific percentile cutoffs** of the future glucose distribution, conditioned on the current input context.

### 3.2 The repository's configured quantiles

The current model configuration uses the quantile tuple:

- \( (0.1, 0.5, 0.9) \)

This means that at each future timestep, the model predicts:

- a 10th-percentile cutoff
- a 50th-percentile cutoff (the median)
- a 90th-percentile cutoff

Suppose the model outputs:

- \(q_{0.1} = 100\)
- \(q_{0.5} = 120\)
- \(q_{0.9} = 145\)

Then the interpretation is:

- about 10% of future outcomes are expected to be at or below 100
- about 50% of future outcomes are expected to be at or below 120
- about 90% of future outcomes are expected to be at or below 145

Equivalently, the interval \([100, 145]\) is the model's estimated **central 80% prediction interval** for that timestep.

### 3.3 Why this matters in forecasting

This move from mean regression to quantile regression is especially important when uncertainty is **not constant**.

In real forecasting problems, some contexts are much more predictable than others. A single mean forecast can hide that fact. Quantile outputs instead reveal:

- the center of the forecast
- the width of uncertainty
- possible asymmetry in uncertainty
- how uncertainty changes with context

This makes quantile regression particularly suitable for medical time-series forecasting, where the conditional spread of future values can vary substantially across physiological states.

---

## 4. Sequence Forecasting and the Concept of Prediction Horizon

Once the reader understands that the repository predicts conditional quantiles rather than only a conditional mean, the next important transition is from **one future value** to a **future sequence**.

This is where the concept of the **forecast horizon** becomes central.

The horizon means:

> how many future timesteps the model predicts

In the repository defaults:

- `encoder_length = 168`
- `prediction_length = 12`

The data is organized on a 5-minute grid. Therefore:

- 168 historical timesteps correspond to 14 hours of history
- 12 future timesteps correspond to 60 minutes of future forecast

So the model does not predict one isolated future glucose scalar. It predicts a **12-step future glucose trajectory**.

This matters because every major architectural choice in the repository is built around that sequence-to-sequence forecasting structure. The model is trying to answer:

> given the recent patient history and the available covariates, what are the likely glucose values over the next 12 timesteps?

This also explains why the output must be interpreted along two dimensions at once:

- **time over the forecast horizon**
- **quantile channels at each timestep**

---

## 5. Semantic Structuring of Model Inputs

At this point, a natural next question is:

> what exactly counts as "the available covariates"?

The repository answers this by using a **semantic data contract** rather than one anonymous flat input matrix. This is one of the most important design choices for understanding the model.

### 5.1 Feature roles

The code distinguishes features by forecasting role:

- **STATIC**: variables that do not change across the sequence for a sample
- **KNOWN**: variables that vary with time but are known ahead of forecast time
- **OBSERVED**: variables observed historically but not known ahead
- **TARGET**: the variable to be forecasted

It also distinguishes features by representation type:

- **CONTINUOUS**
- **CATEGORICAL**

These distinctions create grouped model-facing inputs such as:

- static categorical
- static continuous
- known categorical
- known continuous
- observed categorical
- observed continuous
- target history

### 5.2 Why the grouping is meaningful

This grouping is not cosmetic. It reflects a forecasting reality:

- static subject-level information plays a different role from time-varying signals
- future-known covariates should not be treated the same way as unknown future outcomes
- observed historical covariates are different from the target being predicted

In other words, the repository does not merely ask what the values are; it also asks **what forecasting role each value plays**.

This is especially important for the TFT branch, which is built to reason explicitly about these role distinctions.

---

## 6. Architectural Design: Complementary Modeling Branches

Once the data is understood as semantically structured and the output is understood as a horizon-wise probabilistic forecast, the model architecture becomes easier to motivate.

The core model is a **late-fusion hybrid** with:

- three **TCN** branches, using kernel sizes 3, 5, and 7
- one **TFT** branch
- one post-branch **GRN** fusion layer
- one final **NN head** that outputs quantiles

This architecture reflects a division of labor.

### 6.1 What the TCN branches are good at

TCNs are strong at:

- extracting local and multiscale temporal patterns
- modeling short- to medium-range temporal structure
- using causal convolutions to process history efficiently

### 6.2 What the TFT branch is good at

TFT is strong at:

- respecting semantic feature roles
- incorporating static context
- handling known future covariates
- combining historical and future-aware reasoning over the forecast horizon

This means the hybrid architecture is not arbitrary. It is designed so that:

- the TCN side contributes strong history-pattern extraction
- the TFT side contributes feature-role-aware temporal reasoning

The fusion layer then combines these complementary representations.

---

## 7. Temporal Convolutional Network (TCN) Branch: Historical Signal Modeling

The easiest branch to misunderstand is often the TCN branch, because it does **not** receive the full structured TFT-style input contract.

Instead, the TCN path is deliberately narrower and more history-focused.

### 7.1 What the TCN receives

The repository constructs the TCN input from:

- observed continuous history
- target history

These are concatenated along the feature axis.

So the TCN branch is not given all grouped semantic inputs. It is given a **history-only view** built around:

- what has been observed historically
- what the recent glucose trajectory has been

### 7.2 Why target history is essential

A common question is why the TCN should receive the target history at all.

The answer is that the TCN branch is a history-only forecaster, and one of the strongest clues about future glucose is the recent glucose trajectory itself.

This is natural in time-series forecasting because:

- glucose has temporal continuity
- rises, falls, and short-range trends matter
- the target series itself contains strong local structure
- short-horizon forecasting is often strongly autocorrelated

So target history is included not as a shortcut, but as an essential forecasting signal. Without it, the TCN would be forced to forecast future glucose while being denied the recent glucose path.

### 7.3 What each TCN branch computes

Each TCN branch:

1. encodes the full encoder history using causal residual temporal convolutions
2. summarizes the final encoded timestep
3. expands that summary into one hidden vector per future timestep in the forecast horizon

This gives a **horizon-aligned latent representation** for that branch.

Because the repository uses three kernel sizes:

- 3
- 5
- 7

the model obtains three related but differently biased temporal views of the same history window.

---

## 8. Temporal Fusion Transformer (TFT) Branch: Feature-Aware Forecasting

If the TCN path is the history-specialist branch, the TFT path is the feature-role-aware branch.

The TFT branch receives grouped semantic inputs that conceptually include:

- static variables
- known historical variables
- known future variables
- observed historical variables
- target history

The crucial difference is that TFT explicitly uses **future-known inputs**. That makes it the branch that most directly captures the forecasting distinction between:

- what is historically observed
- what is known ahead of time
- what remains uncertain and must be predicted

In the current repository implementation, the TFT branch can produce both:

1. latent decoder features over the future horizon
2. standalone TFT quantile outputs

However, in the fused architecture, the model uses the **latent decoder features** rather than fusing the final TFT quantile outputs directly.

This is an important conceptual transition because it leads directly to the question of what exactly is fused.

---

## 9. Latent Representation Fusion Mechanism

It is very easy to imagine the fusion step incorrectly as something like:

- each TCN branch produces one future forecast
- TFT produces its quantile forecast
- those output forecasts are then fused

That is **not** what the current code does.

Instead, the model computes the following horizon-aligned latent tensors:

- `tcn3_features`
- `tcn5_features`
- `tcn7_features`
- `tft_features`

These are all **representation tensors**, not final forecast tensors.

The fusion step concatenates them along the feature axis.

So what is fused is:

- one TFT latent feature tensor
- three TCN latent feature tensors

This means fusion happens in **representation space**, before the model has collapsed those representations into final quantile outputs.

This is a crucial architectural idea:

> the model first lets different branches form their own horizon-wise internal views, then learns how to combine those views, and only afterward turns the fused representation into final probabilistic outputs.

---

## 10. Tensor-Level Representation of Model Flow

Once the functional story is clear, tensor shapes help stabilize understanding.

Using the repository defaults and current implementation:

- batch size = \(B\)
- encoder length = 168
- prediction length = 12
- TFT hidden size = 128
- each TCN branch hidden size = 128

### 10.1 TCN input

The TCN input shape is:

\[
[B, 168, \text{num\_observed\_cont} + \text{num\_target}]
\]

### 10.2 TCN latent features

Each TCN branch returns:

\[
[B, 12, 128]
\]

So:

- `tcn3_features`: \([B, 12, 128]\)
- `tcn5_features`: \([B, 12, 128]\)
- `tcn7_features`: \([B, 12, 128]\)

### 10.3 TFT latent decoder features

The TFT latent decoder representation is also:

\[
[B, 12, 128]
\]

So:

- `tft_features`: \([B, 12, 128]\)

### 10.4 Fusion input

Concatenating all four feature tensors along the last dimension yields:

\[
[B, 12, 128 + 128 + 128 + 128] = [B, 12, 512]
\]

Thus:

- `pre_fusion_features`: \([B, 12, 512]\)

### 10.5 Fusion output

The GRN fusion layer maps this back to the TFT hidden width:

\[
[B, 12, 512] \rightarrow [B, 12, 128]
\]

Thus:

- `post_fusion_features`: \([B, 12, 128]\)

### 10.6 Final model output

The final NN head maps the fused hidden representation to:

\[
\text{len(quantiles)} = 3
\]

So the final prediction tensor is:

\[
[B, 12, 3]
\]

That means the output carries two simultaneous structures:

- 12 future timesteps
- 3 quantile channels per timestep

This tensor-shape view is the most concrete way to see how the earlier conceptual discussion maps onto the implementation.

---

## 11. Output Projection Layer and Quantile Mapping

Once the fused hidden representation has been formed, the model passes it through the final `NNHead`.

The head's job is simple but important:

> transform one fused hidden vector per horizon step into the desired forecast channels

The fused model constructs the head with:

- input size = hidden feature width
- output size = `len(quantiles)`

Since the configured quantiles are `(0.1, 0.5, 0.9)`, the output width is 3.

So, for each future timestep, the final head emits three raw values.

This leads to another important conceptual clarification:

- the final head does **not** itself encode quantile theory
- it simply produces the required number of output channels
- the **loss function** gives those channels their probabilistic meaning

So if one asks how the model "knows" its output is a set of quantiles, the most precise answer is:

1. the architecture makes room for one channel per quantile
2. the training loss interprets those channels in the configured quantile order
3. optimization pushes each channel toward its intended quantile behavior

---

## 12. Learning Quantile Functions through Optimization

At this stage, the reader can naturally ask:

> how does one of those output channels become \(q_{0.1}\) rather than just an arbitrary number?

The answer comes from training.

### 12.1 Initial state

At the beginning of training:

- model parameters are initialized
- the final output channels produce untrained numerical guesses

So for one future timestep, the model might initially emit arbitrary values.

### 12.2 Forward pass

For one training example:

1. grouped and historical inputs enter the model
2. TCN and TFT branches compute latent features
3. those features are fused
4. the final head emits three numbers at each future timestep

So the model may output something like:

- \(q_{0.1} = 104\)
- \(q_{0.5} = 120\)
- \(q_{0.9} = 138\)

for one horizon position

### 12.3 Loss and optimization

These outputs are then compared against the true future glucose values using **pinball loss**.

The loss produces gradients, and backpropagation sends those gradients through:

- the final NN head
- the fusion GRN
- the TFT branch
- the TCN branches

The optimizer updates the model parameters.

This process repeats across many batches and epochs.

So the most direct answer to the practical training question is:

> yes, the outputs begin as initialized values and become meaningful through repeated forward passes, backpropagation, and optimizer updates

The next question, then, is why the loss pushes different channels toward different quantiles.

---

## 13. Pinball Loss and Its Role in Quantile Estimation

Pinball loss is the standard loss for quantile regression, and the repository uses it as the core probabilistic training objective.

For one quantile \(q\), one true value \(y\), and one predicted value \(\hat y_q\), the loss is **asymmetric**:

- if the prediction is too low, one slope applies
- if the prediction is too high, another slope applies

This asymmetry is what gives each output channel its identity.

### 13.1 Why \(q_{0.1}\) becomes a lower quantile

When \(q = 0.1\):

- predicting too **high** is penalized heavily
- predicting too **low** is penalized lightly

So the model is strongly discouraged from placing the 10th-percentile estimate too high.

Over many training examples, the best balance point becomes a value such that only about 10% of outcomes fall below it.

### 13.2 Why \(q_{0.9}\) becomes an upper quantile

When \(q = 0.9\):

- predicting too **low** is penalized heavily
- predicting too **high** is penalized lightly

So the model is strongly discouraged from placing the 90th-percentile estimate too low.

Over many training examples, the best balance point becomes a value such that about 90% of outcomes fall below it.

### 13.3 Why \(q_{0.5}\) becomes the median

When \(q = 0.5\), the asymmetry disappears.
The loss treats over- and under-estimation equally.

That makes this channel behave like a median estimator.

### 13.4 The most important intuition

The model does **not** explicitly count:

- "10% below this in this batch"
- "90% below this in this batch"

Instead, it is the **asymmetric penalty structure** that makes the optimizer settle each output channel near the corresponding conditional quantile over many examples.

So the percentages emerge from the optimization dynamics of the loss, not from explicit percentile-counting logic during inference.

---

## 14. Conditional Distribution Interpretation via the CDF

At this point, the narrative can return to the conditional CDF view introduced earlier and connect it directly to the training story.

The predicted quantiles are best understood as **selected cut points of the conditional CDF** of future glucose.

For the conditional distribution \(Y \mid X\):

- \(q_{0.1}\) is the 10th-percentile cutoff
- \(q_{0.5}\) is the median cutoff
- \(q_{0.9}\) is the 90th-percentile cutoff

The model does not necessarily learn the entire CDF explicitly as one smooth object.
Instead, it learns a small number of chosen points on that distribution.

This lens is useful because it unifies:

- the regression intuition
- the quantile-loss intuition
- the interval interpretation

In other words, the final output channels can be understood simultaneously as:

- regression targets for asymmetric losses
- percentile cutoffs of the conditional distribution
- bounds and center of a predictive interval

---

## 15. Horizon Alignment Across Model Components

A further conceptual gap often appears around the word "decoder" and whether the TCN branch also has a horizon.

The answer is yes: in the current fused model, both branches are aligned to the same forecast horizon.

- the **TCN** branches produce horizon-aligned future features
- the **TFT** branch produces decoder-horizon future features

So although the TCN does not have a transformer-style decoder, it still produces:

- one hidden feature vector per future timestep

This is necessary because fusion happens timestep-by-timestep over the same 12-step future window.

Thus:

- TCN forecast length = 12
- TFT decoder horizon = 12

This shared alignment is what makes late fusion over future positions possible.

---

## 16. Bridging Probabilistic Outputs with Point-Based Evaluation

Once the reader accepts that the model's true native output is a full quantile tensor, the next academic question is:

> how should the model be evaluated?

The repository answers this by separating **point-forecast summaries** from **probabilistic summaries**.

### 16.1 Full probabilistic output

The raw model output is:

\[
[B, 12, 3]
\]

This is the actual probabilistic forecast tensor.

### 16.2 Point forecast extraction

For familiar metrics such as MAE and RMSE, the code extracts one representative point forecast by selecting the configured quantile closest to 0.5.

Because the current quantile tuple includes 0.5 exactly, this means:

- point forecast = \(q_{0.5}\)

So the model remains probabilistic, but point metrics are reported using the median forecast.

### 16.3 Why this split is sensible

This preserves both kinds of interpretability:

- the full quantile tensor retains uncertainty information
- MAE and RMSE provide familiar scalar summaries for comparison across runs

Thus:

- the **loss** uses the full quantile output
- the **point metrics** use the median-like forecast
- the **probabilistic metrics** use the quantile structure itself

This is a clean and academically coherent division of responsibilities.

---

## 17. Evaluation Metrics and Their Interpretation

The repository's metric surface can now be understood naturally.

### 17.1 MAE

Mean Absolute Error over the representative point forecast.

Interpretation:

- average absolute glucose forecasting error
- based on the median-like point prediction

### 17.2 RMSE

Root Mean Squared Error over the representative point forecast.

Interpretation:

- emphasizes larger misses more strongly than MAE
- still based on the point forecast

### 17.3 Mean bias

Signed mean residual over the point forecast.

Interpretation:

- positive bias means systematic overprediction
- negative bias means systematic underprediction

### 17.4 Pinball loss

Probabilistic loss over the full quantile tensor.

Interpretation:

- lower is better
- measures how well the predicted quantile channels align with the target distribution

### 17.5 Prediction interval width

The repository uses the outermost quantile pair as the interval bounds:

- lower = \(q_{0.1}\)
- upper = \(q_{0.9}\)

Thus the mean interval width is:

\[
q_{0.9} - q_{0.1}
\]

Interpretation:

- narrower intervals mean sharper forecasts
- but width alone is not enough; the interval must still cover the truth appropriately

### 17.6 Empirical interval coverage

Coverage asks:

> how often does the true value fall inside the predicted interval?

With the current quantiles, the nominal interval is a central 80% interval, so a well-calibrated model should produce empirical coverage near 0.8.

Interpretation:

- too low coverage suggests intervals are too narrow or misplaced
- too high coverage suggests intervals may be overly wide

Together, interval width and empirical coverage provide a first-pass view of probabilistic forecast quality.

---

## 18. Quantiles as Distributional Summaries of a Single Target

At this stage, it is worth stating the central conceptual lesson one final time.

The model does **not** predict multiple different clinical targets.

It predicts **future glucose**.

What changes is that, for each future timestep, the model predicts several quantile cutoffs of the conditional future glucose distribution.

So the quantile channels are best understood as:

- multiple distributional summaries
- of the same target
- over each step of the forecast horizon

This is why the probabilistic model can still be understood from the classical supervised-learning perspective, even though its outputs are richer.

---

## 19. Complementary Conceptual Frameworks

By now, three different explanatory lenses have appeared. Rather than choosing between them, it is best to treat them as complementary.

### 19.1 Lens 1: Classical regression

This is the best entry point.

- many inputs
- one target
- move from one central answer to several conditional cutoffs

### 19.2 Lens 2: Conditional CDF

This is the mathematically cleanest lens.

- quantiles are inverse-CDF cut points of the conditional future glucose distribution

### 19.3 Lens 3: Forecast interval interpretation

This is the most operational forecasting lens.

- lower and upper quantiles define uncertainty intervals over the future horizon
- the middle quantile provides the median forecast

Taken together, these lenses explain why the model can be simultaneously:

- a regression model
- a probabilistic forecaster
- an interval-producing uncertainty model

---

## 20. Architectural and Theoretical Coherence

With all the pieces now in place, the broader academic coherence of the current design becomes clearer.

The repository is coherent because:

- it treats forecasting as a sequence problem rather than a static tabular problem
- it uses a semantic data contract that respects feature roles
- it gives TCN and TFT branches different but complementary information budgets
- it aligns both branches to the same forecast horizon
- it fuses latent representations before final output generation
- it uses a final head that cleanly maps fused hidden states to forecast channels
- it gives those channels probabilistic meaning through pinball loss
- it separates point-forecast evaluation from probabilistic evaluation in a transparent way

So the probabilistic behavior of the model is not an add-on. It is part of the architecture, the loss, and the output interpretation from the start.

---

## 21. Illustrative Example of Quantile Forecast Interpretation

Suppose that for one future timestep the model outputs:

- \(q_{0.1} = 104\)
- \(q_{0.5} = 120\)
- \(q_{0.9} = 142\)

This can now be interpreted in three coherent ways.

### 21.1 Regression-style interpretation

The model has produced:

- a lower conditional estimate
- a median conditional estimate
- an upper conditional estimate

for the same future glucose target.

### 21.2 CDF interpretation

The model estimates that:

- the conditional CDF reaches 0.1 at 104
- the conditional CDF reaches 0.5 at 120
- the conditional CDF reaches 0.9 at 142

### 21.3 Forecast-interval interpretation

The model's central 80% interval is:

\[
[104, 142]
\]

and the median forecast is 120.

If the true future glucose later turns out to be 118:

- it lies inside the interval
- the median forecast error is small

If the true value turns out to be 155:

- it lies above \(q_{0.9}\)
- the interval missed high
- repeated misses of that kind would lower empirical interval coverage

This example shows how the different explanatory lenses all describe the same output in compatible ways.

---

## 22. Concluding Synthesis

The cleanest final summary is:

- the repository solves a **sequence forecasting** problem rather than a static tabular regression problem
- the target is future glucose over a **12-step forecast horizon**
- the input data is grouped semantically into **static, known, observed, and target** roles
- the **TCN** branches use observed continuous inputs plus target history to model history-only temporal structure
- the **TFT** branch uses semantic feature grouping, static context, historical context, and future-known inputs to model feature-aware temporal structure
- the model fuses **latent horizon-aligned feature representations**, not already-decoded branch forecasts
- the final head emits **three output channels per future timestep**, one per configured quantile
- pinball loss trains those channels to behave as \(q_{0.1}\), \(q_{0.5}\), and \(q_{0.9}\)
- MAE and RMSE are computed on the **median-like point forecast**
- interval width and coverage summarize the **probabilistic quality** of the forecast
- the quantile outputs are best understood as **selected percentile cutoffs of the conditional distribution of future glucose**

That is the central conceptual and academic story of the current model.

---

## Appendix A. Compact tensor summary

With default widths and batch size \(B\):

- TCN input: \([B, 168, \text{observed\_cont} + \text{target}]\)
- TCN branch features: \([B, 12, 128]\) each
- TFT decoder features: \([B, 12, 128]\)
- concatenated fusion input: \([B, 12, 512]\)
- post-fusion GRN output: \([B, 12, 128]\)
- final quantile output: \([B, 12, 3]\)

---

## Appendix B. Practical reading lenses

When teaching or learning this model, three lenses tend to help most:

### Lens 1: Classical ML regression

- many inputs
- one target
- extend from one point estimate to several percentile cutoffs

### Lens 2: Conditional CDF

- quantiles are inverse-CDF cut points of the conditional target distribution

### Lens 3: Forecast interval view

- lower and upper quantiles define uncertainty intervals over the future horizon
