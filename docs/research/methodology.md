# Methodology

This file is the canonical methods section for the research companion. It owns the method framing, architecture, probabilistic objective, training protocol, reproducibility hooks, evaluation protocol, and method-specific constraints without duplicating the deeper material that belongs in [`introduction.md`](introduction.md), [`dataset.md`](dataset.md), [`results_and_discussion.md`](results_and_discussion.md), [`references.md`](references.md), and [`../reference/data_and_model_contract.md`](../reference/data_and_model_contract.md).

## 4.1 Method Framing And Research Positioning

This repository treats probabilistic glucose forecasting as a multi-horizon sequence-modeling problem. The current implementation is a late-fusion hybrid that combines Temporal Convolutional Network (TCN) branches, a Temporal Fusion Transformer (TFT) branch, a gated residual fusion block, and a quantile forecasting head [1]-[6].

The method is not presented as a new foundation architecture or as a claim that one existing sequence family is sufficient on its own. It is a repository-specific adaptation of established modeling components to the glucose-forecasting setting. The central method-level claim is narrower and clearer: semantically grouped inputs, parallel TCN and TFT latent branches, gated late fusion, and direct quantile supervision together define a coherent probabilistic forecasting pipeline for this task.

This section stays focused on the method contract itself. Dataset provenance, full preprocessing semantics, literature synthesis, and result interpretation are intentionally deferred to the other research files that already own those concerns.

## 4.2 Rationale For Method Choice

The model combines two complementary sequence-modeling families. TCNs provide causal convolutions, dilation, and efficient extraction of short- and mid-range temporal patterns from history-only inputs [2]. TFT provides a structured mechanism for combining static context, historical observations, and future-known covariates in a multi-horizon forecasting setting [1].

Prior hybrid TCN-TFT work in other forecasting domains motivates using both families together rather than forcing one branch to dominate the entire problem [3], [4]. The repository adopts that intuition in a late-fusion form: TCN and TFT build separate horizon-aligned latent representations and meet only at the fusion layer.

Quantile regression is used because glucose forecasting is intrinsically uncertain. A point-only predictor would collapse the output to one central estimate. Quantile prediction instead preserves lower, central, and upper conditional forecasts of the same future trajectory [5], [6].

## 4.3 Forecasting Pipeline

The repository's end-to-end forecasting pipeline can be summarized in six stages:

1. standardize the raw AZT1D source into one cleaned canonical table
2. construct legal encoder-decoder windows without crossing temporal gaps
3. pack each window into grouped tensors with static, known, observed, and target-history roles
4. run three TCN branches and one TFT branch in parallel
5. concatenate the horizon-aligned latent features and fuse them with a GRN
6. emit per-horizon quantile forecasts and evaluate them on held-out data

## 4.4 Model Architecture

The current model is `FusedModel`, a late-fusion hybrid composed of one TFT branch, three TCN branches, one post-branch GRN, and one final quantile head.

![`../assets/FusedModel_architecture.png`](../assets/FusedModel_architecture.png)

### 4.4.1 Input Representation

The model receives grouped batch tensors rather than one flat feature matrix. At runtime, the DataModule binds the prepared dataset into four semantic roles:

- static features
- known-ahead temporal features
- observed-only temporal features
- target history

This grouping matters because the TCN and TFT branches do not consume identical information budgets. The grouped batch contract preserves causal availability instead of flattening every feature into one anonymous input tensor.

Default sequence lengths are:

- encoder length \(L_e = 168\)
- prediction length \(L_d = 12\)
- sampling interval = 5 minutes

That corresponds to 14 hours of history and 1 hour of forecast horizon in the default configuration. Exact feature semantics, vocabulary definitions, and batch shapes remain documented in [`dataset.md`](dataset.md).

### 4.4.2 Temporal Convolutional Branch

The TCN side consumes only encoder-side observed continuous signals together with target history. The repository instantiates three parallel branches with kernel sizes `3`, `5`, and `7`, so the same history can be viewed through multiple receptive-field biases.

The shared default TCN configuration is:

| channels | dilations | dropout | normalization | prediction length |
|---|---|---|---|---|
| `(64, 64, 128)` | `(1, 2, 4)` | `0.1` | `layer_norm` | `12` |

Each branch is causal and produces horizon-aligned latent features rather than final quantile outputs [2].

![`../assets/TCN_architecture.png`](../assets/TCN_architecture.png)

### 4.4.3 Temporal Fusion Transformer Branch

The TFT branch consumes grouped static features, encoder history, and decoder-known future inputs [1]. Unlike the TCN path, it is explicitly designed to preserve semantic feature roles across the encoder-decoder example axis.

The default TFT hyperparameters are:

| hidden size | attention heads | dropout | attention dropout | layer norm epsilon | encoder length | total sequence length |
|---|---|---|---|---|---|---|
| `128` | `4` | `0.1` | `0.0` | `1e-3` | `168` | `180` |

Categorical embedding cardinalities and some variable counts are runtime-bound from the prepared dataset rather than fixed in the manuscript, so the implementation remains aligned with the actual feature schema used for a run.

![`../assets/TFT_architecture.PNG`](../assets/TFT_architecture.PNG)

### 4.4.4 Latent Fusion Mechanism

The decoder-aligned TFT latent representation is concatenated with the three TCN branch representations along the feature dimension. A GRN, adapted from the TFT design, then compresses the concatenated tensor back to the shared hidden width and acts as the nonlinear gated fusion block [1], [4].

This is a late-fusion method. The repository does not first force each branch to emit its own final probabilistic forecast and then reconcile the outputs. Instead, the branches meet in latent space before the final prediction head.

### 4.4.5 Quantile Forecasting Head

The fused hidden state is passed to a position-wise residual MLP head (`NNHead`) that emits one output channel per requested quantile at each horizon step.

The default head configuration is:

| input size | hidden size | feedforward size | residual blocks | dropout | quantiles |
|---|---|---|---|---|---|
| `128` | `128` | `256` | `2` | `0.1` | `(0.1, 0.5, 0.9)` |

The output tensor therefore has shape \([B, L_d, |\mathcal{Q}|]\), where \(\mathcal{Q}\) is the configured quantile set.

## 4.5 Probabilistic Learning Objective

Training uses pinball loss, also called quantile loss, over the forecast quantile tensor [5], [6]. For one target value \(y\), one predicted quantile \(\hat{y}_q\), and one quantile level \(q\), the loss is:

\[
\mathcal{L}_q(y, \hat{y}_q) =
\begin{cases}
q(y - \hat{y}_q), & y \ge \hat{y}_q \\
(1 - q)(\hat{y}_q - y), & y < \hat{y}_q
\end{cases}
\]

For the configured quantile set \(\mathcal{Q}\) and prediction horizon \(L_d\), the repository uses the mean loss over batch items, horizon steps, and quantile channels:

\[
\mathcal{L} =
\frac{1}{|\mathcal{Q}|L_d}
\sum_{q \in \mathcal{Q}}
\sum_{\tau=1}^{L_d}
\mathcal{L}_q(y_{t+\tau}, \hat{y}_{t+\tau,q})
\]

At \(q = 0.5\), the objective corresponds to median estimation. The outer quantiles define lower and upper conditional cutoffs of the same future target. This section defines the optimization target only; metric interpretation is handled in [`results_and_discussion.md`](results_and_discussion.md).

## 4.6 Training Procedure

Each training step follows the same branch-and-fuse pattern. The grouped batch is first partitioned into the views required by the TCN and TFT branches, then the branch features are fused, projected into quantiles, and supervised with pinball loss.

```mermaid
flowchart TD
    A[Grouped batch] --> B[Split grouped inputs]
    B --> C[TCN history view]
    B --> D[TFT grouped view]
    C --> E1[TCN k=3]
    C --> E2[TCN k=5]
    C --> E3[TCN k=7]
    D --> F[TFT branch]
    E1 --> G[Latent concatenation]
    E2 --> G
    E3 --> G
    F --> G
    G --> H[GRN fusion]
    H --> I[Quantile head]
    I --> J[Forecast quantiles]
    J --> K[Mean pinball loss]
    K --> L[Adam update]
```

Concretely, the split step means that encoder-side tensors are routed into two branch-specific views: a history-only TCN input built from observed continuous variables and target history, and a semantically grouped TFT input built from static, historical, and future-known covariates. Operationally, the repository delegates epoch-loop mechanics to PyTorch Lightning. The model owns forward computation, loss computation, train, validation, and test step semantics, and optimizer construction, while Lightning owns batching, backpropagation, validation scheduling, callback dispatch, and checkpoint management.

## 4.7 Hyperparameters And Experimental Configuration

The table below records the repository's current default experiment configuration. These values are the canonical defaults exposed by the codebase. They define the present implementation, but they do not by themselves claim that the configuration is globally optimal.

| sampling interval | encoder length | prediction length | window stride | split ratio | split mode | batch size | optimizer | learning rate | weight decay | max epochs | early stopping patience | quantiles |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `5` minutes | `168` | `12` | `1` | `70 / 15 / 15` | within-subject chronological | `64` | `Adam` | `1e-3` | `0.0` | `20` | `5` validation checks | `(0.1, 0.5, 0.9)` |

These values come directly from the repository's typed config defaults and provide a reproducible baseline for the present implementation.

## 4.8 Reproducibility And Runtime Traceability

The repository explicitly distinguishes between declarative config and runtime-bound config. Declarative config is built at the entry surface. The runtime-bound variant is produced after the DataModule has inspected the prepared dataset and resolved feature-dependent metadata.

This distinction is methodologically important because the TFT branch depends on runtime-derived categorical and sequence information. The training wrapper therefore prepares the data before constructing the final model instance.

The default artifact directory is `artifacts/main_run/`. A run can emit:

- `run_summary.json`
- `report_index.json`
- `metrics_summary.json`
- checkpoints
- logs and telemetry
- profiler outputs
- structured post-run reports

Together, these artifacts provide a paper trail for the effective config, optimizer settings, runtime tuning choices, and evaluation outputs of a run. Local copies of the most relevant papers are preserved in `docs/publications/`, while the formatted bibliography is maintained in [`references.md`](references.md).

## 4.9 Evaluation Protocol

The default repository experiment uses a `70 / 15 / 15` split with within-subject chronological partitioning. Each legal sample contains `168` historical steps and `12` future target steps on a 5-minute grid.

Held-out evaluation is performed on the forecast quantile tensor rather than on scalar test loss alone. The standard metric set includes:

- MAE on the median-like forecast
- RMSE on the median-like forecast
- pinball loss on the full quantile output
- prediction-interval width
- empirical interval coverage

The repository can also produce grouped evaluations by horizon, subject, and glucose range, but those grouped outputs and their interpretation belong in [`results_and_discussion.md`](results_and_discussion.md) rather than in this methods section.

## 4.10 Validation And Sanity Checks

The current implementation includes several method-side sanity protections:

- causal TCN convolutions prevent future leakage through the convolutional path
- decoder inputs to TFT are restricted to future-known covariates
- legal windows cannot cross temporal discontinuities in the cleaned timeline
- runtime metadata binding keeps model dimensions aligned with the prepared data
- quantile tensors are checked against the configured quantile set during loss and evaluation paths
- validation loss can be used to drive checkpoint ranking and early stopping

These checks do not replace full empirical validation, but they reduce common failure modes such as illegal windows, shape drift, and accidental leakage through the forecasting interface.

## 4.11 Methodological Scope And Constraints

This methods section should be read with several boundaries in mind.

- the repository defaults are baseline experiment settings, not final tuned study settings
- the default split mode answers a within-subject forecasting question rather than unseen-subject generalization
- `window_stride = 1` produces heavily overlapping windows
- the default quantile set is coarse and does not exhaust the possible calibration surface
- the present method is architecturally coherent, but ablation evidence against strong TCN-only and TFT-only baselines is still a later-stage requirement

Those boundaries are part of the method definition itself and should remain visible in the paper-facing narrative instead of being hidden until the discussion section.
