# Forensic Artifact Diagnosis of the Probabilistic Glucose Forecasting Run

## Abstract

This document is a forensic, artifact-tied diagnosis of the uploaded run artifacts in `artifacts/main_run/`. It is not a generic explanation of forecasting metrics. Every major statement in this document is tied to the actual exported files from the run: the shared-report tables, the run summary, the prediction table, the telemetry stream, and the profiler outputs.

The purpose of this document is fivefold:

1. To explain **what is being recorded** by the reporting pipeline.
2. To explain **what each recorded quantity means mathematically and operationally**.
3. To explain **how to interpret the results in this specific run**, using the actual numbers in the artifacts.
4. To identify **failure modes that only become visible when multiple artifacts are read together**.
5. To provide a reading order for the artifact tree so that future evaluations become systematic rather than ad hoc.

---

## 1. Scope of the Artifact Diagnosis

### 1.1 Files examined

This diagnosis was tied directly to the following files in the uploaded archive:

- `artifacts/main_run/run_summary.json`
- `artifacts/main_run/run.log`
- `artifacts/main_run/test_predictions.csv`
- `artifacts/main_run/telemetry.csv`
- `artifacts/main_run/profiler/fit-simple_profiler.txt`
- `artifacts/main_run/profiler/test-simple_profiler.txt`
- `artifacts/main_run/profiler/predict-simple_profiler.txt`
- `artifacts/main_run/reports/data_summary.json`
- `artifacts/main_run/reports/artifacts/shared_report/metrics_summary.json`
- `artifacts/main_run/reports/artifacts/shared_report/scalars.json`
- `artifacts/main_run/reports/artifacts/shared_report/text.json`
- `artifacts/main_run/reports/artifacts/shared_report/tables/by_horizon.csv`
- `artifacts/main_run/reports/artifacts/shared_report/tables/by_subject.csv`
- `artifacts/main_run/reports/artifacts/shared_report/tables/by_glucose_range.csv`
- `artifacts/main_run/reports/artifacts/shared_report/tables/prediction_table.csv`

The HTML reports were not used as primary evidence because the underlying CSV/JSON tables already expose the quantitative content driving those plots.

### 1.2 What kind of run this was

From `run_summary.json`, this run used:

- dataset: **AZT1D**
- sampling interval: **5 minutes**
- encoder length: **168 steps** = **14 hours**
- prediction length: **12 steps** = **60 minutes**
- quantiles: **0.1, 0.5, 0.9**
- batch size: **128**
- optimizer: **Adam**
- learning rate: **0.001**
- device profile resolved as **apple-silicon / mps**
- training epochs in this run summary: **1**

The artifact set therefore describes a **one-hour probabilistic glucose forecast**, emitted every 5 minutes, conditioned on a 14-hour encoder history.

---

## 2. The Architecture Context Matters for Interpreting the Artifacts

The run log shows that the model is not a plain regressor. It is a fused system containing:

- a **TCN branch with kernel size 3**
- a **TCN branch with kernel size 5**
- a **Temporal Fusion Transformer branch**
- a **fusion GRN**
- a downstream **MLP head**
- a **quantile projection / quantile output head**

This architectural fact matters because the artifacts do not merely reflect point-forecast behavior. They reflect how a **multi-branch sequence model** converts shared temporal features into **quantile forecasts**. When the exported artifacts show calibration failure, horizon drift, or systematic bias, those failures should be interpreted as failures of the **fused representation and quantile head**, not merely of a scalar point regressor.

---

## 3. Artifact Topology: What Is Recorded and Where to Look

The run exports information at several layers. Each layer answers a different scientific question.

### 3.1 Dataset-level structural summary

Look at:

- `reports/data_summary.json`

This file answers questions such as:

- How many rows were in the processed dataset?
- How many subjects exist?
- What time span is covered?
- Are there duplicate `(subject_id, timestamp)` rows?
- Are there missing values?
- How many train / val / test rows and windows were created?

From the artifact:

- total processed rows: **300,607**
- total subjects in processed dataset: **25**
- duplicate `(subject, timestamp)` rows: **0**
- missing values by column: **none across the listed columns**
- train rows: **210,411**
- val rows: **45,082**
- test rows: **45,114**
- train windows: **143,237**
- val windows: **27,221**
- test windows: **28,234**

Interpretation: the exported evaluation is built on a dataset with **no missing values in the reported columns** and **no duplicate subject–timestamp rows**, which is important because it means the observed forecast pathologies are not immediately explainable by trivial structural corruption of the processed table.

### 3.2 Run-level summary

Look at:

- `run_summary.json`

This is the run’s compact metadata record. It tells you:

- configuration of the data pipeline
- model configuration
- device profile
- optimizer settings
- checkpoint selection used for evaluation
- top-level test metrics
- number of prediction batches

This is where you verify the experimental context before interpreting downstream metrics.

### 3.3 Global scalar performance summary

Look at:

- `reports/artifacts/shared_report/metrics_summary.json`
- `reports/artifacts/shared_report/scalars.json`

These files answer the question:

> “If I collapse the entire test set into one global summary, what does the model look like?”

This is the correct file for first-pass triage, but not for root-cause analysis.

### 3.4 Conditional / stratified performance tables

Look at:

- `tables/by_horizon.csv`
- `tables/by_subject.csv`
- `tables/by_glucose_range.csv`

These answer:

- How performance changes with forecast distance
- How performance differs by person
- How performance differs by clinically meaningful glucose region

These are the most important files for diagnosis.

### 3.5 Instance-level forecast trace

Look at:

- `tables/prediction_table.csv`
- `test_predictions.csv`

These are row-level forecast records and are indispensable for forensic diagnosis. Each row includes:

- the forecast timestamp
- subject id
- horizon index
- true glucose
- last observed glucose
- q10, q50, q90
- residual
- interval width

This is the only level where you can reconstruct **trajectory behavior**, **lag**, **overreaction**, **systematic drift**, and **coverage failure** one row at a time.

### 3.6 System and execution observability

Look at:

- `telemetry.csv`
- profiler outputs
- TensorBoard event files under `logs/`

These tell you whether poor forecast behavior might be entangled with runtime issues or unstable resource behavior.

---

## 4. Global Performance Summary: What the Run Looks Like Before Stratification

From `metrics_summary.json`, the global scalar summary is:

| Quantity | Value |
|---|---:|
| forecast rows | 338,808 |
| subjects appearing in report | 24 |
| horizons | 12 |
| MAE | 20.1709 |
| RMSE | 27.2063 |
| bias | 12.9791 |
| overall pinball loss | 6.6697 |
| mean interval width | 58.8711 |
| empirical interval coverage | 0.7612 |
| q10 pinball loss | 4.8160 |
| q50 pinball loss | 10.0855 |
| q90 pinball loss | 5.1077 |
| target mean | 145.8902 |
| target std | 47.1307 |
| prediction mean | 158.8693 |
| prediction std | 44.8046 |
| near-zero interval fraction | 0.0000 |

### 4.1 What these numbers mean at face value

The global MAE of **20.17 mg/dL** means that, averaged across all forecast rows, the median forecast is off by about twenty mg/dL. But this number by itself is incomplete.

The global RMSE of **27.21 mg/dL** is materially larger than MAE. This tells you that the error distribution has a substantial tail: the run does not merely make moderate errors everywhere; it also makes a non-trivial number of larger misses.

The global bias of **+12.98 mg/dL** is the first major red flag. It means that, on average, the model predicts glucose values materially **higher** than what actually occurred.

This is not a subtle effect:

- target mean = **145.89**
- prediction mean = **158.87**
- difference = **12.98 mg/dL**

This is strong evidence that the model carries a **systematic upward drift**.

### 4.2 Why the global coverage number is insufficient by itself

The 10–90 interval has an expected nominal coverage of roughly **80%**. The empirical interval coverage is **76.1%**.

At the global level that already implies undercoverage:

- expected: **80.0%**
- observed: **76.1%**
- shortfall: **3.9 percentage points**

However, the deeper problem is not merely that coverage is below target. The deeper problem is **how** it fails. As shown later from the row-level table, the misses are highly asymmetric:

- rows below q10: **20.4%**
- rows above q90: **3.5%**
- rows inside interval: **76.1%**

If the model were simply under-dispersed but otherwise centered, misses below q10 and above q90 would be more balanced. They are not. The overwhelming failure mode is that the true glucose lands **below the lower bound**, which is precisely what one expects from a model with a substantial **positive bias**.

### 4.3 Why “near_zero_interval_fraction = 0” is only a weak reassurance

The near-zero interval fraction is **0.0**, which tells you the quantile head is not collapsing into degenerate intervals. That is useful. It rules out one pathological failure mode: the model is not producing numerically identical q10 and q90 at scale.

But this does **not** imply that the uncertainty is good. A model can have non-degenerate intervals and still be badly calibrated if:

- intervals are too narrow relative to actual uncertainty,
- intervals are shifted upward,
- interval growth with horizon is too weak.

This run exhibits exactly that kind of non-degenerate-but-miscalibrated behavior.

---

## 5. Deterministic Error Metrics: What Is Being Measured and How to Interpret It

### 5.1 Mean Absolute Error (MAE)

**Where to look:**

- `metrics_summary.json`
- `by_horizon.csv`
- `by_subject.csv`
- `by_glucose_range.csv`

**What it records:**  
MAE is the mean absolute residual:

\[
MAE = \frac{1}{N} \sum_i |\hat y_i - y_i|
\]

It measures the average magnitude of error without regard to sign.

**How to interpret it in this run:**  
At the global level, MAE is **20.17 mg/dL**. But the real meaning emerges only after stratification. In this run, MAE is not constant. It varies strongly by:

- horizon,
- subject,
- glucose range.

Therefore, the global MAE should be treated as a population average across several distinct regimes, not as a single coherent description of forecast quality.

### 5.2 RMSE

**What it records:**  
RMSE is the square-root mean squared residual:

\[
RMSE = \sqrt{\frac{1}{N} \sum_i (\hat y_i - y_i)^2}
\]

This weights larger misses more heavily than MAE.

**How to interpret it in this run:**  
RMSE (**27.21**) being noticeably above MAE (**20.17**) implies that the model’s error distribution is not tight and symmetric around modest mistakes. There is a meaningful tail of larger failures.

This is confirmed by the row-level absolute error quantiles:

| absolute error quantile | value |
|---|---:|
| 50th percentile | 14.95 |
| 75th percentile | 26.86 |
| 90th percentile | 43.91 |
| 95th percentile | 57.56 |
| 99th percentile | 86.45 |

Interpretation: once you move beyond the median error regime, the tail grows quickly. The 95th percentile absolute error exceeds **57.6 mg/dL**, which is not a mild miss.

### 5.3 Bias

**What it records:**  
Bias is the signed mean residual:

\[
Bias = \frac{1}{N} \sum_i (\hat y_i - y_i)
\]

**How to interpret it in this run:**  
Bias is **+12.98 mg/dL**, so the model systematically predicts glucose levels that are higher than observed. This is the dominant organizing feature of the entire artifact set.

The residual quantiles make the asymmetry obvious:

| residual quantile | value |
|---|---:|
| 1st percentile | -55.25 |
| 5th percentile | -25.25 |
| 10th percentile | -11.97 |
| 50th percentile | 11.66 |
| 90th percentile | 39.94 |
| 95th percentile | 53.37 |
| 99th percentile | 84.31 |

The median residual is **+11.66 mg/dL**, not close to zero. This means that positive residuals are not just coming from a few outliers. The central mass of the residual distribution itself is shifted upward.

Also note the sign imbalance:

- overprediction rows: **79.7%**
- underprediction rows: **20.3%**

That is extremely strong directional skew.

---

## 6. Probabilistic Forecast Metrics: What the Quantile Outputs Are Recording

### 6.1 Pinball loss

**Where to look:**

- `metrics_summary.json`
- `by_horizon.csv`
- `by_subject.csv`
- `by_glucose_range.csv`

**What it records:**  
Pinball loss is the standard proper scoring rule for quantile regression. For quantile \(q\),

\[
L_q(y, \hat y_q) =
\begin{cases}
q (y - \hat y_q), & y \ge \hat y_q \\
(1-q)(\hat y_q - y), & y < \hat y_q
\end{cases}
\]

It penalizes under- and over-estimation asymmetrically according to the quantile level.

**What it means here:**  
The run reports:

- q10 pinball = **4.8160**
- q50 pinball = **10.0855**
- q90 pinball = **5.1077**
- overall pinball = **6.6697**

The q50 pinball is the highest, which is unsurprising because the median forecast is trying to capture the center of the distribution and is directly exposed to the upward bias. But the real lesson is not the ranking alone. The real lesson is that pinball loss must be interpreted together with interval coverage and interval asymmetry. In this run, the quantile set is not simply noisy; it is **directionally shifted upward**.

### 6.2 Prediction interval width

**What it records:**  
For the 10–90 interval,

\[
Width = \hat y_{0.9} - \hat y_{0.1}
\]

**What it means globally:**  
The mean interval width is **58.87 mg/dL**.

**What the width distribution looks like:**

| width quantile | value |
|---|---:|
| 1st percentile | 39.79 |
| 5th percentile | 44.33 |
| 25th percentile | 49.59 |
| 50th percentile | 55.29 |
| 75th percentile | 65.26 |
| 95th percentile | 86.17 |
| 99th percentile | 98.34 |

There is real uncertainty spread here; the model is not producing trivial intervals. But width alone is not enough. The intervals can be non-trivial and still be wrong if their **center** is shifted or if width grows too slowly with forecast distance.

### 6.3 Interval coverage

**What it records:**  
Coverage is the fraction of rows whose true target lies inside the reported interval.

**What it means globally:**  
Coverage is **76.1%**, versus the nominal target of **80%** for a 10–90 interval.

**What makes this run specifically problematic:**  
The coverage misses are highly one-sided.

- below q10: **20.4%**
- above q90: **3.5%**

Interpretation: when the interval misses, it is much more likely to miss because the true glucose is **lower than the lower quantile**. This is not neutral undercoverage. It is **undercoverage generated by upward shift**.

---

## 7. Horizon-Wise Forensics: The One-Hour Forecast Degrades Fast and the Intervals Do Not Keep Up

**Where to look:**

- `tables/by_horizon.csv`
- `horizon_metrics.html`
- `horizon_bias.html`
- `horizon_coverage.html`

The by-horizon table is one of the most important artifacts because it answers:

> “As the forecast moves from 5 minutes ahead to 60 minutes ahead, does error grow at a rate that is matched by uncertainty?”

### 7.1 Actual by-horizon values

|   horizon_index |   minutes_ahead |     mae |    rmse |     bias |   mean_interval_width |   coverage |
|----------------:|----------------:|--------:|--------:|---------:|----------------------:|-----------:|
|               0 |               5 | 12.57   | 16.6572 | 11.4703  |               58.1096 |   0.964086 |
|               1 |              10 | 12.1101 | 16.9619 |  9.87821 |               57.6453 |   0.943154 |
|               2 |              15 | 14.4615 | 19.3927 | 11.1769  |               58.0746 |   0.897216 |
|               3 |              20 | 16.0484 | 21.4127 | 11.4504  |               58.2207 |   0.85096  |
|               4 |              25 | 17.5839 | 23.5799 | 11.5959  |               58.3345 |   0.810512 |
|               5 |              30 | 19.7054 | 26.0075 | 12.7609  |               58.6378 |   0.762804 |
|               6 |              35 | 21.1211 | 27.635  | 12.9021  |               58.8769 |   0.729829 |
|               7 |              40 | 22.5367 | 29.4409 | 13.2088  |               58.9734 |   0.697067 |
|               8 |              45 | 24.654  | 31.7813 | 14.8077  |               59.5796 |   0.654424 |
|               9 |              50 | 25.8014 | 33.2918 | 14.9547  |               59.8414 |   0.631933 |
|              10 |              55 | 27.1052 | 34.6514 | 15.4987  |               59.9577 |   0.608061 |
|              11 |              60 | 28.3536 | 36.0618 | 16.0443  |               60.2018 |   0.58465  |

### 7.2 What the horizon table says quantitatively

From horizon 0 to horizon 11:

- MAE rises from **12.57** to **28.35**
- RMSE rises from **16.66** to **36.06**
- bias rises from **11.47** to **16.04**
- coverage falls from **96.4%** to **58.5%**
- mean interval width rises only from **58.11** to **60.20**

Expressed as relative change:

- MAE increases by **125.6%**
- RMSE increases by **116.5%**
- bias increases by **39.9%**
- interval width increases by only **3.6%**
- coverage drops by **37.9 percentage points**

### 7.3 Interpretation

This is one of the clearest findings in the entire artifact set:

1. **Error grows strongly with horizon.**
2. **Bias also grows with horizon.**
3. **Coverage collapses with horizon.**
4. **Interval width barely expands.**

This means the uncertainty mechanism is not scaling with the difficulty of the task. Put differently: the forecast gets much harder as you move out toward 60 minutes, but the interval inflation is too weak to reflect that increased uncertainty.

### 7.4 Why the coverage collapse is especially informative

The row-level horizon calibration decomposition shows the asymmetry:

|   horizon_index |   coverage |   below_q10 |   above_q90 |   width |     mae |     bias |
|----------------:|-----------:|------------:|------------:|--------:|--------:|---------:|
|               0 |   0.964086 |   0.034462  |  0.00145215 | 58.1096 | 12.57   | 11.4703  |
|               1 |   0.943154 |   0.0529858 |  0.00386059 | 57.6453 | 12.1101 |  9.87821 |
|               2 |   0.897216 |   0.093646  |  0.00913792 | 58.0746 | 14.4615 | 11.1769  |
|               3 |   0.85096  |   0.132394  |  0.0166466  | 58.2207 | 16.0484 | 11.4504  |
|               4 |   0.810512 |   0.164128  |  0.0253595  | 58.3345 | 17.5839 | 11.5959  |
|               5 |   0.762804 |   0.205178  |  0.0320181  | 58.6378 | 19.7054 | 12.7609  |
|               6 |   0.729829 |   0.229581  |  0.0405894  | 58.8769 | 21.1211 | 12.9021  |
|               7 |   0.697067 |   0.25487   |  0.0480626  | 58.9734 | 22.5367 | 13.2088  |
|               8 |   0.654424 |   0.294397  |  0.0511794  | 59.5796 | 24.654  | 14.8077  |
|               9 |   0.631933 |   0.309981  |  0.058086   | 59.8414 | 25.8014 | 14.9547  |
|              10 |   0.608061 |   0.32978   |  0.0621591  | 59.9577 | 27.1052 | 15.4987  |
|              11 |   0.58465  |   0.347772  |  0.0675781  | 60.2018 | 28.3536 | 16.0443  |

At 5 minutes:

- coverage = **96.4%**
- below q10 = **3.4%**
- above q90 = **0.1%**

At 60 minutes:

- coverage = **58.5%**
- below q10 = **34.8%**
- above q90 = **6.8%**

Interpretation: as horizon lengthens, the probability that the target falls **below q10** rises dramatically, from **3.4%** to **34.8%**. The upper-tail miss rate also increases, but much less.

This tells you the problem is not just uncertainty underestimation. It is that the entire predictive distribution drifts too high as forecast distance increases.

---

## 8. Subject-Level Forensics: Performance Heterogeneity Is Real, and One Subject Is a Tiny-Sample Outlier

**Where to look:**

- `tables/by_subject.csv`
- `subject_metrics.html`

### 8.1 Important structural observation

The data summary says the dataset contains **25 subjects**, and the test split reports **25 test subjects**. However, `by_subject.csv` contains only **24 subjects** in the shared report.

The subject missing from the report is:

- Subject 17

This discrepancy is worth documenting because it means the subject-level artifact layer is not identical to the dataset-level split layer. The most likely explanation is that one subject did not produce valid prediction rows after the rolling-window and/or forecast-export logic, but the artifact alone does not fully explain that discrepancy. It should be treated as a minor report-consistency issue to verify in the pipeline.

### 8.2 Worst-subject table by MAE

| subject_id   |   count |     mae |    rmse |    bias |   coverage |   below_q10 |   above_q90 |   width |
|:-------------|--------:|--------:|--------:|--------:|-----------:|------------:|------------:|--------:|
| Subject 5    |      96 | 40.9125 | 44.7395 | 40.9125 |   0.229167 |    0.770833 |   0         | 55.7775 |
| Subject 10   |   19248 | 28.0897 | 40.2892 | 24.5975 |   0.666823 |    0.319306 |   0.0138716 | 57.6388 |
| Subject 23   |   18060 | 25.1482 | 34.3621 | 15.393  |   0.674086 |    0.265116 |   0.0607973 | 58.7313 |
| Subject 2    |   14268 | 23.8426 | 31.4352 | 14.6031 |   0.707107 |    0.238716 |   0.0541772 | 61.3122 |
| Subject 14   |   14880 | 22.9072 | 30.2202 | 16.0146 |   0.617272 |    0.34422  |   0.0385081 | 50.7814 |
| Subject 11   |   13884 | 22.5914 | 28.1048 | 16.8477 |   0.707145 |    0.262388 |   0.0304667 | 63.0143 |
| Subject 7    |   17592 | 22.2753 | 28.8891 | 17.8837 |   0.722885 |    0.258583 |   0.0185312 | 58.7527 |
| Subject 4    |   20808 | 21.7864 | 27.6163 | 12.7275 |   0.756824 |    0.19579  |   0.0473856 | 64.1898 |

### 8.3 Best-subject table by MAE

| subject_id   |   count |     mae |    rmse |     bias |   coverage |   below_q10 |   above_q90 |   width |
|:-------------|--------:|--------:|--------:|---------:|-----------:|------------:|------------:|--------:|
| Subject 13   |   11280 | 13.8555 | 18.3493 |  9.90141 |   0.873936 |    0.105674 |  0.0203901  | 54.7494 |
| Subject 6    |   18624 | 15.6371 | 21.4961 |  9.67226 |   0.850408 |    0.121241 |  0.0283505  | 57.526  |
| Subject 3    |    7428 | 15.6438 | 19.3176 | 13.2011  |   0.831179 |    0.16734  |  0.00148088 | 56.4514 |
| Subject 19   |   21420 | 16.1701 | 21.7106 | 12.0267  |   0.811718 |    0.169701 |  0.0185808  | 53.8414 |
| Subject 18   |   16788 | 17.0447 | 21.7708 |  9.56884 |   0.834286 |    0.142304 |  0.0234096  | 60.095  |
| Subject 1    |   13908 | 17.7048 | 22.1461 | 13.7483  |   0.788251 |    0.194564 |  0.0171844  | 57.7059 |
| Subject 20   |   17484 | 17.7308 | 22.9923 | 12.5765  |   0.777111 |    0.195321 |  0.0275681  | 55.3108 |
| Subject 16   |   17760 | 18.0827 | 26.0083 | 13.0616  |   0.829336 |    0.146565 |  0.0240991  | 58.9267 |

### 8.4 Subject interpretation

The spread across subjects is substantial.

- best subject by MAE: **Subject 13** with MAE **13.86**
- worst subject by MAE: **Subject 5** with MAE **40.91**

The worst row in the table is **Subject 5**, but it has only **96** prediction rows. That means it is a **tiny-sample outlier** and should not be interpreted as representative of stable subject-level behavior. It is still important, but the small count means it can be dominated by a very short and pathological forecast segment.

More meaningful high-error subjects with larger sample counts include:

- **Subject 10**: count **19,248**, MAE **28.09**, bias **24.60**, coverage **66.7%**
- **Subject 23**: count **18,060**, MAE **25.15**, bias **15.39**, coverage **67.4%**
- **Subject 14**: count **14,880**, MAE **22.91**, bias **16.01**, coverage **61.7%**

The best sustained subject-level performance appears in:

- **Subject 13**: MAE **13.86**, coverage **87.4%**
- **Subject 6**: MAE **15.64**, coverage **85.0%**
- **Subject 3**: MAE **15.64**, coverage **83.1%**

### 8.5 Diagnostic meaning

The subject table tells you the model is not uniformly bad or uniformly good. Instead, it shows **heterogeneous subject sensitivity**:

- some individuals are tracked reasonably,
- others experience large upward bias and lower coverage.

This is exactly the kind of pattern one expects when a fused temporal model is learning a strong global prior but not fully adapting to subject-specific dynamics.

---

## 9. Glucose-Range Forensics: This Is Where the Main Clinical Failure Reveals Itself

**Where to look:**

- `tables/by_glucose_range.csv`
- `glucose_range_metrics.html`

### 9.1 Actual glucose-range table

| range     |   count |   coverage |   below_q10 |   above_q90 |   width |     mae |    rmse |     bias |
|:----------|--------:|-----------:|------------:|------------:|--------:|--------:|--------:|---------:|
| lt_70     |    4573 |   0.201618 |   0.798382  |   0         | 43.8575 | 39.0538 | 46.1425 | 39.0425  |
| 70_to_180 |  268935 |   0.74959  |   0.230595  |   0.0198152 | 54.6184 | 19.3524 | 26.3722 | 15.3664  |
| gt_180    |   65300 |   0.848331 |   0.0533538 |   0.0983155 | 77.437  | 22.2199 | 28.7568 |  1.32165 |

### 9.2 Interpretation by range

#### a. Hypoglycemia region (`lt_70`)

This is the most alarming stratum in the entire report.

- count: **4,573**
- MAE: **39.05**
- RMSE: **46.14**
- bias: **39.04**
- width: **43.86**
- coverage: **20.2%**
- below q10: **79.8%**
- above q90: **0.0%**

This means that in the hypoglycemic range:

- the model overpredicts by about **39.0 mg/dL** on average,
- the 10–90 interval contains the truth only about **20.2%** of the time,
- nearly **79.8%** of targets fall below q10.

This is not a mild calibration issue. It is a **systematic failure to represent low glucose states**.

#### b. Normal glycemic region (`70_to_180`)

This is the bulk regime.

- count: **268,935**
- MAE: **19.35**
- bias: **15.37**
- coverage: **75.0%**

Even in the normal range, the model is materially upward biased, by about **15.4 mg/dL**. Because this range dominates the row count, it heavily shapes the global metrics.

#### c. Hyperglycemic region (`gt_180`)

This regime behaves differently.

- count: **65,300**
- MAE: **22.22**
- bias: **1.32**
- width: **77.44**
- coverage: **84.8%**

The hyperglycemic stratum has much wider intervals (**77.4 mg/dL**) and better coverage (**84.8%**) than the normal and low ranges. Its bias is only **1.3 mg/dL** globally.

Interpretation: the uncertainty mechanism is giving the model much more room in the high-glucose regime, and that helps calibration there. By contrast, the low-glucose regime is assigned the **narrowest** intervals of the three ranges, even though it is the regime where the model needs the most caution.

### 9.3 Core clinical diagnosis from the range table

The worst clinically important failure is not “general error.” It is specifically:

> The model is strongly upward biased in low glucose, and its intervals are too narrow there.

That combination is especially dangerous because it means the system can look numerically stable while systematically missing downward excursions.

---

## 10. Hypoglycemia Horizon Collapse: The Strongest Single Finding in the Artifact Set

The range-by-horizon decomposition for `lt_70` is the clearest forensic evidence in the archive.

### 10.1 Hypoglycemia by horizon

|   horizon_index |   count |     mae |    bias |   coverage |   width |   below_q10 |
|----------------:|--------:|--------:|--------:|-----------:|--------:|------------:|
|               0 |     383 | 10.8933 | 10.8583 |  0.814621  | 35.9742 |    0.185379 |
|               1 |     382 | 14.8746 | 14.8246 |  0.659686  | 37.0661 |    0.340314 |
|               2 |     381 | 19.292  | 19.2417 |  0.47769   | 38.3109 |    0.52231  |
|               3 |     380 | 25.5226 | 25.5226 |  0.265789  | 40.2039 |    0.734211 |
|               4 |     379 | 30.4095 | 30.4095 |  0.147757  | 41.5244 |    0.852243 |
|               5 |     379 | 36.9201 | 36.9201 |  0.0501319 | 43.2802 |    0.949868 |
|               6 |     381 | 42.0749 | 42.0749 |  0         | 44.8377 |    1        |
|               7 |     382 | 47.8398 | 47.8398 |  0         | 46.382  |    1        |
|               8 |     382 | 53.5034 | 53.5034 |  0         | 47.9096 |    1        |
|               9 |     381 | 57.6279 | 57.6279 |  0         | 49.0335 |    1        |
|              10 |     381 | 62.6436 | 62.6436 |  0         | 50.3    |    1        |
|              11 |     382 | 67.0284 | 67.0284 |  0         | 51.4651 |    1        |

### 10.2 Interpretation

At horizon 0 (5 minutes ahead):

- MAE = **10.89**
- bias = **10.86**
- coverage = **81.5%**

By horizon 5 (30 minutes ahead):

- MAE = **36.92**
- bias = **36.92**
- coverage = **5.0%**

By horizon 6 (35 minutes ahead):

- coverage = **0.0%**
- below q10 = **100.0%**

From horizon 6 through horizon 11:

- coverage is **0.0%**
- below q10 is **100.0%**

This means that for hypoglycemic targets at horizons 35–60 minutes ahead, **every single target falls below the model’s lower quantile**. In other words, even the model’s q10 forecast is too high for every low-glucose row in that horizon band.

That is a remarkably strong statement, and it comes directly from the actual row-level stratification. This is not “slightly miscalibrated.” It is a full collapse of lower-tail representation in the low-glucose, longer-horizon regime.

### 10.3 What this implies about model behavior

This pattern implies all of the following simultaneously:

1. The model has an upward prior that becomes stronger as horizon increases.
2. The interval expansion with horizon is too weak.
3. The lower quantile is not sufficiently sensitive to impending downward glucose trajectories.
4. The quantile head is not capturing asymmetric downside uncertainty where it matters most.

This is one of the strongest candidates for targeted model improvement.

---

## 11. Hyperglycemia Has a Different Failure Signature

### 11.1 Hyperglycemia by horizon

|   horizon_index |   count |     mae |       bias |   coverage |   width |   above_q90 |
|----------------:|--------:|--------:|-----------:|-----------:|--------:|------------:|
|               0 |    5458 | 16.4083 |  11.7948   |   0.982045 | 80.5721 |  0.00751191 |
|               1 |    5457 | 15.8526 |  10.1399   |   0.971963 | 80.0334 |  0.00989555 |
|               2 |    5454 | 17.5394 |   9.9295   |   0.951412 | 80.0006 |  0.0168684  |
|               3 |    5450 | 17.5588 |   6.33752  |   0.930826 | 78.8449 |  0.0313761  |
|               4 |    5446 | 20.1988 |   6.28473  |   0.889277 | 78.8847 |  0.0497613  |
|               5 |    5442 | 21.4857 |   3.64882  |   0.860897 | 78.0336 |  0.0727674  |
|               6 |    5439 | 22.8645 |   0.895993 |   0.82938  | 77.2471 |  0.10057    |
|               7 |    5439 | 24.6682 |  -1.33929  |   0.794264 | 76.5563 |  0.129987   |
|               8 |    5435 | 25.6562 |  -3.87078  |   0.778473 | 75.8607 |  0.150874   |
|               9 |    5431 | 27.1296 |  -6.05173  |   0.748481 | 75.4115 |  0.176579   |
|              10 |    5427 | 28.0682 |  -9.50795  |   0.733554 | 74.2945 |  0.201216   |
|              11 |    5422 | 29.3197 | -12.5911   |   0.707119 | 73.449  |  0.234231   |

### 11.2 Interpretation

Hyperglycemia does not collapse the same way. Instead, the regime transitions:

- early horizons: still positively biased, very high coverage
- later horizons: bias drifts downward and upper-tail misses rise

At horizon 11 in `gt_180`:

- bias = **-12.59**
- coverage = **70.7%**
- above q90 = **23.4%**

Interpretation: the model seems to start by overpredicting in high glucose as well, but as horizon extends it begins to **underpredict some high-glucose tails**, which is different from the low-glucose failure mode. This suggests the model is regressing trajectories toward a central prior and losing dynamic range in both directions, but because that prior is already elevated, the low-glucose regime suffers most.

---

## 12. Row-Level Behavioral Diagnosis: The Model Appears to Lean Too Heavily on the Last Observed Level and Then Drift Upward

The prediction table allows a deeper behavioral question:

> How does the predicted change from the last observation compare to the actual future change?

From `prediction_table.csv`:

- mean actual change from last observation to target: **-0.07 mg/dL**
- mean predicted change from last observation to median forecast: **12.91 mg/dL**
- std of actual change: **24.90**
- std of predicted change: **15.09**
- correlation between actual change and predicted change: **0.367**
- regression slope of predicted change on actual change: **0.223**

### 12.1 Interpretation

This is a critical forensic result.

On average, the actual future glucose relative to the last observed value is essentially flat:

- mean actual change ≈ **-0.07**

But the model’s median forecast moves upward by about:

- mean predicted change ≈ **12.91**

That is an enormous discrepancy. It means the model is not merely making noisy forecasts around the current state. It is systematically injecting an **upward displacement** into the median forecast relative to the last observed glucose.

The low slope (**0.223**) and modest change-correlation (**0.367**) also indicate that the model’s forecasted movement is only weakly responsive to the actual future movement. A perfect directional-change model would not necessarily have slope 1.0 here due to forecasting difficulty, but a slope of about **0.22** indicates significant under-reaction to true change magnitude.

### 12.2 Interpretation in plain terms

The model appears to do something like this:

1. start from the current glucose regime,
2. preserve much of that level information,
3. bias the whole forecast path upward,
4. widen the interval only slightly as horizon increases.

That combination explains:

- the strong positive residual median,
- the dominant below-q10 misses,
- the catastrophic hypoglycemia horizon collapse,
- the relative success in high-glucose coverage where intervals are wider.

---

## 13. What Each Artifact Is Good For, and What It Cannot Tell You Alone

### 13.1 `metrics_summary.json`

**Good for:** first-pass triage.  
**Not enough for:** root-cause analysis.

It tells you the run is biased and undercovered, but not where or why.

### 13.2 `by_horizon.csv`

**Good for:** diagnosing temporal degradation and uncertainty growth.  
**Key finding here:** error more than doubles across the one-hour horizon while interval width barely changes.

### 13.3 `by_subject.csv`

**Good for:** identifying heterogeneity and difficult participants.  
**Caution:** tiny-count subjects can look extreme; always inspect `count` before interpreting MAE.

### 13.4 `by_glucose_range.csv`

**Good for:** clinical-risk stratification.  
**Key finding here:** the low-glucose regime is dramatically worse than the other two.

### 13.5 `prediction_table.csv`

**Good for:** real forensics.  
This is where you confirm:

- sign asymmetry,
- dynamic drift,
- horizon failure,
- range collapse.

### 13.6 `telemetry.csv`

**Good for:** ruling out obvious runtime instability.

Summary of telemetry:

- mean CPU percent: **61.99**
- mean RAM percent: **95.27**
- mean RAM used GB: **6.16**
- mean GPU allocated MB: **79.29**
- mean GPU reserved MB: **4484.01**
- mean reported GPU utilization percent: **0.00**

The telemetry stream does not reveal an obvious unstable-memory signature that would explain the forecasting failure. The GPU utilization column is reported as zeros in this artifact, which is likely a metric-collection limitation on the Apple/MPS path rather than evidence of no acceleration.

### 13.7 Profiler outputs

The profiler summaries are dominated by the cumulative training loop accounting. The fit profiler shows total wall-time heavily concentrated in the training epoch and training batch actions. Because the profiler outputs appear cumulative across stages, they are more useful here as execution records than as sharp diagnostic evidence for forecast quality.

---

## 14. A Reading Order for Future Investigations

When diagnosing future runs, the most disciplined reading order is:

1. **`run_summary.json`**  
   Verify model, horizon length, quantiles, optimizer, checkpoint source, device.

2. **`metrics_summary.json`**  
   Ask: Is there global bias? Is coverage on target? Are intervals degenerate?

3. **`by_horizon.csv`**  
   Ask: Does uncertainty grow with horizon enough to match error growth?

4. **`by_glucose_range.csv`**  
   Ask: Is the model failing exactly where clinical risk is highest?

5. **`by_subject.csv`**  
   Ask: Are failures global or clustered in particular participants?

6. **`prediction_table.csv`**  
   Ask: What is the actual row-level mechanism? Upward drift? lag? collapse in a specific region?

7. **Telemetry and profiler**  
   Ask: Could resource behavior have corrupted the run?

This order prevents the common mistake of overinterpreting a single global metric.

---

## 15. Consolidated Forensic Findings for This Specific Run

### 15.1 Primary finding

The run is dominated by **systematic upward bias**.

Evidence:

- global bias = **+12.98 mg/dL**
- prediction mean exceeds target mean by **12.98 mg/dL**
- median residual = **+11.66 mg/dL**
- overprediction on **79.7%** of rows

### 15.2 Secondary finding

The uncertainty intervals are **not collapsed**, but they are **miscalibrated**.

Evidence:

- near-zero interval fraction = **0.0**
- global coverage = **76.1%** vs nominal **80%**
- interval width increases only modestly with horizon despite major error growth

### 15.3 Third finding

The calibration failure is **strongly asymmetric**, not neutral.

Evidence:

- below q10 = **20.4%**
- above q90 = **3.5%**

Interpretation: the predictive distribution is shifted too high.

### 15.4 Fourth finding

The most severe failure is the **hypoglycemia, long-horizon regime**.

Evidence:

- in `lt_70`, overall coverage = **20.2%**
- from horizons 6–11 in `lt_70`, coverage = **0%**
- from horizons 6–11 in `lt_70`, below q10 = **100%**

### 15.5 Fifth finding

The model’s dynamic response appears too weak and too upward-shifted.

Evidence:

- mean actual future change from last observation ≈ **-0.07**
- mean predicted future change ≈ **12.91**
- slope of predicted change on actual change ≈ **0.223**

Interpretation: the model is not centering its median path on the empirical near-flat average future; it is drifting upward.

### 15.6 Sixth finding

Subject-level heterogeneity exists, but some extreme subject rows are tiny-sample and should be interpreted cautiously.

Evidence:

- `Subject 5` has MAE **40.91**, but only **96** rows
- several other large-count subjects also show elevated MAE and bias, so subject heterogeneity is real even after discounting tiny-sample extremes

---

## 16. Practical Interpretation: What the Artifacts Are Telling You About the Model

Putting all layers together, the artifact stack suggests the following model behavior:

1. The fused model learns a reasonably structured forecast surface, not random noise.
2. It preserves level information from the recent past strongly.
3. It emits non-degenerate quantile intervals.
4. But the center of the predictive distribution is shifted upward.
5. As forecast horizon increases, the model’s median path drifts further away from the truth.
6. The uncertainty head does not inflate the interval quickly enough to compensate.
7. This becomes catastrophic in the low-glucose regime, where even the lower quantile becomes too high.

This is not merely “the model needs better MAE.” It is a much more specific diagnosis:

> The current system appears to be a biased, insufficiently downside-sensitive probabilistic forecaster whose uncertainty growth is too weak relative to forecast-distance difficulty.

---

## 17. Where to Look Next, If You Want to Validate This Diagnosis Further

The present artifacts already support the diagnosis strongly, but the most direct next checks would be:

1. **Open `forecast_overview.html`**  
   Visually confirm the upward displacement of the median path relative to the realized glucose trace.

2. **Open `horizon_bias.html` and `horizon_coverage.html`**  
   These should visually show the monotonic increase in positive bias and monotonic coverage collapse already seen in `by_horizon.csv`.

3. **Filter `prediction_table.csv` for `target < 70` and `horizon_index >= 6`**  
   This will directly expose the rows underlying the 0% coverage result.

4. **Inspect training/validation curves in TensorBoard**  
   This helps determine whether the upward drift emerged from undertraining, calibration imbalance, or a more structural inductive bias issue.

---

## 18. Closing Assessment

This run is scientifically interpretable and diagnostically rich because the reporting pipeline records the right layers of evidence. The key conclusion is not simply that the model has non-trivial error. The key conclusion is more precise:

- the model is **upward biased**,
- the probabilistic intervals are **shifted and under-responsive**,
- the failure grows with **forecast horizon**,
- and the most severe breakdown occurs in the **low-glucose regime**, especially beyond roughly **35 minutes ahead**.

That is the forensic reading of the uploaded artifacts.

---

## Appendix A. Key file-to-question map

| Question | File(s) |
|---|---|
| What was the run configuration? | `run_summary.json`, `run.log` |
| How many rows / subjects / windows were in the dataset? | `reports/data_summary.json` |
| What are the headline metrics? | `metrics_summary.json`, `scalars.json` |
| How does performance change with horizon? | `by_horizon.csv`, `horizon_metrics.html`, `horizon_bias.html`, `horizon_coverage.html` |
| Which subjects are difficult? | `by_subject.csv`, `subject_metrics.html` |
| Which glucose regimes are dangerous? | `by_glucose_range.csv`, `glucose_range_metrics.html` |
| What happened on individual forecast rows? | `prediction_table.csv`, `test_predictions.csv` |
| Was the run systemically unstable? | `telemetry.csv`, profiler files, TensorBoard event files |

## Appendix B. Important cautions for interpretation

1. **Do not interpret global MAE without reading the range table.**  
   The low-glucose regime is much worse than the global average suggests.

2. **Do not interpret subject rankings without checking row counts.**  
   Tiny-sample subjects can be extreme.

3. **Do not interpret interval width without checking coverage and miss asymmetry.**  
   Width by itself is not calibration.

4. **Do not treat undercoverage as symmetric unless you inspect q10 vs q90 misses.**  
   In this run the misses are overwhelmingly below q10, which changes the diagnosis completely.


---

## 19. Artifact Navigation Layer (New in Current Pipeline)

A new addition to the pipeline is:

- `report_index.json`

This file acts as a **central artifact registry**. Its purpose is to eliminate ambiguity when navigating the artifact tree.

It provides:
- Direct paths to core summaries (`run_summary.json`, `metrics_summary.json`)
- Links to grouped tables
- Links to prediction exports
- Links to analysis outputs

Interpretation:
This file upgrades the workflow from *file dumping* to **structured artifact indexing**, which is critical for reproducibility and report generation.

---

## 20. Run Health Summary Layer

New artifact:

- `run_health_summary.json`

This file synthesizes multiple signals into a compact diagnostic layer.

It includes:
- Horizon degradation metrics
- Coverage warnings
- Bias summaries
- Threshold-based performance indicators

Interpretation:
This acts as a **machine-readable diagnosis layer**, allowing automated checks such as:
- “Is coverage collapsing?”
- “Is bias increasing with horizon?”

---

## 21. Threshold-Based Accuracy Analysis

Located under:
- `analysis_outputs/threshold_accuracy_summary.csv`
- `analysis_outputs/threshold_accuracy_by_horizon.csv`
- `analysis_outputs/threshold_accuracy_by_glucose_range.csv`

This layer answers:

> “How often are predictions clinically close enough?”

This complements MAE/RMSE by providing:
- % within 10 mg/dL
- % within 20 mg/dL
- % within 30 mg/dL

Interpretation:
This is a **decision-oriented metric layer**, much closer to real-world usability than average error.

---

## 22. Event-Aware Analysis Layer

Located under:
- `analysis_outputs/metrics_by_meal_event.csv`
- `analysis_outputs/metrics_by_insulin_event.csv`
- `analysis_outputs/metrics_by_device_mode.csv`
- `analysis_outputs/predictions_with_event_context.csv`

This layer introduces **contextual evaluation**.

It answers:
- Are meal windows harder?
- Are insulin events harder?
- Does device mode affect performance?

Interpretation:
This transforms the model evaluation from:
- static accuracy → **context-aware performance understanding**

---

## 23. Baseline Comparison Layer

New artifacts include:
- `baseline_persistence_metrics.json`
- `baseline_vs_model_summary.csv`

Purpose:
To evaluate whether the model outperforms a simple baseline (e.g., persistence).

Interpretation:
This prevents misleading conclusions where:
- model appears “good”
- but performs similarly to trivial baselines

---

## 24. Updated Interpretation of This Run

After incorporating all artifact layers, the refined interpretation is:

1. The model is **systematically upward biased**.
2. The probabilistic intervals are **not degenerate**, but are **miscalibrated and asymmetric**.
3. Error increases sharply with horizon, but **uncertainty does not scale accordingly**.
4. The **low-glucose regime remains the dominant failure mode**, especially at longer horizons.
5. The pipeline now provides **sufficient observability to diagnose these failures precisely**.

---

## 25. Final Assessment (Patched)

This run is not merely a model output; it is a **fully observable experiment**.

The artifact ecosystem now supports:
- forensic debugging
- structured reporting
- clinical interpretation
- reproducible evaluation

The primary limitation is no longer lack of visibility.

It is now **model behavior itself**, particularly:
- upward bias
- weak downside sensitivity
- insufficient uncertainty expansion

