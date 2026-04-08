# An Exposition of the Data Contract, Fused Forecasting Model, and Training Procedure

Role: Preserved full-length methods monograph for the data-to-model-to-training
pipeline.
Audience: Researchers and advanced contributors studying implementation and
method choices in depth.
Owns: Detailed methodological exposition, tensor semantics, and optimization
reasoning.
Related docs: [`../../README.md`](../../README.md),
[`../system_walkthrough.md`](../system_walkthrough.md),
[`materials_and_methods.md`](materials_and_methods.md),
[`../current_architecture.md`](../current_architecture.md),
[`../repository_primer.md`](../repository_primer.md),
[`probabilistic_forecasting.md`](probabilistic_forecasting.md).

This document is intentionally not the first deep read after the README. Start
with [`../system_walkthrough.md`](../system_walkthrough.md) if you want the
broad system story first, or [`materials_and_methods.md`](materials_and_methods.md)
if you want the shorter paper-oriented methods path. Use this file when you
want the full monograph-length methods treatment.

## Abstract

This document is a self-contained academic exposition of the glucose forecasting pipeline implemented in the `danCoder93/ENSF617-Probabilistic-Glucose-Forecasting` repository. It aims to answer the practical questions that matter most when one first encounters a modern research codebase:

- What data are entering the pipeline?
- How are the raw tables converted into model-ready tensors?
- What does each tensor mean physically and mathematically?
- How does the fused TCN + TFT model transform those tensors stage by stage?
- Why are those transformations there?
- What objective is the training loop trying to optimize?
- What assumptions, compromises, and current limitations are embedded in the present implementation?

The codebase is not merely a generic Temporal Fusion Transformer implementation. It is a repository-specific hybrid architecture in which a Temporal Convolutional Network (TCN) branch and a Temporal Fusion Transformer (TFT) branch are both built over a shared semantic input contract and are joined by a late-fusion stage before final quantile prediction. In practical terms, the system attempts to forecast future glucose values while also estimating predictive uncertainty through quantiles such as the 10th, 50th, and 90th percentiles. The repository therefore sits at the intersection of three ideas:

1. **physiologically informed multimodal glucose forecasting**, because the inputs include glucose, insulin, carbohydrate, and device-state information;
2. **multi-horizon sequence modeling**, because the model sees a historical window and predicts a future trajectory;
3. **probabilistic forecasting**, because the output is a forecast distribution summary rather than only a single point value.

This document reconstructs the conceptual algorithm that the code implements, maps major theoretical ideas to concrete files, and identifies where the implementation is elegant, where it is transitional, and where it could be improved.

---

## Table of Contents

1. [Purpose of This Document](#1-purpose-of-this-document)
2. [Scientific Objective](#2-scientific-objective)
3. [Dataset Provenance and Research Context](#3-dataset-provenance-and-research-context)
4. [The Foundational Forecasting Question](#4-the-foundational-forecasting-question)
5. [The Current Experimental Contract](#5-the-current-experimental-contract)
6. [Raw Data to Canonical Table](#6-raw-data-to-canonical-table)
7. [Canonical Columns and Their Meaning](#7-canonical-columns-and-their-meaning)
8. [Cleaning and Table-Wide Normalization](#8-cleaning-and-table-wide-normalization)
9. [Why the Preprocessing Choices Matter](#9-why-the-preprocessing-choices-matter)
10. [Feature Semantics: Static, Known, Observed, and Target Variables](#10-feature-semantics-static-known-observed-and-target-variables)
11. [The Semantic Feature Contract in the Current Branch](#11-the-semantic-feature-contract-in-the-current-branch)
12. [Sliding Windows and Sequence Construction](#12-sliding-windows-and-sequence-construction)
13. [Split Policy and Leakage Considerations](#13-split-policy-and-leakage-considerations)
14. [The Exact Batch Contract Emitted by the Dataset](#14-the-exact-batch-contract-emitted-by-the-dataset)
15. [Tensor Shapes: A Complete Reading Guide](#15-tensor-shapes-a-complete-reading-guide)
16. [What Each Tensor Means on the Original Patient Timeline](#16-what-each-tensor-means-on-the-original-patient-timeline)
17. [The Top-Level Configuration Story](#17-the-top-level-configuration-story)
18. [Why the DataModule Exists](#18-why-the-datamodule-exists)
19. [Why the Dataset Exists Separately](#18-why-the-datamodule-exists)
20. [Why Indexing Exists Separately](#20-why-indexing-exists-separately)
21. [The Fused Model at a Glance](#21-the-fused-model-at-a-glance)
22. [Why Fuse TCN and TFT at All?](#22-why-fuse-tcn-and-tft-at-all)
23. [The Temporal Convolutional Network Branch](#23-the-temporal-convolutional-network-branch)
24. [The Temporal Fusion Transformer Branch](#24-the-temporal-fusion-transformer-branch)
25. [Gated Residual Networks in This Repository](#25-gated-residual-networks-in-this-repository)
26. [The Final Quantile Head](#26-the-final-quantile-head)
27. [The Forward Pass, Step by Step](#27-the-forward-pass-step-by-step)
28. [Mathematical Formulation of the Main Computation](#28-mathematical-formulation-of-the-main-computation)
29. [Probabilistic Supervision and Pinball Loss](#29-probabilistic-supervision-and-pinball-loss)
30. [Point Forecast Extraction for Human-Readable Metrics](#30-point-forecast-extraction-for-human-readable-metrics)
31. [What Training Is Actually Optimizing](#31-what-training-is-actually-optimizing)
32. [What the Lightning Trainer Does for the Repository](#32-what-the-lightning-trainer-does-for-the-repository)
33. [Checkpointing, Early Stopping, and Run Policy](#33-checkpointing-early-stopping-and-run-policy)
34. [Why the Repo Uses Runtime-Bound Configuration](#34-why-the-repo-uses-runtime-bound-configuration)
35. [Shape Walkthrough with a Concrete Example](#35-shape-walkthrough-with-a-concrete-example)
36. [Strengths of the Current Design](#36-strengths-of-the-current-design)
37. [Current Transitional Design Choices](#37-current-transitional-design-choices)
38. [Suboptimal or Risky Aspects and How They Could Improve](#38-suboptimal-or-risky-aspects-and-how-they-could-improve)
39. [Interpretation of the Forecast Outputs](#39-interpretation-of-the-forecast-outputs)
40. [What the Repository Is Trying to Achieve Scientifically](#40-what-the-repository-is-trying-to-achieve-scientifically)
41. [A Code-to-Algorithm Crosswalk](#41-a-code-to-algorithm-crosswalk)
42. [Appendix A: Quick Tensor Cheat Sheet](#42-appendix-a-quick-tensor-cheat-sheet)
43. [Appendix B: Important Equations](#43-appendix-b-important-equations)
44. [Appendix C: Suggested Future Improvements](#44-appendix-c-suggested-future-improvements)
45. [References and Provenance Notes](#45-references-and-provenance-notes)

---

## 1. Purpose of This Document

The aim of this document is to provide an intellectual map of the implementation. In this project, that whole is a pipeline that begins with raw diabetes-management data and ends with future glucose quantiles.

---

## 2. Scientific Objective

At the repository level, the branch is implementing a **probabilistic, multi-horizon blood glucose forecasting system** built around a late-fused TCN + TFT architecture. The repository README explicitly frames the problem as forecasting future blood glucose values while supporting uncertainty-aware prediction instead of only point forecasting, and as combining local temporal pattern extraction with longer-range sequence reasoning. The same README also states that the current codebase is organized around a hybrid TCN + TFT architecture and an artifact-rich training/evaluation workflow.

Conceptually, the repository is trying to solve the following scientific problem:

Given a historical sequence of glucose and related covariates for a subject, estimate the future glucose trajectory over a prediction horizon, while preserving enough information about uncertainty that the system can say not only “the future glucose may be 140 mg/dL,” but also “a lower-plausible outcome is around \(q_{0.1}\), the central tendency is around \(q_{0.5}\), and an upper-plausible outcome is around \(q_{0.9}\).”

This is a much more meaningful research target than single-point next-step regression, because glucose dynamics are inherently uncertain. Meals may be partially logged, correction boluses may have delayed effects, device modes may change, and human behavior is never perfectly measured. A probabilistic output therefore fits the problem better than a single deterministic scalar.

---

## 3. Dataset Provenance and Research Context

The current data path is built around the **AZT1D** dataset. The `DataConfig` default URL points to the public Mendeley dataset release, and the README plus data-layer docstrings treat AZT1D as the current main supported dataset. The AZT1D release describes itself as a real-world dataset for type 1 diabetes containing data from 25 individuals using automated insulin delivery systems, including glucose values, insulin data, carbohydrate intake, and device-mode information over roughly 6 to 8 weeks per patient. The dataset documentation and related paper description also indicate that glucose records are reported on a 5-minute basis, which is consistent with the repository’s default `sampling_interval_minutes=5`.

From a research perspective, this matters for two reasons.

First, many glucose forecasting datasets are narrower than they initially appear. Some contain CGM and meal information but not rich insulin detail. Some contain insulin but not device-state information. The AZT1D dataset is richer than many older public datasets because it includes not only glucose and carbohydrate data, but also basal insulin, total bolus insulin, correction-specific insulin, meal-related insulin, device mode, and bolus type.

Second, the dataset is clearly real-world and therefore messy. Real-world medical time series are not created for machine learning convenience. They contain duplicated timestamps, inconsistent categorical strings, state variables that persist between documented events, and event variables that should not be propagated forward. The repository’s preprocessing pipeline is best understood as an attempt to impose a coherent machine-learning contract on that messy reality without losing the clinical semantics of the underlying variables.

---

## 4. The Foundational Forecasting Question

The forecasting question embedded can be written as follows.

Let a subject’s cleaned timeline be a multivariate sequence
\[
x_1, x_2, \dots, x_T,
\]
where each \(x_t\) contains:

- glucose information,
- insulin-related signals,
- carbohydrate intake,
- time-derived covariates,
- device-state covariates,
- and subject identity information.

Given an encoder history of length \(L_e\), the model seeks to predict the next \(L_d\) target values:
\[
y_{t+1}, y_{t+2}, \dots, y_{t+L_d},
\]
where \(y_t\) is glucose in mg/dL.

In the repository defaults:

- encoder length \(L_e = 168\),
- prediction length \(L_d = 12\),
- sampling interval = 5 minutes.

Thus one default sample corresponds to:

- **168 historical steps** = 168 × 5 minutes = 840 minutes = 14 hours,
- **12 forecast steps** = 12 × 5 minutes = 60 minutes.

So, in the default configuration, the model uses **14 hours of history** to predict **the next 1 hour of glucose**.

This default is a design choice, not a dataset-mandated fact. The code explicit about this distinction.

---

## 5. The Current Experimental Contract

The current experimental contract can be summarized as follows.

### 5.1 Data contract

A cleaned dataframe is converted into semantically grouped tensors rather than a single undifferentiated feature matrix.

### 5.2 Modeling contract

The model is a hybrid:

- three TCN branches with different kernel sizes,
- one TFT branch,
- one late-fusion GRN,
- one final neural head that emits quantiles.

### 5.3 Supervision contract

Training is probabilistic and uses **pinball loss** over configured quantiles.

### 5.4 Evaluation contract

Human-readable point metrics such as MAE and RMSE are still reported by extracting a representative point forecast, typically the median quantile when available or closer to it.

### 5.5 Runtime contract

PyTorch Lightning owns the outer epoch loop, checkpointing, optional early stopping, and device/runtime mechanics, while the repository’s own trainer wrapper determines how the model, data module, config, and observability tools are assembled.

These contracts are important because they establish a layered architecture rather than a monolithic script. **That layered design is one of the repository’s strongest engineering features**.

---

## 6. Raw Data to Canonical Table

The preprocessing pipeline begins with a very practical task: taking raw AZT1D files and rewriting them into one canonical processed CSV.

The `AZT1DPreprocessor` does **not** split data into train/validation/test. It does **not** build tensors. It does **not** create windows. Its job is more modest and more foundational:

> Convert vendor-shaped raw exports into one stable processed table with canonical names.

The canonical output columns are:

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

This is scientifically wise. In a mature pipeline, raw-file normalization should happen once and early. Otherwise every downstream stage becomes polluted with raw-format contingency logic.

The preprocessor maps raw columns such as `EventDateTime`, `Basal`, `TotalBolusInsulinDelivered`, `CorrectionDelivered`, `FoodDelivered`, and `CarbSize` into these canonical internal names. It also tolerates multiple raw glucose column spellings, accepting either `CGM` or `Readings (CGM / BGM)` as the glucose source column.

### Why this matters

Without canonicalization, every later module would have to ask questions like:

- “Is glucose stored as `CGM` or as `Readings (CGM / BGM)`?”
- “Is the insulin field called `FoodDelivered` or something else?”
- “Which directories identify the subject?”

That would be disastrous for maintainability. Canonicalization makes all downstream code speak one internal vocabulary.

---

## 7. Canonical Columns and Their Meaning

The cleaned table can be understood as a harmonized patient timeline in which each row corresponds to a standardized time point.

### 7.1 `subject_id`

This identifies the patient. In the raw archive, subject identity is inferred from folder structure. Once extracted, it becomes just another canonical column.

### 7.2 `timestamp`

This is the row’s temporal anchor. It is later converted to datetime, rounded to the nearest minute, sorted, and used for continuity checks.

### 7.3 `glucose_mg_dl`

This is the supervised target variable: glucose concentration in milligrams per deciliter, obtained from **continuous glucose monitoring (CGM) systems**.

In clinical and modeling contexts, CGM glucose is not merely a scalar measurement but a **high-frequency physiological time series sampled approximately every 5 minutes**, obtained from sensors placed in the **subcutaneous tissue**. These sensors measure glucose in the **interstitial fluid**, the fluid surrounding cells, rather than directly within the bloodstream.

Physiologically, glucose transitions from the blood (capillaries) into the interstitial space through diffusion processes. As a result, CGM measurements represent a **filtered and temporally delayed proxy of blood glucose**, rather than an instantaneous observation. This introduces inherent **lag dynamics—typically on the order of 5 to 15 minutes**, which become especially pronounced during periods of rapid glycemic change, such as postprandial excursions or insulin-induced declines.

Consequently, the observed glucose signal reflects not only the current metabolic state but also the **underlying transport dynamics between physiological compartments (blood → interstitial space)**. From a modeling perspective, the target variable is therefore a **lagged and smoothed representation of the underlying physiological system**, requiring temporal models to capture delayed responses and history-dependent effects when learning glucose trajectories.

From a modeling perspective, this variable:

- Encodes **glycemic state trajectories**, including trends, variability, and excursions  
- Serves as the primary signal for identifying **glycemic regimes**, i.e., temporally sustained states such as **hyperglycemia** (>180 mg/dL) and **hypoglycemia** (<70 mg/dL), as well as the transitions and dynamics between these states
- Acts as the core observable through which AID systems **close the loop between sensing and insulin delivery**

The AZT1D dataset explicitly frames CGM as the central signal for **trend analysis, variability metrics, and downstream prediction tasks**.

### 7.4 `basal_insulin_u`

This represents basal insulin in units, corresponding to **continuous background insulin delivery** administered by an insulin pump.

Clinically, basal insulin is designed to:

- Maintain glucose stability in **fasting conditions**  
- Counteract endogenous glucose production (e.g., hepatic glucose output)

This implies that basal insulin offsets glucose that the body **produces internally**, even in the absence of food intake.

Physiologically, the liver continuously releases glucose into the bloodstream through processes such as:

- **Glycogenolysis** (breakdown of stored glycogen)  
- **Gluconeogenesis** (synthesis of glucose from non-carbohydrate substrates)  

This endogenous glucose production is essential for maintaining energy supply during fasting states (e.g., overnight or between meals). However, in individuals with type 1 diabetes, there is insufficient endogenous insulin to regulate this process effectively.

Basal insulin therefore acts as a **background regulatory signal** that suppresses hepatic glucose output, helping to:

- Maintain stable glucose levels during fasting periods  
- Prevent gradual upward drift in glucose when no meals are present  

Without adequate basal insulin, glucose levels would increase even in the absence of food intake due to unregulated hepatic glucose release.

In the dataset:

- Basal insulin is recorded **hourly** and then repeated across the corresponding **5-minute CGM intervals** until the next hourly value becomes active, ensuring alignment with the glucose time series  
- This results in a **piecewise constant signal**, where each basal value persists across multiple timesteps, reflecting its continuous physiological effect rather than discrete events  

From a modeling standpoint:

- It behaves as a **state variable (low-frequency, continuous effect)** rather than an event-driven signal  
- It contributes to **long-horizon glucose dynamics**, influencing slow trends rather than rapid fluctuations  
- It is essential for capturing **baseline glucose drift and stability**, particularly during periods without meals or bolus insulin events  

### 7.5 `bolus_insulin_u`

This represents total bolus insulin in units, corresponding to **discrete insulin delivery events**, i.e., insulin delivered at specific points in time rather than continuously.

In practice, these events occur when:

- A person administers insulin to **cover a meal (carbohydrate intake)**  
- A person delivers a **correction dose** to reduce elevated glucose levels  

In insulin pump systems (as in this dataset), these are triggered either:

- Manually by the user  
- Automatically by the AID system  

Unlike basal insulin, bolus insulin is:

- **Event-triggered**
- Typically administered in response to **meals or high glucose levels**

The dataset highlights that bolus insulin includes:

- Total delivered dose  
- Associated type (standard, correction, automatic)  

In the aligned time series:

- Bolus insulin is represented at the **5-minute resolution of the CGM signal**  
- Most timesteps contain **zero values**, with non-zero values appearing only at the time of delivery  
- Each bolus event is therefore encoded as a **localized spike at a specific timestep (or short window)**  

From a modeling perspective:

- It is a **sparse, impulse-like control signal**, meaning it is mostly zero with occasional sharp spikes corresponding to insulin delivery events  
- These impulses act as **external interventions** whose effects unfold over future timesteps rather than instantaneously  
- It introduces **nonlinear, delayed glucose responses**, reflecting the pharmacodynamics of insulin absorption and action  
- It is one of the primary drivers of **rapid glucose decreases**, particularly following meal coverage or correction dosing  

### 7.6 `correction_insulin_u`

This isolates the **correction-specific component of bolus insulin**, i.e., insulin delivered explicitly to reduce elevated glucose levels.

This distinction is clinically critical because:

- Correction insulin is **reactive**, not anticipatory  
- It encodes the system’s response to **hyperglycemia events**

This means that correction insulin is administered **in response to an observed high glucose level**, rather than in preparation for a future disturbance. In practice:

- It is given *after* glucose has already risen above target  
- It functions as a **feedback control action**, aiming to bring glucose back into range  

In contrast:

- Meal insulin is typically **anticipatory**, given before or during carbohydrate intake to prevent glucose excursions  

In the dataset, this is defined as:

- The portion of the bolus insulin administered to correct high blood glucose levels  

From a modeling standpoint:

- It provides **causal interpretability**, helping distinguish whether insulin acts as a **cause of future glucose changes** or a **response to current glucose levels**  
- It enables separation between:
  - **Drivers of glucose change** (e.g., meals and meal-related insulin)  
  - **Responses to glucose change** (e.g., correction insulin)  

This distinction is important because these two situations follow different dynamics:

- **Meal-driven dynamics (planned)**:  
  carbs → glucose rises → insulin counters it  

- **Feedback-control dynamics (reactive)**:  
  glucose rises → correction insulin → glucose falls  

By separating these, the model can better learn:

- How glucose evolves due to **external inputs (meals)**  
- How the system behaves under **feedback control actions (corrections)**  

Dataset Handling and Representation:

`correction_insulin_u` is handled as a **time-aligned event feature** within the canonical table.

At the preprocessing stage:

- The raw correction-specific bolus value is extracted from the source field corresponding to correction delivery  
- Its timestamp is aligned to the common **5-minute CGM grid**  
- If no correction event exists at a given timestep, the value is set to **0**  

As a result, the column behaves as:

- A **mostly-zero sequence**  
- With occasional **non-zero spikes** at timesteps where a correction dose was delivered  

For each training sample:

- The dataset slices a contiguous input window from the canonical table  
- `correction_insulin_u` appears as one feature channel across all timesteps in that window  
- Most samples therefore contain all zeros in that channel  
- Some samples contain one or more localized spikes marking correction events  

Conceptually, the model does not receive correction insulin as a persistent background state. Instead, it is represented as:

- A **sparse event signal**  
- Tied to specific timestamps  
- Whose effect must be learned over subsequent future timesteps  

### 7.7 `meal_insulin_u`

This represents the **meal-specific component of bolus insulin**, i.e., insulin delivered to cover carbohydrate intake.

In the dataset, this corresponds to:

- The portion of the bolus insulin dedicated to covering meal carbohydrate intake  

Clinically:

- Meal insulin is **anticipatory**, meaning it is administered *before or during* carbohydrate intake  
- It is calculated based on:
  - The amount of carbohydrates consumed  
  - A patient-specific **insulin-to-carbohydrate ratio**  

This implies that meal insulin is intended to **preemptively counteract the expected rise in glucose** following food intake, rather than responding to an already elevated glucose level.

In contrast:

- Correction insulin is **reactive**, administered after glucose has already risen  

From a modeling standpoint:

- It acts as a **feedforward control signal**, representing a planned intervention based on expected future disturbance  
- It is tightly coupled with `carbs_g`, forming a structured relationship: carbs → glucose rises → meal insulin counters it
- It represents a **primary driver of glucose dynamics**, rather than a response to them  
- It introduces **structured, predictable glucose responses**, particularly postprandial (after-meal) glucose curves  

Dataset Handling and Representation

`meal_insulin_u` is handled as a **time-aligned event feature** within the canonical table.

At the preprocessing stage:

- The meal-specific portion of bolus insulin is extracted from the raw bolus logs  
- Its timestamp is aligned to the common **5-minute CGM grid**  
- If no meal-related insulin event exists at a given timestep, the value is set to **0**  

As a result, the column behaves as:

- A **mostly-zero sequence**  
- With **non-zero spikes** at timesteps corresponding to meal-related insulin delivery  

For each training sample:

- The dataset slices a contiguous input window from the canonical table  
- `meal_insulin_u` appears as one feature channel across all timesteps in that window  
- Most timesteps contain zeros, with occasional spikes aligned to meal events  

Conceptually, this means the model receives meal insulin as:

- A **sparse, event-driven control signal**  
- Strongly correlated with `carbs_g`  
- Whose effect unfolds over future timesteps as part of post-meal glucose dynamics  


### 7.8 `carbs_g`

This is carbohydrate intake in grams, representing **exogenous glucose input into the system**.

In diabetes physiology:

- Carbohydrates are the **primary driver of postprandial glucose excursions**
- Their effect is:
  - Delayed (digestion + absorption)
  - Nonlinear (depends on glycemic index, metabolism)

The dataset defines this as:

- The amount of carbohydrates consumed, measured in grams

From a modeling standpoint:

- This is one of the most **causally dominant exogenous variables**
- It drives:
  - Rapid glucose increases
  - Interaction effects with insulin timing

Importantly:

- Missing carb entries are filled with **0 when no event is recorded**, reinforcing its **event-based sparsity**

### 7.9 `device_mode`

This records the operating mode of the insulin delivery system, typically:

- `regular`
- `sleep`
- `exercise`

These modes reflect **contextual physiological and control regimes**, not just metadata.

In the dataset:

- Device mode is extracted from source data (e.g., pump logs or PDFs) and aligned with the common time series  
- It represents the **state of the AID system at each timestamp**  
- Since device mode is not recorded at every 5-minute interval, it is **propagated forward in time** until a new mode change occurs, ensuring full alignment with the CGM grid  

As a result:

- The signal behaves as a **piecewise constant categorical variable**  
- Each timestep has a valid device mode, even if no explicit change occurred at that moment  

Clinically:

- Different modes correspond to:
  - Altered insulin sensitivity  
  - Modified control policies (e.g., more conservative insulin delivery during sleep, or adjustments during exercise)  

From a modeling perspective:

- This is a **contextual regime variable**, not a direct physiological measurement  
- It allows models to learn **conditional dynamics**, meaning:
  - The same inputs (e.g., carbs, insulin) may produce different glucose responses depending on the current device mode  

Dataset Handling and Representation:

`device_mode` is handled as a **time-aligned categorical feature** within the canonical table.

At the preprocessing stage:

- Raw device mode values are extracted and mapped to a **controlled vocabulary** (e.g., `regular`, `sleep`, `exercise`)  
- The mode is aligned to the **5-minute CGM grid**  
- Missing values between mode changes are **forward-filled**, ensuring continuity  

For each training sample:

- The dataset slices a contiguous input window from the canonical table  
- `device_mode` appears as one feature channel across all timesteps in that window  
- Since it is categorical, it is typically:
  - **Encoded numerically** (e.g., integer encoding) or  
  - Transformed via **embeddings** in the model

### 7.10 `bolus_type`

This represents the **categorical type of bolus insulin delivery**, describing the intent and mechanism behind each bolus event.

The dataset defines categories such as:

- `standard` (typically meal-related)  
- `correction` (to reduce high glucose)  
- `automatic` (system-triggered by the AID controller)  

This variable is important because:

- It encodes the **intent behind insulin delivery**, not just the amount  
- It distinguishes between:
  - **Patient-initiated vs system-initiated actions**  
  - **Reactive vs planned interventions**  

In the dataset, bolus type is extracted alongside insulin dose and aligned to the **5-minute CGM grid**. Since bolus events are sparse:

- Most timesteps contain a **default / no-event category**  
- Non-default values appear only at timesteps where a bolus is delivered  

As a result, the signal behaves as a **sparse categorical event feature** with localized spikes in category values.

For each training sample:

- `bolus_type` appears as one feature channel across all timesteps  
- Most entries are the default category, with occasional event-specific labels  

From a modeling standpoint:

- It provides **semantic disambiguation of identical numeric doses**, allowing the model to distinguish *why* insulin was delivered  
- It enables better separation between:
  - **Meal-driven (planned)** vs **correction-driven (reactive)** actions  
- It improves **interpretability of insulin-glucose interactions** by adding context to quantitative insulin signals  

### 7.11 `source_file`

This is not a predictive physiological feature. It is a provenance/debugging column.

---

## 8. Cleaning and Table-Wide Normalization

After raw canonicalization, the branch applies dataframe-level normalization through `load_processed_frame(...)`.

This stage deserves careful attention because it encodes much of the repository’s scientific worldview.

### 8.1 Rows with missing subject, time, or target are dropped

The code explicitly removes rows missing:

- subject identity,
- timestamp,
- or glucose target.

This is logically unavoidable for supervised multi-step forecasting. If any of those are missing, the row cannot be used as part of a legally indexed supervised example.

### 8.2 Timestamps are rounded to the nearest minute

The repository converts timestamps to datetime and rounds them to the minute. This is a small but important harmonization step that reduces minor textual timestamp inconsistencies.

### 8.3 Rows are sorted by subject and time

This creates a deterministic chronological order inside each subject.

### 8.4 Exact duplicates are removed

Duplicate rows can otherwise overweight certain events and confuse continuity checks.

### 8.5 Duplicate timestamps are collapsed

This reflects the fact that real-world medical exports can contain multiple records that refer to the same nominal time point. A windowing algorithm expects a single coherent row per time index, not several contradictory rows.

### 8.6 Basal insulin is forward-filled, backward-filled, and finally zero-filled

This is one of the most conceptually important preprocessing steps.

The code treats basal insulin as a **state variable** on the unified time grid. A missing basal entry at a given time point is therefore interpreted not as “no basal exists,” but rather as “the basal state continues from surrounding context.” The branch first converts the column to numeric, then forward-fills within each subject, then backward-fills leading gaps, then fills any remaining missing values with zero.

This is defensible because basal insulin behaves like an ongoing background delivery rate rather than like a one-time event. Still, it is also a modeling assumption: the code is imposing a piecewise-constant reconstruction of basal insulin.

### 8.7 Event-style quantities are zero-filled

The columns

- `bolus_insulin_u`
- `correction_insulin_u`
- `meal_insulin_u`
- `carbs_g`

are treated as sparse event variables. Once the table is placed on a common 5-minute grid, a missing value in these columns is interpreted as “no event was recorded at this interval,” so missing values become zeros.

This is one of the cleanest preprocessing decisions in the branch. For sparse event quantities, forward-filling would be deeply wrong. If a subject ate 30 grams of carbohydrate at noon, that does not mean they ate 30 grams at 12:05, 12:10, and 12:15.

### 8.8 `device_mode` is treated as persistent state

The code normalizes text, maps certain raw placeholders such as `"0"` to `"regular"`, forward-fills within subject, then fills leading gaps with `"regular"`.

This again reflects a state-like interpretation. Device mode is assumed to persist unless changed.

### 8.9 `bolus_type` is treated as event-local

Unlike `device_mode`, bolus type is *not* forward-filled. Missing values remain tied to “no bolus event here.” This is conceptually correct.

### 8.10 Time features are added

The schema defines default known continuous time-derived features:

- `minute_of_day_sin`
- `minute_of_day_cos`
- `day_of_week_sin`
- `day_of_week_cos`
- `is_weekend`

These are added after base cleaning. Their role is to provide the model with cyclical calendar/time context that is known ahead of time.

### 8.11 Declared categorical vocabularies are normalized

For columns with controlled vocabularies, the branch centralizes category order. This ensures stable integer encoding and stable embedding cardinalities.

### 8.12 Required columns are validated

The transform layer validates that the cleaned dataframe contains the columns the semantic feature grouping expects.

---

## 9. Why the Preprocessing Choices Matter

Preprocessing is not a neutral housekeeping stage. It defines what kind of scientific problem the model is allowed to see.

### 9.1 State variables versus event variables

The branch’s preprocessing makes a principled distinction between **persistent states** and **sparse events**.

- Basal insulin and device mode are persistent states.
- Bolus insulin, correction insulin, meal insulin, and carbohydrates are sparse events.

That distinction is medically sensible and algorithmically essential.

### 9.2 Temporal continuity is taken seriously

The code later refuses to build windows across gaps in the expected 5-minute cadence. This means preprocessing is not pretending the timeline is uniformly sampled when it is not. That is scientifically good practice.

### 9.3 The branch is making imputational assumptions

Even when reasonable, these remain assumptions. For example:

- forward-filling device mode assumes mode persistence;
- forward/backward-filling basal assumes a stable underlying basal state;
- zero-filling events assumes absence of record means absence of event.

Each assumption could be challenged in a future study. What matters here is that the repository’s current assumptions are coherent and explicit in code.

---

## 10. Feature Semantics: Static, Known, Observed, and Target Variables

The repository borrows a core conceptual idea from the TFT literature: not all variables play the same causal-information role.

### 10.1 Static variables

These are time-invariant within a sample. In the current branch, subject identity functions as the main static variable.

### 10.2 Known variables

These are variables whose future values are legitimately known at prediction time. Time-derived calendar features are the clearest example. The future hour’s clock time and day-of-week are known in advance.

### 10.3 Observed variables

These are variables observed historically but not known prospectively. Meal and insulin events belong here under the current contract, as do observed categorical states such as bolus type and device mode.

### 10.4 Target variable

This is the glucose series itself. Historical glucose is available in the encoder. Future glucose is what must be predicted.

This semantic partition is one of the most important intellectual structures in the repository. It prevents future leakage. The model may use future clock features, because those are known. It may not use future meal or future glucose unless the experiment intentionally changes the problem definition.

---

## 11. The Semantic Feature Contract in the Current Branch

The schema comments describe the default feature groups as follows:

- subject identity is static,
- time features are known in advance,
- insulin/carb activity is observed history,
- glucose is the target to be forecast.

In practical terms, the current default grouping is:

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

This is an elegant default contract because it respects the causal availability of information.

At the same time, it is also somewhat conservative. For example, one could argue that certain insulin-delivery schedules or pump-programmed basal patterns might contain known-ahead components in some settings. The current branch avoids making that stronger claim and instead treats insulin-related signals as observed history. That is safe, but it may leave some predictive structure unused.

---

## 12. Sliding Windows and Sequence Construction

Once the cleaned dataframe exists, the repository does not immediately convert everything into tensors. Instead it constructs a list of valid windows.

This is handled by `build_sequence_index(...)`, which produces a list of `SampleIndexEntry` objects. Each entry stores integer row boundaries:

- `encoder_start`
- `encoder_end`
- `decoder_start`
- `decoder_end`

These are half-open Python slicing boundaries over the split dataframe.

### Why use an index instead of immediate tensors?

Because these are different responsibilities:

- **Indexing** answers: “Which windows are legal?”
- **Dataset assembly** answers: “How do I turn one legal window into tensors?”

That separation is excellent design. It makes the pipeline easier to reason about and easier to test.

### Continuity requirement

For a window to be legal, it must lie entirely inside a contiguous segment whose timestep gaps match the expected sampling interval, default 5 minutes.

If the subject timeline contains a discontinuity, the code breaks the timeline into separate segments. No sample may cross that gap.

### Required length

A legal segment must be at least:

\[
L_{ ext{required}} = L_e + L_d
\]

rows long, where \(L_e\) is encoder length and \(L_d\) is decoder/prediction length.

With default settings:

\[
L_{ ext{required}} = 168 + 12 = 180.
\]

Thus a contiguous segment shorter than 180 rows yields no default sample.

### Stride

The repository uses a sliding-window stride parameter `window_stride`, default 1. That means consecutive windows overlap heavily:

- sample 1 may use rows 0–167 to predict 168–179,
- sample 2 may use rows 1–168 to predict 169–180,
- and so on.

This produces many training examples from one long trajectory.

---

## 13. Split Policy and Leakage Considerations

The branch supports several split modes, but the default is:

- `split_by_subject = False`
- `split_within_subject = True`

So the default behavior is to split each subject’s timeline chronologically into train, validation, and test segments.

### Why this is reasonable

For within-subject forecasting, this produces a realistic scenario in which the model sees a subject’s earlier trajectory and is evaluated on that same subject’s later trajectory. That matches many personalized forecasting settings.

### Why it is not leakage-free in the strongest sense

Because the same subject appears in train, validation, and test. If the model learns subject-specific embeddings or patterns, then evaluation is partly measuring within-subject temporal generalization rather than across-subject generalization.

### Alternative split mode

The repository also supports `split_by_subject=True`, which assigns whole subjects to only one split. That is the stronger leakage barrier. It tests generalization to unseen patients.

### Interpretation

Neither policy is universally correct. They answer different scientific questions.

- **Within-subject chronological split** asks:
  “Can the model forecast this patient’s future from this patient’s past?”

- **Subject-held-out split** asks:
  “Can the model generalize to patients it never saw during training?”

The current default is closer to the first question.

---

## 14. The Exact Batch Contract Emitted by the Dataset

For each indexed window, `AZT1DSequenceDataset.__getitem__` returns a dictionary with these keys:

- `static_categorical`
- `static_continuous`
- `encoder_continuous`
- `encoder_categorical`
- `decoder_known_continuous`
- `decoder_known_categorical`
- `target`
- `metadata`

This is the most important interface in the repository.

### 14.1 `static_categorical`

A one-row categorical vector, anchored at the encoder start row. In the default branch this mainly represents subject identity.

### 14.2 `static_continuous`

A one-row continuous vector, also anchored at the encoder start. In the default setup this is often empty.

### 14.3 `encoder_continuous`

A temporal tensor over the encoder window containing:

\[
[ ext{known continuous} \; | \;  ext{observed continuous} \; | \;  ext{target history}]
\]

in that order.

### 14.4 `encoder_categorical`

A temporal tensor over the encoder window containing:

\[
[ ext{known categorical} \; | \;  ext{observed categorical}]
\]

in that order.

### 14.5 `decoder_known_continuous`

A temporal tensor over the decoder horizon containing only future-known continuous features.

### 14.6 `decoder_known_categorical`

A temporal tensor over the decoder horizon containing only future-known categorical features.

### 14.7 `target`

A temporal vector over the decoder horizon containing the future glucose values to be predicted.

### 14.8 `metadata`

Human-readable identifiers and timestamps for debugging and later reporting. This is not a model input.

The dataset deliberately emits zero-width tensors instead of `None` for empty groups, because batching becomes simpler when every key always exists. Later model code can check feature widths.

---

## 15. Tensor Shapes: A Complete Reading Guide

Let:

- \(B\) = batch size
- \(L_e\) = encoder length
- \(L_d\) = decoder length / prediction length
- \(F_{sc}\) = number of static categorical features
- \(F_{s\ell}\) = number of static continuous features
- \(F_{kc}\) = number of known categorical features
- \(F_{k\ell}\) = number of known continuous features
- \(F_{oc}\) = number of observed categorical features
- \(F_{o\ell}\) = number of observed continuous features
- \(F_t\) = number of target channels

In the default branch:

- \(F_{sc} = 1\) (`subject_id`)
- \(F_{s\ell} = 0\)
- \(F_{kc} = 0\)
- \(F_{k\ell} = 5\)
- \(F_{oc} = 2\) (`device_mode`, `bolus_type`)
- \(F_{o\ell} = 5\) (basal, bolus, correction, meal, carbs)
- \(F_t = 1\) (glucose)

Therefore:

### 15.1 `static_categorical`

Shape:
\[
[B, F_{sc}] = [B, 1]
\]

### 15.2 `static_continuous`

Shape:
\[
[B, F_{s\ell}] = [B, 0]
\]
under the default setup.

### 15.3 `encoder_continuous`

Shape:
\[
[B, L_e, F_{k\ell} + F_{o\ell} + F_t]
\]

With defaults:
\[
[B, 168, 5 + 5 + 1] = [B, 168, 11]
\]

### 15.4 `encoder_categorical`

Shape:
\[
[B, L_e, F_{kc} + F_{oc}]
\]

With defaults:
\[
[B, 168, 0 + 2] = [B, 168, 2]
\]

### 15.5 `decoder_known_continuous`

Shape:
\[
[B, L_d, F_{k\ell}]
\]

With defaults:
\[
[B, 12, 5]
\]

### 15.6 `decoder_known_categorical`

Shape:
\[
[B, L_d, F_{kc}]
\]

With defaults:
\[
[B, 12, 0]
\]

### 15.7 `target`

Shape:
\[
[B, L_d]
\]

With defaults:
\[
[B, 12]
\]

These are the core runtime tensors one must understand before the model even begins.

---

## 16. What Each Tensor Means on the Original Patient Timeline

To make the abstraction concrete, imagine one default sample for one patient.

### Encoder interval

Suppose rows 1000 through 1167 form the encoder.
That is 168 steps, or 14 hours of history.

### Decoder interval

Suppose rows 1168 through 1179 form the decoder.
That is 12 steps, or 1 hour of future.

Then:

- `encoder_continuous[i, :, :]` for batch item \(i\) contains the 14-hour historical trajectory of:
  - cyclical time features,
  - observed insulin/carbohydrate signals,
  - historical glucose itself.

- `encoder_categorical[i, :, :]` contains the 14-hour historical trajectory of:
  - device mode,
  - bolus type.

- `decoder_known_continuous[i, :, :]` contains the next hour’s future-known time features:
  - what minute of day it will be,
  - what day of week it will be,
  - whether it is a weekend.

- `target[i, :]` contains the actual future glucose sequence over those 12 steps.

This is the exact information boundary of the forecasting problem. It is a very sensible one.

---

## 17. The Top-Level Configuration Story

The branch uses dataclass-based configuration objects to keep the pipeline coherent.

At the top level:

- `DataConfig` describes data paths, split policy, sequence lengths, loader behavior, and dataset facts.
- `TFTConfig` describes the TFT branch.
- `TCNConfig` describes the TCN branch.
- `Config` groups data + TFT + TCN.
- `TrainConfig`, `SnapshotConfig`, and `ObservabilityConfig` govern runtime policy.

This separation is more than cosmetic. It is what allows the codebase to say:

- “these are dataset facts,”
- “these are model architecture facts,”
- “these are runtime/training facts.”

In research code, that separation is invaluable because it reduces accidental drift between experiments.

---

## 18. Why the DataModule Exists

The repository uses `AZT1DDataModule` in the PyTorch Lightning style.

The DataModule owns:

- downloading / preparing processed data on disk,
- loading the cleaned dataframe,
- building category maps,
- splitting the dataframe,
- indexing legal windows,
- creating dataset instances,
- creating DataLoaders.

This is almost a textbook case of what a DataModule should do: keep the data lifecycle separate from the model lifecycle.

A subtle but important design point is that this DataModule also caches **categorical cardinalities**, because the model’s embedding tables depend on them. The DataModule therefore becomes the place where runtime-discovered data facts are bound into model configuration.

That is one of the branch’s cleanest design ideas.

---

## 19. Why the Dataset Exists Separately

The dataset does not split data. It does not create loaders. It does not fit vocabularies. It does not decide legal windows. It only does one thing:

> Given one indexed sequence window, assemble the exact tensor dictionary required by the model.

This single-responsibility design is excellent. It keeps `__getitem__` conceptually simple, and it avoids the classic “god dataset” anti-pattern in which one file handles downloading, cleaning, splitting, indexing, and batching all at once.

---

## 20. Why Indexing Exists Separately

The indexing layer answers a different question than the dataset.

The dataset asks:
“How do I pack one sample?”

The indexer asks:
“Which samples are valid in the first place?”

Those questions should indeed live in different places because they fail for different reasons.

- Indexing fails when a segment is too short or discontinuous.
- Dataset assembly fails when columns or category maps are wrong.

That separation is very useful for debugging research pipelines.

---

## 21. The Fused Model at a Glance

The repository’s core model is `FusedModel`, a LightningModule that wraps the full hybrid architecture.

High-level flow:

1. Split packed encoder tensors into semantic groups.
2. Build TCN branch input from observed history + target history.
3. Build TFT branch input from static, known, observed, and target groups.
4. Run three TCN branches with different kernel sizes.
5. Run the TFT branch.
6. Concatenate branch outputs at the horizon axis.
7. Apply a fusion GRN.
8. Apply a final neural head.
9. Emit quantiles per forecast horizon step.

This is a **late-fusion architecture**. The TCN outputs are not being passed as auxiliary decoder features into the TFT. Instead, TCN and TFT branch features meet after each branch has already produced horizon-aligned latent representations.

The comments in `fused_model.py` explicitly note that this is a change from an earlier alignment idea. That means the current branch is not merely using TCN as a preprocessor for TFT; it is treating TCN and TFT as parallel views of the forecasting problem.

---

## 22. Why Fuse TCN and TFT at All?

The scientific intuition is straightforward.

### TCN strength

A TCN is very good at causal local-to-mid-range pattern extraction. With dilated causal convolutions, it can efficiently detect motifs such as:

- post-meal rises,
- correction-driven declines,
- short-term oscillatory patterns,
- local rate-of-change structure.

### TFT strength

A TFT is designed for multi-horizon forecasting with mixed input types:

- static variables,
- known future variables,
- observed history,
- target history,
- variable selection,
- longer-range attention-based reasoning.

### Hybrid intuition

The repository’s hybrid design suggests that the authors want:

- TCN for efficient multi-scale temporal pattern extraction on the encoder history,
- TFT for richer semantic handling of future-known inputs and more structured temporal reasoning,
- late fusion so that neither branch is forced into the representational assumptions of the other too early.

This is intellectually coherent. It is a reasonable architectural hypothesis for glucose forecasting.

---

## 23. The Temporal Convolutional Network Branch

The TCN implementation is adapted from the broader TCN literature and open-source lineage, but narrowed to the repository’s use case.

### 23.1 Causal convolutions

A `CausalConv1d` pads only on the left. This ensures that the representation at time \(t\) depends only on times \(\leq t\), never on the future.

For forecasting, this is non-negotiable. If the convolution could see decoder-side future information, evaluation would be invalid.

### 23.2 Residual temporal blocks

Each `TemporalBlock` contains:

- causal convolution,
- layer normalization,
- activation,
- dropout,
- another causal convolution,
- another normalization,
- another activation,
- dropout,
- residual connection.

The residual connection helps optimization by letting the block learn a refinement of the incoming representation instead of forcing each layer to recreate all information from scratch.

### 23.3 Layer normalization instead of batch normalization

The branch intentionally standardizes on layer normalization rather than batch normalization. In heterogeneous medical time series, where batch composition can vary sharply by subject and condition, layer normalization can be a better fit because it does not depend on batch-wide statistics.

### 23.4 Multi-scale TCN branches

The fused model instantiates three TCN branches:

- kernel size 3,
- kernel size 5,
- kernel size 7.

All three operate on the same TCN input tensor, but each has a different local receptive-field bias.

This is a very intuitive design for physiological forecasting. Different glucose dynamics operate at different temporal scales:

- immediate sensor and short-lag response patterns,
- meal-to-glucose response curves,
- insulin action curves,
- broader drift.

A multi-kernel ensemble of TCN branches is one way to expose the model to those scales.

### 23.5 TCN input in the current branch

The fused model defines:

\[
 ext{TCN input size} =  ext{num observed continuous} +  ext{num target channels}.
\]

So the TCN sees:

- observed continuous encoder history,
- plus historical target trajectory.

It does **not** directly consume future-known decoder features. This is a deliberate information-budget difference between TCN and TFT.

This is sensible. The TCN branch is playing the role of a history encoder, not a full future-aware structured forecaster.

---

## 24. The Temporal Fusion Transformer Branch

The TFT branch is based on the TFT architecture introduced by Lim et al., with project-specific adaptations and an implementation lineage closer to NVIDIA’s DeepLearningExamples code.

### 24.1 What TFT is for

TFT was designed for **multi-horizon forecasting** in settings where variables come in different semantic types:

- static covariates,
- known future inputs,
- observed historical variables,
- target history.

That is exactly the kind of setting this repository has.

### 24.2 Separate variable embeddings

A key TFT idea preserved here is that variables are embedded separately rather than immediately mixed into a single dense vector.

Categorical variables get embedding tables.
Continuous variables get learned linear embedding vectors plus bias.

This preserves a “one slot per variable” structure so later variable-selection modules can determine which variables matter.

### 24.3 Seven input families

The TFT embedding comments list seven input types:

1. static categorical,
2. static continuous,
3. temporal known categorical,
4. temporal known continuous,
5. temporal observed categorical,
6. temporal observed continuous,
7. temporal target history.

This taxonomy is conceptually central. The repository’s grouped tensor contract is explicitly designed to feed this TFT worldview.

### 24.4 Continuous embedding mechanics

The TFT continuous embedding uses a learned pointwise linear transform per variable. If a continuous input group tensor has shape:

\[
[B, T, F],
\]

then the learned embedding vectors have shape:

\[
[F, H],
\]

where \(H\) is hidden size.

The resulting embedded tensor has shape:

\[
[B, T, F, H].
\]

In words: **each scalar feature at each time step becomes its own hidden vector**.

That is different from immediately concatenating all continuous features and feeding them through a shared linear layer. The TFT approach tries to preserve variable identity.

### 24.5 Variable selection

Although the fetched snippets do not expose every implementation detail line by line, the TFT architecture is explicitly described as using variable-selection networks. In TFT theory, variable selection learns a context-dependent weighting over variables so that the model can emphasize relevant inputs and suppress irrelevant ones. This is especially important in multivariate clinical time series, where feature importance can vary by subject, context, and time.

### 24.6 Temporal modeling inside TFT

The original TFT architecture combines:

- local processing layers,
- recurrent sequence modeling,
- self-attention for longer-range dependencies,
- gating blocks to suppress unnecessary components.

The repository comments explicitly frame the TFT branch as the future-aware side of the problem, handling static context, known decoder inputs, and richer temporal reasoning across the full example axis.

### 24.7 Why TFT fits this use case

Glucose forecasting is not just a short-lag autoregressive problem. The model benefits from knowing:

- what time of day the future corresponds to,
- which cyclical routines might be active,
- which subject is being modeled,
- how the observed historical signals interact with future-known clock structure.

That is exactly the sort of mixed semantic setting TFT was designed for.

---

## 25. Gated Residual Networks in This Repository

The GRN is a recurring nonlinear processing block used both inside TFT and again at the final fusion stage.

### 25.1 What a GRN does conceptually

A Gated Residual Network can be summarized as:

1. project input into a hidden space,
2. optionally inject context,
3. apply nonlinear transform,
4. use a gate to decide how much transformed content should pass,
5. add a residual shortcut,
6. normalize.

This is a very elegant block because it combines expressive transformation with a built-in “do not over-transform unnecessarily” mechanism.

### 25.2 Why gating matters

In noisy time-series problems, not every transformation should be fully trusted. A gate lets the model attenuate weak or harmful transformations.

### 25.3 Why the same GRN is reused for fusion

The branch intentionally uses the same GRN style both inside TFT and for the post-branch fusion layer. This makes the fused architecture numerically and stylistically aligned with the TFT internals rather than introducing a completely different fusion block.

That is good architectural discipline.

---

## 26. The Final Quantile Head

After branch fusion, the model uses `NNHead` to project the fused hidden representation into final quantile outputs.

The final head is no longer responsible for discovering how to combine TCN and TFT. That work has already been done by the fusion GRN. The head’s job is therefore simpler:

> Read horizon-wise fused latent representations and emit quantile values.

If the configured quantiles are:

\[
(0.1, 0.5, 0.9),
\]

then at each decoder time step the head emits three numbers:

- lower quantile estimate,
- median estimate,
- upper quantile estimate.

So the final output shape is:

\[
[B, L_d, Q],
\]

where \(Q\) is number of quantiles.

With defaults:

\[
[B, 12, 3].
\]

---

## 27. The Forward Pass, Step by Step

We may now reconstruct the forward pass of the fused model in conceptual order.

### Step 1: Receive the semantic batch dictionary

The model receives grouped tensors:

- static categorical
- static continuous
- encoder continuous
- encoder categorical
- decoder known continuous
- decoder known categorical
- target (future ground truth, used during supervision, not as predictive input)
- metadata

### Step 2: Split packed encoder continuous history

The helper `_split_encoder_continuous(...)` interprets `encoder_continuous` as:

\[
[ ext{known continuous} \; | \;  ext{observed continuous} \; | \;  ext{target history}]
\]

It returns:

- `known_history`
- `observed_history`
- `target_history`

This is crucial because the packed tensor is a storage convenience, not the conceptual model contract.

### Step 3: Split packed encoder categorical history

The helper `_split_encoder_categorical(...)` interprets `encoder_categorical` as:

\[
[ ext{known categorical} \; | \;  ext{observed categorical}]
\]

and returns those two groups separately.

### Step 4: Build TCN input

The TCN branch receives historical observed continuous variables plus target history.

If we denote observed encoder history by \(O_{1:L_e}\) and target history by \(Y_{1:L_e}\), then the TCN branch input is roughly

\[
X^{ ext{TCN}} = [O_{1:L_e}, Y_{1:L_e}].
\]

This tensor is then usually transposed into channel-first format for Conv1d:

\[
[B, L_e, F]
ightarrow [B, F, L_e].
\]

### Step 5: Run the three TCN branches

The branch runs:

- TCN with kernel 3,
- TCN with kernel 5,
- TCN with kernel 7.

Each branch transforms the history into a horizon-aligned latent representation. The exact internal details of the final temporal projection are abstracted by the TCN module, but the design intent is clear: each branch emits future-step features.

### Step 6: Build TFT grouped inputs

The TFT branch receives grouped tensors preserving semantic families:

- static categorical,
- static continuous,
- known continuous over encoder+decoder span,
- observed continuous over encoder,
- target history over encoder,
- known categorical if any,
- observed categorical over encoder.

The fused model also performs a constructor-time synthetic initialization step so the TFT’s lazy continuous-embedding parameters are materialized before Lightning configures optimizers. This is a practical engineering fix for Lightning compatibility.

### Step 7: Run the TFT branch

The TFT branch embeds each variable family, performs variable selection and temporal processing, and emits horizon-wise latent features aligned with the decoder horizon.

### Step 8: Concatenate branch features

The fused model explicitly computes a fused feature size equal to:

\[
H_{ ext{TFT}} + H_{ ext{TCN3}} + H_{ ext{TCN5}} + H_{ ext{TCN7}}.
\]

Thus, if each branch produces per-horizon latent vectors, the fusion tensor is their concatenation along the feature dimension.

### Step 9: Apply the fusion GRN

The concatenated branch representation is fed into a GRN that projects it back down to the TFT hidden size. This lets the model learn nonlinear cross-branch interactions.

### Step 10: Apply the final neural head

The final `NNHead` maps each horizon step’s fused hidden vector to quantile outputs.

### Step 11: Return quantile forecasts

Final output:

\[
\hat{Y} \in \mathbb{R}^{B  imes L_d  imes Q}.
\]

---

## 28. Mathematical Formulation of the Main Computation

A high-level mathematical abstraction of the current fused model is:

### 28.1 Data partition

Let

- \(S\) = static features,
- \(K^{enc}\) = encoder-known features,
- \(K^{dec}\) = decoder-known features,
- \(O\) = observed encoder features,
- \(Y^{hist}\) = historical target trajectory.

### 28.2 TCN branch

\[
H^{(3)} = f_{ ext{TCN},3}(O, Y^{hist}),
\]
\[
H^{(5)} = f_{ ext{TCN},5}(O, Y^{hist}),
\]
\[
H^{(7)} = f_{ ext{TCN},7}(O, Y^{hist}).
\]

### 28.3 TFT branch

\[
H^{ ext{TFT}} = f_{ ext{TFT}}(S, K^{enc}, K^{dec}, O, Y^{hist}).
\]

### 28.4 Fusion

\[
H^{ ext{fused}} =  ext{GRN}\left([H^{ ext{TFT}}; H^{(3)}; H^{(5)}; H^{(7)}]
ight),
\]
where \([ \cdot ; \cdot ]\) denotes concatenation along the feature dimension.

### 28.5 Quantile output

\[
\hat{Y}_{q} = g(H^{ ext{fused}})
\]
for quantiles \(q \in \mathcal{Q}\).

So the complete predictive distribution summary is:
\[
\hat{Y} = \{\hat{Y}*{0.1}, \hat{Y}*{0.5}, \hat{Y}_{0.9}\}
\]
under default settings.

---

## 29. Probabilistic Supervision and Pinball Loss

The repository uses **pinball loss**, also called quantile loss, as the central probabilistic supervision objective.

For one quantile \(q \in (0,1)\), target \(y\), and prediction \(\hat{y}_q\), the quantile loss is:

\[
\mathcal{L}_q(y, \hat{y}_q)
=

\max\left((q-1)(y-\hat{y}_q), \; q(y-\hat{y}_q)
ight).
\]

Equivalent piecewise form:

\[
\mathcal{L}_q(y, \hat{y}_q)
=

egin{cases}
q(y-\hat{y}_q), & y \ge \hat{y}_q \
(1-q)(\hat{y}_q-y), & y < \hat{y}_q
\end{cases}
\]

This asymmetric loss is exactly what a quantile regressor should use. It penalizes under-prediction and over-prediction differently depending on which quantile is being estimated.

### Interpretation

- For \(q = 0.5\), the loss becomes median regression.
- For \(q = 0.1\), the model is encouraged to place the prediction so that roughly 10% of true values fall below it.
- For \(q = 0.9\), the model is encouraged to place the prediction so that roughly 90% of true values fall below it.

The repository computes mean pinball loss across all quantiles and all forecast steps. This is appropriate for probabilistic multi-horizon supervision.

---

## 30. Point Forecast Extraction for Human-Readable Metrics

Even though the model is probabilistic, the repository still reports MAE and RMSE. This requires collapsing the quantile forecast into a representative point forecast.

The evaluation helpers include `select_point_prediction(...)`, which, under standard practice, would select the median quantile when present. This is sensible because the median is a robust central point forecast under quantile regression.

Thus, MAE and RMSE are not the primary training objective. They are interpretive secondary metrics.

---

## 31. What Training Is Actually Optimizing

The model is not directly optimizing MAE.
It is not directly optimizing RMSE.
It is optimizing **quantile pinball loss**.

That distinction matters a great deal.

### 31.1 Why optimize pinball loss?

Because the repository wants a predictive interval and central estimate, not just a single number.

### 31.2 Why still measure MAE/RMSE?

Because human readers and many forecasting comparisons still understand point errors more easily than distributional scores.

### 31.3 What interval statistics mean

The evaluation code also computes:

- mean prediction interval width,
- empirical interval coverage.

These tell us two different things:

- **width** measures sharpness: narrower intervals are more informative if still calibrated,
- **coverage** measures calibration-like behavior: do actual targets fall inside the outer interval as often as expected?

Together, they help interpret the probabilistic quality of the model.

---

## 32. What the Lightning Trainer Does for the Repository

The class `FusedModelTrainer` is not the model itself. It is an orchestration layer around PyTorch Lightning’s `Trainer`.

It owns:

- preparing the DataModule,
- binding runtime-discovered config into the model,
- optionally compiling the model,
- building callbacks,
- building the Trainer,
- launching `fit`,
- later reusing the same in-memory state for test/prediction flows.

### Why this wrapper is useful

It centralizes the repository’s policy decisions about training. That means notebooks, scripts, and future automation do not need to duplicate the same orchestration glue.

---

## 33. Checkpointing, Early Stopping, and Run Policy

The branch supports checkpointing and optional early stopping.

### With validation data

If validation windows exist, checkpoints are monitored against a validation metric, and early stopping can use the same monitor.

### Without validation data

The trainer falls back to “last checkpoint only” semantics, since there is no meaningful notion of “best” without a monitored validation signal.

This is a mature design choice. It avoids pretending to know a best checkpoint when no ranking signal exists.

---

## 34. Why the Repo Uses Runtime-Bound Configuration

A particularly elegant idea in this branch is that the final model config is not fully known until the DataModule has seen the cleaned data.

Why?

Because some model facts depend on the actual cleaned dataframe, especially categorical cardinalities. For example, the number of unique device modes or bolus types determines embedding sizes.

Thus the pipeline works like this:

1. start from a declarative config,
2. prepare the data,
3. build category maps,
4. bind discovered cardinalities and feature metadata into the TFT config,
5. instantiate the model.

This is much better than hard-coding guessed cardinalities.

---

## 35. Shape Walkthrough with a Concrete Example

Assume:

- batch size \(B = 64\),
- encoder length \(L_e = 168\),
- decoder length \(L_d = 12\),
- default feature counts.

Then:

### Static categorical

\[
[64, 1]
\]

Example meaning:
one integer subject ID per sample.

### Static continuous

\[
[64, 0]
\]

Currently empty in the default branch.

### Encoder continuous

\[
[64, 168, 11]
\]

Feature order:

- 5 known continuous time features,
- 5 observed continuous insulin/carb features,
- 1 historical glucose channel.

### Encoder categorical

\[
[64, 168, 2]
\]

Feature order:

- device mode,
- bolus type
assuming no known categorical features.

### Decoder known continuous

\[
[64, 12, 5]
\]

These are the next hour’s known-ahead time features.

### Decoder known categorical

\[
[64, 12, 0]
\]

Empty by default.

### Target

\[
[64, 12]
\]

True future glucose values for supervision.

### Model output

\[
[64, 12, 3]
\]

Quantiles:

- 0.1
- 0.5
- 0.9

This means:
for each sample and each future 5-minute step in the next hour,
the model emits three plausible glucose levels.

---

## 36. Strengths of the Current Design

Several aspects of the current branch are particularly strong.

### 36.1 Clean separation of responsibilities

Downloading, preprocessing, transforming, splitting, indexing, dataset assembly, model definition, and training orchestration are all separated. This is excellent research-software design.

### 36.2 Semantic feature grouping

The static / known / observed / target distinction is not only theoretically grounded in TFT; it is also implemented consistently across data and model layers.

### 36.3 Late fusion is conceptually honest

The branch no longer pretends that TCN outputs are just extra future covariates for TFT. Instead it treats TCN and TFT as parallel representational branches.

### 36.4 Probabilistic outputs

Quantile forecasting is well matched to the uncertainty of glucose prediction.

### 36.5 Strong attention to runtime configurability

The trainer wrapper, runtime config validation, and observability layers suggest a repository intended for repeated serious experimentation, not just a one-off demo.

---

## 37. Current Transitional Design Choices

The code comments make it clear that this branch is in a transitional but increasingly disciplined state.

### 37.1 Fallback feature specification synthesis

The DataModule can still synthesize `FeatureSpec` entries from fallback feature groups if `config.data.features` is not fully populated. This is practical for migration, but it is transitional.

### 37.2 Some architecture code inherits upstream TFT lineage

This is not a weakness by itself, but it does mean the branch mixes deeply project-specific code with adapted upstream code. Such hybrids are common in research repositories.

### 37.3 Default split policy reflects one scientific question, not all

The within-subject chronological split is useful, but it is not the same as a cross-subject generalization study.

---

## 38. Suboptimal or Risky Aspects and How They Could Improve

The user specifically asked that suboptimal aspects be explained plainly. Several deserve discussion.

### 38.1 Subject ID as a static categorical feature

**What it does now:**  
The current default contract effectively lets the model know which subject generated each sequence.

**Why this can help:**  
It can improve personalized forecasting because the model can learn subject-specific biases or response tendencies.

**Why it can be risky:**  
If train/validation/test all contain the same subjects, subject ID can make evaluation more personalized and less population-general. That is not inherently wrong, but it changes the meaning of performance.

**How to improve:**  
Run paired studies:

- one with subject ID included,
- one with subject ID excluded,
- one with subject-held-out splits.

This would clarify whether performance reflects true physiological generalization or partly memorized subject identity.

### 38.2 Within-subject chronological splitting as the default

**What it does now:**  
Each subject contributes earlier rows to train and later rows to validation/test.

**Why it is useful:**  
It matches personalized forecasting.

**Why it is limited:**  
It does not answer whether the model works on unseen patients.

**How to improve:**  
Include a standard subject-held-out benchmark mode in the main experiment suite.

### 38.3 Heavy overlap from stride-1 windows

**What it does now:**  
Consecutive training samples differ by only one timestep.

**Why this is common:**  
It enlarges the dataset and improves data efficiency.

**Why it can be problematic:**  
It creates highly correlated samples, which can inflate apparent data size and make gradient steps see near-duplicates repeatedly.

**How to improve:**  
Experiment with larger training stride, especially for ablation studies, and report the tradeoff between statistical efficiency and effective sample diversity.

### 38.4 Basal forward/backward fill may oversimplify physiology

**What it does now:**  
Treats basal insulin as continuous state and fills gaps accordingly.

**Why it is reasonable:**  
Basal is more state-like than event-like.

**Why it may be imperfect:**  
In some devices or exports, missing basal entries may reflect logging limitations rather than a truly constant state. Backward-filling leading gaps is especially assumption-heavy.

**How to improve:**  
Track an auxiliary missingness mask or confidence indicator for imputed basal values.

### 38.5 No explicit missingness channels for imputed variables

**What it does now:**  
Imputation happens, but the model may not be told which values were imputed.

**Why that matters:**  
Imputed values are less trustworthy than observed ones. A model that knows which values were imputed may learn better uncertainty behavior.

**How to improve:**  
Add binary missingness indicators for major channels.

### 38.6 Time features are useful but simple

**What it does now:**  
Uses cyclical clock/day features and weekend flag.

**Why that is good:**  
These are strong low-cost known-ahead covariates.

**What is missing:**  
Potentially richer behavioral context:

- time since last meal,
- time since last bolus,
- recent rolling glucose statistics,
- insulin-on-board approximations,
- carb-on-board approximations.

**How to improve:**  
Add physiologically motivated engineered features or a mechanistic auxiliary branch.

### 38.7 The current TCN branch only sees observed continuous + target history

**What it does now:**  
TCN ignores future-known time covariates directly.

**Why that is defensible:**  
The TCN branch is acting as a history encoder.

**What could be improved:**  
One could test variants where horizon-specific known future context conditions the TCN readout stage without violating causal constraints.

### 38.8 Quantile set is narrow

**What it does now:**  
Default quantiles are 0.1, 0.5, 0.9.

**Why this is practical:**  
It yields a lower/median/upper summary.

**Why it may be limiting:**  
It offers only a coarse view of uncertainty and can hide distributional asymmetry between 0.1 and 0.9.

**How to improve:**  
Use a denser quantile grid, e.g., 0.05 to 0.95.

### 38.9 Calibration is only lightly assessed

**What it does now:**  
Computes interval width and empirical interval coverage.

**Why this is a good start:**  
These are meaningful first-pass uncertainty metrics.

**What is missing:**  
More formal calibration diagnostics such as:

- coverage by horizon,
- calibration curves,
- quantile reliability plots,
- CRPS-style scores or approximations.

### 38.10 The hybrid architecture is plausible but not yet proven optimal

**What it does now:**  
Combines TCN and TFT via late fusion.

**Why that is attractive:**  
It merges complementary inductive biases.

**Why caution is needed:**  
Hybrid models can become large and difficult to interpret. The improvement over a strong pure TFT or pure TCN baseline must be demonstrated, not assumed.

**How to improve:**  
Perform rigorous ablations:

- TFT only,
- single TCN only,
- 3-branch TCN only,
- fused TCN + TFT,
- fused without GRN,
- fused with different kernel sets.

---

## 39. Interpretation of the Forecast Outputs

Suppose at one future step the model predicts:

- \(q_{0.1} = 105\)
- \(q_{0.5} = 128\)
- \(q_{0.9} = 160\)

This does **not** mean there is a 10% chance of exactly 105, a 50% chance of exactly 128, and so on.

Rather:

- the 10th percentile forecast says the true value is expected to fall below 105 only rarely,
- the 50th percentile forecast is the median estimate,
- the 90th percentile forecast says the true value is expected to fall below 160 most of the time.

So the interval \([105, 160]\) is a crude uncertainty band.

### Wide interval

If the interval is wide, the model is uncertain.

### Narrow interval

If the interval is narrow, the model is confident.

### But width alone is not enough

A narrow interval is only good if it still contains the truth often enough.

That is why both width and coverage matter.

---

## 40. What the Repository Is Trying to Achieve Scientifically

The deepest scientific goal of the repository is not simply “predict the next glucose value.”
It is more ambitious:

1. preserve physiologically meaningful inputs,
2. represent different kinds of time-series information according to their causal availability,
3. use a hybrid model to capture both short-range patterns and structured multi-horizon reasoning,
4. predict not only what will happen, but how uncertain that forecast is,
5. make the whole process reproducible and inspectable.

In that sense, the repository is trying to become not just a model implementation but a research artifact.

---

## 41. A Code-to-Algorithm Crosswalk

This section maps major files to algorithmic roles.

### `defaults.py`

Defines baseline research defaults for:

- data lengths,
- split ratios,
- model sizes,
- quantiles,
- runtime policy.

### `src/data/preprocessor.py`

Raw vendor export normalization into canonical CSV.

### `src/data/transforms.py`

Dataframe-wide cleanup, imputation, normalization, time feature generation, categorical normalization, vocabulary fitting.

### `src/data/schema.py`

Declares canonical columns, feature groups, categorical vocabularies, and the semantic meaning of features.

### `src/data/indexing.py`

Determines which windows are valid and how data are split.

### `src/data/dataset.py`

Converts one valid window into one grouped tensor sample.

### `src/data/datamodule.py`

Coordinates the whole data lifecycle and binds runtime-discovered metadata into model config.

### `src/config/data.py`

Data contract.

### `src/config/model.py`

TCN/TFT/top-level config contract.

### `src/models/tcn.py`

Project-specific temporal convolution branch.

### `src/models/tft.py`

TFT implementation and embedding front-end.

### `src/models/grn.py`

Reusable gated residual network block.

### `src/models/fused_model.py`

Main LightningModule that combines all branches, defines forward logic, loss logic, and optimizer behavior.

### `src/evaluation/metrics.py`

Primitive error metrics and probabilistic interval summaries.

### `src/train.py`

Reusable Lightning orchestration wrapper around fit/test/predict flow.

### `src/workflows/training.py`

Higher-level workflow layer that packages fit, evaluation, reports, and run summary generation.

---

## 42. Appendix A: Quick Tensor Cheat Sheet

### Data side

- `static_categorical`: `[B, F_sc]`
- `static_continuous`: `[B, F_sℓ]`
- `encoder_continuous`: `[B, L_e, F_kℓ + F_oℓ + F_t]`
- `encoder_categorical`: `[B, L_e, F_kc + F_oc]`
- `decoder_known_continuous`: `[B, L_d, F_kℓ]`
- `decoder_known_categorical`: `[B, L_d, F_kc]`
- `target`: `[B, L_d]`

### Default branch counts

- `F_sc = 1`
- `F_sℓ = 0`
- `F_kc = 0`
- `F_kℓ = 5`
- `F_oc = 2`
- `F_oℓ = 5`
- `F_t = 1`

### Default branch shapes

- `static_categorical`: `[B, 1]`
- `static_continuous`: `[B, 0]`
- `encoder_continuous`: `[B, 168, 11]`
- `encoder_categorical`: `[B, 168, 2]`
- `decoder_known_continuous`: `[B, 12, 5]`
- `decoder_known_categorical`: `[B, 12, 0]`
- `target`: `[B, 12]`
- `output`: `[B, 12, 3]`

---

## 43. Appendix B: Important Equations

### 43.1 Required window length

\[
L_{ ext{required}} = L_e + L_d
\]

### 43.2 Point metrics

\[
 ext{MAE} = rac{1}{N}\sum_i |\hat{y}_i - y_i|
\]

\[
 ext{RMSE} = \sqrt{rac{1}{N}\sum_i (\hat{y}_i - y_i)^2}
\]

### 43.3 Mean bias

\[
 ext{Bias} = rac{1}{N}\sum_i (\hat{y}_i - y_i)
\]

### 43.4 Quantile pinball loss

\[
\mathcal{L}_q(y,\hat{y}_q) =
\max\left((q-1)(y-\hat{y}_q), q(y-\hat{y}_q)
ight)
\]

### 43.5 Mean interval width

For outermost quantiles:
\[
 ext{MPIW} = rac{1}{N}\sum_i (\hat{y}*{i,q*{high}} - \hat{y}*{i,q*{low}})
\]

### 43.6 Empirical coverage

\[
 ext{Coverage} =
rac{1}{N}\sum_i \mathbf{1}\left[\hat{y}*{i,q*{low}} \le y_i \le \hat{y}*{i,q*{high}}
ight]
\]

---

## 44. Appendix C: Suggested Future Improvements

Below is a practical research agenda suggested by the current branch structure.

### 44.1 Data-side improvements

- add missingness masks,
- add time-since-event features,
- add insulin-on-board approximations,
- add carb-on-board approximations,
- test alternative imputation policies for basal and device mode.

### 44.2 Split/evaluation improvements

- standardize both within-subject and subject-held-out benchmarks,
- report horizon-wise performance,
- report performance by glucose range,
- include calibration diagnostics beyond coarse interval coverage.

### 44.3 Architecture improvements

- ablate TCN-only, TFT-only, and fused variants,
- test denser quantile sets,
- test subject-ID removal,
- test learned or mechanistic future conditioning for the TCN path,
- compare late fusion against earlier fusion alternatives.

### 44.4 Reporting improvements

- store explicit shape summaries in run artifacts,
- log feature-group cardinalities in run summaries,
- emit calibration plots by horizon,
- add representation-level diagnostics for each branch.

---

## 45. References and Provenance Notes

This document was grounded in the current public repository branch and in the canonical literature and dataset sources that the repository itself cites or aligns with.

### Repository sources examined

- GitHub repository root and README
- `defaults.py`
- `src/config/data.py`
- `src/config/model.py`
- `src/data/datamodule.py`
- `src/data/dataset.py`
- `src/data/preprocessor.py`
- `src/data/schema.py`
- `src/data/transforms.py`
- `src/data/indexing.py`
- `src/models/fused_model.py`
- `src/models/tcn.py`
- `src/models/tft.py`
- `src/models/grn.py`
- `src/evaluation/metrics.py`
- `src/train.py`
- `src/workflows/training.py`

### Canonical literature and external references

1. Lim, Arik, Loeff, and Pfister. **Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.**
2. Bai, Kolter, and Koltun. **An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.**
3. Ba, Kiros, and Hinton. **Layer Normalization.**
4. PyTorch Lightning documentation on **LightningDataModule**.
5. AZT1D dataset release and dataset paper description.
