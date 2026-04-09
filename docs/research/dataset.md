# Dataset

## 3.1 Dataset Provenance And Research Context

The experiments in this report use the public **AZT1D** dataset [7]. The prepared data summary stored in [`../../artifacts/main_run/reports/artifacts/shared_report/data/data_summary.json`](../../artifacts/main_run/reports/artifacts/shared_report/data/data_summary.json) records a cleaned modeling table with `300,607` rows from `25` subjects, spanning December 8, 2023 through April 7, 2024. The nominal cadence is `5` minutes, which matches the repository setting `sampling_interval_minutes = 5`.

That cadence does not imply perfect continuity. The subject timelines contain real gaps, including interruptions that last hours or days, and the pipeline preserves those discontinuities rather than interpolating across them. This choice is important because the forecasting task is meant to reflect observed clinical timelines, not an idealized evenly sampled series.

AZT1D is well suited to multimodal glucose forecasting because it includes glucose, carbohydrate intake, basal insulin, total bolus insulin, bolus subtypes, and device operating mode. The same richness also makes the raw data messy in the way real clinical exports usually are: column names vary, events are sparse, state variables persist between explicit updates, and duplicate timestamps appear in the source files.

## 3.2 From Raw Export To Canonical Table

Raw exports are first standardized by `AZT1DPreprocessor`, which rewrites vendor specific files into a canonical CSV with the following columns: `subject_id`, `timestamp`, `glucose_mg_dl`, `basal_insulin_u`, `bolus_insulin_u`, `correction_insulin_u`, `meal_insulin_u`, `carbs_g`, `device_mode`, `bolus_type`, and `source_file`. A later transform stage adds five calendar covariates, namely `minute_of_day_sin`, `minute_of_day_cos`, `day_of_week_sin`, `day_of_week_cos`, and `is_weekend`, which produces the `16` column modeling table used by the forecasting pipeline.

The table level cleaning stage removes rows that lack subject identity, timestamp, or target glucose; rounds timestamps to the nearest minute; sorts rows within subject; and collapses duplicate subject time pairs. According to the saved summary, the resulting modeling table contains no missing values across the declared feature columns.

Imputation rules follow the semantics of each variable. Basal insulin and `device_mode` are treated as persistent states and are forward filled within subject; basal insulin is also backward filled for leading gaps. Event style variables such as `bolus_insulin_u`, `correction_insulin_u`, `meal_insulin_u`, and `carbs_g` are zero filled once aligned to the common grid. `bolus_type` remains tied to discrete events and is not propagated through time. These choices are reasonable, but they should still be read as modeling assumptions rather than as properties guaranteed by the dataset.

## 3.3 Semantic Feature Roles

The cleaned table is not passed to the model as one anonymous matrix. Instead, the repository groups variables by causal availability. Under the default feature contract, `subject_id` is the only static variable. The five calendar covariates are known in advance and are available on both the encoder and decoder axes. Historically observed continuous variables are `basal_insulin_u`, `bolus_insulin_u`, `correction_insulin_u`, `meal_insulin_u`, and `carbs_g`. Historically observed categorical variables are `device_mode` and `bolus_type`. The target variable is `glucose_mg_dl`.

This partition is important for both modeling and interpretation. It preserves the difference between what is genuinely known at forecast time and what is only observed in the past. It also gives the TCN and TFT branches compatible but not identical views of the same sample. Because continuous glucose monitor values are delayed proxies of blood glucose, the historical target series is itself a meaningful part of the input rather than just an outcome label.

## 3.4 Sequence Construction And Split Policy

Once the cleaned dataframe has been built, `AZT1DDataModule` constructs legal forecast windows through `build_sequence_index(...)`. A sample is valid only if it lies entirely inside a contiguous subject segment that respects the expected `5` minute cadence and if the segment is long enough to support both the encoder and decoder. With the default settings, each legal sample contains `168` encoder steps and `12` decoder steps, which corresponds to `14` hours of history and a `1` hour forecast horizon.

The saved artifact set uses a within subject chronological `70 / 15 / 15` split. The shared data summary records `210,411` training rows, `45,082` validation rows, and `45,114` test rows. These become `143,237` training windows, `27,221` validation windows, and `28,234` test windows. The default `window_stride = 1` produces heavily overlapping samples, so the large window count should not be interpreted as the same thing as a large number of independent trajectories.

## 3.5 Batch Interface Used By The Model

`AZT1DSequenceDataset` returns a grouped batch dictionary with the keys `static_categorical`, `static_continuous`, `encoder_continuous`, `encoder_categorical`, `decoder_known_continuous`, `decoder_known_categorical`, `target`, and `metadata`. Under the default feature contract, the main tensor shapes are `static_categorical [B, 1]`, `encoder_continuous [B, 168, 11]`, `encoder_categorical [B, 168, 2]`, `decoder_known_continuous [B, 12, 5]`, and `target [B, 12]`. Empty groups are represented as zero width tensors rather than being omitted.

This grouped interface is more informative than a flat feature matrix. It makes the information boundary of the forecasting task explicit, and it allows the TCN and TFT branches to consume different views of the same sample without reconstructing those views inside the model. The methodology section builds directly on that data contract.
