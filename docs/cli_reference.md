# CLI Argument Map

This document maps the current `src/workflows/cli.py` flags to the typed config objects and workflow parameters used by the training entrypoint.

## How the CLI is wired

At a high level, the CLI flow is:

1. `cli.py` registers flat command-line flags.
2. `_build_cli_configuration(...)` converts parsed values into:
   - `Config(data=..., tft=..., tcn=...)`
   - `TrainConfig`
   - `SnapshotConfig`
   - `ObservabilityConfig`
3. `device-profile` resolution may further adjust some effective training, data, and observability settings before the workflow starts.
4. `main(...)` passes some remaining flags directly to `run_training_workflow(...)` or `run_environment_benchmark_workflow(...)`.

## Important notes

- **Effective runtime behavior may differ from CLI defaults** due to device-profile resolution and environment-aware adjustments.

- **CLI defaults are not always the same as raw dataclass defaults.** The effective defaults used by the CLI come from `cli.py` plus the builder functions in `defaults.py`.
- **Some flags do not map to a config field directly.** They control workflow dispatch, benchmarking, checkpoint selection, or artifact saving.
- **Some booleans are tri-state in the parser.** Flags using `BooleanOptionalAction` default to `None` in argparse, then get converted later to an explicit boolean in `_build_cli_configuration(...)`.
- **`device-profile` is special.** It is not just stored; it is used by `resolve_device_profile(...)` to produce the final effective configs.

---

## 1. Data arguments

| Flag | Type / Expected values | CLI default | Maps to | Effective default after build | Bounds / guidance | What it controls |
| --- | --- | ---: | --- | ---: | --- | --- |
| `--dataset-url` | string or empty-like value | `DEFAULT_AZT1D_URL` | `Config.data.dataset_url` via `build_default_config(dataset_url=...)` | public AZT1D URL | valid URL or `None`-like | Raw dataset download source. Set to empty/none-like only when data already exists locally. |
| `--raw-dir` | path | `ROOT_DIR/data/raw` | `Config.data.raw_dir` | same | existing or creatable path | Location of original downloaded raw files. |
| `--cache-dir` | path | `ROOT_DIR/data/cache` | `Config.data.cache_dir` | same | existing or creatable path | Temporary/cache area for download and extraction artifacts. |
| `--extract-dir` | path | `ROOT_DIR/data/extracted` | `Config.data.extracted_dir` | same | existing or creatable path | Directory for extracted raw contents. |
| `--processed-dir` | path | `ROOT_DIR/data/processed` | `Config.data.processed_dir` | same | existing or creatable path | Directory for processed model-ready data. |
| `--processed-file-name` | filename string | `azt1d_processed.csv` | `Config.data.processed_file_name` | same | usually `.csv` | Output filename inside `processed-dir`. |
| `--output-dir` | path | `DEFAULT_OUTPUT_DIR` | **not a single data field**; used as `output_dir`, `TrainConfig.default_root_dir`, `SnapshotConfig.output_dir`, `ObservabilityConfig.output_dir` | same | existing or creatable path | Root directory for run artifacts, reports, logs, and default checkpoints. |
| `--checkpoint-dir` | path or omitted | `None` | `SnapshotConfig.dirpath` via `build_default_snapshot_config(dirpath=...)` | if `None`, defaults builder uses `output_dir/checkpoints` | valid path or omitted | Overrides where checkpoints are saved. |
| `--encoder-length` | int | `168` | `Config.data.encoder_length` | same | must be `> 0`; usually larger than `prediction-length` | Number of past timesteps fed to the model. |
| `--prediction-length` | int | `12` | `Config.data.prediction_length`; also `TCNConfig.prediction_length`; also used in TFT example length | same | must be `> 0` | Forecast horizon per sample window. |
| `--batch-size` | int | `64` | `Config.data.batch_size` | same | should be `> 0` | DataLoader batch size. Larger values use more memory. |
| `--num-workers` | int | `0` | `Config.data.num_workers` | same | `>= 0` | Number of DataLoader worker processes. `0` means main-process loading. |
| `--prefetch-factor` | int or omitted | `None` | `Config.data.prefetch_factor` | same | `>= 1` when used; most relevant if `num-workers > 0` | Per-worker prefetch queue depth. |
| `--pin-memory` / `--no-pin-memory` | bool | `None` in argparse | `Config.data.pin_memory` | `False` when omitted | boolean | Enables pinned host memory for some accelerator input pipelines. |
| `--persistent-workers` / `--no-persistent-workers` | bool | `None` in argparse | `Config.data.persistent_workers` | `False` when omitted | boolean; mainly useful when `num-workers > 0` | Keeps DataLoader workers alive across epochs. |

### Data fields that are **not** exposed directly in the CLI

These are still part of `DataConfig` defaults, but are not direct top-level CLI flags today:

- `sampling_interval_minutes = 5`
- `window_stride = 1`
- `train_ratio = 0.70`
- `val_ratio = 0.15`
- `test_ratio = 0.15`
- `split_by_subject = False`
- `split_within_subject = True`

---

## 2. Runtime and training arguments

| Flag | Type / Expected values | CLI default | Maps to | Effective default after build | Bounds / guidance | What it controls |
| --- | --- | ---: | --- | ---: | --- | --- |
| `--max-epochs` | int | `20` | `TrainConfig.max_epochs` | same | `> 0` | Maximum number of training epochs. |
| `--device-profile` | choice string | `auto` | **not stored as a simple field**; passed to `resolve_device_profile(...)` | `auto` | must be one of `DEVICE_PROFILE_CHOICES` (defined in `src/workflows/cli.py`) | High-level runtime preset. Used to derive final effective configs for the current environment. |
| `--accelerator` | string | `auto` | `TrainConfig.accelerator` | same | backend-supported value | Lightning accelerator selection. |
| `--devices` | string | `auto` | `TrainConfig.devices` after `_parse_devices(...)` | same logical value | `auto`, int-like string, or backend-specific form | Which devices Lightning should use. |
| `--precision` | string | `32` | `TrainConfig.precision` | converted to `int(32)` when numeric | integer-like or Lightning precision mode such as `16-mixed` | Numeric precision mode. |
| `--gradient-clip-val` | float or omitted | `None` | `TrainConfig.gradient_clip_val` | same | `>= 0` when used | Gradient clipping threshold. |
| `--accumulate-grad-batches` | int | `1` | `TrainConfig.accumulate_grad_batches` | same | `>= 1` | Gradient accumulation steps. |
| `--strategy` | string | `auto` | `TrainConfig.strategy` | same | backend-supported value | Lightning execution / distributed strategy. |
| `--sync-batchnorm` / `--no-sync-batchnorm` | bool | `None` in argparse | `TrainConfig.sync_batchnorm` | `False` when omitted | boolean | Enables synchronized batch norm, mainly for multi-device runs. |
| `--matmul-precision` | string or omitted | `None` | `TrainConfig.matmul_precision` | same | backend-supported value | Matrix multiplication precision hint. |
| `--allow-tf32` / `--no-allow-tf32` | bool | `None` | `TrainConfig.allow_tf32` | same | boolean | Allows TF32 where supported on CUDA hardware. |
| `--cudnn-benchmark` / `--no-cudnn-benchmark` | bool | `None` | `TrainConfig.cudnn_benchmark` | same | boolean | Enables cuDNN autotuning for stable input shapes. |
| `--intraop-threads` | int or omitted | `None` | `TrainConfig.intraop_threads` | same | `>= 1` when used | Torch CPU intra-op thread count. |
| `--interop-threads` | int or omitted | `None` | `TrainConfig.interop_threads` | same | `>= 1` when used | Torch CPU inter-op thread count. |
| `--mps-high-watermark-ratio` | float or omitted | `None` | `TrainConfig.mps_high_watermark_ratio` | same | positive float when used | Apple Silicon MPS allocator tuning. |
| `--mps-low-watermark-ratio` | float or omitted | `None` | `TrainConfig.mps_low_watermark_ratio` | same | positive float when used | Apple Silicon MPS allocator tuning. |
| `--enable-mps-fallback` / `--no-enable-mps-fallback` | bool | `None` | `TrainConfig.enable_mps_fallback` | same | boolean | Allows unsupported MPS ops to fall back to CPU. |
| `--compile-model` / `--no-compile-model` | bool | `None` in argparse | `TrainConfig.compile_model` | `False` when omitted | boolean | Enables Torch compile. |
| `--compile-mode` | string or omitted | `None` | `TrainConfig.compile_mode` | normalized optional string | backend-supported value | Compile mode when compile is enabled. |
| `--compile-fullgraph` | flag | `False` | `TrainConfig.compile_fullgraph` | same | boolean | Requests full-graph compilation. |
| `--learning-rate` | float | `1e-3` | **workflow argument**, not stored in `TrainConfig` | same | `> 0` | Optimizer learning rate passed directly into the training / benchmark workflow. |
| `--weight-decay` | float | `0.0` | **workflow argument**, not stored in `TrainConfig` | same | `>= 0` | Optimizer weight decay passed directly to the workflow. |
| `--optimizer` | string | `adam` | **workflow argument**, not stored in `TrainConfig` | same | workflow-supported optimizer name | Optimizer selection used by the workflow. |
| `--seed` | int | `42` | **workflow argument**, not stored in `TrainConfig` | same | any integer | Random seed passed to the workflow. |
| `--limit-train-batches` | string parsed to int/float | `1.0` | `TrainConfig.limit_train_batches` via `_parse_limit(...)` | `1.0` | positive int or float in `[0, 1]` style | Limits training batches for debugging or benchmarking. |
| `--limit-val-batches` | string parsed to int/float | `1.0` | `TrainConfig.limit_val_batches` | `1.0` | positive int or float in `[0, 1]` style | Limits validation batches. |
| `--limit-test-batches` | string parsed to int/float | `1.0` | `TrainConfig.limit_test_batches` | `1.0` | positive int or float in `[0, 1]` style | Limits test batches. |
| `--early-stopping-patience` | int | `5` | `TrainConfig.early_stopping_patience` | same | `>= 0` or `None` in underlying builder | Early stopping patience. |
| `--fit-ckpt-path` | path or omitted | `None` | **workflow argument**, not stored in `TrainConfig` | normalized optional string | valid path or omitted | Checkpoint path used when fitting/resuming. |
| `--eval-ckpt-path` | string/path | `best` | **workflow argument**, not stored in `TrainConfig` | normalized optional string | often `best` or path | Which checkpoint to use for evaluation/testing. |
| `--deterministic` | flag | `False` | `TrainConfig.deterministic` | same | boolean | Requests deterministic execution. |
| `--fast-dev-run` | flag | `False` | `TrainConfig.fast_dev_run` | same | boolean | Very short end-to-end smoke test mode. |
| `--progress-bar` / `--no-progress-bar` | bool | `None` in argparse | `TrainConfig.enable_progress_bar` | `True` when omitted | boolean | Standard progress bar toggle. |
| `--rich-progress-bar` / `--no-rich-progress-bar` | bool | `None` in argparse | `ObservabilityConfig.enable_rich_progress_bar` | `True` when omitted | boolean | Rich progress bar toggle. |
| `--device-stats` / `--no-device-stats` | bool | `None` in argparse | `ObservabilityConfig.enable_device_stats` | `True` when omitted | boolean | Enables device statistics collection / reporting. |
| `--fail-on-preflight-errors` / `--no-fail-on-preflight-errors` | bool | `None` in argparse | **workflow argument**, not stored in config dataclasses | `True` when omitted | boolean | Whether preflight diagnostic errors should abort the run. |

### Training fields with defaults that are **not** exposed directly in the CLI

The builder also sets these `TrainConfig` defaults, but they are not current top-level CLI flags:

- `log_every_n_steps = 10`
- `num_sanity_val_steps = 2`
- `enable_model_summary = True`

---

## 3. Model arguments

| Flag | Type / Expected values | CLI default | Maps to | Effective default after build | Bounds / guidance | What it controls |
| --- | --- | ---: | --- | ---: | --- | --- |
| `--tcn-channels` | CSV ints | `64,64,128` | `TCNConfig.num_channels` via `_parse_csv_ints(...)` | `(64, 64, 128)` | positive ints | Hidden channel sizes for TCN blocks. |
| `--tcn-dilations` | CSV ints | `1,2,4` | `TCNConfig.dilations` via `_parse_csv_ints(...)` | `(1, 2, 4)` | positive ints | Dilation schedule for temporal convolutions. |
| `--tcn-kernel-size` | int | `3` | `TCNConfig.kernel_size` | same | `> 0`, typically small | TCN convolution kernel size. |
| `--tft-hidden-size` | int | `128` | `TFTConfig.hidden_size` | same | `> 0`; often should align with attention/head constraints | Hidden representation size for TFT. |
| `--tft-n-head` | int | `4` | `TFTConfig.n_head` | same | `> 0`; usually should divide hidden size cleanly | Number of attention heads in TFT. |
| `--quantiles` | CSV floats | `0.1,0.5,0.9` | `TFTConfig.quantiles` via `_parse_csv_floats(...)` | `(0.1, 0.5, 0.9)` | each typically in `(0, 1)` and ordered | Quantile levels for probabilistic forecasting. |

### Model defaults that are **not** exposed directly in the CLI

The builder also fixes these model defaults unless you change code or builder arguments:

- `TCNConfig.num_inputs = 1`
- `TCNConfig.dropout = 0.1`
- `TCNConfig.output_size = 1`
- `TFTConfig.dropout = 0.1`
- `TFTConfig.example_length = encoder_length + prediction_length`

---

## 4. Behavior and workflow-control arguments

These flags mostly change **what the workflow does**, not the persistent config dataclasses.

| Flag | Type / Expected values | CLI default | Maps to | Effective default after build | Bounds / guidance | What it controls |
| --- | --- | ---: | --- | ---: | --- | --- |
| `--rebuild-processed` | flag | `False` | `Config.data.rebuild_processed` | same | boolean | Forces processed data regeneration. |
| `--redownload` | flag | `False` | `Config.data.redownload` | same | boolean | Forces raw dataset re-download. |
| `--run-benchmark-only` | flag | `False` | **workflow dispatch only** | same | boolean | Runs benchmark workflow instead of full train/eval workflow. |
| `--benchmark-train-batches` | int | `10` | **benchmark workflow argument** | same | `> 0` | Number of train batches used in benchmark-only mode. |
| `--run-diagnostics-only` | flag | `False` | **workflow dispatch only** | same | boolean | Prints diagnostics and exits without training. |
| `--skip-test` | flag | `False` | **workflow argument** | same | boolean | Skips test phase. |
| `--skip-predict` | flag | `False` | **workflow argument** | same | boolean | Skips prediction generation/export. |
| `--no-save-predictions` | flag | `False` | **workflow argument** as `save_predictions = not args.no_save_predictions` | saves predictions by default | boolean | Disables saving prediction artifacts. |
| `--disable-checkpoints` | flag | `False` | `SnapshotConfig.enabled = not disable_checkpoints` | checkpoints enabled by default | boolean | Disables checkpoint callback/artifacts. |
| `--save-weights-only` | flag | `False` | `SnapshotConfig.save_weights_only` | same | boolean | Saves weights-only checkpoints where supported. |

---

## 5. Observability arguments

| Flag | Type / Expected values | CLI default | Maps to | Effective default after build | Bounds / guidance | What it controls |
| --- | --- | ---: | --- | ---: | --- | --- |
| `--observability-mode` | string | `baseline` | `ObservabilityConfig.mode` | same | repo-supported mode string | High-level observability preset. |
| `--disable-tensorboard` | flag | `False` | `ObservabilityConfig.enable_tensorboard = not disable_tensorboard` | enabled by default | boolean | Disables TensorBoard logging. |
| `--disable-plot-reports` | flag | `False` | `ObservabilityConfig.enable_plot_reports = not disable_plot_reports` | enabled by default | boolean | Disables generated plot/report artifacts. |
| `--disable-system-telemetry` | flag | `False` | `ObservabilityConfig.enable_system_telemetry = not disable_system_telemetry` | enabled by default | boolean | Disables system telemetry collection. |
| `--disable-gradient-stats` | flag | `False` | `ObservabilityConfig.enable_gradient_stats = not disable_gradient_stats` | enabled by default | boolean | Disables gradient statistics. |
| `--enable-activation-stats` | flag | `False` | `ObservabilityConfig.enable_activation_stats` | disabled by default | boolean | Enables activation statistics collection. |
| `--disable-parameter-histograms` | flag | `False` | `ObservabilityConfig.enable_parameter_histograms = not disable_parameter_histograms` | enabled by default | boolean | Disables parameter histogram logging. |
| `--disable-parameter-scalars` | flag | `False` | `ObservabilityConfig.enable_parameter_scalars = not disable_parameter_scalars` | enabled by default | boolean | Disables parameter scalar logging. |
| `--disable-prediction-figures` | flag | `False` | `ObservabilityConfig.enable_prediction_figures = not disable_prediction_figures` | enabled by default | boolean | Disables prediction figure generation. |
| `--disable-model-graph` | flag | `False` | `ObservabilityConfig.enable_model_graph = not disable_model_graph` | enabled by default | boolean | Disables model graph export/logging. |
| `--disable-model-text` | flag | `False` | `ObservabilityConfig.enable_model_text = not disable_model_text` | enabled by default | boolean | Disables textual model summaries. |
| `--disable-torchview` | flag | `False` | `ObservabilityConfig.enable_torchview = not disable_torchview` | enabled by default | boolean | Disables torchview diagram generation. |
| `--torchview-depth` | int | `4` | `ObservabilityConfig.torchview_depth` | same | `> 0` | Max depth for torchview visualization. |
| `--enable-profiler` | flag | `False` | `ObservabilityConfig.enable_profiler` | same | boolean | Enables runtime profiler. |
| `--profiler-type` | string | `simple` | `ObservabilityConfig.profiler_type` | same | repo-supported profiler type | Selects profiler backend/type. |

### Observability defaults that are **not** exposed directly in the CLI

The builder also enables these by default unless changed in code:

- `enable_text_logging = True`
- `enable_csv_fallback_logger = True`
- `enable_learning_rate_monitor = True`

---

## 6. Quick “what flag should I change?” guide

### Data location and preprocessing

- Change raw dataset URL: `--dataset-url`
- Move raw/cache/extracted/processed folders: `--raw-dir`, `--cache-dir`, `--extract-dir`, `--processed-dir`
- Rename processed CSV: `--processed-file-name`
- Force preprocessing again: `--rebuild-processed`
- Force download again: `--redownload`

### Sequence shape

- Change input history length: `--encoder-length`
- Change forecast horizon: `--prediction-length`

### Data loading

- Larger/smaller batches: `--batch-size`
- More loader workers: `--num-workers`
- Tune worker prefetch: `--prefetch-factor`
- Enable pinned memory: `--pin-memory`
- Keep workers alive: `--persistent-workers`

### Device/runtime

- Let repo auto-pick profile: `--device-profile auto`
- Force a specific runtime preset: `--device-profile <choice>` (see `DEVICE_PROFILE_CHOICES` in `src/workflows/cli.py`)
- Force accelerator/devices manually: `--accelerator`, `--devices`
- Change precision: `--precision`
- Enable compile: `--compile-model`
- Make run deterministic: `--deterministic`

### Optimization

- Change learning rate: `--learning-rate`
- Add weight decay: `--weight-decay`
- Change optimizer: `--optimizer`
- Clip gradients: `--gradient-clip-val`
- Use gradient accumulation: `--accumulate-grad-batches`

### Training duration / debug

- Fewer or more epochs: `--max-epochs`
- Limit batches for quick tests: `--limit-train-batches`, `--limit-val-batches`, `--limit-test-batches`
- Smoke test end-to-end: `--fast-dev-run`
- Diagnostics only: `--run-diagnostics-only`
- Benchmark only: `--run-benchmark-only`

### Model shape

- Change TCN channels: `--tcn-channels`
- Change TCN dilations: `--tcn-dilations`
- Change TCN kernel size: `--tcn-kernel-size`
- Change TFT hidden size: `--tft-hidden-size`
- Change TFT heads: `--tft-n-head`
- Change forecast quantiles: `--quantiles`

### Checkpoints and outputs

- Change artifact root: `--output-dir`
- Change checkpoint directory: `--checkpoint-dir`
- Disable checkpoints: `--disable-checkpoints`
- Save weights only: `--save-weights-only`
- Skip test: `--skip-test`
- Skip predict: `--skip-predict`
- Do not save predictions: `--no-save-predictions`

### Logging / diagnostics / reports

- Change observability preset: `--observability-mode`
- Disable TensorBoard: `--disable-tensorboard`
- Disable telemetry: `--disable-system-telemetry`
- Enable activation stats: `--enable-activation-stats`
- Disable gradient stats: `--disable-gradient-stats`
- Disable torchview: `--disable-torchview`
- Adjust torchview size/detail: `--torchview-depth`
- Enable profiler: `--enable-profiler`
- Choose profiler kind: `--profiler-type`
- Rich progress bar: `--rich-progress-bar`
- Device stats: `--device-stats`

---

## 7. Common example commands

### Full default local run

```bash
python main.py
````

### Short debug run

```bash
python main.py \
  --fast-dev-run \
  --run-diagnostics-only
```

### Apple Silicon debug-style run

```bash
python main.py \
  --device-profile apple-silicon \
  --max-epochs 5 \
  --batch-size 128 \
  --num-workers 1 \
  --observability-mode debug \
  --rich-progress-bar \
  --device-stats \
  --enable-activation-stats
```

### Benchmark-only run

```bash
python main.py \
  --run-benchmark-only \
  --benchmark-train-batches 20
```

### Custom model shape

```bash
python main.py \
  --encoder-length 288 \
  --prediction-length 24 \
  --tcn-channels 64,128,128 \
  --tcn-dilations 1,2,4 \
  --tft-hidden-size 160 \
  --tft-n-head 8 \
  --quantiles 0.05,0.5,0.95
```

---

## 8. Source-of-truth reminder

For maintenance, use these files together:

- `src/workflows/cli.py`
  Defines the public CLI surface and workflow dispatch.
- `defaults.py`
  Defines how parsed CLI values become typed config objects.
- `src/config/data.py`
  Data contract and defaults.
- `src/config/observability.py`
  Observability contract and defaults.
- `src/config/train.py`
  Training/runtime contract and defaults.
- `src/config/snapshot.py`
  Checkpoint policy contract and defaults.

If you add a new CLI flag, update all of:

1. the parser section in `cli.py`
2. `_build_cli_configuration(...)`
3. this document
4. any affected config/dataclass docs

Built from the current `cli.py`, `defaults.py`, and config files in your `danish/dev` branch, including the argument registration in `cli.py`, the config-builder functions in `defaults.py`, and the `DataConfig` / `ObservabilityConfig` contracts.
