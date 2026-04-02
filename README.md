# ENSF617-Probabilistic-Glucose-Forecasting

Hybrid TCN–Transformer (TFT) model for probabilistic blood glucose forecasting from real-world time-series data (CGM, insulin, meals, physiological signals). Captures short- and long-term dependencies with uncertainty-aware predictions; extensible to physics-informed and dynamical system modeling.

## Refactor Notes

The recommended top-level project guides are:

- [docs/codebase_evolution.md](docs/codebase_evolution.md)
- [docs/current_architecture.md](docs/current_architecture.md)

Archived milestone summaries and earlier refactor notes now live under
[docs/history/](docs/history/).

Architecture diagrams now live under [docs/assets/](docs/assets/).

## Testing

### Install dependencies

Local:

```bash
pip install -r requirements.txt
```

For environment-specific PyTorch installs, it is usually better to install the
right Torch build first and then install the rest of the project dependencies:

CUDA:

```bash
pip install -r requirements.txt
```

Then replace the default CPU-only Torch install with the CUDA-enabled wheel
that matches your local CUDA runtime by following the official PyTorch install
selector.

Apple Silicon:

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### Run focused automated tests

Run the full pytest suite that is currently tracked in the repository:

```bash
pytest tests -q
```

Run a few representative modules:

```bash
pytest tests/test_config.py tests/test_fused_model.py tests/test_train.py -q
```

Run the evaluation-focused tests:

```bash
pytest tests/test_evaluation_metrics.py tests/test_evaluation_evaluator.py -q
```

### Manual smoke test

For a heavier end-to-end data pipeline check that can download and prepare the
AZT1D dataset, use the manual smoke script:

```bash
python tests/manual_data_smoke.py
```

This is intentionally separate from the pytest suite because it may touch the
network and real filesystem state.

## Running The Full Pipeline

You can now launch the full train/test workflow from either the script or the
notebook at the repository root.

Run the script:

```bash
python main.py --max-epochs 5 --batch-size 32
```

Or select one explicit runtime profile:

```bash
python main.py --device-profile local-cpu
python main.py --device-profile local-cuda
python main.py --device-profile apple-silicon
python main.py --device-profile slurm-cpu
python main.py --device-profile slurm-cuda
python main.py --device-profile colab-cpu
python main.py --device-profile colab-cuda
```

This entrypoint will:

- add `src/` to the import path
- keep `main.py` as the stable user-facing facade while delegating the heavier
  orchestration to the smaller modules under `src/workflows/`
- download and preprocess AZT1D automatically if the processed CSV is missing
- train the fused model
- run held-out test evaluation when test windows exist
- optionally generate detailed held-out evaluation, prediction exports, and
  report artifacts under `artifacts/main_run/`
- detect the current environment and record the resolved runtime profile in the
  run summary
- run preflight diagnostics so likely environment issues are surfaced before
  training starts when possible
- choose environment-aware defaults for precision and DataLoader workers, while
  still letting explicit CLI flags override them

### Runtime Diagnostics

To inspect environment readiness without starting training, run:

```bash
python main.py --device-profile auto --run-diagnostics-only
```

The diagnostics flow reports the detected environment plus likely
misconfigurations such as:

- missing `torch` or `pytorch-lightning`
- requesting CUDA when no GPU is visible
- requesting MPS when Apple Silicon acceleration is unavailable
- suspicious DataLoader settings for Colab or Apple Silicon
- Slurm-oriented settings outside a Slurm allocation

Explicit runtime flags still win over profile defaults. For example:

```bash
python main.py --device-profile slurm-cuda --precision 32 --devices 1
```

In that case the profile provides the baseline, while `--precision` and
`--devices` remain authoritative.

Additional runtime-control flags are available when you want to keep the
profile but tune a few environment-sensitive knobs:

```bash
python main.py --device-profile colab-cuda --pin-memory --persistent-workers
python main.py --device-profile apple-silicon --no-pin-memory --no-persistent-workers
python main.py --device-profile slurm-cuda --no-progress-bar --no-rich-progress-bar
python main.py --device-profile auto --no-fail-on-preflight-errors
```

Useful controls include:

- `--pin-memory` / `--no-pin-memory`
- `--persistent-workers` / `--no-persistent-workers`
- `--progress-bar` / `--no-progress-bar`
- `--rich-progress-bar` / `--no-rich-progress-bar`
- `--device-stats` / `--no-device-stats`
- `--fail-on-preflight-errors` / `--no-fail-on-preflight-errors`

Profile defaults now adapt a bit more aggressively to the host:

- CUDA profiles prefer `bf16-mixed` automatically when the detected GPU
  reports BF16 support, otherwise they fall back to `16-mixed`
- CPU profiles can now prefer `bf16-mixed` automatically on CPUs that report
  BF16-capable instruction support
- local CPU and local CUDA profiles derive `num_workers` from available CPU
  cores instead of always using one static value
- Apple Silicon keeps the worker pool intentionally small and disables pinned
  memory by default
- CUDA profiles also enable throughput-oriented defaults such as TF32,
  cuDNN benchmark mode, and deeper DataLoader prefetching
- local CUDA and local CPU profiles can enable backend-aware `torch.compile`
  defaults, while Apple Silicon stays more conservative
- CPU and Apple Silicon profiles set Torch thread counts and float32 matmul
  precision more deliberately for local training

Open the notebook:

```bash
jupyter notebook main.ipynb
```

The notebook uses the same shared workflow helpers from `main.py`, so script
and notebook runs stay aligned.

Internally, that stable facade now delegates most of the reusable orchestration
to `src/workflows/`, while the notebook remains a thin interactive surface over
the same shared path.

### Google Colab

If you run this project in Colab, add `src/` to the Python import path before
importing project modules:

```python
import sys
sys.path.insert(0, "/content/ENSF617-Probabilistic-Glucose-Forecasting/src")
```

Then install dependencies and run tests:

```bash
pip install -r requirements.txt
pytest tests -q
```

For a Colab GPU runtime, the new profile-based entrypoint is:

```bash
python main.py --device-profile colab-cuda
```

If you only want to verify that the Colab runtime is configured correctly
before downloading data or starting training, run:

```bash
python main.py --device-profile colab-cuda --run-diagnostics-only
```

Model checkpoints can then be loaded directly from notebooks with standard
Lightning APIs:

```python
from models.fused_model import FusedModel

model = FusedModel.load_from_checkpoint("/content/path/to/checkpoint.ckpt")
```

If you want to run the manual smoke script in Colab, execute it from the project
root after dependencies are installed:

```bash
python tests/manual_data_smoke.py
```

### Slurm Examples

CPU allocation:

```bash
python main.py --device-profile slurm-cpu --output-dir /path/to/job_artifacts
```

GPU allocation:

```bash
python main.py --device-profile slurm-cuda --output-dir /path/to/job_artifacts
```

These profiles automatically bias the runtime toward batch-job-friendly
settings such as quieter progress output and worker counts derived from the
Slurm allocation when available.

### Apple Silicon

On Apple Silicon, prefer the MPS-aware profile:

```bash
python main.py --device-profile apple-silicon
```

The runtime diagnostics layer will warn if the chosen settings look mismatched
for MPS, such as pinned host memory or unsupported mixed-precision requests.
Telemetry logs also record MPS memory usage now, so Apple Silicon runs leave
behind more useful device-level artifacts.
The Apple Silicon profile also exposes MPS allocator/fallback controls through
the runtime flags below.

Additional runtime tuning flags are available when you want to push training
performance further:

- `--prefetch-factor`
- `--gradient-clip-val`
- `--accumulate-grad-batches`
- `--strategy`
- `--sync-batchnorm` / `--no-sync-batchnorm`
- `--matmul-precision`
- `--allow-tf32` / `--no-allow-tf32`
- `--cudnn-benchmark` / `--no-cudnn-benchmark`
- `--intraop-threads`
- `--interop-threads`
- `--mps-high-watermark-ratio`
- `--mps-low-watermark-ratio`
- `--enable-mps-fallback` / `--no-enable-mps-fallback`
- `--compile-model` / `--no-compile-model`
- `--compile-mode`
- `--compile-fullgraph`

For a short environment-only throughput check, you can run a benchmark mode
that performs a tiny training run with test/prediction/reporting disabled:

```bash
python main.py --device-profile auto --run-benchmark-only --benchmark-train-batches 10
```

That writes a compact `benchmark_summary.json` with throughput and memory
figures for the active environment.
