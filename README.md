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

This entrypoint will:

- add `src/` to the import path
- download and preprocess AZT1D automatically if the processed CSV is missing
- train the fused model
- run held-out test evaluation when test windows exist
- optionally generate detailed held-out evaluation, prediction exports, and
  report artifacts under `artifacts/main_run/`

Open the notebook:

```bash
jupyter notebook main.ipynb
```

The notebook uses the same shared workflow helpers from `main.py`, so script
and notebook runs stay aligned.

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
