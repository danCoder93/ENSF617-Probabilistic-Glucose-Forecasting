# ENSF617-Probabilistic-Glucose-Forecasting

Hybrid TCN–Transformer (TFT) model for probabilistic blood glucose forecasting from real-world time-series data (CGM, insulin, meals, physiological signals). Captures short- and long-term dependencies with uncertainty-aware predictions; extensible to physics-informed and dynamical system modeling.

## Refactor Notes

The AZT1D data-layer refactor is summarized in
[docs/data_refactor_summary.md](docs/data_refactor_summary.md).

The model-folder refactor is summarized in
[docs/model_refactor_summary.md](docs/model_refactor_summary.md).

The LightningModule integration work in `FusedModel` is summarized in
[docs/lightning_model_integration_summary.md](docs/lightning_model_integration_summary.md).

The reusable Lightning training-wrapper layer in `src/train.py` is summarized in
[docs/train_wrapper_summary.md](docs/train_wrapper_summary.md).

## Testing

### Install dependencies

Local:

```bash
pip install -r requirements.txt
```

### Run the automated data-layer tests

Run the full pytest suite for the refactored data pipeline:

```bash
pytest tests/data -q
```

Run a single test module:

```bash
pytest tests/data/test_datamodule.py -q
```

### Run model and training-wrapper tests

Run the shared config, model, and training-wrapper tests:

```bash
pytest tests/test_config.py tests/test_fused_model.py tests/test_train.py -q
```

### Manual smoke test

For a heavier end-to-end data pipeline check that can download and prepare the
AZT1D dataset, use the manual smoke script:

```bash
python tests/manual_data_smoke.py
```

This is intentionally separate from the pytest suite because it may touch the
network and real filesystem state.

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
pytest tests/data -q
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
