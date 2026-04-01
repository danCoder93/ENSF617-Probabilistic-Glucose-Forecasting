# ENSF617-Probabilistic-Glucose-Forecasting

Hybrid TCN–Transformer (TFT) model for probabilistic blood glucose forecasting from real-world time-series data (CGM, insulin, meals, physiological signals). Captures short- and long-term dependencies with uncertainty-aware predictions; extensible to physics-informed and dynamical system modeling.

## Refactor Notes

The AZT1D data-layer refactor is summarized in
[docs/data_refactor_summary.md](docs/data_refactor_summary.md).

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
sys.path.append("/content/ENSF617-Probabilistic-Glucose-Forecasting/src")
```

Then install dependencies and run tests:
```bash
pip install -r requirements.txt
pytest tests/data -q
```

If you want to run the manual smoke script in Colab, execute it from the project
root after dependencies are installed:
```bash
python tests/manual_data_smoke.py
```
