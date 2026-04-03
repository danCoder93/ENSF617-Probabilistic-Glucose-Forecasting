# Running the Probabilistic Glucose Forecasting Code

This guide provides step-by-step instructions for running the ENSF617 Probabilistic Glucose Forecasting codebase on Google Colab with CUDA acceleration and locally on Apple Silicon Macs. The codebase includes automatic environment detection and device profiles for seamless setup across platforms.

## Prerequisites

- Python 3.8+
- Git (for cloning the repository)
- For Colab: Google account with access to Google Colab
- For Apple Silicon: macOS with M-series chip (M1/M2/M3)

## Google Colab Setup (CUDA)

### 1. Open in Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "File" → "Open notebook"
3. Select "GitHub" tab
4. Enter the repository URL: `https://github.com/your-org/ENSF617-Probabilistic-Glucose-Forecasting`
5. Open `main.ipynb` or create a new notebook

Alternatively, clone the repository:

```bash
!git clone https://github.com/danCoder93/ENSF617-Probabilistic-Glucose-Forecasting.git -b danish/dev
%cd ENSF617-Probabilistic-Glucose-Forecasting
```

### 2. Enable GPU

1. Click "Runtime" → "Change runtime type"
2. Select "Hardware accelerator" → "GPU"
3. Click "Save"

### 3. Install Dependencies

```bash
!pip install -r requirements.txt
```

The codebase will automatically detect Colab and use the `colab-cuda` profile if GPU is available, or `colab-cpu` otherwise.

### 4. Run the Code

#### Option A: Using the Notebook (Recommended)

Open `main.ipynb` and run the cells in order. The notebook provides an interactive interface for configuration and training.

#### Option B: Using the CLI

```bash
!python main.py --max-epochs 5 --batch-size 32
```

For custom configuration:

```bash
!python main.py \
  --device-profile colab-cuda \
  --max-epochs 10 \
  --batch-size 64 \
  --observability-mode baseline
```

### 5. Access Logs and Telemetry

In Colab, artifacts are saved in the runtime's temporary storage. To persist them:

1. Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

1. Run with output directory on Drive:

```bash
!python main.py \
  --device-profile colab-cuda \
  --max-epochs 10 \
  --batch-size 64 \
  --observability-mode debug \
  --rich-progress-bar \
  --device-stats \
  --enable-activation-stats \
  --output-dir /content/drive/MyDrive/ENSF617/artifacts
```

#### Viewing Logs

- **Console output**: Limited in Colab; use progress bars and printed summaries
- **Text logs**: `artifacts/main_run/run.log` (download or view in Colab file browser)
- **Telemetry CSV**: `artifacts/main_run/telemetry.csv` (download and open in Colab or locally)

- **TensorBoard**: Run in Colab cell:

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/ENSF617/artifacts/main_run/logs
```

#### Colab File Browser

- Click the folder icon in the left sidebar
- Navigate to `artifacts/main_run/`
- Right-click files to download

## Local Apple Silicon Setup (MPS)

### 1. Clone Repository

```bash
git clone https://github.com/your-org/ENSF617-Probabilistic-Glucose-Forecasting.git
cd ENSF617-Probabilistic-Glucose-Forecasting
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

For Apple Silicon, PyTorch with MPS support is included in recent versions. If you encounter MPS issues, reinstall PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Run the Code

The codebase automatically detects Apple Silicon and applies the `apple-silicon` profile.

#### Option A: Using the Notebook

```bash
jupyter notebook main.ipynb
```

Or with JupyterLab:

```bash
jupyter lab main.ipynb
```

#### Option B: Using the CLI

```bash
python main.py --max-epochs 5 --batch-size 32
```

For explicit profile selection:

```bash
python main.py \
  --device-profile apple-silicon \
  --max-epochs 5 \
  --batch-size 64 \
  --observability-mode debug \
  --rich-progress-bar \
  --device-stats \
  --enable-activation-stats
```

### 4. Access Logs and Telemetry

Artifacts are saved locally in `artifacts/main_run/`.

#### Viewing Logs

- **Console output**: Rich progress bars and summaries (more visible than in Colab)
- **Text logs**: `artifacts/main_run/run.log` (open with any text editor)
- **Telemetry CSV**: `artifacts/main_run/telemetry.csv` (open with Excel, pandas, etc.)
- **TensorBoard**:

```bash
tensorboard --logdir artifacts/main_run/logs
```

Then open <http://localhost:6006> in your browser

#### File Locations

- Run summary: `artifacts/main_run/run_summary.json`
- Checkpoints: `artifacts/main_run/checkpoints/`
- Reports: `artifacts/main_run/reports/` (HTML files)
- Model visualizations: `artifacts/main_run/model_viz/`

## Understanding Logs and Telemetry

### Run Summary (JSON)

Contains complete run metadata. Read with:

```python
import json
with open('artifacts/main_run/run_summary.json') as f:
    summary = json.load(f)
print(summary['fit_artifacts'])
```

### Telemetry CSV

Step-by-step metrics including loss, validation loss, learning rate, and system stats.

```python
import pandas as pd
telemetry = pd.read_csv('artifacts/main_run/telemetry.csv')
telemetry.plot(x='step', y=['loss', 'val_loss'])
```

### Text Logs

Plain-text file with lifecycle messages and errors.

### Prediction Tables

- `test_predictions.csv`: Human-readable predictions with quantiles
- `test_predictions.pt`: Raw PyTorch tensors for programmatic analysis

## Debugging and Troubleshooting

### Enable More Logging

Use debug mode for additional diagnostics:

```bash
python main.py --observability-mode debug --max-epochs 5
```

Or trace mode (heavier):

```bash
python main.py --observability-mode trace --max-epochs 2
```

### Common Issues

#### Colab: Out of Memory

- Reduce batch size: `--batch-size 16`
- Disable expensive features: `--disable-torchview --disable-parameter-histograms`

#### Apple Silicon: MPS Fallback

- Some operations may fall back to CPU automatically
- Check logs for "MPS fallback" messages

#### Slow Training

- Enable profiler: `--enable-profiler --profiler-type pytorch`
- Check telemetry for bottlenecks

### Data Download

The code downloads data automatically on first run. For custom data location:

```bash
python main.py --raw-dir /path/to/data/raw
```

## Additional Resources

- [Repository README](README.md) - Quickstart and overview
- [Repository Primer](docs/repository_primer.md) - Detailed system architecture
- [Current Architecture](docs/current_architecture.md) - Code structure guide

## Expected Runtime

- **Colab (GPU)**: ~5-10 minutes for 5 epochs with default settings
- **Apple Silicon**: ~10-15 minutes for 5 epochs (varies by M-chip generation)
- **CPU-only**: ~20-30 minutes for 5 epochs

Adjust `--max-epochs` and `--batch-size` based on your hardware and time constraints.
