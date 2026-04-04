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
  --max-epochs 5 \
  --batch-size 128 \
  --num-workers 4 \
  --pin-memory \
  --persistent-workers \
  --precision 16-mixed \
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
  --batch-size 128 \
  --num-workers 1 \
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

Ahh got it — you want a **clean Markdown block ONLY**, no extra formatting, no `id=`, no explanation. Just paste-ready.

Here it is 👇

---
## TALC GPU Cluster Setup (Slurm)

### 1. Connect to TALC

SSH into the cluster:

```bash
ssh your_username@talc.ucalgary.ca
```

Navigate to your working directory:

```bash
cd ~/projects
```

Clone the repository (if not already done):

```bash
git clone https://github.com/danCoder93/ENSF617-Probabilistic-Glucose-Forecasting.git -b danish/dev
cd ENSF617-Probabilistic-Glucose-Forecasting
```

---

### 2. Setup Environment

#### 2.1 Install your environment if not installed (using conda):

##### Create env with Python 3.11 and proper channel order

```bash
conda create -y -n pytorch python=3.11 -c pytorch -c nvidia -c conda-forge
conda activate pytorch
```

##### Enforce strict channel priority for this env

```bash
conda config --env --set channel_priority strict
```

#### 2.2 Load your environment (example using conda):

```bash
source ~/software/init-conda
conda activate pytorch
```

#### 2.3 Install dependencies (if needed):

##### 2.3.1. Core PyTorch (install in sequence)

```bash
conda install -y pytorch -c pytorch
```

```bash
conda install -y pytorch-cuda=12.1 -c nvidia -c pytorch
```

```bash
conda install -y torchvision torchaudio -c pytorch
```

##### 2.3.2. Core Scientific Stack

```bash
conda install -y numpy pandas -c conda-forge
```

```bash
conda install -y scikit-learn matplotlib -c conda-forge
```

##### 2.3.3. Utilities & Logging

```bash
conda install -y tensorboard psutil requests pytest -c conda-forge
```

##### 2.3.4. Visualization

```bash
conda install -y plotly -c conda-forge
```

##### 2.3.5. PyTorch Ecosystem (use pip to avoid solver issues)

```bash
pip install pytorch-lightning torchmetrics torchview transformers tensorflow
```

##### 2.3.6. Verify Installation

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

### 3. Submit the Job

```bash
sbatch run_glucose.slurm
```

Check job status:

```bash
squeue -u $USER
```

---

### 4. Run Quick Debug Jobs

For faster iteration, modify the command inside the script:

```bash
python main.py \
  --device-profile slurm-cuda \
  --batch-size 128 \
  --max-epochs 20 \
  --pin-memory \
  --precision 16-mixed \
  --observability-mode debug \
  --rich-progress-bar \
  --device-stats \
  --enable-activation-stats \
  --output-dir "$SLURM_SUBMIT_DIR/artifacts/slurm_run"
```

---

### 5. Access Logs and Outputs

After the job completes:

* **Console output**: `glucose_t4-<job_id>.out`
* **Artifacts**:

  ```
  artifacts/glucose_<job_id>/
  ```

Includes:

* model checkpoints
* logs
* prediction outputs
* telemetry CSV

---

### 6. Avoid Slow Configurations

Do NOT use these on the cluster:

```bash
--observability-mode debug
--enable-activation-stats
--device-stats
--rich-progress-bar
```

These significantly slow down training due to logging and synchronization overhead.

---

### 7. Resource Optimization

After your first run:

```bash
seff <job_id>
sacct -j <job_id>
```

---

### 8. Scratch Storage

Use fast local storage during jobs:

```bash
/scratch/${SLURM_JOB_ID}
```

* Faster than network filesystem
* Automatically cleaned after job completion

-----

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
