# xai-cropguard

Explainable AI System for Crop Disease Detection

## Model Download

`best.pt` is not included in the repo (too large for GitHub).

Download from release:
[version-2 assets](https://github.com/hhnaidu/xai-cropguard/releases/tag/version-2)

Place the model at:
`trial/cropguard_best_model/best.pt`

See `README_PROGRESS.md` for the live detailed project progress tracker.
See `REPORT.md` for environment setup and installation logs.

## Recommended setup (Conda, tested)

1. Install Miniconda (if not already installed):

```bash
# Download and run installer (non-interactive)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
```

2. Create and activate the project environment (Python 3.11):

```bash
$HOME/miniconda3/bin/conda create -y -n xai-py311 python=3.11
conda activate xai-py311
```

3. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Verify installation:

```bash
python -c "import tensorflow as tf; import cv2; import datasets; import flask; print('All libraries imported successfully')"
```

Notes:
- If you need desktop OpenCV (with GUI support), install it via conda-forge:
  `conda install -n xai-py311 -c conda-forge opencv`
- The exact installed package versions are pinned in `requirements.txt` (created with `pip freeze`).

---

## Alternative (system venv)

If you prefer a system Python venv, create a Python 3.11 venv, then install the requirements. On some systems, Python 3.11 may not be present in default repos; use `pyenv` or install from official packages.
