# Project Report & Setup Log ✅

This file will be updated with install notes, setup steps, test results, and troubleshooting logs.

## Setup actions

- Created `requirements.txt` with main dependencies (TensorFlow, OpenCV, Flask, etc.).
- Installed packages and saved pinned versions back to `requirements.txt` using `pip freeze` (see log below).

## Verification

Run the following to verify install:

```bash
python -c "import tensorflow as tf; import cv2; import datasets; import flask; print('All libraries imported successfully')"
```

If any import fails, list the error here and do not proceed until fixed.

---

## Logs

- 2026-02-03: Created `requirements.txt` and this `REPORT.md` (initial setup).
- 2026-02-03: Attempted package installation in the project's Python 3.13 venv; failed due to several incompatible wheels (TensorFlow 2.15, OpenCV, scikit-learn). Recommended actions:
  - Create/use a Python 3.11 virtual environment to install `tensorflow==2.15.0`, `opencv-python==4.8.1`, and `scikit-learn==1.3.0`.

- 2026-02-03: Resolved by creating a **Miniconda** environment `xai-py311` with Python 3.11 and installing dependencies there. Notes:
  - TensorFlow 2.15.0, tf-keras-vis, numpy, pandas, scikit-learn, and other packages were installed via `pip` successfully into the conda env.
  - `opencv-python` was not available as a compatible wheel; `opencv-python-headless` (4.11.0.86) was installed via `pip` to satisfy image-processing requirements. If GUI OpenCV is needed, install via `conda install -n xai-py311 -c conda-forge opencv`.
  - Verification command:

    ```bash
    /home/harsh/miniconda3/envs/xai-py311/bin/python -c "import tensorflow as tf; import cv2; import datasets; import flask; print('All libraries imported successfully')"
    ```

    Result: "All libraries imported successfully" (TensorFlow reported GPU not found — expected on this machine).

  - Exact installed package versions were saved to `requirements.txt` using `pip freeze`.

- 2026-02-03: Cleaned and simplified `requirements.txt` to keep only direct project dependencies; backup of the full frozen list saved as `requirements-full.txt`.

## Notes / ToDo

- Add instructions for creating a virtual environment (recommended)
- Add troubleshooting steps for common install errors (GPU drivers, incompatible packages)
- Keep updating this file with any environment or dataset changes
- **Tip:** If someone runs into GPU driver or CUDA compatibility problems, use the CPU-only TensorFlow build:

  ```bash
  pip install tensorflow-cpu==2.15.0
  ```

  Or replace the `tensorflow==2.15.0` line in `requirements.txt` with `tensorflow-cpu==2.15.0` to pin the CPU-only build for the project.

---

## YOLOv8 Training (2026-02-07)

### Dataset Preparation
- **Files created:**
  - `data/processed/yolo/data.yaml` — YOLO dataset config with 23 crop disease classes
  - `scripts/split_yolo_data.py` — Fast train/val splitter (no heavy transforms)
  
- **Dataset split:**
  - Training: 2,447 images (85%)
  - Validation: 432 images (15%)
  - Classes: 23 (Corn, Tomato, Pepper, Potato, Grape diseases + healthy)

- **Training command (optimized for this dataset):**
  ```bash
  cd /home/harsh/xai-cropguard
  /home/harsh/miniconda3/envs/xai-py311/bin/python scripts/train_yolo.py
  ```
  
- **Training parameters chosen:**
  - Model: YOLOv8 Small (yolov8s.pt) — best accuracy/speed tradeoff for 2.4K images
  - Image size: 640×640 — standard YOLO input
  - Batch size: 16 — reasonable for CPU/single GPU
  - Epochs: 100 — sufficient for convergence on this dataset size
  - Patience: 20 — early stopping if no improvement for 20 epochs
  - Device: auto-detect (GPU if available, fallback to CPU)

- **Expected outputs:**
  - Weights: `runs/detect/cropguard_yolo_v1/weights/best.pt`
  - Metrics: `runs/detect/cropguard_yolo_v1/results.csv` (loss, mAP, etc.)
  - Plots: `runs/detect/cropguard_yolo_v1/*.png` (training curves)

### Prediction (after training)
```bash
/home/harsh/miniconda3/envs/xai-py311/bin/python scripts/predict_yolo.py
```
Outputs predictions to: `runs/detect/predictions/`
