# YOLOv8 Crop Disease Detection Training & Deployment Guide

> Last updated: 2026-02-07  
> This guide covers training, evaluation, and deployment of YOLOv8 for crop disease detection.

## Quick Start

### 1. Ensure Ultralytics is Installed
```bash
# Using conda (faster on Linux)
conda install -n xai-py311 -c conda-forge ultralytics -y

# Or using pip
pip install ultralytics --upgrade
```

### 2. Train the Model (Recommended Parameters)
```bash
cd /home/harsh/xai-cropguard

# Run the training script
/home/harsh/miniconda3/envs/xai-py311/bin/python scripts/train_yolo.py
```

**Expected training time:**
- ~10-15 minutes per epoch on GPU (if available)
- ~1-2 hours per epoch on CPU
- Total: ~16-200 hours for 100 epochs (depends on hardware)

**Alternative (direct CLI):**
```bash
python -m ultralytics.yolo detect train \
  data=data/processed/yolo/data.yaml \
  model=yolov8s.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  patience=20 \
  device=0
```

### 3. Evaluate on Validation Set
```bash
python scripts/predict_yolo.py
```
Results will be saved to: `runs/detect/predictions/`

### 4. Export for Raspberry Pi (TFLite)
```bash
# After training completes
python scripts/export_tflite.py
```
Exported model: `runs/detect/cropguard_yolo_v1/weights/best.tflite`

---

## Dataset Structure

```
data/processed/yolo/
├── data.yaml                 # YOLO config (class names, dataset paths)
├── train/
│   ├── images/              # 2,447 training images
│   └── labels/              # YOLO format labels (.txt)
└── val/
    ├── images/              # 432 validation images
    └── labels/              # YOLO format labels
```

## Model Info

| Property | Value |
|---|---|
| Model | YOLOv8 Small (yolov8s) |
| Input Size | 640×640 pixels |
| Classes | 23 crop diseases + variants |
| Framework | PyTorch → TFLite |
| Target | Raspberry Pi 4/5 + Desktop |

## 23 Disease Classes

```
Crop List: Corn, Tomato, Pepper, Potato, Grape

0.  Corn_Cercospora_Leaf_Spot
1.  Corn_Common_Rust
2.  Corn_Healthy
3.  Corn_Northern_Leaf_Blight
4.  Corn_Streak
5.  Pepper_Bacterial_Spot
6.  Pepper_Cercospora
7.  Pepper_Early_Blight
8.  Pepper_Fusarium
9.  Pepper_Healthy
10. Pepper_Late_Blight
11. Pepper_Leaf_Blight
12. Pepper_Leaf_Curl
13. Pepper_Leaf_Mosaic
14. Pepper_Septoria
15. Tomato_Bacterial_Spot
16. Tomato_Early_Blight
17. Tomato_Fusarium
18. Tomato_Healthy
19. Tomato_Late_Blight
20. Tomato_Leaf_Curl
21. Tomato_Mosaic
22. Tomato_Septoria
```

---

## Training Configuration

### Hyperparameters (Optimized for this dataset)

```python
epochs=100              # Sufficient convergence for 2.4K images
batch=16               # Balanced for CPU/GPU memory
imgsz=640              # Standard YOLO resolution
patience=20            # Early stopping (no improvement for 20 epochs)
device=0               # Auto (GPU if available, else CPU)
```

### Why YOLOv8 Small?

- **vs Nano (yolov8n):** Better accuracy, acceptable speed
- **vs Medium (yolov8m):** Slower training, overkill for Raspberry Pi
- **vs Large (yolov8l):** Too slow, doesn't fit on Pi

---

## TFLite Export for Raspberry Pi

### Export Command
```bash
python scripts/export_tflite.py
```

### Deployment on Raspberry Pi

**Setup (Pi side):**
```bash
# Install TensorFlow Lite runtime
pip install tflite-runtime

# Or full TensorFlow (heavier, ~200MB)
pip install tensorflow
```

**Run inference:**
```python
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Load model
interpreter = Interpreter(model_path="best.tflite")
interpreter.allocate_tensors()

# Prepare input (640x640 RGB image)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_data = np.random.rand(1, 640, 640, 3).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get output
detections = interpreter.get_tensor(output_details[0]['index'])
```

---

## Common Issues & Solutions

### `ModuleNotFoundError: No module named 'ultralytics'`
```bash
# Install with conda (faster)
conda install -n xai-py311 -c conda-forge ultralytics -y

# Or pip (might take 10+ mins)
pip install ultralytics --upgrade
```

### Training is very slow (CPU only)
- This is normal! CPU training takes 1-2 hours per epoch
- Reduce batch size to `8` if out of memory
- Reduce epochs to `50` for faster feedback
- Use GPU if available: `device=0` (auto-detect)

### CUDA/GPU errors
- Install PyTorch with GPU support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- Or use CPU only: `pip install torch torchvision torchaudio`

### Out of memory errors
```python
# In training script, reduce batch size
batch=8  # instead of 16
```

---

## Output Files

### After Training
```
runs/detect/cropguard_yolo_v1/
├── weights/
│   ├── best.pt          # Best model (PyTorch)
│   └── last.pt          # Last epoch (PyTorch)
├── results.csv          # Training metrics
├── confusion_matrix.png
├── results.png          # Train/val loss curves
├── F1_curve.png
└── PR_curve.png
```

### After Prediction
```
runs/detect/predictions/
└── image0.jpg           # Images with detected boxes
```

### After TFLite Export
```
runs/detect/cropguard_yolo_v1/weights/
├── best.pt              # PyTorch weights
├── best.tflite          # TensorFlow Lite model
└── best_full_integer_quant.tflite  # Quantized (optional)
```

---

## Next Steps

- [ ] Train YOLOv8 model (100 epochs)
- [ ] Evaluate on validation set
- [ ] Test on sample farm images
- [ ] Export to TFLite
- [ ] Deploy on Raspberry Pi
- [ ] Integrate with Flask dashboard
- [ ] Collect XAI explanations (GradCAM, LIME)

---

## References

- **Ultralytics YOLOv8:** https://docs.ultralytics.com/
- **TFLite Interpreter:** https://www.tensorflow.org/lite/guide/python
- **YOLO Format:** https://docs.ultralytics.com/datasets/detect/#coco-dataset-format
