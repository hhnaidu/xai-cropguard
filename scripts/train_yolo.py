#!/usr/bin/env python3
"""
YOLOv8 Training Script for Crop Disease Detection
Trains YOLOv8 Small model on multiclass crop disease dataset
Optimized for CPU-only training (use device='0' if GPU available)
"""

from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Train the model
print("Starting YOLOv8 training...")
print("Dataset: 2,447 training + 432 validation images")
print("Classes: 23 crop diseases")
print("Duration: ~2000+ hours on CPU (~100 epochs)")
print("-" * 60)

results = model.train(
    data='data/processed/yolo/data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    patience=20,
    device='cpu',  # CPU training (change to 0 if GPU available)
    name='cropguard_yolo_v1',
    project='runs/detect',
    exist_ok=False,
    verbose=True,
    save=True,
    plots=True,
)

print("\n" + "=" * 60)
print("✅ Training complete!")
print("=" * 60)
print(f"Results saved to: runs/detect/cropguard_yolo_v1/")
print(f"Best weights: runs/detect/cropguard_yolo_v1/weights/best.pt")
print(f"\nNext steps:")
print(f"1. Test predictions: python scripts/predict_yolo.py")
print(f"2. Export to TFLite: python scripts/export_tflite.py")
print(f"3. Deploy on Raspberry Pi with TFLite Interpreter")
