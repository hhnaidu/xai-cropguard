#!/usr/bin/env python3
"""
YOLOv8 Prediction Script - Test trained model on validation images
"""
import sys
from pathlib import Path
from ultralytics import YOLO

def predict_on_val(model_path, val_img_dir, output_dir, conf_threshold=0.25):
    """Run inference on validation images"""
    
    # Load trained model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Run prediction
    print(f"Running prediction on images in: {val_img_dir}")
    results = model.predict(
        source=val_img_dir,
        conf=conf_threshold,
        save=True,
        project=output_dir,
        name='predictions',
        exist_ok=True,
    )
    
    print(f"\n✅ Predictions saved to: {output_dir}/predictions/")
    print(f"Total images processed: {len(results)}")
    
    # Print sample detections
    for i, result in enumerate(results[:3]):
        if result.boxes:
            print(f"\nImage {i+1}: {result.boxes.shape[0]} detections")
            for box in result.boxes:
                class_id = int(box.cls)
                conf = float(box.conf)
                class_name = model.names[class_id]
                print(f"  - {class_name}: {conf:.2f}")

if __name__ == '__main__':
    model_path = 'runs/detect/cropguard_yolo_v1/weights/best.pt'
    val_dir = 'data/processed/yolo/val/images'
    
    predict_on_val(
        model_path=model_path,
        val_img_dir=val_dir,
        output_dir='runs/detect',
        conf_threshold=0.25
    )
