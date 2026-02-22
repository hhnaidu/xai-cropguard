#!/usr/bin/env python3
"""
Export YOLOv8 to TensorFlow Lite for Raspberry Pi deployment
"""
from pathlib import Path
from ultralytics import YOLO

def export_to_tflite(model_path, output_dir='models/tflite'):
    """Export trained YOLOv8 model to TFLite format"""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print("Exporting to TensorFlow Lite...")
    export_path = model.export(
        format='tflite',
        imgsz=640,
        half=False,  # Set to True if you have TFLite Delegate for fp16
    )
    
    print(f"\n✅ TFLite model exported successfully!")
    print(f"Output: {export_path}")
    print(f"\nFor Raspberry Pi deployment:")
    print(f"1. Copy {export_path} to Raspberry Pi")
    print(f"2. Use TensorFlow Lite Interpreter to load and run")
    print(f"\nModel properties:")
    print(f"  - Input size: 640×640×3")
    print(f"  - Output: Detection boxes + confidence scores")
    
if __name__ == '__main__':
    model_path = 'runs/detect/cropguard_yolo_v1/weights/best.pt'
    export_to_tflite(model_path=model_path)
