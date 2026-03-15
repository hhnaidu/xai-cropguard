# Model Registry

## Overview
XAI-CropGuard uses a centralized model registry (`models/manifest.json`) to manage all models.

## Current Status (2026-03-15)

### Active Models
- **yolo_primary**: `models_archive/cropguard_best_model.zip`
  - 23-class YOLO detection + classification
  - Primary model used in pipeline
  - Labels: `models/yolo_labels.txt`

### Inactive Models
- **yolo_alt**: `hhnaidu.pt.zip`
  - Alternative YOLO model
  - Kept for comparison/fallback
  - Not currently used

### Disabled Models
- **Keras classifiers**: Disabled pending class label verification
  - Will be enabled once labels are extracted and confirmed

## Usage
```python
from scripts.model_loader import ModelRegistry

registry = ModelRegistry()

model_name, model = registry.get_primary_model()
labels = registry.get_class_labels("yolo_primary")

all_models = registry.list_models()
active_only = registry.list_models(status_filter="active")
```

## Class Labels
23 classes covering tomato, pepper, potato, corn, and apple diseases.

See: `models/yolo_labels.txt` for the complete list.

## Next Steps
1. Extract and verify Keras model class labels
2. Enable Keras models in manifest
3. Implement ensemble prediction (YOLO + Keras)
4. Integrate model_loader into main pipeline
