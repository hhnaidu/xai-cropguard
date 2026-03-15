"""
Unified Model Loader for XAI-CropGuard
Loads models from registry, handles .zip extraction, caches models
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed - YOLO models unavailable")


class ModelRegistry:
    """Manages models from manifest.json"""

    def __init__(self, manifest_path: str = "models/manifest.json"):
        self.manifest_path = Path(manifest_path)

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with self.manifest_path.open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self._model_cache: Dict[str, object] = {}
        self._label_cache: Dict[str, List[str]] = {}
        self._temp_dir: str | None = None

        print("Model Registry initialized")
        print(f"  Version: {self.manifest.get('version', 'unknown')}")
        print(f"  Updated: {self.manifest.get('updated', 'unknown')}")

    def list_models(self, status_filter: Optional[str] = None) -> List[str]:
        models = []
        for name, info in self.manifest["models"].items():
            if status_filter:
                if info.get("status") == status_filter:
                    models.append(name)
            else:
                models.append(name)
        return models

    def get_model_info(self, model_name: str) -> Dict:
        if model_name not in self.manifest["models"]:
            raise ValueError(f"Model '{model_name}' not found in manifest")
        return self.manifest["models"][model_name]

    def _resolve_model_path(self, model_name: str) -> Path:
        info = self.get_model_info(model_name)
        path = Path(info["path"])

        if path.suffix != ".zip":
            return path

        if self._temp_dir is None:
            self._temp_dir = tempfile.mkdtemp(prefix="cropguard_models_")

        extracted_dir = Path(self._temp_dir)
        extracted_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(path, "r") as zip_ref:
            pt_files = [f for f in zip_ref.namelist() if f.endswith(".pt")]
            if not pt_files:
                raise ValueError(f"No .pt file found in {path}")

            pt_file = pt_files[0]
            out_path = extracted_dir / Path(pt_file).name

            if not out_path.exists():
                print(f"Extracting {path.name}...")
                zip_ref.extract(pt_file, extracted_dir)
                extracted = extracted_dir / pt_file
                if extracted != out_path:
                    extracted.rename(out_path)

        return out_path

    def load_model(self, model_name: str, force_reload: bool = False):
        if not force_reload and model_name in self._model_cache:
            print(f"Using cached model: {model_name}")
            return self._model_cache[model_name]

        info = self.get_model_info(model_name)
        if info.get("status") == "disabled":
            raise ValueError(f"Model '{model_name}' is disabled: {info.get('reason', 'No reason given')}")

        model_type = info["type"]
        if model_type == "yolov8":
            if not YOLO_AVAILABLE:
                raise ImportError("ultralytics package required for YOLO models")

            model_path = self._resolve_model_path(model_name)
            print(f"Loading YOLO model: {model_name}")
            print(f"  Path: {model_path}")
            model = YOLO(str(model_path))
            print("  Loaded successfully")
        elif model_type == "keras":
            raise NotImplementedError("Keras models not yet enabled")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self._model_cache[model_name] = model
        return model

    def get_class_labels(self, model_name: str) -> List[str]:
        if model_name in self._label_cache:
            return self._label_cache[model_name]

        info = self.get_model_info(model_name)
        labels_file = info.get("labels_file")
        if not labels_file:
            raise ValueError(f"No labels_file specified for model '{model_name}'")

        labels_path = Path(labels_file)
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        with labels_path.open("r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]

        expected_count = info.get("num_classes")
        if expected_count and len(labels) != expected_count:
            print(f"Warning: Expected {expected_count} classes, found {len(labels)}")

        self._label_cache[model_name] = labels
        return labels

    def get_primary_model(self) -> Tuple[str, object]:
        active_name = self.manifest["pipeline"]["active_model"]
        model = self.load_model(active_name)
        return active_name, model

    def cleanup(self) -> None:
        if self._temp_dir and Path(self._temp_dir).exists():
            shutil.rmtree(self._temp_dir)
            print(f"Cleaned up temp directory: {self._temp_dir}")

    def __del__(self):
        self.cleanup()


def test_registry() -> None:
    print("=" * 70)
    print("MODEL REGISTRY TEST")
    print("=" * 70)

    registry = ModelRegistry()

    print("\nMODELS IN REGISTRY:")
    print("-" * 70)

    all_models = registry.list_models()
    for model_name in all_models:
        info = registry.get_model_info(model_name)
        status = info.get("status", "unknown")
        model_type = info.get("type", "unknown")

        print(f"\n{model_name}:")
        print(f"  Type: {model_type}")
        print(f"  Status: {status}")
        print(f"  Path: {info.get('path', 'N/A')}")

        if status in {"active", "inactive"}:
            try:
                labels = registry.get_class_labels(model_name)
                print(f"  Classes: {len(labels)}")
                print(f"  Sample: {labels[:3]}...")
            except Exception as exc:
                print(f"  Labels error: {exc}")

    print("\n" + "=" * 70)
    print("PRIMARY MODEL")
    print("=" * 70)

    try:
        model_name, model = registry.get_primary_model()
        print(f"Primary model loaded: {model_name}")
        print(f"  Type: {type(model).__name__}")

        labels = registry.get_class_labels(model_name)
        print(f"  Classes: {len(labels)}")
        print("\n  All class labels:")
        for i, label in enumerate(labels):
            print(f"    {i:2d}. {label}")
    except Exception as exc:
        print(f"Error loading primary model: {exc}")

    print("\n" + "=" * 70)
    print("Registry test complete")
    print("=" * 70)

    registry.cleanup()


if __name__ == "__main__":
    test_registry()
