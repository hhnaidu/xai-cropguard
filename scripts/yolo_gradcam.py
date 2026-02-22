#!/usr/bin/env python3
"""
Run YOLOv8 inference and Grad-CAM on the same image.

Example:
  python scripts/yolo_gradcam.py \
    --model trial/cropguard_best_model/best.pt \
    --image trial/test.png \
    --output-dir runs/gradcam
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus, XGradCAM
from ultralytics import YOLO


CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "eigencam": EigenCAM,
    "xgradcam": XGradCAM,
}


def configure_runtime(low_memory: bool) -> None:
    """Limit thread fan-out for RAM-constrained systems."""
    if not low_memory:
        return
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["TORCH_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def cam_quality(cam_map: np.ndarray) -> float:
    """Prefer high-contrast but not over-diffuse CAM maps."""
    cam = np.asarray(cam_map, dtype=np.float32)
    cam = np.clip(cam, 0.0, 1.0)
    p50, p90, p95, p99 = np.percentile(cam, [50, 90, 95, 99])
    active_ratio = float((cam > 0.60).mean())
    contrast = (p95 - p50) + 0.7 * (p99 - p90)
    sparsity_bonus = max(0.0, 1.0 - abs(active_ratio - 0.18))
    return float(contrast + 0.30 * sparsity_bonus)


def is_healthy_class(class_name: str) -> bool:
    return "healthy" in class_name.lower()


def remove_edge_artifacts(cam: np.ndarray, border_percent: float = 0.08) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)
    h, w = cam.shape
    bh = max(1, int(h * border_percent))
    bw = max(1, int(w * border_percent))
    cam_clean = cam.copy()
    cam_clean[:bh, :] = 0.0
    cam_clean[-bh:, :] = 0.0
    cam_clean[:, :bw] = 0.0
    cam_clean[:, -bw:] = 0.0
    vmax = float(cam_clean.max())
    if vmax > 0:
        cam_clean /= vmax
    return cam_clean


def enable_gradients(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(True)


class YOLOOutputWrapper(torch.nn.Module):
    """Expose raw YOLO output tensor to pytorch-grad-cam."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


class YOLOClassTarget:
    """Grad-CAM target: maximize confidence for one class over all anchors."""

    def __init__(self, class_id: int) -> None:
        self.class_id = class_id

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if isinstance(model_output, (list, tuple)):
            model_output = model_output[0]
        if model_output.ndim == 3:
            class_axis = 1
        elif model_output.ndim == 2:
            class_axis = 0
        else:
            raise ValueError(f"Expected 2D/3D output, got shape {tuple(model_output.shape)}")

        channel_idx = 4 + self.class_id
        if channel_idx >= model_output.shape[class_axis]:
            raise ValueError(
                f"class_id={self.class_id} is out of range for output channels={model_output.shape[class_axis]}"
            )

        if model_output.ndim == 3:
            class_scores = model_output[:, channel_idx, :]
        else:
            class_scores = model_output[channel_idx, :]
        return class_scores.max()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 + Grad-CAM inference")
    parser.add_argument("--model", required=True, help="Path to YOLOv8 .pt model")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output-dir", default="runs/gradcam", help="Directory for outputs")
    parser.add_argument("--imgsz", type=int, default=640, help="Square resize size for inference/CAM")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.20, help="YOLO IoU threshold")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cam-method", choices=CAM_METHODS.keys(), default="eigencam")
    parser.add_argument(
        "--target-layer",
        default="-3",
        help="Layer index (e.g. -2) or 'auto' for last Conv2d",
    )
    parser.add_argument(
        "--auto-layer-search",
        action="store_true",
        help="Try fallback layers and choose best CAM contrast (recommended with eigencam)",
    )
    parser.add_argument(
        "--det-index",
        type=int,
        default=0,
        help="Detection index after sorting by confidence (0 = top prediction)",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Force CAM target class id. By default uses class from selected detection.",
    )
    parser.add_argument("--low-memory", action="store_true", help="Reduce thread usage and memory pressure")
    parser.add_argument("--rpi5", action="store_true", help="Apply Raspberry Pi 5 friendly defaults")
    parser.add_argument("--edge-border", type=float, default=0.08, help="CAM border percentage to suppress edge artifacts")
    parser.add_argument("--list-layers", action="store_true", help="Print model layer options and exit")
    return parser.parse_args()


def get_last_conv(model: torch.nn.Module) -> tuple[str, torch.nn.Module]:
    last_name = None
    last_module = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_name = name
            last_module = module
    if last_module is None:
        raise RuntimeError("No Conv2d layers found in model")
    return last_name, last_module


def iter_top_layers(model: torch.nn.Module) -> Iterable[tuple[int, str, torch.nn.Module]]:
    for idx, module in enumerate(model.model):
        yield idx, module.__class__.__name__, module


def resolve_target_layer(
    pytorch_model: torch.nn.Module,
    target_layer_arg: str,
) -> tuple[str, torch.nn.Module]:
    if target_layer_arg == "auto":
        return get_last_conv(pytorch_model)

    try:
        layer_idx = int(target_layer_arg)
    except ValueError as exc:
        raise ValueError("--target-layer must be an int index (e.g. -2) or 'auto'") from exc

    layer = pytorch_model.model[layer_idx]
    layer_name = f"model[{layer_idx}]/{layer.__class__.__name__}"
    return layer_name, layer


def main() -> None:
    args = parse_args()
    if args.rpi5:
        args.device = "cpu"
        args.imgsz = min(args.imgsz, 512)
        args.cam_method = "eigencam"
        args.auto_layer_search = False
        args.low_memory = True

    configure_runtime(args.low_memory)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    infer_model = model
    pytorch_model = model.model
    pytorch_model.eval().to(args.device)
    enable_gradients(pytorch_model)

    if args.list_layers:
        print("Top-level YOLO layers:")
        for idx, cls_name, _ in iter_top_layers(pytorch_model):
            print(f"  [{idx:>2}] {cls_name}")
        conv_name, _ = get_last_conv(pytorch_model)
        print(f"Auto target layer (last Conv2d): {conv_name}")
        return

    image_path = Path(args.image)
    img_bgr_orig = cv2.imread(str(image_path))
    img_bgr = img_bgr_orig
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_bgr = cv2.resize(img_bgr, (args.imgsz, args.imgsz), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = infer_model.predict(
        source=str(image_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        agnostic_nms=True,
        verbose=False,
    )
    result = results[0]

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        raise RuntimeError("No detections found. Lower --conf or try another image.")

    confs = boxes.conf.detach().cpu().numpy()
    clss = boxes.cls.detach().cpu().numpy().astype(int)
    names = [infer_model.names.get(c, str(c)) for c in clss]
    valid_idx = [i for i, cf in enumerate(confs.tolist()) if cf >= args.conf]
    disease_idx = [i for i in valid_idx if not is_healthy_class(names[i])]
    ranked = disease_idx if disease_idx else valid_idx
    if not ranked:
        raise RuntimeError("No detections remained after confidence filtering.")
    ranked = sorted(ranked, key=lambda i: float(confs[i]), reverse=True)
    if args.det_index < 0 or args.det_index >= len(ranked):
        raise IndexError(f"--det-index {args.det_index} out of range for {len(ranked)} detections")
    chosen = int(ranked[args.det_index])

    pred_class = int(boxes.cls[chosen].item())
    class_id = pred_class if args.class_id is None else args.class_id
    class_name = infer_model.names.get(class_id, str(class_id))

    print(f"Selected detection idx={chosen}, conf={confs[chosen]:.4f}, class={pred_class}")
    print(f"CAM target class id={class_id} ({class_name})")

    input_tensor = (
        torch.from_numpy(img_rgb)
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        .div(255.0)
        .to(args.device)
    )

    wrapped_model = YOLOOutputWrapper(pytorch_model)
    layer_candidates = [args.target_layer]
    if args.auto_layer_search:
        layer_candidates.extend(["-3", "-4", "-5", "auto"])

    best = None
    for layer_arg in layer_candidates:
        try:
            layer_name, target_layer = resolve_target_layer(pytorch_model, layer_arg)
            cam_class = CAM_METHODS[args.cam_method]
            with cam_class(model=wrapped_model, target_layers=[target_layer]) as cam:
                if args.cam_method == "eigencam":
                    with torch.no_grad():
                        curr_cam = cam(input_tensor=input_tensor)[0]
                else:
                    targets = [YOLOClassTarget(class_id)]
                    with torch.enable_grad():
                        curr_cam = cam(input_tensor=input_tensor, targets=targets)[0]
            score = cam_quality(curr_cam)
            if best is None or score > best[0]:
                best = (score, layer_name, curr_cam)
        except Exception:
            continue

    if best is None:
        raise RuntimeError("Failed to generate CAM for all candidate layers.")

    _, layer_name, grayscale_cam = best
    grayscale_cam = remove_edge_artifacts(grayscale_cam, border_percent=args.edge_border)
    print(f"Using CAM layer: {layer_name}")
    print(
        f"CAM stats: min={float(grayscale_cam.min()):.3f}, "
        f"max={float(grayscale_cam.max()):.3f}, mean={float(grayscale_cam.mean()):.3f}"
    )

    cam_uint8 = np.uint8(np.clip(grayscale_cam, 0.0, 1.0) * 255)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    cam_rgb = (img_rgb.astype(np.float32) * 0.40 + heatmap_rgb.astype(np.float32) * 0.60).astype(np.uint8)
    cam_bgr = cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR)
    cam_path = out_dir / f"{image_path.stem}_cam.jpg"
    cv2.imwrite(str(cam_path), cam_bgr)

    boxes_xyxy = boxes.xyxy.detach().cpu().numpy()
    orig_h, orig_w = img_bgr_orig.shape[:2]
    detections_img = img_bgr.copy()
    detections_path = out_dir / f"{image_path.stem}_detections.jpg"
    cam_with_boxes = cam_bgr.copy()
    for b, conf, cls in zip(boxes_xyxy, confs.tolist(), clss.tolist()):
        if conf < args.conf:
            continue
        if is_healthy_class(infer_model.names.get(cls, str(cls))):
            continue
        x1 = int((b[0] / max(1, orig_w)) * args.imgsz)
        y1 = int((b[1] / max(1, orig_h)) * args.imgsz)
        x2 = int((b[2] / max(1, orig_w)) * args.imgsz)
        y2 = int((b[3] / max(1, orig_h)) * args.imgsz)
        x1 = max(0, min(args.imgsz - 1, x1))
        y1 = max(0, min(args.imgsz - 1, y1))
        x2 = max(0, min(args.imgsz - 1, x2))
        y2 = max(0, min(args.imgsz - 1, y2))
        label = f"{infer_model.names.get(cls, cls)} {conf:.0%}"
        cv2.rectangle(detections_img, (x1, y1), (x2, y2), (60, 255, 60), 2)
        cv2.putText(detections_img, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 255, 60), 2)
        cv2.rectangle(cam_with_boxes, (x1, y1), (x2, y2), (60, 255, 60), 2)
        cv2.putText(cam_with_boxes, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 255, 60), 2)
    cv2.imwrite(str(detections_path), detections_img)
    cam_boxes_path = out_dir / f"{image_path.stem}_cam_boxes.jpg"
    cv2.imwrite(str(cam_boxes_path), cam_with_boxes)

    panel = np.concatenate([detections_img, cam_with_boxes], axis=1)
    panel_path = out_dir / f"{image_path.stem}_panel.jpg"
    cv2.imwrite(str(panel_path), panel)

    print("\nDetections:")
    for idx, box in enumerate(boxes):
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        xyxy = box.xyxy.squeeze().detach().cpu().numpy().tolist()
        print(f"  {idx:>2}: class={cls} ({infer_model.names.get(cls, cls)}), conf={conf:.4f}, box={xyxy}")

    print("\nSaved files:")
    print(f"  - {detections_path}")
    print(f"  - {cam_path}")
    print(f"  - {cam_boxes_path}")
    print(f"  - {panel_path}")


if __name__ == "__main__":
    main()
