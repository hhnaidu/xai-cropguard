#!/usr/bin/env python3
"""Standalone EigenCAM validator for YOLOv8 models."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from pytorch_grad_cam import EigenCAM
from ultralytics import YOLO

from yolo_gradcam import YOLOOutputWrapper, is_healthy_class, resolve_target_layer


def cam_quality(cam_map: np.ndarray) -> float:
    cam = np.asarray(cam_map, dtype=np.float32)
    cam = np.clip(cam, 0.0, 1.0)
    p50, p90, p95, p99 = np.percentile(cam, [50, 90, 95, 99])
    active_ratio = float((cam > 0.60).mean())
    contrast = (p95 - p50) + 0.7 * (p99 - p90)
    sparsity_bonus = max(0.0, 1.0 - abs(active_ratio - 0.18))
    return float(contrast + 0.30 * sparsity_bonus)


def configure_runtime(low_memory: bool) -> None:
    if not low_memory:
        return
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["TORCH_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


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


def run_eigencam(
    model_path: str,
    image_path: str,
    output_dir: str = "runs/eigencam",
    layer: str = "-3",
    conf: float = 0.35,
    iou: float = 0.20,
    imgsz: int = 640,
    device: str = "cpu",
    low_memory: bool = False,
    auto_layer_search: bool = True,
    edge_border: float = 0.08,
) -> None:
    configure_runtime(low_memory)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(model_path)
    pytorch_model = yolo.model
    pytorch_model.eval().to(device)
    wrapped = YOLOOutputWrapper(pytorch_model)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_640 = cv2.resize(img_rgb, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    input_tensor = torch.from_numpy(img_640).permute(2, 0, 1).float().unsqueeze(0).div(255.0).to(device)

    results = yolo.predict(
        source=image_path,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        agnostic_nms=True,
        verbose=False,
    )
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        print("No detections found. Lower --conf.")
        return

    confs = result.boxes.conf.detach().cpu().numpy()
    clss = result.boxes.cls.detach().cpu().numpy().astype(int)
    names = [yolo.names.get(c, str(c)) for c in clss]
    valid_idx = [i for i, cf in enumerate(confs.tolist()) if cf >= conf]
    disease_idx = [i for i in valid_idx if not is_healthy_class(names[i])]
    ranked = disease_idx if disease_idx else valid_idx
    if not ranked:
        print("No detections after filtering.")
        return
    best_i = int(sorted(ranked, key=lambda i: float(confs[i]), reverse=True)[0])
    disease_name = yolo.names[int(clss[best_i])]
    confidence = float(confs[best_i])
    print(f"Detected: {disease_name} at {confidence:.1%}")

    candidates = [layer, "-3", "-4", "-5", "auto"] if auto_layer_search else [layer]
    chosen = None
    for layer_arg in candidates:
        try:
            layer_name, target_layer = resolve_target_layer(pytorch_model, layer_arg)
            with EigenCAM(model=wrapped, target_layers=[target_layer]) as cam:
                with torch.no_grad():
                    grayscale_cam = cam(input_tensor=input_tensor)[0]
            score = cam_quality(grayscale_cam)
            if chosen is None or score > chosen[0]:
                chosen = (score, layer_name, grayscale_cam)
        except Exception:
            continue

    if chosen is None:
        raise RuntimeError("EigenCAM failed for all candidate layers.")

    _, layer_name, grayscale_cam = chosen
    grayscale_cam = remove_edge_artifacts(grayscale_cam, border_percent=edge_border)
    print(f"Using layer: {layer_name}")
    print(f"CAM min={float(grayscale_cam.min()):.3f}, max={float(grayscale_cam.max()):.3f}, mean={float(grayscale_cam.mean()):.3f}")

    cam_uint8 = np.uint8(np.clip(grayscale_cam, 0.0, 1.0) * 255)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    cam_overlay = (img_640.astype(np.float32) * 0.40 + heatmap_rgb.astype(np.float32) * 0.60).astype(np.uint8)
    cam_bgr = cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR)

    raw_heatmap = cv2.applyColorMap(np.uint8(np.clip(grayscale_cam, 0.0, 1.0) * 255), cv2.COLORMAP_JET)

    boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
    for b, cf, cls in zip(boxes_xyxy, confs.tolist(), clss.tolist()):
        if cf < conf:
            continue
        if is_healthy_class(yolo.names.get(int(cls), str(int(cls)))):
            continue
        x1 = int((b[0] / max(1, orig_w)) * imgsz)
        y1 = int((b[1] / max(1, orig_h)) * imgsz)
        x2 = int((b[2] / max(1, orig_w)) * imgsz)
        y2 = int((b[3] / max(1, orig_h)) * imgsz)
        x1 = max(0, min(imgsz - 1, x1))
        y1 = max(0, min(imgsz - 1, y1))
        x2 = max(0, min(imgsz - 1, x2))
        y2 = max(0, min(imgsz - 1, y2))
        label = f"{yolo.names[int(cls)]} {float(cf):.0%}"
        cv2.rectangle(cam_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(cam_bgr, label, (x1, max(y1 - 8, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)

    stem = Path(image_path).stem
    out_overlay = out_dir / f"{stem}_eigencam.jpg"
    out_raw = out_dir / f"{stem}_raw_heatmap.jpg"
    out_gray = out_dir / f"{stem}_grayscale_cam.npy"

    cv2.imwrite(str(out_overlay), cam_bgr)
    cv2.imwrite(str(out_raw), raw_heatmap)
    np.save(out_gray, grayscale_cam)

    print("Saved:")
    print(f"  - {out_overlay}")
    print(f"  - {out_raw}")
    print(f"  - {out_gray}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test EigenCAM on YOLOv8")
    parser.add_argument("--model", required=True, help="Path to model .pt")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output-dir", default="runs/eigencam")
    parser.add_argument("--layer", default="-3", help="Target layer index or 'auto'")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--rpi5", action="store_true")
    parser.add_argument("--edge-border", type=float, default=0.08)
    parser.add_argument("--no-auto-layer-search", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.rpi5:
        args.device = "cpu"
        args.imgsz = min(args.imgsz, 512)
        args.low_memory = True
        args.no_auto_layer_search = True

    run_eigencam(
        model_path=args.model,
        image_path=args.image,
        output_dir=args.output_dir,
        layer=args.layer,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        low_memory=args.low_memory,
        auto_layer_search=not args.no_auto_layer_search,
        edge_border=args.edge_border,
    )
