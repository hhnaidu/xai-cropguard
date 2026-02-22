#!/usr/bin/env python3
"""Full XAI CropGuard pipeline: YOLO detection + CAM + severity + recommendation."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from textwrap import wrap

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from recommendations import get_recommendation
from severity import compute_severity, draw_severity_bar, get_infected_mask
from scripts.config import DEFAULT_PIPELINE_OUTPUT_DIR, cleanup_pipeline_results
from yolo_gradcam import CAM_METHODS, YOLOClassTarget, YOLOOutputWrapper, enable_gradients, resolve_target_layer

PHONE_STREAM_URL = "http://10.88.29.99:8080/video"


def configure_low_memory(low_memory: bool) -> None:
    """Reduce thread fan-out to lower RAM pressure on small laptops."""
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


def refine_cam_for_display(
    grayscale_cam: np.ndarray,
    boxes: list[list[float]],
    target_size: tuple[int, int],
) -> np.ndarray:
    """Make CAM overlays tighter around likely lesion regions."""
    cam = cv2.resize(np.clip(grayscale_cam, 0.0, 1.0), target_size, interpolation=cv2.INTER_LINEAR)
    low, high = np.percentile(cam, [70, 99])
    if high > low + 1e-6:
        cam = np.clip((cam - low) / (high - low), 0.0, 1.0)
    cam = np.power(cam, 1.35)

    if boxes:
        mask = np.zeros_like(cam, dtype=np.float32)
        for xyxy in boxes:
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            pad = int(0.08 * max(bw, bh))
            xa = max(0, x1 - pad)
            ya = max(0, y1 - pad)
            xb = min(target_size[0] - 1, x2 + pad)
            yb = min(target_size[1] - 1, y2 + pad)
            cv2.rectangle(mask, (xa, ya), (xb, yb), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=11, sigmaY=11)
        mask = np.clip(mask, 0.15, 1.0)
        cam = cam * mask

    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    return cam


def build_vivid_cam_overlay(img_rgb: np.ndarray, cam_map: np.ndarray) -> np.ndarray:
    """Stronger CAM overlay to avoid washed-out heatmaps."""
    cam_uint8 = np.uint8(np.clip(cam_map, 0.0, 1.0) * 255)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return (img_rgb.astype(np.float32) * 0.40 + heatmap_rgb.astype(np.float32) * 0.60).astype(np.uint8)


def remove_edge_artifacts(cam: np.ndarray, border_percent: float = 0.08) -> np.ndarray:
    """Suppress border activation artifacts from CAM maps."""
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


def capture_from_phone(stream_url: str = PHONE_STREAM_URL, countdown: int = 5) -> np.ndarray:
    print(f"Connecting to phone camera at {stream_url}")
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot connect to {stream_url}\n"
            f"Check: 1) Phone IP correct  "
            f"2) Same WiFi  "
            f"3) IP Webcam app running"
        )

    print(f"Connected! Capturing in {countdown} seconds...")
    print("Point phone at the leaf NOW")
    for i in range(countdown, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Failed to capture frame from phone")

    print(f"Captured! Frame size: {frame.shape}")
    return frame


def is_healthy_class(class_name: str) -> bool:
    return "healthy" in class_name.lower()


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    denom = area_a + area_b - inter + 1e-9
    return float(inter / denom)


def nms_indices(xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    if len(xyxy) == 0:
        return []
    order = np.argsort(-scores)
    keep: list[int] = []
    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        remaining = []
        for j in order[1:]:
            if _iou_xyxy(xyxy[i], xyxy[int(j)]) < iou_thresh:
                remaining.append(int(j))
        order = np.array(remaining, dtype=np.int64)
    return keep


def select_boxes(
    boxes_xyxyn: np.ndarray,
    confs: np.ndarray,
    clss: np.ndarray,
    class_names: list[str],
    conf_thresh: float,
    min_box_area_ratio: float,
    post_nms_iou: float,
    prefer_disease: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(boxes_xyxyn) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

    boxes_xyxyn = boxes_xyxyn.astype(np.float32)
    confs = confs.astype(np.float32)
    clss = clss.astype(np.int64)

    widths = np.clip(boxes_xyxyn[:, 2] - boxes_xyxyn[:, 0], 0.0, 1.0)
    heights = np.clip(boxes_xyxyn[:, 3] - boxes_xyxyn[:, 1], 0.0, 1.0)
    areas = widths * heights

    keep = (confs >= conf_thresh) & (areas >= min_box_area_ratio)
    boxes_xyxyn = boxes_xyxyn[keep]
    confs = confs[keep]
    clss = clss[keep]
    if len(boxes_xyxyn) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

    nms_keep = nms_indices(boxes_xyxyn, confs, post_nms_iou)
    boxes_xyxyn = boxes_xyxyn[nms_keep]
    confs = confs[nms_keep]
    clss = clss[nms_keep]

    if prefer_disease:
        disease_mask = np.array([not is_healthy_class(class_names[c]) for c in clss], dtype=bool)
        if disease_mask.any():
            boxes_xyxyn = boxes_xyxyn[disease_mask]
            confs = confs[disease_mask]
            clss = clss[disease_mask]

    return boxes_xyxyn, confs, clss


def run_cam(
    wrapped_model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_layer: torch.nn.Module,
    cam_method: str,
    class_id: int,
) -> np.ndarray:
    cam_class = CAM_METHODS[cam_method]
    with cam_class(model=wrapped_model, target_layers=[target_layer]) as cam:
        if cam_method == "eigencam":
            with torch.no_grad():
                return cam(input_tensor=input_tensor)[0]
        targets = [YOLOClassTarget(class_id)]
        with torch.enable_grad():
            return cam(input_tensor=input_tensor, targets=targets)[0]


def pick_best_eigencam_layer(
    pytorch_model: torch.nn.Module,
    wrapped_model: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_id: int,
    candidates: list[str],
) -> tuple[str, torch.nn.Module, np.ndarray]:
    best = None
    for layer_arg in candidates:
        try:
            layer_name, layer = resolve_target_layer(pytorch_model, layer_arg)
            cam = run_cam(wrapped_model, input_tensor, layer, "eigencam", class_id)
            score = cam_quality(cam)
            if best is None or score > best[0]:
                best = (score, layer_name, layer, cam)
        except Exception:
            continue

    if best is None:
        raise RuntimeError("Failed to compute EigenCAM for candidate layers.")
    return best[1], best[2], best[3]


def build_output_panel(
    img_bgr: np.ndarray,
    grayscale_cam: np.ndarray,
    disease_name: str,
    confidence: float,
    severity_score: float,
    severity_level: str,
    recommendation: str,
    boxes: list[list[float]],
) -> np.ndarray:
    """Create final farmer-facing panel image."""
    target_size = (640, 640)
    img_r = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)

    cam_r = refine_cam_for_display(grayscale_cam, boxes, target_size)
    cam_uint8 = np.uint8(cam_r * 255)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img_r, 0.58, heatmap, 0.42, 0)

    mask = get_infected_mask(grayscale_cam, target_size)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 1000]
    if contours:
        cv2.drawContours(blended, contours, -1, (255, 255, 0), 2)

    for xyxy in boxes:
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cv2.rectangle(blended, (x1, y1), (x2, y2), (80, 255, 80), 2)

    blended = draw_severity_bar(blended, severity_score, severity_level, x=12, y=12, width=240)

    panel_h = 200
    panel = np.full((panel_h, target_size[0], 3), (24, 24, 24), dtype=np.uint8)
    sev_color = {"Low": (0, 200, 0), "Medium": (0, 165, 255), "High": (0, 0, 255)}.get(
        severity_level, (220, 220, 220)
    )

    cv2.putText(panel, f"Disease: {disease_name}", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (120, 255, 120), 2)
    cv2.putText(panel, f"Confidence: {confidence:.1%}", (14, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 1)
    severity_y = 98
    rec_y = 128
    lines_start = 152

    cv2.putText(
        panel,
        f"Severity: {severity_level} ({severity_score:.1f}%)",
        (14, severity_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        sev_color,
        2,
    )
    cv2.putText(panel, "Recommendation:", (14, rec_y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (180, 200, 255), 1)

    lines = wrap(recommendation, width=82)[:3]
    for idx, line in enumerate(lines):
        y = lines_start + idx * 20
        cv2.putText(panel, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)

    return np.vstack([blended, panel])


def build_analysis_panel(
    img_rgb: np.ndarray,
    grayscale_cam: np.ndarray,
    cam_overlay_bgr: np.ndarray,
    disease_name: str,
    confidence: float,
    severity_level: str,
) -> np.ndarray:
    """Create a report-friendly 3-panel explainability figure."""
    size = (640, 640)
    img_a = cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA)
    cam_r = cv2.resize(np.clip(grayscale_cam, 0.0, 1.0), size, interpolation=cv2.INTER_LINEAR)
    raw_heatmap = cv2.applyColorMap(np.uint8(cam_r * 255), cv2.COLORMAP_JET)
    overlay = cv2.resize(cam_overlay_bgr, size, interpolation=cv2.INTER_AREA)

    h = 82
    header = np.full((h, size[0] * 3, 3), (22, 22, 22), dtype=np.uint8)
    cv2.putText(header, "Original Leaf", (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (230, 230, 230), 2)
    cv2.putText(header, "Raw EigenCAM", (size[0] + 18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (230, 230, 230), 2)
    cv2.putText(header, "Overlay + Detection", (2 * size[0] + 18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (230, 230, 230), 2)

    stats = f"min={float(cam_r.min()):.2f}  max={float(cam_r.max()):.2f}  mean={float(cam_r.mean()):.2f}"
    footer = np.full((70, size[0] * 3, 3), (22, 22, 22), dtype=np.uint8)
    cv2.putText(
        footer,
        f"{disease_name} | Conf: {confidence:.1%} | Severity: {severity_level} | {stats}",
        (14, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (180, 245, 180),
        2,
    )

    strip = np.hstack([img_a[:, :, ::-1], raw_heatmap, overlay])
    return np.vstack([header, strip, footer])


def run_pipeline(
    model_path: str,
    image_path: str,
    preloaded_model=None,
    output_dir: str = str(DEFAULT_PIPELINE_OUTPUT_DIR),
    conf_thresh: float = 0.45,
    iou_thresh: float = 0.15,
    imgsz: int = 640,
    layer: str = "-3",
    cam_method: str = "eigencam",
    auto_layer_search: bool = True,
    low_memory: bool = False,
    device: str = "cpu",
    min_box_area_ratio: float = 0.002,
    post_nms_iou: float = 0.20,
    prefer_disease: bool = True,
    edge_border_percent: float = 0.08,
) -> dict | tuple[None, dict]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_stem = Path(image_path).stem

    configure_low_memory(low_memory)

    infer_model = preloaded_model if preloaded_model is not None else YOLO(model_path)
    pytorch_model = infer_model.model
    pytorch_model.eval().to(device)
    enable_gradients(pytorch_model)

    img_bgr_orig = cv2.imread(image_path)
    if img_bgr_orig is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_bgr = cv2.resize(img_bgr_orig, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    names_dict = infer_model.names if isinstance(infer_model.names, dict) else {i: n for i, n in enumerate(infer_model.names)}
    max_cls = int(max(names_dict.keys())) if names_dict else 0
    class_names = [names_dict.get(i, str(i)) for i in range(max_cls + 1)]

    results = infer_model.predict(
        source=image_path,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=imgsz,
        device=device,
        agnostic_nms=True,
        verbose=False,
    )
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None, {
            "error": "No plant detected. Please point camera directly at a leaf and try again."
        }

    raw_confs = result.boxes.conf.detach().cpu().numpy()
    if raw_confs.size == 0 or float(raw_confs.max()) < 0.55:
        return None, {
            "error": (
                "No crop disease detected with sufficient confidence. "
                "Ensure camera is pointing directly at a plant leaf in good lighting."
            )
        }

    boxes_xyxy_orig = result.boxes.xyxy.detach().cpu().numpy()
    boxes_xyxyn = boxes_xyxy_orig.copy()
    boxes_xyxyn[:, [0, 2]] /= max(1, img_bgr_orig.shape[1])
    boxes_xyxyn[:, [1, 3]] /= max(1, img_bgr_orig.shape[0])
    confs = result.boxes.conf.detach().cpu().numpy()
    clss = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
    boxes_xyxyn, confs, clss = select_boxes(
        boxes_xyxyn=boxes_xyxyn,
        confs=confs,
        clss=clss,
        class_names=class_names,
        conf_thresh=conf_thresh,
        min_box_area_ratio=min_box_area_ratio,
        post_nms_iou=post_nms_iou,
        prefer_disease=prefer_disease,
    )
    if len(boxes_xyxyn) == 0:
        return None, {
            "error": (
                "No crop disease detected with sufficient confidence. "
                "Ensure camera is pointing directly at a plant leaf in good lighting."
            )
        }

    disease_mask = np.array([not is_healthy_class(infer_model.names.get(int(c), str(int(c)))) for c in clss], dtype=bool)

    if disease_mask.any():
        best_idx = int(np.argmax(np.where(disease_mask, confs, -1.0)))
        display_boxes_xyxyn = boxes_xyxyn[disease_mask]
        display_confs = confs[disease_mask]
        display_clss = clss[disease_mask]
        best_display_idx = int(np.argmax(display_confs))
        display_boxes_xyxyn = display_boxes_xyxyn[[best_display_idx]]
        display_confs = display_confs[[best_display_idx]]
        display_clss = display_clss[[best_display_idx]]
    else:
        best_idx = int(np.argmax(confs))
        display_boxes_xyxyn = np.empty((0, 4), dtype=np.float32)
        display_confs = np.empty((0,), dtype=np.float32)
        display_clss = np.empty((0,), dtype=np.int64)

    disease_class_id = int(clss[best_idx])
    disease_name = infer_model.names.get(disease_class_id, str(disease_class_id))
    confidence = float(confs[best_idx])

    wrapped_model = YOLOOutputWrapper(pytorch_model)
    input_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).div(255.0).to(device)

    if cam_method == "eigencam" and auto_layer_search and not low_memory:
        layer_name, target_layer, grayscale_cam = pick_best_eigencam_layer(
            pytorch_model,
            wrapped_model,
            input_tensor,
            disease_class_id,
            candidates=[layer, "-3", "-4", "-5", "auto"],
        )
    else:
        if cam_method == "eigencam" and auto_layer_search and low_memory:
            print("Low-memory mode: auto layer search disabled; using --layer only.")
        layer_name, target_layer = resolve_target_layer(pytorch_model, layer)
        grayscale_cam = run_cam(wrapped_model, input_tensor, target_layer, cam_method, disease_class_id)

    grayscale_cam = remove_edge_artifacts(grayscale_cam, border_percent=edge_border_percent)

    severity_score, severity_level = compute_severity(grayscale_cam, confidence)
    recommendation = get_recommendation(disease_name, severity_level)

    box_xyxy = []
    for b in display_boxes_xyxyn:
        x1 = int((b[0] * img_bgr_orig.shape[1]) / max(1, img_bgr_orig.shape[1]) * imgsz)
        y1 = int((b[1] * img_bgr_orig.shape[0]) / max(1, img_bgr_orig.shape[0]) * imgsz)
        x2 = int((b[2] * img_bgr_orig.shape[1]) / max(1, img_bgr_orig.shape[1]) * imgsz)
        y2 = int((b[3] * img_bgr_orig.shape[0]) / max(1, img_bgr_orig.shape[0]) * imgsz)
        x1 = max(0, min(imgsz - 1, x1))
        y1 = max(0, min(imgsz - 1, y1))
        x2 = max(0, min(imgsz - 1, x2))
        y2 = max(0, min(imgsz - 1, y2))
        box_xyxy.append([x1, y1, x2, y2])
    final_panel = build_output_panel(
        img_bgr=img_bgr,
        grayscale_cam=grayscale_cam,
        disease_name=disease_name,
        confidence=confidence,
        severity_score=severity_score,
        severity_level=severity_level,
        recommendation=recommendation,
        boxes=box_xyxy,
    )

    cam_refined = refine_cam_for_display(grayscale_cam, box_xyxy, (imgsz, imgsz))
    cam_vis = build_vivid_cam_overlay(img_rgb, cam_refined)
    cam_boxes = cv2.cvtColor(cam_vis, cv2.COLOR_RGB2BGR)
    for xyxy, conf, cls_id in zip(box_xyxy, display_confs.tolist(), display_clss.tolist()):
        if conf < conf_thresh:
            continue
        if is_healthy_class(infer_model.names.get(cls_id, str(cls_id))):
            continue
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        label = f"{infer_model.names.get(cls_id, cls_id)} {conf:.0%}"
        cv2.rectangle(cam_boxes, (x1, y1), (x2, y2), (60, 255, 60), 2)
        cv2.putText(cam_boxes, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 255, 60), 2)

    out_final = out_dir / f"{image_stem}_final.jpg"
    out_cam_only = out_dir / f"{image_stem}_cam_only.jpg"
    out_cam_boxes = out_dir / f"{image_stem}_cam_boxes.jpg"
    out_gray_npy = out_dir / f"{image_stem}_grayscale_cam.npy"
    out_analysis = out_dir / f"{image_stem}_analysis_panel.jpg"

    cv2.imwrite(str(out_final), final_panel)
    cv2.imwrite(str(out_cam_only), cv2.cvtColor(cam_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_cam_boxes), cam_boxes)
    np.save(out_gray_npy, grayscale_cam)
    analysis_panel = build_analysis_panel(
        img_rgb,
        grayscale_cam,
        cam_boxes,
        disease_name=disease_name,
        confidence=confidence,
        severity_level=severity_level,
    )
    cv2.imwrite(str(out_analysis), analysis_panel)
    cleanup_pipeline_results(out_dir, keep_last=25)

    print(f"Detected   : {disease_name}")
    print(f"Confidence : {confidence:.1%}")
    print(f"Conf/IoU   : {conf_thresh:.2f}/{iou_thresh:.2f}")
    print(f"Post-NMS   : {post_nms_iou:.2f}")
    print(f"Boxes kept : {len(box_xyxy)}")
    print(f"CAM Method : {cam_method}")
    print(f"Device     : {device}")
    print(f"Layer      : {layer_name}")
    print(f"CAM Stats  : min={float(grayscale_cam.min()):.3f}, max={float(grayscale_cam.max()):.3f}, mean={float(grayscale_cam.mean()):.3f}")
    print(f"Severity   : {severity_level} ({severity_score:.1f}%)")
    print(f"Treatment  : {recommendation[:100]}{'...' if len(recommendation) > 100 else ''}")
    print("\nOutputs:")
    print(f"  - {out_final}")
    print(f"  - {out_cam_only}")
    print(f"  - {out_cam_boxes}")
    print(f"  - {out_gray_npy}")
    print(f"  - {out_analysis}")

    return {
        "disease": disease_name,
        "confidence": confidence,
        "severity_score": severity_score,
        "severity_level": severity_level,
        "recommendation": recommendation,
        "output_image": str(out_final),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XAI CropGuard pipeline")
    parser.add_argument("--source", choices=["image", "phone"], default="image", help="Input source type")
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--image", help="Path to leaf image (required when --source image)")
    parser.add_argument("--output-dir", default=str(DEFAULT_PIPELINE_OUTPUT_DIR))
    parser.add_argument("--phone-url", default=PHONE_STREAM_URL, help="Phone camera stream URL")
    parser.add_argument("--countdown", type=int, default=5, help="Countdown seconds before phone capture")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=640, help="Model and CAM input image size")
    parser.add_argument("--layer", default="-3", help="Target layer index or 'auto' (e.g. -2, -3, -4)")
    parser.add_argument("--cam-method", choices=CAM_METHODS.keys(), default="eigencam")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--min-box-area", type=float, default=0.002, help="Drop tiny boxes by normalized area")
    parser.add_argument("--post-nms-iou", type=float, default=0.20, help="Extra NMS IoU after model output")
    parser.add_argument("--prefer-disease", action="store_true", help="Ignore Healthy if disease exists")
    parser.add_argument("--no-prefer-disease", action="store_true", help="Do not prioritize disease over Healthy")
    parser.add_argument("--edge-border", type=float, default=0.08, help="CAM border percentage to suppress edge artifacts")
    parser.add_argument("--no-auto-layer-search", action="store_true", help="Disable fallback layer search for EigenCAM")
    parser.add_argument("--low-memory", action="store_true", help="Reduce thread usage and avoid CAM layer auto-search")
    parser.add_argument("--rpi5", action="store_true", help="Apply Raspberry Pi 5 friendly defaults")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.source == "image" and not args.image:
        raise SystemExit("--image is required when --source image")
    if not args.prefer_disease and not args.no_prefer_disease:
        args.prefer_disease = True
    if args.rpi5:
        args.device = "cpu"
        args.imgsz = min(args.imgsz, 512)
        args.cam_method = "eigencam"
        args.low_memory = True
        args.no_auto_layer_search = True

    prefer_disease = args.prefer_disease and not args.no_prefer_disease

    if args.source == "phone":
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        frame = capture_from_phone(stream_url=args.phone_url, countdown=max(0, args.countdown))
        ts = int(time.time())
        captured_path = out_dir / f"phone_capture_{ts}.jpg"
        cv2.imwrite(str(captured_path), frame)
        image_path = str(captured_path)
        print(f"Saved phone frame: {captured_path}")
        print("Running pipeline...")
    else:
        image_path = args.image

    result = run_pipeline(
        model_path=args.model,
        image_path=image_path,
        output_dir=args.output_dir,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        imgsz=args.imgsz,
        layer=args.layer,
        cam_method=args.cam_method,
        auto_layer_search=not args.no_auto_layer_search,
        low_memory=args.low_memory,
        device=args.device,
        min_box_area_ratio=args.min_box_area,
        post_nms_iou=args.post_nms_iou,
        prefer_disease=prefer_disease,
        edge_border_percent=args.edge_border,
    )
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        err = result[1].get("error", "Pipeline failed.")
        print(f"Error: {err}")
