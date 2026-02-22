#!/usr/bin/env python3
"""Severity scoring utilities for Grad-CAM outputs."""

from __future__ import annotations

import cv2
import numpy as np


def compute_severity(grayscale_cam: np.ndarray, confidence: float) -> tuple[float, str]:
    """
    Compute disease severity from Grad-CAM activation + detector confidence.

    Args:
        grayscale_cam: 2D array in [0.0, 1.0]
        confidence: YOLO confidence in [0.0, 1.0]

    Returns:
        (severity_score_0_to_100, severity_level)
    """
    cam = np.clip(np.asarray(grayscale_cam, dtype=np.float32), 0.0, 1.0)
    conf = float(np.clip(confidence, 0.0, 1.0))

    threshold = 0.5
    infected_ratio = float(np.sum(cam > threshold) / cam.size)
    severity_score = round((infected_ratio * 0.6 + conf * 0.4) * 100.0, 1)
    severity_score = float(np.clip(severity_score, 0.0, 100.0))

    if severity_score < 30:
        level = "Low"
    elif severity_score < 60:
        level = "Medium"
    else:
        level = "High"

    return severity_score, level


def get_infected_mask(grayscale_cam: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    """
    Return binary infected-region mask resized to requested (width, height).
    """
    cam = np.clip(np.asarray(grayscale_cam, dtype=np.float32), 0.0, 1.0)
    cam_resized = cv2.resize(cam, original_size, interpolation=cv2.INTER_LINEAR)
    _, mask = cv2.threshold(np.uint8(cam_resized * 255), 180, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def draw_severity_bar(
    image: np.ndarray,
    severity_score: float,
    severity_level: str,
    x: int = 10,
    y: int = 10,
    width: int = 220,
) -> np.ndarray:
    """Draw a compact severity progress bar on an image copy."""
    img = image.copy()

    color_map = {
        "Low": (0, 200, 0),
        "Medium": (0, 140, 255),
        "High": (0, 0, 255),
    }
    color = color_map.get(severity_level, (255, 255, 255))

    cv2.rectangle(img, (x, y), (x + width, y + 20), (60, 60, 60), -1)
    filled_width = int(width * float(np.clip(severity_score, 0.0, 100.0)) / 100.0)
    cv2.rectangle(img, (x, y), (x + filled_width, y + 20), color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + 20), (170, 170, 170), 1)

    cv2.putText(
        img,
        f"Severity: {severity_level} ({severity_score:.1f}%)",
        (x, y + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )
    return img
