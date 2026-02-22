# XAI CropGuard Progress Tracker

Last updated: 2026-02-20
Primary workspace: `/media/harsh/FDDA-6AD0/xai-cropguard`

This file is the live project monitor for development, validation, and deployment readiness.

## 1. Project Goal

Build a production-ready, Raspberry Pi 5 compatible crop-disease explainability pipeline that:
- detects disease with YOLOv8
- explains predictions with EigenCAM
- computes severity from activation maps
- provides practical treatment recommendations
- produces farmer-facing output panels

## 2. Current Status Summary

Overall status: `Core pipeline complete and stable`

Completed:
- YOLOv8 inference integrated and stable.
- EigenCAM integrated as default XAI method.
- Confidence filtering + extra post-processing (NMS + tiny box filtering) implemented.
- Healthy-vs-disease conflict handling implemented (disease prioritized).
- Edge artifact suppression in CAM implemented.
- Severity scoring and severity bar integrated.
- Recommendation engine integrated with detailed disease-specific advice.
- Final farmer panel + report analysis panel generation implemented.
- RPi5 runtime mode implemented (`--rpi5`) with low-memory settings.

Pending:
- Raspberry Pi deployment packaging and service setup.
- Flask API and offline UI integration.
- Camera capture pipeline integration on Pi.
- End-to-end field validation on real camera feed.

## 3. Pipeline Outputs (Current)

For each image, pipeline currently writes:
- `*_final.jpg` (farmer-facing panel)
- `*_cam_only.jpg` (CAM overlay only)
- `*_cam_boxes.jpg` (CAM with filtered boxes)
- `*_grayscale_cam.npy` (raw CAM array)
- `*_analysis_panel.jpg` (3-panel report figure)

Example recent outputs:
- `/media/harsh/FDDA-6AD0/xai-cropguard/runs/pipeline/id_xh32sn_final.jpg`
- `/media/harsh/FDDA-6AD0/xai-cropguard/runs/pipeline/id_xh32sn_analysis_panel.jpg`

## 4. Key Technical Decisions

Detection and filtering:
- `conf` default: `0.35`
- model NMS IoU default: `0.20`
- extra post-NMS IoU: `0.20`
- tiny-box filter (normalized area): `0.002`
- agnostic NMS enabled

XAI (EigenCAM):
- default target layer: `-3`
- edge artifact masking enabled (default border: `8%`)
- stronger overlay blend (`0.40 image + 0.60 heatmap`)

Severity and contours:
- infected-mask threshold: `180/255`
- morphology cleanup: open + close (5x5)
- contour area gate: `>500 px^2` (removes stray tiny artifacts)

Recommendation logic:
- deterministic rule-based mapping from `class + severity`
- if disease classes exist, healthy classes are ignored for primary diagnosis

## 5. Script Responsibilities

- `scripts/pipeline.py`
  - end-to-end production pipeline
  - post-processing filters
  - severity + recommendation + final panel

- `scripts/yolo_gradcam.py`
  - direct YOLO + CAM debug/inspection script
  - supports layer exploration and RPi profile

- `scripts/eigencam_test.py`
  - focused EigenCAM validation script
  - quick sanity checks for CAM quality

- `scripts/severity.py`
  - severity score, mask extraction, severity bar drawing

- `scripts/recommendations.py`
  - disease recommendation dictionary and lookup

## 6. RPi5 Compatibility Status

Ready and implemented:
- `--rpi5` profile
- CPU execution path
- low-memory thread limits
- reduced image size in RPi profile
- no expensive auto layer search in strict low-memory flow

Needs validation on actual Pi hardware:
- sustained inference latency
- thermals under continuous camera load
- filesystem write throughput and power stability

## 7. Known Risks / Notes

- Confidence values depend on dataset quality and class balance; some classes may still need retraining/calibration.
- CAM localization quality depends on target layer and image quality; current defaults are robust but not perfect for every class.
- Field images with multiple leaves/plants can still produce multiple disease boxes; filtering is configured, but review rules may need tuning per crop.

## 8. Standard Run Commands

Pipeline (recommended):
```bash
cd /media/harsh/FDDA-6AD0/xai-cropguard
/home/harsh/miniconda3/envs/xai-py311/bin/python scripts/pipeline.py \
  --model trial/cropguard_best_model/best.pt \
  --image data/raw/images/id_xh32sn.jpg \
  --rpi5 \
  --output-dir /media/harsh/FDDA-6AD0/xai-cropguard/runs/pipeline
```

GradCAM/EigenCAM inspection:
```bash
/home/harsh/miniconda3/envs/xai-py311/bin/python scripts/yolo_gradcam.py \
  --model trial/cropguard_best_model/best.pt \
  --image data/raw/images/id_xh32sn.jpg \
  --rpi5 \
  --output-dir /media/harsh/FDDA-6AD0/xai-cropguard/runs/gradcam
```

## 9. Validation Snapshot (Latest)

Image: `id_xh32sn.jpg`
- Predicted class: `Pepper_Leaf_Curl`
- Confidence: `83.4%`
- Boxes kept after filtering: `1`
- Severity: `Medium (34.1%)`
- Treatment text: detailed multi-line vector-control guidance
- Edge artifacts: removed
- Stray tiny contours: suppressed with area gate

## 10. How To Keep This Updated (Mandatory Process)

After each meaningful change, update this file in 4 places:

1. `Last updated` date at top.
2. `Current Status Summary` (completed/pending bullets).
3. `Validation Snapshot` with latest run metrics.
4. Append a new entry in `Progress Log` (below).

If a run fails, still log it with failure reason and attempted fix.

Quick append helper (recommended):
```bash
cd /media/harsh/FDDA-6AD0/xai-cropguard
scripts/log_progress.sh "Area" "Change summary" "Success/Failed" "runs/path_or_note"
```

## 11. Progress Log (Append-Only)

Use this format for each new entry:

```text
YYYY-MM-DD HH:MM | Area | Change | Result | Output path(s)
```

Entries:
- 2026-02-20 18:37 | CAM Engine | EigenCAM-first + RPi5 low-memory mode integrated | Success | scripts updated and validated
- 2026-02-20 18:45 | Pipeline Run | USB output path standardized | Success | runs/pipeline/id_x8dn7k_*
- 2026-02-20 19:xx | Post-processing | IoU reduction + vivid overlay + disease-priority filtering | Success | runs/pipeline/id_xh32sn_*
- 2026-02-20 20:xx | CAM cleanup | Edge artifact masking + contour cleanup + recommendation enrichment | Success | runs/pipeline/id_xh32sn_final.jpg
- 2026-02-20 20:54 | Tracking | Added progress logging helper script and docs wiring | Success | scripts/log_progress.sh, README_PROGRESS.md

## 12. Next Execution Plan

1. Freeze this pipeline version as `v1.0-pi-ready`.
2. Add deployment scripts for Pi service startup.
3. Build Flask inference endpoint wrapping `pipeline.py`.
4. Build offline farmer UI with image upload and output preview.
5. Integrate camera capture and continuous inference mode.
