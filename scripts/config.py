"""Centralized runtime configuration and shared path/file helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = REPO_ROOT / "trial" / "cropguard_best_model" / "best.pt"
RESULTS_DIR = REPO_ROOT / "results"
ROVER_LOG_FILE = RESULTS_DIR / "rover_log.csv"
LATEST_IMAGE = RESULTS_DIR / "latest.jpg"
TEMP_CAPTURE_IMAGE = Path(tempfile.gettempdir()) / "cropguard_capture.jpg"
DEFAULT_PIPELINE_OUTPUT_DIR = REPO_ROOT / "runs" / "pipeline"

PIPELINE_OUTPUT_SUFFIXES = (
    "_final.jpg",
    "_cam_only.jpg",
    "_cam_boxes.jpg",
    "_analysis_panel.jpg",
    "_grayscale_cam.npy",
)
PIPELINE_PRIMARY_IMAGE_SUFFIX = "_final.jpg"


def _pipeline_capture_stem(file_name: str) -> str | None:
    for suffix in PIPELINE_OUTPUT_SUFFIXES:
        if file_name.endswith(suffix):
            return file_name[: -len(suffix)]
    return None


def cleanup_pipeline_results(output_dir: str | Path, keep_last: int = 25) -> None:
    """
    Keep the most recent `keep_last` pipeline captures and remove older capture files.

    A capture is identified by one stem plus known suffix files:
    `<stem>_final.jpg`, `<stem>_cam_only.jpg`, `<stem>_cam_boxes.jpg`,
    `<stem>_analysis_panel.jpg`, `<stem>_grayscale_cam.npy`.
    """
    out_path = Path(output_dir)
    if keep_last <= 0 or not out_path.exists() or not out_path.is_dir():
        return

    groups: dict[str, list[Path]] = {}
    for path in out_path.iterdir():
        if not path.is_file():
            continue
        stem = _pipeline_capture_stem(path.name)
        if stem is None:
            continue
        groups.setdefault(stem, []).append(path)

    if len(groups) <= keep_last:
        return

    sorted_stems = sorted(
        groups.keys(),
        key=lambda stem: max(p.stat().st_mtime for p in groups[stem]),
    )
    for old_stem in sorted_stems[: len(groups) - keep_last]:
        for file_path in groups[old_stem]:
            try:
                file_path.unlink()
            except OSError:
                # Cleanup is best-effort; keep processing remaining files.
                pass
