import csv
import os
import sys
import threading
from pathlib import Path

import cv2
from flask import Flask, jsonify, request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.camera import capture_single_frame
from scripts.config import (
    LATEST_IMAGE as CFG_LATEST_IMAGE,
    MODEL_PATH as CFG_MODEL_PATH,
    ROVER_LOG_FILE as CFG_ROVER_LOG_FILE,
    TEMP_CAPTURE_IMAGE as CFG_TEMP_CAPTURE_IMAGE,
)
from scripts.pipeline import run_pipeline

app = Flask(__name__)

TEMP_IMAGE = str(CFG_TEMP_CAPTURE_IMAGE)
LOG_FILE = str(CFG_ROVER_LOG_FILE)
LATEST_IMAGE = str(CFG_LATEST_IMAGE)
MODEL_PATH = str(CFG_MODEL_PATH)
scan_lock = threading.Lock()


@app.route('/capture', methods=['POST'])
def capture():
    if not scan_lock.acquire(blocking=False):
        return jsonify({'error': 'Scan already in progress. Please wait.'})
    try:
        data = request.get_json(silent=True) or {}
        source = data.get('source', 'picamera')

        if isinstance(source, str) and source.isdigit():
            source = int(source)

        frame = capture_single_frame(source)
        cv2.imwrite(TEMP_IMAGE, frame)

        pipeline_result = run_pipeline(model_path=MODEL_PATH, image_path=TEMP_IMAGE)
        if isinstance(pipeline_result, tuple):
            _, result = pipeline_result
        else:
            result = pipeline_result

        if os.path.exists(TEMP_IMAGE):
            os.remove(TEMP_IMAGE)

        if isinstance(result, dict) and 'error' not in result:
            import shutil

            shutil.copy(result['output_image'], LATEST_IMAGE)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        scan_lock.release()


@app.route('/rover_log')
def rover_log():
    if not os.path.exists(LOG_FILE):
        return jsonify([])

    rows = []
    with open(LOG_FILE, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return jsonify(rows)


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'loaded'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
