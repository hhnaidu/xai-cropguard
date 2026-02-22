import argparse
import csv
import os
import sys
from datetime import datetime

import cv2
from ultralytics import YOLO

from scripts.camera import get_frame_generator
from scripts.config import MODEL_PATH, RESULTS_DIR, ROVER_LOG_FILE, TEMP_CAPTURE_IMAGE, cleanup_pipeline_results

sys.path.insert(0, os.path.dirname(__file__))
from scripts.pipeline import run_pipeline

OUTPUT_DIR = str(RESULTS_DIR)
LOG_FILE = str(ROVER_LOG_FILE)
DETECTION_CONF = 0.80
COOLDOWN_FRAMES = 30
MAX_RESULTS = 25
TEMP_IMAGE = str(TEMP_CAPTURE_IMAGE)


def init_log_file():
    if not os.path.exists(LOG_FILE):
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'disease',
                'confidence',
                'severity_level',
                'severity_score',
                'image_path',
            ])


def run_rover(camera_source=0):
    print('Loading detection model...')
    yolo = YOLO(MODEL_PATH)
    print(f'Model loaded. Classes: {len(yolo.names)}')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_log_file()
    frame_count = 0
    detection_count = 0
    cooldown = 0

    print(f'ROVER MODE ACTIVE on source: {camera_source}')
    print('Press Ctrl+C to stop')

    try:
        for frame in get_frame_generator(camera_source):
            frame_count += 1

            if cooldown > 0:
                cooldown -= 1
                continue

            results = yolo.predict(
                source=frame,
                conf=DETECTION_CONF,
                iou=0.2,
                imgsz=320,
                verbose=False,
            )

            disease_boxes = [
                b
                for b in results[0].boxes
                if 'Healthy' not in yolo.names[int(b.cls)]
                and float(b.conf) >= DETECTION_CONF
            ]

            if disease_boxes:
                best = max(disease_boxes, key=lambda b: float(b.conf))
                disease_name = yolo.names[int(best.cls)]
                conf = float(best.conf)

                print(f'\n DISEASE DETECTED: {disease_name} ({conf:.1%})')
                print('Running full EigenCAM analysis...')

                cv2.imwrite(TEMP_IMAGE, frame)

                pipeline_result = run_pipeline(
                    model_path=MODEL_PATH,
                    image_path=TEMP_IMAGE,
                    preloaded_model=yolo,
                    output_dir=OUTPUT_DIR,
                )

                if os.path.exists(TEMP_IMAGE):
                    os.remove(TEMP_IMAGE)

                if isinstance(pipeline_result, tuple):
                    _, result = pipeline_result
                else:
                    result = pipeline_result

                if isinstance(result, dict) and 'error' not in result:
                    with open(LOG_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            result['disease'],
                            result['confidence'],
                            result['severity_level'],
                            result['severity_score'],
                            result['output_image'],
                        ])

                    cleanup_pipeline_results(OUTPUT_DIR, MAX_RESULTS)
                    cooldown = COOLDOWN_FRAMES
                    detection_count += 1
                    print(f'Detection #{detection_count} logged.')
                else:
                    print(f'Pipeline error: {result}')

            if frame_count % 50 == 0:
                print(
                    f'Scanned: {frame_count} frames | '
                    f'Detections: {detection_count} | '
                    f'Cooldown: {cooldown}'
                )

    except KeyboardInterrupt:
        print('\nRover stopped by user.')
    finally:
        print('\n--- SESSION SUMMARY ---')
        print(f'Total frames scanned : {frame_count}')
        print(f'Total detections     : {detection_count}')
        print(f'Log file             : {LOG_FILE}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source',
        default='0',
        help="Camera source: 'picamera', phone URL, or integer",
    )
    args = parser.parse_args()

    source = args.source
    if source.isdigit():
        source = int(source)

    run_rover(source)
