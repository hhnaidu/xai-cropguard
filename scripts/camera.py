import time

import cv2


def capture_single_frame(source):
    if source == "picamera":
        try:
            from picamera2 import Picamera2

            picam2 = Picamera2()
            config = picam2.create_still_configuration(
                main={"size": (1280, 720), "format": "RGB888"}
            )
            picam2.configure(config)
            picam2.start()
            time.sleep(2)
            frame_rgb = picam2.capture_array()
            picam2.stop()
            picam2.close()
            return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise RuntimeError(f"PiCamera capture failed: {e}") from e

    cap = None
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {source}")

        for _ in range(5):
            cap.read()

        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to read frame from stream")

        return frame
    finally:
        if cap is not None:
            cap.release()


def get_frame_generator(source, skip_frames=4):
    if source == "picamera":
        from picamera2 import Picamera2

        picam2 = Picamera2()
        config = picam2.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        counter = 0
        try:
            while True:
                frame_rgb = picam2.capture_array()
                counter += 1
                if counter % (skip_frames + 1) == 0:
                    yield cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        finally:
            picam2.stop()
            picam2.close()
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {source}")

    counter = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            counter += 1
            if counter % (skip_frames + 1) == 0:
                yield frame
    finally:
        cap.release()
