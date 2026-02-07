import os
import re
import time
from datetime import datetime

try:
    from tensorboardX import SummaryWriter
except Exception:
    raise SystemExit("tensorboardX is required. Install with: pip install tensorboardX")

# Path to your YOLO training log (can override with TRAINING_LOG env var)
LOG_PATH = os.environ.get("TRAINING_LOG", "runs/detect/cropguard_yolo_v1/training.log")
TB_DIR = os.environ.get("TB_DIR", "runs/tensorboard_adapter")

os.makedirs(TB_DIR, exist_ok=True)
writer = SummaryWriter(TB_DIR)

# Regex examples for YOLOv8 logs (adjust if your log format differs)
EPOCH_RE = re.compile(r"Epoch\s+(\d+)/(\d+)")
METRICS_RE = re.compile(
    r"box_loss[:=]?\s*([0-9]*\.?[0-9]+).*?cls_loss[:=]?\s*([0-9]*\.?[0-9]+).*?dfl_loss[:=]?\s*([0-9]*\.?[0-9]+).*?metrics/mAP50[:=]?\s*([0-9]*\.?[0-9]+).*?metrics/mAP50-95[:=]?\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE | re.DOTALL,
)

last_pos = 0
current_epoch = 0

print(f"[TB Adapter] Watching {LOG_PATH}")
print(f"[TB Adapter] Writing TensorBoard logs to {TB_DIR}")

while True:
    if not os.path.exists(LOG_PATH):
        time.sleep(2)
        continue

    with open(LOG_PATH, "r", errors="ignore") as f:
        f.seek(last_pos)
        lines = f.readlines()
        last_pos = f.tell()

    for line in lines:
        # Try to parse epoch
        m_epoch = EPOCH_RE.search(line)
        if m_epoch:
            try:
                current_epoch = int(m_epoch.group(1))
            except Exception:
                pass

        # Try to parse metrics
        m = METRICS_RE.search(line)
        if m and current_epoch > 0:
            try:
                box_loss = float(m.group(1))
                cls_loss = float(m.group(2))
                dfl_loss = float(m.group(3))
                map50 = float(m.group(4))
                map5095 = float(m.group(5))

                writer.add_scalar("train/box_loss", box_loss, current_epoch)
                writer.add_scalar("train/cls_loss", cls_loss, current_epoch)
                writer.add_scalar("train/dfl_loss", dfl_loss, current_epoch)
                writer.add_scalar("metrics/mAP50", map50, current_epoch)
                writer.add_scalar("metrics/mAP50-95", map5095, current_epoch)

                writer.flush()
                print(f"[TB Adapter] Epoch {current_epoch} → logged scalars")
            except Exception as e:
                print(f"[TB Adapter] Failed to parse metrics: {e}")

    time.sleep(3)
