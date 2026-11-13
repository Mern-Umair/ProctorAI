"""
Step 2d - Export the object detector used by the web app to ONNX (320 px, static shape).

By default this exports the COCO-pretrained YOLOv8n, whose 80 classes include
`person` (0) and `cell phone` (67). The web app only reads those two classes.
Pass a path to a different .pt (e.g. models/phone_yolov8n.pt) to export the
fine-tuned single-class model instead; then update detectorClasses in web/js/config.js.

Usage:  python scripts/export_detector.py [weights.pt]
"""
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
WEB_MODELS = ROOT / "web" / "models"
IMG = 320

weights = sys.argv[1] if len(sys.argv) > 1 else "yolov8n.pt"
model = YOLO(weights)
out = Path(model.export(format="onnx", imgsz=IMG, opset=12, simplify=True, dynamic=False))
MODELS.mkdir(exist_ok=True)
WEB_MODELS.mkdir(parents=True, exist_ok=True)
dst = MODELS / "detector_yolov8n.onnx"
shutil.move(str(out), dst)
shutil.copy2(dst, WEB_MODELS / dst.name)
print(f"exported {weights} -> {dst} and {WEB_MODELS / dst.name}  ({dst.stat().st_size / 1e6:.1f} MB)")
print("classes:", {int(k): v for k, v in model.names.items()} if len(model.names) < 10 else f"{len(model.names)} COCO classes (person=0, cell phone=67)")
