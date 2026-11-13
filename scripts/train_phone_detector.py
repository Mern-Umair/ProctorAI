"""
Step 2c - Phone-detector experiment.

1. Baseline: evaluate the COCO-pretrained YOLOv8n ("cell phone" = class 67) on the phone
   dataset's test split (labels temporarily remapped 0 -> 67 so Ultralytics can score it).
2. Fine-tune YOLOv8n for a single `phone` class on the same dataset and evaluate on the
   same test split.
3. Export the fine-tuned model to ONNX (320 px) and save metrics + figures.

Usage:  python scripts/train_phone_detector.py [epochs]
"""
import json
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
DATA = PROC / "phone_yolo" / "data.yaml"
MODELS = ROOT / "models"
DOCS = ROOT / "docs" / "figures"
IMG = 320  # small image size -> real-time in browser on CPU

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
COCO_PHONE = 67


def make_coco_labelled_copy() -> Path:
    """Copy of the dataset whose label files use class 67 (COCO 'cell phone') and an 80-name yaml."""
    src = PROC / "phone_yolo"
    dst = PROC / "phone_yolo_coco"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src / "images", dst / "images")
    for split in ("train", "valid", "test"):
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for f in (src / "labels" / split).glob("*.txt"):
            lines = [f"{COCO_PHONE} " + ln.split(" ", 1)[1] for ln in f.read_text().splitlines() if ln.strip()]
            (dst / "labels" / split / f.name).write_text("\n".join(lines))
    names = YOLO("yolov8n.pt").names
    yaml = f"path: {dst.as_posix()}\ntrain: images/train\nval: images/valid\ntest: images/test\nnames:\n"
    yaml += "".join(f"  {i}: {n}\n" for i, n in names.items())
    (dst / "data.yaml").write_text(yaml)
    return dst / "data.yaml"


def val_metrics(model, data, name):
    m = model.val(data=str(data), split="test", imgsz=IMG, device="cpu", plots=False, verbose=False,
                  project=str(ROOT / "runs"), name=name, exist_ok=True)
    return {"test_mAP50": float(m.box.map50), "test_mAP50_95": float(m.box.map),
            "test_precision": float(m.box.mp), "test_recall": float(m.box.mr)}


def main():
    MODELS.mkdir(exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    # ---------------- 1. COCO baseline on the same test split
    coco_yaml = make_coco_labelled_copy()
    baseline = val_metrics(YOLO("yolov8n.pt"), coco_yaml, "phone_coco_baseline_test")
    print("COCO baseline (cell phone):", baseline)

    # ---------------- 2. fine-tune
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA), epochs=EPOCHS, imgsz=IMG, batch=8, device="cpu", workers=0, cache=False,
        project=str(ROOT / "runs"), name="phone_yolov8n", exist_ok=True, seed=42,
        patience=10, optimizer="AdamW", lr0=1e-3, cos_lr=True,
        mosaic=1.0, fliplr=0.5, hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, degrees=5, scale=0.4,
        plots=True, verbose=False,
    )
    run = ROOT / "runs" / "phone_yolov8n"
    best = run / "weights" / "best.pt"
    ft = YOLO(str(best))
    tuned = val_metrics(ft, DATA, "phone_yolov8n_test")
    print("fine-tuned:", tuned)

    # ---------------- 3. export + artefacts
    shutil.copy2(best, MODELS / "phone_yolov8n.pt")
    ft.export(format="onnx", imgsz=IMG, opset=12, simplify=True, dynamic=False)
    shutil.move(str(best.with_suffix(".onnx")), MODELS / "phone_yolov8n.onnx")
    for f in ("results.png", "confusion_matrix.png", "PR_curve.png", "F1_curve.png",
              "labels.jpg", "val_batch0_pred.jpg", "train_batch0.jpg"):
        if (run / f).exists():
            shutil.copy2(run / f, DOCS / f"phone_{f}")
    if (run / "results.csv").exists():
        shutil.copy2(run / "results.csv", MODELS / "phone_training_results.csv")
    metrics = {"epochs": EPOCHS, "imgsz": IMG, "model": "yolov8n fine-tuned (1 class: phone)",
               **tuned, "coco_baseline": baseline}
    json.dump(metrics, open(MODELS / "phone_metrics.json", "w"), indent=2)
    print("saved models/phone_yolov8n.{pt,onnx}, phone_metrics.json, figures")


if __name__ == "__main__":
    main()
