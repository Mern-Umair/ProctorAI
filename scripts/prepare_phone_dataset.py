"""
Convert the COCO-format phone-detection dataset into YOLO format
and write data/processed/phone_yolo/data.yaml for Ultralytics.

COCO bbox = [x_min, y_min, w, h] in pixels
YOLO line = class x_center y_center w h  (all normalised 0..1)
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "phone_detection"
DST = ROOT / "data" / "processed" / "phone_yolo"


def convert_split(split: str) -> int:
    ann = json.loads((SRC / split / "_annotations.coco.json").read_text())
    # Roboflow exports a dummy super-category with id 0; the real class is 'mobile_phone'
    real_ids = {c["id"] for c in ann["categories"] if c["supercategory"] != "none"}
    img_dir = DST / "images" / split
    lbl_dir = DST / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    by_img = {}
    for a in ann["annotations"]:
        if a["category_id"] in real_ids:
            by_img.setdefault(a["image_id"], []).append(a)

    n = 0
    for im in ann["images"]:
        src_img = SRC / split / im["file_name"]
        if not src_img.exists() or src_img.stat().st_size == 0:
            continue
        shutil.copy2(src_img, img_dir / im["file_name"])
        W, H = im["width"], im["height"]
        lines = []
        for a in by_img.get(im["id"], []):
            x, y, w, h = a["bbox"]
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
        (lbl_dir / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines))
        n += 1
    return n


if __name__ == "__main__":
    counts = {s: convert_split(s) for s in ("train", "valid", "test")}
    yaml = f"""# ProctorAI phone detector dataset (YOLO format)
path: {DST.as_posix()}
train: images/train
val: images/valid
test: images/test
names:
  0: phone
"""
    (DST / "data.yaml").write_text(yaml)
    print("converted:", counts)
    print("wrote", DST / "data.yaml")
