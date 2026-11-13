"""
ProctorAI - Dataset downloader
--------------------------------
Downloads the two public datasets used in this project from the Hugging Face Hub:

1. Head-Pose dataset  (StevenLe456/head-pose)  -> used to train the attention / gaze-direction classifier
   * 13,950 rows = 2,790 unique (tilt, pan) poses x 5 augmented copies.
   * We download ONE image per group of 5 (row index % 5 == 0) to get the 2,790 unique poses,
     because augmentation is re-done on the fly during training anyway.
2. Phone-Detection dataset (harshdadiya-wappnet/phone_detection) -> used to fine-tune YOLOv8n
   * COCO-format annotations, train / valid / test splits.

Only the Python standard library is used so this works before the ML environment is installed.
Run:  python -I scripts/download_data.py [--headpose-stride 5]
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HF = "https://huggingface.co"
DS_SERVER = "https://datasets-server.huggingface.co"
UA = {"User-Agent": "ProctorAI-downloader/1.0"}


def fetch(url: str, dest: Path | None = None, retries: int = 5, timeout: int = 60) -> bytes | None:
    """GET a URL with retries. If dest is given, stream to disk and return None."""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if dest is not None:
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(data)
                tmp.replace(dest)
                return None
            return data
        except Exception as e:  # noqa: BLE001
            wait = min(2 ** attempt, 30)
            print(f"  retry {attempt}/{retries} for {url[-60:]} ({e}); sleeping {wait}s", flush=True)
            time.sleep(wait)
    print(f"  FAILED: {url}", flush=True)
    return None


# --------------------------------------------------------------------------- phone dataset
def download_phone_dataset(workers: int = 6) -> None:
    repo = "harshdadiya-wappnet/phone_detection"
    out = RAW / "phone_detection"
    meta = json.loads(fetch(f"{HF}/api/datasets/{repo}"))
    files = [s["rfilename"] for s in meta["siblings"] if s["rfilename"].startswith(("train/", "valid/", "test/"))]
    todo = [f for f in files if not (out / f).exists() or (out / f).stat().st_size == 0]
    print(f"[phone] {len(files)} files total, {len(todo)} to download", flush=True)

    def job(f):
        fetch(f"{HF}/datasets/{repo}/resolve/main/{f}", out / f)
        return f

    done = 0
    with ThreadPoolExecutor(workers) as ex:
        for _ in as_completed([ex.submit(job, f) for f in todo]):
            done += 1
            if done % 25 == 0 or done == len(todo):
                print(f"[phone] {done}/{len(todo)}", flush=True)
    for split in ("train", "valid", "test"):
        n = len(list((out / split).glob("*.jpg")))
        print(f"[phone] {split}: {n} images", flush=True)


# --------------------------------------------------------------------------- head-pose dataset
def download_headpose_dataset(stride: int = 5, workers: int = 6) -> None:
    ds = "StevenLe456/head-pose"
    out = RAW / "head_pose"
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out / "labels.csv"

    existing = {}
    if labels_path.exists():
        with open(labels_path, newline="") as f:
            for row in csv.DictReader(f):
                existing[int(row["idx"])] = row

    total = 13950
    chunk = 100
    rows_meta = []
    for offset in range(0, total, chunk):
        url = f"{DS_SERVER}/rows?dataset={ds}&config=default&split=train&offset={offset}&length={chunk}"
        data = fetch(url)
        if data is None:
            continue
        for r in json.loads(data)["rows"]:
            idx = r["row_idx"]
            if idx % stride != 0:
                continue
            tilt, pan = r["row"]["y"]
            rows_meta.append((idx, tilt, pan, r["row"]["x"]["src"]))
        print(f"[headpose] metadata {min(offset + chunk, total)}/{total}", flush=True)

        # download this chunk's images right away (signed asset URLs expire)
        todo = [m for m in rows_meta if m[0] >= offset and not (img_dir / f"{m[0]:05d}.jpg").exists()]

        def job(m):
            idx, tilt, pan, src = m
            fetch(src, img_dir / f"{idx:05d}.jpg")
            return m

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(job, todo))

    # write labels for everything we have on disk
    with open(labels_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "file", "tilt", "pan"])
        n = 0
        for idx, tilt, pan, _ in sorted(rows_meta):
            p = img_dir / f"{idx:05d}.jpg"
            if p.exists() and p.stat().st_size > 0:
                w.writerow([idx, p.name, tilt, pan])
                n += 1
    print(f"[headpose] {n} images with labels written to {labels_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headpose-stride", type=int, default=5)
    ap.add_argument("--skip-phone", action="store_true")
    ap.add_argument("--skip-headpose", action="store_true")
    args = ap.parse_args()
    if not args.skip_phone:
        download_phone_dataset()
    if not args.skip_headpose:
        download_headpose_dataset(stride=args.headpose_stride)
    print("ALL_DOWNLOADS_DONE", flush=True)
