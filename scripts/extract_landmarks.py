"""
Step 2a - Run MediaPipe FaceLandmarker on every head-pose image and save a
feature matrix for the attention classifier.

For each image we extract landmarks from the original AND the horizontally
flipped image (left <-> right label swapped). This doubles the data and
guarantees the classifier is symmetric.

Output: data/processed/attention_features.npz  with
    X : (n, 1404) float32 features        y : (n,) int class ids
    tilt, pan : original angles           flipped : 0/1      src : source image file
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASS_TO_ID, landmarks_to_features, pose_to_class  # noqa: E402

import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python as mp_python  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "head_pose"
OUT = ROOT / "data" / "processed" / "attention_features.npz"
MODEL = ROOT / "models" / "face_landmarker.task"

FLIP = {"left": "right", "right": "left"}


def make_landmarker():
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def detect(landmarker, bgr: np.ndarray):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]
    return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)


def main():
    rows = list(csv.DictReader(open(RAW / "labels.csv", newline="")))
    print(f"{len(rows)} labelled images")
    lmk = make_landmarker()

    L, X, y, tilts, pans, flips, files = [], [], [], [], [], [], []
    miss = 0
    for r in tqdm(rows, desc="landmarks"):
        img = cv2.imread(str(RAW / "images" / r["file"]))
        if img is None:
            miss += 1
            continue
        tilt, pan = int(r["tilt"]), int(r["pan"])
        cls = pose_to_class(tilt, pan)
        for flipped in (0, 1):
            im = cv2.flip(img, 1) if flipped else img
            lm = detect(lmk, im)
            if lm is None:
                miss += 1
                continue
            label = FLIP.get(cls, cls) if flipped else cls
            L.append(lm[:478])
            X.append(landmarks_to_features(lm))
            y.append(CLASS_TO_ID[label])
            tilts.append(tilt)
            pans.append(-pan if flipped else pan)
            flips.append(flipped)
            files.append(r["file"])

    L = np.stack(L).astype(np.float32)
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # L = raw normalised landmarks (n, 478, 3) so features can be re-derived without re-running MediaPipe
    np.savez_compressed(OUT, L=L, X=X, y=y, tilt=np.array(tilts), pan=np.array(pans),
                        flipped=np.array(flips), src=np.array(files))
    print(f"saved L{L.shape} X{X.shape} -> {OUT}   (no face detected in {miss} images)")
    for c, i in CLASS_TO_ID.items():
        print(f"  {c:7s}: {(y == i).sum()}")


if __name__ == "__main__":
    main()
