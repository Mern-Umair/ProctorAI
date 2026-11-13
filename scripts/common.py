"""
Shared constants + feature engineering for the attention (head-direction) classifier.

IMPORTANT: web/js/features.js re-implements `landmarks_to_features` in JavaScript.
If you change anything here, change it there too, otherwise the browser model will
receive differently-scaled inputs than it was trained on.
"""
from __future__ import annotations

import numpy as np

# Attention classes predicted from face landmarks
CLASSES = ["center", "left", "right", "up", "down"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# MediaPipe FaceLandmarker indices used for normalisation
NOSE_TIP = 1
LEFT_EYE_OUTER = 33     # subject's right eye in image space (image left)
RIGHT_EYE_OUTER = 263   # subject's left eye in image space (image right)
N_LANDMARKS = 468       # we ignore the 10 iris refinement points (468..477)

# Tolerance (degrees) inside which a pose counts as "looking at the screen"
CENTER_TOL = 15


def pose_to_class(tilt: int, pan: int) -> str:
    """Map (tilt, pan) angles from the head-pose dataset to one of CLASSES.

    Verified visually on the dataset (see docs/report):
      tilt > 0 -> head UP,  tilt < 0 -> head DOWN
      pan  > 0 -> face turned towards the image's LEFT  side  -> class 'left'
      pan  < 0 -> face turned towards the image's RIGHT side  -> class 'right'
    Labels are expressed from the *camera's* point of view (what the webcam sees).
    The web UI mirrors them so the student sees their own left/right.
    If the dominant deviation is <= CENTER_TOL -> 'center' (looking at screen).
    """
    if abs(pan) <= CENTER_TOL and abs(tilt) <= CENTER_TOL:
        return "center"
    if abs(pan) >= abs(tilt):
        return "left" if pan > 0 else "right"
    return "up" if tilt > 0 else "down"


# extra landmark indices for explicit head-pose geometry
FOREHEAD = 10
CHIN = 152
LEFT_CHEEK = 234      # image-left face border
RIGHT_CHEEK = 454     # image-right face border
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
N_GEO = 10


def geometric_features(pts: np.ndarray) -> np.ndarray:
    """10 explicit head-pose cues computed from nose-centred, eye-scaled landmarks.

    They encode yaw (left/right turn), pitch (up/down) and roll in a low-dimensional,
    person-independent way, which helps the MLP generalise from a small dataset.
    `pts` must already be centred on the nose tip and divided by the inter-ocular distance.
    """
    le, re = pts[LEFT_EYE_OUTER], pts[RIGHT_EYE_OUTER]
    eye_mid = (le + re) / 2
    lc, rc = pts[LEFT_CHEEK], pts[RIGHT_CHEEK]
    fh, ch = pts[FOREHEAD], pts[CHIN]
    ml, mr = pts[MOUTH_LEFT], pts[MOUTH_RIGHT]
    mouth_mid = (ml + mr) / 2
    face_w = np.linalg.norm(rc[:2] - lc[:2]) + 1e-6
    face_h = np.linalg.norm(ch[:2] - fh[:2]) + 1e-6
    return np.array([
        # yaw cues: nose is at origin, so eye-mid / cheek offsets measure horizontal asymmetry
        -eye_mid[0],                                   # nose x relative to eyes
        (np.linalg.norm(rc[:2]) - np.linalg.norm(lc[:2])) / face_w,   # cheek asymmetry
        (re[2] - le[2]),                               # eye depth difference
        (rc[2] - lc[2]),                               # cheek depth difference
        # pitch cues
        -eye_mid[1],                                   # nose y relative to eyes
        mouth_mid[1],                                  # mouth below nose
        (np.linalg.norm(ch[:2]) - np.linalg.norm(fh[:2])) / face_h,   # chin vs forehead distance
        (fh[2] - ch[2]),                               # forehead/chin depth difference
        # roll + aspect
        np.arctan2(re[1] - le[1], re[0] - le[0]),      # eye-line angle
        face_w / face_h,                               # apparent aspect ratio
    ], dtype=np.float32)


def landmarks_to_features(lm: np.ndarray) -> np.ndarray:
    """Convert (N>=468, 3) normalised MediaPipe landmarks to a 1-D feature vector.

    Steps (translation + scale invariant):
      1. keep first 468 points (x, y, z)
      2. subtract nose-tip so the face is centred at the origin
      3. divide by inter-ocular distance (outer eye corners) so face size does not matter
      4. flatten -> 1404 floats, then append 10 explicit geometry cues -> 1414 floats
    """
    pts = np.asarray(lm, dtype=np.float32)[:N_LANDMARKS].copy()
    pts -= pts[NOSE_TIP]
    scale = np.linalg.norm(pts[RIGHT_EYE_OUTER, :2] - pts[LEFT_EYE_OUTER, :2]) + 1e-6
    pts /= scale
    return np.concatenate([pts.reshape(-1), geometric_features(pts)])


FEATURE_DIM = N_LANDMARKS * 3 + N_GEO
