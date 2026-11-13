# 🛡️ ProctorAI – Real-Time AI Exam Proctoring System

> Deep Learning semester project. ProctorAI watches a student's webcam during an online exam and
> flags the four most common cheating signals **in real time, entirely in the browser**:
> 👀 looking away · 📱 phone in view · 👥 another person in frame · 🚫 student leaves the camera.

**Live demo:** _coming soon (Vercel)_ · **Report:** `docs/` · **Video:** _YouTube link_ · **Slides:** `docs/`

---

## 1. Problem statement

Online exams and remote hiring tests have no invigilator in the room. Human proctoring over video
does not scale (one proctor can watch only a handful of students) and uploading webcam streams to a
server raises privacy and bandwidth problems. ProctorAI replaces the invigilator with two small deep
learning models that run **client-side** (WebAssembly), produce a timestamped violation log, and
summarise the session into a single **integrity score**.

## 2. What it detects

| Signal | How | Model |
|---|---|---|
| Looking away (left / right / up / down) | 468 MediaPipe face landmarks → **our PyTorch MLP** classifies gaze direction | trained in this repo |
| Phone in view | YOLOv8n object detector (`cell phone` class) | COCO-pretrained; fine-tuning experiment included |
| Multiple persons | >1 face from MediaPipe **or** >1 `person` box from YOLO | – |
| Face missing | 0 faces for ≥ 1 s | – |

A state must persist for **≥ 1 s** before it becomes a violation (no false alarms for a blink or a
quick glance), and must be clear for 0.6 s before the event closes. Each event is penalised to give
`integrity score = 100 − Σ penalties` → *Clean (≥85) / Suspicious (60–84) / High risk (<60)*.

## 3. Datasets

| Dataset | Source | Use |
|---|---|---|
| Head-Pose Image Database (93 tilt/pan poses, 15 people) | [HF · StevenLe456/head-pose](https://huggingface.co/datasets/StevenLe456/head-pose) | attention classifier |
| Phone detection, COCO format (529 / 51 / 25) | [HF · harshdadiya-wappnet/phone_detection](https://huggingface.co/datasets/harshdadiya-wappnet/phone_detection) | YOLOv8n fine-tuning |

Angles → classes: `|pan| ≤ 15° and |tilt| ≤ 15°` → **center**, otherwise the dominant angle decides
**left / right / up / down** (labels are from the camera's viewpoint; verified visually on the images).
Each image is also horizontally flipped with left/right swapped, so the classifier is symmetric.

## 4. Models

### 4.1 Attention classifier (ours)
```
features : 468 landmarks (x,y,z) − nose tip, ÷ inter-ocular distance → 1404-d  + 10 head-pose geometry cues = 1414-d
model    : Linear(1414,512)-BN-ReLU-Drop(0.3) → Linear(512,256)-BN-ReLU-Drop(0.3) → Linear(256,5)
training : AdamW lr 2e-3, wd 1e-4, cosine LR, label smoothing 0.05, class-weighted CE,
           Gaussian landmark jitter, 100 epochs, split by source image (80/10/10),
           hyper-parameters picked by a 6-config grid on the validation split (models/attention_grid.json)
```
### 4.2 Object detector
YOLOv8n (COCO) exported to ONNX at 320 px. The web app reads classes `person` and `cell phone`.
`scripts/train_phone_detector.py` additionally fine-tunes YOLOv8n on the phone dataset (15 epochs, 320 px):
mAP@50 rises from **77.9 %** (COCO baseline, class `cell phone`) to **92.7 %** on the same test split
(`models/phone_yolov8n.{pt,onnx}`). The demo ships the COCO model because it also provides the `person`
class and is the more general choice for webcam scenes; swap in the fine-tuned one by changing two lines in `web/js/config.js`.

## 5. Results

_(from `models/*_metrics.json`; the website reads the same files)_

| Model | Metric | Value |
|---|---|---|
| Attention MLP (5 classes) | test accuracy | **89.5 %** |
| Attention MLP (5 classes) | macro-F1 | **0.880** |
| Attention MLP (attentive vs looking-away) | test accuracy | **95.4 %** |
| YOLOv8n COCO (deployed) | phone test mAP@50 / recall | 77.9 % / 51.9 % |
| YOLOv8n fine-tuned, 15 epochs @ 320 px | phone test mAP@50 / mAP@50-95 | **92.7 % / 72.6 %** (recall 81.5 %) |

Figures: `docs/figures/`

## 6. Repository layout
```
ProctorAI/
├── scripts/
│   ├── download_data.py          # fetch both datasets from the Hugging Face Hub
│   ├── prepare_phone_dataset.py  # COCO → YOLO format + data.yaml
│   ├── extract_landmarks.py      # MediaPipe landmarks → 1414-d features (+ flip augmentation)
│   ├── train_attention_model.py  # PyTorch MLP: train, evaluate, export .pt + .onnx
│   ├── train_phone_detector.py   # YOLOv8n fine-tune, test mAP, ONNX export
│   ├── export_detector.py        # export the deployed detector to ONNX for the browser
│   └── common.py                 # shared feature engineering (mirrored in web/js/features.js)
├── models/                       # trained weights (.pt, .onnx) + metrics (.json)
├── web/                          # static site deployed on Vercel (index.html + demo.html)
├── docs/                         # report, slides, figures
└── data/                         # datasets (not committed – see data/README.md)
```

## 7. Reproduce

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install -r requirements.txt

python scripts/download_data.py            # ~100 MB
python scripts/prepare_phone_dataset.py
python scripts/extract_landmarks.py        # MediaPipe on ~2.8k images x2 (flip)
python scripts/train_attention_model.py    # CPU, ~1 min
python scripts/train_phone_detector.py 30  # CPU, ~1-2 h (optional experiment)
python scripts/export_detector.py          # yolov8n COCO -> web/models/detector_yolov8n.onnx

# copy artefacts to the web app and serve it locally
cp models/attention_mlp.onnx models/attention_norm.json models/*_metrics.json web/models/
cd web && npx serve .                       # open http://localhost:3000/demo.html
```

## 8. Tech stack
PyTorch · Ultralytics YOLOv8 · MediaPipe Tasks (FaceLandmarker) · ONNX / ONNX Runtime Web ·
scikit-learn · vanilla JS (ES modules) · Vercel

## 9. Limitations & future work
* Gaze is inferred from head pose – eyes-only glances are not detected.
* Phone detection is appearance-based; a phone fully below the desk is invisible to the camera.
* Future: audio (talking) detection, multi-student teacher dashboard, server-side session storage.

## License
MIT – see `LICENSE`. Datasets keep their original licenses.
