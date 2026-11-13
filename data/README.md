# Data

Raw and processed data are **not** committed (see `.gitignore`). Re-create everything with:

```bash
python scripts/download_data.py          # downloads both datasets from the Hugging Face Hub
python scripts/prepare_phone_dataset.py  # COCO -> YOLO format for Ultralytics
python scripts/extract_landmarks.py      # head-pose images -> MediaPipe landmark features
```

| Dataset | Source | Used for | Size |
|---|---|---|---|
| Head-Pose Image Database | https://huggingface.co/datasets/StevenLe456/head-pose | Attention / gaze-direction classifier | 2,790 unique (tilt, pan) poses (1 of 5 augmented copies per pose) |
| Phone detection (COCO) | https://huggingface.co/datasets/harshdadiya-wappnet/phone_detection | Fine-tuning YOLOv8n phone detector | 529 train / 51 valid / 25 test images |

Layout after running the scripts:

```
data/
├── raw/
│   ├── head_pose/
│   │   ├── images/00000.jpg …          # 384x288 face images
│   │   └── labels.csv                   # idx, file, tilt, pan
│   └── phone_detection/{train,valid,test}/   # images + _annotations.coco.json
└── processed/
    ├── attention_features.npz           # X (n,1404), y, tilt, pan, flipped, file
    └── phone_yolo/                      # images/, labels/, data.yaml
```
