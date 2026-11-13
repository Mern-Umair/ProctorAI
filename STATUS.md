# ProctorAI – work status (updated 2026-10-31 11:30)

## DONE ✅
- Datasets downloaded & verified (`data/raw/head_pose` 2,790 imgs + labels; `data/raw/phone_detection` 529/51/25).
- Phone dataset converted to YOLO format (`data/processed/phone_yolo`).
- Landmarks extracted (`data/processed/attention_features.npz`, 4,599 samples, raw landmarks `L` included).
- **Attention MLP trained (final):** 1414-d features, hidden 512-256, 100 epochs.
  Test accuracy **89.5 %**, macro-F1 **0.880**, attentive-vs-away **95.4 %**. Grid search in `models/attention_grid.json`.
  Artefacts: `models/attention_mlp.{pt,onnx}`, `attention_norm.json`, `attention_metrics.json`, figures in `docs/figures/`.
- **Detector exported:** COCO YOLOv8n → `models/detector_yolov8n.onnx` (320 px, 12.7 MB), copied to `web/models/`.
- COCO baseline on phone test split: mAP50 0.779, mAP50-95 0.639, P 0.949, R 0.519.
- Web app written (`web/`): index.html, demo.html, css, js (config/features/yolo/proctor/landing). JS syntax-checked.
  Python↔JS feature parity verified (max diff 2e-7); ONNX↔PyTorch parity verified.
  `web/models/` has attention_mlp.onnx, attention_norm.json, attention_metrics.json, detector_yolov8n.onnx.
  `web/assets/figures/` has pipeline.png, attention_confusion_matrix.png, attention_training_curves.png.
- Docs drafted: `docs/report.md` (placeholders filled by `scripts/build_report.py`), `docs/presentation_script.md`,
  `docs/linkedin_post.md`, `scripts/build_slides.py`, `scripts/build_report.py`, README.md, LICENSE, data/README.md.
- Env: `.venv` (Python 3.14) has torch, ultralytics, mediapipe, onnx, onnxruntime, opencv, matplotlib, pandas, seaborn,
  python-pptx, python-docx. Local wheels cached in `.wheels/`.

## DONE TODAY ✅
- `docs/ProctorAI_Report.docx` + `.pdf` built (build_report.py); `docs/ProctorAI_Presentation.pptx` built (15 slides, previews in docs/slide_previews).
- Docs/README/slides updated to final architecture (1414 → 512 → 256 → 5, 100 epochs, grid on val).
- `docs/DEPLOY.md` written (GitHub + Vercel + links + video + LinkedIn steps).
- Local server: `cd web && python -m http.server 3000` → http://localhost:3000/demo.html

## DONE 2026-10-31 15:00 ✅
- Phone fine-tune finished (15 ep, 320 px): test mAP@50 **92.7 %** vs COCO baseline 77.9 %; recall 81.5 % vs 51.9 %.
  `models/phone_yolov8n.{pt,onnx}`, `models/phone_metrics.json` (+ copied to web/models), figures `docs/figures/phone_*`.
- Report, slides, README, presentation script, LinkedIn post all updated with final numbers and rebuilt.

## TODO (remaining, in order)
1. Test web app locally: `cd web && python -m http.server 3000` → open http://localhost:3000/demo.html in Chrome, test all 4 violations; tune thresholds in `web/js/config.js` if needed. Take screenshot → `docs/figures/web_demo.png`.
3. `python scripts/build_report.py` → `docs/ProctorAI_Report.docx/.pdf`; `python scripts/build_slides.py` → `docs/ProctorAI_Presentation.pptx`. Review both.
4. Fill results in README.md (section 5).
5. User creates GitHub repo + Vercel: `git init`, commit, push; Vercel → import repo, root directory = `web`.
   Then fill real links in `web/js/config.js` (github, report, slides, youtube) and redeploy.
6. User records voice-over (script in `docs/presentation_script.md`) → YouTube (unlisted) → add link → LinkedIn post (`docs/linkedin_post.md`).

Identity: slides/report = Umair Tahir, SU92-BSCSM-F23-393, Section 7B. Website/GitHub stay generic.
