"""
Build the presentation (docs/ProctorAI_Presentation.pptx) with python-pptx.
Numbers are read from models/*_metrics.json so the deck always matches the trained models.

Usage:  python scripts/build_slides.py
"""
import csv
import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIG = DOCS / "figures"
MODELS = ROOT / "models"
OUT = DOCS / "ProctorAI_Presentation.pptx"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, pose_to_class  # noqa: E402

AUTHOR = "Umair Tahir"
ROLL = "SU92-BSCSM-F23-393"
SECTION = "Section 7B"
COURSE = "Deep Learning – Semester Project"

NAVY = RGBColor(0x0B, 0x10, 0x20)
INDIGO = RGBColor(0x63, 0x66, 0xF1)
VIOLET = RGBColor(0x8B, 0x5C, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xF4, 0xF6, 0xFB)
INK = RGBColor(0x11, 0x18, 0x27)
MUTED = RGBColor(0x5B, 0x66, 0x7A)
GREEN = RGBColor(0x16, 0xA3, 0x4A)
AMBER = RGBColor(0xD9, 0x77, 0x06)
RED = RGBColor(0xDC, 0x26, 0x26)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
W, H = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


# --------------------------------------------------------------------------- helpers
def rect(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
    s.shadow.inherit = False
    return s


def text(slide, x, y, w, h, txt, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT, font="Calibri"):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    lines = txt if isinstance(txt, list) else [txt]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = ln
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = font
    return tb


def bullets(slide, x, y, w, h, items, size=18, color=INK, gap=8):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        sub = it.startswith("  ")
        r = p.add_run()
        r.text = ("      – " if sub else "•  ") + it.strip()
        r.font.size = Pt(size - 3 if sub else size)
        r.font.color.rgb = MUTED if sub else color
        r.font.name = "Calibri"
    return tb


def header(slide, title, n):
    rect(slide, 0, 0, W, Inches(1.1), NAVY)
    rect(slide, 0, Inches(1.1), W, Inches(0.06), INDIGO)
    text(slide, Inches(0.6), Inches(0.22), Inches(10.5), Inches(0.8), title, 30, True, WHITE)
    text(slide, Inches(11.2), Inches(0.35), Inches(1.6), Inches(0.5), f"ProctorAI  ·  {n:02d}", 12, False,
         RGBColor(0xA5, 0xB4, 0xFC), PP_ALIGN.RIGHT)
    rect(slide, 0, Inches(1.16), W, H - Inches(1.16), LIGHT)
    # footer
    text(slide, Inches(0.6), H - Inches(0.45), Inches(8), Inches(0.35),
         f"{AUTHOR}  ·  {ROLL}  ·  {SECTION}  ·  {COURSE}", 10, False, MUTED)


def card(slide, x, y, w, h, title=None, fill=WHITE):
    s = rect(slide, x, y, w, h, fill, RGBColor(0xE2, 0xE8, 0xF0), MSO_SHAPE.ROUNDED_RECTANGLE)
    s.adjustments[0] = 0.06
    if title:
        text(slide, x + Inches(0.25), y + Inches(0.15), w - Inches(0.5), Inches(0.5), title, 16, True, INDIGO)
    return s


def picture(slide, path, x, y, w=None, h=None):
    if not Path(path).exists():
        text(slide, x, y, w or Inches(4), Inches(0.5), f"[missing: {Path(path).name}]", 12, False, RED)
        return None
    kw = {}
    if w:
        kw["width"] = w
    if h:
        kw["height"] = h
    return slide.shapes.add_picture(str(path), x, y, **kw)


def table(slide, x, y, w, rows, col_w=None, size=13, header_fill=INDIGO):
    nrows, ncols = len(rows), len(rows[0])
    shp = slide.shapes.add_table(nrows, ncols, x, y, w, Inches(0.4) * nrows)
    t = shp.table
    if col_w:
        for j, cw in enumerate(col_w):
            t.columns[j].width = cw
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = t.cell(i, j)
            c.text = str(v)
            p = c.text_frame.paragraphs[0]
            p.runs[0].font.size = Pt(size)
            p.runs[0].font.name = "Calibri"
            p.runs[0].font.bold = i == 0
            p.runs[0].font.color.rgb = WHITE if i == 0 else INK
            c.fill.solid()
            c.fill.fore_color.rgb = header_fill if i == 0 else (WHITE if i % 2 else RGBColor(0xF1, 0xF5, 0xF9))
            c.margin_top = c.margin_bottom = Emu(40000)
    return t


# --------------------------------------------------------------------------- data
a = json.load(open(MODELS / "attention_metrics.json"))
rep = a["classification_report"]
acc = f"{a['test_accuracy'] * 100:.1f}%"
f1 = f"{a['test_macro_f1']:.3f}"
n_total = a["n_train"] + a["n_val"] + a["n_test"]
phone = json.load(open(MODELS / "phone_metrics.json")) if (MODELS / "phone_metrics.json").exists() else None

# pick one example image per class for the label slide
examples = {}
lab = ROOT / "data" / "raw" / "head_pose" / "labels.csv"
if lab.exists():
    for r in csv.DictReader(open(lab)):
        t, p = int(r["tilt"]), int(r["pan"])
        c = pose_to_class(t, p)
        key = (c, abs(p) + abs(t))
        if c not in examples or (c == "center" and (t, p) == (0, 0)) or (c != "center" and abs(p) + abs(t) == 60):
            examples[c] = (ROOT / "data" / "raw" / "head_pose" / "images" / r["file"], t, p)

# =========================================================================== slides
# 1 ── title
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, W, H, NAVY)
rect(s, 0, Inches(5.1), W, Inches(0.08), INDIGO)
text(s, Inches(0.9), Inches(1.5), Inches(11.5), Inches(1.2), "🛡️ ProctorAI", 54, True, WHITE)
text(s, Inches(0.9), Inches(2.6), Inches(11.5), Inches(1.0), "Real-Time AI Exam Proctoring System", 32, False,
     RGBColor(0xC7, 0xD2, 0xFE))
text(s, Inches(0.9), Inches(3.5), Inches(11.5), Inches(0.9),
     "Looking away · Phone · Multiple persons · Face missing  —  detected live, 100% in the browser", 18, False,
     RGBColor(0x98, 0xA2, 0xB8))
text(s, Inches(0.9), Inches(5.4), Inches(8), Inches(0.5), AUTHOR, 24, True, WHITE)
text(s, Inches(0.9), Inches(5.95), Inches(8), Inches(0.5), f"Roll No {ROLL}   ·   {SECTION}", 18, False,
     RGBColor(0xC7, 0xD2, 0xFE))
text(s, Inches(0.9), Inches(6.45), Inches(8), Inches(0.5), COURSE, 16, False, RGBColor(0x98, 0xA2, 0xB8))
for i, (lbl, col) in enumerate([("PyTorch", INDIGO), ("YOLOv8", VIOLET), ("MediaPipe", INDIGO), ("ONNX Web", VIOLET), ("Vercel", INDIGO)]):
    b = rect(s, Inches(9.0 + 0.0) + Inches(0.0), Inches(5.45 + i * 0.42), Inches(1.9), Inches(0.34), col, None, MSO_SHAPE.ROUNDED_RECTANGLE)
    b.adjustments[0] = 0.5
    text(s, Inches(9.0), Inches(5.45 + i * 0.42), Inches(1.9), Inches(0.34), lbl, 12, True, WHITE, PP_ALIGN.CENTER)

# 2 ── problem
s = prs.slides.add_slide(BLANK)
header(s, "The problem: online exams, no invigilator", 2)
card(s, Inches(0.6), Inches(1.5), Inches(6.0), Inches(5.3), "Why it matters")
bullets(s, Inches(0.85), Inches(2.1), Inches(5.6), Inches(4.5), [
    "Exams & hiring tests moved online — the invigilator did not",
    "One teacher cannot watch 40 webcam tiles at once",
    "Commercial proctoring is expensive and streams video to servers",
    "  privacy · bandwidth · legal concerns",
    "Cheating behaviours are highly stereotyped and visible",
], 18)
card(s, Inches(6.9), Inches(1.5), Inches(5.8), Inches(5.3), "The four signals we detect")
for i, (ic, t, d, col) in enumerate([
    ("👀", "Looking away", "left / right / up / down for > 1 s", AMBER),
    ("📱", "Phone in view", "held in hand or on the desk", RED),
    ("👥", "Multiple persons", "second face or person box", VIOLET),
    ("🚫", "Face missing", "student leaves the frame", MUTED),
]):
    y = Inches(2.15 + i * 1.1)
    rect(s, Inches(7.15), y, Inches(0.09), Inches(0.85), col)
    text(s, Inches(7.4), y - Inches(0.05), Inches(0.6), Inches(0.6), ic, 24)
    text(s, Inches(8.0), y - Inches(0.02), Inches(4.5), Inches(0.45), t, 18, True)
    text(s, Inches(8.0), y + Inches(0.4), Inches(4.5), Inches(0.45), d, 13, False, MUTED)

# 3 ── objectives
s = prs.slides.add_slide(BLANK)
header(s, "Objectives", 3)
objs = [
    ("Real-time", "Detect all four behaviours live from a normal laptop webcam (≈ 20 fps)."),
    ("Temporal reasoning", "A blink or a quick glance must NOT be a violation — reason over ≥ 1 s windows."),
    ("100 % client-side", "Opens from a link, nothing to install, video never leaves the device."),
    ("Auditable output", "Timestamped timeline + integrity score a teacher can review — not a black-box verdict."),
]
for i, (t, d) in enumerate(objs):
    x = Inches(0.6 + (i % 2) * 6.2)
    y = Inches(1.6 + (i // 2) * 2.6)
    card(s, x, y, Inches(5.9), Inches(2.3))
    n = rect(s, x + Inches(0.3), y + Inches(0.35), Inches(0.7), Inches(0.7), INDIGO, None, MSO_SHAPE.OVAL)
    text(s, x + Inches(0.3), y + Inches(0.42), Inches(0.7), Inches(0.6), str(i + 1), 22, True, WHITE, PP_ALIGN.CENTER)
    text(s, x + Inches(1.2), y + Inches(0.35), Inches(4.4), Inches(0.6), t, 20, True)
    text(s, x + Inches(1.2), y + Inches(0.95), Inches(4.4), Inches(1.2), d, 14, False, MUTED)

# 4 ── datasets
s = prs.slides.add_slide(BLANK)
header(s, "Datasets (public, Hugging Face Hub)", 4)
card(s, Inches(0.6), Inches(1.5), Inches(6.0), Inches(5.3), "🧭 Head-Pose Image Database")
bullets(s, Inches(0.85), Inches(2.1), Inches(5.6), Inches(4.5), [
    "15 people × 93 known (tilt, pan) poses",
    "pan −90° … +90° (15° steps), tilt −90° … +90°",
    "2,790 unique images used (1 of 5 colour-augmented copies)",
    "→ MediaPipe landmarks + horizontal flip",
    f"→ {n_total:,} labelled landmark samples, 5 classes",
    "  StevenLe456/head-pose",
], 16)
card(s, Inches(6.9), Inches(1.5), Inches(5.8), Inches(5.3), "📱 Phone detection (COCO format)")
bullets(s, Inches(7.15), Inches(2.1), Inches(5.4), Inches(4.5), [
    "Roboflow export: 529 train / 51 valid / 25 test images",
    "589 phone bounding boxes in train",
    "Converted to YOLO format (scripts/prepare_phone_dataset.py)",
    "Used to fine-tune YOLOv8n (single class: phone)",
    "  harshdadiya-wappnet/phone_detection",
], 16)

# 5 ── label engineering
s = prs.slides.add_slide(BLANK)
header(s, "From head angles to attention classes", 5)
table(s, Inches(0.6), Inches(1.6), Inches(5.6), [
    ["Rule", "Class"],
    ["|pan| ≤ 15° and |tilt| ≤ 15°", "center (looking at screen)"],
    ["|pan| ≥ |tilt|, pan > 0", "left"],
    ["|pan| ≥ |tilt|, pan < 0", "right"],
    ["|tilt| > |pan|, tilt > 0", "up"],
    ["|tilt| > |pan|, tilt < 0", "down"],
], [Inches(3.0), Inches(2.6)], 13)
bullets(s, Inches(0.6), Inches(4.4), Inches(5.6), Inches(2.4), [
    "Sign convention verified visually on the images",
    "Every image also flipped horizontally with left ↔ right swapped → symmetric classifier",
    "Labels are from the camera's viewpoint; the UI mirrors them for the student",
], 14)
x0 = Inches(6.6)
for i, c in enumerate(CLASSES):
    if c in examples:
        p, t, pn = examples[c]
        cx = x0 + Inches((i % 3) * 2.2)
        cy = Inches(1.6 + (i // 3) * 2.6)
        picture(s, p, cx, cy, w=Inches(2.05))
        text(s, cx, cy + Inches(1.58), Inches(2.05), Inches(0.4), f"{c}  (tilt {t:+d}°, pan {pn:+d}°)", 11, True, INK,
             PP_ALIGN.CENTER)

# 6 ── architecture
s = prs.slides.add_slide(BLANK)
header(s, "System architecture", 6)
picture(s, FIG / "pipeline.png", Inches(0.5), Inches(1.45), w=Inches(12.3))

# 7 ── feature engineering
s = prs.slides.add_slide(BLANK)
header(s, "Feature engineering: invariant to position, size and camera", 7)
card(s, Inches(0.6), Inches(1.5), Inches(6.0), Inches(5.3), "468 landmarks → 1,404 numbers")
bullets(s, Inches(0.85), Inches(2.1), Inches(5.6), Inches(4.5), [
    "Take the first 468 MediaPipe points (x, y, z)",
    "Subtract the nose tip → face centred at the origin",
    "Divide by the inter-ocular distance (outer eye corners) → size-free",
    "Flatten → 1,404-d vector, standardise with training mean / std",
    "Identical code in Python (training) and JavaScript (browser)",
    "  scripts/common.py  ⇄  web/js/features.js",
], 16)
card(s, Inches(6.9), Inches(1.5), Inches(5.8), Inches(5.3), "Formula")
text(s, Inches(7.2), Inches(2.4), Inches(5.2), Inches(1.0), "p̃ᵢ  =  ( pᵢ − p_nose )  /  ‖ (x,y)₂₆₃ − (x,y)₃₃ ‖", 22, True, INK,
     PP_ALIGN.CENTER)
text(s, Inches(7.2), Inches(3.6), Inches(5.2), Inches(2.5), [
    "i = 1 … 468",
    "p_nose = landmark 1 (nose tip)",
    "33 / 263 = outer corners of the two eyes",
    "Result does not change if the face moves, comes closer, or the camera resolution changes.",
], 14, False, MUTED)

# 8 ── classifier
s = prs.slides.add_slide(BLANK)
header(s, "Attention classifier (PyTorch MLP) – trained in this project", 8)
cfg = a.get("config", {})
hid = cfg.get("hidden", [512, 256])
drop = cfg.get("dropout", 0.3)
arch_rows = [["Layer", "Output"], ["Input (standardised features)", str(a.get("feature_dim", 1414))]]
for hsz in hid:
    arch_rows.append([f"Linear → BatchNorm → ReLU → Dropout {drop}", str(hsz)])
arch_rows.append(["Linear (logits)", "5 classes"])
table(s, Inches(0.6), Inches(1.6), Inches(6.2), arch_rows, [Inches(4.4), Inches(1.8)], 14)
card(s, Inches(7.1), Inches(1.5), Inches(5.6), Inches(5.3), "Training recipe")
bullets(s, Inches(7.35), Inches(2.1), Inches(5.2), Inches(4.5), [
    f"AdamW, lr {cfg.get('lr', 2e-3):g}, weight-decay {cfg.get('wd', 1e-4):g}, cosine schedule",
    f"{a['epochs']} epochs, batch 64, best-validation checkpoint",
    f"Cross-entropy + class weights + label smoothing {cfg.get('smoothing', 0.05)}",
    "Augmentation: flip (label-swapped) + Gaussian landmark jitter",
    "Hyper-parameters chosen by a 6-config grid on the VALIDATION split only",
    "Split BY SOURCE IMAGE 80 / 10 / 10 → no leakage",
    f"  train {a['n_train']:,} · val {a['n_val']:,} · test {a['n_test']:,}",
    "Trains in ~2 minutes on CPU → exported to ONNX (≈3.4 MB)",
], 15)

# 9 ── results attention
s = prs.slides.add_slide(BLANK)
header(s, "Results: attention classifier", 9)
for i, (lbl, val, col) in enumerate([("Test accuracy", acc, GREEN), ("Macro-F1", f1, INDIGO),
                                      ("Best val. accuracy", f"{a['best_val_accuracy'] * 100:.1f}%", VIOLET)]):
    x = Inches(0.6 + i * 2.1)
    card(s, x, Inches(1.5), Inches(1.95), Inches(1.4))
    text(s, x, Inches(1.6), Inches(1.95), Inches(0.7), val, 26, True, col, PP_ALIGN.CENTER)
    text(s, x, Inches(2.3), Inches(1.95), Inches(0.5), lbl, 11, False, MUTED, PP_ALIGN.CENTER)
rows = [["Class", "Precision", "Recall", "F1", "n"]]
for c in CLASSES:
    r = rep[c]
    rows.append([c, f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1-score']:.3f}", r["support"]])
table(s, Inches(0.6), Inches(3.15), Inches(6.2), rows, [Inches(1.4), Inches(1.2), Inches(1.2), Inches(1.2), Inches(1.2)], 12)
picture(s, FIG / "attention_confusion_matrix.png", Inches(7.1), Inches(1.45), h=Inches(4.6))
text(s, Inches(7.1), Inches(6.1), Inches(5.6), Inches(0.8),
     "Errors concentrate between neighbouring directions at the 15° boundary — where the label itself is ambiguous.",
     12, False, MUTED)

# 10 ── detector
s = prs.slides.add_slide(BLANK)
header(s, "Object detector: YOLOv8n (phone + person)", 10)
card(s, Inches(0.6), Inches(1.5), Inches(6.0), Inches(5.3), "Deployed model")
bullets(s, Inches(0.85), Inches(2.1), Inches(5.6), Inches(4.5), [
    "YOLOv8n pre-trained on COCO (80 classes)",
    "We read two classes: person (0) and cell phone (67)",
    "Exported to ONNX, 320 × 320, static shape → ≈ 12 MB",
    "Runs every 350 ms, asynchronously, never blocks the face loop",
    "phone conf ≥ 0.40 · person conf ≥ 0.50 · NMS IoU 0.45",
], 16)
card(s, Inches(6.9), Inches(1.5), Inches(5.8), Inches(5.3), "Fine-tuning experiment")
if phone:
    bullets(s, Inches(7.15), Inches(2.1), Inches(5.4), Inches(4.5), [
        f"YOLOv8n fine-tuned {phone['epochs']} epochs @ {phone['imgsz']} px on the phone dataset",
        f"Test mAP@50: {phone['coco_baseline']['test_mAP50'] * 100:.1f}% (COCO) → {phone['test_mAP50'] * 100:.1f}% (fine-tuned)",
        f"Recall: {phone['coco_baseline']['test_recall'] * 100:.1f}% → {phone['test_recall'] * 100:.1f}% · Precision {phone['test_precision'] * 100:.1f}%",
        "Dataset = close-up product photos ≠ webcam scene",
        "→ demo ships the COCO model (general + gives 'person')",
        "Fine-tuned model = one-line swap in web/js/config.js",
    ], 15)
else:
    bullets(s, Inches(7.15), Inches(2.1), Inches(5.4), Inches(4.5), [
        "scripts/train_phone_detector.py fine-tunes YOLOv8n on the phone dataset",
        "Dataset = close-up product photos ≠ webcam scene (domain shift)",
        "COCO weights are the general, conservative choice for webcam scenes",
        "and additionally provide the 'person' class",
        "Swap models with one line in web/js/config.js",
    ], 15)

# 11 ── event engine
s = prs.slides.add_slide(BLANK)
header(s, "Temporal event engine & integrity score", 11)
card(s, Inches(0.6), Inches(1.5), Inches(6.0), Inches(5.3), "Debouncing")
bullets(s, Inches(0.85), Inches(2.1), Inches(5.6), Inches(4.5), [
    "Per-frame flags are noisy (blinks, motion blur, quick glances)",
    "A flag must stay active ≥ 1.0 s to OPEN a violation",
    "…and stay clear ≥ 0.6 s to CLOSE it",
    "Attention probabilities smoothed with an EMA (α = 0.35)",
    "Each event: type, start, end, duration → timeline",
], 16)
card(s, Inches(6.9), Inches(1.5), Inches(5.8), Inches(5.3), "Penalties  →  score = 100 − Σ")
table(s, Inches(7.15), Inches(2.15), Inches(5.3), [
    ["Violation", "Per event", "Per extra second"],
    ["📱 Phone", "15", "1.0"],
    ["👥 Multiple persons", "10", "0.5"],
    ["🚫 Face missing", "5", "0.5 (after 3 s)"],
    ["👀 Looking away", "3", "0.5 (after 3 s)"],
], [Inches(2.3), Inches(1.3), Inches(1.7)], 13)
text(s, Inches(7.15), Inches(4.7), Inches(5.3), Inches(1.5), [
    "≥ 85  →  Clean",
    "60 – 84  →  Suspicious",
    "< 60  →  High risk",
], 16, True, INK)

# 12 ── deployment
s = prs.slides.add_slide(BLANK)
header(s, "Deployment: static site on Vercel, inference in WebAssembly", 12)
card(s, Inches(0.6), Inches(1.5), Inches(6.0), Inches(5.3), "Stack")
bullets(s, Inches(0.85), Inches(2.1), Inches(5.6), Inches(4.5), [
    "HTML + CSS + ES-module JavaScript (no framework, no build step)",
    "MediaPipe Tasks Vision (FaceLandmarker) – WASM",
    "ONNX Runtime Web – runs both of our ONNX models",
    "Models served from the same origin, cached by the browser",
    "Vercel: git push → live URL in seconds",
], 16)
card(s, Inches(6.9), Inches(1.5), Inches(5.8), Inches(5.3), "What the site contains")
bullets(s, Inches(7.15), Inches(2.1), Inches(5.4), Inches(4.5), [
    "Landing page: problem, pipeline, dataset & code details, results",
    "Live demo: webcam, live counters, integrity score, event log",
    "Session report: table, timeline, download JSON / print PDF",
    "Links: GitHub repo, report, slides, YouTube video",
    "🔒 Privacy: no frame ever leaves the device",
], 16)

# 13 ── demo
s = prs.slides.add_slide(BLANK)
header(s, "Live demo", 13)
shot = FIG / "web_demo.png"
if shot.exists():
    picture(s, shot, Inches(0.6), Inches(1.5), h=Inches(5.2))
    bx = Inches(8.4)
else:
    bx = Inches(0.6)
card(s, bx, Inches(1.5), Inches(13.333) - bx - Inches(0.6), Inches(5.3), "Try to cheat – watch it get caught")
bullets(s, bx + Inches(0.25), Inches(2.1), Inches(13.333) - bx - Inches(1.1), Inches(4.5), [
    "Start session → status turns green: Monitoring",
    "Look left / right / up / down > 1 s → 👀 Looking away",
    "Pick up a phone → 📱 Phone detected (red box)",
    "A friend steps in → 👥 Multiple persons",
    "Leave the frame → 🚫 Face missing",
    "End session → full report + integrity score",
], 16)

# 14 ── limitations
s = prs.slides.add_slide(BLANK)
header(s, "Limitations & future work", 14)
card(s, Inches(0.6), Inches(1.5), Inches(6.0), Inches(5.3), "Honest limitations")
bullets(s, Inches(0.85), Inches(2.1), Inches(5.6), Inches(4.5), [
    "Head pose ≠ eye gaze – eyes-only glances are missed",
    "Assumes the camera is roughly at screen level",
    "A phone hidden below the desk is invisible",
    "Lab dataset: 15 people, indoor lighting",
    "Must support – never replace – human judgement",
], 16)
card(s, Inches(6.9), Inches(1.5), Inches(5.8), Inches(5.3), "Next steps")
bullets(s, Inches(7.15), Inches(2.1), Inches(5.4), Inches(4.5), [
    "Iris landmarks → true gaze estimation",
    "Audio: talking / whisper detection",
    "Per-session calibration of the 'center' pose",
    "Teacher dashboard for many students (server-side sessions)",
    "Larger webcam-domain phone dataset → re-fine-tune YOLO",
], 16)

# 15 ── conclusion
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, W, H, NAVY)
text(s, Inches(0.9), Inches(1.2), Inches(11.5), Inches(0.9), "Conclusion", 40, True, WHITE)
bullets(s, Inches(0.9), Inches(2.3), Inches(11.5), Inches(3.2), [
    f"End-to-end DL project: public datasets → trained PyTorch model ({acc} test accuracy) → ONNX → live web app",
    "YOLOv8n detector + temporal event engine → explainable timeline and integrity score",
    "Runs in real time, fully client-side, from a single link – nothing to install, nothing uploaded",
    "Code, weights, report and this deck are public on GitHub; demo deployed on Vercel",
], 18, WHITE, 12)
rect(s, Inches(0.9), Inches(5.3), Inches(11.5), Inches(0.04), INDIGO)
text(s, Inches(0.9), Inches(5.5), Inches(11.5), Inches(0.6), "Thank you — questions?", 28, True, WHITE)
text(s, Inches(0.9), Inches(6.2), Inches(11.5), Inches(0.5), f"{AUTHOR}  ·  {ROLL}  ·  {SECTION}  ·  {COURSE}", 14, False,
     RGBColor(0xC7, 0xD2, 0xFE))

prs.save(OUT)
print("saved", OUT, f"({len(prs.slides)} slides)")
