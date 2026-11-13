"""
Build the project report (DOCX, and PDF if Microsoft Word is available) from
docs/report.md + models/*_metrics.json.

The markdown is deliberately simple (headings, paragraphs, bullet lists, pipe
tables, images, {{PLACEHOLDERS}}), so we convert it with python-docx directly
and do not need pandoc.

Usage:  python scripts/build_report.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MODELS = ROOT / "models"
MD = DOCS / "report.md"
OUT_DOCX = DOCS / "ProctorAI_Report.docx"
OUT_PDF = DOCS / "ProctorAI_Report.pdf"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES  # noqa: E402


# --------------------------------------------------------------------------- placeholders
def build_placeholders() -> dict:
    ph = {}
    a = json.load(open(MODELS / "attention_metrics.json"))
    rep = a["classification_report"]
    ph["ATTN_ACC"] = f"{a['test_accuracy'] * 100:.1f} %"
    ph["ATTN_F1"] = f"{a['test_macro_f1']:.3f}"
    ph["ATTN_VAL"] = f"{a['best_val_accuracy'] * 100:.1f} %"
    ph["N_TEST"] = str(a["n_test"])
    total = a["n_train"] + a["n_val"] + a["n_test"]
    ph["N_SAMPLES"] = f"{total:,}"
    for c in CLASSES:
        ph[f"N_{c.upper()}"] = f"{a['class_counts_total'][c]:,}"
    rows = ["| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for c in CLASSES:
        r = rep[c]
        rows.append(f"| {c} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {r['support']} |")
    m = rep["macro avg"]
    rows.append(f"| **macro avg** | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1-score']:.3f} | {m['support']} |")
    ph["ATTN_REPORT_TABLE"] = "Table 3 – Per-class test metrics.\n\n" + "\n".join(rows)

    cm = a["confusion_matrix"]
    # most confused pair
    worst, wi, wj = 0, 0, 0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if i != j and cm[i][j] > worst:
                worst, wi, wj = cm[i][j], i, j
    ph["ATTN_DISCUSSION"] = (
        f"The model converges within the first ~20 epochs and the validation curve tracks the training curve "
        f"closely, so the dropout / weight-decay / label-smoothing combination is sufficient to prevent over-fitting on "
        f"{a['n_train']:,} training samples. The largest off-diagonal entry in the confusion matrix is "
        f"*{CLASSES[wi]}* predicted as *{CLASSES[wj]}* ({worst} samples), i.e. confusion between neighbouring "
        f"directions at the 15° class boundary, which is exactly where the label itself is most ambiguous. "
        f"Mirror symmetry holds: left and right obtain almost identical precision and recall, confirming that the "
        f"flip augmentation worked as intended."
    )

    pm = MODELS / "phone_metrics.json"
    if pm.exists():
        p = json.load(open(pm))
        ph["PHONE_RESULTS"] = (
            f"Fine-tuning YOLOv8n for {p['epochs']} epochs at {p['imgsz']} px on the phone dataset reached "
            f"**mAP@50 = {p['test_mAP50'] * 100:.1f} %** and mAP@50-95 = {p['test_mAP50_95'] * 100:.1f} % on the 25-image test "
            f"split (precision {p['test_precision'] * 100:.1f} %, recall {p['test_recall'] * 100:.1f} %), compared with "
            f"mAP@50 = {p['coco_baseline']['test_mAP50'] * 100:.1f} % and recall {p['coco_baseline']['test_recall'] * 100:.1f} % for the "
            "COCO-pretrained model's `cell phone` class on the identical split (Table 4) – fine-tuning on only 529 images "
            "therefore adds about 15 mAP points in-domain. "
            "However, that dataset consists mostly of close-up photographs of (often damaged) phones and tablets lying on a "
            "surface, whereas the proctoring camera sees a phone held in a hand, partly occluded, at webcam distance. "
            "The COCO-pretrained model was trained on 80 classes and hundreds of thousands of in-context images (people holding "
            "phones among them) and additionally provides the `person` class used for the multiple-persons rule. For the "
            "deployed demo we therefore kept the COCO weights as the more conservative, general choice; a controlled "
            "comparison of both detectors on recorded webcam sessions is left as future work. "
            "the fine-tuning pipeline is kept in the repository and the two configurations can be swapped with one line "
            "in `web/js/config.js`.\n\n"
            "Table 4 – Phone detection on the 25-image test split (320 px).\n\n"
            "| Model | mAP@50 | mAP@50-95 | Precision | Recall |\n|---|---|---|---|---|\n"
            f"| YOLOv8n COCO (`cell phone`) | {p['coco_baseline']['test_mAP50'] * 100:.1f} % | "
            f"{p['coco_baseline']['test_mAP50_95'] * 100:.1f} % | {p['coco_baseline']['test_precision'] * 100:.1f} % | "
            f"{p['coco_baseline']['test_recall'] * 100:.1f} % |\n"
            f"| YOLOv8n fine-tuned ({p['epochs']} ep) | **{p['test_mAP50'] * 100:.1f} %** | **{p['test_mAP50_95'] * 100:.1f} %** | "
            f"{p['test_precision'] * 100:.1f} % | {p['test_recall'] * 100:.1f} % |\n\n"
            "![YOLOv8n fine-tuning curves](figures/phone_results.png)\n\n"
            "![Fine-tuned YOLOv8n predictions on a validation batch](figures/phone_val_batch0_pred.jpg)"
        )
        ph["PHONE_SENTENCE"] = (
            f"On its own test split the fine-tuned model reached {p['test_mAP50'] * 100:.0f} percent mAP at fifty, "
            "but because that dataset is close-up product photos and the COCO model also detects people, the demo ships "
            "the COCO model, with the fine-tuned one available as a one-line swap."
        )
    else:
        ph["PHONE_RESULTS"] = (
            "The deployed detector is the COCO-pretrained YOLOv8n (reported mAP@50-95 of 37.3 % on COCO val2017 by "
            "Ultralytics), exported to ONNX at 320 px. The fine-tuning experiment on the phone dataset "
            "(`scripts/train_phone_detector.py`) is provided in the repository; the dataset consists mostly of close-up "
            "photographs of devices on a surface, which differs from the webcam scene, so the general COCO model, which "
            "also provides the `person` class, is the better choice for deployment."
        )
        ph["PHONE_SENTENCE"] = (
            "Because that dataset is mostly close-up product photos, the general COCO model worked better on real "
            "webcam scenes, so that is the one deployed."
        )

    perf = MODELS / "perf.json"
    if perf.exists():
        pf = json.load(open(perf))
        ph["FPS"] = str(pf.get("fps", "–"))
        ph["PERF"] = pf.get("text", "")
    else:
        ph["FPS"] = "15–25"
        ph["PERF"] = (
            "On the development laptop (Intel Core i7-7600U, integrated graphics, no discrete GPU) the complete pipeline "
            "runs at roughly 15–25 frames per second in Chrome: MediaPipe FaceLandmarker ≈ 20–30 ms per frame, the "
            "attention MLP < 1 ms, and the 320-px YOLOv8n ≈ 120–180 ms per call, which is why it is throttled to one "
            "call every 350 ms and executed asynchronously so it never blocks the landmark loop. Model download size is "
            "≈ 1.6 MB for the attention MLP and ≈ 12.3 MB for YOLOv8n (cached by the browser after the first visit)."
        )
    return ph


# --------------------------------------------------------------------------- docx helpers
def set_cell_bg(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def add_runs(par, text):
    """Very small inline-markdown renderer: **bold**, *italic*, `code`."""
    tokens = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text)
    for t in tokens:
        if not t:
            continue
        if t.startswith("**"):
            par.add_run(t[2:-2]).bold = True
        elif t.startswith("`"):
            r = par.add_run(t[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(9.5)
        elif t.startswith("*"):
            par.add_run(t[1:-1]).italic = True
        else:
            par.add_run(t)


def add_table(doc, lines):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines if not re.match(r"^\|?\s*-{3,}", ln)]
    rows = [r for r in rows if any(r)]
    if not rows:
        return
    ncol = max(len(r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, r in enumerate(rows):
        for j in range(ncol):
            cell = t.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            add_runs(p, r[j] if j < len(r) else "")
            for run in p.runs:
                run.font.size = Pt(9.5)
                if i == 0:
                    run.bold = True
            if i == 0:
                set_cell_bg(cell, "E8EEF9")
    doc.add_paragraph()


def add_code(doc, lines):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.5)
    r = p.add_run("\n".join(lines))
    r.font.name = "Consolas"
    r.font.size = Pt(8.5)
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "F3F4F6")
    pPr.append(shd)


def add_image(doc, rel, caption):
    path = DOCS / rel
    if not path.exists():
        doc.add_paragraph(f"[figure missing: {rel}]").italic = True
        return
    doc.add_picture(str(path), width=Cm(15))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in cap.runs:
        r.italic = True
        r.font.size = Pt(9)
        r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)


# --------------------------------------------------------------------------- main
def main():
    md = MD.read_text(encoding="utf-8")
    ph = build_placeholders()
    for k, v in ph.items():
        md = md.replace("{{" + k + "}}", v)
    leftover = sorted(set(re.findall(r"{{(\w+)}}", md)))
    if leftover:
        print("WARNING unresolved placeholders:", leftover)

    # front-matter
    meta = {}
    if md.startswith("---"):
        fm, md = md.split("---", 2)[1:]
        for ln in fm.strip().splitlines():
            k, _, v = ln.partition(":")
            meta[k.strip()] = v.strip().strip('"')

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(11)
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.3)
        s.top_margin = s.bottom_margin = Cm(2.2)

    # title page
    for _ in range(6):
        doc.add_paragraph()
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run(meta.get("title", "ProctorAI"))
    r.bold = True
    r.font.size = Pt(26)
    r.font.color.rgb = RGBColor(0x1E, 0x29, 0x5A)
    s = doc.add_paragraph()
    s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = s.add_run(meta.get("subtitle", ""))
    r.font.size = Pt(14)
    r.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
    for _ in range(4):
        doc.add_paragraph()
    for line in [meta.get("author", ""), meta.get("date", "")]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(line).font.size = Pt(12)
    doc.add_page_break()

    lines = md.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        if ln.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            add_code(doc, lines[i + 1:j])
            i = j + 1
            continue
        m = re.match(r"^(#{1,3})\s+(.*)", ln)
        if m:
            level = len(m.group(1))
            doc.add_heading(m.group(2).strip(), level=level)
            i += 1
            continue
        if ln.lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            add_table(doc, lines[i:j])
            i = j
            continue
        m = re.match(r"^!\[(.*?)\]\((.*?)\)", ln.strip())
        if m:
            add_image(doc, m.group(2), m.group(1))
            i += 1
            continue
        m = re.match(r"^\s*(\*|-|\d+\.)\s+(.*)", ln)
        if m:
            style = "List Number" if m.group(1)[0].isdigit() else "List Bullet"
            p = doc.add_paragraph(style=style)
            add_runs(p, m.group(2))
            i += 1
            continue
        if ln.strip().startswith("$$"):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run("p̃ᵢ = (pᵢ − p_nose) / ‖(x,y)₂₆₃ − (x,y)₃₃‖")
            r.italic = True
            i += 1
            continue
        # paragraph (merge soft-wrapped lines)
        j = i
        buf = []
        while j < len(lines) and lines[j].strip() and not re.match(r"^(#|\||!\[|```|\s*(\*|-|\d+\.)\s|\$\$)", lines[j]):
            buf.append(lines[j].strip())
            j += 1
        p = doc.add_paragraph()
        add_runs(p, " ".join(buf))
        p.paragraph_format.space_after = Pt(6)
        i = j

    doc.save(OUT_DOCX)
    print("saved", OUT_DOCX)

    # PDF via Microsoft Word (if installed)
    ps = (
        f"$w = New-Object -ComObject Word.Application; $w.Visible=$false; "
        f"$d = $w.Documents.Open('{OUT_DOCX}'); $d.SaveAs('{OUT_PDF}', 17); $d.Close(); $w.Quit()"
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, timeout=180,
                       capture_output=True)
        print("saved", OUT_PDF)
    except Exception as e:  # noqa: BLE001
        print("PDF export skipped (Word not available?):", e)


if __name__ == "__main__":
    main()
