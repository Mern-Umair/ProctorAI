"""Draw the system-architecture diagram used in the report / slides (docs/figures/pipeline.png)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "pipeline.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(13, 5.2))
ax.set_xlim(0, 13)
ax.set_ylim(0, 5.2)
ax.axis("off")

C = {"in": "#dbeafe", "mp": "#ede9fe", "ours": "#dcfce7", "yolo": "#fef3c7", "logic": "#fee2e2", "out": "#e0f2fe"}


def box(x, y, w, h, title, sub, color, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=color, ec="#334155", lw=1.4))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=10.5,
            fontweight="bold" if bold else "normal", color="#0f172a")
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8.3, color="#475569")


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 lw=1.4, color="#334155"))


# input
box(0.3, 2.0, 1.9, 1.2, "Webcam frame", "640×480 · ~30 fps\nstays on the device", C["in"], True)

# attention path (top)
box(3.0, 3.5, 2.4, 1.2, "MediaPipe landmarks", "FaceLandmarker (WASM)\n478 points · ≤4 faces", C["mp"])
box(6.0, 3.5, 2.4, 1.2, "Feature engineering", "centre on nose · ÷ inter-ocular\n→ 1404-d vector", C["mp"])
box(9.0, 3.5, 2.4, 1.2, "Attention MLP  (trained)", "PyTorch → ONNX\ncenter/left/right/up/down", C["ours"], True)

# object path (bottom)
box(3.0, 0.5, 2.4, 1.2, "Letterbox 320×320", "every 350 ms", C["yolo"])
box(6.0, 0.5, 2.4, 1.2, "YOLOv8n detector", "ONNX Runtime Web\nperson · cell phone", C["yolo"], True)
box(9.0, 0.5, 2.4, 1.2, "Boxes + NMS", "phone conf ≥ 0.40\nperson conf ≥ 0.50", C["yolo"])

# event engine + output
box(11.9, 2.0, 1.0, 1.2, "", "", C["logic"])
ax.text(12.4, 2.62, "Event\nengine", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0f172a")
box(9.0, 2.0, 2.4, 1.2, "Flags per frame", "looking_away · phone\nmultiple_persons · no_face", C["logic"])

arrow(2.2, 2.9, 3.0, 4.0)
arrow(2.2, 2.3, 3.0, 1.2)
arrow(5.4, 4.1, 6.0, 4.1)
arrow(8.4, 4.1, 9.0, 4.1)
arrow(5.4, 1.1, 6.0, 1.1)
arrow(8.4, 1.1, 9.0, 1.1)
arrow(10.2, 3.5, 10.2, 3.2)
arrow(10.2, 1.7, 10.2, 2.0)
arrow(11.4, 2.6, 11.9, 2.6)

ax.text(12.4, 1.55, "≥1 s debounce → violation\ntimeline · penalties\nintegrity score = 100 − Σ",
        ha="center", va="top", fontsize=8, color="#475569")
ax.text(6.5, 5.05, "ProctorAI – real-time, fully client-side proctoring pipeline", ha="center",
        va="center", fontsize=12.5, fontweight="bold", color="#0f172a")

fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print("saved", OUT)
