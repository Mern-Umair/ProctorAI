"""
Step 2b - Train the attention / head-direction classifier (PyTorch MLP) on the
landmark features produced by extract_landmarks.py, evaluate it, and export
weights (.pt) + ONNX for in-browser inference.

Split is done BY SOURCE IMAGE so the flipped copy of a validation image can
never leak into the training set. (Pure numpy/torch - no scikit-learn needed.)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, FEATURE_DIM, landmarks_to_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "attention_features.npz"
MODELS = ROOT / "models"
DOCS = ROOT / "docs" / "figures"
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ----------------------------------------------------------------------------- metrics helpers
def group_split(groups, test_frac, seed):
    """Return (train_idx, test_idx) with whole groups kept together."""
    uniq = np.unique(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(uniq)
    n_test = int(round(len(uniq) * test_frac))
    test_groups = set(uniq[:n_test].tolist())
    mask = np.array([g in test_groups for g in groups])
    return np.where(~mask)[0], np.where(mask)[0]


def confusion(y_true, y_pred, k):
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_report(cm):
    """precision / recall / f1 / support per class + macro averages (like sklearn)."""
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    rep = {c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1-score": float(f1[i]),
               "support": int(cm[i].sum())} for i, c in enumerate(CLASSES)}
    rep["accuracy"] = float(tp.sum() / cm.sum())
    rep["macro avg"] = {"precision": float(prec.mean()), "recall": float(rec.mean()),
                        "f1-score": float(f1.mean()), "support": int(cm.sum())}
    return rep


def format_report(rep):
    lines = [f"{'':10s}{'precision':>10s}{'recall':>10s}{'f1-score':>10s}{'support':>9s}"]
    for c in CLASSES:
        r = rep[c]
        lines.append(f"{c:10s}{r['precision']:10.4f}{r['recall']:10.4f}{r['f1-score']:10.4f}{r['support']:9d}")
    m = rep["macro avg"]
    lines.append(f"{'accuracy':10s}{'':10s}{'':10s}{rep['accuracy']:10.4f}{m['support']:9d}")
    lines.append(f"{'macro avg':10s}{m['precision']:10.4f}{m['recall']:10.4f}{m['f1-score']:10.4f}{m['support']:9d}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- model
class AttentionMLP(nn.Module):
    def __init__(self, in_dim=FEATURE_DIM, hidden=(256, 128), n_classes=len(CLASSES), p=0.3):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main(epochs=60, bs=64, lr=1e-3, hidden=(256, 128), dropout=0.3, jitter=0.02, smoothing=0.05,
         wd=1e-4, save=True, tag=""):
    d = np.load(DATA, allow_pickle=True)
    y, groups = d["y"], d["src"]
    # always re-derive features from the raw landmarks so common.py is the single source of truth
    X = np.stack([landmarks_to_features(l) for l in d["L"]]).astype(np.float32)
    print("features:", X.shape, "classes:", np.bincount(y))

    # train / val / test split by source image (80 / 10 / 10)
    tr, rest = group_split(groups, 0.2, SEED)
    va_i, te_i = group_split(groups[rest], 0.5, SEED + 1)
    va, te = rest[va_i], rest[te_i]
    assert not (set(groups[tr]) & set(groups[te])) and not (set(groups[tr]) & set(groups[va]))
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}")

    # standardise with train statistics (saved for the browser)
    mean, std = X[tr].mean(0), X[tr].std(0) + 1e-6

    def norm(a):
        return (a - mean) / std

    Xt, Xv, Xte = (torch.tensor(norm(a), dtype=torch.float32) for a in (X[tr], X[va], X[te]))
    yt, yv, yte = (torch.tensor(a, dtype=torch.long) for a in (y[tr], y[va], y[te]))

    # class weights for imbalance
    counts = np.bincount(y[tr], minlength=len(CLASSES))
    weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32)

    model = AttentionMLP(hidden=hidden, p=dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(weight=weights, label_smoothing=smoothing)
    cfg = dict(epochs=epochs, bs=bs, lr=lr, hidden=list(hidden), dropout=dropout, jitter=jitter,
               smoothing=smoothing, wd=wd)
    print("config:", cfg)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_state = 0.0, None
    n = len(Xt)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        tot_loss, correct = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xt[idx], yt[idx]
            if jitter > 0:
                xb = xb + jitter * torch.randn_like(xb)  # tiny jitter = landmark noise augmentation
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
            correct += (out.argmax(1) == yb).sum().item()
        sched.step()

        model.eval()
        with torch.no_grad():
            vo = model(Xv)
            vloss = crit(vo, yv).item()
            vacc = (vo.argmax(1) == yv).float().mean().item()
        history["train_loss"].append(tot_loss / n)
        history["val_loss"].append(vloss)
        history["train_acc"].append(correct / n)
        history["val_acc"].append(vacc)
        if vacc > best_acc:
            best_acc, best_state = vacc, {k: v.clone() for k, v in model.state_dict().items()}
        if save and (ep % 5 == 0 or ep == 1):
            print(f"ep {ep:3d}  loss {tot_loss / n:.4f}  acc {correct / n:.4f} | "
                  f"val loss {vloss:.4f}  val acc {vacc:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(1).numpy()
    cm = confusion(y[te], pred, len(CLASSES))
    rep = per_class_report(cm)
    acc, f1 = rep["accuracy"], rep["macro avg"]["f1-score"]
    # what the proctor actually acts on: attentive (center) vs looking away (any other class)
    bin_true, bin_pred = (y[te] != 0), (pred != 0)
    bin_acc = float((bin_true == bin_pred).mean())
    print(f"\n{tag}TEST  accuracy {acc:.4f}   macro-F1 {f1:.4f}   (best val acc {best_acc:.4f})")
    print(f"      binary attentive-vs-away accuracy {bin_acc:.4f}")
    if not save:
        return {"val_acc": best_acc, "test_acc": acc, "test_f1": f1, "bin_acc": bin_acc, "cfg": cfg}
    print(format_report(rep))

    # ---------------- save artefacts
    MODELS.mkdir(exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "classes": CLASSES},
               MODELS / "attention_mlp.pt")
    json.dump({"mean": mean.tolist(), "std": std.tolist(), "classes": CLASSES},
              open(MODELS / "attention_norm.json", "w"))

    # export ONNX  (normalisation is done in JS using attention_norm.json)
    dummy = torch.zeros(1, FEATURE_DIM)
    torch.onnx.export(model, dummy, MODELS / "attention_mlp.onnx", input_names=["features"],
                      output_names=["logits"],
                      dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)

    metrics = {
        "test_accuracy": float(acc), "test_macro_f1": float(f1), "best_val_accuracy": float(best_acc),
        "test_binary_attentive_vs_away_accuracy": bin_acc, "feature_dim": int(FEATURE_DIM), "config": cfg,
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "class_counts_total": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
        "epochs": epochs, "confusion_matrix": cm.tolist(),
        "classification_report": rep, "history": history,
    }
    json.dump(metrics, open(MODELS / "attention_metrics.json", "w"), indent=2)

    # figures
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Attention classifier - test acc {acc:.3f}")
    plt.tight_layout()
    plt.savefig(DOCS / "attention_confusion_matrix.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["val_acc"], label="val")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(DOCS / "attention_training_curves.png", dpi=150)
    plt.close()
    print("saved models/attention_mlp.{pt,onnx}, attention_norm.json, attention_metrics.json, figures")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=str, default="256,128")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--smoothing", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--search", action="store_true", help="run a small hyper-parameter grid (selection on VAL only)")
    args = ap.parse_args()

    if args.search:
        grid = [
            dict(hidden=(256, 128), dropout=0.3, jitter=0.02, smoothing=0.05, lr=1e-3, epochs=60),
            dict(hidden=(256, 128), dropout=0.15, jitter=0.0, smoothing=0.0, lr=1e-3, epochs=80),
            dict(hidden=(512, 256), dropout=0.2, jitter=0.01, smoothing=0.0, lr=1e-3, epochs=80),
            dict(hidden=(512, 256), dropout=0.3, jitter=0.02, smoothing=0.05, lr=2e-3, epochs=100),
            dict(hidden=(128, 64), dropout=0.1, jitter=0.0, smoothing=0.0, lr=2e-3, epochs=80),
            dict(hidden=(512, 256, 128), dropout=0.2, jitter=0.01, smoothing=0.0, lr=1e-3, epochs=100, wd=1e-3),
        ]
        results = []
        for i, g in enumerate(grid):
            torch.manual_seed(SEED)
            results.append(main(save=False, tag=f"[{i}] ", **g))
        results.sort(key=lambda r: -r["val_acc"])
        print("\n==== grid results (sorted by VAL accuracy) ====")
        for r in results:
            print(f"val {r['val_acc']:.4f}  test {r['test_acc']:.4f}  f1 {r['test_f1']:.4f}  bin {r['bin_acc']:.4f}  {r['cfg']}")
        json.dump(results, open(MODELS / "attention_grid.json", "w"), indent=2)
    else:
        main(epochs=args.epochs, lr=args.lr, hidden=tuple(int(h) for h in args.hidden.split(",")),
             dropout=args.dropout, jitter=args.jitter, smoothing=args.smoothing, wd=args.wd)
