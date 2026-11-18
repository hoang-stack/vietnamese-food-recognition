"""Figures and reports/results.md from the Kaggle outputs (no images are ever shown)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from vnfood import analysis  # noqa: E402
from vnfood.metrics import fit_temperature, reliability, softmax  # noqa: E402

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
ACCENT, SECOND = "#2a78d6", "#eb6834"
LABELS = {
    "CLIP zero-shot": "CLIP ViT-B/16, zero-shot",
    "CLIP linear probe": "CLIP ViT-B/16, linear probe",
    "convnext_tiny.fb_in22k_ft_in1k": "ConvNeXt-Tiny, fine-tuned",
    "efficientnet_b0.ra_in1k": "EfficientNet-B0, fine-tuned",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
        }
    )


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_models(rows, out: Path) -> Path:
    rows = sorted(rows, key=lambda r: r["top1"])
    fig, ax = plt.subplots(figsize=(8.6, 0.6 * len(rows) + 1.3))
    for y, r in enumerate(rows):
        lo, hi = r["top1_ci"]
        ax.plot([100 * lo, 100 * hi], [y, y], color=ACCENT, alpha=0.45)
        ax.plot(100 * r["top1"], y, "o", ms=7, color=ACCENT, mec=SURFACE, mew=1.5, zorder=3)
        ax.annotate(
            f"{100 * r['top1']:.1f}%  (top-5 {100 * r['top5']:.1f}%)",
            (100 * hi, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_2,
        )
    ax.set_yticks(range(len(rows)), [LABELS.get(r["model"], r["model"]) for r in rows])
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(min(100 * r["top1_ci"][0] for r in rows) - 3, 104)
    ax.set_xlabel("test top-1 accuracy (%), 95% bootstrap CI")
    ax.set_title("30 Vietnamese dishes, 5,040 test photos")
    return _save(fig, out / "models.png")


def plot_leakage(o, best: str, out: Path) -> Path:
    leak = analysis.leakage(o, best)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    cos = o.audit["test_nn_cos"]
    ax.hist(
        cos,
        bins=np.linspace(0.4, 1.0, 61),
        color=ACCENT,
        alpha=0.85,
        edgecolor=SURFACE,
        linewidth=0.5,
    )
    ax.axvline(analysis.COSINE_DUP, color=SECOND, lw=1.2)
    ax.annotate(
        f"near-duplicate (≥ {analysis.COSINE_DUP})",
        (analysis.COSINE_DUP, ax.get_ylim()[1]),
        xytext=(-6, -12),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=INK_2,
    )
    ax.set_xlabel("cosine similarity of each test photo to its closest training photo (CLIP)")
    ax.set_ylabel("test photos")
    ax.set_title(f"{100 * leak['share']:.1f}% of test photos have a near-copy in training")
    ax = axes[1]
    dup = analysis.near_duplicates(o)
    correct = (o.logits[best]["Test"].argmax(1) == o.labels("Test")).astype(float)
    groups = [
        ("clean test photos", correct[~dup], ACCENT),
        ("all test photos", correct, MUTED),
        ("near-duplicates", correct[dup], SECOND),
    ]
    for i, (_name, c, color) in enumerate(groups):
        lo, hi = analysis.bootstrap_accuracy(c)
        ax.plot([100 * lo, 100 * hi], [i, i], color=color, alpha=0.5)
        ax.plot(100 * c.mean(), i, "o", ms=8, color=color, mec=SURFACE, mew=1.5)
        ax.annotate(
            f"{100 * c.mean():.1f}%  (n = {len(c):,})",
            (100 * hi, i),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_2,
        )
    ax.set_yticks(range(3), [g[0] for g in groups])
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlim(85, 100)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("top-1 accuracy (%), 95% CI")
    ax.set_title(f"Accuracy of {LABELS.get(best, best).split(',')[0]}")
    fig.tight_layout()
    return _save(fig, out / "leakage.png")


def plot_calibration(o, best: str, out: Path) -> Path:
    lg = o.logits[best]
    y_val, y_te = o.labels("Validate"), o.labels("Test")
    t = fit_temperature(lg["Validate"], y_val)
    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(5.4, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.plot([0, 1], [0, 1], color=BASELINE, lw=1.0)
    scaled = softmax(lg["Test"], t)
    for probs, color, label in (
        (softmax(lg["Test"]), SECOND, "raw (label smoothing)"),
        (scaled, ACCENT, f"temperature-scaled, T = {t:.2f}"),
    ):
        pts = reliability(probs, y_te, bins=10)
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            "-o",
            ms=5,
            color=color,
            mec=SURFACE,
            mew=1.0,
            label=label,
        )
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Reliability on test photos")
    axh.hist(
        scaled.max(axis=1),
        bins=np.linspace(0, 1, 21),
        color=ACCENT,
        alpha=0.8,
        edgecolor=SURFACE,
        linewidth=0.5,
    )
    axh.set_xlabel("confidence")
    axh.set_ylabel("photos")
    axh.set_xlim(0, 1.02)
    fig.tight_layout()
    return _save(fig, out / "calibration.png")


def plot_per_class(o, best: str, out: Path) -> Path:
    acc = {r["class"]: r["accuracy"] for r in analysis.per_class_accuracy(o, best)}
    noise = {r["name"]: r["suspect_share"] for r in analysis.label_noise(o)}
    order = sorted(acc, key=acc.get)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 8.0), sharey=True)
    for ax, values, color, label in (
        (axes[0], acc, ACCENT, "test accuracy"),
        (
            axes[1],
            noise,
            SECOND,
            "training photos whose 10 nearest neighbours mostly disagree with their label",
        ),
    ):
        ax.barh(range(len(order)), [100 * values[c] for c in order], height=0.6, color=color)
        ax.set_xlabel(f"{label} (%)", fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), order)
    axes[0].set_xlim(min(100 * min(acc.values()) - 10, 60), 100)
    axes[0].set_title("Per dish (best model)")
    axes[1].set_title("Label-noise candidates")
    fig.tight_layout()
    return _save(fig, out / "per_class.png")


def build_report(kaggle_dir="kaggle/output/main", report_dir="reports") -> Path:
    o = analysis.load(kaggle_dir)
    out = Path(report_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    _style()
    rows = analysis.model_table(o)
    best = rows[0]["model"]
    plot_models(rows, out / "figures")
    plot_leakage(o, best, out / "figures")
    plot_calibration(o, best, out / "figures")
    plot_per_class(o, best, out / "figures")
    leak = {m: analysis.leakage(o, m) for m in o.logits}
    summary = {
        "models": rows,
        "leakage": leak,
        "label_noise": analysis.label_noise(o),
        "confusions": analysis.confusions(o, best, 15),
        "per_class": analysis.per_class_accuracy(o, best),
        "kaggle_metrics": o.metrics,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    lines = [
        "# Results",
        "",
        "| Model | Top-1 (95% CI) | Top-5 | Macro-F1 | ECE raw → scaled | "
        "Top-1, clean test photos |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lo, hi = r["top1_ci"]
        lines.append(
            f"| {LABELS.get(r['model'], r['model'])} | {100 * r['top1']:.1f}% ({100 * lo:.1f}–"
            f"{100 * hi:.1f}) | {100 * r['top5']:.1f}% | {r['macro_f1']:.3f} | "
            f"{r['ece']:.3f} → {r['ece_scaled']:.3f} | "
            f"{100 * leak[r['model']]['accuracy_clean']:.1f}% |"
        )
    lb = leak[best]
    lines += [
        "",
        "## Leakage",
        "",
        f"{lb['near_duplicates']} of {lb['test_images']} test photos ({100 * lb['share']:.1f}%) "
        f"have a near-duplicate in the training set ({lb['exact_phash']} with an identical "
        f"perceptual hash); {100 * lb['dup_same_label']:.1f}% of those duplicates carry the "
        "same label.",
        "",
        "## Most frequent confusions (best model)",
        "",
        "| True | Predicted | Count |",
        "| --- | --- | ---: |",
    ]
    lines += [f"| {c['true']} | {c['predicted']} | {c['count']} |" for c in summary["confusions"]]
    lines += [
        "",
        "## Label-noise candidates per class (top 10)",
        "",
        "| Dish | Share |",
        "| --- | ---: |",
    ]
    lines += [
        f"| {r['name']} | {100 * r['suspect_share']:.1f}% |" for r in summary["label_noise"][:10]
    ]
    md = out / "results.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md
