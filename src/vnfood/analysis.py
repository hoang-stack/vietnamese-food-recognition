"""Analyses of the Kaggle outputs: model comparison, leakage, label noise, calibration, errors.

Inputs (kaggle/output/main): items.json, audit.npz, logits_<model>.npz, metrics.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vnfood.classes import vietnamese_name
from vnfood.metrics import (
    expected_calibration_error,
    fit_temperature,
    per_class_f1,
    softmax,
    top_k_accuracy,
)

# A test image counts as a near-duplicate of the training set when its perceptual hash is
# within this Hamming distance of some training image, or its CLIP embedding has at least this
# cosine similarity with one.
HAMMING_DUP = 8
COSINE_DUP = 0.95


@dataclass
class Outputs:
    classes: list[str]
    items: dict[str, list[tuple[str, int]]]
    audit: dict[str, np.ndarray]
    logits: dict[str, dict[str, np.ndarray]]  # model -> split -> logits
    metrics: dict

    def labels(self, split: str) -> np.ndarray:
        return np.array([y for _, y in self.items[split]])


def load(kaggle_dir: str | Path) -> Outputs:
    d = Path(kaggle_dir)
    items = json.loads((d / "items.json").read_text())
    classes = sorted({rel.split("/")[1] for rel, _ in items["Train"]})
    audit = dict(np.load(d / "audit.npz"))
    logits = {}
    for f in sorted(d.glob("logits_*.npz")):
        z = np.load(f)
        logits[f.stem.removeprefix("logits_")] = {k: z[k] for k in z.files}
    return Outputs(classes, items, audit, logits, json.loads((d / "metrics.json").read_text()))


def bootstrap_accuracy(correct: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = correct[rng.integers(0, len(correct), (n_boot, len(correct)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def model_table(o: Outputs) -> list[dict]:
    """Top-1 / top-5 / macro-F1 / ECE (raw and temperature-scaled) on Test for every model."""
    y_val, y_te = o.labels("Validate"), o.labels("Test")
    rows = []
    for name, lg in o.logits.items():
        t = fit_temperature(lg["Validate"], y_val)
        p, p_t = softmax(lg["Test"]), softmax(lg["Test"], t)
        correct = p.argmax(1) == y_te
        lo, hi = bootstrap_accuracy(correct.astype(float))
        rows.append(
            {
                "model": name,
                "top1": float(correct.mean()),
                "top1_ci": [lo, hi],
                "top5": top_k_accuracy(p, y_te, 5),
                "macro_f1": float(per_class_f1(p, y_te, len(o.classes)).mean()),
                "ece": expected_calibration_error(p, y_te),
                "ece_scaled": expected_calibration_error(p_t, y_te),
                "temperature": t,
            }
        )
    return sorted(rows, key=lambda r: -r["top1"])


def near_duplicates(o: Outputs) -> np.ndarray:
    """Boolean mask over test images that have a near-duplicate in the training set."""
    a = o.audit
    return (a["test_min_hamming"] <= HAMMING_DUP) | (a["test_nn_cos"] >= COSINE_DUP)


def leakage(o: Outputs, model: str) -> dict:
    """How much do near-duplicates inflate test accuracy?"""
    dup = near_duplicates(o)
    y = o.labels("Test")
    correct = softmax(o.logits[model]["Test"]).argmax(1) == y
    nn_label = o.audit["train_labels"][o.audit["test_nn_idx"]]
    return {
        "test_images": int(len(y)),
        "near_duplicates": int(dup.sum()),
        "share": float(dup.mean()),
        "exact_phash": int((o.audit["test_min_hamming"] == 0).sum()),
        "dup_same_label": float((nn_label[dup] == y[dup]).mean()) if dup.any() else float("nan"),
        "accuracy_all": float(correct.mean()),
        "accuracy_duplicates": float(correct[dup].mean()) if dup.any() else float("nan"),
        "accuracy_clean": float(correct[~dup].mean()),
    }


def label_noise(o: Outputs, min_disagree: int = 7) -> list[dict]:
    """Per class: share of training images whose 10 nearest neighbours mostly carry another
    label (>= ``min_disagree`` of 10). These are candidates for review, not proven errors."""
    knn = o.audit["train_knn_labels"]
    y = o.audit["train_labels"]
    disagree = (knn != y[:, None]).sum(axis=1) >= min_disagree
    rows = []
    for c, name in enumerate(o.classes):
        m = y == c
        rows.append(
            {
                "class": name,
                "name": vietnamese_name(name),
                "images": int(m.sum()),
                "suspect_share": float(disagree[m].mean()),
            }
        )
    return sorted(rows, key=lambda r: -r["suspect_share"])


def confusions(o: Outputs, model: str, top: int = 10) -> list[dict]:
    y = o.labels("Test")
    pred = o.logits[model]["Test"].argmax(1)
    pairs: dict[tuple[int, int], int] = {}
    for a, b in zip(y[pred != y], pred[pred != y], strict=True):
        pairs[(a, b)] = pairs.get((a, b), 0) + 1
    out = [
        {
            "true": vietnamese_name(o.classes[a]),
            "predicted": vietnamese_name(o.classes[b]),
            "count": n,
        }
        for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1])
    ]
    return out[:top]


def per_class_accuracy(o: Outputs, model: str) -> list[dict]:
    y = o.labels("Test")
    pred = o.logits[model]["Test"].argmax(1)
    return sorted(
        (
            {
                "class": vietnamese_name(c),
                "accuracy": float((pred[y == i] == i).mean()),
                "images": int((y == i).sum()),
            }
            for i, c in enumerate(o.classes)
            if (y == i).any()
        ),
        key=lambda r: r["accuracy"],
    )
