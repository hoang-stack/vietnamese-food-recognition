import json

import numpy as np
import pytest

from vnfood import analysis
from vnfood.classes import DISHES

CLASSES = sorted(DISHES)


def fake_outputs(tmp_path, n=300, seed=0):
    rng = np.random.default_rng(seed)
    items = {}
    for split in ("Train", "Validate", "Test"):
        labels = rng.integers(0, 30, n)
        items[split] = [[f"{split}/{CLASSES[y]}/{i}.jpg", int(y)] for i, y in enumerate(labels)]
    (tmp_path / "items.json").write_text(json.dumps(items))
    y_tr = np.array([y for _, y in items["Train"]])
    y_te = np.array([y for _, y in items["Test"]])
    knn = np.repeat(y_tr[:, None], 10, axis=1)
    knn[:30] = (y_tr[:30, None] + 1) % 30  # first 30 training images look mislabelled
    dup = np.zeros(n, bool)
    dup[:60] = True
    nn_idx = rng.integers(0, n, n)
    np.savez(
        tmp_path / "audit.npz",
        test_nn_cos=np.where(dup, 0.99, 0.6),
        test_nn_idx=nn_idx,
        test_min_hamming=np.where(dup, 2, 30),
        train_knn_labels=knn,
        train_labels=y_tr,
        test_labels=y_te,
    )
    for name, noise in (("good", 2.0), ("bad", 0.2)):
        logits = {}
        for split in ("Validate", "Test"):
            y = np.array([lab for _, lab in items[split]])
            lg = rng.normal(0, 1, (n, 30))
            lg[np.arange(n), y] += 3.0 * noise
            logits[split] = lg
        np.savez(tmp_path / f"logits_{name}.npz", **logits)
    (tmp_path / "metrics.json").write_text("{}")
    return analysis.load(tmp_path)


def test_model_table_orders_models_and_reports_calibration(tmp_path):
    o = fake_outputs(tmp_path)
    rows = analysis.model_table(o)
    assert [r["model"] for r in rows] == ["good", "bad"]
    assert rows[0]["top1"] > 0.9
    lo, hi = rows[0]["top1_ci"]
    assert lo <= rows[0]["top1"] <= hi
    assert all(r["temperature"] > 0 for r in rows)


def test_leakage_and_label_noise(tmp_path):
    o = fake_outputs(tmp_path)
    leak = analysis.leakage(o, "good")
    assert leak["near_duplicates"] == 60 and leak["share"] == pytest.approx(0.2)
    noise = analysis.label_noise(o)
    assert sum(r["images"] for r in noise) == 300
    assert noise[0]["suspect_share"] > 0
    assert analysis.confusions(o, "bad")[0]["count"] >= 1
    assert len(analysis.per_class_accuracy(o, "good")) == 30
