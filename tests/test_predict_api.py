import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vnfood.api import create_app
from vnfood.classes import DISHES
from vnfood.predict import FoodClassifier, preprocess

CLASSES = sorted(DISHES)
META = {
    "classes": CLASSES,
    "names": [DISHES[c][0] for c in CLASSES],
    "temperature": 1.5,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


class FakeSession:
    """Stands in for onnxruntime: 'Phở' wins for reddish images, 'Bánh mì' otherwise."""

    def run(self, _, feeds):
        x = feeds["image"]
        assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32
        logits = np.zeros((1, 30), np.float32)
        logits[0, CLASSES.index("Pho" if x[0, 0].mean() > x[0, 2].mean() else "Banh mi")] = 5
        return [logits]


def jpeg(color, size=(400, 300)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def test_preprocess_shape_and_normalisation():
    x = preprocess(Image.new("RGB", (640, 320), (124, 116, 104)), META["mean"], META["std"])
    assert x.shape == (1, 3, 224, 224)
    assert abs(float(x.mean())) < 0.05  # the ImageNet mean colour normalises to ~0


def test_classifier_returns_calibrated_top_k():
    clf = FoodClassifier(FakeSession(), META)
    preds = clf.predict(jpeg((200, 30, 30)), k=3)
    assert [p.name for p in preds][0] == "Phở"
    assert len(preds) == 3 and preds[0].probability > preds[1].probability
    total = sum(p.probability for p in clf.predict(jpeg((200, 30, 30)), k=30))
    assert total == pytest.approx(1.0, abs=1e-5)


def test_api():
    with TestClient(create_app(FoodClassifier(FakeSession(), META))) as c:
        r = c.post("/predict", content=jpeg((30, 30, 200)))
        assert r.status_code == 200 and r.json()["predictions"][0]["name"] == "Bánh mì"
        assert c.post("/predict", content=b"").status_code == 400
        assert c.post("/predict", content=b"not an image").status_code == 415
        assert "<html" in c.get("/").text.lower()
