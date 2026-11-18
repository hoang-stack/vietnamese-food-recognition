"""CPU inference with the exported ONNX model: no PyTorch needed at serving time."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from vnfood.metrics import softmax


def preprocess(img: Image.Image, mean, std, size: int = 224, resize: int = 256) -> np.ndarray:
    """Resize the short side to ``resize``, centre-crop ``size``, normalise, NCHW float32
    (the same evaluation transform the model was validated with)."""
    img = img.convert("RGB")
    w, h = img.size
    scale = resize / min(w, h)
    img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array(mean, np.float32)) / np.array(std, np.float32)
    return x.transpose(2, 0, 1)[None]


@dataclass
class Prediction:
    dish: str
    name: str
    probability: float


class FoodClassifier:
    def __init__(self, session, meta: dict):
        self.session, self.meta = session, meta

    @classmethod
    def load(cls, model_path: str | Path, meta_path: str | Path) -> FoodClassifier:
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        return cls(session, json.loads(Path(meta_path).read_text(encoding="utf-8")))

    def predict(self, image_bytes: bytes, k: int = 5) -> list[Prediction]:
        img = Image.open(io.BytesIO(image_bytes))
        x = preprocess(img, self.meta["mean"], self.meta["std"])
        logits = self.session.run(None, {"image": x})[0]
        probs = softmax(logits, self.meta.get("temperature", 1.0))[0]
        top = np.argsort(-probs)[:k]
        return [
            Prediction(self.meta["classes"][i], self.meta["names"][i], float(probs[i])) for i in top
        ]
