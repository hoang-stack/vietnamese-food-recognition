"""Upload a photo, get the five most likely dishes: ``vnfood serve``."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from importlib.resources import files

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from vnfood.predict import FoodClassifier

MAX_BYTES = 10 * 1024 * 1024


def create_app(classifier: FoodClassifier | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.clf = classifier or FoodClassifier.load(
            os.environ.get("VNFOOD_MODEL", "artifacts/efficientnet_b0.onnx"),
            os.environ.get("VNFOOD_META", "artifacts/classes.json"),
        )
        yield

    app = FastAPI(
        title="vnfood",
        lifespan=lifespan,
        description="Recognise 30 Vietnamese dishes (EfficientNet-B0, ONNX, CPU).",
    )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def page() -> str:
        return (files("vnfood") / "static" / "index.html").read_text(encoding="utf-8")

    @app.post("/predict")
    async def predict(request: Request, k: int = 5) -> dict:
        """Body: the raw image bytes (JPEG/PNG/WebP)."""
        body = await request.body()
        if not body:
            raise HTTPException(400, "empty body: send the image bytes")
        if len(body) > MAX_BYTES:
            raise HTTPException(413, "image larger than 10 MB")
        t0 = time.perf_counter()
        try:
            preds = request.app.state.clf.predict(body, k=min(max(k, 1), 30))
        except Exception as exc:  # PIL raises several types for corrupt or unknown files
            raise HTTPException(415, f"could not read the image: {type(exc).__name__}") from exc
        return {
            "predictions": [p.__dict__ for p in preds],
            "elapsed_ms": round(1000 * (time.perf_counter() - t0), 1),
        }

    return app


app = create_app()
