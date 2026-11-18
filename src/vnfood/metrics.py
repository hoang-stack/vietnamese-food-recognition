"""Classification metrics, calibration and a near-duplicate finder for perceptual hashes."""

from __future__ import annotations

import numpy as np


def top_k_accuracy(probs: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == labels[:, None]).any(axis=1).mean())


def per_class_f1(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    pred = probs.argmax(axis=1)
    f1 = np.zeros(n_classes)
    for c in range(n_classes):
        tp = np.sum((pred == c) & (labels == c))
        fp = np.sum((pred == c) & (labels != c))
        fn = np.sum((pred != c) & (labels == c))
        f1[c] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return f1


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    """ECE (Guo et al., 2017): weighted gap between confidence and accuracy per confidence bin."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability(probs: np.ndarray, labels: np.ndarray, bins: int = 10):
    """(mean confidence, accuracy, count) per confidence bin, for reliability diagrams."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return out


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): the T > 0 minimising validation NLL.

    One parameter, so it cannot change which class is predicted, only how confident the model
    is. A golden-section search over log T is enough.
    """

    def nll(log_t: float) -> float:
        p = softmax(logits, np.exp(log_t))
        return float(-np.log(p[np.arange(len(labels)), labels] + 1e-12).mean())

    a, b = np.log(0.05), np.log(20.0)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(80):
        if nll(c) < nll(d):
            b, d = d, c
            c = b - g * (b - a)
        else:
            a, c = c, d
            d = a + g * (b - a)
    return float(np.exp((a + b) / 2))


def hamming_pairs(
    hashes_a: np.ndarray, hashes_b: np.ndarray, max_dist: int = 4, chunk: int = 2048
) -> list[tuple[int, int, int]]:
    """All (i, j, distance) with Hamming(hashes_a[i], hashes_b[j]) <= max_dist (64-bit hashes)."""
    a = hashes_a.astype(np.uint64)
    b = hashes_b.astype(np.uint64)
    out = []
    for s in range(0, len(a), chunk):
        x = a[s : s + chunk, None] ^ b[None, :]
        dist = _popcount(x)
        i, j = np.nonzero(dist <= max_dist)
        out.extend(zip((i + s).tolist(), j.tolist(), dist[i, j].tolist(), strict=True))
    return out


_BYTE_COUNTS = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount(x: np.ndarray) -> np.ndarray:
    """Bits set in each uint64 (np.bitwise_count needs NumPy >= 2.0)."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return _BYTE_COUNTS[x.view(np.uint8).reshape(*x.shape, 8)].sum(axis=-1)
