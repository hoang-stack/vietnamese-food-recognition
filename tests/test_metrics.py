import numpy as np
import pytest

from vnfood import metrics


def test_top_k_and_f1():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.5, 0.4, 0.1]])
    labels = np.array([0, 1, 1])
    assert metrics.top_k_accuracy(probs, labels, 1) == pytest.approx(1 / 3)
    assert metrics.top_k_accuracy(probs, labels, 2) == pytest.approx(1.0)
    np.testing.assert_allclose(metrics.per_class_f1(probs, labels, 3), [2 / 3, 0, 0])


def test_ece_is_zero_for_a_calibrated_model_and_large_for_an_overconfident_one():
    rng = np.random.default_rng(0)
    n = 20_000
    conf = rng.uniform(0.5, 1.0, n)
    correct = rng.random(n) < conf  # accuracy equals confidence: calibrated
    probs = np.stack([conf, 1 - conf], axis=1)
    labels = np.where(correct, 0, 1)
    assert metrics.expected_calibration_error(probs, labels) < 0.02
    over = np.stack([np.full(n, 0.99), np.full(n, 0.01)], axis=1)
    assert metrics.expected_calibration_error(over, labels) > 0.2


def test_temperature_scaling_recovers_the_true_temperature():
    rng = np.random.default_rng(0)
    n, k, true_t = 20_000, 5, 3.0
    logits = rng.normal(0, 4, (n, k))
    p = metrics.softmax(logits, true_t)
    labels = np.array([rng.choice(k, p=row) for row in p])
    assert metrics.fit_temperature(logits, labels) == pytest.approx(true_t, rel=0.1)


def test_hamming_pairs():
    a = np.array([0b1010, 0xFFFF, 0], dtype=np.uint64)
    b = np.array([0b1011, 0xFF00, 0], dtype=np.uint64)
    pairs = metrics.hamming_pairs(a, b, max_dist=1)
    assert sorted(pairs) == [(0, 0, 1), (2, 2, 0)]


def test_popcount_fallback_matches():
    x = np.array([0, 1, 0xFFFF_FFFF_FFFF_FFFF, 0b1011], dtype=np.uint64)
    assert metrics._BYTE_COUNTS[x.view(np.uint8).reshape(4, 8)].sum(axis=-1).tolist() == [
        0,
        1,
        64,
        3,
    ]


def test_every_dish_has_a_name_and_prompt():
    from vnfood.classes import DISHES, prompt

    assert len(DISHES) == 30
    assert all("Vietnamese dish" in prompt(k) for k in DISHES)
