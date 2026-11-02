import numpy as np

from ytmod.evalutil import oof_proba, stratified_rule_quality, wilson


def test_wilson_contains_the_rate_and_is_ordered():
    lo, hi = wilson(5, 50)
    assert lo < 0.1 < hi
    assert 0 <= lo <= hi <= 1


def test_stratified_quality_precision_and_recall():
    # 20 flagged, 15 truly promotion; 100 unflagged sampled out of 2,000, 2 of them promotion
    r = stratified_rule_quality(15, 20, 2, 100, 2000)
    assert abs(r["precision"] - 0.75) < 1e-9
    assert abs(r["estimated_missed"] - 40) < 1e-9
    assert abs(r["recall"] - 15 / 55) < 1e-9
    assert r["recall_ci"][0] < r["recall"] < r["recall_ci"][1]


def test_grouped_oof_keeps_groups_apart():
    # two groups of texts: grouped folds must never train on a group it tests on
    texts = np.array([f"word{i % 2} sample text {i}" for i in range(60)], dtype=object)
    y = np.array([i % 2 for i in range(60)])
    groups = np.repeat(np.arange(6), 10)
    seen = []

    class Spy:
        def fit(self, X, yy):
            seen.append(set(X))
            return self

        def predict_proba(self, X):
            return np.tile([0.5, 0.5], (len(X), 1))

    oof_proba(Spy, texts, y, groups=groups, n_splits=3)
    assert len(seen) == 3
