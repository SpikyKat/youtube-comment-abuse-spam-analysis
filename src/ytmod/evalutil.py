"""Small evaluation helpers: intervals and out-of-fold predictions."""
import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def stratified_rule_quality(flagged_pos: int, flagged_n: int, other_pos: int, other_n: int, other_pop: int) -> dict:
    """Quality of a rule from a labelled sample made of ALL flagged comments plus a random sample of the others.

    Precision is exact when every flagged comment was labelled. Promotion hiding among the unflagged is estimated
    from the random sample and scaled to the whole unflagged group, which gives recall with a (wide) interval."""
    precision = flagged_pos / flagged_n if flagged_n else float("nan")
    lo, hi = wilson(other_pos, other_n)
    p_other = other_pos / other_n if other_n else float("nan")
    est_missed = p_other * other_pop
    out = {"precision": precision, "flagged_labelled": flagged_n, "flagged_positive": flagged_pos,
           "unflagged_sample": other_n, "unflagged_sample_positive": other_pos,
           "unflagged_positive_rate": p_other, "unflagged_positive_rate_ci": (lo, hi),
           "estimated_missed": est_missed, "estimated_missed_ci": (lo * other_pop, hi * other_pop)}
    denom = flagged_pos + est_missed
    out["recall"] = flagged_pos / denom if denom else float("nan")
    out["recall_ci"] = (flagged_pos / (flagged_pos + hi * other_pop) if flagged_pos + hi * other_pop else float("nan"),
                        flagged_pos / (flagged_pos + lo * other_pop) if flagged_pos + lo * other_pop else float("nan"))
    return out


def bootstrap_ci(y, score, metric, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for a metric such as roc_auc_score."""
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            vals.append(metric(y[idx], score[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def oof_proba(make_model, texts, y, groups=None, n_splits: int = 5, seed: int = 0):
    """Out-of-fold probabilities. Uses grouped folds when groups are given, so comments from one video
    (or copies of the same text) never sit in both train and test."""
    texts = np.asarray(texts, dtype=object)
    y = np.asarray(y)
    out = np.zeros(len(y))
    if groups is None:
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(texts, y)
    else:
        splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(texts, y, groups)
    for tr, te in splitter:
        model = make_model().fit(texts[tr], y[tr])
        out[te] = model.predict_proba(texts[te])[:, 1]
    return out


def summarise(y, p) -> dict:
    """ROC AUC and average precision with bootstrap intervals."""
    auc, ap = roc_auc_score(y, p), average_precision_score(y, p)
    return {
        "n": int(len(y)), "positives": int(np.sum(y)), "base_rate": float(np.mean(y)),
        "auc": float(auc), "auc_ci": bootstrap_ci(y, p, roc_auc_score),
        "ap": float(ap), "ap_ci": bootstrap_ci(y, p, average_precision_score),
    }
