"""Moderation policy simulations: a creator-maintained blocked-word list vs a model score."""
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

TOKEN = r"(?u)\b[a-z][a-z']{2,}\b"


def build_blocklist(texts, y, k: int, min_pos: int = 3, min_precision: float = 0.7) -> set:
    """The best list a careful creator could build from labelled examples: single words that appear in at least
    min_pos toxic comments and are toxic at least min_precision of the time, ranked by how many toxic comments
    they catch."""
    vec = CountVectorizer(binary=True, token_pattern=TOKEN, lowercase=True)
    X = vec.fit_transform(texts)
    y = np.asarray(y).astype(bool)
    tp = np.asarray(X[y].sum(axis=0)).ravel()
    total = np.asarray(X.sum(axis=0)).ravel()
    precision = np.divide(tp, total, out=np.zeros_like(tp, dtype=float), where=total > 0)
    ok = np.where((tp >= min_pos) & (precision >= min_precision))[0]
    ranked = ok[np.argsort(-tp[ok])][:k]
    vocab = np.array(vec.get_feature_names_out())
    return set(vocab[ranked])


def blocklist_predict(texts, blocklist: set) -> np.ndarray:
    pat = re.compile(TOKEN)
    return np.array([any(w in blocklist for w in pat.findall(str(t).lower())) for t in texts])


def blocklist_curve(texts, y, ks, groups=None, n_splits: int = 5, seed: int = 0) -> list[dict]:
    """Out-of-fold results for blocklists of different sizes: share of comments held, recall and precision."""
    texts = np.asarray(texts, dtype=object)
    y = np.asarray(y)
    rows = []
    for k in ks:
        pred = np.zeros(len(y), dtype=bool)
        if groups is None:
            splits = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(texts, y)
        else:
            splits = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(texts, y, groups)
        for tr, te in splits:
            pred[te] = blocklist_predict(texts[te], build_blocklist(texts[tr], y[tr], k))
        tp = int((pred & (y == 1)).sum())
        rows.append({"blocklist_size": k, "held_share": float(pred.mean()),
                     "recall": tp / max(int(y.sum()), 1), "precision": tp / max(int(pred.sum()), 1)})
    return rows


def recall_at_held_share(y, score, held_share: float) -> dict:
    """If a reviewer looks at the top `held_share` of comments by score, how many bad ones are in that pile?"""
    y, score = np.asarray(y), np.asarray(score)
    n_hold = max(1, int(round(held_share * len(y))))
    top = np.argsort(-score)[:n_hold]
    caught = int(y[top].sum())
    return {"held_share": held_share, "held": n_hold, "caught": caught,
            "recall": caught / max(int(y.sum()), 1), "precision": caught / n_hold}
