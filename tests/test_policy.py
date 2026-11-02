import numpy as np

from ytmod.policy import blocklist_predict, build_blocklist, recall_at_held_share


def test_blocklist_keeps_only_precise_frequent_words():
    texts = ["you idiot", "idiot move", "total idiot", "nice video", "nice song", "nice idiot joke ha"] * 2
    y = [1, 1, 1, 0, 0, 0] * 2
    bl = build_blocklist(texts, y, k=5)
    assert "idiot" in bl  # toxic in 6 of 8 comments, above the 0.7 cut-off
    assert "nice" not in bl
    assert blocklist_predict(["what an idiot"], bl)[0]


def test_recall_at_held_share_perfect_score():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    r = recall_at_held_share(y, y.astype(float), 0.2)
    assert r["recall"] == 1.0 and r["precision"] == 1.0
