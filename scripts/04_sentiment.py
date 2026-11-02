"""Step 4: three-class sentiment model, then a look at how sentiment relates to flagged and toxic comments.

The sentiment file has 526 repeated texts, so folds are grouped by normalised text. Without that, a copy of a comment
sits in train while its twin is tested, and the score looks better than it is.
"""
import hashlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from common import OUT, ROOT, load_sentiment, save_json
from ytmod.models import sentiment_model

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
CLASSES = ["negative", "neutral", "positive"]

sd = load_sentiment()
sd = sd[sd.text.str.len() > 0].reset_index(drop=True)
norm = sd.text.str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
sd["group"] = norm.map(lambda t: hashlib.md5(t.encode()).hexdigest())
texts, y = sd.text.values, sd.sentiment.values
res = {"rows": len(sd), "distinct_texts": int(sd.group.nunique()), "class_counts": sd.sentiment.value_counts().to_dict()}


def cv_predict(make, grouped, seed=0):
    pred = np.empty(len(y), dtype=object)
    proba = np.zeros((len(y), 3))
    if grouped:
        split = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(texts, y, sd.group)
    else:
        split = StratifiedKFold(5, shuffle=True, random_state=seed).split(texts, y)
    for tr, te in split:
        m = make().fit(texts[tr], y[tr])
        pred[te] = m.predict(texts[te])
        proba[te] = m.predict_proba(texts[te])
    return pred, proba


rows = []
for name, make in (("majority class", lambda: DummyClassifier(strategy="prior")),
                   ("word tf-idf", lambda: sentiment_model("word")),
                   ("word+char tf-idf", lambda: sentiment_model("word_char"))):
    for grouped in (False, True):
        pred, _ = cv_predict(make, grouped)
        rows.append({"model": name, "folds": "grouped by text" if grouped else "random",
                     "accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
                     **{f"f1_{c}": f1_score(y, pred, labels=[c], average="macro") for c in CLASSES}})
tab = pd.DataFrame(rows)
tab.to_csv(OUT / "tables" / "sentiment_cv.csv", index=False)
res["cv"] = tab.round(4).to_dict("records")
print(tab.round(3).to_string(index=False))

best = tab[(tab.model != "majority class") & (tab.folds == "grouped by text")].sort_values("macro_f1").iloc[-1].model
make_best = (lambda: sentiment_model("word")) if best == "word tf-idf" else (lambda: sentiment_model("word_char"))
res["best_model"] = best
pred, proba = cv_predict(make_best, True)
cm = confusion_matrix(y, pred, labels=CLASSES)
res["confusion_grouped"] = {"labels": CLASSES, "matrix": cm.tolist()}

# does the label noise explain part of the gap? rows whose text appears with conflicting labels
conf = sd.groupby("group").sentiment.nunique()
res["texts_with_conflicting_labels"] = int((conf > 1).sum())

# length behaviour
length = sd.text.str.len()
bucket = pd.qcut(length, 4, labels=["shortest 25%", "25-50%", "50-75%", "longest 25%"])
res["accuracy_by_length"] = [{"bucket": b, "n": int((bucket == b).sum()),
                              "accuracy": float(accuracy_score(y[(bucket == b).values], pred[(bucket == b).values]))}
                             for b in bucket.cat.categories]

# ---------------------------------------------------------------- apply to the other two sets (no ground truth)
final = make_best().fit(texts, y)


def apply(df, name):
    p = final.predict_proba(df.text.fillna("").values)
    out = pd.DataFrame(p, columns=final.classes_)
    out["label"] = final.classes_[p.argmax(axis=1)]
    return out


scam = pd.read_csv(ROOT / "data" / "interim" / "scam_scored.csv")
scam_s = apply(scam.rename(columns={"masked_text": "text"}), "scam")
scam["sentiment_pred"] = scam_s.label.values
scam["flagged"] = scam.score.ge(2).values  # medium and high tiers
res["scam_sentiment_share"] = scam.sentiment_pred.value_counts(normalize=True).round(4).to_dict()
res["scam_sentiment_flagged_vs_not"] = {
    "flagged": scam[scam.flagged].sentiment_pred.value_counts(normalize=True).round(3).to_dict(),
    "not_flagged": scam[~scam.flagged].sentiment_pred.value_counts(normalize=True).round(3).to_dict()}
res["training_label_share"] = sd.sentiment.value_counts(normalize=True).round(4).to_dict()

tx = pd.read_csv(ROOT / "data" / "interim" / "toxic_oof.csv")
tx_s = apply(tx, "toxic")
tx["sentiment_pred"] = tx_s.label.values
res["toxic_set_sentiment_by_label"] = {
    "toxic": tx[tx.IsToxic == 1].sentiment_pred.value_counts(normalize=True).round(3).to_dict(),
    "not_toxic": tx[tx.IsToxic == 0].sentiment_pred.value_counts(normalize=True).round(3).to_dict()}
res["toxic_vs_negative_overlap"] = {
    "negative_given_toxic": float((tx[tx.IsToxic == 1].sentiment_pred == "negative").mean()),
    "toxic_given_negative": float((tx[tx.sentiment_pred == "negative"].IsToxic == 1).mean()),
    "toxic_given_not_negative": float((tx[tx.sentiment_pred != "negative"].IsToxic == 1).mean())}
save_json("sentiment", res)

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(figsize=(4.6, 4))
im = ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(3))
ax.set_xticklabels(CLASSES)
ax.set_yticks(range(3))
ax.set_yticklabels(CLASSES)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="black" if cm[i, j] / cm[i].sum() < 0.6 else "white")
ax.set_xlabel("predicted")
ax.set_ylabel("labelled")
ax.set_title("Sentiment (unseen texts), row share")
plt.tight_layout()
plt.savefig(OUT / "figures" / "sentiment_confusion.png")
plt.close()

fig, ax = plt.subplots(figsize=(6, 3.4))
df = pd.DataFrame({"training data": pd.Series(res["training_label_share"]),
                   "scam-topic comments (predicted)": pd.Series(res["scam_sentiment_share"]),
                   "toxicity set (predicted)": tx.sentiment_pred.value_counts(normalize=True)}).reindex(CLASSES)
df.T.plot(kind="barh", stacked=True, ax=ax, color=["#c0392b", "#9fb8cf", "#2f8f5b"])
ax.set_xlabel("share of comments")
ax.set_title("Same model, three comment sections")
plt.tight_layout()
plt.savefig(OUT / "figures" / "sentiment_shift.png")
plt.close()

print("best:", best)
for k in ("texts_with_conflicting_labels", "accuracy_by_length", "scam_sentiment_share", "scam_sentiment_flagged_vs_not",
          "training_label_share", "toxic_set_sentiment_by_label", "toxic_vs_negative_overlap"):
    print(k, res[k])
print(cm)
