"""Step 3: toxicity models on the 1,000 labelled comments.

Two ways of testing: ordinary random folds, and folds that keep each video together. The second asks the harder and
more honest question: does a model trained on some videos work on a video it has never seen?
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

from common import OUT, ROOT, load_toxic, save_json
from ytmod.evalutil import oof_proba, summarise
from ytmod.models import binary_model

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
LABELS = ["IsToxic", "IsAbusive", "IsProvocative", "IsHatespeech", "IsRacist", "IsObscene"]  # at least 100 positives
tx = load_toxic()
texts, groups = tx.text.values, tx.VideoId.values
res = {"labels_modelled": LABELS, "n": len(tx), "videos": int(tx.VideoId.nunique())}

oof = {}  # (kind, scheme, label) -> probabilities
table = []
for kind in ("word", "word_char"):
    for scheme in ("random", "by_video"):
        for lab in LABELS:
            y = tx[lab].astype(int).values
            g = groups if scheme == "by_video" else None
            p = oof_proba(lambda: binary_model(kind), texts, y, groups=g, seed=0)
            oof[(kind, scheme, lab)] = p
            s = summarise(y, p)
            if scheme == "random":  # same test on two more shuffles, to see how much the number moves
                aucs = [roc_auc_score(y, oof_proba(lambda: binary_model(kind), texts, y, groups=None, seed=sd)) for sd in (1, 2)]
                s["auc_other_seeds"] = aucs
            table.append({"model": kind, "scheme": scheme, "label": lab, **s})
res["table"] = table
tab = pd.DataFrame(table)
print(tab[["model", "scheme", "label", "positives", "base_rate", "auc", "ap"]].round(3).to_string(index=False))

# pick the model by average precision on unseen videos
best = tab[tab.scheme == "by_video"].groupby("model").ap.mean().idxmax()
res["best_model"] = best
res["mean_ap_by_video"] = tab[tab.scheme == "by_video"].groupby("model").ap.mean().to_dict()
res["mean_ap_random"] = tab[tab.scheme == "random"].groupby("model").ap.mean().to_dict()

# ---------------------------------------------------------------- IsToxic: thresholds, videos, lengths
y = tx.IsToxic.astype(int).values
p_rand, p_vid = oof[(best, "random", "IsToxic")], oof[(best, "by_video", "IsToxic")]


def at_threshold(y, p, t):
    pred = (p >= t).astype(int)
    return {"threshold": t, "held_share": float(pred.mean()), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred)), "f1": float(f1_score(y, pred))}


def at_recall(y, p, target):
    pr, rc, th = precision_recall_curve(y, p)
    ok = np.where(rc[:-1] >= target)[0]
    i = ok[-1]  # highest threshold that still reaches the recall target
    return {**at_threshold(y, p, float(th[i])), "recall_target": target}


res["isToxic_default_threshold"] = {"random": at_threshold(y, p_rand, 0.5), "by_video": at_threshold(y, p_vid, 0.5)}
res["isToxic_at_recall"] = {s: [at_recall(y, p, r) for r in (0.7, 0.8, 0.9)] for s, p in (("random", p_rand), ("by_video", p_vid))}

rows = []
for v, d in tx.groupby("VideoId"):
    if len(d) < 40:
        continue
    yy, pp = d.IsToxic.astype(int).values, p_vid[d.index]
    both = 0 < yy.sum() < len(yy)
    rows.append({"video": v, "comments": len(d), "toxic_rate": yy.mean(),
                 "auc": roc_auc_score(yy, pp) if both else np.nan,
                 "ap": average_precision_score(yy, pp) if both else np.nan,
                 "recall_at_0.5": recall_score(yy, (pp >= 0.5).astype(int)) if yy.sum() else np.nan,
                 "precision_at_0.5": precision_score(yy, (pp >= 0.5).astype(int), zero_division=0)})
per_video = pd.DataFrame(rows).sort_values("comments", ascending=False)
per_video.to_csv(OUT / "tables" / "toxic_per_video.csv", index=False)
res["per_video"] = per_video.round(3).to_dict("records")

length = tx.text.str.len()
bucket = pd.qcut(length, 4, labels=["shortest 25%", "25-50%", "50-75%", "longest 25%"])
lb = []
for b in bucket.cat.categories:
    m = (bucket == b).values
    lb.append({"length_bucket": b, "comments": int(m.sum()), "toxic_rate": float(y[m].mean()),
               "auc": float(roc_auc_score(y[m], p_vid[m])), "recall_at_0.5": float(recall_score(y[m], (p_vid[m] >= 0.5).astype(int))),
               "precision_at_0.5": float(precision_score(y[m], (p_vid[m] >= 0.5).astype(int), zero_division=0))})
res["by_length"] = lb
res["length_quartile_edges_chars"] = [float(x) for x in length.quantile([0.25, 0.5, 0.75])]

# subtype labels are always inside "toxic": how well does the model separate subtypes among toxic comments only?
sub = {}
toxic_only = tx[tx.IsToxic].reset_index(drop=True)
for lab in ("IsAbusive", "IsProvocative", "IsHatespeech", "IsRacist", "IsObscene"):
    yy = toxic_only[lab].astype(int).values
    if 20 <= yy.sum() <= len(yy) - 20:
        pp = oof_proba(lambda: binary_model(best), toxic_only.text.values, yy, groups=toxic_only.VideoId.values, seed=0)
        sub[lab] = {"positives": int(yy.sum()), "of": len(yy), "auc": float(roc_auc_score(yy, pp)),
                    "ap": float(average_precision_score(yy, pp))}
res["subtype_within_toxic_by_video"] = sub
save_json("toxicity", res)
pd.DataFrame({"comment_id": tx.CommentId, "video_id": tx.VideoId, "text": tx.text, "IsToxic": y,
              "p_random": p_rand, "p_by_video": p_vid}).to_csv(ROOT / "data" / "interim" / "toxic_oof.csv", index=False)

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(figsize=(5.5, 4))
for name, p, c in (("random folds", p_rand, "#2f6f9f"), ("unseen videos", p_vid, "#c0392b")):
    pr, rc, _ = precision_recall_curve(y, p)
    ax.plot(rc, pr, label=f"{name} (AP {average_precision_score(y, p):.2f})", color=c)
ax.axhline(y.mean(), color="grey", ls="--", label=f"base rate {y.mean():.2f}")
ax.set_xlabel("recall")
ax.set_ylabel("precision")
ax.set_title("IsToxic: precision vs recall")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "figures" / "toxic_pr_curve.png")
plt.close()

fig, ax = plt.subplots(figsize=(8, 3.8))
w = 0.38
x = np.arange(len(LABELS))
a = tab[(tab.model == best) & (tab.scheme == "random")].set_index("label").loc[LABELS]
b = tab[(tab.model == best) & (tab.scheme == "by_video")].set_index("label").loc[LABELS]
ax.bar(x - w / 2, a.ap, w, label="random folds", color="#2f6f9f")
ax.bar(x + w / 2, b.ap, w, label="unseen videos", color="#c0392b")
ax.scatter(x, a.base_rate, color="black", marker="_", s=300, label="base rate", zorder=3)
ax.set_xticks(x)
ax.set_xticklabels([l.replace("Is", "") for l in LABELS])
ax.set_ylabel("average precision")
ax.set_title("Per label: ordinary test vs unseen videos")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "figures" / "toxic_ap_by_label.png")
plt.close()

fig, ax = plt.subplots(figsize=(7, 3.4))
pv = per_video.dropna(subset=["auc"])
ax.bar(range(len(pv)), pv.auc, color="#2f6f9f")
ax.set_xticks(range(len(pv)))
ax.set_xticklabels([f"{r.video[:5]}\nn={r.comments}, {r.toxic_rate:.0%} toxic" for r in pv.itertuples()], fontsize=6)
ax.axhline(0.5, color="grey", ls="--")
ax.set_ylabel("AUC on that video")
ax.set_title("Trained on the other videos, tested on this one")
plt.tight_layout()
plt.savefig(OUT / "figures" / "toxic_per_video_auc.png")
plt.close()

print("best model:", best)
print(json.dumps({k: res[k] for k in ("mean_ap_by_video", "mean_ap_random", "isToxic_default_threshold", "isToxic_at_recall", "subtype_within_toxic_by_video")}, indent=1))
print(per_video.round(2).to_string(index=False))
print(pd.DataFrame(lb).round(2).to_string(index=False))
