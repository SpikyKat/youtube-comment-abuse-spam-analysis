"""Step 1: audit the four datasets (sizes, gaps, duplicates, text length, language mix, labels)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUT, load_scam_comments, load_sentiment, load_toxic, load_videos, save_json
from ytmod.rules import language_hint

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
res = {}

# ---------------------------------------------------------------- scam comments
sc, vids = load_scam_comments(), load_videos()
lens = sc.text.str.len()
per_video = sc.groupby("video_id").size()
per_author = sc.groupby("author_id").size()
res["scam"] = {
    "rows": len(sc),
    "empty_after_cleaning": int((sc.text.str.len() == 0).sum()),
    "exact_duplicate_texts": int(sc.text.str.lower().duplicated().sum()),
    "length_mean": lens.mean(), "length_median": lens.median(), "length_p95": lens.quantile(0.95),
    "date_min": sc.date.min(), "date_max": sc.date.max(),
    "videos_with_comments": int(per_video.size),
    "videos_in_list": int(vids.video_id.nunique()),
    "comments_whose_video_not_in_list": int((~sc.video_id.isin(vids.video_id)).sum()),
    "comments_per_video_median": per_video.median(), "comments_per_video_max": per_video.max(),
    "authors": int(per_author.size),
    "authors_with_2plus_comments": int((per_author >= 2).sum()),
    "comments_from_repeat_authors": int(per_author[per_author >= 2].sum()),
    "max_comments_one_author": int(per_author.max()),
    "contains_html_tags": int(sc.raw_content.str.contains("<", regex=False).sum()),
    "language_hint": sc.text.map(language_hint).value_counts().to_dict(),
}
# some videos were returned by more than one search query: count each video once, under its first query
vids_once = vids.drop_duplicates("video_id")
q = sc.merge(vids_once[["video_id", "query"]], on="video_id", how="left")["query"].fillna("(not in list)")
res["scam"]["video_rows_in_list"] = len(vids)
res["scam"]["video_rows_repeating_an_earlier_video"] = int(vids.video_id.duplicated().sum())
res["scam"]["comments_by_query"] = q.value_counts().to_dict()
res["scam"]["videos_by_query"] = vids_once["query"].value_counts().to_dict()
res["scam"]["video_published_min"] = vids.published_at.min()
res["scam"]["video_published_max"] = vids.published_at.max()

# ---------------------------------------------------------------- toxicity
tx = load_toxic()
label_cols = [c for c in tx.columns if c.startswith("Is")]
res["toxic"] = {
    "rows": len(tx),
    "videos": int(tx.VideoId.nunique()),
    "comments_per_video_median": tx.groupby("VideoId").size().median(),
    "comments_per_video_max": int(tx.groupby("VideoId").size().max()),
    "exact_duplicate_texts": int(tx.text.str.lower().duplicated().sum()),
    "length_mean": tx.text.str.len().mean(), "length_median": tx.text.str.len().median(),
    "label_positives": {c: int(tx[c].sum()) for c in label_cols},
    "toxic_but_not_abusive": int((tx.IsToxic & ~tx.IsAbusive).sum()),
    "abusive_but_not_toxic": int((~tx.IsToxic & tx.IsAbusive).sum()),
    "any_subtype_without_toxic": int((tx[[c for c in label_cols if c != "IsToxic"]].any(axis=1) & ~tx.IsToxic).sum()),
    "language_hint": tx.text.map(language_hint).value_counts().to_dict(),
}
# does one video dominate the toxic comments?
by_video = tx.groupby("VideoId").IsToxic.agg(["size", "sum"]).sort_values("size", ascending=False)
res["toxic"]["top_video_share_of_comments"] = float(by_video["size"].iloc[0] / len(tx))
res["toxic"]["toxic_rate_by_video_top5"] = {k: [int(v["size"]), int(v["sum"])] for k, v in by_video.head(5).iterrows()}

# ---------------------------------------------------------------- sentiment
se = load_sentiment()
norm = se.text.str.lower().str.strip()
dup = se[norm.duplicated(keep=False) & (norm != "")]
conflict = dup.assign(n=norm[dup.index]).groupby("n").sentiment.nunique()
res["sentiment"] = {
    "rows": len(se),
    "empty_comments": int((se.text.str.len() == 0).sum()),
    "class_counts": se.sentiment.value_counts().to_dict(),
    "exact_duplicate_texts": int(norm[norm != ""].duplicated().sum()),
    "duplicate_groups_with_conflicting_labels": int((conflict > 1).sum()),
    "length_mean": se.text.str.len().mean(), "length_median": se.text.str.len().median(),
    "length_p95": se.text.str.len().quantile(0.95),
    "language_hint": se.text.sample(4000, random_state=0).map(language_hint).value_counts(normalize=True).to_dict(),
}
save_json("audit", res)

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].hist(sc.text.str.len().clip(upper=400), bins=40, color="#2f6f9f")
ax[0].set_title("Scam-topic comments: length")
ax[1].hist(tx.text.str.len().clip(upper=600), bins=40, color="#2f6f9f")
ax[1].set_title("Toxicity set: length")
ax[2].hist(se.text.str.len().clip(upper=400), bins=40, color="#2f6f9f")
ax[2].set_title("Sentiment set: length")
for a in ax:
    a.set_xlabel("characters (capped)")
plt.tight_layout()
plt.savefig(OUT / "figures" / "audit_lengths.png")
plt.close()

fig, ax = plt.subplots(figsize=(7, 3.8))
pos = pd.Series(res["toxic"]["label_positives"]).sort_values()
ax.barh(pos.index.str.replace("Is", ""), pos.values, color="#c0392b")
for i, v in enumerate(pos.values):
    ax.text(v + 5, i, str(v), va="center", fontsize=8)
ax.set_title("Toxicity set: positives per label (of 1,000 comments)")
plt.tight_layout()
plt.savefig(OUT / "figures" / "audit_toxic_labels.png")
plt.close()

fig, ax = plt.subplots(figsize=(8, 3.4))
monthly = sc.set_index("date").resample("MS").size()
ax.bar(monthly.index.strftime("%Y-%m"), monthly.values, color="#2f6f9f")
ax.set_title("Scam-topic comments by month posted")
ax.set_ylabel("comments")
plt.xticks(rotation=90, fontsize=6)
plt.tight_layout()
plt.savefig(OUT / "figures" / "audit_scam_by_month.png")
plt.close()

import json  # noqa: E402

print(json.dumps(json.load(open(OUT / "results" / "audit.json", encoding="utf-8")), indent=1)[:6000])
