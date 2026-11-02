"""Step 2: score every scam-topic comment with the transparent rules and study where promotion clusters.

There are no labels here, so everything below is a SHARE OF COMMENTS FLAGGED, not an accuracy figure.
"""
import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from common import OUT, ROOT, load_scam_comments, load_videos, save_json
from ytmod.clean import mask_contact
from ytmod.evalutil import wilson
from ytmod.rules import score_comment

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})

sc = load_scam_comments()
vids = load_videos().drop_duplicates("video_id")[["video_id", "query", "title", "channel"]]
sig = [score_comment(r) for r in sc.raw_content]
sc["score"] = [s.score for s in sig]
sc["tier"] = [s.tier for s in sig]
sc["reasons"] = ["; ".join(s.reasons) for s in sig]
sc["n_links"] = [s.n_links for s in sig]
sc["scam_talk"] = [s.scam_talk for s in sig]
sc["language"] = [s.language for s in sig]
sc["masked_text"] = sc.text.map(mask_contact)
sc = sc.merge(vids, on="video_id", how="left")
sc["flagged"] = sc.tier.isin(["medium", "high"])
sc["year"] = sc.date.dt.year

n = len(sc)
res = {"rows": n}
tiers = sc.tier.value_counts().reindex(["high", "medium", "low"]).fillna(0).astype(int)
res["tier_counts"] = tiers.to_dict()
res["tier_share"] = {t: [c / n, *wilson(int(c), n)] for t, c in tiers.items()}
res["flagged_share"] = [int(sc.flagged.sum()) / n, *wilson(int(sc.flagged.sum()), n)]

# which rules fire
all_reasons = Counter(r for rs in sc.reasons for r in rs.split("; ") if r)
res["rule_counts"] = dict(all_reasons.most_common())
prev = pd.DataFrame({"rule": list(all_reasons), "comments": list(all_reasons.values())})
prev["share_of_comments"] = prev.comments / n
prev.sort_values("comments", ascending=False).to_csv(OUT / "tables" / "scam_rule_prevalence.csv", index=False)

# by query
byq = sc.groupby("query").agg(comments=("tier", "size"), high=("tier", lambda s: (s == "high").sum()),
                              flagged=("flagged", "sum"), scam_talk=("scam_talk", "mean"))
byq["flagged_share"] = byq.flagged / byq.comments
byq.to_csv(OUT / "tables" / "scam_by_query.csv")
res["by_query"] = byq.round(4).reset_index().to_dict("records")

# by video: concentration
byv = sc.groupby(["video_id", "title", "channel", "query"]).agg(
    comments=("tier", "size"), flagged=("flagged", "sum"), high=("tier", lambda s: (s == "high").sum()),
    scam_talk=("scam_talk", "mean")).reset_index()
byv["flagged_share"] = byv.flagged / byv.comments
byv.sort_values("flagged", ascending=False).to_csv(OUT / "tables" / "scam_by_video.csv", index=False)
ordered = byv.sort_values("flagged", ascending=False).reset_index(drop=True)
top_k = max(1, int(round(0.1 * len(ordered))))
res["videos"] = len(ordered)
res["top10pct_videos_count"] = top_k
res["top10pct_videos_share_of_flagged"] = float(ordered.flagged.head(top_k).sum() / ordered.flagged.sum())
res["videos_with_zero_flagged"] = int((ordered.flagged == 0).sum())
res["flagged_share_range_videos_20plus"] = [
    float(byv[byv.comments >= 20].flagged_share.min()), float(byv[byv.comments >= 20].flagged_share.max())]

# authors and duplicates
sc["author_n"] = sc.groupby("author_id").author_id.transform("size")
rep = sc[sc.author_n >= 2]
res["repeat_author_comments"] = len(rep)
res["flagged_share_repeat_authors"] = [float(rep.flagged.mean()), *wilson(int(rep.flagged.sum()), len(rep))]
res["flagged_share_single_authors"] = [float(sc[sc.author_n == 1].flagged.mean()),
                                       *wilson(int(sc[sc.author_n == 1].flagged.sum()), int((sc.author_n == 1).sum()))]
norm = sc.text.str.lower().str.strip()
dup = sc[norm.duplicated(keep=False)]
res["duplicate_text_comments"] = len(dup)
res["flagged_share_duplicates"] = [float(dup.flagged.mean()), *wilson(int(dup.flagged.sum()), len(dup))] if len(dup) else None
res["flagged_share_non_duplicates"] = float(sc[~norm.duplicated(keep=False)].flagged.mean())
res["same_text_posted_under_n_videos_max"] = int(sc.assign(n=norm).groupby("n").video_id.nunique().max())

# discussion of scams vs promotion
res["scam_talk_share"] = float(sc.scam_talk.mean())
res["scam_talk_and_flagged"] = int((sc.scam_talk & sc.flagged).sum())
res["flagged_not_scam_talk"] = int((~sc.scam_talk & sc.flagged).sum())

# language
lang = sc.groupby("language").agg(comments=("tier", "size"), flagged=("flagged", "mean"), scam_talk=("scam_talk", "mean"))
res["by_language"] = lang.round(4).reset_index().to_dict("records")

# time: by year
yr = sc.groupby("year").agg(comments=("tier", "size"), flagged=("flagged", "mean"))
res["by_year"] = yr.round(4).reset_index().to_dict("records")

# links that appear
links = [l for l in sc.raw_content.map(lambda r: __import__("ytmod.clean", fromlist=["extract_links"]).extract_links(r)).sum()]
hosts = Counter()
for l in links:
    h = l.lower().split("//")[-1].split("/")[0].replace("www.", "")
    hosts[h] += 1
res["top_link_hosts"] = dict(hosts.most_common(12))
res["n_comments_with_any_link"] = int(sc.raw_content.map(lambda r: len(__import__("ytmod.clean", fromlist=["extract_links"]).extract_links(r)) > 0).sum())

# distinctive words in flagged comments (log-odds with a small prior)
cv = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-z]{4,}\b", min_df=5, binary=True)
X = cv.fit_transform(sc.text.str.lower())
vocab = np.array(cv.get_feature_names_out())
f = np.asarray(X[sc.flagged.values].sum(axis=0)).ravel() + 0.5
l = np.asarray(X[~sc.flagged.values].sum(axis=0)).ravel() + 0.5
logodds = np.log(f / f.sum()) - np.log(l / l.sum())
order = np.argsort(-logodds)
res["distinctive_terms_flagged"] = [str(vocab[i]) for i in order[:25] if f[i] >= 5]
res["distinctive_terms_unflagged"] = [str(vocab[i]) for i in order[::-1][:25] if l[i] >= 5]

save_json("scam_patterns", res)
cols = ["comment_row", "video_id", "query", "date", "author_id", "masked_text", "score", "tier", "reasons", "n_links",
        "scam_talk", "language"]
sc[cols].to_csv(ROOT / "data" / "interim" / "scam_scored.csv", index=False)

# ---------------------------------------------------------------- labelling sheet (stratified by tier)
# Only about 1% of comments are flagged, so a plain random sample would hold almost no promotion. Label every
# flagged comment plus a random sample of the rest; weights are restored in evaluate_labels.py.
SHEET_SIZE = 300
flagged_all = sc[sc.flagged].assign(sample_stratum="flagged")
rest = sc[~sc.flagged]
rest_sample = rest.sample(min(SHEET_SIZE - len(flagged_all), len(rest)), random_state=11).assign(sample_stratum="not_flagged")
sheet = pd.concat([flagged_all, rest_sample]).sample(frac=1, random_state=3)
sheet_out = pd.DataFrame({
    "comment_row": sheet.comment_row, "video_id": sheet.video_id, "text": sheet.text,
    "label_promo_or_scam_attempt": "", "notes": ""})
(ROOT / "data" / "labels").mkdir(exist_ok=True)
sheet_out.to_csv(ROOT / "data" / "labels" / "scam_label_sheet.csv", index=False, encoding="utf-8-sig")
# strata sizes and population sizes are needed later to weight the labelled sample correctly
pd.DataFrame({"sample_stratum": sheet.sample_stratum, "comment_row": sheet.comment_row}).to_csv(
    ROOT / "data" / "labels" / "scam_label_strata.csv", index=False)
pd.Series({"flagged": int(sc.flagged.sum()), "not_flagged": int((~sc.flagged).sum())}).rename("population").to_csv(
    ROOT / "data" / "labels" / "scam_population_by_stratum.csv")

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(figsize=(8, 3.6))
tab = sc.groupby(["query", "tier"]).size().unstack(fill_value=0).reindex(columns=["low", "medium", "high"], fill_value=0)
tab = tab.div(tab.sum(axis=1), axis=0)
tab.plot(kind="barh", stacked=True, ax=ax, color=["#9fb8cf", "#e6a23c", "#c0392b"])
ax.set_xlabel("share of comments")
ax.set_ylabel("")
ax.set_title("Rule-based risk tier by search query")
plt.tight_layout()
plt.savefig(OUT / "figures" / "scam_tier_by_query.png")
plt.close()

fig, ax = plt.subplots(figsize=(5.5, 3.8))
cum = ordered.flagged.cumsum() / ordered.flagged.sum()
ax.plot(np.arange(1, len(cum) + 1) / len(cum) * 100, cum.values * 100, color="#c0392b")
ax.plot([0, 100], [0, 100], "--", color="grey")
ax.set_xlabel("% of videos (most flagged first)")
ax.set_ylabel("% of flagged comments")
ax.set_title("Flagged comments cluster in few videos")
plt.tight_layout()
plt.savefig(OUT / "figures" / "scam_concentration.png")
plt.close()

fig, ax = plt.subplots(figsize=(7, 3.6))
pr = prev.sort_values("comments")
ax.barh(pr.rule, pr.comments, color="#2f6f9f")
ax.set_title("How often each rule fires (of %d comments)" % n)
plt.tight_layout()
plt.savefig(OUT / "figures" / "scam_rule_counts.png")
plt.close()

print(json.dumps(json.load(open(OUT / "results" / "scam_patterns.json", encoding="utf-8")), indent=1)[:7000])
print("\n--- masked examples by tier (for the analyst's eyes, not for the write-up)")
for t in ("high", "medium"):
    print("TIER", t)
    for _, r in sc[sc.tier == t].sort_values("score", ascending=False).head(6).iterrows():
        print(f"  [{r.score}] {r.reasons} :: {r.masked_text[:140]}")
