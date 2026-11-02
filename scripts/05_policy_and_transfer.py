"""Step 5: moderation policy simulation and the transfer check on the scam-topic comments.

Part A (labels exist): a creator's blocked-word list against a model score, on the toxicity set.
Part B (no labels): what the same tools would do to the scam-topic comments. Only workloads and overlaps are reported,
never accuracy, because nobody has labelled those comments yet.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUT, ROOT, load_scam_comments, load_toxic, save_json
from ytmod.clean import extract_links
from ytmod.models import binary_model
from ytmod.policy import blocklist_curve, recall_at_held_share

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
tx = load_toxic()
y = tx.IsToxic.astype(int).values
oof = pd.read_csv(ROOT / "data" / "interim" / "toxic_oof.csv")
res = {}

# ---------------------------------------------------------------- A. blocklist vs model
ks = [10, 25, 50, 100, 200]
block = {"random": blocklist_curve(tx.text.values, y, ks), "by_video": blocklist_curve(tx.text.values, y, ks, groups=tx.VideoId.values)}
res["blocklist"] = block
shares = [0.1, 0.2, 0.3, 0.4, 0.5]
model = {s: [recall_at_held_share(y, oof[c].values, h) for h in shares] for s, c in (("random", "p_random"), ("by_video", "p_by_video"))}
res["model_recall_at_held_share"] = model
res["model_vs_blocklist_by_video"] = []
for row in block["by_video"]:
    r = recall_at_held_share(y, oof.p_by_video.values, row["held_share"])
    res["model_vs_blocklist_by_video"].append({"blocklist_size": row["blocklist_size"], "held_share": row["held_share"],
                                               "blocklist_recall": row["recall"], "blocklist_precision": row["precision"],
                                               "model_recall_same_held_share": r["recall"], "model_precision_same_held_share": r["precision"]})
tab = pd.DataFrame(res["model_vs_blocklist_by_video"])
tab.to_csv(OUT / "tables" / "policy_blocklist_vs_model.csv", index=False)
print(tab.round(3).to_string(index=False))
print(pd.DataFrame(model["by_video"]).round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4.2))
hs = np.linspace(0.02, 1, 50)
for s, c, ls in (("by_video", "p_by_video", "-"), ("random", "p_random", "--")):
    ax.plot(hs * 100, [recall_at_held_share(y, oof[c].values, h)["recall"] * 100 for h in hs], ls, color="#c0392b",
            label=f"model score ({'unseen videos' if s == 'by_video' else 'random folds'})")
ax.plot([0, 100], [0, 100], ":", color="grey", label="no model (random hold)")
for s, ls, mk in (("by_video", "-", "o"), ("random", "--", "s")):
    ax.plot([r["held_share"] * 100 for r in block[s]], [r["recall"] * 100 for r in block[s]], ls, marker=mk, color="#2f6f9f",
            label=f"creator blocklist ({'unseen videos' if s == 'by_video' else 'random folds'})")
ax.set_xlabel("% of comments held for review")
ax.set_ylabel("% of toxic comments caught")
ax.set_title("Review workload vs toxic comments caught")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(OUT / "figures" / "policy_recall_vs_workload.png")
plt.close()

# ---------------------------------------------------------------- B. scam-topic comments, no labels
sc = load_scam_comments()
scored = pd.read_csv(ROOT / "data" / "interim" / "scam_scored.csv")
sc = sc.merge(scored[["comment_row", "query", "masked_text", "score", "tier", "n_links", "scam_talk"]], on="comment_row")
sc["flagged"] = sc.tier.isin(["medium", "high"])
sc["any_link"] = sc.raw_content.map(lambda r: len(extract_links(r)) > 0)

final = binary_model("word_char").fit(tx.text.values, y)
sc["toxic_score"] = final.predict_proba(sc.masked_text.fillna("").values)[:, 1]
n = len(sc)

pol = {
    "hold every comment with a link": sc.any_link,
    "hold comments with scam keywords": sc.scam_talk.astype(bool),
    "hold comments flagged by promotion rules": sc.flagged,
    "toxicity model score 0.5 or more": sc.toxic_score >= 0.5,
}
rows = [{"policy": k, "held": int(v.sum()), "held_share": float(v.mean())} for k, v in pol.items()]
res["scam_policy_workload"] = rows
names = list(pol)
overlap = pd.DataFrame([[int((pol[a] & pol[b]).sum()) for b in names] for a in names], index=names, columns=names)
overlap.to_csv(OUT / "tables" / "scam_policy_overlap.csv")
res["scam_policy_overlap"] = overlap.to_dict()
res["scam_toxic_score"] = {"mean": float(sc.toxic_score.mean()), "share_0.5_plus": float((sc.toxic_score >= 0.5).mean()),
                           "training_set_toxic_rate": float(y.mean())}
res["scam_toxic_score_by_flag"] = {"flagged_mean": float(sc[sc.flagged].toxic_score.mean()),
                                   "not_flagged_mean": float(sc[~sc.flagged].toxic_score.mean())}
res["flagged_comments_missed_by_link_hold"] = int((sc.flagged & ~sc.any_link).sum())
res["flagged_comments_with_link"] = int((sc.flagged & sc.any_link).sum())
res["link_comments_not_flagged"] = int((sc.any_link & ~sc.flagged).sum())
save_json("policy_and_transfer", res)
print(pd.DataFrame(rows).round(4).to_string(index=False))
print(overlap.to_string())
for k in ("scam_toxic_score", "scam_toxic_score_by_flag", "flagged_comments_missed_by_link_hold", "flagged_comments_with_link", "link_comments_not_flagged"):
    print(k, res[k])

fig, ax = plt.subplots(figsize=(6.5, 3.2))
pw = pd.DataFrame(rows)
ax.barh(pw.policy, pw.held_share * 100, color="#2f6f9f")
for i, v in enumerate(pw.held_share * 100):
    ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8)
ax.set_xlabel("% of scam-topic comments held")
ax.set_title("What each policy would put in front of a reviewer")
plt.tight_layout()
plt.savefig(OUT / "figures" / "scam_policy_workload.png")
plt.close()

sc[["comment_row", "video_id", "query", "masked_text", "score", "tier", "toxic_score"]].to_csv(
    ROOT / "data" / "interim" / "scam_with_toxicity.csv", index=False)
