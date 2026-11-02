"""Reviewer's triage queue for the scam-topic comments.

Run the scripts first (they write data/interim), then:  panel serve app/dashboard.py
Authors are hashed and links, emails and phone numbers are masked.
"""
from pathlib import Path

import pandas as pd
import panel as pn

pn.extension("tabulator")
INTERIM = Path(__file__).resolve().parents[1] / "data" / "interim"

scored = pd.read_csv(INTERIM / "scam_scored.csv")
tox = pd.read_csv(INTERIM / "scam_with_toxicity.csv")[["comment_row", "toxic_score"]]
df = scored.merge(tox, on="comment_row").fillna({"reasons": ""})
df["date"] = pd.to_datetime(df["date"]).dt.date
df["masked_text"] = df.masked_text.str.slice(0, 300)

tier = pn.widgets.CheckButtonGroup(name="Rule tier", options=["high", "medium", "low"], value=["high", "medium"])
query = pn.widgets.MultiSelect(name="Search query", options=sorted(df["query"].unique()), size=4)
min_tox = pn.widgets.FloatSlider(name="Minimum toxicity score", start=0, end=1, step=0.05, value=0)
text_filter = pn.widgets.TextInput(name="Contains word", placeholder="optional")


def filtered(tiers, queries, tmin, word):
    d = df[df.tier.isin(tiers) & (df.toxic_score >= tmin)]
    if queries:
        d = d[d["query"].isin(queries)]
    if word:
        d = d[d.masked_text.str.contains(word, case=False, regex=False)]
    return d.sort_values(["score", "toxic_score"], ascending=False)


def summary(tiers, queries, tmin, word):
    d = filtered(tiers, queries, tmin, word)
    return pn.pane.Markdown(f"**{len(d):,}** of {len(df):,} comments shown "
                            f"({len(d) / len(df):.1%}). Rule score is a hand-set heuristic, not a validated classifier.")


def queue(tiers, queries, tmin, word):
    d = filtered(tiers, queries, tmin, word)[["score", "tier", "toxic_score", "reasons", "masked_text", "query", "date", "author_id"]]
    return pn.widgets.Tabulator(d.round({"toxic_score": 2}), pagination="local", page_size=15, show_index=False,
                                disabled=True, sizing_mode="stretch_width", widths={"masked_text": 420, "reasons": 220})


args = dict(tiers=tier, queries=query, tmin=min_tox, word=text_filter)
app = pn.Column(
    "# Comment triage queue",
    "Scam-topic YouTube comments, ordered by rule score. For reviewing, not for deciding: nothing here is labelled yet.",
    pn.Row(pn.Column(tier, min_tox, text_filter), query),
    pn.bind(summary, **args),
    pn.bind(queue, **args),
    max_width=1100,
)
app.servable(title="Comment triage queue")
