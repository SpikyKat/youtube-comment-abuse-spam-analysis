"""Evaluate the scam/spam rules once data/labels/scam_label_sheet.csv has been labelled.

Fill the column label_promo_or_scam_attempt with 1 (someone advertises, solicits or runs a scam) or 0 (anything else,
including victims describing scams and general discussion). Blank rows are ignored.
Run:  python scripts/evaluate_labels.py
"""
import sys

import pandas as pd

from common import OUT, ROOT, save_json
from ytmod.evalutil import stratified_rule_quality

sheet = pd.read_csv(ROOT / "data" / "labels" / "scam_label_sheet.csv", encoding="utf-8-sig")
strata = pd.read_csv(ROOT / "data" / "labels" / "scam_label_strata.csv")
pop = pd.read_csv(ROOT / "data" / "labels" / "scam_population_by_stratum.csv", index_col=0)["population"]

col = "label_promo_or_scam_attempt"
sheet[col] = pd.to_numeric(sheet[col].replace({"yes": 1, "no": 0, "y": 1, "n": 0}), errors="coerce")
labelled = sheet.dropna(subset=[col]).merge(strata, on="comment_row")
if labelled.empty:
    sys.exit("No labels yet. Fill label_promo_or_scam_attempt in data/labels/scam_label_sheet.csv first.")

fl = labelled[labelled.sample_stratum == "flagged"]
ot = labelled[labelled.sample_stratum == "not_flagged"]
total_flagged = int(pop["flagged"])
if len(fl) < total_flagged:
    print(f"Note: only {len(fl)} of {total_flagged} flagged comments are labelled; precision is an estimate.")
res = stratified_rule_quality(int(fl[col].sum()), len(fl), int(ot[col].sum()), len(ot), int(pop["not_flagged"]))
save_json("scam_rule_evaluation", res)
for k, v in res.items():
    print(f"{k}: {v}")
