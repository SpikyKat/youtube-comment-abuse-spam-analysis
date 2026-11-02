# YouTube comments: abuse, spam and sentiment

Case study on four public YouTube comment datasets: toxicity and sentiment models, a transparent rule-based scam/spam
detector, and a simulation of how model-based triage compares with the hold-for-review settings YouTube gives creators.

Read `docs/case_study.md` first. Short version: toxicity is learnable but weaker on unseen videos (AUC 0.74 vs 0.80),
a model beats a blocked-word list modestly at equal workload, and promotion is only about 1% of comments under
scam-topic videos. Scam-rule accuracy is **not measured yet**; it needs the hand labels described below.

## Layout

```
src/ytmod/     cleaning, rules, models, evaluation helpers, policy simulation
scripts/       01_audit ... 05_policy_and_transfer, evaluate_labels
app/           Panel triage dashboard
tests/         pytest
docs/          case study and data card
```

## Run

Place the four CSVs in `data/raw/` (see `docs/data_card.md`), then:

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.\run_all.ps1
.venv\Scripts\panel serve app/dashboard.py
```

## Labelling the scam rules

Fill `label_promo_or_scam_attempt` (1 = advertises, solicits or runs a scam; 0 = anything else, including victims) in
`data/labels/scam_label_sheet.csv`, then run `python scripts/evaluate_labels.py`.

## Status

Raw comments, author hashes and outputs are git-ignored. Do not publish before checking dataset licences and YouTube terms.
