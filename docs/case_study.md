# YouTube comments: abuse, spam and sentiment

A personal analysis project on four public datasets of YouTube comments. The question: what can simple, explainable
models and rules do about abusive and scammy comments, and how does that compare with the tools YouTube already gives
creators?

Not affiliated with YouTube or Google. Everything here is my own analysis of public data. Numbers come from
`outputs/results/*.json`, produced by the scripts in `scripts/`.

## 1. Short version

1. **Toxicity is learnable, but less than random-fold scores suggest.** On 1,000 labelled comments, a word+char
   TF-IDF logistic regression reaches AUC 0.80 with ordinary folds and 0.74 when whole videos are held out
   (average precision 0.79 vs 0.71). The held-out-video number is the honest one for a new video.
2. **A model beats a creator's blocked-word list at the same review workload, but not by a lot.** Holding 43% of
   comments, a blocklist built from labelled examples catches 54% of toxic comments on unseen videos. Ranking by model
   score at the same 43% catches about 62%.
3. **Promotion is rare under scam-topic videos.** My transparent rules flag 27 of 2,754 comments (1.0%). Most comments
   are people discussing or warning about scams. Keyword filters on scam words would hold 12% of comments and almost
   none of the promotional ones.
4. **I cannot yet say how accurate the scam rules are.** There are no labels. A 300-row label sheet is ready; until it
   is filled in, only "share flagged" is reported.
5. **The scam comments cannot test YouTube's current filters.** They are what survived them.

## 2. Data

| Dataset | Rows | Labels | Used for |
|---|---|---|---|
| Scam-topic comments | 2,754 | none | rules, patterns, transfer check |
| Video list | 55 rows, 50 unique videos | search query | context |
| YouToxic English | 1,000 | 12 binary labels | toxicity models |
| Comment sentiment | 18,408 | positive / neutral / negative | sentiment model |

Things the audit found that shape everything else:

- The 1,000 toxic comments come from only **13 videos**; one video has 27% of them. Toxic rates range from about 20% to 74% by video.
- Subtype labels are nested: no comment is abusive without also being toxic. Homophobic and radicalism have no positives and sexist has one, so only six labels are modelled.
- The sentiment set has 526 repeated texts, four of them with conflicting labels.
- 5 videos hold half of the flagged scam-topic comments; 32 of 49 videos have none.
- The scam comments span 2020 to 2026, 2,628 authors; 104 authors posted more than once.

See `data_card.md` for sources and handling.

## 3. Toxicity

Two test schemes, same model: ordinary 5-fold, and 5-fold that keeps each video's comments together.

| Label | Positives | AUC random | AUC unseen videos | AP random | AP unseen videos |
|---|---|---|---|---|---|
| Toxic | 462 | 0.80 | 0.74 | 0.79 | 0.71 |
| Abusive | 353 | 0.83 | 0.76 | 0.76 | 0.67 |
| Provocative | 161 | 0.76 | 0.71 | 0.44 | 0.31 |
| Hate speech | 138 | 0.80 | 0.76 | 0.37 | 0.31 |
| Racist | 125 | 0.83 | 0.79 | 0.36 | 0.32 |
| Obscene | 100 | 0.86 | 0.81 | 0.48 | 0.46 |

Rare labels have AUCs that look fine but average precision only a little above the base rate (10 to 16%), so they are
not usable on their own. Among comments already toxic, the model separates abusive from non-abusive better (AUC 0.82).

At the default 0.5 threshold on unseen videos the toxic model holds 44% of comments with precision 0.66 and recall 0.63.
Reaching 80% recall means holding 64% of comments at precision 0.58. That is a lot of reviewing; this is a small,
weak model.

Per video the AUC ranges from 0.70 to 0.79 across the five videos with at least 40 comments. By comment length it ranges
from 0.69 to 0.76; the model is not clearly worse on short comments.

## 4. Sentiment

Folds grouped by normalised text so duplicates do not leak.

| Model | Accuracy | Macro F1 |
|---|---|---|
| Majority class | 0.62 | 0.26 |
| Word TF-IDF | 0.76 | 0.68 |
| Word+char TF-IDF | 0.77 | 0.70 |

Negative is the hardest class (F1 0.57); neutral is confused with both neighbours. Random folds are about 0.01 higher
than grouped folds, so duplicates matter little here.

The same model on other comment sections: it calls 62% of scam-topic comments neutral and 23% negative, against 62%
positive in its own training data. That shift is the main reason not to read sentiment predictions on a new domain as
measurements. On the toxicity set, 72% of toxic comments are predicted negative, but only 56% of predicted-negative
comments are toxic: negative sentiment is not abuse.

## 5. Scam and spam rules

A hand-weighted scoring of observable signals: external links, shorteners, phone numbers, emails, named contact apps,
requests to get in touch, promotional wording, money amounts, capitals and emoji. Comments that talk about scams
without any link, number or email are discounted. Weights are my judgement and unvalidated.

- 27 of 2,754 comments flagged (0.98%, 95% interval 0.67% to 1.42%).
- Per query flagged: unpaid internship 17 of 1,187, edtech 5 of 618, placement fraud 3 of 323, fake job 2 of 626.
- Repeat authors were flagged 1.7% of the time against 0.9% for single posters.
- Reading the ten highest-scoring flagged comments myself, about six were real promotion and four were false positives
  (victims, warnings, a Hinglish "me"). This is one person's read of ten comments, not a measurement.

**Evaluation plan.** All 27 flagged comments plus a random 273 of the rest are in `data/labels/scam_label_sheet.csv`.
Once labelled, `scripts/evaluate_labels.py` weights the sample to give precision exactly and recall with an interval.
Because only about 1% of the unflagged will be promotion, the recall interval will be wide.

## 6. Compared with what YouTube offers creators today

Source: YouTube Help, "Manage comments and settings" (support.google.com/youtube/answer/9483359), checked 2026-11-02.

| | YouTube today | This project |
|---|---|---|
| Hold level | None / Basic / Strict / Hold all; Basic and Strict use AI | model score with an adjustable threshold |
| Words | Creator blocked-word list; matches held up to 60 days | learned word and character features |
| Links | Option to hold every comment with a link | rule weighs link type, shortener, contact asks |
| Trusted people | Approved-users list skips the filters | repeat-author signal only, unused in rules |
| Review | "Held" tab: approve, remove or report | triage queue ordered by score, with reasons |

What the simulation says about the trade-off, on the toxicity set:

- A blocklist and a model both trade workload for recall. At matched workload the model catches 2 to 8 points more toxic comments.
- On the scam-topic comments, holding every link would hold 1.7% of comments and reach 12 of my 27 flagged ones;
  15 flagged ones contain no link. It also holds 34 link comments the rules did not flag. Without labels I cannot say which side is right.
- A scam-keyword list would hold 12% of comments, nearly all discussion.

What I cannot say: how YouTube's own Basic and Strict levels perform. Their internals are not public, and this data
has no record of what they held. I do not claim my approach beats them. YouTube's Transparency Report covers comment
removals; I did not extract figures from it and do not quote any.

## 7. Limits

- 1,000 toxic labels from 13 videos, English. Results will not carry to other topics or languages; the scam comments
  include Hinglish and some Indic script that the toxicity model never saw.
- Survivor bias in the scam comments (section 1, point 5).
- Labels were made by others under unknown guidelines; "toxic" is a judgement.
- Rule weights are unvalidated, scam accuracy is unmeasured, sentiment on new domains is unvalidated.
- Moving the toxicity model onto the scam comments is a domain shift with no ground truth: 21% score 0.5 or higher, but
  nothing says those are abusive.
- Intervals on the model metrics are bootstrap over comments, which ignores the video clustering and is therefore too narrow.

## 8. Ethics, privacy and licence

- Raw comments and author handles are not published; `data/raw`, `data/interim`, `data/labels` and `outputs` are git-ignored.
- Authors are hashed with a fixed salt. That is pseudonymisation, not anonymity.
- The triage output masks links, emails and phone numbers.
- Offensive text is not quoted in this write-up.
- Before making the repo public: check each dataset's licence and YouTube's terms, and my employment contract's outside-work clause.
- A model like this should rank items for a human reviewer, not remove comments on its own.

## 9. Reproduce

```
python -m venv .venv ; .venv\Scripts\pip install -r requirements.txt
.\run_all.ps1
```
