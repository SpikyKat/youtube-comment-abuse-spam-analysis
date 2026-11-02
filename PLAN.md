# Case study plan: abuse, scam and sentiment signals in YouTube comments

Working title: **"Keeping comments safe: what four public datasets say about abuse, scams and sentiment on YouTube, and where today's tools fall short."**
Written from the point of view of a trust and safety analyst. Plan first; nothing below has been built yet.

## 1. The four datasets (inspected, not yet cleaned)
| File | Rows | What it has | Labels? | Main limits |
|---|---|---|---|---|
| `scam-youtube-dataset-comments.csv` | 2,754 | platform, video_id, date, comment text, author handle | **None** | Unlabelled; text is HTML-escaped (`&#39;`, `<a href=...>`); author handles are personal data |
| `scam-youtube-video-ids.csv` | 55 | query, video_id, title, channel, published_at | n/a | Four search queries: unpaid internship scam India (25 videos), edtech scam India (10), fake job scam India (10), placement fraud India (10) |
| `youtoxic_english_1000.csv` | 1,000 | comment text + 12 yes/no labels | **Yes, multi-label** | Small. Toxic 462, abusive 353, provocative 161, hate speech 138, racist 125, obscene 100, threat 21, religious hate 12, nationalist 8, sexist 1, homophobic 0, radicalism 0. Sample rows look like US policing and race topics, which differ from Indian scam topics |
| `YoutubeCommentsDataSet.csv` | 18,408 | comment, sentiment | **Yes** (positive 11,432, neutral 4,638, negative 2,338) | 44 empty comments; no video id or date; sample rows look like tech and payments topics |

The files were in WinRAR temp folders, so I copied them to `data/raw/` in this project. **Raw data stays out of Git** until the licences are checked (Kaggle and YouTube terms may not allow republishing comment text). Author handles are hashed before any output leaves the machine.

## 2. Questions the case study answers
1. **What does scam and spam look like** in comments under Indian scam-topic videos? (links, phone or Telegram or WhatsApp handles, "DM me" and testimonial-style promotion, duplicated text, repeat authors, bursts in time, which queries and channels attract them)
2. **How well can a simple model find abusive and toxic comments** from the labelled set, label by label, and where does it fail? (precision, recall, false positives, threshold trade-offs)
3. **Does sentiment add anything** for moderation, or is it a separate signal? (compare sentiment with toxicity and with scam patterns on the data that allows it)
4. **How does today's YouTube approach compare** with what the data suggests could be added? (section 3)
5. **If a platform used our rules and model, what would it cost and gain?** (policy simulation: share of bad comments caught vs good comments held for review)

## 3. Comparison with current methods (sourced from YouTube Help, to be re-checked when writing)
What YouTube offers today:
- **Automated holds for review.** Creators choose None, Basic, Strict or Hold all; held comments wait in a Held tab for approve, remove or report. YouTube says the detection uses AI and can be wrong.
- **Blocked words and links.** Creators can list blocked words and phrases and hold comments containing links or hashtags; an approved-users list skips these filters.
- **Policy.** The spam, deceptive practices and scams policy covers comments and links, including fake job offers, fake support contacts and impersonation, with penalties from removal to channel termination.
- **Enforcement at scale.** The Transparency Report says almost all removed comments are first flagged by automated systems; check the live page for current figures.

Where the data suggests gaps to discuss (hypotheses to test, not conclusions):
- Generic filters work on words and links; scam comments here may use handles, spaced-out numbers and promotional stories that a word list misses.
- Risk is concentrated: some videos (scam-topic) and some repeat authors attract most promotion, so **video-level and author-level risk scores** could prioritise review.
- Held-for-review pushes work onto creators; a **ranked queue with reasons** could save time.
- English-trained models can fail on Hinglish and other Indian languages; measure it.

## 4. System to build
```
raw CSVs -> clean (unescape HTML, normalise, hash authors) -> features and rules
          -> models: toxicity (multi-label), sentiment, scam/spam score
          -> policy layer: allow / hold for review / remove-suggest, with thresholds and reasons
          -> evaluation: metrics, error analysis, policy simulation
          -> dashboard: moderation triage view -> case study write-up
```
- **Toxicity:** TF-IDF + logistic regression per label as the baseline, with class weights. Only the 6 labels with at least 100 positives are modelled; rarer ones are reported as not learnable from this data.
- **Sentiment:** a classifier trained on the 18k-comment set, evaluated with cross-validation, plus a check of how it transfers to the scam comments.
- **Scam and spam:** no ground truth exists. So I would (a) write transparent rules, (b) **hand-label a random sample of about 300 scam-set comments** (you do the labelling; I build the labelling sheet), and (c) measure the rules and any model against those labels with confidence intervals. Without those labels, no scam accuracy number is honest.
- **Combined score and policy simulation:** compare "YouTube-style" (hold links and blocked words) vs our rules vs rules plus models.
- **Code quality goals (supports the D. E. Shaw prep):** a small Python package (`src/`), type hints, `pytest` tests for cleaning and rules, a Panel dashboard.
- Classical models only (free, fast, explainable). Transformers are a stretch goal, not part of the plan.

## 5. Deliverables
1. Case study write-up (about 8 to 10 pages): problem, data, method, findings, comparison with current methods, recommendations, limits and ethics.
2. Repo: `src/`, tests, notebooks, scripts, README with a reproducible run (data not included).
3. Dashboard: moderation triage view with filters by video, author risk and reason.
4. A data card: sources, licences (to be checked), known biases, what the models cannot do.

## 6. Schedule (5 days, about 5 to 6 hours a day)
| Day | Work | Done when |
|---|---|---|
| 1 | Data audit and cleaning: nulls, duplicates, language mix, lengths, HTML cleanup, hashing authors; EDA on all four files; repo skeleton and tests for the cleaning functions | Cleaning is tested and a data-quality table exists |
| 2 | Scam and spam analysis: patterns, repeat authors, bursts, video and query breakdown; rules v1; build the labelling sheet and label about 300 comments | Rules run, labels exist |
| 3 | Toxicity models: baselines per label, threshold study, error analysis (what it misses and why) | Per-label metrics with intervals |
| 4 | Sentiment model; transfer check to scam comments; combined score; policy simulation against the hand labels | Simulation table: caught vs held |
| 5 | Dashboard; write-up; README and data card; licence and privacy check before anything is published | Case study and repo complete |

## 7. What can be done with this data (short answer)
A toxicity detector, a scam and spam pattern detector, a sentiment model, a combined moderation triage score, a policy simulation, a moderator dashboard, a written case study, and an analysis of how scam promotion clusters by video, query, channel, author and time. It is not enough for a production moderation system: 1,000 labelled toxic comments and zero labelled scam comments are too few, which is why the case study must say so.

## 8. Limits to state in the write-up
Small and topic-specific toxicity labels; no scam ground truth until hand-labelled; sentiment data from other topics; no video-level context for two of the files; English-centred models; labels are subjective; any model can wrongly flag legitimate comments, so every threshold is a trade-off between missed scams and wrongly held comments.

## 9. Decisions needed from you
1. **Existing work:** your resumes already list this project as built. Is there earlier work, or are we building it now? Until the build exists, resume wording should say "in progress".
2. **Hand labelling:** will you label about 300 comments (some may be offensive)? If not, I will not report scam accuracy.
3. **Publishing:** keep the repo private until licences and platform terms are checked, and publish code and a data card but not the raw comments.
4. **Dashboard tool:** Panel (matches the D. E. Shaw job) or Streamlit (faster).
