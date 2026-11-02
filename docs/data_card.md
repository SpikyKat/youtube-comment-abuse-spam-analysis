# Data card

Raw files are not in this repository. Put them in `data/raw/` with these names.

| File | Rows | What it is | Handling |
|---|---|---|---|
| `scam-youtube-dataset-comments.csv` | 2,754 | Comments on videos found by four scam-related search queries; no labels | HTML unescaped, links and contacts masked in outputs, authors hashed |
| `scam-youtube-video-ids.csv` | 55 | Video IDs, titles, channels and the query that found them; 50 unique videos | de-duplicated before joins |
| `youtoxic_english_1000.csv` | 1,000 | English comments from 13 videos with 12 binary labels (toxic, abusive, hate speech and others) | six labels with 100+ positives modelled |
| `YoutubeCommentsDataSet.csv` | 18,408 | Comments labelled positive, neutral, negative | 44 empty rows dropped, grouped by text for testing |

Known issues: the scam comments are survivors of YouTube's own filtering; the toxicity set covers few videos; the
sentiment set has 526 repeated texts; label guidelines for the two labelled sets were not available to me.

Source, licence and terms-of-use for each file still need to be confirmed and recorded here before any public release.
