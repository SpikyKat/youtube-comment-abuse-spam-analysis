"""Text models: toxicity labels and sentiment. Classical, fast and explainable on purpose."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def _vectorizer(kind: str):
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, strip_accents="unicode")
    if kind == "word":
        return word
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
    return FeatureUnion([("word", word), ("char", char)])


def binary_model(kind: str = "word", C: float = 2.0):
    """One yes/no label (for example IsToxic)."""
    return Pipeline([
        ("tfidf", _vectorizer(kind)),
        ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced")),
    ])


def sentiment_model(kind: str = "word", C: float = 2.0):
    """Three classes: positive, neutral, negative."""
    return Pipeline([
        ("tfidf", _vectorizer(kind)),
        ("clf", LogisticRegression(C=C, max_iter=3000, class_weight="balanced")),
    ])
