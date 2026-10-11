"""Headline sentiment classifiers: lexicon, classical ML pipelines and FinBERT, behind one interface."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from . import lexicon
from .config import FINBERT_MODEL, LABELS
from .text import clean_text, normalise

MODEL_NAMES = {
    "lexicon": "Finance lexicon (rules)",
    "logreg": "TF-IDF + logistic regression",
    "svm": "Word + char TF-IDF + linear SVM",
    "nb": "TF-IDF + complement naive Bayes",
    "finbert": "FinBERT (transformer)",
}


def build_pipeline(name: str) -> Pipeline:
    """Scikit-learn pipelines that take raw headlines and output class probabilities."""
    words = TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    if name == "logreg":
        return Pipeline([("tfidf", words),
                         ("clf", LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"))])
    if name == "svm":
        chars = TfidfVectorizer(preprocessor=normalise, analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                sublinear_tf=True, max_features=40000)
        feats = FeatureUnion([("words", words), ("chars", chars)])
        svm = CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3)
        return Pipeline([("tfidf", feats), ("clf", svm)])
    if name == "nb":
        return Pipeline([("tfidf", words), ("clf", ComplementNB(alpha=0.3))])
    raise ValueError(f"Unknown model {name!r}")


def finbert_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("transformers", "torch"))


class FinBERT:
    """Hugging Face sequence classifier (ProsusAI/finbert by default), loaded lazily."""

    def __init__(self, model_name: str = FINBERT_MODEL, batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.batch_size = batch_size
        id2label = {i: lab.lower() for i, lab in self.model.config.id2label.items()}
        self.order = [next(i for i, lab in id2label.items() if lab.startswith(name[:3])) for name in LABELS]

    def predict_proba(self, texts) -> np.ndarray:
        out = []
        texts = [str(t) for t in texts]
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[i:i + self.batch_size], padding=True, truncation=True, max_length=64,
                                   return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                probs = self.torch.softmax(self.model(**batch).logits, dim=-1).cpu().numpy()
            out.append(probs[:, self.order])
        return np.vstack(out) if out else np.empty((0, 3))


@dataclass
class SentimentEngine:
    """Scores headlines with one backend: ``lexicon``, ``logreg``, ``svm``, ``nb`` or ``finbert``."""

    backend: str = "logreg"
    model: object = field(default=None, repr=False)

    @classmethod
    def train(cls, backend: str, corpus: pd.DataFrame | None = None) -> SentimentEngine:
        if backend == "lexicon":
            return cls("lexicon")
        if backend == "finbert":
            return cls("finbert", FinBERT())
        if corpus is None:
            from .demo import labelled_corpus
            corpus = labelled_corpus()
        pipe = build_pipeline(backend).fit(corpus["headline"], corpus["label"])
        return cls(backend, pipe)

    def predict_proba(self, texts) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, 3))
        if self.backend == "lexicon":
            return lexicon.predict(texts)[1]
        if self.backend == "finbert":
            return self.model.predict_proba(texts)
        probs = self.model.predict_proba(texts)
        cols = [list(self.model.classes_).index(lab) for lab in LABELS]
        return probs[:, cols]

    def score(self, texts) -> pd.DataFrame:
        """Label, class probabilities, polarity score (P(pos) - P(neg)) and confidence per headline."""
        p = self.predict_proba(texts)
        df = pd.DataFrame(p, columns=["p_negative", "p_neutral", "p_positive"])
        df["label"] = np.array(LABELS)[p.argmax(axis=1)] if len(p) else []
        df["score"] = df["p_positive"] - df["p_negative"]
        df["confidence"] = p.max(axis=1) if len(p) else []
        return df

    def explain(self, text: str, top: int = 8) -> pd.DataFrame:
        """Word contributions towards positive minus negative, for linear models and the lexicon."""
        if self.backend == "lexicon":
            words = {w: lexicon.LEXICON[w] for w in set(clean_text(text).split()) if w in lexicon.LEXICON}
            return pd.DataFrame({"term": list(words), "weight": list(words.values())}) \
                .sort_values("weight", key=abs, ascending=False).head(top)
        if self.backend != "logreg":
            return pd.DataFrame(columns=["term", "weight"])
        vec, clf = self.model.named_steps["tfidf"], self.model.named_steps["clf"]
        x = vec.transform([text])
        classes = list(clf.classes_)
        coef = clf.coef_[classes.index("positive")] - clf.coef_[classes.index("negative")]
        idx = x.nonzero()[1]
        terms = vec.get_feature_names_out()[idx]
        contrib = np.asarray(x[0, idx].todense()).ravel() * coef[idx]
        return pd.DataFrame({"term": terms, "weight": contrib}).sort_values("weight", key=abs, ascending=False) \
            .head(top)

    def save(self, path):
        joblib.dump({"backend": self.backend, "model": self.model}, path)

    @classmethod
    def load(cls, path) -> SentimentEngine:
        d = joblib.load(path)
        return cls(d["backend"], d["model"])


def _norm(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1)
    return p / p.sum(axis=1, keepdims=True)


def benchmark(corpus: pd.DataFrame, models=("lexicon", "nb", "logreg", "svm"), folds: int = 5,
              seed: int = 0) -> dict:
    """Cross-validated comparison of classifiers.

    Uses stratified K-fold, or GroupKFold when the corpus has a ``group`` column (e.g. one group per
    source document, so near-duplicates never straddle train and test). Returns a metrics table and
    pooled confusion matrices.
    """
    X, y = corpus["headline"].to_numpy(), corpus["label"].to_numpy()
    if "group" in corpus:
        splits = list(GroupKFold(n_splits=folds).split(X, y, corpus["group"]))
    else:
        splits = list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y))
    rows, cms = [], {}
    for name in models:
        pred = np.empty(len(y), dtype=object)
        proba = np.zeros((len(y), 3))
        for tr, te in splits:
            eng = SentimentEngine.train(name, corpus.iloc[tr])
            p = eng.predict_proba(X[te])
            proba[te] = p
            pred[te] = np.array(LABELS)[p.argmax(axis=1)]
        f1s = f1_score(y, pred, labels=list(LABELS), average=None)
        rows.append({"model": MODEL_NAMES[name], "key": name, "accuracy": accuracy_score(y, pred),
                     "macro_f1": f1_score(y, pred, average="macro"),
                     "log_loss": log_loss(y, _norm(proba), labels=list(LABELS)),
                     **{f"f1_{lab}": f for lab, f in zip(LABELS, f1s, strict=True)}})
        cms[name] = pd.DataFrame(confusion_matrix(y, pred, labels=list(LABELS)), index=LABELS, columns=LABELS)
    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return {"table": table, "confusion": cms, "n": len(y), "folds": folds}
