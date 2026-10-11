import numpy as np
import pandas as pd

from finsent import data, lexicon, models
from finsent.text import clean_text


def test_clean_text_keeps_negations_and_direction():
    out = clean_text("Apple does NOT expect shares to go down! https://x.com $AAPL")
    assert "not" in out and "down" in out
    assert "http" not in out and "aapl" not in out


def test_lexicon_handles_phrases_and_negation():
    assert lexicon.score("Microsoft beats estimates and raises guidance") > 0.5
    assert lexicon.score("Tesla cuts guidance as deliveries plunge") < -0.5
    assert lexicon.score("Bank does not expect layoffs") > 0
    assert lexicon.score("Company to hold annual meeting") == 0


def test_engine_outputs_valid_probabilities(engine):
    out = engine.score(["Nvidia shares surge on strong data center demand", "Apple misses estimates",
                        "JPMorgan to present at conference"])
    probs = out[["p_negative", "p_neutral", "p_positive"]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1)
    assert list(out["label"]) == ["positive", "negative", "neutral"]
    assert out["score"].between(-1, 1).all()


def test_explain_returns_signed_terms(engine):
    e = engine.explain("Tesla shares plunge after weak demand")
    assert not e.empty and (e["weight"] < 0).any()


def test_benchmark_ranks_models_and_beats_chance(corpus):
    res = models.benchmark(corpus, ("lexicon", "logreg"), folds=3)
    t = res["table"]
    assert set(t["key"]) == {"lexicon", "logreg"}
    assert (t["accuracy"] > 0.5).all()
    assert t.iloc[0]["key"] == "logreg"  # learned model beats rules on in-domain data
    assert res["confusion"]["logreg"].to_numpy().sum() == len(corpus)


def test_engine_round_trip(tmp_path, engine):
    path = tmp_path / "m.joblib"
    engine.save(path)
    again = models.SentimentEngine.load(path)
    h = ["Amazon stock jumps after upbeat outlook"]
    assert np.allclose(again.predict_proba(h), engine.predict_proba(h))


def test_parse_uploaded_files():
    news = data.parse_uploaded_news(pd.DataFrame({"Date": ["2024-01-02 17:30", "bad"], "Title": ["A", "B"]}), "XYZ")
    assert len(news) == 1 and news["ticker"].iloc[0] == "XYZ"
    assert str(news["published"].dt.tz) == "America/New_York"
    lab = data.parse_labelled_corpus(pd.DataFrame({"sentence": ["x", "y", "z"], "sentiment": ["pos", "-1", "?"]}))
    assert list(lab["label"]) == ["positive", "negative"]
