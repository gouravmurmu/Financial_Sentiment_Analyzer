from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_renders_every_tab_on_demo_data():
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 5
    assert at.metric[0].label == "Last close"


def test_app_with_single_ticker_and_lexicon():
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.sidebar.multiselect[0].set_value(["NVDA"])
    at.sidebar.selectbox[1].set_value("lexicon")
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_cli_score(capsys):
    import main

    main.main(["score", "Apple beats estimates", "--model", "lexicon"])
    assert "positive" in capsys.readouterr().out
