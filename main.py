"""Command line entry point.

    python main.py score "Apple beats estimates and raises guidance"
    python main.py analyse --tickers AAPL TSLA --demo
    python main.py benchmark --corpus my_labelled_headlines.csv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from finsent import data, demo, index, models, predict, signals
from finsent.config import BENCHMARK, DEFAULT_TICKERS


def cmd_score(args):
    eng = models.SentimentEngine.train(args.model)
    out = eng.score(args.headlines)
    out.insert(0, "headline", args.headlines)
    print(out[["headline", "label", "score", "confidence"]].to_string(index=False))


def cmd_benchmark(args):
    corpus = data.parse_labelled_corpus(pd.read_csv(args.corpus)) if args.corpus else demo.labelled_corpus()
    names = [m for m in models.MODEL_NAMES if m != "finbert" or models.finbert_available()]
    res = models.benchmark(corpus, names, folds=args.folds)
    print(f"{res['n']:,} labelled headlines, {res['folds']}-fold cross-validation\n")
    print(res["table"].drop(columns="key").round(3).to_string(index=False))


def cmd_analyse(args):
    start = (pd.Timestamp.today() - pd.DateOffset(years=args.years)).normalize()
    if args.demo:
        prices, news = demo.market_data(args.tickers, start=start)
    else:
        prices = data.fetch_prices(args.tickers + [BENCHMARK], start)
        news, errors = data.fetch_news(args.tickers)
        for e in errors:
            print("warning:", e, file=sys.stderr)
    eng = models.SentimentEngine.train(args.model)
    scored = pd.concat([news.reset_index(drop=True), eng.score(news["headline"])], axis=1)
    merged = index.merge_returns(index.daily_index(scored, prices.index), prices, BENCHMARK)
    for tk, g in merged.groupby("ticker"):
        g = g.sort_values("date")
        print(f"\n=== {tk}: {int(g['n'].sum()):,} headlines, latest sentiment index {g['index'].iloc[-1]:+.2f}")
        ll = signals.lead_lag(g.set_index("date")["daily"].fillna(0), g.set_index("date")["ret"], 2)
        print(ll[["lag", "corr", "p_value"]].round(3).to_string(index=False))
        if g["next_ret"].notna().sum() >= 320:
            print(predict.run(g, kinds=("logit",)).scores.round(3).to_string(index=False))
    if args.out:
        merged.to_csv(args.out, index=False)
        print(f"\nDaily sentiment and returns saved to {args.out}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Financial news sentiment toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)
    model_names = list(models.MODEL_NAMES)

    s = sub.add_parser("score", help="score one or more headlines")
    s.add_argument("headlines", nargs="+")
    s.add_argument("--model", default="logreg", choices=model_names)
    s.set_defaults(fn=cmd_score)

    b = sub.add_parser("benchmark", help="cross-validate every classifier")
    b.add_argument("--corpus", help="CSV with text and label columns (default: built-in demo corpus)")
    b.add_argument("--folds", type=int, default=5)
    b.set_defaults(fn=cmd_benchmark)

    a = sub.add_parser("analyse", help="build sentiment indices and test them against prices")
    a.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS[:3])
    a.add_argument("--years", type=int, default=3)
    a.add_argument("--model", default="logreg", choices=model_names)
    a.add_argument("--demo", action="store_true", help="use simulated data instead of live feeds")
    a.add_argument("--out", help="write the daily table to this CSV")
    a.set_defaults(fn=cmd_analyse)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
