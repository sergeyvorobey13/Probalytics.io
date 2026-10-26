# Probalytics.io

**Prediction-market strategies, backtested on real Kalshi and Polymarket data from [Probalytics](https://probalytics.io).**

Each folder is one self-contained strategy test: the code, the result tables and the charts. Every test was run on [Probalytics](https://probalytics.io) data — trade fills, order book snapshots and the cross-platform Market Graph — queried directly with SQL.

> 📊 **Want to run your own?** Get access at [app.probalytics.io](https://app.probalytics.io) and read the docs at [docs.probalytics.io](https://docs.probalytics.io).

---

## Strategies

| Folder | Question | Answer |
|---|---|---|
| [`copy_trading/`](copy_trading/) | Does copying Polymarket's best wallets make money? | **No.** The top wallets are genuinely skilled (+$14 per $100 at their own prices), but a copier 30 s late with 0.5¢ slippage loses **$4.27 per $100** and lost money in all 6 monthly test windows. [Full report →](copy_trading/report_copy_trading/REPORT.md) |
| [`mboformer/`](mboformer/) | Does a tiny "event-history" transformer ([arXiv:2610.02917](https://arxiv.org/abs/2610.02917)) predict Polymarket order books well enough to trade? | **The science replicates in part, the trade doesn't.** Every taker strategy fails after fees and 0.25 s latency. The one lead is maker quoting gated by MBOFormer in BTC 15-min markets (+$4.09 per window at 5 shares, 38/40 windows profitable), but it does not hold in ETH and is fragile. Final results: [`mboformer/mboformer_v3/`](mboformer/mboformer_v3/). [Write-up →](mboformer/MBOFORMER_FINDINGS.md) |
| [`market_graph/`](market_graph/) | Do Kalshi and Polymarket agree on the same football match, and is there free money between them? | **They agree, no free money.** For Leyton Orient vs. Cambridge United, the win/draw/win mids sum to 1.035 (Kalshi) and 1.01 (Polymarket); buying every outcome at the cheapest ask across both venues costs 1.16, so no arbitrage. |
| [`longshot/`](longshot/) | Are cheap Kalshi contracts (< $0.20) overpriced, so that fading them pays? | Equity curves in [`longshot/`](longshot/). |

## Run it yourself

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env        # add your Probalytics credentials

cd copy_trading
../venv/bin/python test_copy_trading.py --selftest   # no database needed
../venv/bin/python test_copy_trading.py              # full backtest
```

Run each script from inside its own folder. Query results are cached in `cache_*/` on the first run, so reruns are fast. Model weights, caches and the largest raw data dumps are not committed; the scripts regenerate them.

| Strategy | Entry point |
|---|---|
| Copy trading | `copy_trading/test_copy_trading.py`, charts: `copy_trading/report_copy_trading/make_report_assets.py` |
| MBOFormer | `mboformer/mboformer_v3.py` (leak-free, final); `test_mboformer.py` and `retest_mboformer.py` are earlier versions |
| Market Graph | `market_graph/market_graph_demo.py`, API smoke test: `market_graph/test_market_graph.py` |
| Longshot fade | `longshot/test_longshot.py` |

---

## Built on Probalytics

<p align="center"><b>All of this research ran on <a href="https://probalytics.io">Probalytics.io</a> — the data platform for prediction markets.</b></p>

Probalytics gives you Kalshi and Polymarket data in one place, ready for serious research:

- **Every trade fill** — price, size, side and fee, with taker wallet IDs on Polymarket. That is what made the copy-trading test possible.
- **Order book snapshots** — depth per side over time, enough to train and test order-book models like MBOFormer.
- **Plain SQL on ClickHouse** — connect with any ClickHouse client and pull exactly the trades and books you need. No scraping.
- **Market Graph** — links the same event across Kalshi and Polymarket and maps how markets relate (equivalent, implies, outcome sets), queryable with Cypher or ready-made endpoints. Cross-venue comparisons like the one in `market_graph/` take a few API calls instead of manual market matching.

**👉 [Get started at probalytics.io](https://probalytics.io)** · [Docs](https://docs.probalytics.io) · [Dashboard & API keys](https://app.probalytics.io)

---

*These are research backtests, not trading advice. Results cover the periods stated in each report.*
