# Does a tiny "event-history" transformer predict Polymarket order books?

> **Superseded — read [`mboformer_v3/`](mboformer_v3/) for the final results.** The runs described below (`test_mboformer.py`, `retest_mboformer.py`) treated book decreases as executions using fills printed up to 6 s later, which leaked future information, and they used Polymarket's January 2026 fee instead of the current 0.07·p(1−p). `mboformer_v3.py` fixes both. Its verdict: every taker strategy fails at 0.25 s latency; the only lead is MBOFormer-gated maker quoting in BTC (+$4.09 per window at 5 shares, 38/40 windows profitable). That lead does not hold in ETH (−$0.41), so the pre-registered test fails overall.

**Source:** Chris Preuss' LinkedIn post. It describes the paper *"Event History Over Scale: Compact Transformers for Low-Latency Limit Order Book Forecasting"* by Schaurecker, Strand, O'Sullivan & Jakob (ETH Zurich, 2026), [arXiv:2610.02917](https://arxiv.org/abs/2610.02917).

**Test:** `test_mboformer.py`, run on Probalytics `orderbook_snapshots` and `fills` data.

**Charts:** `mboformer_1_macro_f1.png` to `mboformer_6_what_snapshots_miss.png`.

**Raw numbers:** `mboformer_results/*.csv`

## What the paper claims

The paper's model is MBOFormer, a 7,203-parameter transformer that reads the last 128 order-book events. An event is an order added, an order cancelled or a trade. MBOFormer predicts whether the smoothed mid price will go up, stay flat or go down a few steps ahead. The paper finds that it predicts as well as or better than models up to 400× bigger that only look at order-book *snapshots*. It also runs in under a millisecond on one laptop CPU thread. The authors did not test whether this makes money.

## What we did

- **Markets.** We used 59 Polymarket contracts in three groups:
  - BTC 15-minute "Up or Down" (20 contracts)
  - ETH 15-minute "Up or Down" (19 contracts; one had no book data)
  - Live sports and esports matches (20 contracts)

  Each contract contributes 15 minutes of order book, about 11.7M book changes in total. This produced 9.1M events and 1.17M decision points.
- **Events.** Probalytics stores a full book every time the book changes. Comparing each book with the previous one gives the events. A size decrease that coincides with a printed trade is labelled an execution.
- **Models compared:**
  - MBOFormer, at exactly 7,203 parameters
  - MBOFusion, which adds a slow "market context" branch
  - **L2-Former**: the same tiny transformer, but fed the last 16 order-book *snapshots* instead of events. This isolates the paper's claim: same model, different input.
  - A logistic regression on 12 classic order-book signals
  - An "always predict flat" baseline
- **Splits.** The first 12 contracts of each group train the models, the next 3 tune them and the last 5 test them. Test contracts are never seen during training.
- **Fixes and additions to the paper:**
  - The up/flat/down threshold comes from training data only. The paper computed it on the test set.
  - Every comparison gets a 95% confidence interval.
  - A trading test with the spread and Polymarket's taker fee.

**How long one "step" is.** A decision is made after every 10 book changes. That is about 0.03 s in BTC markets, 0.06 s in ETH and 0.07 s in sports. So a 10-step forecast looks 0.3–0.7 seconds ahead, and a 100-step forecast looks 3–7 seconds ahead.

## Findings in plain language

### 1. The paper's main claim holds up on Polymarket: reading events beats reading snapshots.

- **Which model wins.** In all 12 settings (3 market groups × 4 horizons), an event model (MBOFormer or MBOFusion) had the best score. The paper reported 11 of 12. In BTC at the two shortest horizons the lead over the snapshot model is too small to be sure of.
- **MBOFormer against the snapshot model with the same backbone.** MBOFormer is clearly better in 9 of 12 settings. It is clearly *worse* in one: BTC at the shortest horizon, by −0.018. The remaining two are too close to call.
- **Size of the gain.** The gain is usually 0.03–0.11 macro-F1. Macro-F1 is the average hit quality across up, flat and down; 0.2 is what "always guess flat" scores. On a typical score of 0.5–0.7, that is a real improvement.
- **Where the gain is largest.** The edge grows with the forecast horizon in BTC and sports. Sports at 100 steps gains +0.11. ETH gains the most at the shortest horizon (+0.09).
- **Against classic signals.** MBOFormer beats the logistic regression in all 12 settings. The deep model adds something over simple order-book imbalance and order-flow signals.

### 2. Why events help: snapshots hide part of the story.

- Between two snapshots taken 10 book changes apart, **13–21% of all order activity is invisible**. These are orders that were placed and pulled again in between, so the net change is zero. That answers the post's closing question, "what does an L2 pipeline lose between snapshots?" (chart 6).
- About 51% of events are new orders, 34–44% are cancellations and only 5–16% are trades. Most of what happens in the book never prints as a trade.

### 3. Prediction is not profit. Trading every signal loses money in every setting.

The test buys at the ask (or sells at the bid) when a signal fires, then exits h steps later. The fee is Polymarket's taker fee, 0.25·p·(p(1−p))², which tops out at 0.78¢ per share at a price of 50¢.

- **Every signal, MBOFormer:** the result is **negative after costs in all 12 settings**. The losses range from −0.6¢ to −3.0¢ per share per trade.
- **Before costs, the direction is right.** At the mid price, profits run from about 0¢ (sports, shortest horizon) to +1.8¢ per share. The model gets direction right more often than not, but the bid-ask spread (often 1–2¢) eats the edge.
- **Even perfect foresight loses at short horizons.** An "oracle" that knows the true up/flat/down label in advance **still loses money at the 10- and 20-step horizons** in every group. In sports it loses after fees at every horizon. At these speeds the spread is bigger than the typical move, so no classifier of this kind can make money as a taker there.

### 4. One possible exception, not proven: very confident signals at longer horizons.

Trading only the 10% most confident MBOFormer signals turned positive after spread and fees in some settings:

| Setting | Profit per share after spread + fee (95% CI) |
|---|---|
| BTC, 100 steps (~3 s) | +1.4¢ (+1.1 to +1.7) |
| ETH, 50 steps (~3 s) | +1.4¢ (+0.4 to +2.0) |
| ETH, 100 steps (~6 s) | +4.7¢ (+2.5 to +6.9) |

The result survived a one-step entry delay. Do **not** treat it as a strategy yet, for these reasons:

- **Look-ahead in the cutoff.** The "top 10%" cutoff was set from the confidence scores on the test set itself.
- **Little independent evidence.** There are only 5 test contracts per group. The trades overlap heavily: one every few hundredths of a second, each held for seconds, so they are not independent bets.
- **Latency is understated.** A one-step delay is only 30–70 ms. Probalytics' own delivery delay is about 45 ms median and up to about 1 s at the 99th percentile. Sending a real order to Polymarket adds more.
- **Assumed fee and size.** The fee formula is an assumption. Each trade is 1 share, so there is no market impact.
- **A likely simpler explanation.** In 15-minute crypto markets, prices often drift strongly toward 0 or 1 as the window ends. The confident signals may simply be catching that drift, which a much simpler rule could also catch.

### 5. Speed: tiny models are fast, but speed doesn't matter here.

| Model | Median time, 1 CPU thread |
|---|---|
| MBOFormer | 1.2 ms |
| L2-Former (snapshot model) | 0.7 ms |

The paper's 0.48 ms was on a newer Apple M2 chip. Both models are already faster than the market produces decisions (30–70 ms) and much faster than the data arrives. Shaving model latency gains nothing on Polymarket; the feed and the order round trip are the bottleneck.

## Clean re-test of point 4 (`retest_mboformer.py`, charts in `mboformer_retest/`)

**Correction to the first run.** One ETH contract had no book data. A bug in the train/validation/test split fallback then left ETH with only 2 test contracts and 1 validation contract, not 5 and 3. The ETH trading numbers in point 4 therefore rest on 2 contracts. The bug is fixed: the split is now 60/15/25 of whatever contracts exist.

**Rules fixed in the script before any result existed:**

- **Cutoff.** The "top 10% most confident" cutoff is set on the 3 *validation* contracts, never on test data.
- **Test data.** 30 *fresh* BTC and 30 fresh ETH 15-minute contracts, closing 23 Oct 00:00 – 24 Oct 06:00 UTC. All of them come after every contract used before.
- **Strategy.** h = 100 steps (2–4 s hold). Orders fill 0.5 s after the signal (0, 0.25 and 1 s are also reported). One position at a time. Prices are walked through the real book for 1 and 100 shares. Taker fee on entry and exit.
- **Pass rule.** Profitable after fees in **both** BTC and ETH, Holm-adjusted contract-clustered bootstrap p < 0.05, **and** better than the simple rules:
  - momentum;
  - "favourite drifts toward 0 or 1";
  - a random direction placed at MBOFormer's own signal times.

### Result: FAIL. The strategy is not profitable.

| Fresh contracts, pre-registered spec | Profit per share per trade after spread + fee (95% CI) | Contracts in profit |
|---|---|---|
| BTC 15m | −1.03¢ (−1.17 to −0.88) | 0 of 30 |
| ETH 15m | −0.10¢ (−0.51 to +0.33) | 14 of 30 |

### What the re-test shows

- **The prediction itself holds up.** On the 30 fresh contracts MBOFormer still beats the snapshot model by +0.03 to +0.08 macro-F1, and every confidence interval is above zero.
- **Its direction calls are real.** At the mid price the confident signals earn +1.2¢ (BTC) and +2.3¢ (ETH) per share, 0.5 s after the signal. Placing a random direction at the same moments earns about 0¢. The edge is in the direction, not in the timing.
- **The edge decays within a fraction of a second.** Using the profit per share for confident signals at h=100 on fresh contracts:

  | Fill delay | BTC | ETH |
  |---|---|---|
  | 0 s (instant) | +0.29¢ (22/30 contracts in profit) | +1.86¢ (29/30) |
  | 0.25 s | −0.73¢ | +0.21¢ (CI includes 0) |
  | 0.5 s | −1.03¢ | −0.10¢ |
  | 1 s | −1.31¢ | −0.60¢ |

  The money is made in the first quarter-second. Probalytics data alone arrives about 45 ms late at the median and up to about 1 s at the 99th percentile. A retail order to Polymarket takes longer still, so the instant-fill row is not achievable.
- **Size makes it worse.** At 100 shares per trade, walking the book costs a further 0.6–1.7¢ per share. Every setting loses money, even with instant fills.
- **Simple rules lose too.** Momentum and drift-to-0/1 lose −0.8 to −2.5¢ in every setting. The "drift toward resolution" story does not explain the first-run profit either. MBOFormer's direction calls are better than the simple rules, but not by enough to pay the spread and fee.
- **Why the first run looked profitable:**
  1. It assumed instant fills.
  2. ETH had only 2 test contracts, which inflated +4.7¢ (on fresh data it is +1.9¢ even with instant fills).
  3. Overlapping trades counted the same move many times, which made the confidence intervals look tighter than they were.

## Bottom line

- **Is it profitable?** No. With a realistic 0.5 s from signal to fill, the confident-signal strategy loses money in BTC (−1.0¢ per share per trade, 0 of 30 fresh contracts in profit) and breaks even at best in ETH (−0.1¢). It fails the pre-registered test. It would only pay with sub-quarter-second execution, and even then only at tiny size: about $0.30–0.45 per 15-minute window at 1 share, and a loss at 100 shares.
- **The science replicates.** On Polymarket, a 7,203-parameter model reading the raw event stream predicts short-term price direction better than the same model reading snapshots. It also beats classic order-book signals. This holds across crypto and sports markets and on contracts it never saw.
- **The trade does not, at least not simply.** Taking liquidity on every signal loses money everywhere, because the spread and fees are larger than the moves being predicted. Even perfect foresight loses at sub-second horizons.
- **The only promising part needs a proper test.** That is the top-confidence signals over several seconds. A proper test needs: the confidence cutoff fixed on the validation contracts, more test contracts, realistic 0.5–1 s latency, and a comparison with a simple "drift toward resolution" rule.
- **A better use.** Rather than trading on the signal directly, use it to decide when a maker (resting) order is likely to be filled or run over. Makers earn the spread instead of paying it.

## Known limits of this test

- Polymarket exposes price levels, not individual orders. Our "events" are therefore one level coarser than the paper's order-by-order (L3) data.
- Only the top 10 levels per side are used.
- "Execution" is matched to fills within a few seconds of the book change, so it is an approximation.
- MBOFusion was trained with one seed instead of three because of the CPU budget. Epochs were capped at 12, and several MBOFormer runs hit that cap while still improving, so its scores are, if anything, slightly understated.
- Results cover October 2026 only.
