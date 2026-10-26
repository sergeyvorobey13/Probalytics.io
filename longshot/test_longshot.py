"""
Longshot-Fade Backtest — Probalytics Data
Testing whether cheap Kalshi contracts (<$0.20) are overpriced,
by fading them and measuring return on capital.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
import matplotlib.pyplot as plt
from clickhouse_driver import Client

# ─────────────────────────────────────────────────────────────
# STEP 1 — Connect to Probalytics (ClickHouse)
# ─────────────────────────────────────────────────────────────
client = Client(
    host=os.environ["PROBALYTICS_CLICKHOUSE_HOST"],
    user=os.environ["PROBALYTICS_CLICKHOUSE_USERNAME"],
    password=os.environ["PROBALYTICS_CLICKHOUSE_PASSWORD"],
    database="probalytics",
    secure=True,
    port=9440,
)

# Sanity check: confirm the connection works before doing anything else
rows, columns = client.execute(
    "SELECT title FROM markets WHERE platform = 'KALSHI' AND status = 'ACTIVE' LIMIT 5",
    with_column_types=True,
)
for row in rows:
    print(row)


# ─────────────────────────────────────────────────────────────
# STEP 2 — Build the universe: resolved binary markets
# ─────────────────────────────────────────────────────────────
markets_query = """
SELECT id, title, category, resolution_winning_outcome_id
FROM markets
WHERE platform = 'KALSHI'
  AND market_type = 'BINARY'
  AND status = 'RESOLVED'
  AND resolution_resolved_at >= now() - INTERVAL 180 DAY
"""

rows, columns = client.execute(markets_query, with_column_types=True)
col_names = [c[0] for c in columns]
markets_df = pd.DataFrame(rows, columns=col_names)

print(f"Found {len(markets_df)} resolved markets")
print(markets_df.head())


# ─────────────────────────────────────────────────────────────
# STEP 3 — Pull the longshot fills and tag the fade
# ─────────────────────────────────────────────────────────────
fills_query = """
SELECT
    f.market_id,
    f.timestamp,
    f.outcome.id AS outcome_id,
    f.price,
    f.size,
    m.resolution_winning_outcome_id,
    (f.outcome.id != m.resolution_winning_outcome_id) AS fade_won
FROM fills f
JOIN markets m ON f.market_id = m.id
WHERE f.platform = 'KALSHI'
  AND m.market_type = 'BINARY'
  AND m.status = 'RESOLVED'
  AND f.price < 0.20
  AND f.timestamp <= m.resolution_resolved_at - INTERVAL 1 HOUR
  AND f.timestamp >= now() - INTERVAL 30 DAY
"""

rows, columns = client.execute(fills_query, with_column_types=True)
col_names = [c[0] for c in columns]
fills_df = pd.DataFrame(rows, columns=col_names)
fills_df.to_csv("fills_kalshi.csv", index=False)

print(f"Found {len(fills_df)} longshot trades")
print(fills_df.head())


# ─────────────────────────────────────────────────────────────
# STEP 4 — Score the strategy (dollar-weighted, not averaged %)
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("fills_kalshi.csv")
df["fade_won"] = df["fade_won"].astype(bool)

# Fading a longshot means buying the complement outcome at (1 - price)
df["fade_stake"] = df["size"] * (1 - df["price"])
df["dollar_pnl"] = df.apply(
    lambda r: r["size"] * r["price"] if r["fade_won"] else -r["size"] * (1 - r["price"]),
    axis=1
)

total_staked = df["fade_stake"].sum()
total_pnl = df["dollar_pnl"].sum()
print(f"Total staked (on the fade): {total_staked:,.2f}")
print(f"Total P&L: {total_pnl:,.2f}")
print(f"Return on capital: {total_pnl / total_staked * 100:.2f}%")


# ─────────────────────────────────────────────────────────────
# STEP 5 — Out-of-sample check: split the data in half
# ─────────────────────────────────────────────────────────────
df["timestamp"] = pd.to_datetime(df["timestamp"])
midpoint = df["timestamp"].median()
first_half = df[df["timestamp"] < midpoint]
second_half = df[df["timestamp"] >= midpoint]

for name, half in [("First half", first_half), ("Second half", second_half)]:
    staked = half["fade_stake"].sum()
    pnl = half["dollar_pnl"].sum()
    print(f"{name}: {len(half):,} trades, staked {staked:,.2f}, "
          f"P&L {pnl:,.2f}, return {pnl/staked*100:.2f}%")


# ─────────────────────────────────────────────────────────────
# STEP 6 — Plot cumulative RETURN ON CAPITAL (not raw dollars)
# ─────────────────────────────────────────────────────────────
df = df.sort_values("timestamp")
df["cum_pnl"] = df["dollar_pnl"].cumsum()
df["cum_staked"] = df["fade_stake"].cumsum()
df["cum_return_pct"] = df["cum_pnl"] / df["cum_staked"] * 100

plt.figure(figsize=(10, 5))
plt.plot(df["timestamp"], df["cum_return_pct"])
plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
plt.title("Longshot-fade strategy — cumulative return on capital (30 days, Kalshi)")
plt.xlabel("Date")
plt.ylabel("Cumulative return (%)")
plt.tight_layout()
plt.savefig("equity_curve_pct_kalshi.png")
print("Saved chart to equity_curve_pct_kalshi.png")