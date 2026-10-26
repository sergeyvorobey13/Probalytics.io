"""Profit curves from copy_trading_trades.csv (written by test_copy_trading.py)."""

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker
import pandas as pd

STAKE = 10                      # dollars per copied trade, for the left panel
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

path = sys.argv[1] if len(sys.argv) > 1 else "copy_trading_trades.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "copy_trading_profit_curve.png"
t = pd.read_csv(path, parse_dates=["entry_ts"]).sort_values("entry_ts")
copy_lag = t.loc[t["lag_s"] > 0, "lag_s"].iloc[0]


def series(group, lag):
    s = t[(t["group"] == group) & (t["lag_s"] == lag)]
    return s["entry_ts"], s["ret"]


plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE)


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.axhline(0, color=INK2, linewidth=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_ylabel(ylabel)
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def end_label(ax, x, y, text, dy=0):
    ax.annotate(text, (x.iloc[-1], y.iloc[-1]), xytext=(6, dy), textcoords="offset points",
                va="center", fontsize=9, color=INK)


def money(v):
    return f"{'-' if v < 0 else ''}\\${abs(v):,.0f}"


# Left: the account of someone copying the top wallets, $STAKE per trade
for lag, color, name in [(0, BLUE, "at the leaders' own prices (impossible)"),
                         (copy_lag, ORANGE, f"you, copying {copy_lag}s late")]:
    x, r = series("top", lag)
    y = (r * STAKE).cumsum()
    a1.plot(x, y, color=color, linewidth=2, label=name, solid_capstyle="round")
    end_label(a1, x, y, money(y.iloc[-1]))
style(a1, f"Profit from copying the top wallets (\\${STAKE} per trade)", "cumulative profit, \\$")
a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: money(v)))
a1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)

# Right: running return per $100 copied, by which wallets you copy
ends = []
for g, color, name in [("top", ORANGE, "top wallets"), ("random", AQUA, "random wallets"),
                       ("bottom", YELLOW, "bottom wallets")]:
    x, r = series(g, copy_lag)
    y = 100 * r.cumsum() / range(1, len(r) + 1)
    x, y = x.iloc[200:], y.iloc[200:]          # the first few hundred trades are just noise
    a2.plot(x, y, color=color, linewidth=2, label=name, solid_capstyle="round")
    ends.append((y.iloc[-1], x, y))
# nudge end labels apart when lines finish close together
ends.sort(key=lambda e: e[0])
prev = None
for v, x, y in ends:
    dy = 0 if prev is None or v - prev > 1.5 else 9
    end_label(a2, x, y, f"{v:+.1f}", dy=dy)
    prev = v
style(a2, f"Running return per \\$100 copied ({copy_lag}s late)", "\\$ profit per \\$100 copied")
a2.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)

fig.tight_layout()
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"saved {out}")
