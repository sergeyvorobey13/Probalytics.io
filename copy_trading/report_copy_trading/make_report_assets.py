"""Render the B-roll frames for REPORT.md: 16:9 charts and code snapshots.

Reads the CSVs written by test_copy_trading.py (one folder up) and writes PNGs
into charts/ and code/ next to this file.

    venv/bin/python report_copy_trading/make_report_assets.py
"""

import io
import keyword
import tokenize
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CHARTS, CODE = HERE / "charts", HERE / "code"
CHARTS.mkdir(exist_ok=True)
CODE.mkdir(exist_ok=True)

# dark chart tokens (dataviz reference palette, dark steps)
SURFACE, PLANE = "#1a1a19", "#0d0d0d"
INK, INK2, MUTED = "#f0efec", "#c3c2b7", "#898781"
GRID, AXIS = "#2c2c2a", "#383835"
BLUE, ORANGE, AQUA, YELLOW = "#3987e5", "#d95926", "#199e70", "#c98500"
GOOD, BAD = "#0ca30c", "#e66767"
W, H, DPI = 16, 9, 120                      # 1920 x 1080

ANCHOR = pd.Timestamp("2026-10-24")          # day the backtest was run
FORM, TEST, BUFFER, FOLDS = 60, 30, 14, 6

plt.rcParams.update({
    "font.size": 20, "font.family": "DejaVu Sans",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})

res = pd.read_csv(ROOT / "copy_trading_results.csv")
folds = pd.read_csv(ROOT / "copy_trading_folds.csv")
sig = pd.read_csv(ROOT / "copy_trading_significance.csv")
mirror = res[res["exit"] == "mirror"]


def cell(group, lag, slip):
    r = mirror[(mirror.group == group) & (mirror.lag_s == lag) & (mirror.slip == slip)].iloc[0]
    return r


def money(v):
    return f"{'-' if v < 0 else '+' if v > 0 else ''}\\${abs(v):,.0f}"


def frame(title, subtitle=None):
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)
    fig.text(0.05, 0.93, title, fontsize=34, weight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.05, 0.855, subtitle, fontsize=20, color=INK2, va="top")
    fig.text(0.95, 0.03, "Polymarket fills, Feb–Oct 2026 · walk-forward backtest",
             fontsize=13, color=MUTED, ha="right")
    return fig


def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=8)


def save(fig, name):
    fig.savefig(CHARTS / name, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved", name)


# ── 1. hero: the money curve ────────────────────────────────────────────────
t = pd.read_csv(ROOT / "copy_trading_trades.csv", parse_dates=["entry_ts"])
t["exit_ts"] = pd.to_datetime(t["exit_ts"], format="mixed", errors="coerce")
t = t.sort_values("entry_ts")
STAKE = 10

fig = frame("Copying the best Polymarket traders",
            f"Cumulative profit at \\${STAKE} per copied trade, top-200 wallets, 6 months out of sample")
ax = fig.add_axes([0.11, 0.17, 0.66, 0.62])
style(ax)
for lag, color, name in [(0, BLUE, "Leaders' own prices"),
                         (30, ORANGE, "You, 30 s later")]:
    s = t[(t.group == "top") & (t.lag_s == lag)]
    y = (s["ret"] * STAKE).cumsum()
    ax.plot(s["entry_ts"], y, color=color, linewidth=3.5, solid_capstyle="round")
    ax.annotate(f"{name}\n{money(y.iloc[-1])}", (s["entry_ts"].iloc[-1], y.iloc[-1]),
                xytext=(14, 0), textcoords="offset points", va="center",
                fontsize=19, color=INK, linespacing=1.4)
ax.axhline(0, color=AXIS, linewidth=1.5)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: money(v)))
loc = mdates.MonthLocator()
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
save(fig, "01_profit_curve.png")


# ── 2. the edge decays with every second of delay ───────────────────────────
lags = [0, 1, 10, 30, 60, 300]
labels = ["0 s\n(leader)", "1 s", "10 s", "30 s", "60 s", "5 min"]
fig = frame("The edge disappears in seconds",
            "Return per \\$100 copied from the top wallets, by how late you are")
ax = fig.add_axes([0.08, 0.17, 0.74, 0.62])
style(ax)
x = np.arange(len(lags))
for slip, color, name in [(0.0, BLUE, "perfect fills"), (0.005, ORANGE, "+0.5¢ slippage")]:
    y = [100 * cell("top", l, slip)["ret"] for l in lags]
    ax.plot(x, y, color=color, linewidth=3.5, marker="o", markersize=12,
            markeredgecolor=SURFACE, markeredgewidth=2.5)
    ax.annotate(name, (x[-1], y[-1]), xytext=(18, 0), textcoords="offset points",
                va="center", fontsize=19, color=INK)
first = 100 * cell("top", 0, 0.0)["ret"]
ax.annotate(f"+\\${first:.0f}", (0, first), xytext=(18, 6), textcoords="offset points",
            fontsize=22, weight="bold", color=INK)
ax.axhline(0, color=INK2, linewidth=1.5)
ax.set_xticks(x, labels)
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: money(v)))
save(fig, "02_delay_decay.png")


# ── 3. top vs random vs bottom ──────────────────────────────────────────────
fig = frame("Skill is real, but you can't buy it",
            "Return per \\$100: at the leaders' own prices vs. what a copier actually gets (30 s, +0.5¢)")
ax = fig.add_axes([0.08, 0.15, 0.86, 0.64])
style(ax)
groups = [("top", "Top 200"), ("random", "Random 200"), ("bottom", "Bottom 200")]
x = np.arange(3)
wd = 0.36
for i, (lag, slip, color, name) in enumerate([(0, 0.0, BLUE, "Leader's own price"),
                                               (30, 0.005, ORANGE, "Copier")]):
    vals = [100 * cell(g, lag, slip)["ret"] for g, _ in groups]
    xs = x + (i - 0.5) * (wd + 0.02)
    ax.bar(xs, vals, width=wd, color=color, label=name, zorder=2)
    for xx, v in zip(xs, vals):
        ax.text(xx, v + (0.6 if v >= 0 else -0.6), f"{v:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=20, color=INK)
ax.axhline(0, color=INK2, linewidth=1.5)
ax.set_xticks(x, [n for _, n in groups], fontsize=22, color=INK2)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: money(v)))
ax.set_ylim(-13.5, 17)
ax.legend(frameon=False, loc="upper right", fontsize=19, labelcolor=INK)
save(fig, "03_groups.png")


# ── 4. every fold loses ─────────────────────────────────────────────────────
def fold_label(k):
    end = ANCHOR - pd.Timedelta(days=BUFFER + k * TEST)
    start = end - pd.Timedelta(days=TEST)
    return f"{start:%d %b}\n– {end:%d %b}"

fig = frame("Six months, six losing months",
            "Copier return per \\$100, top wallets, each 30-day test window (30 s late, +0.5¢)")
ax = fig.add_axes([0.08, 0.17, 0.86, 0.62])
style(ax)
f = folds.sort_values("fold", ascending=False)
vals = 100 * f["top"].to_numpy()
x = np.arange(len(f))
ax.bar(x, vals, width=0.6, color=ORANGE, zorder=2)
for xx, v in zip(x, vals):
    ax.text(xx, v - 0.4, f"{v:+.1f}", ha="center", va="top", fontsize=20, color=INK)
ax.axhline(0, color=INK2, linewidth=1.5)
ax.set_xticks(x, [fold_label(k) for k in f["fold"]], fontsize=17)
ax.set_ylim(vals.min() * 1.25, 2)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: money(v)))
save(fig, "04_folds.png")


# ── 5. holding time ─────────────────────────────────────────────────────────
top30 = t[(t.group == "top") & (t.lag_s == 30)].copy()
top30["hold_min"] = (top30["exit_ts"] - top30["entry_ts"]).dt.total_seconds() / 60
edges = [0, 5, 15, 60, 240, 1440, 10080, np.inf]
names = ["< 5 min", "5–15 min", "15–60 min", "1–4 h", "4–24 h", "1–7 days", "> 7 days"]
top30["bucket"] = pd.cut(top30["hold_min"], edges, labels=names, right=False)
hb = top30.dropna(subset=["bucket"]).groupby("bucket", observed=False)["ret"].agg(["size", "mean"])
hb["share"] = 100 * hb["size"] / hb["size"].sum()
hb["ret100"] = 100 * hb["mean"]

fig = frame("Fast trades are where copiers lose",
            "Top wallets, copied 30 s late: how long the position is held")
a1 = fig.add_axes([0.13, 0.17, 0.36, 0.56])
a2 = fig.add_axes([0.57, 0.17, 0.38, 0.56])
y = np.arange(len(hb))[::-1]
style(a1, "x")
a1.barh(y, hb["share"], color=BLUE, height=0.62, zorder=2)
for yy, v in zip(y, hb["share"]):
    a1.text(v + 0.8, yy, f"{v:.0f}%", va="center", fontsize=18, color=INK)
a1.set_yticks(y, names, fontsize=18, color=INK2)
a1.set_title("Share of trades", loc="left", fontsize=22, pad=14)
a1.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
style(a2, "x")
cols = [BAD if v < 0 else GOOD for v in hb["ret100"]]
a2.barh(y, hb["ret100"], color=cols, height=0.62, zorder=2)
for yy, v in zip(y, hb["ret100"]):
    a2.text(v + (0.3 if v >= 0 else -0.3), yy, f"{v:+.1f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=18, color=INK)
a2.axvline(0, color=INK2, linewidth=1.5)
a2.set_yticks(y, [""] * len(y))
a2.set_xlim(hb["ret100"].min() - 3, hb["ret100"].max() + 3)
a2.set_title("Return per \\$100 copied", loc="left", fontsize=22, pad=14)
a2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: money(v)))
save(fig, "05_holding_time.png")
med = top30["hold_min"].median()
print(f"holding: median {med:.1f} min, buckets:\n{hb.round(2)}")


# ── 6. the experiment design: walk-forward timeline ─────────────────────────
fig = frame("How the test works: pick, then copy — never peek",
            "Each fold ranks wallets on 60 days of history, then copies them for the next 30 days")
ax = fig.add_axes([0.13, 0.15, 0.82, 0.64])
style(ax, "x")
for k in range(FOLDS):
    te = ANCHOR - pd.Timedelta(days=BUFFER + k * TEST)
    ts = te - pd.Timedelta(days=TEST)
    fs = ts - pd.Timedelta(days=FORM)
    yy = k
    ax.barh(yy, (ts - fs).days, left=mdates.date2num(fs), color=BLUE, height=0.55, zorder=2)
    ax.barh(yy, (te - ts).days - 1, left=mdates.date2num(ts) + 1, color=ORANGE, height=0.55, zorder=2)
    if k == 0:
        ax.text(mdates.date2num(fs) + FORM / 2, yy, "PICK (60 d)", ha="center", va="center",
                fontsize=17, color="#ffffff", weight="bold")
        ax.text(mdates.date2num(ts) + TEST / 2, yy, "COPY", ha="center", va="center",
                fontsize=17, color="#ffffff", weight="bold")
ax.set_yticks(range(FOLDS), [f"Fold {k + 1}" for k in range(FOLDS)], fontsize=18, color=INK2)
ax.invert_yaxis()
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
save(fig, "06_walk_forward.png")


# ── 7. verdict card ─────────────────────────────────────────────────────────
fig = frame("The verdict", "Pre-registered tests at 30 s delay, +0.5¢ slippage, Holm-corrected")
rows = [
    ("Copying the top wallets makes money", sig.iloc[0]),
    ("Top wallets beat random wallets", sig.iloc[1]),
    ("Top wallets beat the worst wallets", sig.iloc[2]),
]
for i, (claim, r) in enumerate(rows):
    yy = 0.70 - i * 0.18
    ok = r["verdict"] == "YES"
    fig.text(0.07, yy, "✓" if ok else "✗", fontsize=54, color=GOOD if ok else BAD,
             va="center", weight="bold")
    fig.text(0.14, yy + 0.02, claim, fontsize=30, color=INK, va="center")
    fig.text(0.14, yy - 0.045,
             f"estimate {100 * r['est']:+.1f} per \\$100  ·  95% CI [{100 * r['lo']:+.1f}, "
             f"{100 * r['hi']:+.1f}]  ·  p = {r['p_holm']:.3f}",
             fontsize=19, color=INK2, va="center")
save(fig, "07_verdict.png")


# ── code snapshots (no pygments here, so a tiny tokenize-based highlighter) ──
SYN = dict(kw="#c792ea", str="#c3e88d", num="#f78c6c", com="#7f8c98",
           fn="#82aaff", default="#e6e6e3", op="#89ddff")


def highlight(src):
    """Yield (line, [(text, color), ...]) for each source line."""
    lines = src.splitlines()
    colored = [[] for _ in lines]
    prev_end = [(1, 0)]
    toks = list(tokenize.generate_tokens(io.StringIO(src + "\n").readline))
    after_def = False
    for tok in toks:
        if tok.type in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER,
                        tokenize.INDENT, tokenize.DEDENT):
            continue
        (sr, sc), (er, ec) = tok.start, tok.end
        if tok.type == tokenize.COMMENT:
            c = SYN["com"]
        elif tok.type == tokenize.STRING or tok.type == getattr(tokenize, "FSTRING_START", -1):
            c = SYN["str"]
        elif tok.type in (getattr(tokenize, "FSTRING_MIDDLE", -1), getattr(tokenize, "FSTRING_END", -1)):
            c = SYN["str"]
        elif tok.type == tokenize.NUMBER:
            c = SYN["num"]
        elif tok.type == tokenize.NAME and keyword.iskeyword(tok.string):
            c = SYN["kw"]
        elif tok.type == tokenize.NAME and after_def:
            c = SYN["fn"]
        elif tok.type == tokenize.OP:
            c = SYN["op"]
        else:
            c = SYN["default"]
        after_def = tok.type == tokenize.NAME and tok.string in ("def", "class")
        for r in range(sr, er + 1):
            line = lines[r - 1] if r - 1 < len(lines) else ""
            a = sc if r == sr else 0
            b = ec if r == er else len(line)
            colored[r - 1].append((a, line[a:b], c))
    out = []
    for line, parts in zip(lines, colored):
        parts.sort()
        segs, pos = [], 0
        for a, text, c in parts:
            if a > pos:
                segs.append((line[pos:a], SYN["default"]))
            segs.append((text, c))
            pos = a + len(text)
        if pos < len(line):
            segs.append((line[pos:], SYN["default"]))
        out.append(segs)
    return out


def snapshot(name, caption, src, first_line):
    src = src.rstrip("\n")
    n = len(src.splitlines())
    width = max(len(l) for l in src.splitlines())
    # fit the code box: usable width/height in points over char width / line height
    fs = min(22, 0.90 * W * 72 * 0.90 / (0.602 * (width + 6)), 0.80 * H * 72 * 0.86 / (1.55 * n))
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=PLANE)
    fig.text(0.05, 0.94, caption, fontsize=28, color=INK, weight="bold", va="top")
    # editor window
    win = fig.add_axes([0.05, 0.06, 0.90, 0.80])
    win.set_xlim(0, 1)
    win.set_ylim(0, 1)
    win.axis("off")
    win.add_patch(matplotlib.patches.FancyBboxPatch(
        (0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.015",
        facecolor="#1e2127", edgecolor="#2c313a", linewidth=1.5, transform=win.transAxes))
    for i, c in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        win.add_patch(matplotlib.patches.Circle((0.02 + i * 0.018, 0.96), 0.0055,
                                                color=c, transform=win.transAxes))
    win.text(0.5, 0.96, "test_copy_trading.py", ha="center", va="center",
             fontsize=15, color=MUTED, family="DejaVu Sans")
    top = 0.90
    step = (fs * 1.55) / (0.80 * H * 72)          # line height in axes units
    cw = (fs * 0.602) / (0.90 * W * 72)           # monospace char width in axes units
    for i, segs in enumerate(highlight(src)):
        yy = top - i * step
        win.text(0.015, yy, f"{first_line + i:>4}", fontsize=fs, family="DejaVu Sans Mono",
                 color="#4b5263", va="top")
        col = 0
        for text, c in segs:
            win.text(0.07 + col * cw, yy, text, fontsize=fs, family="DejaVu Sans Mono",
                     color=c, va="top")
            col += len(text)
    fig.savefig(CODE / name, dpi=DPI, facecolor=PLANE)
    plt.close(fig)
    print("saved", name)


code = (ROOT / "test_copy_trading.py").read_text().splitlines()


def lines_of(start_pat, end_pat=None, n=None, skip=()):
    """Source lines from the line matching start_pat up to (excl.) end_pat."""
    i = next(k for k, l in enumerate(code) if l.startswith(start_pat))
    if n is not None:
        j = i + n
    else:
        j = next(k for k in range(i + 1, len(code)) if code[k].startswith(end_pat))
    body = [l for k, l in enumerate(code[i:j]) if k not in skip]
    while body and not body[-1].strip():
        body.pop()
    return "\n".join(body), i + 1


src, ln = lines_of("FORM_DAYS = ", "ANCHOR = ")
snapshot("c1_settings.png", "Every rule fixed before looking at results", src, ln)
src, ln = lines_of("def fold_windows", "def combine_formation")
snapshot("c2_walk_forward.png", "Walk-forward: pick on the past, copy in the future", src, ln)
src, ln = lines_of("def combine_formation", "def pick_groups")
snapshot("c3_scoring.png", "Ranking traders by how reliably they beat the price", src, ln)
src, ln = lines_of("def simulate", n=22)
snapshot("c4_copy_entry.png", "Copying late: you get the NEXT real trade, not the leader's", src, ln)
src, ln = lines_of("def trade_returns", "def _units")
snapshot("c5_returns.png", "Profit per dollar, after fees and slippage", src, ln)
