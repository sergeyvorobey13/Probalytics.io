"""
Market Graph smoke test — Probalytics (early access)
Exercises every "Querying the graph" feature on real markets:
https://docs.probalytics.io/docs/market-graph#querying-the-graph

  1. Cypher endpoint     POST /api/graph/cypher
  2. Preconfigured       GET  /api/graph/q/{market,relations,underlying,outcome_set}?id=
  3. Doc example queries (IMPLIES chain, instrument, game, cross-platform pairs)
  4. Price-rule check    graph relations joined to last Yes fill price in ClickHouse

Needs in .env:
  PROBALYTICS_API_KEY_ID, PROBALYTICS_API_KEY_SECRET   (app.probalytics.io → API Keys)
  PROBALYTICS_CLICKHOUSE_*                             (only for step 4)
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

BASE = "https://api.probalytics.io/api/graph"
KEY_ID = os.environ.get("PROBALYTICS_API_KEY_ID")
KEY_SECRET = os.environ.get("PROBALYTICS_API_KEY_SECRET")
if not (KEY_ID and KEY_SECRET):
    sys.exit("Set PROBALYTICS_API_KEY_ID and PROBALYTICS_API_KEY_SECRET in .env")
HEADERS = {"Authorization": f"Bearer {KEY_ID}:{KEY_SECRET}",
           "Content-Type": "application/json"}

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 70)


# ─────────────────────────────────────────────────────────────
# HTTP helpers — 429 means "server busy", so back off and retry
# ─────────────────────────────────────────────────────────────
def _request(url, body=None, retries=5):
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers=HEADERS,
                                     method="POST" if data else "GET")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            msg = e.read().decode(errors="replace")[:300]
            if e.code == 403:
                sys.exit(f"403 — account has no market-graph access (early access). {msg}")
            raise RuntimeError(f"HTTP {e.code} for {url}: {msg}") from None


def cypher(query, **params):
    t0 = time.time()
    rows = _request(f"{BASE}/cypher", {"query": query, "params": params})["rows"]
    print(f"  [cypher] {len(rows)} rows in {time.time() - t0:.2f}s")
    return pd.DataFrame(rows)


def preset(name, market_id):
    return _request(f"{BASE}/q/{name}?" + urllib.parse.urlencode({"id": market_id}))


def section(title):
    print(f"\n{'─' * 70}\n{title}\n{'─' * 70}")


# ─────────────────────────────────────────────────────────────
# STEP 1 — Cypher: graph inventory + real seed markets
# ─────────────────────────────────────────────────────────────
section("1. Graph inventory")
print(cypher("MATCH (m:Market) RETURN m.domain AS domain, m.platform AS platform, "
             "count(*) AS markets ORDER BY markets DESC"))
print(cypher("MATCH ()-[r:RELATION]->() RETURN r.relation AS relation, "
             "r.cross_platform AS cross_platform, count(*) AS edges ORDER BY edges DESC"))

# Seed: an active cross-platform pair from the docs' example query
pairs = cypher("""
MATCH (a:Market)-[r:RELATION]->(b:Market)
WHERE r.cross_platform AND r.divergence < 0.005 AND a.status = 'ACTIVE' AND b.status = 'ACTIVE'
RETURN a.id AS a_id, a.platform AS a_plat, a.title AS a_title,
       b.id AS b_id, b.platform AS b_plat, b.title AS b_title,
       r.relation AS relation, r.exact AS exact, r.divergence AS divergence,
       r.divergence_n AS divergence_n, r.holds_if AS holds_if
LIMIT 100""")
section("Open cross-platform pairs with low divergence (doc example)")
print(pairs[["relation", "exact", "divergence", "divergence_n", "holds_if", "a_plat", "a_title", "b_title"]]
      .head(15).to_string(index=False) if len(pairs) else "  none")
if pairs.empty:
    sys.exit("No seed pairs found — nothing further to test")
seed = pairs.iloc[0]["a_id"]


# ─────────────────────────────────────────────────────────────
# STEP 2 — Preconfigured queries on the seed market
# ─────────────────────────────────────────────────────────────
for name in ["market", "relations", "underlying", "outcome_set"]:
    section(f"2. GET /q/{name}?id={seed}")
    res = preset(name, seed)
    rows = res.get("rows", res) if isinstance(res, dict) else res
    if isinstance(rows, list):
        print(f"  {len(rows)} rows")
        df = pd.DataFrame(rows)
        if name == "market":
            print(df.T.to_string(header=False))
        else:
            print(df.drop(columns=[c for c in df if c.endswith(("id", "url", "underlying"))])
                  .head(10).to_string(index=False) if rows else "  (empty)")
    else:
        print(json.dumps(rows, indent=2, default=str)[:1500])


# ─────────────────────────────────────────────────────────────
# STEP 3 — Remaining doc example queries
# ─────────────────────────────────────────────────────────────
section("3a. Everything the seed IMPLIES (nearest-neighbour chain)")
implier = cypher("MATCH (a:Market)-[r:RELATION]->(:Market) WHERE r.relation = 'IMPLIES' "
                 "AND a.status = 'ACTIVE' RETURN a.id AS id, a.title AS title, a.platform AS platform, "
                 "a.direction AS direction, a.lo AS lo, a.hi AS hi LIMIT 1")
if len(implier):
    s = implier.iloc[0]
    print(f"  from: [{s['platform']}] {s['title']}  ({s['direction']} lo={s['lo']} hi={s['hi']})")
    print(cypher("""
        MATCH (a:Market {id: $id})-[r:RELATION* 1..30 (e, n | WHERE e.relation = 'IMPLIES')]->(b:Market)
        RETURN DISTINCT b.platform AS platform, b.direction AS direction, b.lo AS lo, b.hi AS hi, b.title AS title""",
                 id=s["id"]).sort_values("lo").to_string(index=False))

section("3b. All markets on one instrument (UST10Y)")
print(cypher("MATCH (m:Market)-[:ASSERTS]->(u:Underlying) WHERE u.asset = 'UST10Y' "
             "RETURN m.platform AS platform, u.measure AS measure, count(*) AS markets"))

section("3c. Both platforms' markets for one game")
# A full scan over games exhausts the server's buffer pool, so take the game of one
# active cross-platform sports pair instead
game = cypher("""
    MATCH (a:Market)-[r:RELATION]->(b:Market)
    WHERE r.cross_platform AND a.domain = 'sports' AND a.status = 'ACTIVE' AND a.game_id IS NOT NULL
    RETURN a.game_id AS game LIMIT 1""")
if len(game):
    print("  game:", game.iloc[0]["game"])
    print(cypher("""
        MATCH (g:Game {id: $game})<-[:OF_GAME]-(u:Underlying)<-[:ASSERTS]-(m:Market)
        RETURN m.platform AS platform, m.title AS title, u.measure AS measure, m.lo AS lo, m.hi AS hi""",
                 game=game.iloc[0]["game"]).head(20).to_string(index=False))


# ─────────────────────────────────────────────────────────────
# STEP 4 — Price rules vs. last traded Yes price (ClickHouse)
# ─────────────────────────────────────────────────────────────
section("4. Price-rule check on the cross-platform pairs")
from clickhouse_driver import Client

ch = Client(host=os.environ["PROBALYTICS_CLICKHOUSE_HOST"],
            user=os.environ["PROBALYTICS_CLICKHOUSE_USERNAME"],
            password=os.environ["PROBALYTICS_CLICKHOUSE_PASSWORD"],
            database="probalytics", secure=True, port=9440)

ids = list(set(pairs["a_id"]) | set(pairs["b_id"]))
# Yes = outcome index 0 on both platforms (see markets docs); last fill in the past 7 days
rows = ch.execute("""
SELECT toString(m.id) AS id, argMax(toFloat64(f.price), f.timestamp) AS p_yes, max(f.timestamp) AS last_ts
FROM fills AS f
INNER JOIN (SELECT id, outcomes[1].id AS yes_id FROM markets FINAL WHERE toString(id) IN %(ids)s) AS m
    ON f.market_id = m.id AND f.outcome.id = m.yes_id
WHERE toString(f.market_id) IN %(ids)s AND f.timestamp > now() - INTERVAL 7 DAY
GROUP BY id""", {"ids": ids})
px = {r[0]: r[1] for r in rows}
print(f"  priced {len(px)}/{len(ids)} markets")

TOL = 0.02  # tolerance for "=" rules and ticks/staleness
RULES = {
    "IMPLIES":    lambda a, b: a <= b + TOL,
    "EQUIVALENT": lambda a, b: abs(a - b) <= TOL,
    "MUTEX":      lambda a, b: a + b <= 1 + TOL,
    "INVERSE":    lambda a, b: abs(a + b - 1) <= TOL,
    "NEAR":       lambda a, b: abs(a - b) <= 0.10,
}
pairs["p_a"] = pairs["a_id"].map(px)
pairs["p_b"] = pairs["b_id"].map(px)
priced = pairs.dropna(subset=["p_a", "p_b"]).copy()
priced["holds"] = [RULES[r](a, b) for r, a, b in zip(priced["relation"], priced["p_a"], priced["p_b"])]
priced["gap"] = (priced["p_a"] - priced["p_b"]).round(3)
print(priced[["relation", "a_plat", "a_title", "p_a", "b_plat", "p_b", "gap", "holds"]]
      .sort_values("holds").to_string(index=False))
if len(priced):
    print(f"\n  rule holds on {priced['holds'].mean():.0%} of {len(priced)} priced pairs")
