"""
Market Graph demo run — one real match, start to finish
Question: do Kalshi and Polymarket agree on Leyton Orient vs. Cambridge United (31 Oct 2026)?

  STEP 1  Start from one Kalshi market                     GET  /q/market
  STEP 2  Ask the graph what it is linked to               GET  /q/relations
  STEP 3  Get the whole outcome set (win / draw / win)     GET  /q/outcome_set
  STEP 4  Find the twin of every outcome on Polymarket     POST /cypher
  STEP 5  Pull live orderbooks for all six markets         ClickHouse orderbook_snapshots
  STEP 6  Check the price rules                            EQUIVALENT: A = B, outcome set: sum = 1
  STEP 7  Look for free money                              buy every outcome at the cheapest ask

Writes market_graph_demo.json — the numbers behind market_graph_test_run.html
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
import urllib.request
from datetime import datetime, timezone
from uuid import UUID

from clickhouse_driver import Client

GRAPH = "https://api.probalytics.io/api/graph"
AUTH = {"Authorization": f"Bearer {os.environ['PROBALYTICS_API_KEY_ID']}:{os.environ['PROBALYTICS_API_KEY_SECRET']}",
        "Content-Type": "application/json"}
SEED = "7545f970-71d1-514e-8eba-3fcf2986dfcc"   # Kalshi: "Cambridge wins"


def preset(name, market_id):
    req = urllib.request.Request(f"{GRAPH}/q/{name}?id={market_id}", headers=AUTH)
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["rows"]


def cypher(query, **params):
    req = urllib.request.Request(f"{GRAPH}/cypher", json.dumps({"query": query, "params": params}).encode(),
                                 AUTH, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["rows"]


out = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

# ─────────────────────────────────────────────────────────────
# STEP 1 — the starting market
# ─────────────────────────────────────────────────────────────
m = preset("market", SEED)[0]
out["seed"] = {k: m[k] for k in ["id", "platform", "title", "game_id", "measure", "direction", "lo", "hi", "status"]}
print("STEP 1", out["seed"])

# ─────────────────────────────────────────────────────────────
# STEP 2 — what the graph links it to
# ─────────────────────────────────────────────────────────────
rel = preset("relations", SEED)
out["relations"] = [{"relation": r["relation"], "peer_platform": r["peer_platform"], "peer_title": r["peer_title"],
                     "exact": r["exact"], "divergence": r["divergence"], "divergence_n": r["divergence_n"]}
                    for r in rel]
print("STEP 2", *out["relations"], sep="\n  ")

# ─────────────────────────────────────────────────────────────
# STEP 3 — the outcome set: Kalshi's three options for this match
# ─────────────────────────────────────────────────────────────
oset = preset("outcome_set", SEED)
out["outcome_set"] = {"kind": oset[0]["kind"], "members": oset[0]["members"],
                      "markets": [{"id": r["id"], "title": r["title"]} for r in oset]}
print("STEP 3", out["outcome_set"]["kind"], [r["title"] for r in oset])

# ─────────────────────────────────────────────────────────────
# STEP 4 — the Polymarket twin of each Kalshi outcome
# ─────────────────────────────────────────────────────────────
TWINS = """
MATCH (k:Market)-[r:RELATION]-(p:Market)
WHERE k.id IN $ids AND r.relation = 'EQUIVALENT' AND p.platform = 'POLYMARKET'
RETURN k.id AS kalshi_id, k.title AS kalshi, p.id AS poly_id, p.title AS polymarket
"""
twins = cypher(TWINS, ids=[r["id"] for r in oset])
out["twins_query"] = TWINS.strip()
out["twins"] = twins
print("STEP 4", *[(t["kalshi"], t["polymarket"]) for t in twins], sep="\n  ")

# ─────────────────────────────────────────────────────────────
# STEP 5 — live orderbooks: best bid / best ask on the Yes side
# ─────────────────────────────────────────────────────────────
ch = Client(host=os.environ["PROBALYTICS_CLICKHOUSE_HOST"], user=os.environ["PROBALYTICS_CLICKHOUSE_USERNAME"],
            password=os.environ["PROBALYTICS_CLICKHOUSE_PASSWORD"], database="probalytics", secure=True, port=9440)
ids = [UUID(t["kalshi_id"]) for t in twins] + [UUID(t["poly_id"]) for t in twins]
BOOKS = """
SELECT toString(market_id) AS id,
       argMax(bids[1].price, (timestamp, path_index)) AS best_bid,
       argMax(asks[1].price, (timestamp, path_index)) AS best_ask,
       max(timestamp) AS as_of
FROM orderbook_snapshots
WHERE market_id IN %(ids)s
  AND outcome.name = 'Yes' AND state = 'VERIFIED'
  AND timestamp > now() - INTERVAL 1 DAY
GROUP BY id
"""
book = {r[0]: {"bid": float(r[1]), "ask": float(r[2]), "as_of": r[3].strftime("%H:%M:%S")}
        for r in ch.execute(BOOKS, {"ids": ids})}
out["books_query"] = BOOKS.strip()

# 30-minute mid-price history for the chart
HIST = """
SELECT toString(market_id) AS id, toStartOfInterval(timestamp, INTERVAL 30 MINUTE) AS t,
       argMax((bids[1].price + asks[1].price) / 2, (timestamp, path_index)) AS mid
FROM orderbook_snapshots
WHERE market_id IN %(ids)s AND outcome.name = 'Yes' AND state = 'VERIFIED'
  AND length(bids) > 0 AND length(asks) > 0 AND timestamp > now() - INTERVAL 2 DAY
GROUP BY id, t ORDER BY t
"""
hist = {}
for mid_id, t, mid in ch.execute(HIST, {"ids": ids}):
    hist.setdefault(mid_id, []).append([t.strftime("%Y-%m-%dT%H:%M"), round(float(mid), 4)])

# ─────────────────────────────────────────────────────────────
# STEP 6 — price rules
#   EQUIVALENT:   P(Kalshi) = P(Polymarket)
#   outcome set:  P(home) + P(draw) + P(away) = 1 on each platform
# ─────────────────────────────────────────────────────────────
rows = []
for t in twins:
    k, p = book[t["kalshi_id"]], book[t["poly_id"]]
    k_mid, p_mid = (k["bid"] + k["ask"]) / 2, (p["bid"] + p["ask"]) / 2
    rows.append({"outcome": t["kalshi"], "kalshi_id": t["kalshi_id"], "poly_id": t["poly_id"],
                 "kalshi_bid": k["bid"], "kalshi_ask": k["ask"], "kalshi_mid": round(k_mid, 4),
                 "poly_bid": p["bid"], "poly_ask": p["ask"], "poly_mid": round(p_mid, 4),
                 "gap_cents": round(abs(k_mid - p_mid) * 100, 1),
                 # books overlap → no one can buy on one platform and sell higher on the other
                 "books_overlap": k["bid"] <= p["ask"] and p["bid"] <= k["ask"],
                 "kalshi_as_of": k["as_of"], "poly_as_of": p["as_of"],
                 "kalshi_hist": hist.get(t["kalshi_id"], []), "poly_hist": hist.get(t["poly_id"], [])})
out["pairs"] = rows
out["sum_kalshi_mid"] = round(sum(r["kalshi_mid"] for r in rows), 4)
out["sum_poly_mid"] = round(sum(r["poly_mid"] for r in rows), 4)

# ─────────────────────────────────────────────────────────────
# STEP 7 — free money? buy every outcome at the cheapest ask on either platform.
#   One outcome always pays $1, so a total cost below $1 would be a sure profit.
# ─────────────────────────────────────────────────────────────
out["cheapest_basket"] = round(sum(min(r["kalshi_ask"], r["poly_ask"]) for r in rows), 4)

for r in rows:
    print(f"STEP 6 {r['outcome']:<20} Kalshi {r['kalshi_bid']:.2f}/{r['kalshi_ask']:.2f}  "
          f"Poly {r['poly_bid']:.2f}/{r['poly_ask']:.2f}  gap {r['gap_cents']}¢  overlap={r['books_overlap']}")
print(f"STEP 6 sum of mids  Kalshi {out['sum_kalshi_mid']}  Polymarket {out['sum_poly_mid']}")
print(f"STEP 7 cheapest basket of all three outcomes: ${out['cheapest_basket']} (sure profit only if < $1)")

json.dump(out, open("market_graph_demo.json", "w"), indent=1, default=str)
print("wrote market_graph_demo.json")
