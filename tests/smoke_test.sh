#!/usr/bin/env bash
# Runs the full pipeline on made-up data in a temporary copy of the project.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
cp -r "$HERE/src" "$HERE/tests" "$TMP/"
mkdir -p "$TMP/docs"
cd "$TMP"
export CFB_OFFLINE=1 CFB_SEASON=2026
python tests/make_sample_data.py "$TMP"
python src/backtest.py
python src/predict.py --now 2026-10-27T15:00Z
python - <<'EOF'
import json
p = json.load(open("docs/data/predictions.json"))
r = json.load(open("docs/data/ratings.json"))
res = json.load(open("docs/data/results.json"))
assert len(p["games"]) > 0, "no predictions"
assert all(g["home"]["proj"] is not None for g in p["games"])
assert any(g["qb"]["home"]["status"] == "out" or g["qb"]["away"]["status"] == "out" for g in p["games"])
assert len(r["teams"]) > 10 and res["summary"]["season"]["spread"]["w"] > 0
h = json.load(open("docs/data/history/index.json"))
assert len(h["seasons"]) >= 5, h
g = json.load(open(f"docs/data/history/{h['seasons'][-1]}.json"))["games"][0]
assert all(k in g for k in ("su", "ml_u", "ats", "ou")), g
assert any(x["splits"] and x["splits"]["rlm"] for x in p["games"]), "expected a reverse line movement flag"
assert any(x["splits"] and x["splits"]["sharp"] for x in p["games"]), "expected a sharp money flag"
news = [q["news"] for x in p["games"] for q in x["qb"].values() if q.get("news")]
assert any(n["suggested"] == "questionable" for n in news), "expected unverified QB news"
assert sum(1 for x in p["games"] for q in x["qb"].values() if q["status"] == "out") >= 2, "confirmation file not applied"
assert any(q.get("if_out") for x in p["games"] for q in x["qb"].values() if q.get("news")), "expected an if-he-sits projection for news"
assert any(q.get("auto") and q["status"] == "out" for x in p["games"] for q in x["qb"].values()), "confirmed news should apply"
print(f"smoke test passed: {len(p['games'])} predictions, {len(r['teams'])} rated teams")
EOF
rm -rf "$TMP"
