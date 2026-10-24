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
assert len(r["teams"]) > 10 and res["summary"]["season"]["ats"]["w"] > 0
print(f"smoke test passed: {len(p['games'])} predictions, {len(r['teams'])} rated teams")
EOF
rm -rf "$TMP"
