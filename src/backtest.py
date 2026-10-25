"""Step 7: walk-forward backtest. The only judge of whether a change helps.

For each test season, the ML model trains on earlier seasons only, and every
week's ratings use only games played before that week. Every game is graded
four ways (winner, moneyline, spread, over/under) and saved for the website's
History page.

    python src/backtest.py                       # all feature groups
    python src/backtest.py --groups CORE         # test a smaller model
    python src/backtest.py --groups CORE STYLE --w 0.7
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import features  # noqa: E402
import grading  # noqa: E402
from common import DATA, SITE_DATA, load_games, write_json  # noqa: E402
from model import MarketAdjust, Model  # noqa: E402
from ratings import RatingEngine  # noqa: E402


def build_table(games=None):
    """Feature table of every finished game (ratings settings come from config)."""
    games = load_games() if games is None else games
    engine = RatingEngine(games)
    seasons = sorted(s for s in games["season"].unique() if s > config.FIRST_SEASON)
    table = features.build(games, engine, seasons=seasons)
    return table[table["completed"].astype(bool) & table["actual_margin"].notna()]


def walk_forward(table, seasons, groups=None, w=None):
    out = []
    for s in seasons:
        train, test = table[table["season"] < s], table[table["season"] == s].copy()
        if len(test) == 0 or train["season"].nunique() < 2:
            continue
        test["pred_margin"], test["pred_total"] = Model(groups, w).fit(train).predict(test)
        if config.MARKET_ADJUST and out:
            # learn the movement adjustment from earlier seasons' out-of-sample projections only
            prior = pd.concat(out)
            adj = MarketAdjust().fit(prior, prior["pred_margin_base"], prior["pred_total_base"])
            test["pred_margin_base"], test["pred_total_base"] = test["pred_margin"], test["pred_total"]
            test["pred_margin"], test["pred_total"] = adj.apply(test, test["pred_margin"], test["pred_total"])
        else:
            test["pred_margin_base"], test["pred_total_base"] = test["pred_margin"], test["pred_total"]
        out.append(test)
    if not out:
        raise SystemExit("Not enough seasons to backtest yet.")
    return grading.grade(pd.concat(out))


def our_picks_open(bt):
    """The '★ Our picks' rule graded at the opening line (seasons with openers): 5+ point edge from week 5 on."""
    o = bt[bt["spread_open"].notna()]
    if o.empty:
        return None
    e = o["pred_margin"] + o["spread_open"]
    cover = np.sign(e) * (o["actual_margin"] + o["spread_open"])
    m = (e.abs() >= config.OUR_PICK_MIN) & ((o["season_type"] == "postseason") | (o["week"] >= config.OUR_PICK_FROM_WEEK))
    w, l, p = int((cover[m] > 0).sum()), int((cover[m] < 0).sum()), int((cover[m] == 0).sum())
    cc = (np.sign(e) * (o["actual_margin"] + o["spread"]))[m & o["spread"].notna()]  # same picks at the close
    wc, lc, pc = int((cc > 0).sum()), int((cc < 0).sum()), int((cc == 0).sum())
    return {"w": w, "l": l, "p": p, "pct": round(w / (w + l), 4) if w + l else None,
            "units": round(w * 100 / 110 - l, 2), "from": int(o["season"].min()), "to": int(o["season"].max()),
            "close": {"w": wc, "l": lc, "p": pc, "pct": round(wc / (wc + lc), 4) if wc + lc else None,
                      "units": round(wc * 100 / 110 - lc, 2)}}


def accuracy(bt):
    d = bt[bt["spread"].notna()]
    return {
        "games": int(len(bt)),
        "mae_model": round(float((bt["actual_margin"] - bt["pred_margin"]).abs().mean()), 2),
        "mae_ratings_only": round(float((bt["actual_margin"] - bt["rating_margin"]).abs().mean()), 2),
        "mae_market": round(float((d["actual_margin"] + d["spread"]).abs().mean()), 2) if len(d) else None,
        "mae_market_same_games_model": round(float((d["actual_margin"] - d["pred_margin"]).abs().mean()), 2) if len(d) else None,
        "margin_sd": round(float((bt["actual_margin"] - bt["pred_margin"]).std()), 2),
    }


def run(groups=None, w=None, save=True, table=None, quiet=False):
    t0 = time.time()
    table = build_table() if table is None else table
    if not quiet:
        print(f"feature table: {len(table)} games in {time.time() - t0:.0f}s")
    bt = walk_forward(table, config.BACKTEST_SEASONS, groups, w)

    overall = {**grading.summary(bt), **accuracy(bt), "our_picks_open": our_picks_open(bt)}
    by_season = {int(s): {**grading.summary(d), **accuracy(d)} for s, d in bt.groupby("season")}
    edges = {
        "spread": grading.edge_table(bt, "edge", "ats"),
        "total": grading.edge_table(bt, "total_edge", "ou"),
        "winner_by_confidence": [
            {"min_win_chance": c, **grading.record(bt.loc[bt["p_pick"] >= c, "su"])} for c in (0.5, 0.6, 0.7, 0.8, 0.9)],
    }
    resid = bt["actual_margin"] - bt["rating_margin"]
    qb_effect = {
        "home_qb_out_games": int(bt["qb_out_home"].sum()), "away_qb_out_games": int(bt["qb_out_away"].sum()),
        "home_qb_out_points": round(float(resid[bt["qb_out_home"] == 1].mean()), 2) if bt["qb_out_home"].sum() else None,
        "away_qb_out_points": round(float(-resid[bt["qb_out_away"] == 1].mean()), 2) if bt["qb_out_away"].sum() else None,
    }
    if not quiet:
        print_report(overall, by_season, edges, qb_effect, groups or config.FEATURE_GROUPS, config.BLEND_W if w is None else w)
    if save:
        settings = {"groups": groups or config.FEATURE_GROUPS, "blend_w": config.BLEND_W if w is None else w,
                    "rating_alpha": config.RATING_ALPHA, "half_life_days": config.HALF_LIFE_DAYS,
                    "blowout_cap": config.BLOWOUT_CAP, "strong_edge": config.STRONG_EDGE}
        write_json({"generated": pd.Timestamp.now(tz="UTC"), "settings": settings, "overall": overall,
                    "by_season": by_season, "edges": edges, "qb_effect": qb_effect}, SITE_DATA, "backtest.json")
        write_history(bt)
        bt.to_csv(os.path.join(DATA, "backtest_games.csv"), index=False)
    return bt, overall


def write_history(bt):
    """One file per season for the History page, plus an index."""
    for s, d in bt.groupby("season"):
        write_json({"season": int(s), "source": "backtest", "games": grading.history_rows(d)},
                   SITE_DATA, "history", f"{int(s)}.json")
    update_history_index([int(s) for s in bt["season"].unique()])


def update_history_index(seasons):
    import json
    path = os.path.join(SITE_DATA, "history", "index.json")
    have = json.load(open(path))["seasons"] if os.path.exists(path) else []
    write_json({"seasons": sorted(set(have) | set(seasons), reverse=True), "current": config.THIS_SEASON,
                "strong_edge": config.STRONG_EDGE, "sharp_gap": config.SHARP_GAP, "public_min": config.PUBLIC_MIN,
                "rlm_move": config.RLM_MIN_MOVE}, path)


def print_report(overall, by_season, edges, qb_effect, groups, w):
    f = lambda r: "   –  " if r.get("pct") is None else f"{r['pct']:6.1%}"
    print(f"\nGroups: {' + '.join(groups)}   blend w = {w}   alpha = {config.RATING_ALPHA}   half-life = {config.HALF_LIFE_DAYS}")
    print(f"{'season':>7} {'games':>6} {'MAE':>6} {'MAE mkt':>8} {'Winner':>7} {'ML u':>7} {'Spread':>7} {'ATS u':>7} {'O/U':>7}")
    rows = list(by_season.items()) + [("ALL", overall)]
    for s, r in rows:
        print(f"{s:>7} {r['games']:>6} {r['mae_model']:6.2f} {r['mae_market'] or float('nan'):8.2f} {f(r['winner']):>7} "
              f"{r['moneyline']['units']:7.1f} {f(r['spread']):>7} {r['spread']['units']:7.1f} {f(r['total']):>7}")
    print("Spread record by minimum edge:", "  ".join(
        f"{e['min_edge']}+: {f(e)} ({e['games']})" for e in edges["spread"]))
    print("Winner record by win chance:", "  ".join(
        f"{int(e['min_win_chance'] * 100)}%+: {f(e)}" for e in edges["winner_by_confidence"]))
    print(f"Value moneylines: {overall['value_ml']['bets']} bets, {overall['value_ml']['units']:+.1f} units")
    print(f"Real results scatter around projections by {overall['margin_sd']} points (MARGIN_SD is {config.MARGIN_SD}).")
    print("Break-even against the spread at -110 odds is 52.4%.")
    print("QB out (points lost vs. ratings):", qb_effect)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", choices=list(features.FEATURE_GROUPS))
    ap.add_argument("--w", type=float, help="blend weight on the rating projection")
    args = ap.parse_args()
    run(args.groups, args.w)
