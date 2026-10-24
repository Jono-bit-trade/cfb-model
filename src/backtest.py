"""Step 7: walk-forward backtest. The only judge of whether a change helps.

For each test season, the ML model trains on earlier seasons only, and every
week's ratings use only games played before that week.

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
from common import DATA, SITE_DATA, load_games, write_json  # noqa: E402
from model import Model, ats_edge, grade, grade_total  # noqa: E402
from ratings import RatingEngine  # noqa: E402


def pct(x):
    x = pd.Series(x).dropna()
    x = x[x != 0.5]
    return float(x.mean()) if len(x) else float("nan")


def summarize(df):
    has_line = df["spread"].notna()
    d = df[has_line]
    out = {
        "games": int(len(df)),
        "games_with_line": int(has_line.sum()),
        "mae_model": float((d["actual_margin"] - d["pred_margin"]).abs().mean()),
        "mae_ratings_only": float((d["actual_margin"] - d["rating_margin"]).abs().mean()),
        "mae_market": float((d["actual_margin"] + d["spread"]).abs().mean()),
        "ats_pct": pct(d["ats"]),
        "ats_ratings_only_pct": pct(d["ats_ratings"]),
        "su_pct": float((np.sign(df["pred_margin"]) == np.sign(df["actual_margin"])).mean()),
        "ou_pct": pct(d["ou"]),
        "ats_by_edge": {},
    }
    for lo, hi, label in [(0, 3, "0-3"), (3, 7, "3-7"), (7, 99, "7+")]:
        e = d[(d["edge"].abs() >= lo) & (d["edge"].abs() < hi)]
        out["ats_by_edge"][label] = {"games": int(len(e)), "ats_pct": pct(e["ats"])}
    return out


def run(groups=None, w=None, save=True):
    t0 = time.time()
    games = load_games()
    engine = RatingEngine(games)
    seasons = sorted(s for s in games["season"].unique() if s > config.FIRST_SEASON)
    table = features.build(games, engine, seasons=seasons)
    table = table[table["completed"].astype(bool) & table["actual_margin"].notna()]
    print(f"feature table: {len(table)} games in {time.time() - t0:.0f}s")

    results = []
    for s in config.BACKTEST_SEASONS:
        train, test = table[table["season"] < s], table[table["season"] == s].copy()
        if len(test) == 0 or train["season"].nunique() < 2:
            continue
        m = Model(groups, w).fit(train)
        test["pred_margin"], test["pred_total"] = m.predict(test)
        results.append(test)
    if not results:
        raise SystemExit("Not enough seasons to backtest yet.")
    bt = pd.concat(results)
    bt["edge"] = ats_edge(bt["pred_margin"], bt["spread"])
    bt["ats"] = [grade(a, s, e) for a, s, e in zip(bt["actual_margin"], bt["spread"], bt["edge"])]
    bt["ats_ratings"] = [grade(a, s, e) for a, s, e in zip(bt["actual_margin"], bt["spread"],
                                                          ats_edge(bt["rating_margin"], bt["spread"]))]
    bt["ou"] = [grade_total(a, t, p) for a, t, p in zip(bt["actual_total"], bt["total"], bt["pred_total"])]

    overall = summarize(bt)
    by_season = {int(s): summarize(df) for s, df in bt.groupby("season")}

    # How many points is a missing starting QB worth in this data?
    resid = bt["actual_margin"] - bt["rating_margin"]
    qb_effect = {
        "home_qb_out_games": int(bt["qb_out_home"].sum()),
        "away_qb_out_games": int(bt["qb_out_away"].sum()),
        "home_qb_out_residual": float(resid[bt["qb_out_home"] == 1].mean()) if bt["qb_out_home"].sum() else None,
        "away_qb_out_residual": float(resid[bt["qb_out_away"] == 1].mean()) if bt["qb_out_away"].sum() else None,
    }

    print_report(overall, by_season, qb_effect, groups or config.FEATURE_GROUPS, config.BLEND_W if w is None else w)
    if save:
        write_json({"generated": pd.Timestamp.now(tz="UTC"), "groups": groups or config.FEATURE_GROUPS,
                    "blend_w": config.BLEND_W if w is None else w, "overall": overall,
                    "by_season": by_season, "qb_effect": qb_effect}, SITE_DATA, "backtest.json")
        bt.to_csv(os.path.join(DATA, "backtest_games.csv"), index=False)
    return overall, by_season


def print_report(overall, by_season, qb_effect, groups, w):
    f = lambda x: "  -  " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:5.1%}"
    print(f"\nGroups: {' + '.join(groups)}   blend w = {w}")
    print(f"{'season':>7} {'games':>6} {'MAE':>6} {'MAE mkt':>8} {'ATS':>6} {'ATS rat':>8} {'SU':>6} {'O/U':>6}")
    for s, r in by_season.items():
        print(f"{s:>7} {r['games_with_line']:>6} {r['mae_model']:6.2f} {r['mae_market']:8.2f} "
              f"{f(r['ats_pct']):>6} {f(r['ats_ratings_only_pct']):>8} {f(r['su_pct']):>6} {f(r['ou_pct']):>6}")
    r = overall
    print(f"{'ALL':>7} {r['games_with_line']:>6} {r['mae_model']:6.2f} {r['mae_market']:8.2f} "
          f"{f(r['ats_pct']):>6} {f(r['ats_ratings_only_pct']):>8} {f(r['su_pct']):>6} {f(r['ou_pct']):>6}")
    print("ATS by edge size:", ", ".join(f"{k}: {f(v['ats_pct'])} ({v['games']} g)" for k, v in r["ats_by_edge"].items()))
    print("Break-even at -110 odds is 52.4%.")
    print("QB out:", qb_effect)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", choices=list(features.FEATURE_GROUPS))
    ap.add_argument("--w", type=float, help="blend weight on the rating projection")
    args = ap.parse_args()
    run(args.groups, args.w)
