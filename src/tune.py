"""Find the model settings and pick threshold that work best, without fooling ourselves.

1. Try every combination of rating settings, feature groups and blend weight,
   scoring each on the TUNING seasons only (all backtest seasons but the last 3).
2. On the tuning seasons, find the smallest edge where the spread record is
   60%+ over a meaningful number of games.
3. Check the winning settings and that edge on the HOLDOUT seasons (the last 3),
   which played no part in choosing them. That holdout number is the honest one.

The winning settings are saved to data/tuned.json. config.py loads them, so the
weekly predictions of upcoming games use exactly the tuned model.

    python src/tune.py          # takes a while: it rebuilds ratings for each setting
"""
import itertools
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import grading  # noqa: E402
from backtest import build_table  # noqa: E402
from common import DATA, SITE_DATA, load_games, write_json  # noqa: E402
from model import Model  # noqa: E402

ALPHAS = [6.0, 12.0, 20.0]
HALF_LIVES = [45, 70, 120]
GROUP_SETS = [["CORE"], ["CORE", "STYLE"], ["CORE", "STYLE", "CONTEXT", "SITUATION"]]
BLEND_WS = [0.3, 0.5, 0.7, 1.0]
EDGE_CUTS = [0, 2, 3, 4, 5, 6, 7, 8, 10, 12]
MIN_GAMES_FOR_THRESHOLD = 150   # don't trust a 60% that rests on a handful of games
TARGET = 0.60


def walk_forward_ml(table, seasons, groups):
    """ML predictions for each season, trained only on earlier seasons."""
    out = []
    for s in seasons:
        train, test = table[table["season"] < s], table[table["season"] == s].copy()
        if len(test) == 0 or train["season"].nunique() < 2:
            continue
        m = Model(groups, w=0.0).fit(train)       # w=0 -> pure ML output
        test["ml_margin"], test["ml_total"] = m.predict(test)
        out.append(test)
    return pd.concat(out) if out else pd.DataFrame()


def blend(df, w):
    d = df.copy()
    d["pred_margin"] = w * d["rating_margin"] + (1 - w) * d["ml_margin"]
    d["pred_total"] = w * d["rating_total"] + (1 - w) * d["ml_total"]
    return grading.grade(d)


def score(g):
    """Lower is better: average miss on the margin (stable), tie-break on spread record."""
    mae = float((g["actual_margin"] - g["pred_margin"]).abs().mean())
    ats = grading.record(g["ats"])["pct"] or 0.5
    return mae, ats


def threshold(g):
    """Smallest edge with a 60%+ spread record on enough games; else the best-performing edge."""
    ladder = grading.edge_table(g, "edge", "ats", EDGE_CUTS)
    ok = [r for r in ladder if r["games"] >= MIN_GAMES_FOR_THRESHOLD and (r["pct"] or 0) >= TARGET]
    if ok:
        return ok[0]["min_edge"], ladder, True
    # No edge size reached the target: keep the default "strong pick" threshold rather than
    # relabeling every pick, and say so in the report.
    return 3.0, ladder, False


def main():
    t0 = time.time()
    games = load_games()
    seasons = config.BACKTEST_SEASONS
    if len(seasons) < 5:
        raise SystemExit("Need at least 5 backtest seasons to tune and still hold some out.")
    tune_seasons, holdout = seasons[:-3], seasons[-3:]
    print(f"Tuning on {tune_seasons[0]}-{tune_seasons[-1]}, holding out {holdout[0]}-{holdout[-1]}")

    results = []
    for alpha, half_life in itertools.product(ALPHAS, HALF_LIVES):
        config.RATING_ALPHA, config.HALF_LIFE_DAYS = alpha, half_life
        table = build_table(games)
        for groups in GROUP_SETS:
            ml = walk_forward_ml(table, seasons, groups)
            for w in BLEND_WS:
                g = blend(ml, w)
                tune_g = g[g["season"].isin(tune_seasons)]
                mae, ats = score(tune_g)
                results.append({"alpha": alpha, "half_life": half_life, "groups": groups, "w": w,
                                "tune_mae": round(mae, 3), "tune_ats": round(ats, 4), "_g": g})
        print(f"  alpha {alpha:>4}, half-life {half_life:>3}: best so far MAE "
              f"{min(r['tune_mae'] for r in results):.3f}  ({time.time() - t0:.0f}s)")

    results.sort(key=lambda r: (r["tune_mae"], -r["tune_ats"]))
    best = results[0]
    g = best["_g"]
    tune_g, hold_g = g[g["season"].isin(tune_seasons)], g[g["season"].isin(holdout)]
    edge, tune_ladder, found = threshold(tune_g)
    hold_ladder = grading.edge_table(hold_g, "edge", "ats", EDGE_CUTS)
    hold_at = grading.record(hold_g.loc[hold_g["edge"].abs() >= edge, "ats"])
    hold_all = grading.summary(hold_g)

    settings = {"RATING_ALPHA": best["alpha"], "HALF_LIFE_DAYS": best["half_life"],
                "FEATURE_GROUPS": best["groups"], "BLEND_W": best["w"], "STRONG_EDGE": float(edge)}
    report = {
        "generated": pd.Timestamp.now(tz="UTC").isoformat(), "settings": settings,
        "tune_seasons": tune_seasons, "holdout_seasons": holdout,
        "found_60_on_tuning": found, "target": TARGET,
        "tuning_spread_by_edge": tune_ladder, "holdout_spread_by_edge": hold_ladder,
        "holdout_at_strong_edge": hold_at, "holdout_summary": hold_all,
        "top_settings": [{k: v for k, v in r.items() if k != "_g"} for r in results[:10]],
    }
    write_json(report, DATA, "tuned.json")
    write_json(report, SITE_DATA, "tuning.json")

    f = lambda r: "  –  " if r.get("pct") is None else f"{r['pct']:.1%}"
    print("\nBest settings (chosen on tuning seasons only):", settings)
    print("Tuning seasons, spread record by edge:  ", "  ".join(f"{r['min_edge']}+: {f(r)} ({r['games']})" for r in tune_ladder))
    print("Holdout seasons, spread record by edge: ", "  ".join(f"{r['min_edge']}+: {f(r)} ({r['games']})" for r in hold_ladder))
    verdict = "reached" if found else "did not reach"
    print(f"\nOn tuning seasons the model {verdict} {TARGET:.0%} against the spread at {edge}+ point edges.")
    print(f"On the holdout seasons it went {hold_at['w']}-{hold_at['l']}-{hold_at['p']} ({f(hold_at)}) at that edge. "
          "That holdout figure is the honest estimate of future results.")
    print(f"Holdout overall: winners {f(hold_all['winner'])}, spread {f(hold_all['spread'])}, "
          f"over/under {f(hold_all['total'])}, moneyline {hold_all['moneyline']['units']:+.1f} units")
    print(f"Saved to data/tuned.json; the weekly predictions now use these settings. ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
