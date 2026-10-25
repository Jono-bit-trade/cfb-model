"""Grade projections against what happened, the same way for every page.

Four ways a game is graded:
  * Winner (straight up): did the model's projected winner win?
  * Moneyline: betting the projected winner at its moneyline odds, in units won or lost.
    Also "value" moneylines: only bet a side when the model's win chance beats the
    market's (vig-free) chance by at least VALUE_MIN_PROB.
  * Spread (ATS): the side the model likes against the spread, at standard -110 odds.
  * Over/under: the model's projected total against the market total, at -110.

One unit = the amount risked on each bet.
"""
import math

import numpy as np
import pandas as pd

import config

WIN_AT_110 = 100 / 110  # profit per unit risked on a -110 bet


def win_prob(margin, sd=None):
    """Chance the home team wins, from the projected margin (normal model)."""
    sd = sd or config.MARGIN_SD
    m = np.asarray(margin, dtype=float)
    return 0.5 * (1 + np.vectorize(math.erf, otypes=[float])(m / (sd * math.sqrt(2))))


def implied(ml):
    """Market win chance implied by American odds (includes the book's cut)."""
    ml = np.asarray(ml, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


def ml_profit(ml, won):
    """Units won betting 1 unit at American odds `ml`. Loss = -1."""
    ml = np.asarray(ml, dtype=float)
    won = np.asarray(won, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pay = np.where(ml > 0, ml / 100, 100 / np.abs(ml))
    out = np.where(won == 1, pay, -1.0)
    return np.where(np.isnan(ml) | np.isnan(won), np.nan, out)


def _result(score, pick_positive):
    """1 win, 0 loss, 0.5 push, nan when ungradable. score > 0 means 'positive side' won."""
    out = np.where(score == 0, 0.5, np.where((score > 0) == pick_positive, 1.0, 0.0))
    return np.where(np.isnan(score), np.nan, out)


def grade(df):
    """Adds pick and result columns. Needs pred_margin, pred_total, spread, total,
    home_ml, away_ml, and (for finished games) actual_margin, actual_total."""
    d = df.copy()
    for c in ("spread", "total", "home_ml", "away_ml", "actual_margin", "actual_total"):
        if c not in d:
            d[c] = np.nan
    d[["spread", "total", "home_ml", "away_ml"]] = d[["spread", "total", "home_ml", "away_ml"]].astype(float)
    home, away = d["home"].to_numpy(), d["away"].to_numpy()
    actual_m, actual_t = d["actual_margin"].astype(float).to_numpy(), d["actual_total"].astype(float).to_numpy()

    # Winner
    d["su_home"] = d["pred_margin"] > 0
    d["su_pick"] = np.where(d["su_home"], home, away)
    d["su"] = np.where(np.isnan(actual_m) | (actual_m == 0), np.nan, ((actual_m > 0) == d["su_home"]).astype(float))
    d["p_home"] = win_prob(d["pred_margin"])
    d["p_pick"] = np.where(d["su_home"], d["p_home"], 1 - d["p_home"])

    # Moneyline: bet the projected winner at its price
    d["ml_odds"] = np.where(d["su_home"], d["home_ml"], d["away_ml"])
    d["ml_units"] = ml_profit(d["ml_odds"], d["su"])

    # Value moneyline: bet a side only when the model's chance beats the market's
    ih, ia = implied(d["home_ml"]), implied(d["away_ml"])
    fair_home = ih / (ih + ia)
    gap = d["p_home"] - fair_home
    d["val_side"] = np.where(gap >= config.VALUE_MIN_PROB, "home", np.where(gap <= -config.VALUE_MIN_PROB, "away", ""))
    d.loc[np.isnan(fair_home), "val_side"] = ""
    d["val_pick"] = np.where(d["val_side"] == "home", home, np.where(d["val_side"] == "away", away, None))
    d["val_odds"] = np.where(d["val_side"] == "home", d["home_ml"], np.where(d["val_side"] == "away", d["away_ml"], np.nan))
    d["val_gap"] = np.abs(gap)
    val_won = np.where(d["val_side"] == "home", (actual_m > 0).astype(float), (actual_m < 0).astype(float))
    val_won = np.where(np.isnan(actual_m) | (actual_m == 0) | (d["val_side"] == ""), np.nan, val_won)
    d["val_units"] = ml_profit(d["val_odds"], val_won)

    # Spread
    d["edge"] = d["pred_margin"] + d["spread"]          # > 0: model likes home against the spread
    d["ats_home"] = d["edge"] > 0
    d["ats_pick"] = np.where(d["spread"].isna() | (d["edge"] == 0), None, np.where(d["ats_home"], home, away))
    d["ats_line"] = np.where(d["ats_home"], d["spread"], -d["spread"])
    d["ats"] = _result(actual_m + d["spread"].to_numpy(), d["ats_home"].to_numpy())
    d.loc[d["edge"] == 0, "ats"] = np.nan

    # Over/under
    d["total_edge"] = d["pred_total"] - d["total"]
    d["ou_over"] = d["total_edge"] > 0
    d["ou_pick"] = np.where(d["total"].isna() | (d["total_edge"] == 0), None, np.where(d["ou_over"], "Over", "Under"))
    d["ou"] = _result(actual_t - d["total"].to_numpy(), d["ou_over"].to_numpy())
    d.loc[d["total_edge"] == 0, "ou"] = np.nan
    return d


def units_110(results):
    r = pd.Series(results).dropna()
    return float((r == 1).sum() * WIN_AT_110 - (r == 0).sum())


def record(results):
    r = pd.Series(results).dropna()
    w, l, p = int((r == 1).sum()), int((r == 0).sum()), int((r == 0.5).sum())
    return {"w": w, "l": l, "p": p, "pct": round(w / (w + l), 4) if w + l else None}


def summary(g, min_edge=0.0, min_total_edge=0.0):
    """Records and units for a graded table, optionally only bigger disagreements."""
    ats = g[g["edge"].abs() >= min_edge]
    ou = g[g["total_edge"].abs() >= min_total_edge]
    ml = g[g["ml_units"].notna()]
    val = g[g["val_units"].notna()]
    return {
        "winner": record(g["su"]),
        "moneyline": {**record(ml["su"]), "units": round(float(ml["ml_units"].sum()), 2), "bets": int(len(ml))},
        "value_ml": {**record(np.where(val["val_units"] > 0, 1.0, 0.0)), "units": round(float(val["val_units"].sum()), 2),
                     "bets": int(len(val))},
        "spread": {**record(ats["ats"]), "units": round(units_110(ats["ats"]), 2)},
        "total": {**record(ou["ou"]), "units": round(units_110(ou["ou"]), 2)},
    }


def our_picks(g, lo=None, hi=None):
    """'Our picks': spread picks with an edge in the sweet spot (3-7 points by default)."""
    import config
    lo = config.STRONG_EDGE if lo is None else lo
    hi = config.SWEET_SPOT_MAX if hi is None else hi
    e = g["edge"].abs()
    p = g[(e >= lo) & (e <= hi)]
    return {**record(p["ats"]), "units": round(units_110(p["ats"]), 2), "picks": int(p["ats"].notna().sum())}


def edge_table(g, col="edge", result="ats", cuts=(0, 2, 3, 4, 5, 6, 7, 8, 10, 12)):
    """Record at or above each edge size: shows where (if anywhere) the model clears 60%."""
    rows = []
    for c in cuts:
        r = record(g.loc[g[col].abs() >= c, result])
        rows.append({"min_edge": c, **r, "games": r["w"] + r["l"] + r["p"]})
    return rows


def history_rows(g):
    """Compact per-game rows for the website's History page."""
    out = []
    for r in g.sort_values("start").itertuples():
        def n(v, d=1):
            return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), d)
        out.append({
            "id": int(r.game_id), "wk": int(r.week) if not pd.isna(r.week) else None,
            "post": getattr(r, "season_type", "regular") == "postseason",
            "date": pd.Timestamp(r.start).isoformat(), "h": r.home, "a": r.away,
            "hs": n(r.home_pts, 0), "as": n(r.away_pts, 0),
            "pm": n(r.pred_margin), "pt": n(r.pred_total),
            "sp": n(r.spread), "tot": n(r.total), "hml": n(r.home_ml, 0), "aml": n(r.away_ml, 0),
            "pw": n(r.p_pick, 3), "su": n(r.su), "ml_odds": n(r.ml_odds, 0), "ml_u": n(r.ml_units, 2),
            "val": r.val_pick, "val_odds": n(r.val_odds, 0), "val_u": n(r.val_units, 2), "val_gap": n(r.val_gap, 3),
            "ats_pick": r.ats_pick, "ats_line": n(r.ats_line), "edge": n(abs(r.edge)) if not pd.isna(r.edge) else None,
            "ats": n(r.ats), "ou_pick": r.ou_pick, "te": n(abs(r.total_edge)) if not pd.isna(r.total_edge) else None,
            "ou": n(r.ou), "bf": bool(getattr(r, "backfill", False)),
            "sp_open": n(getattr(r, "spread_open", None)), "tot_open": n(getattr(r, "total_open", None)),
            "bh": n(getattr(r, "bets_home_pct", None), 0), "mh": n(getattr(r, "money_home_pct", None), 0),
            "bo": n(getattr(r, "bets_over_pct", None), 0), "mo": n(getattr(r, "money_over_pct", None), 0),
        })
    return out
