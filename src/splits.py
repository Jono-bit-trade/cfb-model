"""Betting splits: share of bets vs. share of money, and the flags built from them.

There is no free source for splits, so they come from data/splits.csv, filled in
from a splits service you use (for example DraftKings splits on VSiN, or Action
Network). Percentages are for the HOME team (spread) and the OVER (total):

    season,week,home,away,bets_home_pct,money_home_pct,bets_over_pct,money_over_pct
    2026,9,Texas,Ole Miss,38,61,70,52

Flags:
* Sharp money: the money share is at least SHARP_GAP points higher than the bet
  share on one side (fewer, bigger bets, which books tend to respect).
* Reverse line movement (RLM): at least PUBLIC_MIN% of bets are on one side, yet
  the spread moved RLM_MIN_MOVE+ points the other way since opening.
"""
import os

import numpy as np
import pandas as pd

import config
from common import DATA

FILE = os.path.join(DATA, "splits.csv")
COLS = ["season", "week", "home", "away", "bets_home_pct", "money_home_pct", "bets_over_pct", "money_over_pct"]


def load():
    if not os.path.exists(FILE):
        return pd.DataFrame(columns=COLS)
    s = pd.read_csv(FILE, comment="#")
    return s.drop_duplicates(["season", "week", "home"], keep="last") if len(s) else s


def attach(rows, splits=None):
    """Adds split columns and flags to a table of games (needs season, week, home, line_move)."""
    splits = load() if splits is None else splits
    rows = rows.copy()
    for c in COLS[4:]:
        rows[c] = np.nan
    if len(splits):
        s = splits.set_index(["season", "week", "home"])
        key = list(zip(rows["season"], rows["week"], rows["home"]))
        for c in COLS[4:]:
            rows[c] = [s[c].get(k, np.nan) if k in s.index else np.nan for k in key]
    gap = rows["money_home_pct"] - rows["bets_home_pct"]
    rows["sharp_side"] = np.where(gap >= config.SHARP_GAP, "home", np.where(gap <= -config.SHARP_GAP, "away", ""))
    tgap = rows["money_over_pct"] - rows["bets_over_pct"]
    rows["sharp_total"] = np.where(tgap >= config.SHARP_GAP, "Over", np.where(tgap <= -config.SHARP_GAP, "Under", ""))
    public = np.where(rows["bets_home_pct"] >= config.PUBLIC_MIN, "home",
                      np.where(rows["bets_home_pct"] <= 100 - config.PUBLIC_MIN, "away", ""))
    mv = rows.get("line_move", pd.Series(np.nan, index=rows.index))
    rlm_home = (public == "away") & (mv >= config.RLM_MIN_MOVE)   # public on away, line moved toward home
    rlm_away = (public == "home") & (mv <= -config.RLM_MIN_MOVE)
    rows["rlm_side"] = np.where(rlm_home, "home", np.where(rlm_away, "away", ""))
    rows["public_side"] = public
    return rows
