"""Step 4: is the starting quarterback out?

History (training data) comes from box scores: before each game, the starter is
the passer with the most attempts over the team's previous LOOKBACK games that
season. If he throws no passes in the game, the game is marked qb_out = 1.

The current week comes from data/qb_status.csv, which you fill in from the
conference availability reports:

    season,week,team,qb,status,note
    2026,9,Texas,Arch Manning,questionable,ankle

status: available | probable | questionable | doubtful | out
out/doubtful count as out. questionable is projected both ways on the site.
"""
import os

import pandas as pd

from common import DATA

LOOKBACK = 3
STATUS_FILE = os.path.join(DATA, "qb_status.csv")
OUT_STATUSES = {"out", "doubtful"}


def history_flags(passers, games):
    """game_id, team, starter, qb_out for every completed game with box-score data."""
    if passers.empty:
        return pd.DataFrame(columns=["game_id", "team", "starter", "qb_out"])
    att = passers.merge(games[["game_id", "start"]], on="game_id", how="inner").sort_values("start")
    flags = []
    for (season, team), df in att.groupby(["season", "team"], sort=False):
        history = []
        for gid, gm in df.groupby("game_id", sort=False):
            recent = {}
            for game_att in history[-LOOKBACK:]:
                for p, a in game_att.items():
                    recent[p] = recent.get(p, 0) + a
            starter = max(recent, key=recent.get) if recent else None
            played = set(gm.loc[gm["att"] > 0, "player"])
            flags.append({"game_id": gid, "team": team, "starter": starter,
                          "qb_out": int(starter is not None and starter not in played)})
            history.append(dict(zip(gm["player"], gm["att"])))
    return pd.DataFrame(flags)


def current_starters(passers, games, season):
    """Each team's expected starter right now: most attempts over its last LOOKBACK games."""
    if passers.empty:
        return {}
    att = passers[passers["season"] == season].merge(games[["game_id", "start"]], on="game_id")
    out = {}
    for team, df in att.sort_values("start").groupby("team"):
        last_games = df["game_id"].drop_duplicates().tail(LOOKBACK)
        recent = df[df["game_id"].isin(last_games)].groupby("player")["att"].sum()
        if len(recent):
            out[team] = recent.idxmax()
    return out


def load_status():
    if not os.path.exists(STATUS_FILE):
        return pd.DataFrame(columns=["season", "week", "team", "qb", "status", "note"])
    s = pd.read_csv(STATUS_FILE, comment="#")
    s["status"] = s["status"].astype(str).str.strip().str.lower()
    return s


def status_for(season, week, team, status_table):
    """(status, qb, note) for a team's upcoming game, or ('unknown', None, None)."""
    if status_table.empty:
        return "unknown", None, None
    m = status_table[(status_table["season"] == season) & (status_table["week"] == week)
                     & (status_table["team"] == team)]
    if m.empty:
        return "unknown", None, None
    row = m.iloc[-1]
    note = row.get("note")
    return row["status"], row.get("qb"), (None if pd.isna(note) else note)
