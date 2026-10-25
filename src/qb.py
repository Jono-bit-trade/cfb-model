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

import numpy as np
import pandas as pd

from common import DATA

LOOKBACK = 3
STATUS_FILE = os.path.join(DATA, "qb_status.csv")
CONFIRM_DIR = os.path.join(DATA, "qb_confirm")  # one small file per confirmation, made from the website's links
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
    """Your QB statuses: data/qb_status.csv plus every confirmation file in data/qb_confirm/.
    Later rows win; confirmation files are named season-week-team-timestamp and read in that order."""
    cols = ["season", "week", "team", "qb", "status", "note"]
    parts = []
    if os.path.exists(STATUS_FILE):
        parts.append(pd.read_csv(STATUS_FILE, comment="#"))
    if os.path.isdir(CONFIRM_DIR):
        for f in sorted(os.listdir(CONFIRM_DIR)):  # names end in a timestamp, so later confirmations win
            if f.endswith(".csv"):
                try:
                    parts.append(pd.read_csv(os.path.join(CONFIRM_DIR, f), comment="#"))
                except Exception as e:
                    print(f"  ! couldn't read {f}: {e}")
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=cols)
    s = pd.concat(parts, ignore_index=True)
    s["status"] = s["status"].astype(str).str.strip().str.lower()
    s["team"] = s["team"].astype(str).str.strip()
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


def quality_table(passers, qb_ppa, games, lookback=LOOKBACK):
    """Before every game (played or upcoming), each team's expected starter and backup
    and how efficient each has been (expected points added per pass, attempt-weighted),
    using only earlier games. Carries over from last season for returning players.

    Returns game_id, team, qb_starter_ppa, qb_backup_ppa.
    """
    cols = ["game_id", "team", "qb_starter_ppa", "qb_backup_ppa"]
    if passers.empty or qb_ppa.empty:
        return pd.DataFrame(columns=cols)
    g = games[["game_id", "season", "week", "season_type", "start", "home", "away"]]
    att = passers.merge(g[["game_id", "week", "season_type", "start"]], on="game_id", how="inner")
    q = qb_ppa.dropna(subset=["ppa_pass"]).drop_duplicates(["season", "week", "season_type", "team", "player"])
    att = att.merge(q[["season", "week", "season_type", "team", "player", "ppa_pass"]],
                    on=["season", "week", "season_type", "team", "player"], how="left")
    by_game = {k: v for k, v in att.groupby(["game_id", "team"])}

    timeline = pd.concat([g.rename(columns={"home": "team"})[["game_id", "start", "team"]],
                          g.rename(columns={"away": "team"})[["game_id", "start", "team"]]]).sort_values("start")
    out = []
    for team, tl in timeline.groupby("team", sort=False):
        recent_games = []          # [{player: attempts}] for the last few games
        totals = {}                # player -> [sum(att * ppa), sum(att)] over every earlier game
        for gid in tl["game_id"]:
            recent = {}
            for game_att in recent_games[-lookback:]:
                for p, a in game_att.items():
                    recent[p] = recent.get(p, 0) + a
            ranked = sorted(recent, key=recent.get, reverse=True)
            rating = lambda p: (totals[p][0] / totals[p][1]) if p in totals and totals[p][1] >= 20 else np.nan
            out.append({"game_id": gid, "team": team,
                        "qb_starter_ppa": rating(ranked[0]) if ranked else np.nan,
                        "qb_backup_ppa": rating(ranked[1]) if len(ranked) > 1 else np.nan})
            gm = by_game.get((gid, team))
            if gm is not None:
                recent_games.append(dict(zip(gm["player"], gm["att"])))
                for p, a, v in zip(gm["player"], gm["att"], gm["ppa_pass"]):
                    if a > 0 and not pd.isna(v):
                        s = totals.setdefault(p, [0.0, 0.0])
                        s[0] += a * v
                        s[1] += a
    return pd.DataFrame(out, columns=cols)
