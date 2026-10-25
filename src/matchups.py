"""Matchup pages: one JSON file per upcoming game with the deeper numbers behind it.

docs/data/matchups/{game_id}.json holds, for both teams:
  * season stats with national ranks (scoring, efficiency, success rate, explosiveness,
    finishing drives, havoc, turnovers, pace, run/pass splits)
  * records straight up, against the spread and over/under
  * the last five games with score, line and whether they covered
and every earlier meeting between the two teams (2013 on) with the line and result.
"""
import glob
import os

import numpy as np
import pandas as pd

from common import SITE_DATA, load, write_json

# (key, label, source, column, side, higher_is_better, format)
#   side "off" = the team's own number, "def" = what opponents did against it
STATS = [
    ("ppg", "Points per game", "score", "pts", "off", True, "1"),
    ("papg", "Points allowed per game", "score", "opp_pts", "off", False, "1"),
    ("epa_off", "Offense: points added per play", "clean", "off_ppa", "off", True, "3"),
    ("epa_def", "Defense: points added per play allowed", "clean", "def_ppa", "off", False, "3"),
    ("succ_off", "Offense: success rate", "clean", "off_success", "off", True, "pct"),
    ("succ_def", "Defense: success rate allowed", "clean", "def_success", "off", False, "pct"),
    ("expl_off", "Offense: explosiveness", "clean", "off_explosive", "off", True, "2"),
    ("expl_def", "Defense: explosiveness allowed", "clean", "def_explosive", "off", False, "2"),
    ("pass_off", "Passing: points added per pass", "clean", "off_pass_ppa", "off", True, "3"),
    ("rush_off", "Rushing: points added per rush", "clean", "off_rush_ppa", "off", True, "3"),
    ("pass_def", "Pass defense: per pass allowed", "clean", "def_pass_ppa", "off", False, "3"),
    ("rush_def", "Run defense: per rush allowed", "clean", "def_rush_ppa", "off", False, "3"),
    ("line_off", "Offensive line yards", "clean", "off_line_yards", "off", True, "2"),
    ("stuff_def", "Defense: stuff rate", "clean", "def_stuff", "off", True, "pct"),
    ("ppo_off", "Points per scoring chance", "adv", "off_ppo", "off", True, "2"),
    ("ppo_def", "Points per scoring chance allowed", "adv", "def_ppo", "off", False, "2"),
    ("havoc_def", "Havoc created", "adv", "def_havoc", "off", True, "pct"),
    ("havoc_off", "Havoc allowed", "adv", "off_havoc", "off", False, "pct"),
    ("ppd_off", "Points per drive", "drives", "ppd", "off", True, "2"),
    ("ppd_def", "Points per drive allowed", "drives", "ppd", "def", False, "2"),
    ("field_off", "Average starting field position (yards to go)", "drives", "start_ytg", "off", False, "1"),
    ("to_margin", "Turnover margin per game", "box", "to_margin", "off", True, "2"),
    ("pen", "Penalty yards per game", "box", "pen_yards", "off", False, "1"),
    ("plays", "Plays per game (pace)", "adv", "off_plays", "off", None, "1"),
]


def _team_games(games, season):
    g = games[(games["season"] == season) & games["completed"].astype(bool) & games["home_pts"].notna()]
    home = pd.DataFrame({"game_id": g["game_id"], "start": g["start"], "team": g["home"], "opp": g["away"],
                         "pts": g["home_pts"], "opp_pts": g["away_pts"], "site": np.where(g["neutral"], "N", "H"),
                         "fbs": g["home_class"].eq("fbs")})
    away = pd.DataFrame({"game_id": g["game_id"], "start": g["start"], "team": g["away"], "opp": g["home"],
                         "pts": g["away_pts"], "opp_pts": g["home_pts"], "site": np.where(g["neutral"], "N", "A"),
                         "fbs": g["away_class"].eq("fbs")})
    return pd.concat([home, away], ignore_index=True)


def season_stats(games, season):
    """Per-team season averages and national ranks for every stat in STATS (FBS teams only)."""
    tg = _team_games(games, season)
    fbs = set(tg.loc[tg["fbs"], "team"])
    frames = {"score": tg[["game_id", "team", "pts", "opp_pts"]]}
    for name in ("adv_clean", "adv", "drives", "teambox"):
        d = load(name)
        d = d[d["season"] == season] if len(d) and "season" in d else d
        frames[{"adv_clean": "clean", "teambox": "box"}.get(name, name)] = d
    box = frames["box"]
    if len(box):
        opp = box[["game_id", "team", "turnovers"]].rename(columns={"team": "opp_team", "turnovers": "takeaways"})
        b = box.merge(opp, on="game_id")
        b = b[b["team"] != b["opp_team"]]
        b["to_margin"] = b["takeaways"] - b["turnovers"]
        frames["box"] = b
    drv = frames["drives"]
    if len(drv):
        o = drv[["game_id", "team", "ppd"]].rename(columns={"team": "opp_team", "ppd": "opp_ppd"})
        d2 = drv.merge(o, on="game_id")
        frames["drives_def"] = d2[d2["team"] != d2["opp_team"]]

    table = pd.DataFrame(index=sorted(fbs))
    for key, _, src, col, side, _, _ in STATS:
        f = frames["drives_def"] if (src == "drives" and side == "def") else frames.get(src)
        c = "opp_ppd" if (src == "drives" and side == "def") else col
        if f is None or not len(f) or c not in f:
            continue
        table[key] = f.groupby("team")[c].mean().reindex(table.index)
    ranks = pd.DataFrame(index=table.index)
    for key, _, _, _, _, better, _ in STATS:
        if key in table and better is not None:
            ranks[key] = table[key].rank(ascending=not better, method="min")
    return table, ranks, len(table)


def _lines():
    lines = load("lines")
    return lines.set_index("game_id")[["spread", "total"]] if len(lines) else pd.DataFrame(columns=["spread", "total"])


def _game_row(g, team, lines):
    """One past game from `team`'s side: score, line, and whether it covered / went over."""
    home = g["home"] == team
    pts, opp_pts = (g["home_pts"], g["away_pts"]) if home else (g["away_pts"], g["home_pts"])
    ln = lines.loc[g["game_id"]] if g["game_id"] in lines.index else None
    spread = None if ln is None or pd.isna(ln["spread"]) else float(ln["spread"] if home else -ln["spread"])
    total = None if ln is None or pd.isna(ln["total"]) else float(ln["total"])
    margin = pts - opp_pts
    cover = None if spread is None else ("W" if margin + spread > 0 else "L" if margin + spread < 0 else "P")
    ou = None if total is None else ("O" if pts + opp_pts > total else "U" if pts + opp_pts < total else "P")
    return {"date": pd.Timestamp(g["start"]).isoformat(), "season": int(g["season"]),
            "opp": g["away"] if home else g["home"], "site": "N" if g["neutral"] else ("H" if home else "A"),
            "pts": int(pts), "opp_pts": int(opp_pts), "result": "W" if margin > 0 else "L" if margin < 0 else "T",
            "spread": spread, "cover": cover, "total": total, "ou": ou}


def _records(rows):
    def wl(key, w, l_):
        a = sum(1 for r in rows if r[key] == w)
        b = sum(1 for r in rows if r[key] == l_)
        p = sum(1 for r in rows if r[key] in ("P", "T"))
        return f"{a}–{b}" + (f"–{p}" if p else "")
    return {"su": wl("result", "W", "L"), "ats": wl("cover", "W", "L"), "ou": wl("ou", "O", "U")}


def _fmt(v, f):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), 4 if f == "pct" else int(f))


def write_matchups(out_games, games, season, ranks_ap=None, ratings=None):
    """Write docs/data/matchups/{id}.json for each game on the board; remove stale files."""
    outdir = os.path.join(SITE_DATA, "matchups")
    os.makedirs(outdir, exist_ok=True)
    table, ranks, n_teams = season_stats(games, season)
    lines = _lines()
    done = games[games["completed"].astype(bool) & games["home_pts"].notna()].sort_values("start")
    this = done[done["season"] == season]
    rating_by = {t["team"]: t for t in (ratings or [])}
    keep = set()
    for og in out_games:
        gid = og["id"]
        teams = {}
        for side in ("away", "home"):
            t = og[side]["team"]
            played = this[(this["home"] == t) | (this["away"] == t)]
            rows = [_game_row(g, t, lines) for _, g in played.iterrows()]
            stats = []
            for key, label, _, _, _, better, f in STATS:
                if key not in table:
                    continue
                v = table.at[t, key] if t in table.index else np.nan
                rk = ranks.at[t, key] if (t in ranks.index and key in ranks) else np.nan
                stats.append({"key": key, "label": label, "value": _fmt(v, f), "fmt": f,
                              "rank": None if pd.isna(rk) else int(rk), "better": better})
            rt = rating_by.get(t)
            teams[side] = {"team": t, "rank_ap": (ranks_ap or {}).get(t), "records": _records(rows),
                           "recent": rows[-5:][::-1], "stats": stats,
                           "rating": {k: rt.get(k) for k in ("rank", "overall", "offense", "defense")} if rt else None}
        a, h = og["away"]["team"], og["home"]["team"]
        meet = done[((done["home"] == a) & (done["away"] == h)) | ((done["home"] == h) & (done["away"] == a))]
        h2h = [_game_row(g, h, lines) for _, g in meet.iterrows()][::-1]
        write_json({"id": gid, "season": season, "n_teams": n_teams, "teams": teams, "h2h": h2h,
                    "h2h_record": _records(h2h)}, outdir, f"{gid}.json")
        keep.add(f"{gid}.json")
    for f in glob.glob(os.path.join(outdir, "*.json")):
        if os.path.basename(f) not in keep:
            os.remove(f)
    print(f"wrote {len(keep)} matchup pages")
