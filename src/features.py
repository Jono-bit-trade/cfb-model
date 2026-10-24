"""Steps 3-5: one row per game with every model input.

Every value uses only information available before that game kicked off:
ratings come from games before the game's week, rolling stats use shift(1),
and qb_out for upcoming games comes from the availability report file.
"""
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import qb
import weather
from common import load, load_file, miles
from ratings import RatingEngine, project, team_keys

CENTRAL = ZoneInfo("America/Chicago")

FEATURE_GROUPS = {
    "CORE": ["rating_margin", "rating_total", "epa_matchup", "neutral", "conf_game",
             "wind", "gust", "temp", "rain", "dome", "qb_out_home", "qb_out_away"],
    "STYLE": ["success_matchup", "explosive_matchup", "ppo_matchup", "havoc_matchup",
              "pace_sum", "to_margin_diff", "pen_diff"],
    "CONTEXT": ["rest_diff", "travel_diff", "elev_diff", "tz_diff", "local_kick_hour",
                "new_coach_home", "new_coach_away"],
    "SITUATION": ["home_off_bye", "away_off_bye", "home_after_loss", "away_after_loss"],
}
TOTAL_FEATURES = ["rating_total", "epa_sum", "success_sum", "explosive_sum", "pace_sum",
                  "wind", "gust", "temp", "rain", "dome", "neutral", "qb_out_home", "qb_out_away"]

ROLL_COLS = ["off_ppa", "def_ppa", "off_success", "def_success", "off_explosive", "def_explosive",
             "off_ppo", "def_ppo", "off_havoc", "def_havoc", "off_plays", "pen_yards", "to_margin"]
ROLL_N = 5


def features_for(groups):
    return [c for gname in groups for c in FEATURE_GROUPS[gname]]


# ------------------------------------------------------------ team timelines
def team_timeline(games):
    """One row per team per game (all seasons), with rolling pre-game stats."""
    cols = ["game_id", "season", "week", "start", "completed", "venue_id", "neutral"]
    home = games[cols + ["home", "away", "home_pts", "away_pts"]].rename(
        columns={"home": "team", "away": "opp", "home_pts": "pts", "away_pts": "opp_pts"})
    home["is_home"] = ~home["neutral"].astype(bool)
    away = games[cols + ["away", "home", "away_pts", "home_pts"]].rename(
        columns={"away": "team", "home": "opp", "away_pts": "pts", "home_pts": "opp_pts"})
    away["is_home"] = False
    tl = pd.concat([home, away], ignore_index=True).sort_values("start")

    ppa, adv, box = load("ppa"), load("adv"), load("teambox")
    if len(ppa):
        tl = tl.merge(ppa[["game_id", "team", "off_ppa", "def_ppa"]].drop_duplicates(["game_id", "team"]),
                      on=["game_id", "team"], how="left")
    if len(adv):
        keep = ["game_id", "team"] + [c for c in adv.columns if c.startswith(("off_", "def_"))]
        tl = tl.merge(adv[keep].drop_duplicates(["game_id", "team"]), on=["game_id", "team"], how="left")
    if len(box):
        b = box.drop_duplicates(["game_id", "team"])
        opp_to = b[["game_id", "team", "turnovers"]].rename(columns={"team": "opp", "turnovers": "opp_turnovers"})
        tl = tl.merge(b[["game_id", "team", "turnovers", "pen_yards"]], on=["game_id", "team"], how="left")
        tl = tl.merge(opp_to, on=["game_id", "opp"], how="left")
        tl["to_margin"] = tl["opp_turnovers"] - tl["turnovers"]
    for c in ROLL_COLS:
        if c not in tl:
            tl[c] = np.nan

    tl = tl.sort_values("start").reset_index(drop=True)
    grp = tl.groupby("team", sort=False)
    # Rolling means over the team's previous games (carrying over from last season early on).
    for c in ROLL_COLS:
        tl[f"r_{c}"] = grp[c].transform(lambda s: s.shift(1).rolling(ROLL_N, min_periods=1).mean())
    tl["prev_start"] = grp["start"].shift(1)
    tl["rest"] = ((tl["start"] - tl["prev_start"]).dt.total_seconds() / 86400).clip(upper=21).fillna(21)
    tl["prev_margin"] = grp["pts"].shift(1) - grp["opp_pts"].shift(1)
    tl.loc[tl["season"] != grp["season"].shift(1), "prev_margin"] = np.nan  # no carry across seasons
    return tl


def home_venues(games):
    """Each team's usual home stadium (most common non-neutral home venue)."""
    h = games[~games["neutral"].astype(bool) & games["venue_id"].notna()]
    return h.groupby("home")["venue_id"].agg(lambda s: s.value_counts().index[0])


def _tz_offset(tzname, when):
    try:
        return when.astimezone(ZoneInfo(tzname)).utcoffset().total_seconds() / 3600
    except Exception:
        return np.nan


# --------------------------------------------------------------- main build
def build(games, engine=None, seasons=None, live_cutoff=None):
    """Feature table for the given seasons.

    Completed and past games use ratings as of the start of their week
    (Tuesday, Central time). Games not yet played use ratings as of
    `live_cutoff` (normally now).
    """
    engine = engine or RatingEngine(games)
    games = team_keys(games)
    target = games[games["season"].isin(seasons)] if seasons is not None else games
    target = target.copy()

    # ---- ratings snapshot per week bucket
    start_ct = target["start"].dt.tz_convert(CENTRAL)
    target["bucket_start"] = start_ct.dt.tz_localize(None).dt.to_period("W-MON").dt.start_time
    target["cutoff"] = target["bucket_start"].dt.tz_localize(CENTRAL).dt.tz_convert("UTC")
    if live_cutoff is not None:
        live = ~target["completed"].astype(bool)
        target.loc[live, "cutoff"] = pd.Timestamp(live_cutoff)

    proj = []
    for (season, cutoff), grp in target.groupby(["season", "cutoff"]):
        r, avg, hfa = engine.as_of(season, cutoff)
        for gm in grp.itertuples():
            hp, ap = project(r, avg, hfa, gm.home_key, gm.away_key, bool(gm.neutral))
            proj.append({"game_id": gm.game_id, "proj_home": hp, "proj_away": ap, "hfa": hfa,
                         "home_overall": r["overall"].get(gm.home_key, np.nan),
                         "away_overall": r["overall"].get(gm.away_key, np.nan)})
    t = target.merge(pd.DataFrame(proj), on="game_id", how="left")
    t["rating_margin"] = t["proj_home"] - t["proj_away"]
    t["rating_total"] = t["proj_home"] + t["proj_away"]

    # ---- rolling stats, rest, previous result
    tl = team_timeline(games)
    rcols = [f"r_{c}" for c in ROLL_COLS] + ["rest", "prev_margin"]
    side = lambda pfx: tl[["game_id", "team"] + rcols].rename(
        columns={c: f"{pfx}_{c}" for c in rcols}).rename(columns={"team": pfx})
    t = t.merge(side("home"), on=["game_id", "home"], how="left")
    t = t.merge(side("away"), on=["game_id", "away"], how="left")

    H, A = "home_r_", "away_r_"
    t["epa_matchup"] = (t[H + "off_ppa"] + t[A + "def_ppa"]) - (t[A + "off_ppa"] + t[H + "def_ppa"])
    t["epa_sum"] = (t[H + "off_ppa"] + t[A + "def_ppa"]) + (t[A + "off_ppa"] + t[H + "def_ppa"])
    t["success_matchup"] = (t[H + "off_success"] + t[A + "def_success"]) - (t[A + "off_success"] + t[H + "def_success"])
    t["success_sum"] = (t[H + "off_success"] + t[A + "def_success"]) + (t[A + "off_success"] + t[H + "def_success"])
    t["explosive_matchup"] = (t[H + "off_explosive"] + t[A + "def_explosive"]) - (t[A + "off_explosive"] + t[H + "def_explosive"])
    t["explosive_sum"] = (t[H + "off_explosive"] + t[A + "def_explosive"]) + (t[A + "off_explosive"] + t[H + "def_explosive"])
    t["ppo_matchup"] = (t[H + "off_ppo"] + t[A + "def_ppo"]) - (t[A + "off_ppo"] + t[H + "def_ppo"])
    # havoc: defense's havoc created + opponent offense's havoc allowed (good for that defense)
    t["havoc_matchup"] = (t[H + "def_havoc"] + t[A + "off_havoc"]) - (t[A + "def_havoc"] + t[H + "off_havoc"])
    t["pace_sum"] = t[H + "off_plays"] + t[A + "off_plays"]
    t["to_margin_diff"] = t[H + "to_margin"] - t[A + "to_margin"]
    t["pen_diff"] = t[H + "pen_yards"] - t[A + "pen_yards"]

    # ---- situations (Step 5)
    t["rest_diff"] = t["home_rest"] - t["away_rest"]
    t["home_off_bye"] = (t["home_rest"] >= 12).astype(float)
    t["away_off_bye"] = (t["away_rest"] >= 12).astype(float)
    t["home_after_loss"] = (t["home_prev_margin"] < 0).astype(float).where(t["home_prev_margin"].notna())
    t["away_after_loss"] = (t["away_prev_margin"] < 0).astype(float).where(t["away_prev_margin"].notna())

    # ---- venue: travel, altitude, time zones, dome
    venues = load_file("venues.csv")
    if len(venues):
        v = venues.drop_duplicates("venue_id").set_index("venue_id")
        hv = home_venues(games)
        def loc(team_series, col):
            return team_series.map(hv).map(v[col])
        t["v_lat"], t["v_lon"] = t["venue_id"].map(v["lat"]), t["venue_id"].map(v["lon"])
        t["v_elev"], t["v_tz"] = t["venue_id"].map(v["elevation"]), t["venue_id"].map(v["timezone"])
        t["dome"] = t["venue_id"].map(v["dome"]).fillna(False).astype(float)
        for s in ("home", "away"):
            t[f"{s}_lat"], t[f"{s}_lon"] = loc(t[s], "lat"), loc(t[s], "lon")
            t[f"{s}_elev"], t[f"{s}_tz"] = loc(t[s], "elevation"), loc(t[s], "timezone")
            t[f"{s}_travel"] = [miles(a, b, c, d) for a, b, c, d in
                                zip(t[f"{s}_lat"], t[f"{s}_lon"], t["v_lat"], t["v_lon"])]
        t["travel_diff"] = t["away_travel"].fillna(0) - t["home_travel"].fillna(0)
        t["elev_diff"] = t["v_elev"] - t["away_elev"]
        starts = t["start"].dt.to_pydatetime()
        t["tz_diff"] = [_tz_offset(vt, s) - _tz_offset(at, s) if isinstance(vt, str) and isinstance(at, str) else np.nan
                        for vt, at, s in zip(t["v_tz"], t["away_tz"], starts)]
        t["local_kick_hour"] = [s.astimezone(ZoneInfo(vt)).hour + s.astimezone(ZoneInfo(vt)).minute / 60
                                if isinstance(vt, str) else np.nan for vt, s in zip(t["v_tz"], starts)]
    else:
        for c in ["travel_diff", "elev_diff", "tz_diff", "local_kick_hour", "dome"]:
            t[c] = np.nan

    # ---- coaching change
    for s in ("home", "away"):
        t[f"new_coach_{s}"] = np.nan
    for season in t["season"].unique():
        nc = engine.new_coach(season)
        if len(nc):
            m = t["season"] == season
            t.loc[m, "new_coach_home"] = t.loc[m, "home"].map(nc)
            t.loc[m, "new_coach_away"] = t.loc[m, "away"].map(nc)

    # ---- QB out (history from box scores)
    flags = qb.history_flags(load("passers"), games)
    for s in ("home", "away"):
        f = flags.rename(columns={"team": s, "qb_out": f"qb_out_{s}", "starter": f"starter_{s}"})
        t = t.merge(f[["game_id", s, f"qb_out_{s}", f"starter_{s}"]], on=["game_id", s], how="left")
        t[f"qb_out_{s}"] = t[f"qb_out_{s}"].fillna(0.0)

    # ---- weather
    wx = weather.table()
    t = t.merge(wx, on="game_id", how="left")
    for c in ["wind", "gust", "rain"]:
        t[c] = np.where(t["dome"] == 1, 0.0, t[c])
    t["temp"] = np.where(t["dome"] == 1, 70.0, t["temp"])

    # ---- market and targets
    lines = load("lines")
    if len(lines):
        t = t.merge(lines.drop_duplicates("game_id")[["game_id", "provider", "spread", "spread_open", "total", "total_open"]],
                    on="game_id", how="left")
    else:
        for c in ["provider", "spread", "spread_open", "total", "total_open"]:
            t[c] = np.nan
    t["actual_margin"] = t["home_pts"] - t["away_pts"]
    t["actual_total"] = t["home_pts"] + t["away_pts"]
    for c in ["neutral", "conf_game"]:
        t[c] = t[c].astype(float)
    return t.sort_values("start").reset_index(drop=True)
