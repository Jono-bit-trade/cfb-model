"""Builds a fake but realistic data/ folder so the whole pipeline can be tested
without an API key or network. Never used for real predictions.

    python tests/make_sample_data.py /path/to/test-root

The fake world has known truths the model should discover: a home edge,
a starting QB being out costs points, wind lowers scoring, and roster data
(returning production, talent) predicts next season's strength.
"""
import os
import sys

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data")
FIRST, LAST = 2013, 2026
NOW = pd.Timestamp("2026-10-27T15:00Z")

CONFS = ["Alpha", "Bravo", "Coastal", "Delta", "Eastern", "Frontier"]
TEAMS = [f"{c} {i}" for c in ["Austin", "Boone", "Carver", "Dalton", "Easton", "Fulton", "Grady", "Hollis",
                               "Irving", "Jasper"] for i in ["State", "Tech", "U"]][:60]
FCS_TEAMS = [f"FCS College {i}" for i in range(1, 13)]
TZ = [(-122, "America/Los_Angeles"), (-110, "America/Denver"), (-92, "America/Chicago"), (-78, "America/New_York")]


def save(df, name, season):
    os.makedirs(os.path.join(DATA, name), exist_ok=True)
    df.to_csv(os.path.join(DATA, name, f"{season}.csv"), index=False)


def main():
    os.makedirs(DATA, exist_ok=True)
    n = len(TEAMS)
    conf = {t: CONFS[i % 6] for i, t in enumerate(TEAMS)}
    # venues
    venues = []
    for i, t in enumerate(TEAMS + ["Neutral Dome"]):
        lon = rng.uniform(-122, -72)
        tz = min(TZ, key=lambda z: abs(z[0] - lon))[1]
        venues.append({"venue_id": 100 + i, "name": f"{t} Stadium", "city": "Town", "state": "ST", "timezone": tz,
                       "lat": rng.uniform(29, 47), "lon": lon, "elevation": rng.choice([150, 300, 600, 1600, 2100]),
                       "dome": bool(i % 17 == 0 or t == "Neutral Dome"), "grass": True})
    pd.DataFrame(venues).to_csv(os.path.join(DATA, "venues.csv"), index=False)
    home_venue = {t: 100 + i for i, t in enumerate(TEAMS)}
    neutral_venue = 100 + n

    base_talent = {t: rng.normal(0, 1) for t in TEAMS}
    off = {t: 3 * base_talent[t] + rng.normal(0, 4) for t in TEAMS}
    dfn = {t: -3 * base_talent[t] + rng.normal(0, 4) for t in TEAMS}
    coach = {t: f"Coach {t}" for t in TEAMS}
    game_id = 400000000
    weather_rows = []

    for season in range(FIRST, LAST + 1):
        # roster inputs, which drive this season's change in strength
        talent = {t: base_talent[t] + rng.normal(0, 0.3) for t in TEAMS}
        ret = {t: float(np.clip(rng.normal(0.6, 0.18), 0.1, 0.98)) for t in TEAMS}
        recruit = {t: 200 + 40 * base_talent[t] + rng.normal(0, 15) for t in TEAMS}
        portal = {t: rng.normal(0, 6) for t in TEAMS}
        new_coach = {t: (season > FIRST and rng.random() < 0.1) for t in TEAMS}
        for t in TEAMS:
            if new_coach[t]:
                coach[t] = f"Coach {t} {season}"
            keep = 0.35 + 0.5 * ret[t]
            off[t] = keep * off[t] + (1 - keep) * 4 * talent[t] + 0.15 * portal[t] + rng.normal(0, 2.5)
            dfn[t] = keep * dfn[t] - (1 - keep) * 4 * talent[t] - 0.15 * portal[t] + rng.normal(0, 2.5)
        save(pd.DataFrame([{"season": season, "team": t, "talent": 700 + 60 * talent[t]} for t in TEAMS]), "talent", season)
        save(pd.DataFrame([{"season": season, "team": t, "ret_ppa": ret[t], "ret_pass": ret[t]} for t in TEAMS]), "returning", season)
        save(pd.DataFrame([{"season": season, "team": t, "recruit_pts": recruit[t]} for t in TEAMS]), "recruiting", season)
        if season >= 2018:
            save(pd.DataFrame([{"season": season, "team": t, "portal_in": max(portal[t], 0), "portal_out": max(-portal[t], 0),
                                "portal_net": portal[t]} for t in TEAMS]), "portal", season)
        save(pd.DataFrame([{"season": season, "team": t, "coach": coach[t], "games": 12} for t in TEAMS]), "coaches", season)
        save(pd.DataFrame([{"season": season, "team": t, "elo": 1500 + 25 * (off[t] - dfn[t])} for t in TEAMS]), "elo", season)

        starter = {t: f"{t} QB{season}" for t in TEAMS}
        first_sat = pd.Timestamp(f"{season}-09-01", tz="America/Chicago")
        first_sat += pd.Timedelta(days=(5 - first_sat.weekday()) % 7)
        g_rows, l_rows, p_rows, a_rows, b_rows, pass_rows = [], [], [], [], [], []
        for week in range(1, 13):
            order = list(rng.permutation(TEAMS))
            pairs = [(order[i], order[i + 1]) for i in range(0, n, 2)]
            if week in (1, 2):  # FCS cupcakes
                pairs = pairs[:-3] + [(order[-1 - i], FCS_TEAMS[(week * 3 + i) % len(FCS_TEAMS)]) for i in range(3)]
            for home, away in pairs:
                if away in FCS_TEAMS or rng.random() < 0.5:
                    pass
                else:
                    home, away = away, home
                game_id += 1
                neutral = rng.random() < 0.03
                venue = neutral_venue if neutral else home_venue[home]
                start = (first_sat + pd.Timedelta(days=7 * (week - 1), hours=int(rng.choice([11, 14, 18, 21])))).tz_convert("UTC")
                completed = start < NOW
                v = venues[venue - 100]
                is_dome = v["dome"]
                wind = 0.0 if is_dome else float(max(0, rng.gamma(2.2, 4)))
                temp = 70.0 if is_dome else float(rng.normal(62 - (week * 1.5), 12))
                rain = 0.0 if is_dome else float(max(0, rng.normal(-0.05, 0.08)))
                qb_out = {t: (t in TEAMS and rng.random() < 0.06) for t in (home, away)}
                o = lambda t: off.get(t, -14.0)
                d = lambda t: dfn.get(t, 12.0)
                hfa = 0 if neutral else 2.6
                wind_pen = -0.18 * max(wind - 10, 0)
                exp_h = 27 + o(home) + d(away) + hfa / 2 - 6.5 * qb_out[home] + wind_pen
                exp_a = 27 + o(away) + d(home) - hfa / 2 - 6.5 * qb_out[away] + wind_pen
                hp = int(max(0, round(exp_h + rng.normal(0, 11))))
                ap = int(max(0, round(exp_a + rng.normal(0, 11))))
                is_fbs = lambda t: "fbs" if t in TEAMS else "fcs"
                g_rows.append({"game_id": game_id, "season": season, "week": week, "season_type": "regular",
                               "start": start.isoformat(), "start_tbd": False, "completed": completed, "neutral": neutral,
                               "conf_game": conf.get(home) == conf.get(away), "venue_id": venue, "venue": v["name"],
                               "home": home, "away": away, "home_class": is_fbs(home), "away_class": is_fbs(away),
                               "home_conf": conf.get(home, "FCS"), "away_conf": conf.get(away, "FCS"),
                               "home_pts": hp if completed else None, "away_pts": ap if completed else None})
                true_margin = exp_h - exp_a
                spread = round((-true_margin + rng.normal(0, 2.8)) * 2) / 2
                total = round((exp_h + exp_a + rng.normal(0, 3)) * 2) / 2
                l_rows.append({"game_id": game_id, "season": season, "provider": "consensus", "spread": spread,
                               "spread_open": spread + rng.choice([-1.5, -1, 0, 0, 1, 1.5]), "total": total,
                               "total_open": total + rng.choice([-2, 0, 2])})
                if not is_dome:
                    weather_rows.append({"game_id": game_id, "kind": "actual" if completed else "forecast",
                                         "temp": round(temp, 1), "rain": round(rain, 2), "wind": round(wind, 1),
                                         "gust": round(wind * 1.6, 1), "fetched_at": NOW.isoformat()})
                if not completed:
                    continue
                for team, opp, pts, opp_pts in ((home, away, hp, ap), (away, home, ap, hp)):
                    p_rows.append({"game_id": game_id, "season": season, "team": team, "opponent": opp,
                                   "off_ppa": 0.02 * (pts - 27) + 0.25 + rng.normal(0, 0.05),
                                   "def_ppa": 0.02 * (opp_pts - 27) + 0.25 + rng.normal(0, 0.05)})
                    a_rows.append({"game_id": game_id, "season": season, "team": team, "opponent": opp,
                                   "off_plays": rng.normal(70, 6), "off_success": 0.42 + 0.004 * (pts - 27) + rng.normal(0, 0.03),
                                   "off_explosive": 1.25 + rng.normal(0, 0.1), "off_ppo": 4 + 0.03 * (pts - 27) + rng.normal(0, 0.4),
                                   "off_havoc": 0.16 + rng.normal(0, 0.03), "def_plays": rng.normal(70, 6),
                                   "def_success": 0.42 + 0.004 * (opp_pts - 27) + rng.normal(0, 0.03),
                                   "def_explosive": 1.25 + rng.normal(0, 0.1), "def_ppo": 4 + 0.03 * (opp_pts - 27) + rng.normal(0, 0.4),
                                   "def_havoc": 0.16 + rng.normal(0, 0.03)})
                    b_rows.append({"game_id": game_id, "season": season, "team": team, "turnovers": rng.poisson(1.4),
                                   "fumbles_lost": rng.poisson(0.5), "fumbles_rec": rng.poisson(0.5), "interceptions": rng.poisson(0.8),
                                   "pen_yards": rng.normal(55, 15)})
                    if team in TEAMS:
                        if qb_out[team]:
                            pass_rows.append({"season": season, "game_id": game_id, "team": team, "player_id": 2,
                                              "player": f"{team} Backup", "att": int(rng.integers(20, 34))})
                        else:
                            pass_rows.append({"season": season, "game_id": game_id, "team": team, "player_id": 1,
                                              "player": starter[team], "att": int(rng.integers(22, 40))})
                            if rng.random() < 0.3:
                                pass_rows.append({"season": season, "game_id": game_id, "team": team, "player_id": 2,
                                                  "player": f"{team} Backup", "att": int(rng.integers(1, 6))})
                    if rng.random() < 0.02 and team in TEAMS:
                        starter[team] = f"{team} Backup"  # mid-season benching
        save(pd.DataFrame(g_rows), "games", season)
        save(pd.DataFrame(l_rows), "lines", season)
        save(pd.DataFrame(p_rows), "ppa", season)
        save(pd.DataFrame(a_rows), "adv", season)
        save(pd.DataFrame(b_rows), "teambox", season)
        save(pd.DataFrame(pass_rows), "passers", season)

    pd.DataFrame(weather_rows).to_csv(os.path.join(DATA, "weather.csv"), index=False)
    # This week's QB report: one out, one questionable.
    upcoming = pd.read_csv(os.path.join(DATA, "games", f"{LAST}.csv"))
    upcoming = upcoming[upcoming["week"] == 9]
    a, b = upcoming.iloc[0], upcoming.iloc[1]
    pd.DataFrame([
        {"season": LAST, "week": 9, "team": a["home"], "qb": f"{a['home']} QB{LAST}", "status": "out", "note": "shoulder"},
        {"season": LAST, "week": 9, "team": b["away"], "qb": f"{b['away']} QB{LAST}", "status": "questionable", "note": "ankle"},
    ]).to_csv(os.path.join(DATA, "qb_status.csv"), index=False)
    print("sample data written to", DATA)


if __name__ == "__main__":
    main()
