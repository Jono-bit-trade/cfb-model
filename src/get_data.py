"""Step 2: download CollegeFootballData into small CSV files under data/.

First run downloads every season since FIRST_SEASON (several hundred API calls).
Later runs only refresh the current season, so the weekly job stays cheap.

    python src/get_data.py              # history (once) + current season
    python src/get_data.py --season 2025  # one season only
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from common import CALLS, OutOfCalls, fetch, g, has_season, load, save_season, dataset_dir, DATA  # noqa: E402

SEASON_TYPES = ["regular", "postseason"]


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------
# Parsers: raw API JSON -> compact table. Each tolerates old and new field
# names, so a CFBD rename only needs a fix here.
# --------------------------------------------------------------------------
def parse_games(raw, season):
    rows = []
    for x in raw:
        home_pts, away_pts = g(x, "homePoints", "home_points"), g(x, "awayPoints", "away_points")
        rows.append({
            "game_id": g(x, "id"),
            "season": season,
            "week": g(x, "week"),
            "season_type": g(x, "seasonType", "season_type", default="regular"),
            "start": g(x, "startDate", "start_date"),
            "start_tbd": bool(g(x, "startTimeTBD", "start_time_tbd", default=False)),
            "completed": bool(g(x, "completed", default=home_pts is not None)),
            "neutral": bool(g(x, "neutralSite", "neutral_site", default=False)),
            "conf_game": bool(g(x, "conferenceGame", "conference_game", default=False)),
            "venue_id": g(x, "venueId", "venue_id"),
            "venue": g(x, "venue"),
            "home": g(x, "homeTeam", "home_team"),
            "away": g(x, "awayTeam", "away_team"),
            "home_class": str(g(x, "homeClassification", "homeDivision", "home_division", default="")).lower(),
            "away_class": str(g(x, "awayClassification", "awayDivision", "away_division", default="")).lower(),
            "home_conf": g(x, "homeConference", "home_conference"),
            "away_conf": g(x, "awayConference", "away_conference"),
            "home_pts": home_pts,
            "away_pts": away_pts,
        })
    return pd.DataFrame(rows)


def parse_lines(raw, season):
    rows = []
    prefer = [p.lower() for p in config.LINE_PROVIDERS]
    for x in raw:
        lines = g(x, "lines", default=[]) or []
        if not lines:
            continue
        rank = lambda l: prefer.index(str(g(l, "provider", default="")).lower()) if str(g(l, "provider", default="")).lower() in prefer else len(prefer)
        usable = [l for l in lines if g(l, "spread") is not None or g(l, "overUnder", "over_under") is not None]
        if not usable:
            continue
        best = sorted(usable, key=rank)[0]
        rows.append({
            "game_id": g(x, "id"),
            "season": season,
            "provider": g(best, "provider"),
            "spread": _num(g(best, "spread")),            # home perspective: -7 = home favored by 7
            "spread_open": _num(g(best, "spreadOpen", "spread_open")),
            "total": _num(g(best, "overUnder", "over_under")),
            "total_open": _num(g(best, "overUnderOpen", "over_under_open")),
            "home_ml": _num(_first_ml(usable, rank, "homeMoneyline", "home_moneyline")),
            "away_ml": _num(_first_ml(usable, rank, "awayMoneyline", "away_moneyline")),
        })
    return pd.DataFrame(rows)


def _first_ml(lines, rank, *names):
    """Moneyline from the most preferred book that posted one (not every book does)."""
    for l in sorted(lines, key=rank):
        v = g(l, *names)
        if v is not None:
            return v
    return None


def parse_venues(raw):
    rows = []
    for x in raw:
        rows.append({
            "venue_id": g(x, "id"),
            "name": g(x, "name"),
            "city": g(x, "city"),
            "state": g(x, "state"),
            "timezone": g(x, "timezone"),
            "lat": _num(g(x, "latitude", "location.y", "location.latitude")),
            "lon": _num(g(x, "longitude", "location.x", "location.longitude")),
            "elevation": _num(g(x, "elevation")),
            "dome": bool(g(x, "dome", default=False)),
            "grass": g(x, "grass"),
        })
    return pd.DataFrame(rows)


def parse_ppa(raw, season):
    rows = []
    for x in raw:
        rows.append({
            "game_id": g(x, "gameId", "game_id"),
            "season": season,
            "team": g(x, "team"),
            "opponent": g(x, "opponent"),
            "off_ppa": _num(g(x, "offense.overall")),
            "def_ppa": _num(g(x, "defense.overall")),
        })
    return pd.DataFrame(rows)


def parse_adv(raw, season):
    rows = []
    for x in raw:
        row = {"game_id": g(x, "gameId", "game_id"), "season": season, "team": g(x, "team"), "opponent": g(x, "opponent")}
        for side, key in (("off", "offense"), ("def", "defense")):
            row[f"{side}_plays"] = _num(g(x, f"{key}.plays"))
            row[f"{side}_success"] = _num(g(x, f"{key}.successRate", f"{key}.success_rate"))
            row[f"{side}_explosive"] = _num(g(x, f"{key}.explosiveness"))
            row[f"{side}_ppo"] = _num(g(x, f"{key}.pointsPerOpportunity", f"{key}.points_per_opportunity"))
            row[f"{side}_havoc"] = _num(g(x, f"{key}.havoc.total"))
        rows.append(row)
    return pd.DataFrame(rows)


def _stat(stats, *categories):
    for s in stats or []:
        if g(s, "category") in categories:
            return g(s, "stat")
    return None


def parse_teambox(raw, season):
    rows = []
    for x in raw:
        for t in g(x, "teams", default=[]) or []:
            stats = g(t, "stats", default=[])
            pen = _stat(stats, "totalPenaltiesYards", "total_penalties_yards")
            rows.append({
                "game_id": g(x, "id"),
                "season": season,
                "team": g(t, "team", "school"),
                "turnovers": _num(_stat(stats, "turnovers")),
                "fumbles_lost": _num(_stat(stats, "fumblesLost", "fumbles_lost")),
                "fumbles_rec": _num(_stat(stats, "fumblesRecovered", "fumbles_recovered")),
                "interceptions": _num(_stat(stats, "interceptions")),
                "pen_yards": _num(str(pen).split("-")[-1]) if pen else float("nan"),
            })
    return pd.DataFrame(rows)


def parse_passers(raw, season):
    """Pass attempts per passer per game, for the QB-out flag (Step 4)."""
    rows = []
    for x in raw:
        for t in g(x, "teams", default=[]) or []:
            for cat in g(t, "categories", default=[]) or []:
                if g(cat, "name") != "passing":
                    continue
                for typ in g(cat, "types", default=[]) or []:
                    if g(typ, "name") not in ("C/ATT", "c/att"):
                        continue
                    for a in g(typ, "athletes", default=[]) or []:
                        stat = str(g(a, "stat", default="0/0"))
                        att = _num(stat.split("/")[-1]) if "/" in stat else _num(stat)
                        rows.append({"season": season, "game_id": g(x, "id"), "team": g(t, "team", "school"),
                                     "player_id": g(a, "id"), "player": g(a, "name"), "att": att})
    return pd.DataFrame(rows)


def parse_simple(raw, season, value_fields, team_fields=("team", "school")):
    rows = []
    for x in raw:
        row = {"season": season, "team": g(x, *team_fields)}
        for out, names in value_fields.items():
            row[out] = _num(g(x, *names))
        rows.append(row)
    return pd.DataFrame(rows)


def parse_portal(raw, season):
    """Net transfer talent per team: stars gained minus stars lost."""
    gained, lost = {}, {}
    for x in raw:
        stars = _num(g(x, "stars"))
        stars = 2.0 if pd.isna(stars) else stars
        dest, origin = g(x, "destination"), g(x, "origin")
        if dest:
            gained[dest] = gained.get(dest, 0) + stars
        if origin:
            lost[origin] = lost.get(origin, 0) + stars
    teams = set(gained) | set(lost)
    return pd.DataFrame([{"season": season, "team": t, "portal_in": gained.get(t, 0), "portal_out": lost.get(t, 0),
                          "portal_net": gained.get(t, 0) - lost.get(t, 0)} for t in teams])


def parse_cfbd_weather(raw, season):
    """CFBD's own game-day weather (Patreon tiers): already matched to each game's kickoff."""
    rows = []
    for x in raw:
        wind = _num(g(x, "windSpeed", "wind_speed"))
        gust = _num(g(x, "windGust", "wind_gust"))
        rows.append({"game_id": g(x, "id"), "season": season,
                     "indoors": bool(g(x, "gameIndoors", "game_indoors", default=False)),
                     "temp": _num(g(x, "temperature")), "wind": wind,
                     "gust": gust if not pd.isna(gust) else wind,
                     "rain": _num(g(x, "precipitation")), "snow": _num(g(x, "snowfall")),
                     "condition": g(x, "weatherCondition", "weather_condition")})
    return pd.DataFrame(rows)


WEATHER_ACCESS = {"ok": True}


def refresh_cfbd_weather(season, current):
    """2 calls per season. Skipped for the rest of the run if the plan doesn't include weather."""
    if not WEATHER_ACCESS["ok"] or (has_season("cfbd_weather", season) and not current):
        return
    try:
        rows = []
        for st in SEASON_TYPES:
            rows += fetch("/games/weather", year=season, seasonType=st) or []
        save_season(parse_cfbd_weather(rows, season), "cfbd_weather", season)
        print("  saved cfbd_weather")
    except OutOfCalls:
        raise
    except Exception as e:
        WEATHER_ACCESS["ok"] = False
        print(f"  ! CFBD weather not available on this plan ({e}); using the free weather source instead")


def parse_coaches(raw, season):
    rows = []
    for c in raw:
        name = f"{g(c, 'firstName', 'first_name', default='')} {g(c, 'lastName', 'last_name', default='')}".strip()
        for s in g(c, "seasons", default=[]) or []:
            if g(s, "year") == season:
                rows.append({"season": season, "team": g(s, "school"), "coach": name,
                             "games": _num(g(s, "games"))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Download plan
# --------------------------------------------------------------------------
def _save_partial(frames, name, season, done_file, done_weeks):
    df = pd.concat([f for f in frames if len(f)], ignore_index=True) if frames else pd.DataFrame()
    if len(df):
        save_season(df.drop_duplicates(), name, season)
    with open(done_file, "w") as f:
        f.write("\n".join(sorted(done_weeks)))


def _both_types(endpoint, season, **params):
    out = []
    for st in SEASON_TYPES:
        try:
            out += fetch(endpoint, year=season, seasonType=st, **params) or []
        except OutOfCalls:
            raise
        except Exception as e:  # some endpoints/years have no postseason data
            print(f"  ! {endpoint} {season} {st}: {e}")
    return out


SEASONAL = {
    # name: (download function, parser)
    "games": (lambda s: _both_types("/games", s, classification="fbs"), parse_games),
    "lines": (lambda s: _both_types("/lines", s), parse_lines),
    "ppa": (lambda s: _both_types("/ppa/games", s), parse_ppa),
    "adv": (lambda s: _both_types("/stats/game/advanced", s), parse_adv),
}
PRESEASON = {
    "talent": (lambda s: fetch("/talent", year=s), lambda r, s: parse_simple(r, s, {"talent": ("talent",)})),
    "returning": (lambda s: fetch("/player/returning", year=s),
                  lambda r, s: parse_simple(r, s, {"ret_ppa": ("percentPPA", "percent_ppa"),
                                                   "ret_pass": ("percentPassingPPA", "percent_passing_ppa")})),
    "recruiting": (lambda s: fetch("/recruiting/teams", year=s), lambda r, s: parse_simple(r, s, {"recruit_pts": ("points",)})),
    "portal": (lambda s: fetch("/player/portal", year=s), parse_portal),
    "coaches": (lambda s: fetch("/coaches", year=s), parse_coaches),
    "elo": (lambda s: fetch("/ratings/elo", year=s), lambda r, s: parse_simple(r, s, {"elo": ("elo",)})),
}
WEEKLY = {
    "teambox": ("/games/teams", parse_teambox),
    "passers": ("/games/players", parse_passers),
}


# Columns added in later versions: an older saved file without them is re-downloaded once.
REQUIRED_COLUMNS = {"lines": ["home_ml", "away_ml"]}


def _outdated(name, season):
    need = REQUIRED_COLUMNS.get(name)
    if not need:
        return False
    header = open(os.path.join(DATA, name, f"{season}.csv")).readline()
    return not all(col in header for col in need)


def refresh_season(season, current):
    print(f"Season {season}{' (current)' if current else ''}")
    for name, (download, parser) in SEASONAL.items():
        if has_season(name, season) and not current and not _outdated(name, season):
            continue
        try:
            df = parser(download(season), season).drop_duplicates()
            if name in ("games", "lines") and len(df):
                df = df.drop_duplicates("game_id", keep="last")
            save_season(df, name, season)
            print(f"  saved {name}")
        except OutOfCalls:
            raise
        except Exception as e:
            print(f"  ! {name}: {e}")

    for name, (download, parser) in PRESEASON.items():
        if has_season(name, season):
            continue  # preseason inputs don't change during a season
        try:
            save_season(parser(download(season) or [], season), name, season)
            print(f"  saved {name}")
        except OutOfCalls:
            raise
        except Exception as e:
            print(f"  ! {name}: {e} (skipped; not every season has this data)")

    refresh_cfbd_weather(season, current)

    # Per-week endpoints: only fetch weeks with completed games we don't have yet.
    games = load_games_season(season)
    if games.empty:
        return
    for name, (endpoint, parser) in WEEKLY.items():
        path = os.path.join(dataset_dir(name), f"{season}.csv")
        have = pd.read_csv(path) if os.path.exists(path) and os.path.getsize(path) > 0 else pd.DataFrame()
        have_ids = set(have["game_id"]) if "game_id" in have else set()
        done_file = os.path.join(dataset_dir(name), f"weeks_done_{season}.txt")
        done_weeks = set(open(done_file).read().split()) if os.path.exists(done_file) else set()
        need = games[games["completed"].astype(bool) & ~games["game_id"].isin(have_ids)]
        need = need[~(need["season_type"] + ":" + need["week"].astype(str)).isin(done_weeks)]
        if need.empty:
            continue
        frames = [have] if len(have) else []
        for (week, stype), _ in need.groupby(["week", "season_type"]):
            try:
                frames.append(parser(fetch(endpoint, year=season, week=int(week), seasonType=stype) or [], season))
                week_games = games[(games["week"] == week) & (games["season_type"] == stype)]
                if week_games["completed"].astype(bool).all():
                    done_weeks.add(f"{stype}:{week}")  # whole week final: never fetch it again
            except OutOfCalls:
                _save_partial(frames, name, season, done_file, done_weeks)
                raise
            except Exception as e:
                print(f"  ! {name} week {week} {stype}: {e}")
        with open(done_file, "w") as f:
            f.write("\n".join(sorted(done_weeks)))
        df = pd.concat([f for f in frames if len(f)], ignore_index=True) if frames else pd.DataFrame()
        if len(df):
            df = df.drop_duplicates()
            save_season(df, name, season)
            print(f"  saved {name} ({len(need)} new games)")


def check_allowance():
    """Print and save the plan's tier and remaining calls (the /info endpoint is free)."""
    import json
    try:
        info = fetch("/info") or {}
        rem = g(info, "remainingCalls", "remaining_calls")
        if rem is not None:
            CALLS["remaining"], CALLS["used"] = int(rem), 0
        tier = g(info, "patronLevel", "patron_level", default=0)
        print(f"CFBD plan: tier {tier}, {rem if rem is not None else 'unknown'} calls left this month")
        with open(os.path.join(DATA, "api_status.json"), "w") as f:
            json.dump({"tier": tier, "remaining_calls": rem, "monthly_limit": g(info, "monthlyLimit", "monthly_limit"),
                       "weather_feature": g(info, "features.weather"),
                       "checked_at": pd.Timestamp.now(tz="UTC").isoformat()}, f, indent=1)
    except Exception as e:
        print(f"(could not read the API allowance: {e})")


def load_games_season(season):
    path = os.path.join(DATA, "games", f"{season}.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, help="refresh just this season")
    ap.add_argument("--current-only", action="store_true", help="skip the history backfill")
    ap.add_argument("--quick", action="store_true", help="only refresh this season's schedule and betting lines (4 calls)")
    ap.add_argument("--from-season", type=int, default=config.FIRST_SEASON,
                    help="oldest season to backfill (use a later year if your API plan has few calls)")
    args = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    if args.quick:
        s = config.THIS_SEASON
        for name in ("games", "lines"):
            download, parser = SEASONAL[name]
            save_season(parser(download(s), s), name, s)
            print(f"saved {name} {s}")
        return
    check_allowance()
    venues_path = os.path.join(DATA, "venues.csv")
    if not os.path.exists(venues_path):
        try:
            parse_venues(fetch("/venues")).to_csv(venues_path, index=False)
            print("saved venues")
        except Exception as e:
            print(f"! venues: {e}")
    try:
        if args.season:
            refresh_season(args.season, current=args.season == config.THIS_SEASON)
        else:
            # This season first, then history newest-first: if the API allowance runs
            # out, the seasons that matter most are already saved.
            refresh_season(config.THIS_SEASON, current=True)
            if not args.current_only:
                for s in range(config.THIS_SEASON - 1, args.from_season - 1, -1):
                    refresh_season(s, current=False)
    except OutOfCalls as e:
        print(f"! {e}. Everything downloaded so far is saved; the rest continues next month or on a bigger plan.")

    print("Done. Rows per dataset:")
    for name in list(SEASONAL) + list(PRESEASON) + list(WEEKLY):
        df = load(name)
        print(f"  {name:11s} {len(df):7d}")


if __name__ == "__main__":
    main()
