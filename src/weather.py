"""Step 4: weather at kickoff, free from Open-Meteo (no key needed).

* Past games -> the weather that actually happened (training data).
* Upcoming games -> the forecast for the kickoff window.

Both are averaged from kickoff through WEATHER_WINDOW_HOURS later; the
strongest gust in that window is kept. Domes get no weather.
Results are cached in data/weather.csv so each past game is fetched once.

API docs: https://open-meteo.com/en/docs and
https://open-meteo.com/en/docs/historical-weather-api
"""
import os
import time

import pandas as pd

import config
from common import DATA

CACHE = os.path.join(DATA, "weather.csv")
FIELDS = "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
WORKERS = 6  # stadium-seasons fetched at the same time (well under the free per-minute limit)
COLS = ["game_id", "kind", "temp", "rain", "wind", "gust", "fetched_at"]


def load_cache():
    if os.path.exists(CACHE):
        return pd.read_csv(CACHE)
    return pd.DataFrame(columns=COLS)


class RateLimited(Exception):
    """Open-Meteo's free hourly/daily allowance is used up: stop and resume next run."""


def _hourly(url, lat, lon, start_date, end_date):
    import requests

    p = {"latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date,
         "hourly": FIELDS, "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
         "precipitation_unit": "inch", "timezone": "UTC"}
    r = requests.get(url, params=p, timeout=30)
    if r.status_code == 429:
        raise RateLimited(r.text[:200])
    r.raise_for_status()
    h = pd.DataFrame(r.json()["hourly"])
    h["time"] = pd.to_datetime(h["time"], utc=True)
    return h.set_index("time")


def _save(cache, new_rows):
    if not new_rows:
        return cache
    new = pd.DataFrame(new_rows)
    cache = cache[~cache.set_index(["game_id", "kind"]).index.isin(new.set_index(["game_id", "kind"]).index)]
    cache = pd.concat([cache, new], ignore_index=True)
    cache.to_csv(CACHE, index=False)
    return cache


def _window(hourly, kickoff):
    kick = pd.Timestamp(kickoff).floor("h")
    w = hourly.loc[kick: kick + pd.Timedelta(hours=config.WEATHER_WINDOW_HOURS)]
    if w.empty:
        return None
    return {"temp": round(w["temperature_2m"].mean(), 1), "rain": round(w["precipitation"].sum(), 2),
            "wind": round(w["wind_speed_10m"].mean(), 1), "gust": round(w["wind_gusts_10m"].max(), 1)}


def _attach_venues(games, venues):
    v = venues.set_index("venue_id")
    games = games.copy()
    games["lat"] = games["venue_id"].map(v["lat"])
    games["lon"] = games["venue_id"].map(v["lon"])
    games["dome"] = games["venue_id"].map(v["dome"]).fillna(False).astype(bool)
    return games[games["lat"].notna() & ~games["dome"]]


def update(games, venues, upcoming_ids=()):
    """Fill the cache: actual weather for finished games, forecasts for upcoming ones."""
    if config.OFFLINE or venues.empty:
        print("weather: offline, skipped")
        return load_cache()
    cache = load_cache()
    now = pd.Timestamp.now(tz="UTC")
    calls, errors = 0, 0
    new_rows = []
    started = time.time()

    # 1) History: one call per venue-season covers every game played there.
    done = set(cache.loc[cache["kind"] == "actual", "game_id"])
    past = _attach_venues(games[games["completed"].astype(bool) & ~games["game_id"].isin(done)
                                & ~games["start_tbd"].astype(bool)], venues)
    past = past[past["start"] < now - pd.Timedelta(hours=6)]
    groups = list(past.sort_values("start", ascending=False).groupby(["venue_id", "season"], sort=False))
    stop = False

    def fetch_group(item):
        (venue, season), grp = item
        first, last = grp["start"].min(), grp["start"].max() + pd.Timedelta(hours=6)
        url = ARCHIVE if last < now - pd.Timedelta(days=6) else FORECAST
        return item, _hourly(url, grp["lat"].iloc[0], grp["lon"].iloc[0],
                             first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d"))

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i in range(0, len(groups), WORKERS):
            minutes = (time.time() - started) / 60
            if stop or calls >= config.WEATHER_MAX_CALLS or minutes >= config.WEATHER_MAX_MINUTES or errors >= 10:
                print(f"weather: stopping history after {calls} calls / {minutes:.0f} min; the rest fills on later runs")
                break
            futures = [pool.submit(fetch_group, item) for item in groups[i:i + WORKERS]]
            for fut in futures:
                try:
                    ((venue, season), grp), hourly = fut.result()
                    calls += 1
                except RateLimited as e:
                    print(f"weather: free allowance used up after {calls} calls ({e}); the rest fills on later runs")
                    stop = True
                    continue
                except Exception as e:
                    errors += 1
                    print(f"  ! weather: {e}")
                    continue
                for gm in grp.itertuples():
                    wx = _window(hourly, gm.start)
                    if wx:
                        new_rows.append({"game_id": gm.game_id, "kind": "actual", **wx, "fetched_at": now.isoformat()})
            if calls and calls % 48 < WORKERS:
                cache, new_rows = _save(cache, new_rows), []

    # 2) Forecasts for this week's games (refreshed every run).
    upcoming = _attach_venues(games[games["game_id"].isin(upcoming_ids)], venues)
    upcoming = upcoming[upcoming["start"] < now + pd.Timedelta(days=15)]  # forecast horizon
    for venue, grp in upcoming.groupby("venue_id"):
        first, last = grp["start"].min(), grp["start"].max() + pd.Timedelta(hours=6)
        try:
            hourly = _hourly(FORECAST, grp["lat"].iloc[0], grp["lon"].iloc[0],
                             first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d"))
        except RateLimited:
            print("weather: allowance used up; this week's forecasts refresh on the next run")
            break
        except Exception as e:
            print(f"  ! forecast venue {venue}: {e}")
            continue
        for gm in grp.itertuples():
            if gm.start_tbd:
                continue
            wx = _window(hourly, gm.start)
            if wx:
                new_rows.append({"game_id": gm.game_id, "kind": "forecast", **wx, "fetched_at": now.isoformat()})

    cache = _save(cache, new_rows)
    print(f"weather: history calls this run: {calls}; cache now holds {len(cache)} games")
    return cache


def table():
    """One weather row per game: actual weather if known, else the latest forecast."""
    c = load_cache()
    if c.empty:
        return pd.DataFrame(columns=["game_id", "temp", "rain", "wind", "gust", "wx_kind", "wx_time"])
    c["rank"] = (c["kind"] == "actual").astype(int)
    c = c.sort_values(["rank", "fetched_at"]).groupby("game_id").tail(1)
    return c.rename(columns={"kind": "wx_kind", "fetched_at": "wx_time"})[
        ["game_id", "temp", "rain", "wind", "gust", "wx_kind", "wx_time"]]
