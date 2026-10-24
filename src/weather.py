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
COLS = ["game_id", "kind", "temp", "rain", "wind", "gust", "fetched_at"]


def load_cache():
    if os.path.exists(CACHE):
        return pd.read_csv(CACHE)
    return pd.DataFrame(columns=COLS)


def _hourly(url, lat, lon, start_date, end_date):
    import requests

    p = {"latitude": lat, "longitude": lon, "start_date": start_date, "end_date": end_date,
         "hourly": FIELDS, "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
         "precipitation_unit": "inch", "timezone": "UTC"}
    for attempt in range(4):
        r = requests.get(url, params=p, timeout=60)
        if r.status_code == 429:
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
        h = pd.DataFrame(r.json()["hourly"])
        h["time"] = pd.to_datetime(h["time"], utc=True)
        return h.set_index("time")
    r.raise_for_status()


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
    calls = 0
    new_rows = []

    # 1) History: one call per venue-season covers every game played there.
    done = set(cache.loc[cache["kind"] == "actual", "game_id"])
    past = _attach_venues(games[games["completed"].astype(bool) & ~games["game_id"].isin(done)
                                & ~games["start_tbd"].astype(bool)], venues)
    past = past[past["start"] < now - pd.Timedelta(hours=6)]
    for (venue, season), grp in past.sort_values("start", ascending=False).groupby(["venue_id", "season"], sort=False):
        if calls >= config.WEATHER_MAX_CALLS:
            print(f"weather: reached {calls} calls; the rest of the history fills on later runs")
            break
        first, last = grp["start"].min(), grp["start"].max() + pd.Timedelta(hours=6)
        url = ARCHIVE if last < now - pd.Timedelta(days=6) else FORECAST
        try:
            hourly = _hourly(url, grp["lat"].iloc[0], grp["lon"].iloc[0],
                             first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d"))
            calls += 1
        except Exception as e:
            print(f"  ! weather venue {venue} {season}: {e}")
            continue
        for gm in grp.itertuples():
            wx = _window(hourly, gm.start)
            if wx:
                new_rows.append({"game_id": gm.game_id, "kind": "actual", **wx, "fetched_at": now.isoformat()})

    # 2) Forecasts for this week's games (refreshed every run).
    upcoming = _attach_venues(games[games["game_id"].isin(upcoming_ids)], venues)
    upcoming = upcoming[upcoming["start"] < now + pd.Timedelta(days=15)]  # forecast horizon
    for venue, grp in upcoming.groupby("venue_id"):
        first, last = grp["start"].min(), grp["start"].max() + pd.Timedelta(hours=6)
        try:
            hourly = _hourly(FORECAST, grp["lat"].iloc[0], grp["lon"].iloc[0],
                             first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d"))
        except Exception as e:
            print(f"  ! forecast venue {venue}: {e}")
            continue
        for gm in grp.itertuples():
            if gm.start_tbd:
                continue
            wx = _window(hourly, gm.start)
            if wx:
                new_rows.append({"game_id": gm.game_id, "kind": "forecast", **wx, "fetched_at": now.isoformat()})

    if new_rows:
        new = pd.DataFrame(new_rows)
        cache = cache[~cache.set_index(["game_id", "kind"]).index.isin(new.set_index(["game_id", "kind"]).index)]
        cache = pd.concat([cache, new], ignore_index=True)
        cache.to_csv(CACHE, index=False)
    print(f"weather: {len(new_rows)} games updated with {calls} history calls")
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
