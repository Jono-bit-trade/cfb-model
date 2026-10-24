"""Shared helpers: file locations, the CFBD API client, and loaders."""
import glob
import json
import math
import os
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SITE_DATA = os.path.join(ROOT, "docs", "data")
CFBD_BASE = "https://api.collegefootballdata.com"


# ---------------------------------------------------------------- API client
def api_key():
    key = os.environ.get("CFBD_KEY")
    if not key and os.path.exists(os.path.join(ROOT, "key.txt")):
        key = open(os.path.join(ROOT, "key.txt")).read().strip()
    if not key:
        raise SystemExit("No API key: set the CFBD_KEY secret (GitHub) or create key.txt (your computer).")
    return key


def fetch(endpoint, **params):
    """GET a CFBD endpoint and return the parsed JSON (retries on rate limits)."""
    import requests

    headers = {"Authorization": f"Bearer {api_key()}", "Accept": "application/json"}
    params = {k: v for k, v in params.items() if v is not None}
    for attempt in range(5):
        r = requests.get(CFBD_BASE + endpoint, headers=headers, params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        time.sleep(0.3)
        return r.json()
    r.raise_for_status()


# ----------------------------------------------------- tolerant field access
def g(d, *names, default=None):
    """First present key among names. Handles camelCase vs snake_case renames.

    A dotted name ("offense.ppa") walks into nested dicts.
    """
    if d is None:
        return default
    for name in names:
        cur, ok = d, True
        for part in name.split("."):
            if isinstance(cur, dict) and part in cur and cur[part] is not None:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return default


# ------------------------------------------------------------- data storage
def dataset_dir(name):
    path = os.path.join(DATA, name)
    os.makedirs(path, exist_ok=True)
    return path


def save_season(df, name, season):
    df.to_csv(os.path.join(dataset_dir(name), f"{season}.csv"), index=False)


def has_season(name, season):
    return os.path.exists(os.path.join(DATA, name, f"{season}.csv"))


def load(name):
    """All seasons of a dataset as one table (empty table if missing)."""
    files = sorted(glob.glob(os.path.join(DATA, name, "*.csv")))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_file(name):
    path = os.path.join(DATA, name)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def _sanitize(o):
    """NaN/inf are not valid JSON: turn them into null, recursively."""
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            return _sanitize(o.item())
        except (ValueError, AttributeError):
            pass
    return o


def write_json(obj, *parts):
    path = os.path.join(*parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_sanitize(obj), f, indent=1, default=_json_default, allow_nan=False)


def _json_default(o):
    if hasattr(o, "isoformat"):
        return o.isoformat()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


# ------------------------------------------------------------------- geometry
def miles(lat1, lon1, lat2, lon2):
    if any(pd.isna(v) for v in (lat1, lon1, lat2, lon2)):
        return float("nan")
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 3959 * 2 * math.asin(math.sqrt(a))


def load_games():
    """All games, with parsed kickoff times, sorted by kickoff."""
    games = load("games")
    if games.empty:
        raise SystemExit("No games downloaded yet: run `python src/get_data.py` first.")
    games["start"] = pd.to_datetime(games["start"], utc=True, format="mixed")
    return games.sort_values("start").reset_index(drop=True)
