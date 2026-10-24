"""All the settings you are likely to tune, in one place.

Change a value, rerun `python src/backtest.py`, and keep the change only if
the backtest improves across several seasons.
"""
import os

# ---- Seasons --------------------------------------------------------------
FIRST_SEASON = 2013          # oldest season downloaded
import datetime as _dt

_today = _dt.date.today()
# A season runs Aug-Jan, so January/February still belong to last year's season.
THIS_SEASON = int(os.environ.get("CFB_SEASON") or (_today.year if _today.month >= 3 else _today.year - 1))
BACKTEST_SEASONS = list(range(2018, THIS_SEASON))  # seasons the backtest scores

# ---- Step 3: power ratings -----------------------------------------------
RATING_ALPHA = 12.0          # higher = trust the preseason prior longer
HALF_LIFE_DAYS = 70          # recency weighting inside a season
BLOWOUT_CAP = 38             # a winning margin never counts above this many points
DEFAULT_HFA = 2.5            # home edge (points) until history estimates it
PRIOR_FALLBACK_SHRINK = 0.6  # last season's rating x this, when no prior model

# ---- Step 6: machine learning blend --------------------------------------
BLEND_W = 0.5                # weight on the plain rating projection (1.0 = ratings only)
FEATURE_GROUPS = ["CORE", "STYLE", "CONTEXT", "SITUATION"]  # drop a group to test without it

# ---- Picks ----------------------------------------------------------------
MIN_EDGE = 0.0               # only show a pick when |edge| >= this many points
STRONG_EDGE = 3.0            # label picks at or above this as "strong" (the tuner suggests a value)
MARGIN_SD = 16.0             # spread of real results around the projection, for win chances
VALUE_MIN_PROB = 0.04        # value moneyline: model's win chance must beat the market's by this much

# ---- Weather ---------------------------------------------------------------
WEATHER_WINDOW_HOURS = 3     # average kickoff hour through this many hours after
WEATHER_MAX_CALLS = int(os.environ.get("WEATHER_MAX_CALLS", 600))  # per run, for the history backfill

# ---- Sportsbook preference for the market line ---------------------------
LINE_PROVIDERS = ["consensus", "DraftKings", "ESPN Bet", "Bovada", "William Hill (New Jersey)", "teamrankings", "numberfire"]

# ---- Offline mode (tests, or no network) ---------------------------------
OFFLINE = os.environ.get("CFB_OFFLINE") == "1"

# ---- Tuned settings ----------------------------------------------------------
# `python src/tune.py` saves the best settings to data/tuned.json; they replace the
# defaults above, so the backtest, the History page and this week's predictions all
# use the same tuned model. Delete data/tuned.json to go back to the defaults.
import json as _json

_TUNED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "tuned.json")
TUNED = False
if os.path.exists(_TUNED) and os.environ.get("CFB_IGNORE_TUNED") != "1":
    try:
        for _k, _v in _json.load(open(_TUNED)).get("settings", {}).items():
            globals()[_k] = _v
        TUNED = True
    except (ValueError, OSError):
        pass
