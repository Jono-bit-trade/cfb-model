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
BACKTEST_SEASONS = list(range(2016, THIS_SEASON))  # seasons the backtest scores (needs data from 2 seasons before)

# ---- Step 3: power ratings -----------------------------------------------
RATING_ALPHA = 12.0          # higher = trust the preseason prior longer
HALF_LIFE_DAYS = 70          # recency weighting inside a season
BLOWOUT_CAP = 38             # a winning margin never counts above this many points
DEFAULT_HFA = 2.5            # home edge (points) until history estimates it
PRIOR_FALLBACK_SHRINK = 0.6  # last season's rating x this, when no prior model
EFF_ALPHA = 4.0              # efficiency (EPA per play) ratings: higher = trust last season longer (tested 4/8/16)
EFF_PRIOR_SHRINK = 0.6       # efficiency ratings start at last season's x this

# ---- Step 6: machine learning blend --------------------------------------
BLEND_W = 0.5                # weight on the plain rating projection (1.0 = ratings only)
BLEND_W_TOTAL = 0.4          # same idea for totals; None = use BLEND_W
GBM_DEPTH = 2                # machine-learning trees: depth, learning rate, rounds, minimum games per leaf
GBM_LEARNING_RATE = 0.025    # (chosen on 2016-22 from 12 combinations; 2023-25 held out)
GBM_ROUNDS = 650
GBM_MIN_LEAF = 40
TOTALS_TRAIN_SEASONS = 5     # totals model trains on the most recent 5 seasons (3/5/7/all tested; spreads use all)
SEASON_HALF_LIFE = None      # ML layer: a season this many years older counts half (None = all seasons equal; 3 and 6 tested, no gain)
MARKET_ADJUST = True         # second step from line movement, where opening lines exist (2023 on): lowered 2024-25 margin error
MARKET_ADJUST_TOTALS = False # the same for totals didn't help in testing
WIND_TOTAL_PER_MPH = -0.12   # totals: points per mph of wind above WIND_TOTAL_FROM_MPH (learned 2016-22; dome and
WIND_TOTAL_FROM_MPH = 10     #   rain patterns were tested too but didn't hold up in 2023-25)
MARKET_ADJUST_ALPHA = 50.0   # its ridge penalty: higher = smaller adjustments
FEATURE_GROUPS = ["CORE", "STYLE", "CONTEXT", "SITUATION"]  # drop a group to test without it

# ---- Picks ----------------------------------------------------------------
MIN_EDGE = 0.0               # only show a pick when |edge| >= this many points
BET_EARLY_MAX_MOVE = 1.0     # "Bet early" only while the line has moved less than this against the model
STRONG_EDGE = 3.0            # label picks at or above this as "strong" (the tuner suggests a value)
SHARP_GAP = 10              # sharp money: money % exceeds bet % on a side by this many points
PUBLIC_MIN = 60             # reverse line movement: at least this % of bets on one side...
RLM_MIN_MOVE = 1.0          # ...while the spread moved at least this far the other way
MARGIN_SD = 16.6             # spread of real results around the projection, for win chances (measured 2016-25)
VALUE_MIN_PROB = 0.04        # value moneyline: model's win chance must beat the market's by this much

# ---- Weather ---------------------------------------------------------------
WEATHER_WINDOW_HOURS = 3     # average kickoff hour through this many hours after
WEATHER_MAX_CALLS = int(os.environ.get("WEATHER_MAX_CALLS", 1500))  # per run, for the history backfill
WEATHER_MAX_MINUTES = 12     # per run: the history backfill never holds up predictions longer than this

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

# Games with an opening spread this big: the model leans toward the opening line by an amount learned
# from past seasons (it underrates big favorites by 4-6 points otherwise). Tested 2023-25.
BIG_SPREAD = 21.0
# Spread edges in this range are the "sweet spot"; bigger edges usually mean the model is missing news.
SWEET_SPOT_MAX = 7.0
