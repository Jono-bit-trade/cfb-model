"""Step 6: the machine learning blend.

Gradient-boosted trees learn how much each input moves the final margin (and
total) beyond the plain rating projection. The final projection blends the two:

    margin = BLEND_W * rating_margin + (1 - BLEND_W) * ml_margin
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import config
from features import TOTAL_FEATURES, features_for


def _gbm():
    return HistGradientBoostingRegressor(max_depth=config.GBM_DEPTH, learning_rate=config.GBM_LEARNING_RATE,
                                         max_iter=config.GBM_ROUNDS, min_samples_leaf=config.GBM_MIN_LEAF,
                                         l2_regularization=1.0, random_state=7)


class Model:
    def __init__(self, groups=None, w=None):
        self.groups = groups or config.FEATURE_GROUPS
        self.features = features_for(self.groups)
        from features import TOTAL_EXTRAS
        self.total_features = TOTAL_FEATURES + (["total_move"] if "MARKET" in self.groups else []) + \
            [c for gname in self.groups for c in TOTAL_EXTRAS.get(gname, []) if c not in TOTAL_FEATURES]
        self.w = config.BLEND_W if w is None else w

    def fit(self, table):
        train = table[table["actual_margin"].notna() & table["rating_margin"].notna()]
        # Leave out inputs with (almost) no data, e.g. a stat the API doesn't provide.
        usable = lambda cols: [c for c in cols if c in train and train[c].notna().mean() >= 0.02 and train[c].nunique() > 1]
        self.margin_cols, self.total_cols = usable(self.features), usable(self.total_features)
        self.dropped = sorted(set(self.features + self.total_features) - set(self.margin_cols + self.total_cols))
        w = season_weights(train["season"])
        self.margin = _gbm().fit(train[self.margin_cols], train["actual_margin"], sample_weight=w)
        # Totals learn from recent seasons only: scoring levels drift (rules, pace, offenses).
        tt = train
        if config.TOTALS_TRAIN_SEASONS:
            tt = train[train["season"] > train["season"].max() - config.TOTALS_TRAIN_SEASONS]
        self.total = _gbm().fit(tt[self.total_cols], tt["actual_total"], sample_weight=season_weights(tt["season"]))
        self.n_train = len(train)
        return self

    def predict(self, rows):
        ml_m = self.margin.predict(rows[self.margin_cols])
        ml_t = self.total.predict(rows[self.total_cols])
        margin = self.w * rows["rating_margin"].to_numpy() + (1 - self.w) * ml_m
        wt = config.BLEND_W_TOTAL if config.BLEND_W_TOTAL is not None else self.w
        total = wt * rows["rating_total"].to_numpy() + (1 - wt) * ml_t
        total = total + wind_adjustment(rows)
        return margin, total


def wind_adjustment(rows):
    """Lower the projected total in windy outdoor games (learned on 2016-22, held on 2023-25):
    the model alone underrated how much wind above ~10 mph suppresses scoring."""
    if not config.WIND_TOTAL_PER_MPH or "wind" not in rows:
        return 0.0
    wind = rows["wind"].astype(float).fillna(0).to_numpy()
    if "dome" in rows:
        wind = np.where(rows["dome"].fillna(0).to_numpy() == 1, 0.0, wind)
    return config.WIND_TOTAL_PER_MPH * np.clip(wind - config.WIND_TOTAL_FROM_MPH, 0, None)


def season_weights(seasons):
    """Recent seasons count more: a season SEASON_HALF_LIFE years older counts half as much."""
    if not config.SEASON_HALF_LIFE:
        return None
    s = seasons.to_numpy(dtype=float)
    return 0.5 ** ((s.max() - s) / config.SEASON_HALF_LIFE)


class MarketAdjust:
    """Second step, used only where opening lines exist (2023 on in CFBD).

    Learns from out-of-sample projections how much the line's movement since opening
    says about the result the projection missed, e.g. "the final margin beat our
    projection by about b points for every point the line moved toward the home team".
    A small ridge regression on movement only, so it can't simply copy the market.
    """

    MARGIN_X = ["line_move", "line_move_x_edge", "big_spread_gap", "early_gap_1_2", "early_gap_3_4"]
    TOTAL_X = ["total_move"]

    @staticmethod
    def _frame(rows, margin, total):
        x = pd.DataFrame(index=rows.index)
        x["line_move"] = rows["spread_open"] - rows["spread"]
        edge_open = margin + rows["spread_open"]
        x["line_move_x_edge"] = x["line_move"] * np.sign(edge_open)  # moved toward / away from the model
        x["total_move"] = rows["total"] - rows["total_open"]
        # In lopsided games (opening spread of BIG_SPREAD+ points) the model shrinks mismatches toward
        # the middle and the market is better: learn how far to lean toward the opening line there.
        big = rows["spread_open"].abs() >= config.BIG_SPREAD
        x["big_spread_gap"] = np.where(big, -rows["spread_open"] - margin, 0.0)
        # Early in the season the ratings lean on preseason estimates and the market knows more:
        # learn how far to lean toward the opening line in weeks 1-2 and 3-4.
        gap = -rows["spread_open"] - margin
        reg = rows["season_type"].eq("regular") if "season_type" in rows else True
        wk = rows["week"] if "week" in rows else pd.Series(99, index=rows.index)
        x["early_gap_1_2"] = np.where(reg & (wk <= 2), gap, 0.0)
        x["early_gap_3_4"] = np.where(reg & (wk >= 3) & (wk <= 4), gap, 0.0)
        x.loc[rows["spread_open"].isna(), ["big_spread_gap", "early_gap_1_2", "early_gap_3_4"]] = np.nan
        return x

    def fit(self, rows, margin, total):
        from sklearn.linear_model import Ridge
        x = self._frame(rows, margin, total)
        mm = x[self.MARGIN_X].notna().all(axis=1) & rows["actual_margin"].notna()
        tm = x[self.TOTAL_X].notna().all(axis=1) & rows["actual_total"].notna()
        self.n_margin, self.n_total = int(mm.sum()), int(tm.sum())
        self.margin = Ridge(alpha=config.MARKET_ADJUST_ALPHA).fit(
            x.loc[mm, self.MARGIN_X], (rows["actual_margin"] - margin)[mm]) if mm.sum() >= 200 else None
        self.total = Ridge(alpha=config.MARKET_ADJUST_ALPHA).fit(
            x.loc[tm, self.TOTAL_X], (rows["actual_total"] - total)[tm]) if tm.sum() >= 200 else None
        return self

    def apply(self, rows, margin, total):
        margin, total = np.asarray(margin, dtype=float).copy(), np.asarray(total, dtype=float).copy()
        x = self._frame(rows, margin, total)
        if self.margin is not None:
            ok = x[self.MARGIN_X].notna().all(axis=1).to_numpy()
            margin[ok] += self.margin.predict(x.loc[ok, self.MARGIN_X])
        if self.total is not None and config.MARKET_ADJUST_TOTALS:
            ok = x[self.TOTAL_X].notna().all(axis=1).to_numpy()
            total[ok] += self.total.predict(x.loc[ok, self.TOTAL_X])
        return margin, total


def split_score(margin, total):
    """Projected (home points, away points) from margin and total."""
    return (total + margin) / 2, (total - margin) / 2


def ats_edge(pred_margin, spread):
    """Positive = the model likes the home team against the spread (spread is home's line)."""
    return pred_margin + spread


def grade(actual_margin, spread, edge):
    """1 win, 0 loss, 0.5 push, nan if no line or game not played."""
    if np.isnan(spread) or np.isnan(actual_margin) or np.isnan(edge) or edge == 0:
        return np.nan
    cover = actual_margin + spread
    if cover == 0:
        return 0.5
    return 1.0 if (cover > 0) == (edge > 0) else 0.0


def grade_total(actual_total, total, pred_total):
    if np.isnan(total) or np.isnan(actual_total) or pred_total == total:
        return np.nan
    if actual_total == total:
        return 0.5
    return 1.0 if (actual_total > total) == (pred_total > total) else 0.0
