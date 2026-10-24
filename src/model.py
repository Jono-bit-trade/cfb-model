"""Step 6: the machine learning blend.

Gradient-boosted trees learn how much each input moves the final margin (and
total) beyond the plain rating projection. The final projection blends the two:

    margin = BLEND_W * rating_margin + (1 - BLEND_W) * ml_margin
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import config
from features import TOTAL_FEATURES, features_for


def _gbm():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.04, max_iter=350,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=7)


class Model:
    def __init__(self, groups=None, w=None):
        self.features = features_for(groups or config.FEATURE_GROUPS)
        self.w = config.BLEND_W if w is None else w

    def fit(self, table):
        train = table[table["actual_margin"].notna() & table["rating_margin"].notna()]
        self.margin = _gbm().fit(train[self.features], train["actual_margin"])
        self.total = _gbm().fit(train[TOTAL_FEATURES], train["actual_total"])
        self.n_train = len(train)
        return self

    def predict(self, rows):
        ml_m = self.margin.predict(rows[self.features])
        ml_t = self.total.predict(rows[TOTAL_FEATURES])
        margin = self.w * rows["rating_margin"].to_numpy() + (1 - self.w) * ml_m
        total = self.w * rows["rating_total"].to_numpy() + (1 - self.w) * ml_t
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
