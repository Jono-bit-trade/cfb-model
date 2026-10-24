"""Step 3: opponent-adjusted offense and defense ratings, in points.

Each team's score in a game is modeled as

    points = average + offense[team] + defense[opponent] + home edge

`offense` = points scored above an average team, `defense` = points allowed
above average (lower is better), overall = offense - defense.

Ratings start each season at a preseason prior built from last season's final
rating plus roster data (returning production, talent, recruiting, transfer
portal, coaching change). A ridge regression then moves each team away from its
prior as games are played; RATING_ALPHA sets how fast.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import config
from common import load

FCS = "FCS"  # every non-FBS opponent is pooled into one team


def team_keys(games):
    """Add home_key/away_key: the team name, or FCS for non-FBS teams."""
    games = games.copy()
    for side in ("home", "away"):
        cls = games[f"{side}_class"].fillna("").astype(str).str.lower()
        is_fbs = cls.isin(["fbs", "", "nan", "none"])
        games[f"{side}_key"] = np.where(is_fbs, games[side], FCS)
    return games


def _long_rows(games):
    """Two rows per game: each side's scoring, with the margin capped."""
    h, a = games["home_pts"].astype(float), games["away_pts"].astype(float)
    cap = config.BLOWOUT_CAP
    h_c = np.where(h - a > cap, a + cap, h)
    a_c = np.where(a - h > cap, h + cap, a)
    home_flag = np.where(games["neutral"].astype(bool), 0.0, 1.0)
    rows = pd.DataFrame({
        "off": np.concatenate([games["home_key"], games["away_key"]]),
        "def": np.concatenate([games["away_key"], games["home_key"]]),
        "home": np.concatenate([home_flag, -home_flag]),
        "pts": np.concatenate([h_c, a_c]),
        "start": np.concatenate([games["start"], games["start"]]),
    })
    return rows


def _fit(rows, alpha, y, weight=None, with_home=False):
    X = pd.get_dummies(rows[["off", "def"]], prefix=["off", "def"], prefix_sep="|").astype(float)
    if with_home:
        X["home"] = rows["home"].to_numpy() / 2
    m = Ridge(alpha=alpha, fit_intercept=True).fit(X, y, sample_weight=weight)
    c = pd.Series(m.coef_, index=X.columns)
    off = c[[k for k in c.index if k.startswith("off|")]].rename(lambda s: s[4:])
    dfn = c[[k for k in c.index if k.startswith("def|")]].rename(lambda s: s[4:])
    hfa = float(c["home"]) if with_home else None
    return off, dfn, float(m.intercept_), hfa


class RatingEngine:
    """Builds ratings for any season and date, using only earlier information."""

    def __init__(self, games):
        self.games = team_keys(games)
        self.completed = self.games[self.games["completed"].astype(bool) & self.games["home_pts"].notna()]
        self._final, self._priors = {}, {}
        self.inputs = {name: load(name) for name in ("returning", "talent", "recruiting", "portal", "coaches")}

    # ---------------------------------------------------- end-of-season fits
    def season_final(self, season):
        """Ratings fit on a whole finished season, no prior. Used for priors and HFA."""
        if season in self._final:
            return self._final[season]
        played = self.completed[self.completed["season"] == season]
        if len(played) < 50:
            self._final[season] = None
            return None
        rows = _long_rows(played)
        off, dfn, avg, hfa = _fit(rows, alpha=4.0, y=rows["pts"], with_home=True)
        r = pd.DataFrame({"off": off, "def": dfn}).fillna(0.0)
        r["overall"] = r["off"] - r["def"]
        self._final[season] = {"ratings": r, "avg": avg, "hfa": hfa}
        return self._final[season]

    def hfa_for(self, season):
        vals = [self.season_final(s)["hfa"] for s in range(season - 5, season) if self.season_final(s)]
        return float(np.mean(vals)) if vals else config.DEFAULT_HFA

    # ---------------------------------------------------------- the priors
    def _prior_features(self, season, teams):
        last = self.season_final(season - 1)
        f = pd.DataFrame(index=sorted(teams))
        if last is not None:
            lr = last["ratings"]
            f["last_overall"] = lr["overall"].reindex(f.index)
            f["last_off"] = lr["off"].reindex(f.index)
            f["last_def"] = lr["def"].reindex(f.index)
            new_team = f["last_overall"].isna()
            fbs_last = lr.drop(index=FCS, errors="ignore")
            f.loc[new_team, "last_overall"] = fbs_last["overall"].quantile(0.1)
            f.loc[new_team, "last_off"] = fbs_last["off"].quantile(0.15)
            f.loc[new_team, "last_def"] = fbs_last["def"].quantile(0.85)
        else:
            f["last_overall"] = f["last_off"] = f["last_def"] = 0.0

        def season_value(name, col, s):
            df = self.inputs.get(name, pd.DataFrame())
            if df.empty or col not in df:
                return pd.Series(dtype=float)
            return df[df["season"] == s].groupby("team")[col].mean()

        f["ret_ppa"] = season_value("returning", "ret_ppa", season).reindex(f.index)
        talent = season_value("talent", "talent", season)
        if talent.empty:
            talent = season_value("talent", "talent", season - 1)
        f["talent"] = talent.reindex(f.index)
        rec = [season_value("recruiting", "recruit_pts", s).reindex(f.index) for s in range(season - 3, season + 1)]
        f["recruiting"] = pd.concat(rec, axis=1).mean(axis=1) if rec else np.nan
        f["portal_net"] = season_value("portal", "portal_net", season).reindex(f.index)
        f["new_coach"] = self.new_coach(season).reindex(f.index).astype(float)
        return f

    def new_coach(self, season):
        """1 if the team's head coach is in his first season there."""
        c = self.inputs.get("coaches", pd.DataFrame())
        if c.empty or season - 1 not in set(c["season"]):
            return pd.Series(dtype=float)
        first = c.groupby(["team", "coach"])["season"].min().rename("first").reset_index()
        cur = c[c["season"] == season].merge(first, on=["team", "coach"])
        cur["new"] = (cur["first"] == season).astype(float)
        return cur.groupby("team")["new"].max()

    PRIOR_COLS = ["last_overall", "ret_ppa", "talent", "recruiting", "portal_net", "new_coach"]

    def priors(self, season):
        """Preseason offense/defense prior per team, plus the league average."""
        if season in self._priors:
            return self._priors[season]
        teams = set(self.games.loc[self.games["season"] == season, "home_key"]) | \
            set(self.games.loc[self.games["season"] == season, "away_key"]) | {FCS}
        f = self._prior_features(season, teams)
        last = self.season_final(season - 1)
        shrink = config.PRIOR_FALLBACK_SHRINK
        predicted = shrink * f["last_overall"]

        # Learn the prior model from earlier seasons: roster data -> final rating.
        train = []
        for s in range(config.FIRST_SEASON + 1, season):
            final = self.season_final(s)
            if final is None or self.season_final(s - 1) is None:
                continue
            fs = self._prior_features(s, set(final["ratings"].index) - {FCS})
            fs["target"] = final["ratings"]["overall"].reindex(fs.index)
            train.append(fs)
        if len(train) >= 2:
            t = pd.concat(train)
            cols = [c for c in self.PRIOR_COLS if t[c].notna().mean() > 0.5 and f[c].notna().mean() > 0.5]
            if "last_overall" in cols:
                mu, sd = t[cols].mean(), t[cols].std().replace(0, 1)
                X = ((t[cols] - mu) / sd).fillna(0)
                model = Ridge(alpha=5.0).fit(X, t["target"])
                Xn = ((f[cols] - mu) / sd).fillna(0)
                fbs = f.index != FCS
                predicted = predicted.copy()
                predicted[fbs] = model.predict(Xn[fbs])

        base_off, base_def = shrink * f["last_off"], shrink * f["last_def"]
        delta = predicted - (base_off - base_def)
        pri = pd.DataFrame({"off": base_off + delta / 2, "def": base_def - delta / 2}).fillna(0.0)
        pri["overall"] = pri["off"] - pri["def"]
        avg = last["avg"] if last else 27.0
        self._priors[season] = {"ratings": pri, "avg": avg, "features": f}
        return self._priors[season]

    # ------------------------------------------------------ in-season ratings
    def as_of(self, season, cutoff):
        """Ratings using only this season's games that kicked off before `cutoff`."""
        cutoff = pd.Timestamp(cutoff)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff
        pri = self.priors(season)
        hfa = self.hfa_for(season)
        played = self.completed[(self.completed["season"] == season) & (self.completed["start"] < cutoff)]
        prior_r = pri["ratings"]
        if len(played) == 0:
            r = prior_r.copy()
            r["games"] = 0
            return r, pri["avg"], hfa

        rows = _long_rows(played)
        p_off = rows["off"].map(prior_r["off"]).fillna(0.0)
        p_def = rows["def"].map(prior_r["def"]).fillna(0.0)
        y = rows["pts"] - pri["avg"] - p_off - p_def - rows["home"] * hfa / 2
        age = (cutoff - pd.to_datetime(rows["start"], utc=True)).dt.days.clip(lower=0)
        w = 0.5 ** (age / config.HALF_LIFE_DAYS)
        off, dfn, intercept, _ = _fit(rows, alpha=config.RATING_ALPHA, y=y, weight=w)

        teams = prior_r.index.union(off.index)
        r = pd.DataFrame(index=teams)
        r["off"] = prior_r["off"].reindex(teams).fillna(0) + off.reindex(teams).fillna(0)
        r["def"] = prior_r["def"].reindex(teams).fillna(0) + dfn.reindex(teams).fillna(0)
        r["overall"] = r["off"] - r["def"]
        counts = pd.concat([played["home_key"], played["away_key"]]).value_counts()
        r["games"] = counts.reindex(teams).fillna(0).astype(int)
        return r, pri["avg"] + intercept, hfa


def project(r, avg, hfa, home, away, neutral):
    """Projected (home points, away points)."""
    h = 0.0 if neutral else hfa / 2
    get = lambda t, col: r[col].get(t, r[col].get(FCS, 0.0))
    home_pts = avg + get(home, "off") + get(away, "def") + h
    away_pts = avg + get(away, "off") + get(home, "def") - h
    return home_pts, away_pts
