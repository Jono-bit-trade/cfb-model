"""Step 8: this week's predictions, the season record, and ratings for the site.

    python src/predict.py

Writes docs/data/predictions.json, results.json and ratings.json, and keeps
data/predictions_log.csv: the last prediction made before each kickoff, which
is what the published record is graded on.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import features  # noqa: E402
import qb  # noqa: E402
import splits  # noqa: E402
import weather  # noqa: E402
from common import DATA, SITE_DATA, load, load_file, load_games, write_json  # noqa: E402
import grading  # noqa: E402
from backtest import update_history_index  # noqa: E402
from model import Model, ats_edge, split_score  # noqa: E402
from ratings import FCS, RatingEngine, team_keys  # noqa: E402
from features import CENTRAL  # noqa: E402

LOG = os.path.join(DATA, "predictions_log.csv")


def clean(v):
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else round(float(v), 1)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def bet_early(edge, line_move):
    """Strong edge, and the line is still close to where it opened.

    In 2023-2025, model picks with 5+ point edges went 54.6% at the closing line when
    the line hadn't moved, ~49% when it had already moved toward the model (the edge
    was gone) and ~51% when it moved against the model (the money disagreed).
    """
    if np.isnan(edge) or abs(edge) < config.STRONG_EDGE:
        return False
    toward_model = 0.0 if np.isnan(line_move) else line_move * np.sign(edge)
    return bool(-config.BET_EARLY_MAX_MOVE < toward_model < 0.5)


def locked_pick(log, r):
    """The early pick recorded for this game, if it has locked."""
    if log is None or not len(log) or r.game_id not in log.index or "lock_at" not in log:
        return None
    x = log.loc[r.game_id]
    if isinstance(x, pd.DataFrame):
        x = x.iloc[-1]
    if not isinstance(x.get("lock_at"), str) or pd.isna(x.get("lock_spread")):
        return None
    edge = x["lock_pred_margin"] + x["lock_spread"]
    team, line = (r.home, x["lock_spread"]) if edge > 0 else (r.away, -x["lock_spread"])
    return {"text": fmt_line(team, line), "at": x["lock_at"], "edge": clean(abs(edge)),
            "line_text": fmt_line(r.home, x["lock_spread"])}


def text(v):
    return v if isinstance(v, str) and v.strip() else None


def fmt_line(team, line):
    if line is None or np.isnan(line):
        return None
    if abs(line) < 0.05:
        return f"{team} PK"
    return f"{team} {'+' if line > 0 else '−'}{abs(line):.1f}"


def records(games, season):
    done = games[(games["season"] == season) & games["completed"].astype(bool) & games["home_pts"].notna()]
    rec = {}
    for gm in done.itertuples():
        for team, won in ((gm.home, gm.home_pts > gm.away_pts), (gm.away, gm.away_pts > gm.home_pts)):
            w, l = rec.get(team, (0, 0))
            rec[team] = (w + int(won), l + int(not won))
    return {t: f"{w}–{l}" for t, (w, l) in rec.items()}


def upcoming_games(games, now):
    future = games[~games["completed"].astype(bool) & (games["start"] > now - pd.Timedelta(hours=5))]
    window = future[future["start"] <= now + pd.Timedelta(days=8)]
    if window.empty and len(future):  # offseason or a bye week: show the next slate
        first = future["start"].min()
        window = future[future["start"] <= first + pd.Timedelta(days=7)]
    return window


def main(now=None):
    now = pd.Timestamp(now) if now else pd.Timestamp.now(tz="UTC")
    now = now.tz_localize("UTC") if now.tzinfo is None else now
    games = load_games()
    venues = load_file("venues.csv")
    upcoming = upcoming_games(games, now)
    print(f"{len(upcoming)} upcoming games")

    weather.update(games, venues, upcoming_ids=set(upcoming["game_id"]))

    engine = RatingEngine(games)
    seasons = sorted(s for s in games["season"].unique() if s > config.FIRST_SEASON)
    table = features.build(games, engine, seasons=seasons, live_cutoff=now)
    hist = table[table["completed"].astype(bool) & table["actual_margin"].notna()]
    model = Model().fit(hist)
    print(f"model trained on {model.n_train} games" + (f"; inputs with no data left out: {', '.join(model.dropped)}" if model.dropped else ""))

    rows = table[table["game_id"].isin(upcoming["game_id"])].copy()
    season = config.THIS_SEASON
    status = qb.load_status()
    starters = qb.current_starters(load("passers"), games, season)
    rec = records(games, season)

    # Current-week QB status overrides the box-score flag.
    alt_rows = []
    for i, r in rows.iterrows():
        for s in ("home", "away"):
            st, name, note = qb.status_for(r["season"], r["week"], r[s], status)
            rows.at[i, f"qb_status_{s}"] = st
            rows.at[i, f"qb_name_{s}"] = name or starters.get(r[s])
            rows.at[i, f"qb_note_{s}"] = note
            rows.at[i, f"qb_out_{s}"] = 1.0 if st in qb.OUT_STATUSES else 0.0
    if len(rows):
        rows["pred_margin"], rows["pred_total"] = model.predict(rows)
        # Questionable QB: also project the game as if he sits.
        for s in ("home", "away"):
            q = rows[f"qb_status_{s}"] == "questionable"
            if q.any():
                alt = rows[q].copy()
                alt[f"qb_out_{s}"] = 1.0
                m, t = model.predict(alt)
                rows.loc[q, f"alt_margin_{s}"], rows.loc[q, f"alt_total_{s}"] = m, t
        rows = grading.grade(rows)
        rows["line_move"] = rows["spread_open"] - rows["spread"]
        rows = splits.attach(rows)

    backfill_log(table, season)
    if len(rows):
        update_log(rows, now)  # records early (locked) picks before the board is written
        save_line_snapshot(rows, now)
    log_now = pd.read_csv(LOG).set_index("game_id") if os.path.exists(LOG) else pd.DataFrame()
    out_games = []
    for r in rows.sort_values("start").itertuples():
        d = r._asdict()
        hp, ap = split_score(r.pred_margin, r.pred_total)
        edge = ats_edge(r.pred_margin, r.spread)
        pick_side = None if np.isnan(edge) or abs(edge) < config.MIN_EDGE else ("home" if edge > 0 else "away")
        pick_team = r.home if pick_side == "home" else r.away if pick_side == "away" else None
        pick_line = (r.spread if pick_side == "home" else -r.spread) if pick_side else None
        total_pick = None
        if not np.isnan(r.total):
            total_pick = f"{'Over' if r.pred_total > r.total else 'Under'} {r.total:.1f}"
        qbinfo = {}
        for s in ("home", "away"):
            alt = None
            am = d.get(f"alt_margin_{s}")
            if am is not None and not (isinstance(am, float) and np.isnan(am)):
                ah, aa = split_score(am, d[f"alt_total_{s}"])
                alt = {"home": clean(ah), "away": clean(aa)}
            qbinfo[s] = {"name": text(d.get(f"qb_name_{s}")), "status": text(d.get(f"qb_status_{s}")),
                         "note": text(d.get(f"qb_note_{s}")), "if_out": alt}
        out_games.append({
            "id": int(r.game_id), "kickoff": r.start.isoformat(), "time_tbd": bool(r.start_tbd),
            "week": int(r.week) if not pd.isna(r.week) else None, "venue": text(r.venue), "neutral": bool(r.neutral),
            "home": {"team": r.home, "record": rec.get(r.home, "0–0"), "proj": clean(hp),
                     "rating": clean(r.home_overall), "conf": r.home_conf},
            "away": {"team": r.away, "record": rec.get(r.away, "0–0"), "proj": clean(ap),
                     "rating": clean(r.away_overall), "conf": r.away_conf},
            "market": {"spread": clean(r.spread), "spread_open": clean(r.spread_open), "total": clean(r.total),
                       "line_move": clean(r.spread_open - r.spread) if not (np.isnan(r.spread_open) or np.isnan(r.spread)) else None,
                       "total_open": clean(r.total_open), "provider": r.provider if isinstance(r.provider, str) else None,
                       "line_text": fmt_line(r.home, r.spread)},
            "model": {"margin": clean(r.pred_margin), "total": clean(r.pred_total),
                      "line_text": fmt_line(r.home, -r.pred_margin), "rating_margin": clean(r.rating_margin)},
            "moneyline": {"home": clean(r.home_ml), "away": clean(r.away_ml),
                          "win_chance": clean(round(100 * r.p_pick, 1)),
                          "value": {"team": r.val_pick, "odds": clean(r.val_odds),
                                    "gap": clean(round(100 * r.val_gap, 1))} if isinstance(r.val_pick, str) and r.val_pick else None},
            "splits": None if np.isnan(r.bets_home_pct) else {
                "bets_home": clean(r.bets_home_pct), "money_home": clean(r.money_home_pct),
                "bets_over": clean(r.bets_over_pct), "money_over": clean(r.money_over_pct),
                "sharp": r.home if r.sharp_side == "home" else r.away if r.sharp_side == "away" else None,
                "sharp_total": r.sharp_total or None,
                "public": r.home if r.public_side == "home" else r.away if r.public_side == "away" else None,
                "rlm": r.home if r.rlm_side == "home" else r.away if r.rlm_side == "away" else None},
            "bet_early": bet_early(edge, r.spread_open - r.spread if not (np.isnan(r.spread_open) or np.isnan(r.spread)) else np.nan),
            "locked": locked_pick(log_now, r),
            "edge": clean(abs(edge)) if not np.isnan(edge) else None,
            "pick": {"team": pick_team, "line": clean(pick_line), "text": fmt_line(pick_team, pick_line) if pick_team else None,
                     "strong": bool(pick_side and abs(edge) >= config.STRONG_EDGE)},
            "winner": r.home if r.pred_margin > 0 else r.away,
            "total_pick": total_pick,
            "weather": {"dome": bool(r.dome == 1), "temp": clean(r.temp), "wind": clean(r.wind), "gust": clean(r.gust),
                        "rain": clean(r.rain), "kind": r.wx_kind if isinstance(r.wx_kind, str) else None,
                        "updated": r.wx_time if isinstance(r.wx_time, str) else None},
            "qb": qbinfo,
        })

    weeks = rows["week"].dropna().astype(int)
    label = f"Week {weeks.mode().iloc[0]}" if len(weeks) else "Upcoming games"
    if len(rows) and (rows["season_type"] == "postseason").all():
        label = "Bowl season"
    write_json({"updated": now, "season": season, "label": label, "games": out_games,
                "settings": {"blend_w": config.BLEND_W, "strong_edge": config.STRONG_EDGE}},
               SITE_DATA, "predictions.json")

    write_results(games, season, now)
    write_ratings(engine, games, season, now, rec)
    print(f"wrote {len(out_games)} predictions")


def backfill_log(table, season):
    """First launch mid-season: replay the model on this season's finished games.

    Each replayed game uses ratings from before its week and a model trained on
    earlier seasons only, and is marked made_at = "backfill" so the site can say so.
    """
    log = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame(columns=["season"])
    if (log["season"] == season).any():
        return
    done = table[(table["season"] == season) & table["completed"].astype(bool) & table["actual_margin"].notna()].copy()
    train = table[(table["season"] < season) & table["actual_margin"].notna()]
    if done.empty or train["season"].nunique() < 2:
        return
    done["pred_margin"], done["pred_total"] = Model().fit(train).predict(done)
    new = pd.DataFrame({
        "game_id": done["game_id"], "season": done["season"], "week": done["week"],
        "start": done["start"].map(lambda x: x.isoformat()), "home": done["home"], "away": done["away"],
        "pred_margin": done["pred_margin"].round(2), "pred_total": done["pred_total"].round(2),
        "spread": done["spread"], "total": done["total"], "home_ml": done["home_ml"], "away_ml": done["away_ml"],
        "spread_open": done["spread_open"], "total_open": done["total_open"],
        "made_at": "backfill"})
    pd.concat([log, new], ignore_index=True).to_csv(LOG, index=False)
    print(f"backfilled {len(new)} finished games of {season} into the log")


LOCK_COLS = ["lock_pred_margin", "lock_pred_total", "lock_spread", "lock_total", "lock_home_ml", "lock_away_ml", "lock_at"]


def in_lock_window(now, start):
    """Early picks lock on the first run after the previous weekend: Sunday to Tuesday
    (Central), or any time once kickoff is under 3 days away."""
    day = now.tz_convert(CENTRAL).weekday()  # Monday = 0 ... Sunday = 6
    return day in (6, 0, 1) or (start - now) < pd.Timedelta(days=3)


LINE_HISTORY = os.path.join(DATA, "line_history.csv")


def save_line_snapshot(rows, now):
    """Every run appends the current line for each upcoming game, so picks can later be
    graded at the line on any day (for example the Wednesday line)."""
    if not len(rows):
        return
    snap = pd.DataFrame({"game_id": rows["game_id"], "at": now.isoformat(),
                         "weekday": now.tz_convert(CENTRAL).strftime("%a"),
                         "spread": rows["spread"], "total": rows["total"], "home_ml": rows["home_ml"],
                         "away_ml": rows["away_ml"], "pred_margin": rows["pred_margin"].round(2),
                         "pred_total": rows["pred_total"].round(2)})
    snap = snap[pd.to_datetime(rows["start"], utc=True) > now]
    header = not os.path.exists(LINE_HISTORY)
    snap.to_csv(LINE_HISTORY, mode="a", header=header, index=False)


def update_log(rows, now):
    """Latest pre-kickoff projection per game, plus the early (locked) pick, which never changes."""
    log = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame()
    for c in LOCK_COLS:
        if len(log) and c not in log:
            log[c] = np.nan
    if len(rows):
        new = pd.DataFrame({
            "game_id": rows["game_id"], "season": rows["season"], "week": rows["week"],
            "start": rows["start"].map(lambda x: x.isoformat()), "home": rows["home"], "away": rows["away"],
            "pred_margin": rows["pred_margin"].round(2), "pred_total": rows["pred_total"].round(2),
            "spread": rows["spread"], "total": rows["total"], "home_ml": rows["home_ml"], "away_ml": rows["away_ml"],
            "spread_open": rows["spread_open"], "total_open": rows["total_open"],
            "bets_home_pct": rows["bets_home_pct"], "money_home_pct": rows["money_home_pct"],
            "bets_over_pct": rows["bets_over_pct"], "money_over_pct": rows["money_over_pct"],
            "made_at": now.isoformat()})
        new = new[pd.to_datetime(new["start"], utc=True, format="mixed") > now]  # never change a game that started
        # Keep any lock already recorded; otherwise lock now if this run is in the lock window.
        old_locks = log.set_index("game_id")[LOCK_COLS] if len(log) and "lock_at" in log else pd.DataFrame(columns=LOCK_COLS)
        old_locks = old_locks[old_locks["lock_at"].notna()]
        starts = pd.to_datetime(new["start"], utc=True, format="mixed")
        can_lock = np.array([in_lock_window(now, s) for s in starts]) & new["spread"].notna().to_numpy()
        for c in LOCK_COLS:
            new[c] = new["game_id"].map(old_locks[c]) if len(old_locks) else np.nan
        fresh = new["lock_at"].isna().to_numpy() & can_lock
        for src, dst in (("pred_margin", "lock_pred_margin"), ("pred_total", "lock_pred_total"), ("spread", "lock_spread"),
                         ("total", "lock_total"), ("home_ml", "lock_home_ml"), ("away_ml", "lock_away_ml")):
            new.loc[fresh, dst] = new.loc[fresh, src]
        new["lock_at"] = new["lock_at"].astype(object)
        new.loc[fresh, "lock_at"] = now.isoformat()
        if len(log):
            log = log[~log["game_id"].isin(new["game_id"])]
        log = pd.concat([log, new], ignore_index=True)
    if len(log):
        log.to_csv(LOG, index=False)


def _locked_view(res):
    """The same games, priced at the early (locked) pick instead of the last one before kickoff.

    Games replayed before launch use the opening line; a live game that never locked
    (first seen inside 3 days of kickoff) falls back to its last line.
    """
    v = res.copy()
    for c in LOCK_COLS:
        if c not in v:
            v[c] = np.nan
    backfill = v["made_at"] == "backfill"
    v["lock_spread"] = v["lock_spread"].fillna(v["spread_open"].where(backfill)).fillna(v["spread"])
    v["lock_total"] = v["lock_total"].fillna(v["total_open"].where(backfill)).fillna(v["total"])
    v["lock_pred_margin"] = v["lock_pred_margin"].fillna(v["pred_margin"])
    v["lock_pred_total"] = v["lock_pred_total"].fillna(v["pred_total"])
    v["lock_home_ml"] = v["lock_home_ml"].fillna(v["home_ml"])
    v["lock_away_ml"] = v["lock_away_ml"].fillna(v["away_ml"])
    return v.assign(spread=v["lock_spread"], total=v["lock_total"], pred_margin=v["lock_pred_margin"],
                    pred_total=v["lock_pred_total"], home_ml=v["lock_home_ml"], away_ml=v["lock_away_ml"])


def write_results(games, season, now=None):
    """Grade every logged pick (winner, moneyline, spread, over/under) at both the closing
    line and the early locked line, for the History page and the board's final scores."""
    now = now or pd.Timestamp.now(tz="UTC")
    if not os.path.exists(LOG):
        write_json({"season": season, "summary": {}, "games": []}, SITE_DATA, "results.json")
        write_json({"games": []}, SITE_DATA, "finals.json")
        return
    log = pd.read_csv(LOG)
    for c in ("home_ml", "away_ml", "spread_open", "total_open"):
        if c not in log:
            log[c] = np.nan
    done = games[games["completed"].astype(bool) & games["home_pts"].notna()][
        ["game_id", "home_pts", "away_pts", "season_type"]]
    res = log.merge(done, on="game_id")
    lines = load("lines")  # opening lines for older log rows that didn't store them
    if len(lines):
        first = lines.drop_duplicates("game_id").set_index("game_id")
        for c in ("spread_open", "total_open"):
            if c in first:
                res[c] = res[c].fillna(res["game_id"].map(first[c]))
    res["actual_margin"] = res["home_pts"] - res["away_pts"]
    res["actual_total"] = res["home_pts"] + res["away_pts"]
    res["backfill"] = res["made_at"] == "backfill"
    g = grading.grade(res)
    lk = grading.grade(_locked_view(res)).set_index("game_id")
    cur = g[g["season"] == season]
    cur_lk = lk[lk["season"] == season]
    summary = {"season": grading.summary(cur), "season_strong": grading.summary(cur, config.STRONG_EDGE, config.STRONG_EDGE),
               "season_locked": grading.summary(cur_lk), "season_locked_strong":
                   grading.summary(cur_lk, config.STRONG_EDGE, config.STRONG_EDGE),
               "all_time": grading.summary(g)}

    rows = grading.history_rows(cur)
    for r in rows:  # attach the locked-line grading to each game
        if r["id"] in cur_lk.index:
            x = cur_lk.loc[r["id"]]
            f = lambda v, d=1: None if pd.isna(v) else round(float(v), d)
            r.update({"lk_sp": f(x["spread"]), "lk_pm": f(x["pred_margin"]), "lk_tot": f(x["total"]),
                      "lk_pt": f(x["pred_total"]), "lk_ats": f(x["ats"]), "lk_ou": f(x["ou"]),
                      "lk_live": isinstance(x.get("lock_at"), str) and x.get("lock_at") != "backfill"})
    write_json({"season": season, "summary": summary, "games": rows[::-1],
                "backfilled_games": int(cur["backfill"].sum()), "strong_edge": config.STRONG_EDGE},
               SITE_DATA, "results.json")
    write_json({"season": season, "source": "live", "games": rows}, SITE_DATA, "history", f"{season}.json")
    update_history_index([season])

    # Final scores for the board: games that ended in the last 4 days.
    recent = [r for r in rows if pd.Timestamp(r["date"]) > now - pd.Timedelta(days=4)]
    write_json({"updated": now, "games": sorted(recent, key=lambda r: r["date"], reverse=True)}, SITE_DATA, "finals.json")


def write_ratings(engine, games, season, now, rec):
    r, avg, hfa = engine.as_of(season, now)
    g = team_keys(games[games["season"] == season])
    conf = pd.concat([g[["home_key", "home_conf"]].set_axis(["team", "conf"], axis=1),
                      g[["away_key", "away_conf"]].set_axis(["team", "conf"], axis=1)]).dropna().drop_duplicates("team")
    conf = conf.set_index("team")["conf"]
    r = r.drop(index=FCS, errors="ignore")
    r = r[r.index.isin(conf.index)]
    teams = []
    for team, x in r.sort_values("overall", ascending=False).iterrows():
        teams.append({"team": team, "conf": conf.get(team), "record": rec.get(team, "0–0"),
                      "overall": round(x["overall"], 2), "offense": round(avg + x["off"], 2),
                      "defense": round(avg + x["def"], 2), "games": int(x["games"])})
    write_json({"updated": now, "season": season, "avg_points": round(avg, 2), "home_edge": round(hfa, 2),
                "teams": teams}, SITE_DATA, "ratings.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", help="pretend it is this time (ISO), for testing")
    main(ap.parse_args().now)
