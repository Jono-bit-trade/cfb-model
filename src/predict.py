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
import weather  # noqa: E402
from common import DATA, SITE_DATA, load, load_file, load_games, write_json  # noqa: E402
import grading  # noqa: E402
from backtest import update_history_index  # noqa: E402
from model import Model, ats_edge, split_score  # noqa: E402
from ratings import FCS, RatingEngine, team_keys  # noqa: E402

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
                                    "gap": clean(round(100 * r.val_gap, 1))} if r.val_pick else None},
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

    backfill_log(table, season)
    update_log(rows, now)
    write_results(games, season)
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


def update_log(rows, now):
    log = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame()
    if len(rows):
        new = pd.DataFrame({
            "game_id": rows["game_id"], "season": rows["season"], "week": rows["week"],
            "start": rows["start"].map(lambda x: x.isoformat()), "home": rows["home"], "away": rows["away"],
            "pred_margin": rows["pred_margin"].round(2), "pred_total": rows["pred_total"].round(2),
            "spread": rows["spread"], "total": rows["total"], "home_ml": rows["home_ml"], "away_ml": rows["away_ml"],
            "spread_open": rows["spread_open"], "total_open": rows["total_open"],
            "made_at": now.isoformat()})
        new = new[pd.to_datetime(new["start"], utc=True, format="mixed") > now]  # never change a game that started
        if len(log):
            log = log[~log["game_id"].isin(new["game_id"])]
        log = pd.concat([log, new], ignore_index=True)
    if len(log):
        log.to_csv(LOG, index=False)


def write_results(games, season):
    """Grade every logged pick (winner, moneyline, spread, over/under) for the Record and History pages."""
    if not os.path.exists(LOG):
        write_json({"season": season, "summary": {}, "games": []}, SITE_DATA, "results.json")
        return
    log = pd.read_csv(LOG)
    for c in ("home_ml", "away_ml"):
        if c not in log:
            log[c] = np.nan
    done = games[games["completed"].astype(bool) & games["home_pts"].notna()][
        ["game_id", "home_pts", "away_pts", "season_type"]]
    res = log.merge(done, on="game_id")
    lines = load("lines")  # opening lines for older log rows that didn't store them
    for c in ("spread_open", "total_open"):
        if len(lines) and c in lines:
            opens = lines.drop_duplicates("game_id").set_index("game_id")[c]
            res[c] = res[c].fillna(res["game_id"].map(opens)) if c in res else res["game_id"].map(opens)
    res["actual_margin"] = res["home_pts"] - res["away_pts"]
    res["actual_total"] = res["home_pts"] + res["away_pts"]
    res["backfill"] = res["made_at"] == "backfill"
    g = grading.grade(res)
    cur = g[g["season"] == season]
    summary = {"season": grading.summary(cur), "season_strong": grading.summary(cur, config.STRONG_EDGE, config.STRONG_EDGE),
               "all_time": grading.summary(g)}
    rows = grading.history_rows(cur)[::-1]  # newest first
    write_json({"season": season, "summary": summary, "games": rows,
                "backfilled_games": int(cur["backfill"].sum()), "strong_edge": config.STRONG_EDGE},
               SITE_DATA, "results.json")
    # This season also appears on the History page, graded the same way as past seasons.
    write_json({"season": season, "source": "live", "games": grading.history_rows(cur)},
               SITE_DATA, "history", f"{season}.json")
    update_history_index([season])


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
