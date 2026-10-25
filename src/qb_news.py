"""Scan news feeds for injury reports about each team's starting quarterback.

Reads public RSS feeds (ESPN, CBS Sports, Yahoo) plus a Google News search for
each upcoming starter's name, and keeps articles from the last few days that
name the starter together with an injury word (questionable, doubtful, ruled
out, ...). The result is only a suggestion: the website shows the headline and
link, and nothing changes in the model until you confirm it.

    python src/qb_news.py            # writes data/qb_news.csv

Feeds are read once per run (about one request per upcoming starter), the way a
news reader would.
"""
import html
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
import qb  # noqa: E402
from common import DATA, load, load_games  # noqa: E402

OUT = os.path.join(DATA, "qb_news.csv")
COLS = ["season", "week", "team", "qb", "suggested", "headline", "link", "source", "published", "fetched"]
FEEDS = {
    "ESPN": "https://www.espn.com/espn/rss/ncf/news",
    "CBS Sports": "https://www.cbssports.com/rss/headlines/college-football/",
    "Yahoo Sports": "https://sports.yahoo.com/college-football/rss/",
}
GOOGLE = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
HEADERS = {"User-Agent": "Mozilla/5.0 (personal college football model; RSS reader)"}
MAX_AGE_DAYS = 6
WORKERS = 4

# Ordered most to least serious; the first match near the player's name wins.
PATTERNS = [
    ("out", r"ruled out|will not play|won't play|will miss|to miss|out for (the )?(season|year|game|saturday|week)|"
            r"season[- ]ending|sidelined|undergo(es)? surgery|out indefinitely|not expected to play|done for the season"),
    ("doubtful", r"doubtful|unlikely to play"),
    ("questionable", r"questionable|game[- ]time decision|day[- ]to[- ]day|status (is )?(unclear|uncertain|in doubt)|"
                     r"uncertain|injur(y|ed|ies)|concussion|limited in practice|banged up|left the game|"
                     r"knocked out of|exited|hurt|ankle|shoulder|knee|hamstring|wrist|elbow"),
    ("available", r"expected to (play|start)|will start|cleared to (play|return)|full go|returns? to practice|"
                  r"no limitations|probable|good to go"),
]
SEVERITY = {"out": 3, "doubtful": 2, "questionable": 1, "available": 0}


def _text(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def read_feed(url):
    """List of {title, link, published, text, source} from an RSS feed (empty on any error)."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception as e:
        print(f"  ! {url[:80]}: {e}")
        return []
    items = []
    for it in root.iter("item"):
        title = _text(it.findtext("title"))
        try:
            pub = parsedate_to_datetime(it.findtext("pubDate")).astimezone(tz=None)
            pub = pd.Timestamp(pub).tz_convert("UTC")
        except Exception:
            pub = None
        src = it.find("source")
        items.append({"title": title, "link": (it.findtext("link") or "").strip(), "published": pub,
                      "text": f"{title}. {_text(it.findtext('description'))}",
                      "source": _text(src.text) if src is not None and src.text else None})
    return items


def classify(text, name, team):
    """Status suggested by an article, or None if it doesn't clearly concern this quarterback."""
    low = text.lower()
    full, last = name.lower(), name.split()[-1].lower()
    if full in low:
        anchor = low.index(full)
    elif re.search(rf"\b{re.escape(last)}\b", low) and team.lower() in low:
        anchor = re.search(rf"\b{re.escape(last)}\b", low).start()
    else:
        return None
    window = low[max(0, anchor - 160): anchor + 260]  # words near the name, not elsewhere in a roundup
    for status, pat in PATTERNS:
        if re.search(pat, window):
            return status
    return None


def upcoming_starters(now):
    games = load_games()
    season = config.THIS_SEASON
    soon = games[(games["season"] == season) & ~games["completed"].astype(bool)
                 & (games["start"] >= now - pd.Timedelta(hours=6)) & (games["start"] <= now + pd.Timedelta(days=8))]
    starters = qb.current_starters(load("passers"), games, season)
    out = []
    for r in soon.itertuples():
        for team in (r.home, r.away):
            if starters.get(team):
                out.append({"season": season, "week": int(r.week), "team": team, "qb": starters[team]})
    return pd.DataFrame(out).drop_duplicates(["team"]) if out else pd.DataFrame(columns=["season", "week", "team", "qb"])


def main(now=None):
    if os.environ.get("CFB_OFFLINE"):
        print("offline: skipping the QB news check")
        return
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    teams = upcoming_starters(now)
    if teams.empty:
        print("no upcoming starters to check")
        pd.DataFrame(columns=COLS).to_csv(OUT, index=False)
        return
    t0 = time.time()
    general = []
    for src, url in FEEDS.items():
        for it in read_feed(url):
            it["source"] = it["source"] or src
            general.append(it)

    def search(row):
        q = f'"{row.qb}" {row.team} (injury OR questionable OR doubtful OR "ruled out" OR status) when:{MAX_AGE_DAYS}d'
        time.sleep(0.3)
        return row.team, read_feed(GOOGLE.format(q=quote(q)))

    with ThreadPoolExecutor(WORKERS) as ex:
        searched = dict(ex.map(search, teams.itertuples()))

    cutoff = now - pd.Timedelta(days=MAX_AGE_DAYS)
    rows = []
    for r in teams.itertuples():
        best = None
        for it in general + searched.get(r.team, []):
            if it["published"] is not None and it["published"] < cutoff:
                continue
            st = classify(it["text"], r.qb, r.team)
            if st is None:
                continue
            key = (it["published"] or cutoff, SEVERITY[st])  # newest first, then most serious
            if best is None or key > best[0]:
                best = (key, st, it)
        if best:
            _, st, it = best
            rows.append({"season": r.season, "week": r.week, "team": r.team, "qb": r.qb, "suggested": st,
                         "headline": it["title"][:240], "link": it["link"], "source": it["source"],
                         "published": it["published"].isoformat() if it["published"] is not None else None,
                         "fetched": now.isoformat()})
    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(OUT, index=False)
    flagged = df[df["suggested"] != "available"]
    print(f"QB news: checked {len(teams)} starters in {time.time() - t0:.0f}s; {len(flagged)} possible injuries, "
          f"{len(df) - len(flagged)} 'expected to play' reports")
    for r in flagged.itertuples():
        print(f"  {r.team}: {r.qb} -> {r.suggested} | {r.headline[:90]}")


def load_news():
    if not os.path.exists(OUT):
        return pd.DataFrame(columns=COLS)
    try:
        return pd.read_csv(OUT)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=COLS)


if __name__ == "__main__":
    main()
