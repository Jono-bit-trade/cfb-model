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

# Ordered most to least serious; the injury words must sit right next to the player's name.
PATTERNS = [
    # recaps and "he's fine" stories: he played or is cleared, so not an injury flag
    ("available", r"expected to (play|start)|will start|set to start|cleared|full go|returns? (after|for|to|from)|returned|"
                  r"plays? through|played through|no limitations|probable|good to go|shuts? down .{0,30}speculation|"
                  r"not (seriously|significantly) (hurt|injured)|on track to play"),
    ("out", r"ruled out|will not play|won't play|will miss|to miss|out for (the )?(season|year|game|saturday|week)|"
            r"season[- ]ending|sidelined|undergo(es)? surgery|out indefinitely|not expected to play|done for the season|\bout\b"),
    ("doubtful", r"doubtful|unlikely to play"),
    ("questionable", r"questionable|game[- ]time decision|day[- ]to[- ]day|status (is )?(unclear|uncertain|in doubt)|"
                     r"uncertain|injur(y|ed)\b|concussion|limited in practice|banged up|left the game|"
                     r"knocked out of|exited|ankle|shoulder|knee|hamstring|wrist|elbow|"
                     r"injury update|dealing with"),
]
SEVERITY = {"out": 3, "doubtful": 2, "questionable": 1, "available": 0}
SKIP_SOURCES = re.compile(r"youtube|podcast|reddit", re.I)
NEAR = 70  # characters between the name and the injury words
PROPER = re.compile(r"\b[A-Z][a-z'’.-]+(?: [A-Z][a-z'’.-]+)+\b")  # "Hollywood Smothers", "Kirby Smart"


def _text(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def read_feed(url):
    """List of {title, link, published, desc, source} from an RSS feed (empty on any error)."""
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
            pub = pd.Timestamp(parsedate_to_datetime(it.findtext("pubDate"))).tz_convert("UTC")
        except Exception:
            pub = None
        src = it.find("source")
        source = _text(src.text) if src is not None and src.text else None
        # Google News puts " - Source" at the end of each title
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        items.append({"title": title, "link": (it.findtext("link") or "").strip(), "published": pub,
                      "desc": _text(it.findtext("description")), "source": source})
    return items


def _sentences(text):
    return [x for x in re.split(r"(?<=[.!?|:;])\s+|\s[-–—]\s", text) if x]


def classify(text, name, team):
    """Status suggested by an article, or None unless the injury words clearly refer to this quarterback.

    The words must be in the same sentence as his name, within a few words of it, with no other
    person's name in between (so "X ruled out adds pressure to <QB>" doesn't flag the QB)."""
    full, last = name, name.split()[-1]
    best = None
    for sent in _sentences(text):
        low = sent.lower()
        spots = [m.start() for m in re.finditer(re.escape(full.lower()), low)]
        if not spots and team.lower() in low:
            spots = [m.start() for m in re.finditer(rf"\b{re.escape(last.lower())}\b", low)]
        if not spots:
            continue
        others = [(m.start(), m.end()) for m in PROPER.finditer(sent)
                  if last not in m.group(0) and team.lower() not in m.group(0).lower()]
        for status, pat in PATTERNS:
            for m in re.finditer(pat, low):
                for at in spots:
                    lo, hi = sorted((at, m.start()))
                    if hi - lo > NEAR:
                        continue
                    if any(lo < a and b < hi for a, b in others):
                        continue  # someone else is named between them: the words are about that person
                    gap = abs(m.start() - at)
                    if any(min(abs(m.start() - a), abs(m.start() - b)) < gap for a, b in others):
                        continue  # another person's name is closer to the injury words than his is
                    if best is None or SEVERITY[status] > SEVERITY[best] and best != "available":
                        best = status
            if best == "available":
                return best  # a recap or "cleared" story outranks injury words in the same piece
    return best


def upcoming_starters(now):
    games = load_games()
    season = config.THIS_SEASON
    soon = games[(games["season"] == season) & ~games["completed"].astype(bool)
                 & (games["start"] >= now - pd.Timedelta(hours=6)) & (games["start"] <= now + pd.Timedelta(days=8))]
    starters = qb.current_starters(load("passers"), games, season)
    played = games[(games["season"] == season) & games["completed"].astype(bool)]
    last = pd.concat([played[["home", "start"]].rename(columns={"home": "team"}),
                      played[["away", "start"]].rename(columns={"away": "team"})]).groupby("team")["start"].max()
    out = []
    for r in soon.itertuples():
        for team in (r.home, r.away):
            if starters.get(team):
                out.append({"season": season, "week": int(r.week), "team": team, "qb": starters[team],
                            "last_game": last.get(team, pd.NaT)})
    return pd.DataFrame(out).drop_duplicates(["team"]) if out else pd.DataFrame(columns=["season", "week", "team", "qb", "last_game"])


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

    rows = []
    for r in teams.itertuples():
        # only news since the team's last game (older stories are about a game already played)
        cutoff = max(now - pd.Timedelta(days=MAX_AGE_DAYS), r.last_game + pd.Timedelta(hours=4)) \
            if not pd.isna(r.last_game) else now - pd.Timedelta(days=MAX_AGE_DAYS)
        best = None
        for it in general + searched.get(r.team, []):
            if it["published"] is not None and it["published"] < cutoff:
                continue
            if it["source"] and SKIP_SOURCES.search(it["source"]) or SKIP_SOURCES.search(it["title"]):
                continue
            st = classify(it["title"], r.qb, r.team) or classify(it["desc"], r.qb, r.team)
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
