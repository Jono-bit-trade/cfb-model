"""Checks that the CFBD parsers read the API's JSON shape correctly.

    python tests/test_parsers.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import get_data as gd  # noqa: E402

GAME = {"id": 401, "season": 2025, "week": 3, "seasonType": "regular", "startDate": "2025-09-13T19:30:00.000Z",
        "startTimeTBD": False, "completed": True, "neutralSite": False, "conferenceGame": True, "venueId": 3958,
        "venue": "Memorial Stadium", "homeTeam": "Oklahoma", "homeClassification": "fbs", "homeConference": "SEC",
        "homePoints": 24, "awayTeam": "Temple", "awayClassification": "fbs", "awayConference": "American Athletic",
        "awayPoints": 3}
LINES = [{"id": 401, "homeTeam": "Oklahoma", "awayTeam": "Temple", "lines": [
    {"provider": "Bovada", "spread": -21.5, "spreadOpen": -20, "overUnder": 50.5, "overUnderOpen": 51,
     "homeMoneyline": -1800, "awayMoneyline": 950},
    {"provider": "DraftKings", "spread": -22.0, "spreadOpen": -21, "overUnder": 49.5, "overUnderOpen": 50,
     "homeMoneyline": None, "awayMoneyline": None}]}]
VENUE = [{"id": 3958, "name": "Memorial Stadium", "city": "Norman", "state": "OK", "timezone": "America/Chicago",
          "latitude": 35.2059, "longitude": -97.4423, "elevation": 357.5, "dome": False, "grass": True}]
VENUE_OLD = [{"id": 1, "name": "X", "location": {"x": -97.4, "y": 35.2}, "dome": True}]
PPA = [{"gameId": 401, "season": 2025, "week": 3, "team": "Oklahoma", "opponent": "Temple",
        "offense": {"overall": 0.31, "passing": 0.4}, "defense": {"overall": -0.05}}]
ADV = [{"gameId": 401, "team": "Oklahoma", "opponent": "Temple",
        "offense": {"plays": 72, "successRate": 0.48, "explosiveness": 1.31, "pointsPerOpportunity": 4.6,
                    "havoc": {"total": 0.12}},
        "defense": {"plays": 61, "successRate": 0.33, "explosiveness": 1.1, "pointsPerOpportunity": 2.0,
                    "havoc": {"total": 0.21}}}]
BOX = [{"id": 401, "teams": [{"team": "Oklahoma", "homeAway": "home", "points": 24, "stats": [
    {"category": "turnovers", "stat": "1"}, {"category": "fumblesLost", "stat": "0"},
    {"category": "fumblesRecovered", "stat": "2"}, {"category": "totalPenaltiesYards", "stat": "7-65"}]}]}]
PLAYERS = [{"id": 401, "teams": [{"team": "Oklahoma", "categories": [
    {"name": "passing", "types": [
        {"name": "C/ATT", "athletes": [{"id": "1", "name": "John Mateer", "stat": "21/30"},
                                       {"id": "2", "name": "Backup QB", "stat": "1/2"}]},
        {"name": "YDS", "athletes": [{"id": "1", "name": "John Mateer", "stat": "250"}]}]},
    {"name": "rushing", "types": []}]}]}]
PORTAL = [{"season": 2025, "firstName": "A", "lastName": "B", "origin": "Temple", "destination": "Oklahoma", "stars": 4},
          {"season": 2025, "firstName": "C", "lastName": "D", "origin": "Oklahoma", "destination": None, "stars": None}]
COACHES = [{"firstName": "Brent", "lastName": "Venables", "seasons": [{"school": "Oklahoma", "year": 2025, "games": 13}]}]


def main():
    g = gd.parse_games([GAME], 2025).iloc[0]
    assert (g.home, g.away, g.home_pts, g.away_pts, g.venue_id) == ("Oklahoma", "Temple", 24, 3, 3958)
    assert g.completed and g.conf_game and not g.neutral and g.home_class == "fbs"

    snake = {"id": 9, "home_team": "A", "away_team": "B", "home_points": 7, "away_points": 3, "start_date": "x"}
    assert gd.parse_games([snake], 2019).iloc[0].home == "A", "old snake_case names still work"

    line = gd.parse_lines(LINES, 2025).iloc[0]
    assert line.provider == "DraftKings" and line.spread == -22.0 and line.spread_open == -21 and line.total == 49.5
    assert line.home_ml == -1800 and line.away_ml == 950, "moneyline falls back to the next book that has one"

    v = gd.parse_venues(VENUE).iloc[0]
    assert v.lat == 35.2059 and v.timezone == "America/Chicago" and not v.dome
    vo = gd.parse_venues(VENUE_OLD).iloc[0]
    assert vo.lat == 35.2 and vo.lon == -97.4 and vo.dome

    p = gd.parse_ppa(PPA, 2025).iloc[0]
    assert p.off_ppa == 0.31 and p.def_ppa == -0.05

    a = gd.parse_adv(ADV, 2025).iloc[0]
    assert a.off_success == 0.48 and a.def_havoc == 0.21 and a.off_ppo == 4.6

    b = gd.parse_teambox(BOX, 2025).iloc[0]
    assert b.turnovers == 1 and b.fumbles_rec == 2 and b.pen_yards == 65

    q = gd.parse_passers(PLAYERS, 2025)
    assert list(q.att) == [30, 2] and q.iloc[0].player == "John Mateer"

    port = gd.parse_portal(PORTAL, 2025).set_index("team")
    assert port.loc["Oklahoma", "portal_net"] == 4 - 2 and port.loc["Temple", "portal_net"] == -4

    c = gd.parse_coaches(COACHES, 2025).iloc[0]
    assert c.team == "Oklahoma" and c.coach == "Brent Venables"
    wx = gd.parse_cfbd_weather([{"id": 401, "gameIndoors": False, "temperature": 48.2, "windSpeed": 14.0,
                                  "precipitation": 0.1, "weatherCondition": "Rain"}], 2025).iloc[0]
    assert wx.temp == 48.2 and wx.wind == 14.0 and wx.gust == 14.0 and not wx.indoors
    ac = gd.parse_adv_clean([{"gameId": 401, "team": "Oklahoma", "opponent": "Temple",
                               "offense": {"ppa": 0.31, "successRate": 0.47, "explosiveness": 1.2, "lineYardsAverage": 3.1,
                                           "stuffRate": 0.15, "passingPlays": {"ppa": 0.4}, "rushingPlays": {"ppa": 0.2},
                                           "passingDowns": {"successRate": 0.3}},
                               "defense": {"ppa": -0.1, "successRate": 0.3}}], 2025).iloc[0]
    assert ac.off_pass_ppa == 0.4 and ac.off_line_yards == 3.1 and ac.def_ppa == -0.1 and ac.off_pd_success == 0.3

    dr = gd.parse_drives([{"gameId": 401, "offense": "Oklahoma", "startYardsToGoal": 75, "startOffenseScore": 0, "endOffenseScore": 7},
                          {"gameId": 401, "offense": "Oklahoma", "startYardsToGoal": 60, "startOffenseScore": 7, "endOffenseScore": 7},
                          {"gameId": 401, "offense": "Temple", "startYardsToGoal": 80, "driveResult": "FG"}], 2025).set_index("team")
    assert dr.loc["Oklahoma", "ppd"] == 3.5 and dr.loc["Oklahoma", "start_ytg"] == 67.5 and dr.loc["Temple", "ppd"] == 3

    la = gd.parse_lines_all(LINES, 2025)
    assert len(la) == 2 and set(la.provider) == {"Bovada", "DraftKings"}

    q = gd.parse_qb_ppa([{"week": 3, "name": "John Mateer", "team": "Oklahoma", "averagePPA": {"all": 0.3, "pass": 0.35}}], 2025).iloc[0]
    assert q.ppa_pass == 0.35 and q.player == "John Mateer"

    e = gd.parse_elo_week([{"team": "Oklahoma", "elo": 1702}], 2025, 3, "regular").iloc[0]
    assert e.elo == 1702 and e.week == 3
    print("parser tests passed")


if __name__ == "__main__":
    main()
