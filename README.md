# Jono's Picks

A college football prediction model and website. Every week it projects every FBS game's score, compares the projection with the betting line, and publishes the picks with the kickoff weather forecast and each team's starting-quarterback status.

Website: **https://jono-bit-trade.github.io/jonos-picks/** (live once setup step 2 below is done)

## One-time setup (about 5 minutes)

1. **Add your API key as a secret.** In this repository go to **Settings → Secrets and variables → Actions → New repository secret**. Name: `CFBD_KEY`. Value: the key CollegeFootballData emailed you. Never put the key in a file here.
2. **Turn on the website.** Go to **Settings → Pages**. Under "Build and deployment", set Source to **Deploy from a branch**, Branch to **main**, folder **/docs**, then **Save**. (Free GitHub accounts need the repository to be public for Pages.)
3. **Run it for the first time.** Go to the **Actions** tab, choose **weekly-predictions**, then **Run workflow**. The first run downloads every season since 2013, which takes 20 to 40 minutes. When it finishes, the website shows this week's games.

After that it runs on its own:

| When (Central time) | What happens |
| --- | --- |
| Sunday evening | Saturday's results are in: ratings rebuild and each game's **early pick locks** at the current line |
| Monday night | Full refresh, picks and the backtest |
| Wednesday to Saturday mornings | Latest lines, kickoff weather forecasts and quarterback status |
| Thursday and Friday nights, Saturday afternoon and night | Scores every 2 hours, so finished games show their results on the board |

## Your weekly job: quarterback status

The model checks whether each team's starting QB is out. Past games are figured out automatically from box scores. For this week's games, it reads `data/qb_status.csv`, which you update from the conference availability reports. Open the file on GitHub, click the pencil icon, add a line, and commit:

```
season,week,team,qb,status,note
2026,9,Texas,Arch Manning,questionable,ankle
```

- `status` is one of: `available`, `probable`, `questionable`, `doubtful`, `out`
- `out` and `doubtful` are projected with the starter missing.
- `questionable` shows both projections on the site: if he plays and if he sits.
- Team names must match the site's spelling. Teams you don't list are assumed to have their starter.

The next morning's run picks it up, or click **Run workflow** to update right away.

## Betting splits (optional): sharp money and reverse line movement

There's no free source for betting splits (share of bets vs. share of money), so you add them from a service you use, such as DraftKings splits on VSiN or Action Network. Add a line per game to `data/splits.csv`; percentages are for the home team's spread and for the over:

```
season,week,home,away,bets_home_pct,money_home_pct,bets_over_pct,money_over_pct
2026,9,Texas,Ole Miss,38,61,70,52
```

The weekly board then flags:

- **Sharp money:** the money share beats the bet share by 10+ points on one side (fewer, bigger bets).
- **Reverse line movement:** 60%+ of bets are on one side, yet the spread moved 1+ point the other way since opening.

Every logged split is saved with the pick, so the History page's "Sharp money and reverse line movement" table measures over time whether these signals win. The thresholds are `SHARP_GAP`, `PUBLIC_MIN` and `RLM_MIN_MOVE` in `src/config.py`.

## How the model works

| Step | File | What it does |
| --- | --- | --- |
| Data | `src/get_data.py` | Downloads games, betting lines, efficiency stats, box scores, talent, returning production, recruiting, transfer portal, coaches and stadiums from CollegeFootballData |
| Ratings | `src/ratings.py` | Opponent-adjusted offense and defense ratings in points, starting each season from a roster-based preseason prior |
| Weather | `src/weather.py` | Free kickoff forecasts and past game-day weather from Open-Meteo |
| QB out | `src/qb.py` | Starting QB out flag: box scores for past games, `data/qb_status.csv` for this week |
| Features | `src/features.py` | Efficiency matchups, pace, turnovers, penalties, rest, travel, altitude, time zones, coaching changes, situations |
| Model | `src/model.py` | Gradient-boosted trees blended with the rating projection |
| Backtest | `src/backtest.py` | Replays past seasons week by week using only information available at the time |
| Site | `src/predict.py`, `docs/` | Writes the week's picks, ratings and record for the website |

All tunable settings are in `src/config.py`. To test a change, edit a value and run the backtest (or ask Claude to). Keep a change only if it improves results across several seasons.

## History: how it would have done

The **History** page grades every game since 2018 four ways, the way the sites it was modeled on do:

- **Picking winners**: did the projected winner win?
- **Moneyline**: betting the projected winner at its posted odds, in units won or lost. Tick "Value moneylines only" to see just the bets where the model's win chance beat the odds.
- **Against the spread** and **Over/under**: the model's side against the line, at standard -110 odds.

Filter by season, team or minimum edge. "Where the model is strongest" shows the spread record at each edge size and highlights any level that holds 60% or better over at least 50 games.

Past seasons are replayed walk-forward: each game uses only ratings and a model built from games before it, so the history isn't flattered by hindsight.

## Making it more accurate: the tuner

Run **Actions → tune-model → Run workflow** (20 to 60 minutes). It tries combinations of rating settings, feature groups and blend weights on the older seasons, picks the best, then finds the smallest edge where the spread record reached 60%. It then checks those choices on the 3 newest seasons, which played no part in choosing them; the How it works page shows that holdout result. The winning settings are saved to `data/tuned.json` and used for this week's predictions too, so upcoming picks and the History page always come from the same model.

A realistic target: 60%+ picking winners is normal; against the spread, every game, 53 to 56% is very good. 60% against the spread is only realistic on the subset of games with the biggest edges, and only if the holdout seasons confirm it.

## Early picks and "Bet early"

Lines tend to move toward the side the model likes: in 2023-2025, lines moved toward the model's side about 70% of the time when it disagreed with the opener by 5+ points, and its picks went about 57% against the opening line, versus about 51% against the closing line. So the model's value is early in the week.

- **Locked pick:** the first projection after the previous weekend's games is saved with the line at that moment and never changes. Every game is graded at both the locked line and the closing line.
- **Bet early:** marks games where the model's edge is 3+ points and the line is still within about a point of where it opened. In 2023-2025, those picks won 54.6% at 5+ point edges; once a line had already moved toward the pick, results fell to about 49% (the edge was gone), and when it moved against the pick, about 51%.
- **Totals too:** the same "Bet early" rule applies to over/unders: a 3+ point gap between the model's total and the market's, with the total still near its opening number. In 2023-2025, 5+ point total edges won 57% at the close, and 60% when the total hadn't moved (116 games, so treat it as promising, not proven).
- **Line history chart:** each game card has a "Line history" panel charting the spread and total from open to now, with the model's number as a dashed line.
- **Line snapshots:** every run saves the current lines to `data/line_history.csv`, so picks can later be graded at the line on a specific day, such as Wednesday.
- **History → Grade against:** switch between the closing line and the early line (the locked pick this season, the opening line for past seasons).

## Recent settings changes (tested on 2023-2025, chosen on 2016-2022)

- **Line-movement adjustment (on, spreads):** where an opening line exists (2023 on), a second step adjusts the projected margin based on how the spread has moved, learned only from earlier seasons' out-of-sample projections. It lowered the margin error in both 2024 and 2025. The same idea for totals didn't help, so it's off for totals.
- **Machine-learning settings:** shallower trees learning more slowly, faster-reacting efficiency ratings, and a separate blend weight for totals.
- **Recent seasons weighted more:** tested (3- and 6-year half-lives) and left off: it didn't improve results. Team strength already comes from the current season, starting from last season's rating.
- **New data sources (tested one at a time):** pregame Elo ratings and efficiency with garbage time removed (including passing/rushing splits and line play) are on: together they cut the margin error in both 2016-2022 and 2023-2025. Quarterback efficiency, points per drive/field position and sportsbook disagreement were downloaded and tested but didn't improve unseen seasons, so they're off (the data stays for future tests).
- **Blowout spreads:** in games with an opening spread of 21+ points the model used to underrate the favorite by 4-6 points. The line-movement step now learns how far to lean toward the opening line in those games (2024-25: average miss 12.69 -> 12.44, 3+ point edges vs the opener 56.0% -> 58.1%).
- **Sweet spot:** 3-7 point spread edges at the current line are the main plays (about 52-54% at the close). Bigger edges have gone about 50% at the close because they usually mean the model is missing news, so they get a check-the-news label instead.
- **AP rankings (RANK):** each team's AP rank going into the game is now an input. It cut the margin error slightly in both 2016-2022 (13.19 → 13.17) and 2023-2025 (12.72 → 12.69). On their own, rankings don't beat the line: ranked favorites cover about 48.5%, and big underdogs against top-5 teams win less often than their moneyline implies.
- **Training years:** the spread model learns from every season (more history helped), while the totals model learns from the most recent 5 seasons, because scoring levels drift with rule changes and pace. Tested 3, 5, 7 and all seasons, and with and without 2020.
- **Wind and totals:** projected totals drop about 0.12 points per mph of wind above 10 mph (using the kickoff forecast for upcoming games). Learned on 2016-2022; on 2023-2025 it lifted the model's over/under record in windy games from 49.9% to 51.0%. Dome and rain patterns looked like Under spots in older seasons but didn't hold up, so they aren't used.
- **Win chances** are calibrated to the 16.6-point scatter measured in the backtest.

## The record

`data/predictions_log.csv` keeps the last projection made before each kickoff, and the Record page grades those. When the site first launches mid-season, earlier games of the season are filled in by replaying the model with only pre-game information, and the site labels them as backfilled.

## API usage

The first download uses roughly 700 CollegeFootballData calls. After that, upkeep is about 80 calls a month. If your plan's monthly limit is lower than the first download needs, run the backfill in pieces: on your computer, `python src/get_data.py --from-season 2019`, or ask Claude to change the workflow.

## Running it on your own computer (optional)

```
pip install -r requirements.txt
echo YOUR_KEY > key.txt        # key.txt is never uploaded (.gitignore)
python src/get_data.py
python src/predict.py
python src/backtest.py
cd docs && python -m http.server   # then open http://localhost:8000
```

Tests (no key needed): `python tests/test_parsers.py` and `bash tests/smoke_test.sh`.

Projections are for entertainment and analysis, not betting advice.
