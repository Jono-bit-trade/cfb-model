"use strict";

const TZ = "America/Chicago";
const state = { predictions: null, ratings: null, results: null, backtest: null, filter: "all", sort: "kickoff",
                ratingSort: { key: "overall", dir: -1 }, search: "" };

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const lineText = (team, line) => line === null || line === undefined ? "–" : Math.abs(line) < 0.05 ? `${team} PK` : `${team} ${line > 0 ? "+" : "−"}${Math.abs(line).toFixed(1)}`;
const pct = (v) => (v === null || v === undefined ? "–" : `${(v * 100).toFixed(1)}%`);

async function getJSON(name) {
  try {
    const r = await fetch(`data/${name}.json`, { cache: "no-cache" });
    return r.ok ? await r.json() : null;
  } catch { return null; }
}

function kickoffText(iso, tbd) {
  const d = new Date(iso);
  const day = new Intl.DateTimeFormat("en-US", { timeZone: TZ, weekday: "short", month: "short", day: "numeric" }).format(d);
  if (tbd) return { time: "Time TBA", day };
  const time = new Intl.DateTimeFormat("en-US", { timeZone: TZ, hour: "numeric", minute: "2-digit" }).format(d);
  return { time: `${time} CT`, day };
}

function updatedText(iso) {
  if (!iso) return "";
  return new Intl.DateTimeFormat("en-US", { timeZone: TZ, weekday: "short", month: "short", day: "numeric",
    hour: "numeric", minute: "2-digit" }).format(new Date(iso)) + " CT";
}

/* ------------------------------------------------------------ the field strip */
function fieldSVG(g, strong) {
  const R = 24, W = 300, mid = W / 2, scale = 140 / R;
  const clamp = (v) => Math.max(-R, Math.min(R, v));
  const x = (margin) => mid + clamp(margin) * scale;           // right = home favored
  const model = g.model.margin;
  const market = g.market.spread === null ? null : -g.market.spread;
  let lines = "";
  for (let p = -R; p <= R; p += 3) {
    const major = p === 0;
    lines += `<line x1="${x(p)}" x2="${x(p)}" y1="${major ? 14 : 19}" y2="${major ? 34 : 29}" stroke="var(--rule)" stroke-width="${major ? 2 : 1}"/>`;
  }
  const color = strong ? "var(--pylon)" : "var(--turf)";
  let span = "";
  if (market !== null) {
    const a = Math.min(x(market), x(model)), b = Math.max(x(market), x(model));
    span = `<rect x="${a}" y="21" width="${Math.max(b - a, 1)}" height="6" rx="3" fill="${color}" opacity=".35"/>`;
  }
  const marketMark = market === null ? "" :
    `<circle cx="${x(market)}" cy="24" r="6" fill="var(--surface)" stroke="var(--ink)" stroke-width="2"/>
     <text x="${x(market)}" y="9" text-anchor="middle" font-size="11" fill="var(--muted)">Line</text>`;
  const modelMark = `<circle class="model-dot" cx="${x(model)}" cy="24" r="6" fill="${color}"/>
     <text x="${x(model)}" y="44" text-anchor="middle" font-size="11" fill="var(--muted)">Model</text>`;
  const label = market === null
    ? `Model has ${g.winner} by ${num(Math.abs(model))}`
    : `Betting line has ${market >= 0 ? g.home.team : g.away.team} by ${num(Math.abs(market))}; model has ${model >= 0 ? g.home.team : g.away.team} by ${num(Math.abs(model))}`;
  return `<div class="field"><svg viewBox="0 0 ${W} 46" role="img" aria-label="${esc(label)}">
      <line x1="${x(-R)}" x2="${x(R)}" y1="24" y2="24" stroke="var(--rule)" stroke-width="1"/>
      ${lines}${span}${marketMark}${modelMark}</svg>
      <div class="ends"><span>${esc(g.away.team)} favored</span><span>Even</span><span>${esc(g.home.team)} favored</span></div></div>`;
}

/* ------------------------------------------------------------------- tags */
function weatherTag(w) {
  if (!w) return "";
  if (w.dome) return `<span class="tag">Dome</span>`;
  if (w.temp === null && w.wind === null) return `<span class="tag">Forecast not out yet</span>`;
  const bits = [];
  if (w.temp !== null) bits.push(`${Math.round(w.temp)}°F`);
  if (w.wind !== null) bits.push(`wind ${Math.round(w.wind)} mph`);
  if (w.gust !== null && w.gust - (w.wind ?? 0) >= 6) bits.push(`gusts ${Math.round(w.gust)}`);
  if (w.rain !== null && w.rain >= 0.05) bits.push(`${num(w.rain, 2)} in rain`);
  return `<span class="tag ${isWeatherGame(w) ? "warn" : ""}" title="${w.kind === "forecast" ? "Kickoff forecast" : "Observed"}">${esc(bits.join(", "))}</span>`;
}
const isWeatherGame = (w) => !!w && !w.dome && ((w.wind ?? 0) >= 15 || (w.gust ?? 0) >= 25 || (w.rain ?? 0) >= 0.1 || (w.temp ?? 60) <= 32);

function qbTags(g) {
  let out = "";
  for (const side of ["away", "home"]) {
    const q = g.qb[side];
    if (!q || !q.status || ["unknown", "available"].includes(q.status)) continue;
    const who = q.name ? `${q.name}` : `${g[side].team} QB`;
    let text = `${who}: ${q.status}${q.note ? ` (${q.note})` : ""}`;
    if (q.if_out) text += `. If he sits: ${g.away.team} ${num(q.if_out.away)}, ${g.home.team} ${num(q.if_out.home)}`;
    out += `<span class="tag ${q.status === "probable" ? "" : "warn"}">${esc(text)}</span>`;
  }
  return out;
}
const hasQbNews = (g) => ["away", "home"].some((s) => g.qb[s] && !["unknown", "available"].includes(g.qb[s].status));

/* ------------------------------------------------------------------ board */
function gameRow(g) {
  const k = kickoffText(g.kickoff, g.time_tbd);
  const homeWins = g.model.margin >= 0;
  const team = (side, wins) => `<div class="team ${wins ? "winner" : "loser"}">
      <span class="name">${esc(g[side].team)}<span class="rec">${esc(g[side].record)}</span></span>
      <span class="score" aria-label="projected ${num(g[side].proj)} points">${Math.round(g[side].proj)}</span></div>`;
  const m = g.market;
  const moved = m.spread !== null && m.spread_open !== null && m.spread_open !== m.spread;
  const numbers = `<dl class="numbers">
      <div><dt>Line</dt><dd>${esc(m.line_text || "None yet")}</dd></div>
      <div><dt>Model</dt><dd>${esc(g.model.line_text)}</dd></div>
      <div><dt>Total</dt><dd>${m.total === null ? "–" : num(m.total)}</dd></div>
      <div><dt>Model total</dt><dd>${num(g.model.total)}</dd></div>
      ${moved ? `<div><dt>Opened</dt><dd>${esc(lineText(g.home.team, m.spread_open))}</dd></div>` : ""}
    </dl>`;
  const strong = g.pick && g.pick.strong;
  const pick = g.pick && g.pick.text
    ? `<span class="pick">${esc(g.pick.text)}<small>${num(g.edge)}-point edge</small></span>`
    : `<span class="pick">${esc(g.winner)} to win<small>no line yet</small></span>`;
  return `<li class="game ${strong ? "strong" : ""}">
      <div class="kick"><time datetime="${esc(g.kickoff)}">${esc(k.time)}</time>${esc(k.day)}<br>${esc(g.venue || "")}${g.neutral ? " (neutral)" : ""}</div>
      <div class="teams">${team("away", !homeWins)}${team("home", homeWins)}</div>
      <div class="lines">${fieldSVG(g, strong)}${numbers}</div>
      <div class="pick-row">${pick}${g.total_pick ? `<span class="tag">${esc(g.total_pick)}</span>` : ""}${weatherTag(g.weather)}${qbTags(g)}</div>
    </li>`;
}

function renderBoard() {
  const p = state.predictions;
  if (!p) {
    $("#board-title").textContent = "No projections yet";
    $("#board-sub").textContent = "The weekly update hasn't run. Once it does, this week's games appear here.";
    return;
  }
  $("#board-title").textContent = `${p.label} projections`;
  const strongN = p.games.filter((g) => g.pick && g.pick.strong).length;
  $("#board-sub").textContent = `${p.games.length} games, ${strongN} strong picks. Projected scores are the model's; lines are the market's.`;
  $("#strong-edge").textContent = p.settings?.strong_edge ?? 3;

  let games = [...p.games];
  if (state.filter === "strong") games = games.filter((g) => g.pick && g.pick.strong);
  if (state.filter === "qb") games = games.filter(hasQbNews);
  if (state.filter === "weather") games = games.filter((g) => isWeatherGame(g.weather));
  if (state.sort === "edge") games.sort((a, b) => (b.edge ?? -1) - (a.edge ?? -1));
  $("#games").innerHTML = games.map(gameRow).join("");
  $("#games-empty").hidden = games.length > 0;

  const s = state.results?.summary?.season;
  if (s && s.ats && s.ats.w + s.ats.l > 0) {
    $("#season-line").innerHTML = `${s.ats.w}–${s.ats.l}${s.ats.p ? `–${s.ats.p}` : ""} against the spread<br><span>${s.su.w}–${s.su.l} picking winners this season</span>`;
  }
}

/* ---------------------------------------------------------------- ratings */
function renderRatings() {
  const r = state.ratings;
  const body = $("#ratings-table tbody");
  if (!r) { body.innerHTML = `<tr><td colspan="7">Ratings appear after the first weekly update.</td></tr>`; return; }
  $("#ratings-sub").textContent = `${r.season} season, ${r.teams.length} teams. Average team scores ${num(r.avg_points)} points; home field is worth ${num(r.home_edge)}.`;
  const { key, dir } = state.ratingSort;
  const rank = new Map([...r.teams].sort((a, b) => b.overall - a.overall).map((t, i) => [t.team, i + 1]));
  const recordWins = (t) => parseInt(String(t.record).split("–")[0], 10) || 0;
  let teams = r.teams.filter((t) => t.team.toLowerCase().includes(state.search));
  teams.sort((a, b) => {
    const av = key === "record" ? recordWins(a) : a[key], bv = key === "record" ? recordWins(b) : b[key];
    return (typeof av === "string" ? av.localeCompare(bv) : av - bv) * dir;
  });
  const max = Math.max(...r.teams.map((t) => Math.abs(t.overall)), 1);
  body.innerHTML = teams.map((t) => `<tr>
      <td class="num">${rank.get(t.team)}</td><td class="team-cell">${esc(t.team)}</td><td>${esc(t.conf || "")}</td>
      <td class="num">${esc(t.record)}</td>
      <td class="num">${num(t.overall, 2)}<span class="bar" style="width:${Math.max(0, t.overall) / max * 60}px"></span></td>
      <td class="num">${num(t.offense, 2)}</td><td class="num">${num(t.defense, 2)}</td></tr>`).join("");
  document.querySelectorAll("#ratings-table th button").forEach((b) => {
    b.removeAttribute("aria-sort");
    if (b.dataset.sort === key) b.setAttribute("aria-sort", dir < 0 ? "descending" : "ascending");
  });
}

/* ----------------------------------------------------------------- record */
function wl(x) { return x && x.w + x.l + x.p > 0 ? `${x.w}–${x.l}${x.p ? `–${x.p}` : ""}` : "–"; }
function renderRecord() {
  const res = state.results;
  const body = $("#results-table tbody");
  if (!res || !res.games.length) {
    $("#tally").innerHTML = "";
    body.innerHTML = `<tr><td colspan="6">Results appear once games the model picked have finished.</td></tr>`;
    return;
  }
  const s = res.summary.season;
  $("#record-title").textContent = `${res.season} record`;
  const tile = (label, x) => `<div><b>${wl(x)}</b><span>${label}${x && x.pct !== null ? `, ${pct(x.pct)}` : ""}</span></div>`;
  $("#tally").innerHTML = tile("Against the spread", s.ats) + tile("Strong picks against the spread", s.ats_strong)
    + tile("Picking winners", s.su) + tile("Over/under", s.ou);
  const note = $("#backfill-note");
  note.hidden = !res.backfilled_games;
  note.textContent = res.backfilled_games
    ? `${res.backfilled_games} games from before the site launched were filled in by replaying the model with only the information available before each game. Every pick since launch was recorded before kickoff.`
    : "";
  const mark = (v) => v === 1 ? `<span class="res-w">Won</span>` : v === 0 ? `<span class="res-l">Lost</span>` : v === 0.5 ? `<span class="res-p">Push</span>` : "–";
  body.innerHTML = res.games.map((g) => `<tr>
      <td>${g.week ?? ""}</td><td>${esc(g.away)} at ${esc(g.home)}</td>
      <td class="num">${Math.round(g.proj[1])}–${Math.round(g.proj[0])}</td><td class="num">${g.final[1]}–${g.final[0]}</td>
      <td>${esc(g.pick || "–")}</td><td>${mark(g.ats)}</td></tr>`).join("");
}

/* -------------------------------------------------------------- backtest */
function renderBacktest() {
  const b = state.backtest;
  const body = $("#backtest-table tbody");
  if (!b) { body.innerHTML = `<tr><td colspan="7">Run the backtest to fill this in.</td></tr>`; return; }
  const row = (label, r) => `<tr><td>${label}</td><td class="num">${r.games_with_line}</td><td class="num">${pct(r.ats_pct)}</td>
      <td class="num">${pct(r.su_pct)}</td><td class="num">${pct(r.ou_pct)}</td><td class="num">${num(r.mae_model)}</td><td class="num">${num(r.mae_market)}</td></tr>`;
  body.innerHTML = Object.entries(b.by_season).map(([s, r]) => row(s, r)).join("") + row("<b>All</b>", b.overall);
}

/* ---------------------------------------------------------------- router */
function show(tab) {
  const tabs = ["board", "ratings", "record", "how"];
  if (!tabs.includes(tab)) tab = "board";
  tabs.forEach((t) => { $(`#view-${t}`).hidden = t !== tab; });
  document.querySelectorAll(".tabs a").forEach((a) => {
    if (a.dataset.tab === tab) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
}

async function init() {
  [state.predictions, state.ratings, state.results, state.backtest] =
    await Promise.all(["predictions", "ratings", "results", "backtest"].map(getJSON));
  renderBoard(); renderRatings(); renderRecord(); renderBacktest();
  $("#updated").textContent = state.predictions
    ? `Updated ${updatedText(state.predictions.updated)}. Kickoff forecasts and quarterback status refresh every morning Wednesday through Saturday.` : "";

  document.querySelectorAll("[data-filter]").forEach((b) => b.addEventListener("click", () => {
    state.filter = b.dataset.filter;
    document.querySelectorAll("[data-filter]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    renderBoard();
  }));
  $("#sort").addEventListener("change", (e) => { state.sort = e.target.value; renderBoard(); });
  $("#team-search").addEventListener("input", (e) => { state.search = e.target.value.trim().toLowerCase(); renderRatings(); });
  document.querySelectorAll("#ratings-table th button").forEach((b) => b.addEventListener("click", () => {
    const k = b.dataset.sort;
    state.ratingSort = { key: k, dir: state.ratingSort.key === k ? -state.ratingSort.dir : (k === "defense" || k === "team" || k === "conf" ? 1 : -1) };
    renderRatings();
  }));
  window.addEventListener("hashchange", () => show(location.hash.slice(1)));
  show(location.hash.slice(1));
}
init();
