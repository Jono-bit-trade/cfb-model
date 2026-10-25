"use strict";

const TZ = "America/Chicago";
const state = { predictions: null, ratings: null, results: null, backtest: null, filter: "all", sort: "kickoff",
                ratingSort: { key: "overall", dir: -1 }, search: "" };

const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const lineText = (team, line) => line === null || line === undefined ? "–" : Math.abs(line) < 0.05 ? `${team} PK` : `${team} ${line > 0 ? "+" : "−"}${Math.abs(line).toFixed(1)}`;
const odds = (v) => v === null || v === undefined ? "–" : v > 0 ? `+${Math.round(v)}` : `−${Math.abs(Math.round(v))}`;
const moveTag = (g) => {
  const mv = g.market && g.market.line_move;
  if (mv === null || mv === undefined || Math.abs(mv) < 0.5) return "";
  const toward = mv > 0 ? g.home.team : g.away.team;
  const withPick = g.pick && g.pick.team ? (g.pick.team === toward ? "with the pick" : "against the pick") : "";
  return `<span class="tag ${withPick === "against the pick" ? "warn" : ""}" title="Change from the opening spread">Line moved ${num(Math.abs(mv))} toward ${esc(toward)}${withPick ? `, ${withPick}` : ""}</span>`;
};
const splitTags = (g) => {
  const s = g.splits;
  if (!s) return "";
  let out = "";
  if (s.rlm) {
    const pub = s.public, pubPct = pub === g.home.team ? s.bets_home : 100 - s.bets_home;
    out += `<span class="tag rlm">Reverse line movement: ${Math.round(pubPct)}% of bets on ${esc(pub)}, line moved ${num(Math.abs(g.market.line_move))} toward ${esc(s.rlm)}</span>`;
  }
  if (s.sharp) {
    const bets = s.sharp === g.home.team ? s.bets_home : 100 - s.bets_home, money = s.sharp === g.home.team ? s.money_home : 100 - s.money_home;
    out += `<span class="tag sharp">Sharp money: ${esc(s.sharp)} (${Math.round(money)}% of money, ${Math.round(bets)}% of bets)</span>`;
  }
  if (s.sharp_total) out += `<span class="tag sharp">Sharp money: ${s.sharp_total}</span>`;
  return out;
};
const hasSharp = (g) => g.splits && (g.splits.sharp || g.splits.rlm || g.splits.sharp_total);
const valueTag = (g) => g.moneyline && g.moneyline.value
  ? `<span class="tag" title="The model gives this side a ${num(g.moneyline.value.gap, 0)}-point better win chance than the moneyline implies">Value moneyline: ${esc(g.moneyline.value.team)} ${odds(g.moneyline.value.odds)}</span>` : "";
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
      ${g.moneyline && g.moneyline.home !== null ? `<div><dt>Moneyline</dt><dd>${esc(g.away.team)} ${odds(g.moneyline.away)}, ${esc(g.home.team)} ${odds(g.moneyline.home)}</dd></div>` : ""}
      ${g.splits ? `<div><dt>Bets / money</dt><dd>${esc(g.home.team)} ${Math.round(g.splits.bets_home)}% / ${Math.round(g.splits.money_home)}%</dd></div>` : ""}
      ${g.moneyline ? `<div><dt>Win chance</dt><dd>${esc(g.winner)} ${num(g.moneyline.win_chance, 0)}%</dd></div>` : ""}
    </dl>`;
  const strong = g.pick && g.pick.strong;
  const pick = g.pick && g.pick.text
    ? `<span class="pick">${esc(g.pick.text)}<small>${num(g.edge)}-point edge</small></span>`
    : `<span class="pick">${esc(g.winner)} to win<small>no line yet</small></span>`;
  return `<li class="game ${strong ? "strong" : ""}">
      <div class="kick"><time datetime="${esc(g.kickoff)}">${esc(k.time)}</time>${esc(k.day)}<br>${esc(g.venue || "")}${g.neutral ? " (neutral)" : ""}</div>
      <div class="teams">${team("away", !homeWins)}${team("home", homeWins)}</div>
      <div class="lines">${fieldSVG(g, strong)}${numbers}</div>
      <div class="pick-row">${pick}${g.total_pick ? `<span class="tag">${esc(g.total_pick)}</span>` : ""}${valueTag(g)}${moveTag(g)}${splitTags(g)}${weatherTag(g.weather)}${qbTags(g)}</div>
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
  if (state.filter === "sharp") games = games.filter(hasSharp);
  if (state.sort === "edge") games.sort((a, b) => (b.edge ?? -1) - (a.edge ?? -1));
  $("#games").innerHTML = games.map(gameRow).join("");
  $("#games-empty").hidden = games.length > 0;

  const s = state.results?.summary?.season;
  if (s && s.spread && s.spread.w + s.spread.l > 0) {
    $("#season-line").innerHTML = `${wl(s.spread)} against the spread<br><span>${wl(s.winner)} picking winners this season</span>`;
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

/* ---------------------------------------------------------------- history */
function wl(x) { return x && x.w + x.l + x.p > 0 ? `${x.w}–${x.l}${x.p ? `–${x.p}` : ""}` : "–"; }
const H = { index: null, cache: {}, season: null, team: "", edge: 0, valueOnly: false };

function tallyOf(rows, key) {
  let w = 0, l = 0, p = 0;
  for (const r of rows) { const v = r[key]; if (v === 1) w++; else if (v === 0) l++; else if (v === 0.5) p++; }
  return { w, l, p, pct: w + l ? w / (w + l) : null };
}
const units110 = (t) => t.w * (100 / 110) - t.l;
const sumUnits = (rows, key) => rows.reduce((s, r) => s + (r[key] ?? 0), 0);
const unitText = (u) => `<span class="units ${u > 0.005 ? "pos" : u < -0.005 ? "neg" : ""}">${u >= 0 ? "+" : "−"}${Math.abs(u).toFixed(1)} units</span>`;
const mark = (v) => v === 1 ? `<span class="mark w">Won</span>` : v === 0 ? `<span class="mark l">Lost</span>` : v === 0.5 ? `<span class="mark p">Push</span>` : "";
const signed = (v) => v === null || v === undefined ? "" : Math.abs(v) < 0.05 ? "PK" : `${v > 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}`;

async function loadSeason(s) {
  if (!H.cache[s]) H.cache[s] = (await getJSON(`history/${s}`))?.games ?? [];
  return H.cache[s];
}

async function initHistory() {
  H.index = await getJSON("history/index");
  const sel = $("#h-season");
  if (!H.index || !H.index.seasons.length) {
    $("#h-tally").innerHTML = "";
    $("#h-note").textContent = "History appears after the first backtest runs (Monday nights, or Run workflow on GitHub).";
    return;
  }
  sel.innerHTML = H.index.seasons.map((s) => `<option value="${s}">${s}${s === H.index.current ? " (this season)" : ""}</option>`).join("")
    + `<option value="all">All seasons</option>`;
  H.season = H.index.seasons.includes(H.index.current) ? String(H.index.current) : String(H.index.seasons[0]);
  sel.value = H.season;
  sel.addEventListener("change", () => { H.season = sel.value; renderHistory(); });
  $("#h-team").addEventListener("input", (e) => { H.team = e.target.value.trim().toLowerCase(); renderHistory(); });
  $("#h-edge").addEventListener("change", (e) => { H.edge = Number(e.target.value); renderHistory(); });
  $("#h-value").addEventListener("change", (e) => { H.valueOnly = e.target.checked; renderHistory(); });
  renderHistory();
}

async function renderHistory() {
  if (!H.index) return;
  const seasons = H.season === "all" ? H.index.seasons : [Number(H.season)];
  let rows = (await Promise.all(seasons.map(loadSeason))).flat();
  if (H.team) rows = rows.filter((r) => r.h.toLowerCase().includes(H.team) || r.a.toLowerCase().includes(H.team));
  const teamRows = rows;
  const spreadRows = rows.filter((r) => (r.edge ?? -1) >= H.edge);
  const totalRows = rows.filter((r) => (r.te ?? -1) >= H.edge);
  const shown = H.edge > 0 ? spreadRows : rows;

  const su = tallyOf(shown, "su"), ats = tallyOf(spreadRows, "ats"), ou = tallyOf(totalRows, "ou");
  const mlRows = H.valueOnly ? shown.filter((r) => r.val_u !== null) : shown.filter((r) => r.ml_u !== null);
  const mlUnits = sumUnits(mlRows, H.valueOnly ? "val_u" : "ml_u");
  const mlWins = mlRows.filter((r) => (H.valueOnly ? r.val_u : r.ml_u) > 0).length;
  const tile = (label, rec, extra = "") => `<div><b>${wl(rec)}</b><span>${label}${rec.pct !== null ? `, ${pct(rec.pct)}` : ""}</span>${extra}</div>`;
  $("#h-tally").innerHTML =
    tile("Picking winners", su)
    + `<div><b>${mlRows.length ? `${mlWins}–${mlRows.length - mlWins}` : "–"}</b><span>${H.valueOnly ? "Value moneylines" : "Moneyline on the projected winner"}</span>${mlRows.length ? unitText(mlUnits) : ""}</div>`
    + tile("Against the spread", ats, ats.w + ats.l ? unitText(units110(ats)) : "")
    + tile("Over/under", ou, ou.w + ou.l ? unitText(units110(ou)) : "");

  const bf = shown.filter((r) => r.bf).length;
  const label = H.season === "all" ? "all seasons" : H.season;
  $("#h-note").textContent = `${shown.length} games in ${label}${H.team ? ` involving "${H.team}"` : ""}`
    + `${H.edge ? `, where the model and the spread disagree by ${H.edge}+ points (over/under uses the same gap on the total)` : ""}. `
    + "Spread and over/under units assume standard −110 odds; moneyline units use each game's posted odds. "
    + "Past seasons were replayed using only information available before each game"
    + (bf ? `; ${bf} games this season were filled in the same way before launch.` : ".");

  // Edge ladder: where does the record clear 60%?
  const cuts = [0, 2, 3, 4, 5, 6, 7, 8, 10, 12];
  $("#h-edges tbody").innerHTML = cuts.map((c) => {
    const a = tallyOf(teamRows.filter((r) => (r.edge ?? -1) >= c), "ats");
    const o = tallyOf(teamRows.filter((r) => (r.te ?? -1) >= c), "ou");
    const n = a.w + a.l + a.p;
    const hit = a.pct !== null && a.pct >= 0.6 && n >= 50;
    return `<tr class="${hit ? "hit60" : ""}"><td>${c === 0 ? "Every game" : `${c}+ points`}</td><td class="num">${n}</td>
      <td class="num">${wl(a)}</td><td class="num">${pct(a.pct)}</td><td class="num">${wl(o)} ${o.pct !== null ? `(${pct(o.pct)})` : ""}</td></tr>`;
  }).join("");

  // Line movement: who the money backed, and how the model did with or against it.
  const coverOf = (r, line, home) => {
    if (r.hs === null || line === null || line === undefined) return null;
    const c = (r.hs - r.as) + line;            // > 0: home covered
    if (c === 0) return 0.5;
    return (c > 0) === home ? 1 : 0;
  };
  $("#h-moves tbody").innerHTML = [0.5, 1.5, 3, 5].map((c) => {
    const moved = teamRows.filter((r) => r.sp_open !== null && r.sp_open !== undefined && r.sp !== null && Math.abs(r.sp_open - r.sp) >= c);
    const towardHome = (r) => r.sp_open - r.sp > 0;
    const open = tallyOf(moved.map((r) => ({ v: coverOf(r, r.sp_open, towardHome(r)) })), "v");
    const close = tallyOf(moved.map((r) => ({ v: coverOf(r, r.sp, towardHome(r)) })), "v");
    const withMove = tallyOf(moved.filter((r) => r.ats_pick && r.ats_pick === (towardHome(r) ? r.h : r.a)), "ats");
    const against = tallyOf(moved.filter((r) => r.ats_pick && r.ats_pick !== (towardHome(r) ? r.h : r.a)), "ats");
    const cell = (t) => `${wl(t)}${t.pct !== null ? ` (${pct(t.pct)})` : ""}`;
    return `<tr><td>${c}+ points</td><td class="num">${moved.length}</td><td class="num">${pct(open.pct)}</td>
      <td class="num">${pct(close.pct)}</td><td class="num">${cell(withMove)}</td><td class="num">${cell(against)}</td></tr>`;
  }).join("");

  // Sharp money and reverse line movement (only games where splits were logged)
  const SG = H.index.sharp_gap ?? 10, PM = H.index.public_min ?? 60, RM = H.index.rlm_move ?? 1;
  const withSplits = teamRows.filter((r) => r.bh !== null && r.bh !== undefined && r.mh !== null && r.mh !== undefined);
  const side = (r, home) => home ? r.h : r.a;
  const sharpHome = (r) => r.mh - r.bh >= SG ? true : r.mh - r.bh <= -SG ? false : null;
  const rlmHome = (r) => {
    if (r.sp_open === null || r.sp_open === undefined || r.sp === null) return null;
    const mv = r.sp_open - r.sp;
    if (r.bh <= 100 - PM && mv >= RM) return true;
    if (r.bh >= PM && mv <= -RM) return false;
    return null;
  };
  const signals = [
    ["Side with sharp money", withSplits.filter((r) => sharpHome(r) !== null).map((r) => ({ v: coverOf(r, r.sp, sharpHome(r)) }))],
    ["Reverse line movement side", withSplits.filter((r) => rlmHome(r) !== null).map((r) => ({ v: coverOf(r, r.sp, rlmHome(r)) }))],
    ["Model picks with the sharp side", withSplits.filter((r) => sharpHome(r) !== null && r.ats_pick === side(r, sharpHome(r))).map((r) => ({ v: r.ats }))],
    ["Model picks against the sharp side", withSplits.filter((r) => sharpHome(r) !== null && r.ats_pick && r.ats_pick !== side(r, sharpHome(r))).map((r) => ({ v: r.ats }))],
  ];
  $("#h-sharp-note").textContent = withSplits.length
    ? `${withSplits.length} games with logged betting splits. All records are against the closing spread.`
    : "No betting splits logged for this selection yet. They build up as data/splits.csv is filled in each week.";
  $("#h-sharp tbody").innerHTML = withSplits.length ? signals.map(([label, rs]) => {
    const t = tallyOf(rs, "v");
    return `<tr><td>${label}</td><td class="num">${t.w + t.l + t.p}</td><td class="num">${wl(t)}</td><td class="num">${pct(t.pct)}</td></tr>`;
  }).join("") : "";

  const list = [...shown].sort((x, y) => y.date.localeCompare(x.date));
  const LIMIT = 400;
  const fmtDay = (iso) => new Intl.DateTimeFormat("en-US", { timeZone: TZ, month: "short", day: "numeric", year: H.season === "all" ? "numeric" : undefined }).format(new Date(iso));
  $("#h-games tbody").innerHTML = list.slice(0, LIMIT).map((r) => {
    const ph = (r.pt + r.pm) / 2, pa = (r.pt - r.pm) / 2;
    const winner = r.pm >= 0 ? r.h : r.a;
    const mlCell = H.valueOnly
      ? (r.val ? `${esc(r.val)} ${odds(r.val_odds)}<span class="small">${r.val_u >= 0 ? "+" : "−"}${Math.abs(r.val_u).toFixed(2)} units</span>` : "")
      : (r.ml_odds !== null ? `${odds(r.ml_odds)}<span class="small">${r.ml_u >= 0 ? "+" : "−"}${Math.abs(r.ml_u ?? 0).toFixed(2)} units</span>` : "–");
    return `<tr>
      <td>${r.post ? "Bowl" : r.wk ?? ""}<span class="small">${fmtDay(r.date)}</span></td>
      <td>${esc(r.a)} at ${esc(r.h)}${r.bf ? `<span class="bf">replayed</span>` : ""}</td>
      <td class="num">${r.as}–${r.hs}</td><td class="num">${Math.round(pa)}–${Math.round(ph)}</td>
      <td>${esc(winner)} ${mark(r.su)}<span class="small">${r.pw !== null ? `${Math.round(r.pw * 100)}% win chance` : ""}</span></td>
      <td>${mlCell}</td>
      <td>${r.ats_pick ? `${esc(r.ats_pick)} ${signed(r.ats_line)} ${mark(r.ats)}<span class="small">${num(r.edge)}-point edge</span>` : "No line"}</td>
      <td>${r.ou_pick ? `${r.ou_pick} ${num(r.tot)} ${mark(r.ou)}<span class="small">${num(r.te)}-point edge</span>` : "–"}</td></tr>`;
  }).join("") || `<tr><td colspan="8">No games match these filters.</td></tr>`;
  $("#h-more").hidden = list.length <= LIMIT;
  $("#h-more").textContent = `Showing the newest ${LIMIT} of ${list.length} games. Narrow it with the season or team filters.`;
}

/* -------------------------------------------------------------- backtest */
function renderBacktest() {
  const b = state.backtest;
  const body = $("#backtest-table tbody");
  if (!b) { body.innerHTML = `<tr><td colspan="7">Run the backtest to fill this in.</td></tr>`; return; }
  const row = (label, r) => `<tr><td>${label}</td><td class="num">${r.games}</td><td class="num">${pct(r.spread.pct)}</td>
      <td class="num">${pct(r.winner.pct)}</td><td class="num">${pct(r.total.pct)}</td><td class="num">${num(r.mae_model)}</td><td class="num">${num(r.mae_market)}</td></tr>`;
  body.innerHTML = Object.entries(b.by_season).map(([s, r]) => row(s, r)).join("") + row("<b>All</b>", b.overall);
}

/* ---------------------------------------------------------------- router */
function show(tab) {
  const tabs = ["board", "ratings", "history", "how"];
  if (!tabs.includes(tab)) tab = "board";
  tabs.forEach((t) => { $(`#view-${t}`).hidden = t !== tab; });
  document.querySelectorAll(".tabs a").forEach((a) => {
    if (a.dataset.tab === tab) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
}

async function init() {
  [state.predictions, state.ratings, state.results, state.backtest] =
    await Promise.all(["predictions", "ratings", "results", "backtest"].map(getJSON));
  renderBoard(); renderRatings(); renderBacktest(); initHistory();
  const tuning = await getJSON("tuning");
  if (tuning) {
    const r = tuning.holdout_at_strong_edge, s = tuning.settings;
    const n = $("#tuning-note");
    n.hidden = false;
    n.textContent = `Settings were tuned on ${tuning.tune_seasons[0]}–${tuning.tune_seasons.at(-1)} and then checked on ${tuning.holdout_seasons[0]}–${tuning.holdout_seasons.at(-1)}, seasons the tuning never saw. `
      + `On those, strong picks (${s.STRONG_EDGE}+ point edges) went ${r.w}–${r.l}${r.p ? `–${r.p}` : ""}${r.pct !== null ? ` (${pct(r.pct)})` : ""} against the spread.`;
  }
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
