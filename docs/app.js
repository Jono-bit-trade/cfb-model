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
const betEarlyTag = (g) => g.bet_early && g.pick && g.pick.team
  ? `<span class="tag rlm" title="The model's edge is ${num(g.edge)} points and the line is still close to where it opened. In 2023-2025, picks like this won 54.6% at 5+ point edges; once the line had moved toward the pick, the edge was gone.">Bet now: ${esc(g.pick.text)}</span>` : "";
const totalTags = (g) => {
  let out = "";
  const mv = g.total_move;
  if (mv !== null && mv !== undefined && Math.abs(mv) >= 0.5) {
    const dir = mv > 0 ? "up" : "down";
    const pickOver = g.total_pick && g.total_pick.startsWith("Over");
    const withPick = g.total_pick ? ((mv > 0) === pickOver ? "with the pick" : "against the pick") : "";
    out += `<span class="tag ${withPick === "against the pick" ? "warn" : ""}" title="Change from the opening total">Total moved ${dir} ${num(Math.abs(mv))}${withPick ? `, ${withPick}` : ""}</span>`;
  }
  if (g.total_bet_early && g.total_pick) out += `<span class="tag rlm" title="The model's total is 3+ points from the market and the total is still close to where it opened. In 2023-2025, 5+ point total edges won 57% at the close, and 60% when the total hadn't moved (small sample).">Bet now: ${esc(g.total_pick)}</span>`;
  return out;
};

/* Line history chart: the market line over time against the model's number (dashed). */
function lineChart(points, key, model, title, fmt) {
  const pts = points.filter((p) => p[key] !== null && p[key] !== undefined);
  if (pts.length < 2 || model === null || model === undefined) return "";
  const W = 300, H = 104, L = 8, R = 8, T = 22, B = 26;
  const vals = pts.map((p) => p[key]).concat([model]);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (hi - lo < 2) { lo -= 1; hi += 1; }
  const x = (i) => L + (pts.length === 1 ? 0 : i * (W - L - R) / (pts.length - 1));
  const y = (v) => T + (hi - v) * (H - T - B) / (hi - lo);
  const path = pts.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(p[key]).toFixed(1)}`).join(" ");
  const label = `${title}: ${pts.map((p) => `${p.label} ${fmt(p[key])}`).join(", ")}; model ${fmt(model)}`;
  return `<figure class="mini-chart"><figcaption>${esc(title)}</figcaption>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label)}">
      <line x1="${L}" x2="${W - R}" y1="${y(model)}" y2="${y(model)}" stroke="var(--pylon)" stroke-width="1.5" stroke-dasharray="4 4"/>
      <text x="${W - R}" y="${y(model) - 5}" text-anchor="end" font-size="11" fill="var(--pylon)">Model ${esc(fmt(model))}</text>
      <path d="${path}" fill="none" stroke="var(--turf)" stroke-width="2"/>
      ${pts.map((p, i) => `<circle cx="${x(i)}" cy="${y(p[key])}" r="3.5" fill="var(--turf)"/>
        <text x="${x(i)}" y="${y(p[key]) - 8}" text-anchor="${i === 0 ? "start" : i === pts.length - 1 ? "end" : "middle"}" font-size="11" fill="var(--ink)">${esc(fmt(p[key]))}</text>
        <text x="${x(i)}" y="${H - 6}" text-anchor="${i === 0 ? "start" : i === pts.length - 1 ? "end" : "middle"}" font-size="11" fill="var(--muted)">${esc(p.label)}</text>`).join("")}
    </svg></figure>`;
}
function lineCharts(g) {
  const h = g.line_history || [];
  if (h.length < 2) return "";
  const spread = lineChart(h, "spread", -g.model.margin, `Spread, ${g.home.team}`, (v) => signed(v));
  const total = lineChart(h, "total", g.model.total, "Total", (v) => num(v));
  return spread || total ? `<div class="movement"><p class="mini-title">Line movement <span>open to now, model dashed</span></p><div class="mini-charts">${spread}${total}</div></div>` : "";
}
function lineHistory(g) {
  const h = g.line_history || [];
  if (h.length < 2) return "";
  const spread = lineChart(h, "spread", -g.model.margin, `Spread, ${g.home.team}`, (v) => signed(v));
  const total = lineChart(h, "total", g.model.total, "Total", (v) => num(v));
  if (!spread && !total) return "";
  return `<details class="line-history"><summary>Line history</summary><div class="mini-charts">${spread}${total}</div></details>`;
}
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

/* ---------------------------------------------------------- weather icons */
const WX_ICON = (() => {
  const sun = (cx = 12, cy = 12, r = 4.2) => `<g class="i-sun"><circle cx="${cx}" cy="${cy}" r="${r}"/>${[0, 45, 90, 135, 180, 225, 270, 315].map((a) => {
    const t = (a * Math.PI) / 180, r1 = r + 2.2, r2 = r + 4.4;
    return `<line x1="${(cx + r1 * Math.cos(t)).toFixed(1)}" y1="${(cy + r1 * Math.sin(t)).toFixed(1)}" x2="${(cx + r2 * Math.cos(t)).toFixed(1)}" y2="${(cy + r2 * Math.sin(t)).toFixed(1)}"/>`;
  }).join("")}</g>`;
  const moon = (dx = 0, dy = 0) => `<path class="i-moon" transform="translate(${dx} ${dy})" d="M15.5 4.5a7.5 7.5 0 1 0 4 13.6A6.2 6.2 0 0 1 15.5 4.5z"/>`;
  const cloud = (dx = 0, dy = 0, cls = "i-cloud") => `<path class="${cls}" transform="translate(${dx} ${dy})" d="M7 19h10.5a4 4 0 0 0 .4-8 5.5 5.5 0 0 0-10.6-1.3A4.7 4.7 0 0 0 7 19z"/>`;
  const drops = `<g class="i-rain"><line x1="8.5" y1="20.5" x2="7.5" y2="23"/><line x1="12.5" y1="20.5" x2="11.5" y2="23"/><line x1="16.5" y1="20.5" x2="15.5" y2="23"/></g>`;
  const flakes = `<g class="i-snow"><circle cx="8" cy="21.6" r="1.1"/><circle cx="12" cy="22.6" r="1.1"/><circle cx="16" cy="21.6" r="1.1"/></g>`;
  const bolt = `<path class="i-bolt" d="M12.8 15.5 9.6 20.6h2.6l-1.2 3.4 3.8-5.4h-2.6l1.3-3.1z"/>`;
  const wrap = (body) => `<svg class="wx-ico" viewBox="0 0 24 24" aria-hidden="true">${body}</svg>`;
  return {
    clear: (night) => wrap(night ? moon(-1, 0) : sun()),
    partly: (night) => wrap((night ? moon(-4, -3) : sun(9, 8.5, 3.4)) + cloud(1.5, 1.5)),
    cloudy: () => wrap(cloud(-1.5, -2.5, "i-cloud back") + cloud(1, 1)),
    fog: () => wrap(`<g class="i-fog"><line x1="4" y1="8" x2="20" y2="8"/><line x1="2.5" y1="12" x2="18.5" y2="12"/><line x1="5.5" y1="16" x2="21.5" y2="16"/><line x1="4" y1="20" x2="17" y2="20"/></g>`),
    rain: () => wrap(cloud(0, -3.5) + drops),
    storm: () => wrap(cloud(0, -4.5, "i-cloud dark") + bolt + `<g class="i-rain"><line x1="7.5" y1="19.5" x2="6.5" y2="22"/><line x1="17.5" y1="19.5" x2="16.5" y2="22"/></g>`),
    snow: () => wrap(cloud(0, -3.5) + flakes),
    wind: () => wrap(`<g class="i-wind"><path d="M3 9h11.5a2.6 2.6 0 1 0-2.6-2.6"/><path d="M3 13.5h16a2.8 2.8 0 1 1-2.8 2.8"/><path d="M3 18h7.5"/></g>`),
    dome: () => wrap(`<g class="i-dome"><path d="M2.5 18.5a9.5 8 0 0 1 19 0z"/><path d="M8 18.5a4 7.6 0 0 1 8 0M12 10.5v8M5 14.5h14"/><line x1="1.5" y1="20.5" x2="22.5" y2="20.5"/></g>`),
  };
})();

function wxKind(w) {
  if (w.dome) return ["dome", "Indoors"];
  const c = w.code;
  let k = c === null || c === undefined ? null
    : c >= 95 ? "storm" : (c >= 71 && c <= 77) || c === 85 || c === 86 ? "snow"
    : c >= 51 ? "rain" : c >= 45 ? "fog" : c === 3 ? "cloudy" : c === 2 ? "partly" : "clear";
  if (!k) k = (w.rain ?? 0) >= 0.1 ? ((w.temp ?? 50) <= 32 ? "snow" : "rain") : "partly";
  if ((k === "clear" || k === "partly" || k === "cloudy") && ((w.wind ?? 0) >= 15 || (w.gust ?? 0) >= 25)) k = "wind";
  const label = { clear: w.night ? "Clear" : "Sunny", partly: "Partly cloudy", cloudy: "Cloudy", fog: "Fog", rain: "Rain",
    storm: "Thunderstorms", snow: "Snow", wind: "Windy", dome: "Indoors" }[k];
  return [k, label];
}

function wxChip(w, compact = false) {
  if (!w) return "";
  if (!w.dome && w.temp === null && w.wind === null) return `<span class="wx pending" title="Kickoff forecast comes out about a week before the game">Forecast soon</span>`;
  const [k, label] = wxKind(w);
  const icon = WX_ICON[k](w.night);
  if (w.dome) return `<span class="wx dome" title="Played indoors">${icon}<span class="wx-t">Dome</span></span>`;
  const extra = [];
  if (w.wind !== null) extra.push(Math.round(w.wind) < 2 ? "calm" : `${Math.round(w.wind)} mph`);
  if (!compact && w.gust !== null && w.gust - (w.wind ?? 0) >= 6) extra.push(`gusts ${Math.round(w.gust)}`);
  if (!compact && w.rain !== null && w.rain >= 0.05) extra.push(`${num(w.rain, 2)} in`);
  const tip = `${label} at kickoff (${w.kind === "forecast" ? "forecast" : "observed"})`;
  return `<span class="wx ${isWeatherGame(w) ? "bad" : ""}" title="${esc(tip)}">${icon}<span class="wx-t">${w.temp !== null ? `${Math.round(w.temp)}°` : ""}</span><span class="wx-s">${esc(label)}${extra.length ? " · " + esc(extra.join(" · ")) : ""}</span></span>`;
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

const REPO = (() => {
  const owner = location.hostname.endsWith(".github.io") ? location.hostname.split(".")[0] : "Jono-bit-trade";
  const repo = location.hostname.endsWith(".github.io") ? (location.pathname.split("/")[1] || "jonos-picks") : "jonos-picks";
  return `https://github.com/${owner}/${repo}`;
})();
const slug = (t) => String(t).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
function confirmLink(g, side, qbName, status, label, note) {
  const season = state.predictions.season, week = g.week;
  const csvCell = (v) => `"${String(v ?? "").replace(/"/g, "'")}"`;
  const body = `season,week,team,qb,status,note\n${season},${week},${csvCell(g[side].team)},${csvCell(qbName)},${status},${csvCell(note)}\n`;
  const file = `data/qb_confirm/${season}-w${String(week).padStart(2, "0")}-${slug(g[side].team)}-${Date.now()}.csv`;
  const url = `${REPO}/new/main?filename=${encodeURIComponent(file)}&value=${encodeURIComponent(body)}`;
  return `<a class="qb-btn ${status}" href="${esc(url)}" target="_blank" rel="noopener" title="Opens GitHub with the file filled in: click Commit changes to apply">${esc(label)}</a>`;
}
const timeAgo = (iso) => {
  if (!iso) return "";
  const h = (Date.now() - new Date(iso).getTime()) / 3.6e6;
  return h < 1 ? "just now" : h < 24 ? `${Math.round(h)}h ago` : `${Math.round(h / 24)}d ago`;
};
function qbTags(g) {
  let out = "";
  for (const side of ["away", "home"]) {
    const q = g.qb[side];
    if (!q) continue;
    const who = q.name ? `${q.name}` : `${g[side].team} QB`;
    const ifOut = q.if_out ? `. If he sits: ${short(g.away.team)} ${num(q.if_out.away)}, ${short(g.home.team)} ${num(q.if_out.home)}` : "";
    const confirmed = q.status && !["unknown"].includes(q.status);
    if (q.auto && q.news) {
      const n = q.news, name = n.qb || who;
      const ev = (n.evidence || []).map((e) => `<li>${esc(e)}</li>`).join("");
      out += `<div class="qb-news auto">
          <div class="qb-news-head"><span class="qb-news-label">QB out · confirmed by news · applied to picks</span><b>${esc(name)} (${esc(short(g[side].team))}) is out</b></div>
          <a class="qb-news-link" href="${esc(n.link)}" target="_blank" rel="noopener">${esc(n.headline)}</a>
          <div class="qb-news-src">${esc(n.source || "")}${n.published ? ` · ${esc(timeAgo(n.published))}` : ""}${n.missed_last ? " · didn't play last game" : ""}</div>
          ${ev ? `<ul class="qb-news-ev">${ev}</ul>` : ""}
          <div class="qb-news-actions"><span>Wrong?</span>
            ${confirmLink(g, side, name, "available", "He's playing", "override of news")}
            ${confirmLink(g, side, name, "questionable", "Only questionable", "override of news")}</div>
        </div>`;
      continue;
    }
    if (confirmed && q.status !== "available") {
      out += `<span class="tag ${q.status === "probable" ? "" : "warn"}">${esc(`${who}: ${q.status}${q.note ? ` (${q.note})` : ""}${ifOut}`)}</span>`;
    }
    if (q.news && !confirmed) {
      const n = q.news, name = n.qb || who;
      const word = { out: "may be out", doubtful: "doubtful", questionable: "questionable", available: "expected to play" }[n.suggested] || n.suggested;
      out += `<div class="qb-news ${n.suggested}">
          <div class="qb-news-head"><span class="qb-news-label">QB news · unverified</span><b>${esc(name)} (${esc(short(g[side].team))}) ${esc(word)}</b>${n.suggested !== "available" ? esc(ifOut) : ""}</div>
          <a class="qb-news-link" href="${esc(n.link)}" target="_blank" rel="noopener">${esc(n.headline)}</a>
          <div class="qb-news-src">${esc(n.source || "")}${n.published ? ` · ${esc(timeAgo(n.published))}` : ""}</div>
          <div class="qb-news-actions"><span>Verify:</span>
            ${confirmLink(g, side, name, "out", "He's out", n.headline)}
            ${confirmLink(g, side, name, "questionable", "Questionable", n.headline)}
            ${confirmLink(g, side, name, "available", "He's playing", n.headline)}</div>
        </div>`;
    }
  }
  return out;
}
const hasQbNews = (g) => ["away", "home"].some((s) => g.qb[s] && (!["unknown", "available"].includes(g.qb[s].status) || (g.qb[s].news && g.qb[s].news.suggested !== "available")));

/* ------------------------------------------------------------------ board */
/* ------------------------------------------------------------- team marks */
function contrastText(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex || "");
  if (!m) return "#fff";
  const n = parseInt(m[1], 16), r = n >> 16, g = (n >> 8) & 255, b = n & 255;
  return (0.299 * r + 0.587 * g + 0.114 * b) > 160 ? "#0f1a2b" : "#fff";
}
function abbrOf(team) {
  const t = state.teams && state.teams[team];
  return (t && t.a) || team.split(/\s+/).map((w) => w[0]).join("").slice(0, 4).toUpperCase();
}
function badgeHTML(team, size) {
  const t = state.teams && state.teams[team];
  const bg = (t && t.c) ? (t.c.startsWith("#") ? t.c : `#${t.c}`) : "#7a8a82";
  return `<span class="logo badge" style="width:${size}px;height:${size}px;background:${bg};color:${contrastText(bg)};font-size:${Math.round(size * 0.34)}px" aria-hidden="true">${esc(abbrOf(team))}</span>`;
}
function logo(team, size = 32) {
  const t = state.teams && state.teams[team];
  if (t && t.l) return `<img class="logo" src="${esc(t.l)}" alt="" width="${size}" height="${size}" loading="lazy" data-team="${esc(team)}" data-size="${size}">`;
  return badgeHTML(team, size);
}
// a logo that fails to load becomes the team-colored badge
document.addEventListener("error", (e) => {
  const img = e.target;
  if (img && img.tagName === "IMG" && img.classList.contains("logo")) {
    img.outerHTML = badgeHTML(img.dataset.team, Number(img.dataset.size) || 32);
  }
}, true);
const short = (team) => abbrOf(team);

/* --------------------------------------------------------------- game card */
function gameRow(g) {
  const k = kickoffText(g.kickoff, g.time_tbd);
  const homeWins = g.model.margin >= 0;
  const teamRow = (side, wins) => `<div class="team-row ${wins ? "winner" : "loser"}">
      ${logo(g[side].team, 34)}
      <div class="team-id"><span class="name">${esc(g[side].team)}</span><span class="rec">${esc(g[side].record)}</span></div>
      <span class="score" aria-label="projected ${num(g[side].proj)} points">${Math.round(g[side].proj)}</span></div>`;
  const m = g.market, strong = !!g.our_pick;
  const moved = m.spread !== null && m.spread_open !== null && m.spread_open !== m.spread;
  const totalEdge = m.total !== null ? Math.abs(g.model.total - m.total) : null;
  const ml = g.moneyline;
  const mlPick = ml && ml.value ? `${esc(short(ml.value.team))} ${odds(ml.value.odds)}<span>value bet</span>`
    : `${esc(short(g.winner))}<span>${ml ? `${num(ml.win_chance, 0)}% to win` : "to win"}</span>`;
  const spreadPick = g.pick && g.pick.text
    ? `${esc(lineText(short(g.pick.team), g.pick.line))}<span>${num(g.edge)}-pt edge${g.our_pick ? " · ★ our pick" : ""}</span>` : `–<span>no line yet</span>`;
  const markets = `<table class="mk">
      <thead><tr><th scope="col"><span class="sr">Market</span></th><th scope="col">Line</th><th scope="col">Model</th><th scope="col">Pick</th></tr></thead>
      <tbody>
        <tr><th scope="row">Spread</th>
          <td>${m.spread !== null ? esc(lineText(short(g.home.team), m.spread)) : "–"}${moved ? `<span>opened ${esc(signed(m.spread_open))}</span>` : ""}</td>
          <td>${esc(lineText(short(g.home.team), -g.model.margin))}</td>
          <td class="pick ${strong ? "strong" : g.pick && g.pick.caution ? "caution" : ""}">${spreadPick}</td></tr>
        <tr><th scope="row">Total</th>
          <td>${m.total !== null ? num(m.total) : "–"}${m.total_open !== null && m.total_open !== m.total ? `<span>opened ${num(m.total_open)}</span>` : ""}</td>
          <td>${num(g.model.total)}</td>
          <td class="pick ${g.total_bet_early ? "strong" : ""}">${g.total_pick ? `${esc(g.total_pick)}<span>${num(totalEdge)}-pt edge</span>` : "–"}</td></tr>
        <tr><th scope="row">Moneyline</th>
          <td>${ml && ml.home !== null ? `${esc(short(g.away.team))} ${odds(ml.away)}<span>${esc(short(g.home.team))} ${odds(ml.home)}</span>` : "–"}</td>
          <td>${ml ? `${num(ml.win_chance, 0)}%<span>${esc(short(g.winner))}</span>` : "–"}</td>
          <td class="pick">${mlPick}</td></tr>
      </tbody></table>`;
  const cautionTag = g.pick && g.pick.caution
    ? `<span class="tag caution" title="Edges this big hold up when bet early, but once the line has settled near kickoff they've gone about 50%: make sure there's no injury or lineup news the model is missing">Big edge (${num(g.edge)} pts): double-check injury and lineup news</span>` : "";
  const leanTag = !g.our_pick && g.week && g.week < 5 && g.edge !== null && g.edge >= 5
    ? `<span class="tag" title="Weeks 1–4 rely on preseason estimates: 5+ point edges went about 52% then, so these aren't counted as our picks">Early-season lean: ${esc(g.pick.text || "")}</span>` : "";
  const ourTag = g.our_pick ? `<span class="tag ours" title="5+ point edge${g.our_pick.locked ? " at the locked early line" : " (not locked yet)"}: one of our picks. Bet it early, before the line moves.">★ Our pick: ${esc(g.our_pick.text)}${g.our_pick.locked ? " (locked)" : ""}</span>` : "";
  const badges = `${ourTag}${leanTag}${cautionTag}${betEarlyTag(g)}${g.total_bet_early ? totalTags(g).replace(/<span class="tag (warn)?"[^>]*>Total moved[^<]*<\/span>/, "") : ""}${qbTags(g)}${splitTags(g)}`;
  const details = `<details class="more"><summary>Details</summary>
      <div class="more-body">
        ${fieldSVG(g, strong)}
        <div class="tags">${g.locked ? `<span class="tag">Locked pick: ${esc(g.locked.text)}</span>` : ""}${moveTag(g)}${totalTags(g).replace(/<span class="tag rlm"[^]*$/, "")}${g.splits ? `<span class="tag">Bets / money on ${esc(short(g.home.team))}: ${Math.round(g.splits.bets_home)}% / ${Math.round(g.splits.money_home)}%</span>` : ""}</div>
      </div></details>`;
  return `<li class="game ${strong ? "strong" : ""}">
      <div class="meta"><time datetime="${esc(g.kickoff)}">${esc(k.day)}, ${esc(k.time)}</time>${g.network ? `<span class="net">${esc(g.network)}</span>` : ""}<span class="venue">${esc(g.venue || "")}${g.neutral ? " (neutral site)" : ""}</span>${wxChip(g.weather)}</div>
      <a class="matchup" href="#game/${g.id}" aria-label="Full matchup: ${esc(g.away.team)} at ${esc(g.home.team)}">${teamRow("away", !homeWins)}${teamRow("home", homeWins)}<span class="open-matchup">Full matchup →</span></a>
      <div class="markets">${markets}</div>
      ${badges.trim() ? `<div class="badges">${badges}</div>` : ""}
      ${lineCharts(g)}
      ${details}
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
  $("#board-sub").textContent = `${p.games.length} games, ${p.games.filter((g) => g.our_pick).length} of our picks (5+ point edges at the early line, week 5 on). Projected scores are the model's; lines are the market's.`;
  $("#strong-edge").textContent = p.settings?.strong_edge ?? 3;

  let games = [...p.games];
  if (state.filter === "strong") games = games.filter((g) => g.our_pick);
  if (state.filter === "qb") games = games.filter(hasQbNews);
  if (state.filter === "weather") games = games.filter((g) => isWeatherGame(g.weather));
  if (state.filter === "sharp") games = games.filter(hasSharp);
  if (state.filter === "early") games = games.filter((g) => g.bet_early || g.total_bet_early);
  if (state.sort === "edge") games.sort((a, b) => (b.edge ?? -1) - (a.edge ?? -1));
  $("#games").innerHTML = games.map(gameRow).join("");
  $("#games-empty").hidden = games.length > 0;

  renderBoardRecords();
  const sum = state.results?.summary || {}, s = sum.season, op = sum.our_picks;
  const units = (u) => `${u >= 0 ? "+" : "−"}${Math.abs(u).toFixed(1)}u`;
  if (op && op.w + op.l > 0) {
    const wk = sum.our_picks_week, bt = sum.our_picks_backtest;
    $("#season-line").innerHTML = `<span class="op-label">Our picks</span> ${wl(op)} · ${units(op.units)}`
      + `<br><span>${wk && wk.w + wk.l > 0 ? `Week ${wk.week}: ${wl(wk)} · ` : ""}${s ? `${wl(s.winner)} picking winners` : ""}</span>`
      + (bt ? `<br><span title="Same rule in the backtest, graded at the opening line">${bt.from}–${bt.to} backtest: ${wl(bt)}</span>` : "");
  } else if (s && s.spread && s.spread.w + s.spread.l > 0) {
    $("#season-line").innerHTML = `${wl(s.spread)} against the spread<br><span>${wl(s.winner)} picking winners this season</span>`;
  }
}

/* ----------------------------------------------------------- final scores */
function renderFinals() {
  const f = state.finals;
  const box = $("#finals");
  if (!f || !f.games.length) { box.hidden = true; return; }
  box.hidden = false;
  const t = (k) => tallyOf(f.games, k);
  const su = t("su"), ats = t("ats"), lk = t("lk_ats"), ou = t("ou");
  $("#finals-sub").textContent = `Last 4 days: winners ${wl(su)}, spread ${wl(ats)} at the closing line, ${wl(lk)} at the locked line, over/under ${wl(ou)}.`;
  $("#finals-list").innerHTML = f.games.map((r) => {
    const homeWon = r.hs > r.as;
    const lockText = r.lk_sp !== null && r.lk_sp !== undefined && r.lk_pm !== null
      ? (() => { const e = r.lk_pm + r.lk_sp; const team = e > 0 ? r.h : r.a; const line = e > 0 ? r.lk_sp : -r.lk_sp;
                 return `${esc(team)} ${signed(line)} ${mark(r.lk_ats)}`; })() : "–";
    return `<li class="final">
      <div class="final-score"><span class="${homeWon ? "" : "won"}">${logo(r.a, 24)}<b>${esc(r.a)}</b><em>${r.as}</em></span><span class="${homeWon ? "won" : ""}">${logo(r.h, 24)}<b>${esc(r.h)}</b><em>${r.hs}</em></span></div>
      <dl class="numbers final-picks">
        <div><dt>Winner</dt><dd>${esc(r.pm >= 0 ? r.h : r.a)} ${mark(r.su)}</dd></div>
        <div><dt>Spread (close)</dt><dd>${r.ats_pick ? `${esc(r.ats_pick)} ${signed(r.ats_line)} ${mark(r.ats)}` : "–"}</dd></div>
        <div><dt>Locked pick</dt><dd>${lockText}</dd></div>
        <div><dt>Total</dt><dd>${r.ou_pick ? `${r.ou_pick} ${num(r.tot)} ${mark(r.ou)}` : "–"}</dd></div>
      </dl></li>`;
  }).join("");
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
      <td class="num">${rank.get(t.team)}</td><td class="team-cell"><span class="with-logo">${logo(t.team, 24)}${esc(t.team)}</span></td><td>${esc(t.conf || "")}</td>
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
const H = { index: null, cache: {}, season: null, team: "", edge: 0, valueOnly: false, line: "close", week: "all", sort: "newest" };
const weekKey = (r) => r.post ? "bowls" : String(r.wk ?? "");
const weekName = (k) => k === "bowls" ? "Bowls" : `Week ${k}`;
const weekOrder = (k) => k === "bowls" ? 99 : Number(k);

/* Re-grade a game at the early line: the locked pick this season, the opening line before that. */
function atEarlyLine(r) {
  const L = r.lk_sp ?? r.sp_open ?? null, PM = r.lk_pm ?? r.pm;
  const T = r.lk_tot ?? r.tot_open ?? null, PT = r.lk_pt ?? r.pt;
  const out = { ...r, sp: L, tot: T };
  const res = (score, positive) => score === 0 ? 0.5 : (score > 0) === positive ? 1 : 0;
  if (L === null || r.hs === null) { Object.assign(out, { ats: null, ats_pick: null, edge: null }); }
  else {
    const e = PM + L, home = e > 0;
    Object.assign(out, { edge: Math.abs(e), ats_pick: e === 0 ? null : home ? r.h : r.a, ats_line: home ? L : -L,
                         ats: e === 0 ? null : res((r.hs - r.as) + L, home) });
  }
  if (T === null || r.hs === null) { Object.assign(out, { ou: null, ou_pick: null, te: null }); }
  else {
    const te = PT - T;
    Object.assign(out, { te: Math.abs(te), ou_pick: te === 0 ? null : te > 0 ? "Over" : "Under",
                         ou: te === 0 ? null : res((r.hs + r.as) - T, te > 0) });
  }
  return out;
}

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
  $("#h-edge").addEventListener("change", (e) => {
    H.edge = e.target.value === "ours" ? "ours" : Number(e.target.value);
    if (H.edge === "ours") { H.line = "early"; $("#h-line").value = "early"; }  // our picks are graded at the line you'd bet
    renderHistory();
  });
  $("#h-value").addEventListener("change", (e) => { H.valueOnly = e.target.checked; renderHistory(); });
  $("#h-line").addEventListener("change", (e) => { H.line = e.target.value; renderHistory(); });
  $("#h-week").addEventListener("change", (e) => { H.week = e.target.value; renderHistory(); });
  $("#h-sort").addEventListener("change", (e) => { H.sort = e.target.value; renderHistory(); });
  $("#h-weeks").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.target.closest("tr[data-week]")) e.target.click();
  });
  $("#h-weeks").addEventListener("click", (e) => {
    const tr = e.target.closest("tr[data-week]");
    if (!tr) return;
    H.week = H.week === tr.dataset.week ? "all" : tr.dataset.week;
    $("#h-week").value = H.week;
    renderHistory();
  });
  renderHistory();
}

const OUR_FROM_WEEK = 5;
const inOurWeeks = (r) => r.post || (r.wk ?? 0) >= OUR_FROM_WEEK;
const edgeOk = (v, r) => H.edge === "ours" ? v !== null && v !== undefined && v >= 5 && (!r || inOurWeeks(r)) : (v ?? -1) >= H.edge;
async function renderHistory() {
  if (!H.index) return;
  const seasons = H.season === "all" ? H.index.seasons : [Number(H.season)];
  let rows = (await Promise.all(seasons.map(loadSeason))).flat();
  if (H.line === "early") rows = rows.map(atEarlyLine).filter((r) => r.sp !== null || r.tot !== null);
  if (H.team) rows = rows.filter((r) => r.h.toLowerCase().includes(H.team) || r.a.toLowerCase().includes(H.team));

  // Week dropdown follows the weeks present in the selected season(s)
  const weeks = [...new Set(rows.map(weekKey).filter((k) => k))].sort((a, b) => weekOrder(a) - weekOrder(b));
  const weekSel = $("#h-week");
  weekSel.innerHTML = `<option value="all">All weeks</option>` + weeks.map((k) => `<option value="${k}">${weekName(k)}</option>`).join("");
  if (H.week !== "all" && !weeks.includes(H.week)) H.week = "all";
  weekSel.value = H.week;

  // Week-by-week records (before narrowing to one week)
  const byWeek = new Map();
  for (const r of rows) { const k = weekKey(r); if (!byWeek.has(k)) byWeek.set(k, []); byWeek.get(k).push(r); }
  // ★ Our picks are always graded at the early line (5+ point edge there), whatever the "Grade against" setting
  const oursOf = (rs) => {
    const picks = rs.map((r) => (H.line === "early" ? r : atEarlyLine(r))).filter((r) => r.edge !== null && r.edge >= 5 && r.ats !== null && inOurWeeks(r));
    const t = tallyOf(picks, "ats");
    return { ...t, units: units110(t) };
  };
  const weekRec = (rs) => {
    const e = rs.filter((r) => edgeOk(r.edge, r)), t = rs.filter((r) => edgeOk(r.te, r));
    const ml = rs.filter((r) => (H.valueOnly ? r.val_u : r.ml_u) !== null && (H.valueOnly ? r.val_u : r.ml_u) !== undefined);
    const u = sumUnits(ml, H.valueOnly ? "val_u" : "ml_u");
    return { su: tallyOf(rs, "su"), ats: tallyOf(e, "ats"), ou: tallyOf(t, "ou"), u, mlN: ml.length, ours: oursOf(rs) };
  };
  const cellR = (t) => t.w + t.l + t.p ? `${wl(t)}${t.pct !== null ? ` <span class="small-inline">${pct(t.pct)}</span>` : ""}` : "–";
  const unitCell = (u, n) => n ? `<span class="${u > 0.005 ? "res-w" : u < -0.005 ? "res-l" : ""}">${u >= 0 ? "+" : "−"}${Math.abs(u).toFixed(1)}u</span>` : "–";
  $("#h-weeks tbody").innerHTML = [...byWeek.entries()].sort((a, b) => weekOrder(a[0]) - weekOrder(b[0])).map(([k, rs]) => {
    const x = weekRec(rs);
    return `<tr data-week="${k}" class="clickable ${H.week === k ? "selected" : ""}" tabindex="0"><td>${weekName(k)}</td><td class="num">${rs.length}</td><td class="num ours-col">${cellR(x.ours)}${x.ours.w + x.ours.l ? ` ${unitCell(x.ours.units, 1)}` : ""}</td>
      <td class="num">${cellR(x.su)}</td><td class="num">${cellR(x.ats)}</td><td class="num">${cellR(x.ou)}</td><td class="num">${unitCell(x.u, x.mlN)}</td></tr>`;
  }).join("");
  if (H.week !== "all") rows = rows.filter((r) => weekKey(r) === H.week);
  const teamRows = rows;
  const spreadRows = rows.filter((r) => edgeOk(r.edge, r));
  const totalRows = rows.filter((r) => edgeOk(r.te, r));
  const shown = H.edge === "ours" || H.edge > 0 ? spreadRows : rows;

  const su = tallyOf(shown, "su"), ats = tallyOf(spreadRows, "ats"), ou = tallyOf(totalRows, "ou");
  const mlRows = H.valueOnly ? shown.filter((r) => r.val_u !== null) : shown.filter((r) => r.ml_u !== null);
  const mlUnits = sumUnits(mlRows, H.valueOnly ? "val_u" : "ml_u");
  const mlWins = mlRows.filter((r) => (H.valueOnly ? r.val_u : r.ml_u) > 0).length;
  const tile = (label, rec, extra = "") => `<div><b>${wl(rec)}</b><span>${label}${rec.pct !== null ? `, ${pct(rec.pct)}` : ""}</span>${extra}</div>`;
  const ours = oursOf(rows);
  $("#h-tally").innerHTML =
    `<div class="ours"><b>${ours.w + ours.l + ours.p ? wl(ours) : "–"}</b><span>★ Our picks${ours.pct !== null ? `, ${pct(ours.pct)}` : ""}</span>${ours.w + ours.l ? unitText(ours.units) : ""}<span class="close-note">5+ point edge at the early line, week 5 on</span></div>`
    + tile("Picking winners", su)
    + `<div><b>${mlRows.length ? `${mlWins}–${mlRows.length - mlWins}` : "–"}</b><span>${H.valueOnly ? "Value moneylines" : "Moneyline on the projected winner"}</span>${mlRows.length ? unitText(mlUnits) : ""}</div>`
    + tile("Against the spread", ats, ats.w + ats.l ? unitText(units110(ats)) : "")
    + tile("Over/under", ou, ou.w + ou.l ? unitText(units110(ou)) : "");

  const bf = shown.filter((r) => r.bf).length;
  const label = H.season === "all" ? "all seasons" : H.season;
  const lineNote = H.line === "early"
    ? "Graded at the early line: the pick locked on Sunday or Monday this season, and the opening line for earlier seasons (only games with an opening line are included). "
    : "Graded at the closing line, the last line before kickoff. ";
  $("#h-note").textContent = lineNote + `${shown.length} games in ${label}${H.team ? ` involving "${H.team}"` : ""}`
    + `${H.edge === "ours" ? ", our picks only: the model and the early line disagree by 5+ points, from week 5 on" : H.edge ? `, where the model and the spread disagree by ${H.edge}+ points (over/under uses the same gap on the total)` : ""}. `
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

  // Over/under movement
  const ouCover = (r, line, over) => {
    if (r.hs === null || line === null || line === undefined) return null;
    const c = (r.hs + r.as) - line;
    if (c === 0) return 0.5;
    return (c > 0) === over ? 1 : 0;
  };
  $("#h-tmoves tbody").innerHTML = [1, 2, 3, 5].map((c) => {
    const moved = teamRows.filter((r) => r.tot_open !== null && r.tot_open !== undefined && r.tot !== null && Math.abs(r.tot - r.tot_open) >= c);
    const up = (r) => r.tot - r.tot_open > 0;
    const open = tallyOf(moved.map((r) => ({ v: ouCover(r, r.tot_open, up(r)) })), "v");
    const close = tallyOf(moved.map((r) => ({ v: ouCover(r, r.tot, up(r)) })), "v");
    const withMove = tallyOf(moved.filter((r) => r.ou_pick && (r.ou_pick === "Over") === up(r)), "ou");
    const against = tallyOf(moved.filter((r) => r.ou_pick && (r.ou_pick === "Over") !== up(r)), "ou");
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

  const list = H.sort === "week"
    ? [...shown].sort((x, y) => (x.date.slice(0, 4) - y.date.slice(0, 4)) || (weekOrder(weekKey(x)) - weekOrder(weekKey(y))) || x.date.localeCompare(y.date))
    : [...shown].sort((x, y) => y.date.localeCompare(x.date));
  const LIMIT = 400;
  const fmtDay = (iso) => new Intl.DateTimeFormat("en-US", { timeZone: TZ, month: "short", day: "numeric", year: H.season === "all" ? "numeric" : undefined }).format(new Date(iso));
  let lastWeek = null;
  const header = (r) => {
    const k = `${r.date.slice(0, 4)}-${weekKey(r)}`;
    if (H.sort !== "week" || k === lastWeek) return "";
    lastWeek = k;
    const rs = list.filter((x) => `${x.date.slice(0, 4)}-${weekKey(x)}` === k);
    const x = weekRec(rs);
    return `<tr class="week-row"><th colspan="8" scope="colgroup">${H.season === "all" ? `${r.date.slice(0, 4)} ` : ""}${weekName(weekKey(r))}
      <span>Winners ${wl(x.su)}, spread ${wl(x.ats)}, over/under ${wl(x.ou)}</span></th></tr>`;
  };
  $("#h-games tbody").innerHTML = list.slice(0, LIMIT).map((r) => header(r) + (() => {
    const ph = (r.pt + r.pm) / 2, pa = (r.pt - r.pm) / 2;
    const winner = r.pm >= 0 ? r.h : r.a;
    const mlCell = H.valueOnly
      ? (r.val ? `${esc(r.val)} ${odds(r.val_odds)}<span class="small">${r.val_u >= 0 ? "+" : "−"}${Math.abs(r.val_u).toFixed(2)} units</span>` : "")
      : (r.ml_odds !== null ? `${odds(r.ml_odds)}<span class="small">${r.ml_u >= 0 ? "+" : "−"}${Math.abs(r.ml_u ?? 0).toFixed(2)} units</span>` : "–");
    return `<tr>
      <td>${r.post ? "Bowl" : r.wk ?? ""}<span class="small">${fmtDay(r.date)}</span></td>
      <td><span class="with-logo">${logo(r.a, 20)}${esc(r.a)}</span><span class="with-logo at">at ${logo(r.h, 20)}${esc(r.h)}</span>${r.bf ? `<span class="bf">replayed</span>` : ""}</td>
      <td class="num">${r.as}–${r.hs}</td><td class="num">${Math.round(pa)}–${Math.round(ph)}</td>
      <td>${esc(winner)} ${mark(r.su)}<span class="small">${r.pw !== null ? `${Math.round(r.pw * 100)}% win chance` : ""}</span></td>
      <td>${mlCell}</td>
      <td>${r.ats_pick ? `${esc(r.ats_pick)} ${signed(r.ats_line)} ${mark(r.ats)}<span class="small">${num(r.edge)}-point edge</span>` : "No line"}</td>
      <td>${r.ou_pick ? `${r.ou_pick} ${num(r.tot)} ${mark(r.ou)}<span class="small">${num(r.te)}-point edge</span>` : "–"}</td></tr>`;
  })()).join("") || `<tr><td colspan="8">No games match these filters.</td></tr>`;
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
  const tabs = ["board", "ratings", "history", "how", "game"];
  let gameId = null;
  if (tab.startsWith("game/")) { gameId = tab.slice(5); tab = "game"; }
  if (!tabs.includes(tab) || (tab === "game" && !gameId)) tab = "board";
  tabs.forEach((t) => { $(`#view-${t}`).hidden = t !== tab; });
  if (tab === "game") { renderGame(gameId); window.scrollTo(0, 0); tab = "board"; }
  else document.title = "Jono's Picks — college football projections";
  document.querySelectorAll(".tabs a").forEach((a) => {
    if (a.dataset.tab === tab) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
  });
}

async function init() {
  [state.predictions, state.ratings, state.results, state.backtest, state.finals, state.teams] =
    await Promise.all(["predictions", "ratings", "results", "backtest", "finals", "teams"].map(getJSON));
  state.teams = state.teams || {};
  renderBoard(); renderFinals(); renderRatings(); renderBacktest(); initHistory();
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


/* ------------------------------------------------------------ matchup page */
const ordinal = (n) => { const s = ["th", "st", "nd", "rd"], v = n % 100; return n + (s[(v - 20) % 10] || s[v] || s[0]); };
function statValue(st) {
  if (st.value === null || st.value === undefined) return "–";
  if (st.fmt === "pct") return `${(st.value * 100).toFixed(1)}%`;
  return Number(st.value).toFixed(Number(st.fmt));
}
function rankChip(rank, n) {
  if (!rank) return "";
  const cls = rank <= Math.ceil(n * 0.2) ? "top" : rank > Math.floor(n * 0.75) ? "low" : "";
  return `<span class="rk ${cls}">${ordinal(rank)}</span>`;
}
function formList(rows) {
  if (!rows.length) return `<p class="muted">No games yet.</p>`;
  return `<ol class="form">${rows.map((r) => `<li>
      <span class="res ${r.result}">${r.result}</span>
      <span class="opp">${r.site === "A" ? "at " : r.site === "N" ? "vs " : ""}${logo(r.opp, 18)}${esc(short(r.opp))}</span>
      <span class="sc">${r.pts}–${r.opp_pts}</span>
      <span class="ats ${r.cover || ""}" title="${r.spread !== null ? `Line ${signed(r.spread)}` : "No line"}">${r.cover ? (r.cover === "W" ? "Covered" : r.cover === "L" ? "Missed" : "Push") : ""}${r.spread !== null ? ` <small>${esc(signed(r.spread))}</small>` : ""}</span>
    </li>`).join("")}</ol>`;
}
async function renderGame(id) {
  const body = $("#game-body");
  const g = (state.predictions?.games || []).find((x) => String(x.id) === String(id));
  if (!g) { body.innerHTML = `<p>This game isn't on the board any more. <a href="#board">See this week's games</a>.</p>`; return; }
  body.innerHTML = `<p class="muted">Loading matchup…</p>`;
  const m = await getJSON(`matchups/${id}`);
  const k = kickoffText(g.kickoff, g.time_tbd);
  const side = (s) => {
    const t = m?.teams?.[s], rk = t?.rank_ap ? `<span class="ap">#${t.rank_ap}</span>` : "";
    return `<div class="hero-team ${s}">
        ${logo(g[s].team, 72)}
        <div class="hero-name">${rk}${esc(g[s].team)}</div>
        <div class="hero-rec">${esc(g[s].record)}${t?.rating ? ` · ${ordinal(t.rating.rank)} in our ratings` : ""}</div>
        <div class="hero-proj">${Math.round(g[s].proj)}</div>
      </div>`;
  };
  const hero = `<div class="hero">
      ${side("away")}
      <div class="hero-mid"><span class="at">${g.neutral ? "vs" : "at"}</span>
        <div class="hero-meta"><time datetime="${esc(g.kickoff)}">${esc(k.day)}, ${esc(k.time)}</time>${g.network ? ` · ${esc(g.network)}` : ""}</div>
        <div class="hero-venue">${esc(g.venue || "")}${g.neutral ? " (neutral)" : ""}</div>
        ${wxChip(g.weather)}
        <div class="hero-label">Projected score</div>
      </div>
      ${side("home")}
    </div>`;
  // markets table and badges re-used from the board card
  const card = document.createElement("div");
  card.innerHTML = gameRow(g);
  const markets = card.querySelector(".markets")?.outerHTML || "";
  const badges = card.querySelector(".badges")?.outerHTML || "";
  let html = hero + `<div class="game-panels">${markets}${badges}</div>` + (lineCharts(g) ? `<section class="panel">${lineCharts(g)}</section>` : "");
  if (!m) { body.innerHTML = html + `<p class="muted">Detailed team stats will appear after the next update.</p>`; return; }
  const A = m.teams.away, H = m.teams.home, n = m.n_teams;
  const recRow = (label, key) => `<tr><td class="l">${esc(A.records[key])}</td><th scope="row">${label}</th><td class="r">${esc(H.records[key])}</td></tr>`;
  const rows = A.stats.map((sa, i) => {
    const sh = H.stats[i];
    let aBetter = null;
    if (sa.better !== null && sa.value !== null && sh && sh.value !== null && sa.value !== sh.value) aBetter = (sa.value > sh.value) === sa.better;
    return `<tr>
        <td class="l ${aBetter === true ? "edge" : ""}">${statValue(sa)} ${rankChip(sa.rank, n)}</td>
        <th scope="row">${esc(sa.label)}</th>
        <td class="r ${aBetter === false ? "edge" : ""}">${rankChip(sh?.rank, n)} ${sh ? statValue(sh) : "–"}</td></tr>`;
  }).join("");
  const thead = `<thead><tr><th scope="col" class="l">${logo(A.team, 22)} ${esc(short(A.team))}</th><th scope="col"></th><th scope="col" class="r">${esc(short(H.team))} ${logo(H.team, 22)}</th></tr></thead>`;
  html += `<section class="panel"><h2>Team comparison <span>season averages · rank among ${n} FBS teams · better side highlighted</span></h2>
      <table class="compare">${thead}<tbody>
        ${A.rating && H.rating ? `<tr><td class="l">${num(A.rating.overall)} ${rankChip(A.rating.rank, n)}</td><th scope="row">Our rating (points better than average)</th><td class="r">${rankChip(H.rating.rank, n)} ${num(H.rating.overall)}</td></tr>` : ""}
        ${recRow("Record", "su")}${recRow("Against the spread", "ats")}${recRow("Over / under", "ou")}
        ${rows}</tbody></table></section>`;
  html += `<section class="panel"><h2>Recent form <span>last 5 games · result against the spread</span></h2>
      <div class="two-col"><div><h3>${logo(A.team, 20)} ${esc(A.team)}</h3>${formList(A.recent)}</div>
      <div><h3>${logo(H.team, 20)} ${esc(H.team)}</h3>${formList(H.recent)}</div></div></section>`;
  const h2h = m.h2h || [];
  html += `<section class="panel"><h2>Past meetings <span>${h2h.length ? `${esc(H.team)} is ${esc(m.h2h_record.su)} straight up and ${esc(m.h2h_record.ats)} against the spread since 2013` : "no meetings since 2013"}</span></h2>
      ${h2h.length ? `<div class="table-wrap"><table class="h2h"><thead><tr><th scope="col">Date</th><th scope="col">Winner</th><th scope="col" class="num">Score</th><th scope="col" class="num">${esc(short(H.team))} line</th><th scope="col">${esc(short(H.team))} ATS</th><th scope="col" class="num">Total</th></tr></thead><tbody>
        ${h2h.map((r) => {
          const hw = r.result === "W", winner = hw ? H.team : r.result === "L" ? r.opp : "Tie";
          const d = new Date(r.date).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "America/Chicago" });
          return `<tr><td>${esc(d)}${r.site === "N" ? " <small>(neutral)</small>" : r.site === "A" ? ` <small>at ${esc(short(r.opp))}</small>` : ` <small>at ${esc(short(H.team))}</small>`}</td>
            <td><span class="cell-logo">${logo(winner, 18)}${esc(winner)}</span></td>
            <td class="num">${Math.max(r.pts, r.opp_pts)}–${Math.min(r.pts, r.opp_pts)}</td>
            <td class="num">${r.spread !== null ? esc(signed(r.spread)) : "–"}</td>
            <td><span class="ats ${r.cover || ""}">${r.cover === "W" ? "Covered" : r.cover === "L" ? "Missed" : r.cover === "P" ? "Push" : "–"}</span></td>
            <td class="num">${r.total !== null ? `${num(r.total)} <small>${r.ou === "O" ? "over" : r.ou === "U" ? "under" : "push"}</small>` : "–"}</td></tr>`;
        }).join("")}</tbody></table></div>` : ""}</section>`;
  body.innerHTML = html;
  document.title = `${g.away.team} at ${g.home.team} · Jono's Picks`;
}


/* -------------------------------------------- record tiles on the board */
function renderBoardRecords() {
  const sum = state.results?.summary, box = $("#board-records");
  if (!sum || !sum.season) { box.hidden = true; return; }
  const u = (x) => (x && x.w + x.l ? unitText(x.units) : "");
  const tile = (label, rec, extra = "", cls = "") => `<div class="${cls}"><b>${rec && rec.w + rec.l + (rec.p || 0) ? wl(rec) : "–"}</b><span>${label}${rec && rec.pct !== null && rec.pct !== undefined ? `, ${pct(rec.pct)}` : ""}</span>${extra}</div>`;
  const row = (title, sub, d) => `<div class="rec-row"><h2>${title}${sub ? ` <span>${sub}</span>` : ""}</h2><div class="tally">
      ${tile("★ Our picks", d.ours, u(d.ours), "ours")}
      ${tile("All spread picks", d.ats, u(d.ats) + (d.atsClose ? `<span class="close-note">closing line ${wl(d.atsClose)}</span>` : ""))}
      ${tile("Picking winners", d.su)}
      ${tile("Over/under", d.ou, u(d.ou))}</div></div>`;
  let html = "";
  const w = sum.week;
  if (w) html += row(`Week ${w.week} results`, `${w.games} games`, { ours: w.our_picks, ats: w.locked.spread, atsClose: w.close.spread, su: w.close.winner, ou: w.locked.total });
  html += row(`${state.results.season} season`, "", { ours: sum.our_picks, ats: sum.season_locked.spread, atsClose: sum.season.spread, su: sum.season.winner, ou: sum.season_locked.total });
  box.innerHTML = html + `<p class="rec-note">Spreads and totals are graded at the early line, when you'd bet (the locked Sunday–Tuesday line, or the opener). ★ Our picks are 5+ point edges from week 5 on (weeks 1–4 rely too much on preseason estimates). Units assume −110.</p>`;
  box.hidden = false;
  $("#season-line").hidden = true;  // the tiles replace the one-line summary
}
