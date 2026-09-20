/* Jev wiki demo -- front end.
 *
 * One SSE connection to the server, which holds one connection to the
 * Wikimedia firehose for exactly as long as this page is open.
 */
'use strict';

const BASE = location.pathname.replace(/[^/]*$/, '');
const SVG_NS = 'http://www.w3.org/2000/svg';

const MARGIN = { top: 16, right: 14, bottom: 48, left: 176 };
const R = 4.5;
const BAND = 10;
const RAIL_MAX = 8;
const DISHONEST_BELOW = 0.5;
const HOVER_RADIUS = 16;
const SID = crypto.randomUUID();

const el = {
  svg: document.getElementById('scatter'),
  tip: document.getElementById('tip'),
  rail: document.getElementById('rail'),
  detail: document.getElementById('detail'),
  stats: document.getElementById('stats'),
  pausebar: document.getElementById('pausebar'),
  resume: document.getElementById('resume'),
  skippedNow: document.getElementById('skipped-now'),
  skippedTotal: document.getElementById('skipped-total'),
  failed: document.getElementById('failed'),
  seenRate: document.getElementById('seen-rate'),
  scopeRate: document.getElementById('scope-rate'),
  throttled: document.getElementById('throttled'),
  judgedRate: document.getElementById('judged-rate'),
  totalJudged: document.getElementById('total-judged'),
  spend: document.getElementById('spend'),
  tokens: document.getElementById('tokens'),
};

const state = {
  labels: [],
  rungs: [],
  criteria: {},
  edits: new Map(),
  order: [],
  columns: [],
  selected: null,
  plot: null,
  hovered: null,
  paused: false,
};

function svg(tag, attrs) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    node.setAttribute(k, v);
  }
  return node;
}

function elem(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function colorOf(label) {
  return `var(--c-${label.replace(/ /g, '-')})`;
}

function signed(n) {
  return `${n > 0 ? '+' : ''}${n}`;
}

function plotBox() {
  const rect = el.svg.getBoundingClientRect();
  return {
    left: MARGIN.left,
    top: MARGIN.top,
    width: Math.max(320, rect.width - MARGIN.left - MARGIN.right),
    height: Math.max(200, rect.height - MARGIN.top - MARGIN.bottom),
  };
}

function yOf(score) {
  const p = state.plot;
  return p.top + (1 - score / 4) * p.height;
}

function drawAxes() {
  const p = state.plot;
  el.svg.replaceChildren();

  const axes = svg('g', {});
  for (let i = 0; i < 5; i += 1) {
    const y = yOf(i);
    axes.appendChild(svg('line', {
      class: 'grid', x1: p.left, x2: p.left + p.width, y1: y, y2: y,
    }));
    const rung = svg('text', {
      class: 'rung-label', x: p.left - 26, y: y + 4,
      'text-anchor': 'end',
    });
    rung.textContent = state.rungs[i];
    axes.appendChild(rung);
    const num = svg('text', {
      class: 'rung-number', x: p.left - 8, y: y + 4,
      'text-anchor': 'end',
    });
    num.textContent = i;
    axes.appendChild(num);
  }

  const colWidth = p.width / state.labels.length;
  state.labels.forEach((label, i) => {
    const x = p.left + i * colWidth;
    if (i > 0) {
      axes.appendChild(svg('line', {
        class: 'col-sep', x1: x, x2: x, y1: p.top, y2: p.top + p.height,
      }));
    }
    const name = svg('text', {
      class: 'col-label', x: x + colWidth / 2, y: p.top + p.height + 20,
      fill: colorOf(label),
    });
    name.textContent = label;
    const title = svg('title', {});
    title.textContent = state.criteria[label] || '';
    name.appendChild(title);
    axes.appendChild(name);

    const count = svg('text', {
      class: 'col-count', x: x + colWidth / 2, y: p.top + p.height + 36,
      id: `count-${i}`,
    });
    count.textContent = '0 edits · 0%';
    axes.appendChild(count);
  });

  el.svg.appendChild(axes);
  const dots = svg('g', { id: 'dots' });
  el.svg.appendChild(dots);
  return dots;
}

function makeDot(edit, index) {
  const g = svg('g', { class: 'dot enter' });
  g.appendChild(svg('circle', { class: 'halo', r: R + 4 }));
  if (edit.summary_honest.p < DISHONEST_BELOW) {
    g.appendChild(svg('circle', { class: 'ring', r: R + 2.6 }));
  }
  g.appendChild(svg('circle', {
    class: 'body', r: R, fill: colorOf(edit.action.label),
  }));
  g.dataset.id = edit.id;
  g.dataset.index = index;
  return g;
}

function columnCenter(i) {
  const p = state.plot;
  return p.left + (i + 0.5) * (p.width / state.labels.length);
}

function layoutColumn(i) {
  const p = state.plot;
  const colWidth = p.width / state.labels.length;
  const center = columnCenter(i);
  const bands = new Map();
  for (const dot of state.columns[i]) {
    const band = Math.round((yOf(dot.score) - p.top) / BAND);
    if (!bands.has(band)) bands.set(band, []);
    bands.get(band).push(dot);
  }
  for (const [band, dots] of bands) {
    const y = p.top + band * BAND;
    const step = Math.min(2 * R + 1.5, (colWidth - 10) / dots.length);
    dots.forEach((dot, k) => {
      const x = center + (k - (dots.length - 1) / 2) * step;
      dot.x = x;
      dot.y = y;
      dot.node.style.transform = `translate(${x.toFixed(1)}px, ` +
        `${y.toFixed(1)}px)`;
    });
  }
}

function updateCounts() {
  const total = state.order.length || 1;
  state.labels.forEach((label, i) => {
    const n = state.columns[i].length;
    const node = el.svg.querySelector(`#count-${i}`);
    node.textContent = `${n} edit${n === 1 ? '' : 's'} · ` +
      `${Math.round((n / total) * 100)}%`;
  });
}

function relayout() {
  state.plot = plotBox();
  const dots = drawAxes();
  state.columns.forEach((column) => {
    for (const dot of column) dots.appendChild(dot.node);
  });
  state.columns.forEach((_, i) => layoutColumn(i));
  updateCounts();
  markSelected();
}

function addDot(edit) {
  const i = state.labels.indexOf(edit.action.label);
  const node = makeDot(edit, i);
  el.svg.querySelector('#dots').appendChild(node);
  state.columns[i].push({
    node, edit, score: edit.misleading.score, x: columnCenter(i), y: 0,
  });
  node.style.transform =
    `translate(${columnCenter(i).toFixed(1)}px, ${state.plot.top - 26}px)`;
  node.getBoundingClientRect();
  requestAnimationFrame(() => {
    node.classList.remove('enter');
    layoutColumn(i);
  });
  updateCounts();
}

/* ---- the chronological rail: order, which the scatter cannot show -- */

function railLine(edit) {
  const row = elem('div', 'rail-row');
  row.dataset.id = edit.id;
  row.addEventListener('click', () => select(edit.id));
  const swatch = elem('span', 'rail-swatch');
  swatch.style.background = colorOf(edit.action.label);
  const title = elem('span', 'rail-title', edit.title);
  const cat = elem('span', 'rail-cat', edit.action.label);
  cat.style.color = colorOf(edit.action.label);
  const score = elem(
    'span', 'rail-score', edit.misleading.score.toFixed(2));
  row.append(swatch, title, cat, score);
  if (edit.summary_honest.p < DISHONEST_BELOW) {
    row.appendChild(elem('span', 'rail-flag', 'summary dishonest'));
  }
  return row;
}

function renderRail() {
  const ids = state.order.slice(-RAIL_MAX).reverse();
  el.rail.replaceChildren(
    ...ids.map((id) => railLine(state.edits.get(id))));
  markSelected();
}

function markSelected() {
  for (const node of el.svg.querySelectorAll('.dot')) {
    node.classList.toggle('selected', node.dataset.id === state.selected);
  }
  for (const node of el.rail.children) {
    node.classList.toggle('selected', node.dataset.id === state.selected);
  }
}

/* ---- the detail pane: one edit, all three judgments, persistent ---- */

function jdBlock(primitive, question) {
  const block = elem('div', 'jd-block');
  const head = elem('div', 'jd-block-head');
  head.append(
    elem('span', 'jd-prim', primitive), elem('span', 'jd-q', question));
  block.appendChild(head);
  return block;
}

function jdStat(key, value, note) {
  const wrap = elem('div', 'jd-stat');
  wrap.append(elem('span', 'jd-k', key), elem('span', 'jd-v', value));
  if (note) wrap.appendChild(elem('span', 'jd-note', note));
  return wrap;
}

function jdBars(caption, entries, chosen, color) {
  const wrap = elem('div', 'jd-dist');
  wrap.appendChild(elem('div', 'jd-cap', caption));
  for (const [name, p] of entries) {
    const row = elem('div', name === chosen ? 'jd-bar chosen' : 'jd-bar');
    const track = elem('span', 'jd-bar-track');
    const fill = elem('span', 'jd-bar-fill');
    fill.style.width = `${(p * 100).toFixed(1)}%`;
    if (color) fill.style.background = color;
    track.appendChild(fill);
    row.append(
      elem('span', 'jd-bar-label', name), track,
      elem('span', 'jd-bar-num', p.toFixed(3)));
    wrap.appendChild(row);
  }
  return wrap;
}

function betweenText(score) {
  const lo = Math.floor(score);
  const hi = Math.ceil(score);
  if (lo === hi) return `${lo} ${state.rungs[lo]} exactly`;
  return `${lo} ${state.rungs[lo]} → ${hi} ${state.rungs[hi]}`;
}

function diffLine(sign, text, kind) {
  const row = elem('div', kind);
  row.append(
    elem('span', 'sign', sign),
    elem('span', text ? 'text' : 'none', text || '(nothing)'));
  return row;
}

function emptyDetail() {
  const frag = document.createDocumentFragment();
  frag.appendChild(elem('div', 'jd-empty-title', 'No edit selected'));
  frag.appendChild(elem('div', 'jd-empty-body',
    'Click any dot in the scatter, or any line in the list under it. ' +
    'That pauses the live stream and shows the whole judgment of that ' +
    'one edit here: Choice, Score, Noul, and the diff it read.'));
  el.detail.replaceChildren(frag);
}

function renderDetail(edit) {
  const frag = document.createDocumentFragment();

  const head = elem('div', 'jd-head');
  head.appendChild(elem('div', 'jd-kicker', 'Selected edit'));
  const title = elem('a', 'jd-title', edit.title);
  title.href = edit.url;
  title.target = '_blank';
  title.rel = 'noreferrer';
  head.appendChild(title);
  head.appendChild(jdStat('editor', edit.user));
  head.appendChild(jdStat('byte delta', `${signed(edit.byte_delta)} bytes`));
  const sum = elem('div', 'jd-stat');
  sum.append(
    elem('span', 'jd-k', 'edit summary the editor wrote'),
    elem('span', 'jd-summary', edit.summary || '(none given)'));
  head.appendChild(sum);
  frag.appendChild(head);

  const choice = jdBlock('Choice', 'what is this edit doing?');
  choice.appendChild(jdStat('chosen label', edit.action.label));
  choice.appendChild(
    jdStat('confidence', edit.action.confidence.toFixed(3)));
  const actionEntries = Object.entries(edit.action.probabilities)
    .sort((a, b) => b[1] - a[1]);
  choice.appendChild(jdBars(
    'probability over all 7 labels', actionEntries, edit.action.label,
    colorOf(edit.action.label)));
  frag.appendChild(choice);

  const score = jdBlock(
    'Score', 'how misleading would this be to a reader?');
  score.appendChild(jdStat(
    'score', `${edit.misleading.score} of 4`,
    'probability-weighted mean, not rounded'));
  score.appendChild(
    jdStat('sits between rungs', betweenText(edit.misleading.score)));
  score.appendChild(
    jdStat('confidence', edit.misleading.confidence.toFixed(3)));
  const rungEntries = Object.entries(edit.misleading.probabilities)
    .sort((a, b) => Number(a[0]) - Number(b[0]))
    .map(([k, p]) => [`${k} · ${state.rungs[Number(k)]}`, p]);
  score.appendChild(jdBars(
    'probability over the 5 rubric rungs', rungEntries, null, null));
  frag.appendChild(score);

  const noul = jdBlock(
    'Noul', 'does the summary honestly describe the change?');
  noul.appendChild(jdStat(
    'probability summary is honest', edit.summary_honest.p.toFixed(4),
    'a Noul answer carries no confidence'));
  noul.appendChild(jdBars('probability', [
    ['honest', edit.summary_honest.p],
    ['dishonest', 1 - edit.summary_honest.p],
  ], null, null));
  frag.appendChild(noul);

  const diff = elem('div', 'jd-block');
  const dhead = elem('div', 'jd-block-head');
  dhead.append(
    elem('span', 'jd-prim plain', 'Diff'),
    elem('span', 'jd-q', 'the text Jev read, as Wikipedia rendered it'));
  diff.appendChild(dhead);
  const body = elem('div', 'jd-diff');
  body.appendChild(diffLine('+', edit.added, 'added'));
  body.appendChild(diffLine('−', edit.removed, 'removed'));
  diff.appendChild(body);
  frag.appendChild(diff);

  el.detail.replaceChildren(frag);
}

/* ---- pausing: the server stops judging, and drops what arrives ----- */

function setPaused(on) {
  fetch(`${BASE}api/pause?sid=${SID}&on=${on ? 1 : 0}`)
    .then((r) => {
      if (!r.ok) throw new Error(`pause failed: HTTP ${r.status}`);
    })
    .catch((exc) => fail(String(exc)));
}

function applyPaused(on) {
  state.paused = on;
  el.pausebar.hidden = !on;
  el.stats.classList.toggle('frozen', on);
  document.body.classList.toggle('is-paused', on);
}

function select(id) {
  state.selected = String(id);
  renderDetail(state.edits.get(state.selected));
  markSelected();
  if (!state.paused) {
    applyPaused(true);
    el.skippedNow.textContent = '0';
    setPaused(true);
  }
}

el.resume.addEventListener('click', () => {
  applyPaused(false);
  setPaused(false);
});

window.addEventListener('keydown', (ev) => {
  if (ev.key === 'Escape' && state.paused) el.resume.click();
});

/* ---- hover: a light preview, just enough to pick a dot ------------- */

function buildTip(edit) {
  const frag = document.createDocumentFragment();
  frag.appendChild(elem('div', 'tip-title', edit.title));
  const line = elem('div', 'tip-line');
  const cat = elem('span', 'tip-cat', edit.action.label);
  cat.style.color = colorOf(edit.action.label);
  line.append(
    cat, elem('span', 'tip-score',
      `Score ${edit.misleading.score.toFixed(2)} of 4`));
  frag.appendChild(line);
  frag.appendChild(elem('div', 'tip-hint', 'click to pause and inspect'));
  el.tip.replaceChildren(frag);
}

function placeTip(x, y) {
  const pad = 12;
  const w = el.tip.offsetWidth;
  const h = el.tip.offsetHeight;
  let left = x + 18;
  if (left + w + pad > window.innerWidth) left = x - 18 - w;
  left = Math.max(pad, Math.min(left, window.innerWidth - w - pad));
  let top = y - h / 2;
  top = Math.max(pad, Math.min(top, window.innerHeight - h - pad));
  el.tip.style.transform = `translate(${Math.round(left)}px, ` +
    `${Math.round(top)}px)`;
}

function nearestDot(x, y) {
  let best = null;
  let bestD = HOVER_RADIUS * HOVER_RADIUS;
  for (const column of state.columns) {
    for (const dot of column) {
      const d = (dot.x - x) ** 2 + (dot.y - y) ** 2;
      if (d <= bestD) {
        bestD = d;
        best = dot;
      }
    }
  }
  return best;
}

function hover(dot) {
  const id = dot ? dot.edit.id : null;
  if (id === state.hovered) return;
  state.hovered = id;
  for (const node of el.svg.querySelectorAll('.dot.hovered')) {
    node.classList.remove('hovered');
  }
  if (!dot) {
    el.tip.hidden = true;
    return;
  }
  dot.node.classList.add('hovered');
  buildTip(dot.edit);
  el.tip.hidden = false;
}

let pointer = null;
let pending = false;

function onPointer() {
  pending = false;
  if (!pointer || !state.plot) return;
  const rect = el.svg.getBoundingClientRect();
  const dot = nearestDot(pointer.x - rect.left, pointer.y - rect.top);
  hover(dot);
  if (dot) placeTip(pointer.x, pointer.y);
}

el.svg.addEventListener('pointermove', (ev) => {
  pointer = { x: ev.clientX, y: ev.clientY };
  if (pending) return;
  pending = true;
  requestAnimationFrame(onPointer);
});

el.svg.addEventListener('click', (ev) => {
  if (!state.plot) return;
  const rect = el.svg.getBoundingClientRect();
  const dot = nearestDot(ev.clientX - rect.left, ev.clientY - rect.top);
  if (dot) select(dot.edit.id);
});

el.svg.addEventListener('pointerleave', () => {
  pointer = null;
  hover(null);
});

/* ---- the stream ---------------------------------------------------- */

function onEdit(edit) {
  edit.id = String(edit.id);
  state.edits.set(edit.id, edit);
  state.order.push(edit.id);
  addDot(edit);
  renderRail();
}

function onRates(d) {
  el.seenRate.textContent = d.seen_per_sec.toFixed(2);
  el.scopeRate.textContent = d.in_scope_per_sec.toFixed(2);
  el.judgedRate.textContent = d.judged_per_sec.toFixed(2);
  el.throttled.textContent =
    `per second · ${d.throttled} throttled by Wikipedia`;
  el.totalJudged.textContent = d.total_judged;
  el.skippedTotal.textContent = d.skipped_while_paused;
  el.spend.textContent = `$${d.spend_usd.toFixed(6)}`;
  el.tokens.textContent =
    `${d.input_tokens} in / ${d.output_tokens} out tokens`;
}

function onPause(d) {
  applyPaused(d.paused);
  el.skippedNow.textContent = d.skipped_now;
  el.skippedTotal.textContent = d.skipped_total;
}

function fail(message) {
  el.failed.hidden = false;
  el.failed.textContent = message;
}

emptyDetail();

const source = new EventSource(`${BASE}api/stream?sid=${SID}`);

source.addEventListener('meta', (ev) => {
  const meta = JSON.parse(ev.data);
  state.labels = meta.action_labels;
  state.rungs = meta.misleading_rungs;
  state.criteria = meta.action_criteria;
  state.columns = state.labels.map(() => []);
  relayout();
});

source.addEventListener('edit', (ev) => onEdit(JSON.parse(ev.data)));

source.addEventListener('rates', (ev) => onRates(JSON.parse(ev.data)));

source.addEventListener('paused', (ev) => onPause(JSON.parse(ev.data)));

source.addEventListener('error', (ev) => {
  fail(ev.data ? JSON.parse(ev.data).error : 'stream dropped');
});

window.addEventListener('resize', () => {
  if (state.plot) relayout();
});
