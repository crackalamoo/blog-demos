/* Jev readout demo -- front end.
 *
 * One rendered landing page at a time; the arrows move through them.  The
 * rows in the panel are the ten questions, all answered by one request.
 */
'use strict';

const BASE = location.pathname.replace(/[^/]*$/, '');

const el = {
  name: document.getElementById('name'),
  site: document.getElementById('site'),
  url: document.getElementById('url'),
  origin: document.getElementById('origin'),
  model: document.getElementById('model'),
  qcount: document.getElementById('qcount'),
  rows: document.getElementById('rows'),
  count: document.getElementById('count'),
  prev: document.getElementById('prev'),
  next: document.getElementById('next'),
};

const DESIGN_WIDTH = 1280;

function fit() {
  const box = document.querySelector('.viewport');
  document.documentElement.style.setProperty(
    '--zoom', String(box.clientWidth / DESIGN_WIDTH));
}

// Display names for the ten questions in questions.py, in order. Keys
// are the function names; rungs and options are the Score's and
// Choice's own, as Jev returns them.
const QUESTIONS = [
  {
    key: 'usable_today', kind: 'noul',
    label: 'Can you use this today?',
  },
  {
    key: 'has_price', kind: 'noul',
    label: 'Is there a price on the page?',
  },
  {
    key: 'names_customer', kind: 'noul',
    label: 'Does it name a real customer?',
  },
  {
    key: 'says_what_it_does', kind: 'noul',
    label: 'Does it say what it does?',
  },
  {
    key: 'readiness', kind: 'score',
    label: 'How ready is it, really?',
    rungs: ['idea', 'prototype', 'private beta', 'shipping', 'mature'],
  },
  {
    key: 'shipped_vs_roadmap', kind: 'score',
    label: 'Shipped, or roadmap?',
    rungs: ['all roadmap', 'mostly roadmap', 'half', 'mostly shipped',
            'all shipped'],
  },
  {
    key: 'concreteness', kind: 'score',
    label: 'How concrete are the claims?',
    rungs: ['adjectives', 'vague', 'mixed', 'specific', 'numbers'],
  },
  {
    key: 'category', kind: 'choice',
    label: 'What is this, actually?',
    options: ['dev tool', 'consumer app', 'marketplace', 'consultancy',
              'research project', 'other'],
  },
  {
    key: 'omission', kind: 'choice',
    label: 'What is most conspicuously missing?',
    options: ['price', 'what it does', 'who built it', 'proof it works',
              'nothing - the page is complete'],
  },
  {
    key: 'coyness', kind: 'score',
    label: 'How coy is the page?',
    rungs: ['candid', 'reticent', 'evasive', 'coy', 'hiding'],
  },
];

const state = {
  pages: [],
  at: 0,
  source: null,
  nodes: {},
};

function pct(p) {
  return (p * 100).toFixed(0) + '%';
}

function three(x) {
  return x.toFixed(3);
}

function bar(track, p) {
  requestAnimationFrame(() => {
    track.firstChild.style.width = pct(p);
  });
}

function add(parent, tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  parent.appendChild(node);
  return node;
}

function track(parent) {
  const t = add(parent, 'div', 'track');
  add(t, 'div', 'fill');
  return t;
}

function buildRow(q) {
  const row = add(el.rows, 'div', 'row');
  const head = add(row, 'div', 'head');
  add(head, 'div', 'q', q.label);
  add(head, 'div', 'kind', q.kind);
  const body = add(row, 'div', 'body');
  state.nodes[q.key] = { row, body, q };
}

function fillNoul(node, d) {
  const wrap = add(node.body, 'div', 'noul');
  const t = track(wrap);
  add(wrap, 'div', 'val', three(d.p));
  bar(t, d.p);
}

function fillScore(node, d) {
  const ladder = add(node.body, 'div', 'ladder');
  const nearest = Math.round(d.score);
  node.q.rungs.forEach((name, i) => {
    const rung = add(ladder, 'div', 'rung' + (i === nearest ? ' at' : ''));
    add(rung, 'div', 'i', String(i));
    add(rung, 'div', 'nm', name);
    const t = track(rung);
    const p = d.probabilities[String(i)];
    add(rung, 'div', 'pv', pct(p));
    bar(t, p);
  });
  const tail = add(node.body, 'div', 'tail');
  add(tail, 'div', 'big', 'score ' + d.score.toFixed(2));
  add(tail, 'div', null, 'confidence ' + d.confidence.toFixed(2));
}

function fillChoice(node, d) {
  const opts = add(node.body, 'div', 'opts');
  for (const name of node.q.options) {
    const opt = add(opts, 'div',
      'opt' + (name === d.choice ? ' chosen' : ''));
    add(opt, 'div', 'nm', name);
    const t = track(opt);
    const p = d.probabilities[name];
    add(opt, 'div', 'pv', pct(p));
    bar(t, p);
  }
  const tail = add(node.body, 'div', 'tail');
  add(tail, 'div', 'big', d.choice);
  add(tail, 'div', null, 'confidence ' + d.confidence.toFixed(2));
}

function fill(d) {
  const node = state.nodes[d.key];
  node.body.textContent = '';
  if (d.kind === 'noul') fillNoul(node, d);
  else if (d.kind === 'score') fillScore(node, d);
  else fillChoice(node, d);
  node.row.classList.add('filled');
}

function fail(message) {
  let node = el.rows.querySelector('.failed');
  if (!node) node = add(el.rows, 'div', 'failed');
  node.textContent = message;
}

function show(index) {
  if (state.source) state.source.close();
  state.at = (index + state.pages.length) % state.pages.length;
  const page = state.pages[state.at];

  el.name.textContent = page.name;
  el.site.src = page.site;
  el.url.textContent = page.live
    ? page.site
    : 'https://' + page.name.toLowerCase().replace(/[^a-z]/g, '') + '.example';
  el.origin.textContent = page.live ? 'live site' : 'rendered specimen';
  el.origin.className = 'origin' + (page.live ? ' live' : '');
  el.count.textContent =
    String(state.at + 1).padStart(2, '0') + ' / ' +
    String(state.pages.length).padStart(2, '0');
  el.rows.scrollTop = 0;

  el.rows.textContent = '';
  state.nodes = {};
  for (const q of QUESTIONS) buildRow(q);

  const source = new EventSource(
    BASE + 'api/read?page=' + encodeURIComponent(page.id));
  state.source = source;

  source.addEventListener('judged', (ev) => fill(JSON.parse(ev.data)));
  source.addEventListener('done', () => source.close());
  source.addEventListener('error', (ev) => {
    if (source !== state.source) return;
    source.close();
    fail(ev.data ? JSON.parse(ev.data).error : 'read dropped');
  });
}

window.addEventListener('resize', fit);
fit();

el.prev.addEventListener('click', () => show(state.at - 1));
el.next.addEventListener('click', () => show(state.at + 1));

document.addEventListener('keydown', (ev) => {
  if (ev.key === 'ArrowLeft') show(state.at - 1);
  if (ev.key === 'ArrowRight') show(state.at + 1);
});

fetch(BASE + 'api/pages')
  .then((r) => r.json())
  .then((data) => {
    state.pages = data.pages;
    el.model.textContent = 'model ' + data.model.name;
    el.qcount.textContent = QUESTIONS.length + ' judgments per page';
    show(0);
  });
