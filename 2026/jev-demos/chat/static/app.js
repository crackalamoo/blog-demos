/* Jev chat demo -- front end.
 *
 * ChatGPT's side is appended text.  Jev's side is a choice: the candidates it
 * was offered appear in the chooser as it is asked, then each takes the
 * probability Jev gave it and the chosen one settles into the thread.
 * Hovering a settled turn brings its distribution back.
 *
 * One click runs one round -- a ChatGPT turn and Jev's reply -- and the client
 * carries the resume index, so the server holds no per-session state.
 */
'use strict';

/* The demo is mounted under a prefix (/chat/), so URLs resolve against the
 * page rather than the site root. */
const BASE = location.pathname.replace(/[^/]*$/, '');

const PANEL_W = 268;            // widest the panel gets
const PANEL_MIN_W = 186;        // narrowest before it is worth reading
const PANEL_GAP = 22;           // between the panel and the message column
const PANEL_MARGIN = 16;        // smallest gap to the viewport edge

/* Above this many candidates the rows tighten, so ten of them still stand a
 * chance of fitting the viewport without scrolling. */
const DENSE_FROM = 6;

/* How long the chooser lingers after a turn settles.  The rows hold whole
 * sentences now, so this is a reading pause, not a flourish. */
const CHOOSER_HOLD_MS = 5000;

const el = {
  main: document.querySelector('main'),
  thread: document.getElementById('thread'),
  chooser: document.getElementById('chooser'),
  run: document.getElementById('btn-run'),
};

const state = {
  source: null,
  meta: null,
  turns: new Map(),      // turn index -> {node, body, speaker}
  rows: new Map(),       // candidate text -> the chooser row showing it
  live: null,            // the message node currently holding the chooser
  holdTimer: 0,
  round: 0,              // the round the next click runs
  streaming: false,      // while true the running turn owns the chooser
};

/* ------------------------------------------------------------------ thread */

function atBottom() {
  return el.main.scrollHeight - el.main.scrollTop - el.main.clientHeight < 80;
}

function follow(wasAtBottom) {
  if (wasAtBottom) el.main.scrollTop = el.main.scrollHeight;
}

function openTurn(turn, speaker) {
  const node = document.createElement('div');
  node.className = `msg ${speaker}`;
  const body = document.createElement('div');
  node.appendChild(body);
  el.thread.appendChild(node);
  state.turns.set(turn, { node, body, speaker });
  el.main.scrollTop = el.main.scrollHeight;
}

function appendChunk(turn, text) {
  const stuck = atBottom();
  state.turns.get(turn).body.append(text);
  follow(stuck);
}

function settleChoice(turn, data) {
  const stuck = atBottom();
  const { node, body } = state.turns.get(turn);
  body.textContent = data.chosen;
  body.classList.add('fresh');
  node.style.setProperty('--conf', data.confidence.toFixed(3));
  if (data.low) node.classList.add('low');
  node.dataset.choice = JSON.stringify(data);
  node.addEventListener('mouseenter', () => {
    if (state.streaming) return;           // the running turn owns the chooser
    showChooser(node, JSON.parse(node.dataset.choice));
  });
  follow(stuck);
}

/* ----------------------------------------------------------------- chooser */

/* The candidates arrive before Jev is asked, so they are laid out first with
 * their meters at rest and filled once the probabilities come back. */
function openChooser(anchor, candidates) {
  clearTimeout(state.holdTimer);
  state.live = anchor;
  state.rows.clear();
  el.chooser.replaceChildren();
  el.chooser.classList.remove('low', 'settled');
  el.chooser.classList.toggle('dense', candidates.length > DENSE_FROM);

  for (const text of candidates) {
    const row = document.createElement('div');
    row.className = 'cand';
    const label = document.createElement('span');
    label.className = 'opt';
    label.textContent = text;
    const meter = document.createElement('span');
    meter.className = 'meter';
    meter.appendChild(document.createElement('i'));
    const p = document.createElement('span');
    p.className = 'p';
    row.append(label, meter, p);
    el.chooser.appendChild(row);
    state.rows.set(text, row);
  }

  placeChooser(anchor);
  el.chooser.classList.add('on');
}

function fillChooser(data) {
  el.chooser.classList.toggle('low', !!data.low);
  el.chooser.classList.add('settled');
  for (const cand of data.ranked) {
    const row = state.rows.get(cand.text);
    row.classList.toggle('is-chosen', cand.text === data.chosen);
    row.querySelector('.meter i').style.width =
      `${Math.min(100, cand.p * 100).toFixed(1)}%`;
    row.querySelector('.p').textContent =
      (cand.p * 100).toFixed(cand.p < 0.1 ? 1 : 0);
    el.chooser.appendChild(row);          // reorder: likeliest first
  }
  placeChooser(state.live);
}

function showChooser(anchor, data) {
  openChooser(anchor, data.ranked.map((c) => c.text));
  fillChooser(data);
}

function placeChooser(anchor) {
  const message = anchor.getBoundingClientRect();
  const column = el.thread.getBoundingClientRect();
  /* Jev sits at the right edge of the column, so its candidates sit beside it;
   * whichever gutter is wider takes the panel, and the panel narrows to fit
   * that gutter rather than overlapping the column. */
  const right = window.innerWidth - column.right - PANEL_GAP - PANEL_MARGIN;
  const left = column.left - PANEL_GAP - PANEL_MARGIN;
  const width = Math.min(PANEL_W, Math.max(PANEL_MIN_W, Math.max(right, left)));
  el.chooser.style.width = `${Math.round(width)}px`;
  el.chooser.style.maxHeight = `${window.innerHeight - 2 * PANEL_MARGIN}px`;

  const x = right >= left
    ? Math.min(column.right + PANEL_GAP, window.innerWidth - width - PANEL_MARGIN)
    : Math.max(PANEL_MARGIN, column.left - width - PANEL_GAP);
  const height = el.chooser.offsetHeight;
  const y = Math.max(
    PANEL_MARGIN,
    Math.min(message.top - 10, window.innerHeight - height - PANEL_MARGIN),
  );
  el.chooser.style.left = `${Math.round(x)}px`;
  el.chooser.style.top = `${Math.round(y)}px`;
}

function hideChooser(delay) {
  clearTimeout(state.holdTimer);
  state.holdTimer = setTimeout(() => {
    el.chooser.classList.remove('on');
    state.live = null;
  }, delay);
}

document.addEventListener('mouseout', (ev) => {
  if (!state.streaming && ev.target === state.live) hideChooser(160);
});

/* ------------------------------------------------------------------ stream */

function fail(message) {
  const node = document.createElement('div');
  node.className = 'failed';
  node.textContent = message;
  el.thread.appendChild(node);
  el.main.scrollTop = el.main.scrollHeight;
  closeStream();
  spend();
}

function spend() {
  el.run.disabled = true;
  el.run.classList.add('spent');
}

function runRound() {
  el.chooser.classList.remove('on');
  state.live = null;
  state.streaming = true;
  el.run.disabled = true;

  const source = new EventSource(`${BASE}api/stream?round=${state.round}`);
  state.source = source;

  source.addEventListener('meta', (ev) => { state.meta = JSON.parse(ev.data); });

  source.addEventListener('turn_start', (ev) => {
    const d = JSON.parse(ev.data);
    openTurn(d.turn, d.speaker);
  });

  source.addEventListener('chunk', (ev) => {
    const d = JSON.parse(ev.data);
    appendChunk(d.turn, d.text);
  });

  source.addEventListener('candidates', (ev) => {
    const d = JSON.parse(ev.data);
    openChooser(state.turns.get(d.turn).node, d.candidates);
  });

  source.addEventListener('choice', (ev) => {
    const d = JSON.parse(ev.data);
    fillChooser(d);
    settleChoice(d.turn, d);
  });

  source.addEventListener('done', (ev) => {
    const d = JSON.parse(ev.data);
    state.streaming = false;
    hideChooser(CHOOSER_HOLD_MS);
    closeStream();
    if (d.next_round === null) {
      spend();
    } else {
      state.round = d.next_round;
      el.run.textContent = 'next';
      el.run.disabled = false;
    }
  });

  source.addEventListener('error', (ev) => {
    /* Named `error` events come from the server with a reason; the bare
     * EventSource error has no data and means the connection dropped. */
    fail(ev.data ? JSON.parse(ev.data).error : 'stream dropped');
  });
}

function closeStream() {
  if (state.source) state.source.close();
  state.source = null;
  state.streaming = false;
}

el.run.addEventListener('click', runRound);

window.addEventListener('resize', () => {
  if (state.live) placeChooser(state.live);
});
