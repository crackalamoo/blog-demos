/* Jev dim demo -- front end.
 *
 * Paragraphs are judged in chunks of neighbors, one request per chunk, and
 * each verdict is applied the moment its chunk lands, so the page resolves in
 * patches rather than all at once.  A new query drops the stream, which is
 * what cancels the sweep on the server too.
 */
'use strict';

/* The demo is mounted under a prefix (/dim/), so URLs resolve against the
 * page rather than the site root. */
const BASE = location.pathname.replace(/[^/]*$/, '');

const DEBOUNCE_MS = 500;

const MIN_OPACITY = 0.085;   // receded, but the shape of the text remains

/* A logistic on the probability rather than the probability itself: the
 * middle of the range is pushed to one end or the other, so a query leaves a
 * sparse page instead of a uniformly grey one. */
const STEEP = 12;
const MIDPOINT = 0.5;

const HIT = 0.5;             // at or above this, the paragraph is a match

const el = {
  header: document.querySelector('header'),
  query: document.getElementById('query'),
  story: document.getElementById('story'),
  count: document.getElementById('count'),
  prev: document.getElementById('prev'),
  next: document.getElementById('next'),
};

const state = {
  source: null,
  paragraphs: [],
  asked: '',
  debounce: 0,
  hits: [],
  current: -1,       // a paragraph index, not a place in `hits`
};

function weight(p) {
  const s = (x) => 1 / (1 + Math.exp(-STEEP * (x - MIDPOINT)));
  const lo = s(0);
  return (s(p) - lo) / (s(1) - lo);
}

function apply(index, p) {
  const node = state.paragraphs[index];
  const w = weight(p);
  node.style.setProperty('--o', (MIN_OPACITY + (1 - MIN_OPACITY) * w)
    .toFixed(3));
  node.classList.toggle('hit', p >= HIT);
  if (p >= HIT) insert(index);
}

/* The hit list stays in story order however the verdicts arrive, and the
 * current hit is held by paragraph index so that a hit landing above it only
 * moves its number in the counter, never the reader's place. */
function insert(index) {
  let at = 0;
  while (at < state.hits.length && state.hits[at] < index) at++;
  if (state.hits[at] === index) return;
  state.hits.splice(at, 0, index);
  count();
}

function count() {
  const at = state.hits.indexOf(state.current);
  el.count.textContent = `${at + 1} / ${state.hits.length}`;
}

function step(delta) {
  if (!state.hits.length) return;
  const at = state.hits.indexOf(state.current);
  const n = state.hits.length;
  const to = at < 0 ? (delta > 0 ? 0 : n - 1) : (at + delta + n) % n;
  mark(state.hits[to]);
}

function mark(index) {
  const node = state.paragraphs[index];
  if (!node) throw new Error(`no paragraph ${index}`);
  const was = state.paragraphs[state.current];
  if (was) was.classList.remove('current');
  node.classList.add('current');
  state.current = index;
  node.scrollIntoView({behavior: 'smooth', block: 'center'});
  count();
}

function reset() {
  for (const node of state.paragraphs) {
    node.style.removeProperty('--o');
    node.classList.remove('hit', 'current');
  }
  state.hits = [];
  state.current = -1;
  count();
  const failed = el.story.querySelector('.failed');
  if (failed) failed.remove();
}

function fail(message) {
  let node = el.story.querySelector('.failed');
  if (!node) {
    node = document.createElement('p');
    node.className = 'failed';
    el.story.prepend(node);
  }
  node.textContent = message;
}

function stop() {
  if (state.source) state.source.close();
  state.source = null;
  el.header.classList.remove('sweeping');
}

function sweep() {
  const query = el.query.value.trim();
  if (!query || query === state.asked) return;
  stop();
  state.asked = query;
  reset();
  el.header.classList.add('sweeping');

  const source = new EventSource(`${BASE}api/sweep?q=${encodeURIComponent(query)}`);
  state.source = source;

  source.addEventListener('judged', (ev) => {
    const d = JSON.parse(ev.data);
    apply(d.i, d.p);
  });

  source.addEventListener('done', () => stop());

  source.addEventListener('error', (ev) => {
    /* Named `error` events come from the server with a reason; the bare
     * EventSource error has no data and means the connection dropped. */
    if (source !== state.source) return;     // a newer query dropped this one
    stop();
    fail(ev.data ? JSON.parse(ev.data).error : 'sweep dropped');
  });
}

el.query.addEventListener('input', () => {
  clearTimeout(state.debounce);
  state.debounce = setTimeout(sweep, DEBOUNCE_MS);
});

el.query.addEventListener('keydown', (ev) => {
  if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') {
    ev.preventDefault();
    step(ev.key === 'ArrowDown' ? 1 : -1);
    return;
  }
  if (ev.key !== 'Enter') return;
  clearTimeout(state.debounce);
  if (el.query.value.trim() === state.asked) step(ev.shiftKey ? -1 : 1);
  else sweep();
});

el.prev.addEventListener('click', () => step(-1));
el.next.addEventListener('click', () => step(1));

fetch(`${BASE}api/story`)
  .then((r) => r.json())
  .then((story) => {
    for (const text of story.paragraphs) {
      const node = document.createElement('p');
      node.textContent = text;
      el.story.appendChild(node);
      state.paragraphs.push(node);
    }
  });
