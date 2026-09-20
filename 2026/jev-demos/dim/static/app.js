/* Jev dim demo -- front end.
 *
 * Every paragraph is judged against the query on its own, and each verdict is
 * applied the moment it lands, so the page resolves as a wave rather than all
 * at once.  A new query drops the stream, which is what cancels the sweep on
 * the server too.
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

const el = {
  header: document.querySelector('header'),
  query: document.getElementById('query'),
  story: document.getElementById('story'),
};

const state = {
  source: null,
  paragraphs: [],
  asked: '',
  debounce: 0,
};

function weight(p) {
  const s = (x) => 1 / (1 + Math.exp(-STEEP * (x - MIDPOINT)));
  const lo = s(0);
  return (s(p) - lo) / (s(1) - lo);
}

function apply(node, p) {
  const w = weight(p);
  node.style.setProperty('--o', (MIN_OPACITY + (1 - MIN_OPACITY) * w)
    .toFixed(3));
}

function reset() {
  for (const node of state.paragraphs) {
    node.style.removeProperty('--o');
  }
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
    apply(state.paragraphs[d.i], d.p);
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
  if (ev.key !== 'Enter') return;
  clearTimeout(state.debounce);
  sweep();
});

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
