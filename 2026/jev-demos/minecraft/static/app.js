/* Jev minecraft demo -- front end.
 *
 * The left pane is prismarine-viewer, served by the bot process itself, with
 * the world's seed and the milestones reached so far along its bottom edge.
 * The right pane is the decision: the state the bot read off the world, and
 * the actions that were legal when it read it, ranked by the probability
 * Jev gave each one, with the chosen one marked.  Above the state is the
 * current objective, written by ChatGPT and carried in that state.
 */
'use strict';

const BASE = location.pathname.replace(/[^/]*$/, '');

/* ?layout=panel drops the viewer and stacks the readouts into one tall
 * column, for compositing beside a real Minecraft client's video. */
const PANEL = new URLSearchParams(location.search).get('layout') === 'panel';
const PANEL_ACTIONS = 6;

const el = {
  viewer: document.getElementById('viewer'),
  state: document.getElementById('state'),
  actions: document.getElementById('actions'),
  seed: document.getElementById('seed'),
  milestones: document.getElementById('milestones'),
  objective: document.getElementById('objective'),
  note: document.getElementById('note'),
  objectiveModel: document.getElementById('objective-model'),
  choiceModel: document.getElementById('choice-model'),
  confidence: document.getElementById('confidence'),
  confidenceFill: document.getElementById('confidence-fill'),
  actionCount: document.getElementById('action-count'),
  usage: document.getElementById('usage'),
};

if (PANEL) {
  document.documentElement.dataset.layout = 'panel';
  el.viewer.remove();
}

function pct(p) {
  if (p >= 0.995) return '100%';
  if (p > 0 && p < 0.005) return '<1%';
  return (p * 100).toFixed(p < 0.095 ? 1 : 0) + '%';
}

function renderNote(note) {
  const text = note ? 'what is going wrong: ' + note : '';
  if (el.note.textContent === text) return;
  el.note.textContent = text;
  el.note.classList.toggle('on', Boolean(note));
}

function renderObjective(objective) {
  if (!objective || el.objective.textContent === objective) return;
  el.objective.textContent = objective;
  el.objective.classList.remove('fresh');
  void el.objective.offsetWidth;
  el.objective.classList.add('fresh');
}

function colour(json) {
  return json
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/"([^"]*)":/g, '<span class="k">"$1"</span>:')
    .replace(/: "([^"]*)"/g, ': <span class="s">"$1"</span>')
    .replace(/: (-?\d+(?:\.\d+)?|true|false|null)/g,
      ': <span class="n">$1</span>');
}

function renderState(state) {
  el.state.innerHTML = colour(JSON.stringify(state, null, 2));
}

function renderActions(actions, chosen, probabilities) {
  el.actions.textContent = '';
  const ranked = actions.slice().sort(
    (a, b) => probabilities[b.label] - probabilities[a.label]);
  const top = probabilities[ranked[0].label] || 1;
  const shown = PANEL ? ranked.slice(0, PANEL_ACTIONS) : ranked;
  if (PANEL && !shown.some((a) => a.label === chosen)) {
    shown.push(ranked.find((a) => a.label === chosen));
  }
  for (const action of shown) {
    const p = probabilities[action.label];
    const row = document.createElement('li');
    row.className = 'action' + (action.label === chosen ? ' chosen' : '');

    const value = document.createElement('div');
    value.className = 'value';
    value.textContent = pct(p);
    row.appendChild(value);

    const bar = document.createElement('div');
    bar.className = 'p';
    const fill = document.createElement('i');
    fill.style.width = Math.max(p / top * 100, 1.5) + '%';
    bar.appendChild(fill);
    row.appendChild(bar);

    const label = document.createElement('div');
    label.className = 'label';
    label.textContent = action.label;
    if (action.label === chosen) {
      const mark = document.createElement('span');
      mark.className = 'mark';
      mark.textContent = 'chosen';
      label.appendChild(mark);
    }
    row.appendChild(label);

    const why = document.createElement('div');
    why.className = 'why';
    why.textContent = action.description;
    row.appendChild(why);

    el.actions.appendChild(row);
  }

  const hidden = ranked.filter((a) => !shown.includes(a));
  if (hidden.length) {
    const rest = document.createElement('li');
    rest.className = 'more';
    rest.textContent = hidden.length + ' more legal actions, none above '
      + pct(probabilities[hidden[0].label]);
    el.actions.appendChild(rest);
  }
}

function renderVerdict(decision) {
  const c = decision.confidence;
  el.confidence.textContent = c.toFixed(2);
  el.confidenceFill.style.width = (c * 100).toFixed(1) + '%';
  el.actionCount.textContent = decision.actions.length
    + ' legal actions \u00b7 decision ' + decision.decision;
  el.usage.textContent = decision.usage.input_tokens.toLocaleString()
    + ' input tokens';
}

function showViewer(reported) {
  if (PANEL) return;
  // The bot names itself 127.0.0.1, which is the wrong machine whenever the
  // page is open from anywhere but the host.
  const parsed = new URL(reported);
  parsed.hostname = location.hostname;
  const url = parsed.href;
  if (el.viewer.dataset.url === url) return;
  el.viewer.dataset.url = url;
  el.viewer.src = url;
  el.viewer.classList.add('live');
}

function renderMilestone(milestone) {
  const row = document.createElement('li');
  row.className = 'milestone'
    + (milestone.name === 'death' ? ' death' : '');
  const name = document.createElement('span');
  name.textContent = milestone.detail === null
    ? milestone.name
    : milestone.name + ': ' + milestone.detail;
  row.appendChild(name);
  const at = document.createElement('span');
  at.className = 'at';
  at.textContent = ' \u00b7 ' + milestone.decision;
  row.appendChild(at);
  el.milestones.appendChild(row);
  el.milestones.scrollTop = el.milestones.scrollHeight;
}

const source = new EventSource(BASE + 'api/stream');

source.addEventListener('meta', (event) => {
  const meta = JSON.parse(event.data);
  el.objectiveModel.textContent = 'written by ' + meta.objective_model.name;
  el.choiceModel.textContent = 'chosen by ' + meta.model.name;
  renderObjective(meta.objective);
  renderNote(meta.note);
});

source.addEventListener('session', (event) => {
  const session = JSON.parse(event.data);
  el.seed.textContent = 'seed ' + session.seed;
  el.milestones.textContent = '';
  el.objective.textContent = 'waiting for the first objective';
  showViewer(session.viewer);
});

source.addEventListener('milestone', (event) => {
  renderMilestone(JSON.parse(event.data));
});

source.addEventListener('decision', (event) => {
  const decision = JSON.parse(event.data);
  renderObjective(decision.objective);
  renderNote(decision.note);
  renderState(decision.state);
  renderVerdict(decision);
  renderActions(decision.actions, decision.chosen, decision.probabilities);
});
