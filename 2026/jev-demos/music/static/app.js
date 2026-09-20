/* Jev music demo -- front end.
 *
 * Decisions are buffered per bar and revealed a whole bar at a time: all 24
 * slots of a bar are decided in one forward pass.  Reveal timing is cosmetic,
 * so setTimeout is fine there.
 *
 * Audio is scheduled on Tone.Transport in musical time (bars:beats:sixteenths).
 * No setTimeout touches audio -- it drifts audibly over even a few bars.
 *
 * The stream is endless: bars are scheduled and painted as they arrive, the
 * roll translates under a fixed playhead, and bars behind it are culled.
 */
'use strict';

const CELL_W = 16;          // pixels per 16th-note step
const ROW_H = 6;            // pixels per semitone row
const PITCH_ROWS = 24;      // n0 .. n23
const REST_STRIP = 10;      // extra strip under each lane for rest markers
const SLOTS_PER_BAR = 16;

const VOICE_COLORS = {
  melody: 'var(--melody)',
  harmony: 'var(--harmony)',
};

const LOW_CONFIDENCE = 0.5;

/* With a live model, becomes the measured round-trip of the bar's request. */
const BAR_PAUSE_MS = 450;

const GHOST_HOLD_MS = 320;

/* Where the playhead sits, as a fraction of the roll viewport. */
const PLAYHEAD_ANCHOR = 0.4;

/* Empty bars drawn past the newest one, so the grid always bleeds off the
 * right edge instead of stopping in mid-air. */
const LEAD_BARS = 4;

/* Bars kept behind the playhead before their nodes are dropped. */
const KEEP_BEHIND_BARS = 3;

/* Lane width is a constant.  Resizing it resizes #roll's compositing layer,
 * which re-rasterizes the whole masked roll for a frame -- that reads as a
 * flash of the entire roll on every new bar.  The drawn window is held inside
 * this many bars instead, by dropping bars off the left. */
const WINDOW_BARS = KEEP_BEHIND_BARS + LEAD_BARS + 8;

/* The trace grows forever otherwise. */
const TRACE_MAX_ENTRIES = 120;

/* Releases land this much early, so a note never overlaps the next attack of
 * the same pitch in the same voice (one PolySynth voice, two owners). */
const RELEASE_GAP_S = 0.02;

/* The demo is mounted under a prefix (/music/), so every URL it fetches is
 * resolved against the page rather than the site root. */
const BASE = location.pathname.replace(/[^/]*$/, '');

const el = {
  scroll: document.querySelector('.roll-scroll'),
  roll: document.getElementById('roll'),
  playhead: document.getElementById('playhead'),
  trace: document.getElementById('trace'),
  transport: document.getElementById('btn-transport'),
};

const state = {
  meta: null,
  source: null,
  run: 0,                // bumped per run; stale reveal loops see it and stop
  barQueue: [],          // bars buffered by the stream, awaiting reveal
  pending: null,         // {bar, decisions} currently arriving
  revealing: false,
  lastRevealAt: 0,
  revealedBars: 0,
  finalBar: null,        // set only by `done`, i.e. a bounded run
  lanes: new Map(),      // voice name -> {voice, gridEl, sounding, bars}
  originBar: 0,          // bar drawn at x = 0; rebased as bars are culled
  drawnThroughBar: -1,
  sounding: new Map(),   // note id -> live note, for extending holds
  playing: false,
  stepSeconds: 1,
  synths: null,
  rafHandle: null,
};

function transport() {
  return Tone.getTransport ? Tone.getTransport() : Tone.Transport;
}

function optionRow(option) {
  // Row 0 is drawn at the bottom of the lane, so higher pitch sits higher.
  return PITCH_ROWS - 1 - Number(option.slice(1));
}

function bbs(step) {
  const bar = Math.floor(step / SLOTS_PER_BAR);
  const beat = Math.floor((step % SLOTS_PER_BAR) / 4);
  const sixteenth = step % 4;
  return `${bar}:${beat}:${sixteenth}`;
}

/* `busy`: something is running and a click will stop it. */
function setBusy(busy) {
  el.transport.classList.toggle('busy', busy);
  el.transport.setAttribute('aria-label', busy ? 'stop' : 'play');
}

/* ------------------------------------------------------------- piano roll */

function buildRoll(meta) {
  el.roll.innerHTML = '';
  el.roll.style.transform = '';
  state.lanes.clear();
  /* Nothing is drawn before bar 0: an empty grid bleeding off the left edge
   * reads as clipped rather than as continuing.  The left fade only starts
   * acting once bar 0 has scrolled into it, by which point it is leaving. */
  state.originBar = 0;
  state.drawnThroughBar = -1;

  for (const voice of meta.voices) {
    const lane = document.createElement('div');
    lane.className = 'lane';
    lane.style.setProperty('--voice-color', VOICE_COLORS[voice.name]);

    const gridEl = document.createElement('div');
    gridEl.className = 'lane-grid';
    gridEl.style.height = `${PITCH_ROWS * ROW_H + REST_STRIP}px`;
    gridEl.style.width = `${WINDOW_BARS * SLOTS_PER_BAR * CELL_W}px`;

    for (let row = 0; row <= PITCH_ROWS; row += 1) {
      const line = document.createElement('div');
      line.className = 'rowline' + (row % 12 === 0 ? ' octave' : '');
      line.style.top = `${row * ROW_H}px`;
      gridEl.appendChild(line);
    }

    // Each lane has its own base_midi, so the tonic rows differ per lane.
    const tonic = meta.tonal_center;
    if (tonic) {
      for (let p = 0; p < PITCH_ROWS; p += 1) {
        if ((voice.base_midi + p) % 12 !== tonic.pitch_class) continue;
        const band = document.createElement('div');
        band.className = 'tonic';
        band.style.top = `${optionRow('n' + p) * ROW_H}px`;
        band.style.height = `${ROW_H}px`;
        gridEl.appendChild(band);
      }
    }

    lane.appendChild(gridEl);
    el.roll.appendChild(lane);
    state.lanes.set(voice.name, { voice, gridEl, sounding: null, bars: new Map() });
  }

  el.playhead.style.left = `${PLAYHEAD_ANCHOR * 100}%`;
  drawBarsThrough(LEAD_BARS);
  scrollTo(0);
}

function barLeft(bar) {
  return (bar - state.originBar) * SLOTS_PER_BAR * CELL_W;
}

/* Empty bar nodes: the grid is the canvas, so it exists ahead of the music. */
function drawBarsThrough(bar) {
  for (let b = state.drawnThroughBar + 1; b <= bar; b += 1) {
    for (const lane of state.lanes.values()) {
      const node = document.createElement('div');
      node.className = 'bar';
      node.style.left = `${barLeft(b)}px`;
      node.style.width = `${SLOTS_PER_BAR * CELL_W}px`;
      for (let step = 0; step < SLOTS_PER_BAR; step += 1) {
        const line = document.createElement('div');
        const kind = step === 0 ? ' bar' : (step % 4 === 0 ? ' beat' : '');
        line.className = `gridline${kind}`;
        line.style.left = `${step * CELL_W}px`;
        node.appendChild(line);
      }
      lane.gridEl.appendChild(node);
      lane.bars.set(b, node);
    }
  }
  state.drawnThroughBar = Math.max(state.drawnThroughBar, bar);
  dropBarsBefore(state.drawnThroughBar + 1 - WINDOW_BARS);
}

function cullBars(playBar) {
  dropBarsBefore(playBar - KEEP_BEHIND_BARS);
}

/* Drop bars behind `cutoff` and rebase the origin onto it, so neither the node
 * count nor the translate offset grows without bound.  Rebasing moves every
 * remaining bar, so the transform -- which is expressed relative to the origin
 * -- must be recomputed before the frame is painted, or the whole roll jumps
 * one bar sideways for one frame. */
function dropBarsBefore(cutoff) {
  if (cutoff <= state.originBar) return;
  for (const lane of state.lanes.values()) {
    for (const [bar, node] of lane.bars) {
      if (bar < cutoff) { node.remove(); lane.bars.delete(bar); }
    }
  }
  state.originBar = cutoff;
  for (const lane of state.lanes.values()) {
    for (const [bar, node] of lane.bars) node.style.left = `${barLeft(bar)}px`;
  }
  if (state.playing) scrollTo(currentStep());

  const cutoffStep = cutoff * SLOTS_PER_BAR;
  for (const [id, note] of state.sounding) {
    if (note.end <= cutoffStep) state.sounding.delete(id);
  }
}

function currentStep() {
  return transport().seconds / state.stepSeconds;
}

function scrollTo(step) {
  const anchor = el.scroll.clientWidth * PLAYHEAD_ANCHOR;
  const x = (step - state.originBar * SLOTS_PER_BAR) * CELL_W;
  el.roll.style.transform = `translateX(${anchor - x}px)`;
}

/* `ghosts` collects the runner-up markers; the caller clears a whole bar of
 * them at once. */
function paintDecision(decision, ghosts) {
  const lane = state.lanes.get(decision.voice);
  if (!lane) return;
  const barNode = lane.bars.get(decision.bar);
  if (!barNode) return;
  const { voice } = lane;
  // abs_step, not step: `step` is bar-relative, so bars would stack on bar 1.
  // The bar node supplies the bar's own offset, so subtract it back out here.
  const left = (decision.abs_step - decision.bar * SLOTS_PER_BAR) * CELL_W;
  const spanW = voice.steps_per_slot * CELL_W - 2;
  const low = decision.confidence < LOW_CONFIDENCE;

  if (decision.chosen === 'rest') {
    const mark = document.createElement('div');
    mark.className = 'rest';
    mark.style.left = `${left + 2}px`;
    mark.style.width = `${spanW - 2}px`;
    mark.style.top = `${PITCH_ROWS * ROW_H + 4}px`;
    barNode.appendChild(mark);
    lane.sounding = null;
    return;
  }

  if (decision.chosen === 'hold') {
    if (!lane.sounding) return;          // hold with nothing sounding: silence
    const tail = document.createElement('div');
    tail.className = 'cell tail';
    tail.style.left = `${left}px`;
    tail.style.width = `${spanW}px`;
    tail.style.top = `${lane.sounding.row * ROW_H + 1}px`;
    tail.style.height = `${ROW_H - 2}px`;
    tail.style.opacity = String(0.25 + 0.3 * decision.confidence);
    barNode.appendChild(tail);
    return;
  }

  // Flicker across the top candidates before settling.
  for (const cand of decision.top.slice(0, 3)) {
    if (cand.option === decision.chosen) continue;
    if (cand.option === 'rest' || cand.option === 'hold') continue;
    const ghost = document.createElement('div');
    ghost.className = 'cell ghost';
    ghost.style.left = `${left}px`;
    ghost.style.width = `${spanW}px`;
    ghost.style.top = `${optionRow(cand.option) * ROW_H}px`;
    ghost.style.height = `${ROW_H}px`;
    barNode.appendChild(ghost);
    ghosts.push(ghost);
  }

  const row = optionRow(decision.chosen);
  const cell = document.createElement('div');
  cell.className = 'cell appear' + (low ? ' low' : '');
  cell.style.left = `${left}px`;
  cell.style.width = `${spanW}px`;
  cell.style.top = `${row * ROW_H}px`;
  cell.style.height = `${ROW_H}px`;
  cell.style.opacity = String(0.4 + 0.6 * decision.confidence);
  barNode.appendChild(cell);
  lane.sounding = { row, cell };
}

/* ------------------------------------------------------------------ trace */

function traceBarBreak() {
  if (!el.trace.childElementCount) return;
  const li = document.createElement('li');
  li.className = 'gap';
  el.trace.appendChild(li);
}

function traceDecision(decision) {
  const low = decision.confidence < LOW_CONFIDENCE;
  const li = document.createElement('li');
  li.className = 'entry' + (low ? ' low' : '');
  li.style.setProperty('--voice-color', VOICE_COLORS[decision.voice]);

  const top = document.createElement('div');
  top.className = 'entry-top';
  top.innerHTML =
    '<span class="dot"></span>' +
    `<span class="addr">${decision.bar + 1}.${String(decision.step).padStart(2, '0')}</span>` +
    `<span class="chosen">${decision.label}</span>` +
    `<span class="conf">${decision.confidence.toFixed(2)}</span>`;
  li.appendChild(top);

  const bars = document.createElement('div');
  bars.className = 'bars';
  for (const cand of decision.top.slice(0, 3)) {
    const row = document.createElement('div');
    row.className = 'prob' + (cand.option === decision.chosen ? ' is-chosen' : '');
    row.innerHTML =
      `<span>${cand.label}</span>` +
      `<span class="meter"><i style="width:${(cand.p * 100).toFixed(1)}%"></i></span>` +
      `<span>${cand.p.toFixed(3)}</span>`;
    bars.appendChild(row);
  }
  li.appendChild(bars);

  el.trace.appendChild(li);
  return li;
}

function capTrace() {
  while (el.trace.childElementCount > TRACE_MAX_ENTRIES) {
    el.trace.firstElementChild.remove();
  }
}

/* ----------------------------------------------------------------- reveal */

/* One bar, all at once: nothing inside a bar happened in an order. */
function revealBar(bar) {
  drawBarsThrough(bar.index + LEAD_BARS);

  const ghosts = [];
  traceBarBreak();
  let last = null;
  for (const decision of bar.decisions) {
    paintDecision(decision, ghosts);
    last = traceDecision(decision);
  }
  capTrace();
  if (last) last.scrollIntoView({ block: 'nearest' });
  state.lastRevealAt = performance.now();
  state.revealedBars += 1;

  scheduleNotes(bar.notes);
  setTimeout(() => ghosts.forEach((g) => g.remove()), GHOST_HOLD_MS);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function drainBars(run) {
  if (state.revealing) return;
  state.revealing = true;
  try {
    while (state.barQueue.length && state.run === run) {
      // Measured from the previous reveal, so a slow stream is not punished
      // with a second pause on top of its own wait.
      const waited = performance.now() - state.lastRevealAt;
      if (waited < BAR_PAUSE_MS) await sleep(BAR_PAUSE_MS - waited);
      if (state.run !== run) return;
      revealBar(state.barQueue.shift());
      if (!state.playing && state.revealedBars >= playAfterBars()) startPlayback();
    }
  } finally {
    state.revealing = false;
  }
}

function playAfterBars() {
  const lookahead = state.meta && state.meta.lookahead_bars;
  return Math.max(1, lookahead || 2);
}

/* ----------------------------------------------------------------- stream */

function generate() {
  stopPlayback();
  if (state.source) state.source.close();
  const run = state.run + 1;
  state.run = run;
  state.barQueue = [];
  state.pending = null;
  state.lastRevealAt = 0;
  state.revealedBars = 0;
  state.finalBar = null;
  el.trace.innerHTML = '';
  setBusy(true);

  /* `?bars=N` on the page bounds the stream, for a finite run. */
  const bars = new URLSearchParams(location.search).get('bars');
  const source = new EventSource(BASE + 'api/stream' + (bars ? `?bars=${bars}` : ''));
  state.source = source;

  source.addEventListener('meta', (ev) => {
    state.meta = JSON.parse(ev.data);
    buildRoll(state.meta);
  });

  source.addEventListener('bar_start', (ev) => {
    state.pending = { index: JSON.parse(ev.data).bar, decisions: [], notes: [] };
  });

  /* Buffer only: per-slot arrival order is a transport artifact and must not
   * reach the screen. */
  source.addEventListener('decision', (ev) => {
    if (state.pending) state.pending.decisions.push(JSON.parse(ev.data));
  });

  source.addEventListener('bar_done', (ev) => {
    if (!state.pending) return;
    state.pending.notes = JSON.parse(ev.data).notes;
    state.barQueue.push(state.pending);
    state.pending = null;
    drainBars(run);
  });

  /* Only a bounded stream ends. */
  source.addEventListener('done', (ev) => {
    state.finalBar = JSON.parse(ev.data).piece.bars.length - 1;
    source.close();
    state.source = null;
  });

  source.onerror = () => {
    source.close();
    state.source = null;
    if (!state.playing) setBusy(false);
  };
}

/* ------------------------------------------------------------------ audio */

function buildSynths() {
  if (state.synths) return state.synths;
  const reverb = new Tone.Reverb({ decay: 2.2, wet: 0.18 }).toDestination();
  state.synths = {
    melody: new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: 'triangle' },
      envelope: { attack: 0.01, decay: 0.2, sustain: 0.5, release: 0.4 },
      volume: -8,
    }).connect(reverb),
    harmony: new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: 'sine' },
      envelope: { attack: 0.06, decay: 0.3, sustain: 0.7, release: 0.8 },
      volume: -15,
    }).connect(reverb),
  };
  return state.synths;
}

/* A note held past its own bar line has no final duration when its bar is
 * scheduled, so attack and release go on the Transport as separate events.  A
 * later bar that extends the note re-sends it with the same id and a larger
 * `steps`; the pending release is cleared and rescheduled.  The stream stays
 * LOOKAHEAD_BARS ahead of the playhead, so a release being moved is always
 * seconds away from firing. */
function scheduleNotes(notes) {
  const t = transport();
  const synths = buildSynths();
  for (const incoming of notes) {
    const known = state.sounding.get(incoming.id);
    if (known) {
      known.end = incoming.start + incoming.steps;
      scheduleRelease(t, known);
      continue;
    }
    const note = {
      voice: incoming.voice,
      freq: Tone.Frequency(incoming.midi, 'midi').toFrequency(),
      velocity: 0.45 + 0.5 * incoming.confidence,
      end: incoming.start + incoming.steps,
      releaseId: null,
    };
    t.scheduleOnce((time) => {
      synths[note.voice].triggerAttack(note.freq, time, note.velocity);
    }, bbs(incoming.start));
    scheduleRelease(t, note);
    state.sounding.set(incoming.id, note);
  }
}

function scheduleRelease(t, note) {
  if (note.releaseId !== null) t.clear(note.releaseId);
  const at = Math.max(0, new Tone.Time(bbs(note.end)).toSeconds() - RELEASE_GAP_S);
  note.releaseId = t.scheduleOnce((time) => {
    state.synths[note.voice].triggerRelease(note.freq, time);
  }, at);
}

function startPlayback() {
  const t = transport();
  t.bpm.value = state.meta.tempo_bpm;
  t.position = 0;
  state.stepSeconds = new Tone.Time('16n').toSeconds();
  t.start();
  state.playing = true;
  el.playhead.classList.add('on');

  let lastBar = -1;

  // One loop drives both the playhead and the scroll, so they cannot drift.
  // Culling comes first: it rebases the origin the transform is measured from.
  const follow = () => {
    const step = currentStep();
    if (state.finalBar !== null && step >= (state.finalBar + 1) * SLOTS_PER_BAR) {
      stopPlayback();
      return;
    }
    const bar = Math.floor(step / SLOTS_PER_BAR);
    if (bar !== lastBar) { lastBar = bar; cullBars(bar); }
    scrollTo(step);
    state.rafHandle = requestAnimationFrame(follow);
  };
  state.rafHandle = requestAnimationFrame(follow);
}

function stopPlayback() {
  const t = transport();
  t.stop();
  t.cancel(0);
  state.sounding.clear();
  state.playing = false;
  if (state.synths) Object.values(state.synths).forEach((s) => s.releaseAll && s.releaseAll());
  if (state.rafHandle) { cancelAnimationFrame(state.rafHandle); state.rafHandle = null; }
  el.playhead.classList.remove('on');
  setBusy(false);
}

el.transport.addEventListener('click', () => {
  if (el.transport.classList.contains('busy')) {
    state.run += 1;                    // stops any reveal loop in flight
    state.barQueue = [];
    if (state.source) { state.source.close(); state.source = null; }
    stopPlayback();
    return;
  }
  Tone.start();                        // must happen inside the user gesture
  generate();
});

fetch(BASE + 'api/meta')
  .then((r) => r.json())
  .then((meta) => {
    state.meta = meta;
    buildRoll(meta);
  })
  .catch(() => { el.transport.disabled = true; });
