# Learning curve: the calibrated physics model (stage 4, 8 parameters) against an LSTM as
# the training record shrinks from 20 to 10 to 5 water years on Blackwood Creek.
#
# Folds are crossval.py's 'blocked' scheme: 5 contiguous 5-year blocks of the 25 ordinary
# water years. For each fold the pool is the other 20 years; N = 5 and 10 are the first N
# years of one fixed random permutation of that pool (so the subsets are nested, 5 in 10 in
# 20), and N = 20 is the whole pool. Both models get exactly the same N years:
#   physics  calibrated on all N with calibrate.py's differential evolution settings;
#   LSTM     trained on the first N - V of them (chronologically), early-stopped on the last
#            V = max(1, round(0.2 N)), with the 'physics' inputs (prcp, tmean, pet) and
#            learned.py's architecture and hyperparameters, 3 seeds.
# So the data budget is the same, but the LSTM spends part of it on validation.
# Weather covers the whole record for both: the physics model spins up from 1980 and the
# LSTM's 365-day window reaches back before the training years. Observed flow is never an input.
#
# The physics model is also fit with the leak's range loosened (model.WIDE_LEAK), same years.
#
# Optionally also the transformer (learned.py's architecture, same inputs and split as the LSTM),
# trained with batch 128 instead of HP's 256 to stay well under the memory limit on MPS.
#
# Runs everything in one process, sequentially. Every fit is cached under
# data/learning_curve_<gauge>/, so a rerun only reports.
# Usage: python learning_curve.py [--device mps|cpu] [--seeds 3] [--report]
#        python learning_curve.py --models transformer --sizes 20 --seeds 1 --time-limit 35
import argparse
import json
import os
import signal
import subprocess
import threading
import time
import numpy as np
from scipy.optimize import differential_evolution
from model import STAGES, WIDE_LEAK, nse, log_nse
import learned
from learned import d, obs, ORD, DRY, YEARS, SCHEMES, INPUTS, GAUGE, pos, late_ratio

OUT = f'data/learning_curve_{GAUGE}'
FOLDS = SCHEMES['blocked']
SIZES = [20, 10, 5]
INPUT_SET = 'physics'
MEM_LIMIT_GB = 6.0
HARD_LIMIT_MIN = 60
TRANSFORMER_BATCH = 128  # HP['batch'] = 256 sits at the memory cap for the transformer on MPS
TF_SEC_GUESS = {20: 170, 10: 100, 5: 55}  # transformer s/fit before any fit at that N has run
REPORT_MARGIN_MIN = 1.0  # time left for the report when the transformer run stops early
BUDGET_MIN = 45  # if the projected total runs past this, drop the third LSTM seed
T0 = time.time()


def elapsed():
    return (time.time() - T0) / 60


def subset(k, n):
    """The n training years for fold k: the first n of a fixed permutation of its 20-year pool."""
    pool = [y for y in YEARS if y not in FOLDS[k]]
    assert len(pool) == 20
    perm = np.random.default_rng([2026, k]).permutation(pool)
    return sorted(int(y) for y in perm[:n])


def lstm_split(years):
    n_val = max(1, round(0.2 * len(years)))
    return years[:-n_val], years[-n_val:]


# ------------------------------------------------------------------ safety
def mem_gb():
    """Resident memory of this process plus what the MPS driver holds (counted separately,
    to be conservative about unified memory)."""
    rss = int(subprocess.run(['ps', '-o', 'rss=', '-p', str(os.getpid())],
                             capture_output=True, text=True).stdout.strip() or 0) / 1024**2
    mps = 0.0
    try:
        import torch
        if torch.backends.mps.is_available():
            mps = torch.mps.driver_allocated_memory() / 1024**3
    except Exception:
        pass
    return rss, mps


def check_memory(tag=''):
    rss, mps = mem_gb()
    if rss + mps > MEM_LIMIT_GB:
        raise SystemExit(f'ABORT: memory {rss:.2f} GB RSS + {mps:.2f} GB MPS exceeds {MEM_LIMIT_GB} GB {tag}')
    return rss, mps


def on_alarm(*_):
    raise SystemExit(f'ABORT: hard time limit reached after {elapsed():.1f} min')


def start_watchdog(period=0.5):
    """Check memory every `period` s from a background thread and hard-exit over the limit
    (check_memory between fits can't catch a spike inside one fit)."""
    def loop():
        while True:
            rss, mps = mem_gb()
            if rss + mps > MEM_LIMIT_GB:
                print(f'ABORT watchdog: {rss:.2f} GB RSS + {mps:.2f} GB MPS > {MEM_LIMIT_GB} GB', flush=True)
                os._exit(3)
            time.sleep(period)
    threading.Thread(target=loop, daemon=True).start()


# ------------------------------------------------------------------ fits
PHYSICS = {'physics': STAGES[-1][3], 'physics_wide': WIDE_LEAK}  # name -> parameter ranges


def phys_path(k, n, name='physics'):
    return f'{OUT}/{name}_f{k}_n{n}.npz'


def lstm_path(k, n, s):
    return f'{OUT}/lstm_{INPUT_SET}_f{k}_n{n}_s{s}.npz'


def tf_path(k, n, s):
    return f'{OUT}/transformer_{INPUT_SET}_f{k}_n{n}_s{s}.npz'


def fit_physics(k, n, name='physics'):
    p = phys_path(k, n, name)
    if os.path.exists(p):
        return None
    t = time.time()
    _, fn, inputs, _ = STAGES[-1]
    params = PHYSICS[name]
    years = subset(k, n)
    train = ORD & d.wy.isin(years).values
    forcing = [d[c].values for c in inputs]

    def loss(x):
        return -nse(fn(*forcing, *x)[train], obs[train])

    r = differential_evolution(loss, [(lo, hi) for _, lo, hi in params],
                               seed=0, tol=1e-6, maxiter=300, polish=True)
    sim = fn(*forcing, *r.x)
    np.savez(p, sim=sim, x=r.x, years=years, nse_train=nse(sim[train], obs[train]),
             seconds=time.time() - t)
    return time.time() - t


def fit_lstm(k, n, s, device):
    p = lstm_path(k, n, s)
    if os.path.exists(p):
        return None
    years = subset(k, n)
    tr, va = lstm_split(years)
    r = learned.train('lstm', INPUTS[INPUT_SET], tr, va, s, device=device, threads=2)
    np.savez(p, **r, train_years=tr, val_years=va, device=device)
    return r['seconds']


def fit_transformer(k, n, s, device):
    p = tf_path(k, n, s)
    if os.path.exists(p):
        return None
    years = subset(k, n)
    tr, va = lstm_split(years)
    hp_batch = learned.HP['batch']
    learned.HP['batch'] = TRANSFORMER_BATCH  # override for this fit only
    try:
        r = learned.train('transformer', INPUTS[INPUT_SET], tr, va, s, device=device, threads=2)
    finally:
        learned.HP['batch'] = hp_batch
        # hand cached MPS blocks back between fits: with the allocator capped (watermark 0.3,
        # 4 GB here) the cache left by earlier fits in the same process ran fold 4 out of memory
        import gc
        import torch
        gc.collect()
        if device == 'mps':
            torch.mps.empty_cache()
    np.savez(p, **r, train_years=tr, val_years=va, device=device, batch=TRANSFORMER_BATCH)
    return r['seconds']


def run_transformer(runs, device, limit_min=None):
    """runs: ordered (n, seed) pairs; each runs all 5 folds before the next pair starts.
    With limit_min, a fit isn't started unless the last fit's time at that N (or TF_SEC_GUESS)
    still fits before the limit with REPORT_MARGIN_MIN left over, so the run stops cleanly
    between fits (cached fits resume later) instead of being killed by the alarm mid-fit."""
    todo = [(k, n, s) for n, s in runs for k in range(5) if not os.path.exists(tf_path(k, n, s))]
    print(f'[{elapsed():5.1f} min] {len(todo)} transformer fits to run (batch {TRANSFORMER_BATCH}), '
          f'order {runs}', flush=True)
    last = {}
    for i, (k, n, s) in enumerate(todo, 1):
        est = last.get(n, TF_SEC_GUESS.get(n, 200)) / 60
        if limit_min is not None and elapsed() + est > limit_min - REPORT_MARGIN_MIN:
            print(f'[{elapsed():5.1f} min] stopping before transformer fold {k} N={n} seed {s}: '
                  f'~{est:.1f} min fit would pass the {limit_min:.0f} min limit '
                  f'({len(todo) - i + 1} fits left, cached fits resume later)', flush=True)
            break
        sec = fit_transformer(k, n, s, device)
        last[n] = sec
        rss, mps = check_memory(f'after transformer fold {k} N={n} seed {s}')
        print(f'[{elapsed():5.1f} min] transformer seed {s} {i}/{len(todo)} fold {k} N={n}: {sec:.0f}s  '
              f'(mem {rss:.2f} GB RSS, {mps:.2f} GB MPS)', flush=True)


def run(seeds, device, sizes=SIZES):
    os.makedirs(OUT, exist_ok=True)
    jobs = [(k, n) for n in sizes for k in range(5)]
    todo = [(k, n, name) for name in PHYSICS for k, n in jobs if not os.path.exists(phys_path(k, n, name))]
    print(f'[{elapsed():5.1f} min] {len(todo)} physics fits to run', flush=True)
    for i, (k, n, name) in enumerate(todo, 1):
        sec = fit_physics(k, n, name)
        rss, mps = check_memory()
        print(f'[{elapsed():5.1f} min] {name} {i}/{len(todo)} fold {k} N={n}: {sec:.0f}s  '
              f'(mem {rss:.2f} GB)', flush=True)

    run_seeds, seed_minutes = seeds, []
    for s in range(seeds):
        todo = [j for j in jobs if not os.path.exists(lstm_path(*j, s))]
        if s >= 2 and todo and seed_minutes:
            projected = elapsed() + np.mean(seed_minutes) * (seeds - s)
            if projected > BUDGET_MIN:
                run_seeds = s
                print(f'[{elapsed():5.1f} min] projected {projected:.0f} min > {BUDGET_MIN}: '
                      f'stopping at {s} LSTM seeds', flush=True)
                break
        t_seed = time.time()
        for i, (k, n) in enumerate(todo, 1):
            sec = fit_lstm(k, n, s, device)
            rss, mps = check_memory(f'after lstm fold {k} N={n} seed {s}')
            print(f'[{elapsed():5.1f} min] lstm seed {s} {i}/{len(todo)} fold {k} N={n}: {sec:.0f}s  '
                  f'(mem {rss:.2f} GB RSS, {mps:.2f} GB MPS)', flush=True)
        if todo:
            seed_minutes.append((time.time() - t_seed) / 60)
    return run_seeds


# ------------------------------------------------------------------ scoring
def fold_mask(k):
    return ORD & d.wy.isin(FOLDS[k]).values


def scores(get):
    """get(k) is fold k's prediction for every day. Pooled scores use each ordinary year's
    held-out prediction; drought scores use every fold's fit (no fit trains on droughts)."""
    sim = np.full(len(d), np.nan)
    for k in range(5):
        sim[fold_mask(k)] = get(k)[fold_mask(k)]
    per_fold_dry = [nse(pos(get(k)[DRY]), obs[DRY]) for k in range(5)]
    ens_dry = np.mean([get(k) for k in range(5)], 0)
    return dict(nse=nse(pos(sim[ORD]), obs[ORD]), lognse=log_nse(pos(sim[ORD]), obs[ORD]),
                late=late_ratio(sim), drought_mean=float(np.mean(per_fold_dry)),
                drought_ens=nse(pos(ens_dry[DRY]), obs[DRY]), drought_folds=per_fold_dry,
                heldout_folds=[nse(pos(get(k)[fold_mask(k)]), obs[fold_mask(k)]) for k in range(5)])


def report(seeds):
    phys = {(k, n): np.load(phys_path(k, n)) for n in SIZES for k in range(5)}
    wide = {(k, n): np.load(phys_path(k, n, 'physics_wide')) for n in SIZES for k in range(5)}
    lstm = {}
    for s in range(seeds):
        if all(os.path.exists(lstm_path(k, n, s)) for n in SIZES for k in range(5)):
            for n in SIZES:
                for k in range(5):
                    lstm[k, n, s] = np.load(lstm_path(k, n, s))
    S = sorted({s for _, _, s in lstm})
    print(f'\nLSTM seeds complete: {len(S)}')

    res = {}
    for n in SIZES:
        P = scores(lambda k: phys[k, n]['sim'])
        E = scores(lambda k: np.mean([lstm[k, n, s]['sim'] for s in S], 0))
        per = [scores(lambda k, s=s: lstm[k, n, s]['sim']) for s in S]
        res[n] = dict(physics=P, physics_wide=scores(lambda k: wide[k, n]['sim']), lstm_ens=E, lstm_seeds=per)

    def rng(n, key):
        v = np.array([x[key] for x in res[n]['lstm_seeds']])
        return f'{v.mean():.3f} [{v.min():.3f}..{v.max():.3f}]'

    rows = [('nse', 'pooled held-out NSE'), ('drought_mean', 'drought NSE, mean of 5 folds'),
            ('drought_ens', 'drought NSE, 5-fold ensemble'), ('lognse', 'pooled held-out log-NSE'),
            ('late', 'Aug-Sep model/obs median')]
    print(f'\n{"":32s}{"N":>4s} {"physics":>9s} {"LSTM ens":>9s}  {"LSTM per-seed mean [min..max]":>30s}  {"phys-LSTMens":>12s}')
    for key, label in rows:
        for n in SIZES[::-1]:
            p, e = res[n]['physics'][key], res[n]['lstm_ens'][key]
            print(f'{label:32s}{n:4d} {p:9.3f} {e:9.3f}  {rng(n, key):>30s}  {p - e:+12.3f}')
        print()

    print(f'physics with the leak loosened (0-{WIDE_LEAK[4][2]} mm/day), vs the usual range:')
    for key, label in rows:
        print(f'  {label:32s}' + '  '.join(f'N={n} {res[n]["physics_wide"][key]:.3f} ({res[n]["physics"][key]:.3f})'
                                           for n in SIZES[::-1]))
    print('  fitted leak (mm/day) by N: ' + '; '.join(
        f'{n}: {[round(float(wide[k, n]["x"][4]), 1) for k in range(5)]}' for n in SIZES[::-1]))

    print('\nper-fold held-out NSE (fold: physics / LSTM ens), and training years')
    for n in SIZES[::-1]:
        for k in range(5):
            print(f'  N={n:2d} fold {k}: {res[n]["physics"]["heldout_folds"][k]:6.3f} / '
                  f'{res[n]["lstm_ens"]["heldout_folds"][k]:6.3f}   drought {res[n]["physics"]["drought_folds"][k]:6.3f} / '
                  f'{res[n]["lstm_ens"]["drought_folds"][k]:6.3f}   years {subset(k, n)}')

    # sanity check against crossval.py
    cv = json.load(open(f'data/{GAUGE}_crossval.json'))
    R = {r['fold']: np.array(r['sim_test']) for r in cv if r['stage'] == '4: + soil' and r['scheme'] == 'blocked'}
    ref = np.full(len(d), np.nan)
    for k in range(5):
        ref[fold_mask(k)] = R[k]
    ref_nse = nse(ref[ORD], obs[ORD])
    here = res[20]['physics']['nse']
    diff = max(np.max(np.abs(phys[k, 20]['sim'][fold_mask(k)] - R[k])) for k in range(5))
    print(f'\nsanity: physics N=20 pooled NSE {here:.4f} vs crossval.py blocked stage 4 {ref_nse:.4f} '
          f'-> {"REPRODUCED" if abs(here - ref_nse) < 5e-4 else "MISMATCH"} (max |sim diff| {diff:.2e} mm/day)')

    print('\nruntimes and fitting diagnostics:')
    for n in SIZES[::-1]:
        ps = [float(phys[k, n]['seconds']) for k in range(5)]
        ls = [float(lstm[k, n, s]['seconds']) for k in range(5) for s in S]
        be = [int(lstm[k, n, s]['best_epoch']) for k in range(5) for s in S]
        ep = [len(lstm[k, n, s]['val_hist']) for k in range(5) for s in S]
        vbest = [float(np.max(lstm[k, n, s]['val_hist'])) for k in range(5) for s in S]
        ptr = [float(phys[k, n]['nse_train']) for k in range(5)]
        print(f'  N={n:2d}: physics {np.mean(ps):5.1f} s/fit, train NSE {np.mean(ptr):.3f}; '
              f'LSTM {np.mean(ls):5.1f} s/fit, best epoch median {np.median(be):.0f} '
              f'(range {min(be)}..{max(be)}), epochs run median {np.median(ep):.0f}, '
              f'best val NSE median {np.median(vbest):.3f} (range {min(vbest):.2f}..{max(vbest):.2f})')
    tot_p = sum(float(z['seconds']) for z in phys.values())
    tot_l = sum(float(z['seconds']) for z in lstm.values())
    print(f'  total fit time: physics {tot_p / 60:.1f} min, LSTM {tot_l / 60:.1f} min')

    # transformer, at whichever sizes have all 5 folds for some seeds
    tf_seeds = {}
    for n in SIZES:
        T = [s for s in range(10) if all(os.path.exists(tf_path(k, n, s)) for k in range(5))]
        if not T:
            continue
        tf = {(k, s): np.load(tf_path(k, n, s)) for k in range(5) for s in T}
        tf_seeds[n] = T
        res[n]['transformer_seeds'] = [scores(lambda k, s=s: tf[k, s]['sim']) for s in T]
        res[n]['transformer_ens'] = scores(lambda k: np.mean([tf[k, s]['sim'] for s in T], 0))
        print(f'\nN={n}: transformer ({len(T)} seed{"s" * (len(T) > 1)}: {T}, batch '
              f'{sorted({int(z["batch"]) for z in tf.values()})}) against physics and LSTM')
        print(f'{"":36s}' + ''.join(f'{h:>33s}' for _, h in rows))
        table = [('physics', res[n]['physics'], None),
                 ('physics, leak loosened', res[n]['physics_wide'], None),
                 (f'LSTM ens ({len(S)} seeds) [seed range]', res[n]['lstm_ens'], res[n]['lstm_seeds']),
                 ('LSTM seed 0', res[n]['lstm_seeds'][0], None),
                 (f'transformer ens ({len(T)} seed{"s" * (len(T) > 1)})', res[n]['transformer_ens'],
                  res[n]['transformer_seeds'] if len(T) > 1 else None)]
        for name, r, per in table:
            line = f'{name:36s}'
            for key, _ in rows:
                cell = f'{r[key]:.3f}'
                if per:
                    v = [x[key] for x in per]
                    cell += f' [{min(v):.3f}..{max(v):.3f}]'
                line += f'{cell:>33s}'
            print(line)
        ts = [float(z['seconds']) for z in tf.values()]
        be = [int(z['best_epoch']) for z in tf.values()]
        ep = [len(z['val_hist']) for z in tf.values()]
        print(f'  per-fold held-out NSE: {np.round(res[n]["transformer_ens"]["heldout_folds"], 3).tolist()}; '
              f'drought NSE: {np.round(res[n]["transformer_ens"]["drought_folds"], 3).tolist()}')
        print(f'  transformer: {len(ts)} fits, {sum(ts) / 60:.1f} min total, {np.mean(ts):.0f} s/fit; '
              f'best epoch {be}, epochs run {ep}')

    json.dump(dict(seeds=S, transformer_seeds={str(n): T for n, T in tf_seeds.items()},
                   transformer_batch=TRANSFORMER_BATCH, results={str(n): r for n, r in res.items()},
                   sanity=dict(here=here, crossval=ref_nse)),
              open(f'{OUT}/summary.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='mps')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--report', action='store_true', help='only report cached fits')
    ap.add_argument('--models', default='physics,lstm', help='comma-separated: physics,lstm,transformer')
    ap.add_argument('--sizes', default=','.join(map(str, SIZES)), help='training sizes to fit, e.g. 20')
    ap.add_argument('--tf-runs', default=None,
                    help='transformer fits in this order, as N:seed pairs, e.g. 5:0,20:1,20:2 '
                         '(default: every seed < --seeds at every --sizes)')
    ap.add_argument('--time-limit', type=float, default=HARD_LIMIT_MIN, help='hard limit, minutes')
    a = ap.parse_args()
    models = a.models.split(',')
    sizes = [int(n) for n in a.sizes.split(',')]
    assert set(models) <= {'physics', 'lstm', 'transformer'} and set(sizes) <= set(SIZES)
    signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(int(a.time_limit * 60))
    start_watchdog()
    # --seeds sets how many LSTM seeds to fit and report; when the LSTM isn't being fit, the
    # report uses every complete LSTM seed in the cache (so a transformer-only run doesn't shrink it)
    seeds = a.seeds if 'lstm' in models else 10
    if not a.report:
        if 'physics' in models or 'lstm' in models:
            run_seeds = run(a.seeds if 'lstm' in models else 0, a.device, sizes)
            if 'lstm' in models:
                seeds = run_seeds
        if 'transformer' in models:
            runs = ([tuple(map(int, r.split(':'))) for r in a.tf_runs.split(',')] if a.tf_runs
                    else [(n, s) for s in range(a.seeds) for n in sizes])
            assert {n for n, _ in runs} <= set(SIZES)
            run_transformer(runs, a.device, a.time_limit)
    report(seeds)
