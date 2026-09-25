# Cross-validate every stage over the 25 ordinary water years: blocked (5 contiguous
# 5-year folds) and interleaved (fold k = every 5th year). Each fold is fit on the other
# 20 years with the same DE settings as calibrate.py and scored on its own 5. Run after
# calibrate.py; post_figures.py reads the results.
# Usage: python crossval.py [gauge_id] [--refit]
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import differential_evolution
from camels import load, ordinary_years, BLACKWOOD
from model import STAGES, nse, log_nse

GAUGE = next((a for a in sys.argv[1:] if not a.startswith('-')), BLACKWOOD)
CACHE = f'data/{GAUGE}_crossval.json'
d, _ = load(GAUGE)
obs = d.q.values
ORD = ordinary_years(d)
YEARS = sorted(d.wy[ORD].unique().tolist())
assert len(YEARS) == 25, YEARS

SCHEMES = {
    'blocked': [YEARS[5 * k:5 * k + 5] for k in range(5)],
    'interleaved': [[y for i, y in enumerate(YEARS) if i % 5 == k] for k in range(5)],
}


def fit(job):
    scheme, k, s = job
    name, fn, inputs, params = STAGES[s]
    test_wys = SCHEMES[scheme][k]
    test = ORD & d.wy.isin(test_wys).values
    train = ORD & ~test
    forcing = [d[c].values for c in inputs]

    def loss(x):
        return -nse(fn(*forcing, *x)[train], obs[train])

    r = differential_evolution(loss, [(lo, hi) for _, lo, hi in params],
                               seed=0, tol=1e-6, maxiter=300, polish=True)
    sim = fn(*forcing, *r.x)
    return dict(scheme=scheme, fold=k, stage=name, test_wys=test_wys,
                params={p[0]: float(v) for p, v in zip(params, r.x)},
                nse_train=float(nse(sim[train], obs[train])),
                nse_test=float(nse(sim[test], obs[test])),
                lognse_train=float(log_nse(sim[train], obs[train])),
                lognse_test=float(log_nse(sim[test], obs[test])),
                sim_test=sim[test].tolist())


def run():
    if os.path.exists(CACHE) and '--refit' not in sys.argv:
        return json.load(open(CACHE))
    jobs = [(sc, k, s) for sc in SCHEMES for k in range(5) for s in range(len(STAGES))]
    # slowest (stage 4) first so the pool stays busy
    jobs.sort(key=lambda j: -j[2])
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        out = list(ex.map(fit, jobs))
    json.dump(out, open(CACHE, 'w'))
    return out


def report(results, scheme):
    names = [s[0] for s in STAGES]
    R = {(r['fold'], r['stage']): r for r in results if r['scheme'] == scheme}
    folds = SCHEMES[scheme]
    print(f'\n===== {scheme} folds =====')
    wy_p = d[ORD].groupby('wy').prcp.sum()
    wy_q = d[ORD].groupby('wy').q.sum()
    print('fold  years                      P mm/yr  Q mm/yr  Q/P')
    for k, f in enumerate(folds):
        p, q = wy_p[f].mean(), wy_q[f].mean()
        print(f'{k}     {",".join(map(str, f)):26s} {p:7.0f}  {q:7.0f}  {q / p:.2f}')
    print(f'all                              {wy_p.mean():7.0f}  {wy_q.mean():7.0f}')

    for metric in ['nse_train', 'nse_test', 'lognse_test']:
        print(f'\n{metric}:  fold | ' + ' | '.join(n[:2] for n in names))
        for k in range(5):
            print(f'  {k}  ' + '  '.join(f'{R[k, n][metric]:7.3f}' for n in names))
        if metric != 'nse_train':
            v = np.array([[R[k, n][metric] for n in names] for k in range(5)])
            print('  mean ' + '  '.join(f'{x:7.3f}' for x in v.mean(0)))
            print('  med  ' + '  '.join(f'{x:7.3f}' for x in np.median(v, 0)))

    print('\nwins on test NSE (more complex stage better):')
    for a, b in [(1, 2), (1, 3), (2, 3)]:
        diffs = [R[k, names[b]]['nse_test'] - R[k, names[a]]['nse_test'] for k in range(5)]
        print(f'  stage {b + 1} vs {a + 1}: {sum(x > 0 for x in diffs)}/5, '
              f'diffs {" ".join(f"{x:+.3f}" for x in diffs)}')

    # pooled held-out: each ordinary year predicted by the fold fit that didn't see it
    print('\npooled held-out over all 25 years:  NSE   logNSE')
    for n in names:
        sim = np.empty(len(obs))
        for k, f in enumerate(folds):
            m = ORD & d.wy.isin(f).values
            sim[m] = R[k, n]['sim_test']
        print(f'  {n:18s} {nse(sim[ORD], obs[ORD]):.3f}  {log_nse(sim[ORD], obs[ORD]):.3f}')

    _, _, _, params = STAGES[-1]
    print('\nstage 4 params per fold (* = within 1% of a bound)')
    print('  fold ' + ' '.join(f'{p[0]:>7s}' for p in params) + '  K2 half-life d')
    for k in range(5):
        x = R[k, names[-1]]['params']
        cells = []
        for p, lo, hi in params:
            at = min(x[p] - lo, hi - x[p]) < 0.01 * (hi - lo)
            cells.append(f'{x[p]:6.3f}' + ('*' if at else ' '))
        hl = np.log(2) / -np.log(1 - x['K2'])
        print(f'  {k}    ' + ' '.join(cells) + f'  {hl:7.1f}')


if __name__ == '__main__':
    res = run()
    for sc in SCHEMES:
        report(res, sc)
