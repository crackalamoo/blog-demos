# Calibrate each stage on all 25 ordinary (non-drought) years. Also fits sets B and C and a
# drought-years fit. Held-out scores come from crossval.py.
# Usage: python calibrate.py [gauge_id]
import json
import numpy as np
from scipy.optimize import differential_evolution
from camels import load, ordinary_years, drought_years, gauge_arg
from model import STAGES, nse, log_nse

GAUGE = gauge_arg()
d, _ = load(GAUGE)
obs = d.q.values
ORD = ordinary_years(d)


def calibrate(fn, inputs, params, objective=nse, mask=ORD, bounds=None):
    forcing = [d[c].values for c in inputs]
    bounds = bounds or [(lo, hi) for _, lo, hi in params]

    def loss(x):
        return -objective(fn(*forcing, *x)[mask], obs[mask])

    fit = differential_evolution(loss, bounds, seed=0, tol=1e-6, maxiter=300, polish=True)
    sim = fn(*forcing, *fit.x)
    return dict(params={p[0]: float(v) for p, v in zip(params, fit.x)},
                nse=nse(sim[ORD], obs[ORD]))


results = {}
for stage, fn, inputs, params in STAGES:  # every stage is fit on NSE
    results[stage] = calibrate(fn, inputs, params)
_, full_fn, full_inputs, full_params = STAGES[-1]
FULL = (full_fn, full_inputs, full_params)
# The full model fit on NSE of log flow, which weights a dry-summer trickle as much as a melt peak
results['4: + soil, log-flow objective'] = calibrate(*FULL, objective=log_nse)
# The full model fit on the drought years instead
results['drought'] = calibrate(*FULL, mask=drought_years(d))

# Set A is the best fit ('4: + soil'). Sets B and C are the best fits under one restriction
# each: B's slow store has at least twice A's half-life, C's soil at most half A's capacity.
# The slow store keeps 1 - K2 of its water a day, so doubling the half-life means keeping
# the square root of that.
A = results['4: + soil']['params']
bounds = {n: (lo, hi) for n, lo, hi in full_params}
restrict = {'set B': {'K2': (bounds['K2'][0], 1 - np.sqrt(1 - A['K2']))},
            'set C': {'FC': (bounds['FC'][0], A['FC'] / 2)}}
for name, over in restrict.items():
    results[name] = calibrate(*FULL, bounds=list({**bounds, **over}.values()))

for name, r in results.items():
    print(name, {k: round(v, 3) for k, v in r['params'].items()},
          f'NSE on ordinary years {r["nse"]:.3f}')

with open(f'data/{GAUGE}_calibration.json', 'w') as f:
    json.dump(results, f, indent=1)
