# Calibrate each stage on all 25 ordinary (non-drought) years. Also fits sets B and C and a
# drought-years fit. Held-out scores come from crossval.py.
# Usage: python calibrate.py [gauge_id]
import json
import numpy as np
from scipy.optimize import differential_evolution
from camels import load, ordinary_years, drought_years, gauge_arg
from model import STAGES, FULL_LO, FULL_HI, nse, log_nse, to_unit

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

# Set A is the best fit ('4: + soil'). Sets B and C score within NEAR of A and are as
# different from it as possible: B is the set farthest from A, C the set farthest from
# both A and B (maximizing the smaller of its two distances). Distance is Euclidean after
# rescaling each parameter to 0-1 over its range, with the two stores' half-lives on a
# log scale, so every parameter counts the same.
NEAR = 0.01
lo, hi = FULL_LO, FULL_HI


def farthest_from(refs, floor):
    forcing = [d[c].values for c in full_inputs]
    refs = [to_unit(r) for r in refs]

    def loss(x):
        shortfall = max(0.0, floor - nse(full_fn(*forcing, *x)[ORD], obs[ORD]))
        return -min(np.linalg.norm(to_unit(x) - r) for r in refs) + 1000 * shortfall

    fit = differential_evolution(loss, list(zip(lo, hi)), seed=0, tol=1e-8, maxiter=400, polish=False)
    sim = full_fn(*forcing, *fit.x)
    return fit.x, dict(params={p[0]: float(v) for p, v in zip(full_params, fit.x)}, nse=nse(sim[ORD], obs[ORD]))


x_a = np.array(list(results['4: + soil']['params'].values()))
floor = results['4: + soil']['nse'] - NEAR
x_b, results['set B'] = farthest_from([x_a], floor)
_, results['set C'] = farthest_from([x_a, x_b], floor)

for name, r in results.items():
    print(name, {k: round(v, 3) for k, v in r['params'].items()},
          f'NSE on ordinary years {r["nse"]:.3f}')

with open(f'data/{GAUGE}_calibration.json', 'w') as f:
    json.dump(results, f, indent=1)
