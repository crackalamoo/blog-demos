# Calibrate each stage on ordinary (non-drought) years.
# Usage: python calibrate.py [gauge_id]
import json
from scipy.optimize import differential_evolution
from camels import load, calibration_years, gauge_arg
from model import STAGES, nse, log_nse

GAUGE = gauge_arg()
d, _ = load(GAUGE)
obs = d.q.values
CAL = calibration_years(d)

# Every stage is fit on NSE. The full model is also fit on NSE of log flow, which
# weights a dry-summer trickle as much as a melt peak.
RUNS = [(*stage, nse) for stage in STAGES]
RUNS.append(('4: + soil, log-flow objective', *STAGES[-1][1:], log_nse))

results = {}
for stage, fn, inputs, params, objective in RUNS:
    forcing = [d[c].values for c in inputs]
    bounds = [(lo, hi) for _, lo, hi in params]

    def loss(x):
        return -objective(fn(*forcing, *x)[CAL], obs[CAL])

    fit = differential_evolution(loss, bounds, seed=0, tol=1e-6, maxiter=300, polish=True)
    sim = fn(*forcing, *fit.x)
    results[stage] = dict(params={p[0]: float(v) for p, v in zip(params, fit.x)},
                          nse_cal=nse(sim[CAL], obs[CAL]))
    print(stage, {k: round(v, 3) for k, v in results[stage]['params'].items()},
          f'NSE {results[stage]["nse_cal"]:.3f}')

with open(f'data/{GAUGE}_calibration.json', 'w') as f:
    json.dump(results, f, indent=1)
