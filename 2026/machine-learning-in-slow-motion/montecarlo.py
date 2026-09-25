# Monte Carlo: sample random parameter sets within physical ranges and score
# each on the ordinary years, on both flow and log flow.
# Usage: python montecarlo.py [gauge_id]
import numpy as np
from camels import load, ordinary_years, gauge_arg
from model import STAGES, score_many

N = 1_000_000
GAUGE = gauge_arg()
_, _, _, params = STAGES[-1]
names = [p[0] for p in params]
lo = np.array([p[1] for p in params])
hi = np.array([p[2] for p in params])

d, _ = load(GAUGE)
P, T, PET, obs = (d[c].values for c in ['prcp', 'tmean', 'pet', 'q'])
ORD = ordinary_years(d)

rng = np.random.default_rng(0)
X = lo + (hi - lo) * rng.random((N, len(params)))
scores = score_many(X, P, T, PET, np.nan_to_num(obs), ORD)
score, score_log = scores[:, 0], scores[:, 1]
np.savez(f'data/{GAUGE}_montecarlo.npz', X=X, score=score, score_log=score_log, names=names)

for label, s in [('NSE', score), ('log-flow NSE', score_log)]:
    print(f'best of {N:,} random sets on {label}: {s.max():.3f}')
    for th in [0.60, 0.65, 0.68, 0.70]:
        print(f'  > {th}: {(s > th).sum()} sets')
