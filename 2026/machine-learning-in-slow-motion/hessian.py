# The shape of the loss around the best fit (set A): its Hessian, and how far that local,
# quadratic picture holds. Run after calibrate.py.
#
# The loss is L = 1 - NSE. Near A, L(A + d) ~ L(A) + g.d + 1/2 d.H.d, with d in the
# rescaled coordinates of model.to_unit. The leak (PERC) and evaporation threshold (LP)
# sit at the top of their ranges, where the loss still slopes, so they're held fixed and
# the Hessian is over the other six. The model's thresholds (min, if) make the loss
# slightly jagged at fine scale, so derivatives are finite differences over steps of a
# few percent of each range, checked at several step sizes.
import json
import numpy as np
from camels import load, ordinary_years, BLACKWOOD
from model import STAGES, score_many, full_model, to_unit, from_unit

d, _ = load(BLACKWOOD)
ORD = ordinary_years(d)
P, T, PET, obs = (d[c].values for c in ['prcp', 'tmean', 'pet', 'q'])
names = [p[0] for p in STAGES[-1][3]]
calib = json.load(open(f'data/{BLACKWOOD}_calibration.json'))
sets = {k: np.array([calib[r]['params'][n] for n in names])
        for k, r in [('A', '4: + soil'), ('B', 'set B'), ('C', 'set C')]}
uA = to_unit(sets['A'])
FREE = [j for j, n in enumerate(names) if n not in ('PERC', 'LP')]
AT_LIMIT = [j for j, n in enumerate(names) if n in ('PERC', 'LP')]


def loss(U):
    """1 - NSE on the ordinary years, for each row of rescaled parameters U."""
    U = np.atleast_2d(U)
    return 1 - score_many(from_unit(U), P, T, PET, np.nan_to_num(obs), ORD)[:, 0]


def hessian(h):
    """Central finite differences over the free parameters, all points scored in one batch."""
    n = len(FREE)
    pts, keys = [uA], [()]
    for a in range(n):
        for sa in (1, -1):
            u = uA.copy(); u[FREE[a]] += sa * h; pts.append(u); keys.append((a, sa))
            for b in range(a + 1, n):
                for sb in (1, -1):
                    u = uA.copy(); u[FREE[a]] += sa * h; u[FREE[b]] += sb * h
                    pts.append(u); keys.append((a, sa, b, sb))
    L = dict(zip(keys, loss(np.array(pts))))
    H = np.empty((n, n))
    for a in range(n):
        H[a, a] = (L[a, 1] - 2 * L[()] + L[a, -1]) / h**2
        for b in range(a + 1, n):
            H[a, b] = H[b, a] = (L[a, 1, b, 1] - L[a, 1, b, -1] - L[a, -1, b, 1] + L[a, -1, b, -1]) / (4 * h**2)
    return H


def gauss_newton(h):
    """2/D J^T J, where J is how each day's simulated flow moves with each free parameter."""
    o = obs[ORD]
    J = np.empty((ORD.sum(), len(FREE)))
    for a, j in enumerate(FREE):
        up, dn = uA.copy(), uA.copy()
        up[j] += h; dn[j] -= h
        J[:, a] = (full_model(P, T, PET, *from_unit(up))[ORD] - full_model(P, T, PET, *from_unit(dn))[ORD]) / (2 * h)
    return 2 / np.sum((o - o.mean()) ** 2) * J.T @ J


def slope_at_limit(h):
    """Rise in loss per unit step inward for each parameter held at its limit (one-sided)."""
    out = {}
    for j in AT_LIMIT:
        u = uA.copy(); u[j] -= h
        out[names[j]] = float((loss(u)[0] - loss(uA)[0]) / h)
    return out


if __name__ == '__main__':
    L0 = loss(uA)[0]
    print(f'A: NSE {1 - L0:.4f}; rescaled position ' + ' '.join(f'{n}={v:.2f}' for n, v in zip(names, uA)))
    fn = [names[j] for j in FREE]

    print('\nEigenvalues (stiffest first) at several step sizes h, full Hessian vs Gauss-Newton:')
    for h in (0.005, 0.01, 0.02, 0.05):
        ev = np.linalg.eigvalsh(hessian(h))[::-1]
        evg = np.linalg.eigvalsh(gauss_newton(h))[::-1]
        print(f'  h={h:<5} full ' + ' '.join(f'{v:9.3g}' for v in ev) + '   GN ' + ' '.join(f'{v:9.3g}' for v in evg))

    H = hessian(0.02)
    lam, V = np.linalg.eigh(H)
    lam, V = lam[::-1], V[:, ::-1]
    print('\nEigenvectors at h=0.02 (columns: stiffest to sloppiest), and the NSE drop 0.01 allows along each:')
    print('        ' + ' '.join(f'{k:>8}' for k in range(1, len(lam) + 1)))
    for a, n in enumerate(fn):
        print(f'  {n:6s}' + ' '.join(f'{V[a, k]:8.2f}' for k in range(len(lam))))
    print('  lambda' + ' '.join(f'{v:8.3g}' for v in lam))
    print('  reach ' + ' '.join(f'{np.sqrt(2 * 0.01 / v) if v > 0 else np.inf:8.3f}' for v in lam))
    print('\nSlope of the loss inward from the limit, per unit step:', slope_at_limit(0.02))

    print('\nB and C seen from A (rescaled coordinates):')
    for k in ('B', 'C'):
        dv = to_unit(sets[k]) - uA
        df = dv[FREE]
        coef = V.T @ df  # displacement along each eigenvector
        pred = 0.5 * df @ H @ df
        print(f'  {k}: actual NSE drop {loss(to_unit(sets[k]))[0] - L0:.4f}; quadratic over the free six predicts '
              f'{pred:.4f}; move at the limit ' + ' '.join(f'{names[j]} {dv[j]:+.2f}' for j in AT_LIMIT))
        print('     share of the free move along each eigenvector: '
              + ' '.join(f'{c**2 / (df @ df):.2f}' for c in coef))
