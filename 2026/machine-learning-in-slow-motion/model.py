# The model, built up one piece at a time. Each stage takes some of: daily
# precipitation P, mean temperature T, and potential evaporation PET (mm/day),
# and returns simulated river flow Q (mm/day).
import numpy as np
from numba import njit, prange

LOG_EPS = 0.01  # mm/day, keeps log(flow) finite on (near-)zero days


@njit
def bucket(P, K):
    """Stage 1: one bucket. Everything that falls goes in; a fraction K drains out each day."""
    S = 0.0
    Q = np.empty(len(P))
    for t in range(len(P)):
        S += P[t]
        Q[t] = K * S
        S -= Q[t]
    return Q


@njit
def snow_step(snowpack, P, T, TT, CFMAX):
    """Below TT, precipitation is stored as snow; above it, snow melts CFMAX mm per degree."""
    if T < TT:
        return snowpack + P, 0.0
    melt = min(CFMAX * (T - TT), snowpack)
    return snowpack - melt, P + melt


@njit
def snow_bucket(P, T, K, TT, CFMAX):
    """Stage 2: a snowpack in front of the bucket."""
    snowpack, S = 0.0, 0.0
    Q = np.empty(len(P))
    for t in range(len(P)):
        snowpack, water = snow_step(snowpack, P[t], T[t], TT, CFMAX)
        S += water
        Q[t] = K * S
        S -= Q[t]
    return Q


@njit
def snow_two_stores(P, T, TT, CFMAX, K1, K2, PERC):
    """Stage 3: groundwater split into a fast upper store (drains at K1) and a slow
    lower store (drains at K2), with up to PERC mm/day leaking from upper to lower."""
    snowpack, upper, lower = 0.0, 0.0, 0.0
    Q = np.empty(len(P))
    for t in range(len(P)):
        snowpack, water = snow_step(snowpack, P[t], T[t], TT, CFMAX)
        upper, lower, Q[t] = groundwater_step(upper, lower, water, K1, K2, PERC)
    return Q


@njit
def groundwater_step(upper, lower, water, K1, K2, PERC):
    upper += water
    perc = min(PERC, upper)
    upper -= perc
    lower += perc
    q_fast, q_slow = K1 * upper, K2 * lower
    return upper - q_fast, lower - q_slow, q_fast + q_slow


@njit
def soil_step(SM, water, PET, FC, BETA, LP):
    """The soil holds up to FC mm. The wetter it is, the more incoming water passes
    through to groundwater (shape set by BETA). Plants evaporate at the full PET rate
    once the soil is more than LP full, and proportionally less below that."""
    recharge = water * (SM / FC) ** BETA
    SM += water - recharge
    if SM > FC:
        recharge += SM - FC
        SM = FC
    ET = min(PET * min(SM / (LP * FC), 1.0), SM)
    return SM - ET, recharge


@njit
def full_model(P, T, PET, TT, CFMAX, K1, K2, PERC, FC, BETA, LP):
    """Stage 4: soil and evaporation between the snow and the groundwater."""
    snowpack, SM, upper, lower = 0.0, 0.0, 0.0, 0.0
    Q = np.empty(len(P))
    for t in range(len(P)):
        snowpack, water = snow_step(snowpack, P[t], T[t], TT, CFMAX)
        SM, recharge = soil_step(SM, water, PET[t], FC, BETA, LP)
        upper, lower, Q[t] = groundwater_step(upper, lower, recharge, K1, K2, PERC)
    return Q


@njit(parallel=True)
def score_many(X, P, T, PET, obs, mask):
    """Calibration scores of the full model for every parameter set (row) in X.
    Column 0 is NSE on flow (dominated by the big peaks), column 1 is NSE on
    log flow (every order of magnitude counts, so low flows matter too)."""
    o = obs[mask]
    lo = np.log(o + LOG_EPS)
    denom = np.sum((o - o.mean()) ** 2)
    ldenom = np.sum((lo - lo.mean()) ** 2)
    out = np.empty((len(X), 2))
    for i in prange(len(X)):
        x = X[i]
        Q = full_model(P, T, PET, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7])[mask]
        out[i, 0] = 1 - np.sum((Q - o) ** 2) / denom
        out[i, 1] = 1 - np.sum((np.log(Q + LOG_EPS) - lo) ** 2) / ldenom
    return out


# Parameter ranges are the HBV-light ranges in Seibert & Vis (2012), Table A3, except
# stage 1-2's K, which HBV-light doesn't have.
SNOW = [('TT', -1.5, 2.5), ('CFMAX', 1, 10)]
GROUNDWATER = [('K1', 0.01, 0.4), ('K2', 0.001, 0.15), ('PERC', 0, 3)]
SOIL = [('FC', 50, 500), ('BETA', 1, 6), ('LP', 0.3, 1)]
# (name, model, forcing columns it takes from camels.load, parameters)
STAGES = [
    ('1: bucket', bucket, ['prcp'], [('K', 0.001, 0.5)]),
    ('2: + snow', snow_bucket, ['prcp', 'tmean'], [('K', 0.001, 0.5)] + SNOW),
    ('3: + two stores', snow_two_stores, ['prcp', 'tmean'], SNOW + GROUNDWATER),
    ('4: + soil', full_model, ['prcp', 'tmean', 'pet'], SNOW + GROUNDWATER + SOIL),
]
# The full model's ranges with the leak loosened far past HBV-light's 3 mm/day. The fit
# pushes against that bound; with this one it settles inside it.
WIDE_LEAK = [(n, lo, 20 if n == 'PERC' else hi) for n, lo, hi in STAGES[-1][3]]


# Rescaled coordinates for the full model's parameters: each one mapped to 0-1 over its
# range, with the two stores' rates as half-lives on a log scale (0 = fastest, 1 = slowest).
# Distances and curvatures in these coordinates weigh every parameter the same.
FULL_LO = np.array([p[1] for p in STAGES[-1][3]])
FULL_HI = np.array([p[2] for p in STAGES[-1][3]])
STORES = [j for j, p in enumerate(STAGES[-1][3]) if p[0] in ('K1', 'K2')]


def log_half_life(k):
    """Log of the days a store draining at k takes to halve; it keeps 1 - k of its water a day."""
    return np.log(np.log(2) / -np.log(1 - k))


def to_unit(x):
    u = (np.asarray(x, float) - FULL_LO) / (FULL_HI - FULL_LO)
    for j in STORES:
        a, b = log_half_life(FULL_HI[j]), log_half_life(FULL_LO[j])
        u[..., j] = (log_half_life(np.asarray(x, float)[..., j]) - a) / (b - a)
    return u


def from_unit(u):
    u = np.asarray(u, float)
    x = FULL_LO + u * (FULL_HI - FULL_LO)
    for j in STORES:
        a, b = log_half_life(FULL_HI[j]), log_half_life(FULL_LO[j])
        x[..., j] = 1 - 0.5 ** (1 / np.exp(a + u[..., j] * (b - a)))
    return x


def nse(sim, obs):
    return 1 - np.sum((sim - obs) ** 2) / np.sum((obs - obs.mean()) ** 2)


def log_nse(sim, obs):
    return nse(np.log(sim + LOG_EPS), np.log(obs + LOG_EPS))
