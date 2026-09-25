# Learned models (LSTM, transformer) against the calibrated physics model on Blackwood Creek.
# Same folds as crossval.py, same target (daily flow), same loss (MSE on standardized flow,
# i.e. NSE within one basin). Each model sees a 365-day window of daily inputs and predicts
# flow on the window's last day (Kratzert et al. 2019). No observed flow is ever an input.
#
# Every fit caches its full prediction series under data/learned/, so reruns only report.
# Usage: python learned.py [--workers N] [--device cpu|mps] [--seeds 3] [--report]
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from camels import load, ordinary_years, drought_years, BLACKWOOD
from model import nse, log_nse, full_model

GAUGE = BLACKWOOD
OUT = f'data/learned_{GAUGE}'
LOOKBACK = 365
N_VAL_YEARS = 2  # the last 2 training years of each fold, for early stopping only

# Fixed up front, never tuned on test years.
HP = dict(batch=256, lr=1e-3, max_epochs=60, patience=10, clip=1.0,
          lstm=dict(hidden=64, dropout=0.4),
          transformer=dict(d_model=64, heads=4, layers=2, ff=128, dropout=0.2))
INPUTS = {
    'physics': ['prcp', 'tmean', 'pet'],  # exactly what the physics model gets
    'standard': ['prcp', 'tmax', 'tmin', 'pet', 'doy_sin', 'doy_cos'],
}
MODELS = ['lstm', 'transformer']

d, _ = load(GAUGE)
d['doy_sin'] = np.sin(2 * np.pi * d.index.dayofyear / 365.25)
d['doy_cos'] = np.cos(2 * np.pi * d.index.dayofyear / 365.25)
obs = d.q.values
ORD = ordinary_years(d)
DRY = drought_years(d)
YEARS = sorted(d.wy[ORD].unique().tolist())
assert len(YEARS) == 25
SCHEMES = {  # identical to crossval.py
    'blocked': [YEARS[5 * k:5 * k + 5] for k in range(5)],
    'interleaved': [[y for i, y in enumerate(YEARS) if i % 5 == k] for k in range(5)],
}
# The forcing starts 1980-01-01 but WY1981 starts 1980-10-01, so its first 90 days lack a full
# 365-day window. Windows reaching before 1980 are padded with the training mean of each input
# (0 once standardized), with real day-of-year values; the physics model likewise starts
# 1980-01-01 from empty stores. REAL_WINDOW marks days whose window is all real data.
PAD = LOOKBACK - 1
REAL_WINDOW = np.arange(len(d)) >= PAD
PAD_DATES = pd.date_range(end=d.index[0] - pd.Timedelta(days=1), periods=PAD)


def jobs(seeds):
    out = []
    for m in MODELS:
        for inp in INPUTS:
            for s in range(seeds):
                for sc, folds in SCHEMES.items():
                    for k in range(5):
                        out.append((m, inp, sc, k, s))
                out.append((m, inp, 'all', 0, s))  # all 25 ordinary years, scored on droughts
    return out


def split(scheme, k):
    """Training, validation and test years for one fold."""
    test = [] if scheme == 'all' else SCHEMES[scheme][k]
    train = [y for y in YEARS if y not in test]
    return train[:-N_VAL_YEARS], train[-N_VAL_YEARS:], test


def path(m, inp, sc, k, s):
    return f'{OUT}/{m}_{inp}_{sc}{k}_s{s}.npz'


def build(m, n_in):
    import torch
    from torch import nn

    class LSTM(nn.Module):
        def __init__(self, hidden, dropout):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            h, _ = self.lstm(x)
            return self.head(self.drop(h[:, -1])).squeeze(-1)

    import torch.nn.functional as F

    class EncoderLayer(nn.Module):
        """Post-norm encoder layer, the same computation as nn.TransformerEncoderLayer (ReLU,
        dropout on attention weights and both residual branches). With last_only it returns
        just the last position, attending from that one query to every day: identical to
        the full layer's last row, since everything after attention is position-wise."""

        def __init__(self, d_model, heads, ff, dropout):
            super().__init__()
            self.heads, self.p = heads, dropout
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.out = nn.Linear(d_model, d_model)
            self.ff = nn.Sequential(nn.Linear(d_model, ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff, d_model))
            self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
            self.drop = nn.Dropout(dropout)

        def forward(self, x, last_only=False):
            B, T, D = x.shape
            q, k, v = self.qkv(x).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
            if last_only:
                q, x = q[:, :, -1:], x[:, -1:]
            a = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p if self.training else 0.0)
            x = self.norm1(x + self.drop(self.out(a.transpose(1, 2).reshape(B, x.shape[1], D))))
            return self.norm2(x + self.drop(self.ff(x)))

    class Transformer(nn.Module):
        def __init__(self, d_model, heads, layers, ff, dropout):
            super().__init__()
            self.inp = nn.Linear(n_in, d_model)
            pos = np.arange(LOOKBACK)[:, None]  # sinusoidal positional encoding
            div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
            pe = np.zeros((LOOKBACK, d_model), np.float32)
            pe[:, 0::2], pe[:, 1::2] = np.sin(pos * div), np.cos(pos * div)
            self.register_buffer('pe', torch.from_numpy(pe))
            self.layers = nn.ModuleList([EncoderLayer(d_model, heads, ff, dropout) for _ in range(layers)])
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x):
            h = self.drop(self.inp(x) + self.pe)
            for i, layer in enumerate(self.layers):
                h = layer(h, last_only=i == len(self.layers) - 1)
            return self.head(h[:, -1]).squeeze(-1)  # read out at the day being predicted

    return LSTM(**HP['lstm']) if m == 'lstm' else Transformer(**HP['transformer'])


def fit(job, device='cpu', threads=1, log=False):
    m, inp, sc, k, s = job
    p = path(*job)
    if os.path.exists(p):
        return p
    train_y, val_y, _ = split(sc, k)
    r = train(m, INPUTS[inp], train_y, val_y, s, device, threads, log)
    os.makedirs(OUT, exist_ok=True)
    np.savez(p, **r, device=device)
    return p


def train(m, cols, train_y, val_y, s, device='cpu', threads=1, log=False):
    """Fit one network on the ordinary days of train_y, early-stopping on those of val_y.
    Returns the prediction for every day of the record plus training diagnostics."""
    import torch
    torch.set_num_threads(threads)
    torch.manual_seed(s)
    np.random.seed(s)
    rng = np.random.default_rng(s)
    t0 = time.time()

    days = lambda ys: np.flatnonzero(ORD & d.wy.isin(ys).values)
    tr, va = days(train_y), days(val_y)
    # normalization from the training days only (window inputs are then standardized with it)
    X = d[cols].values.astype(np.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0)
    for j, c in enumerate(cols):
        if c.startswith('doy'):
            mu[j], sd[j] = 0.0, 1.0
    X = (X - mu) / sd
    pad = np.zeros((PAD, len(cols)), np.float32)
    for j, c in enumerate(cols):
        if c.startswith('doy'):
            f = np.sin if c == 'doy_sin' else np.cos
            pad[:, j] = f(2 * np.pi * PAD_DATES.dayofyear / 365.25)
    X = np.concatenate([pad, X])
    q_mu, q_sd = obs[tr].mean(), obs[tr].std()
    y = ((np.nan_to_num(obs) - q_mu) / q_sd).astype(np.float32)

    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    win = Xt.unfold(0, LOOKBACK, 1).transpose(1, 2)  # win[i] = days i-364 .. i, predicts day i

    def batch(idx):
        i = torch.as_tensor(idx, device=device)
        return win[i], yt[torch.as_tensor(idx, device=device)]

    net = build(m, len(cols)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=HP['lr'])

    # Inference chunk. The transformer's first layer materializes a (chunk, heads, 365, 365)
    # attention matrix, ~2.2 GB per tensor at 1024 windows, so it predicts in smaller chunks.
    # Chunking doesn't change the predictions (eval mode: no dropout, no batch statistics).
    chunk = 1024 if m == 'lstm' else 128

    def predict(idx):
        net.eval()
        out = []
        with torch.no_grad():
            for b in range(0, len(idx), chunk):
                out.append(net(batch(idx[b:b + chunk])[0]).cpu().numpy())
        return np.concatenate(out) * q_sd + q_mu

    best, best_state, since, hist, ep_sec = -np.inf, None, 0, [], []
    for epoch in range(HP['max_epochs']):
        te = time.time()
        net.train()
        perm = rng.permutation(tr)
        for b in range(0, len(perm), HP['batch']):
            xb, yb = batch(perm[b:b + HP['batch']])
            loss = torch.mean((net(xb) - yb) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), HP['clip'])
            opt.step()
        v = nse(predict(va), obs[va])
        if device == 'mps':
            # eval mode takes the transformer's fused inference path, whose cached blocks the
            # training step can't reuse: releasing them keeps ~1 GB of headroom under a capped
            # allocator (without it an N=10 transformer fit ran out of memory). No effect on results.
            torch.mps.empty_cache()
        hist.append(float(v))
        ep_sec.append(time.time() - te)
        if log:
            print(f'epoch {epoch}: val NSE {v:.3f}, {ep_sec[-1]:.1f}s', flush=True)
        if v > best:
            best, since = v, 0
            best_state = {n: t.detach().clone() for n, t in net.state_dict().items()}
        else:
            since += 1
            if since >= HP['patience']:
                break
    net.load_state_dict(best_state)
    sim = predict(np.arange(len(d)))
    return dict(sim=sim, val_hist=np.array(hist), best_epoch=int(np.argmax(hist)),
                seconds=time.time() - t0, epoch_seconds=np.array(ep_sec))


def _worker(args):
    job, device, threads = args
    t = time.time()
    fit(job, device, threads)
    return job, time.time() - t


# ------------------------------------------------------------------ scoring
def pos(x):
    """Flow can't be negative: every metric below scores predictions clipped at 0."""
    return np.maximum(x, 0)


LATE = ORD & d.index.month.isin([8, 9])


def late_ratio(sim):
    return float(np.median(pos(sim[LATE]) / obs[LATE]))


def pooled(get, scheme):
    """Each ordinary year predicted by the fold fit that didn't see it."""
    sim = np.full(len(d), np.nan)
    for k, f in enumerate(SCHEMES[scheme]):
        m = ORD & d.wy.isin(f).values
        sim[m] = get(scheme, k)[m]
    return sim


def scores(get):
    b, i, a = pooled(get, 'blocked'), pooled(get, 'interleaved'), get('all', 0)
    return dict(blocked=nse(pos(b[ORD]), obs[ORD]), interleaved=nse(pos(i[ORD]), obs[ORD]),
                drought=nse(pos(a[DRY]), obs[DRY]), lognse_blocked=log_nse(pos(b[ORD]), obs[ORD]),
                late_blocked=late_ratio(b),
                lognse_interleaved=log_nse(pos(i[ORD]), obs[ORD]),
                drought_lognse=log_nse(pos(a[DRY]), obs[DRY]),
                train_all=nse(pos(a[ORD]), obs[ORD]), late_all=late_ratio(a),
                neg_frac_blocked=float(np.mean(b[ORD] < 0)),
                blocked_realwin=nse(pos(b[ORD & REAL_WINDOW]), obs[ORD & REAL_WINDOW]),
                blocked_unclipped=nse(b[ORD], obs[ORD]))


def physics_scores():
    cv = json.load(open(f'data/{GAUGE}_crossval.json'))
    R = {(r['scheme'], r['fold']): np.array(r['sim_test']) for r in cv if r['stage'] == '4: + soil'}

    def get(sc, k):
        if sc == 'all':
            p = json.load(open(f'data/{GAUGE}_calibration.json'))['4: + soil']['params']
            return full_model(d.prcp.values, d.tmean.values, d.pet.values, *p.values())
        sim = np.full(len(d), np.nan)
        sim[ORD & d.wy.isin(SCHEMES[sc][k]).values] = R[sc, k]
        return sim
    return scores(get)


COLS = [('blocked', 'blocked NSE'), ('interleaved', 'interl. NSE'), ('drought', 'drought NSE'),
        ('lognse_blocked', 'logNSE blk'), ('late_blocked', 'Aug-Sep ratio')]
EXTRA = [('lognse_interleaved', 'logNSE int'), ('drought_lognse', 'drought logNSE'),
         ('train_all', 'NSE ord (fit on all)'), ('late_all', 'Aug-Sep (fit on all)'),
         ('neg_frac_blocked', 'frac<0 blk'), ('blocked_realwin', 'blk NSE, real windows'), ('blocked_unclipped', 'blk NSE unclipped')]


def report(seeds):
    rows = {'physics (HBV, 8 params)': dict(ens=physics_scores(), seeds=None)}
    meta = {}
    for m in MODELS:
        for inp in INPUTS:
            loaded = {}
            for s in range(seeds):
                for sc in list(SCHEMES) + ['all']:
                    for k in range(1 if sc == 'all' else 5):
                        z = np.load(path(m, inp, sc, k, s))
                        loaded[sc, k, s] = z['sim']
                        meta.setdefault((m, inp), []).append((float(z['seconds']), int(z['best_epoch']),
                                                              len(z['val_hist'])))
            per_seed = [scores(lambda sc, k, s=s: loaded[sc, k, s]) for s in range(seeds)]
            ens = scores(lambda sc, k: np.mean([loaded[sc, k, s] for s in range(seeds)], 0))
            rows[f'{m} / {inp} inputs'] = dict(ens=ens, seeds=per_seed)

    for cols, title in [(COLS, 'MAIN TABLE'), (EXTRA, 'EXTRA')]:
        print(f'\n===== {title}: ensemble of {seeds} seeds [seed mean ± sd; min..max] =====')
        print(f'{"":34s}' + ''.join(f'{h:>28s}' for _, h in cols))
        for name, r in rows.items():
            line = f'{name:34s}'
            for key, _ in cols:
                e = r['ens'][key]
                if r['seeds'] is None:
                    line += f'{e:28.3f}'
                else:
                    v = np.array([x[key] for x in r['seeds']])
                    line += f'{f"{e:.3f} [{v.mean():.3f}±{v.std(ddof=1):.3f}; {v.min():.2f}..{v.max():.2f}]":>28s}'
            print(line)

    print('\nphysics late-summer ratio, fit on all 25 years (in-sample), for reference: '
          f'{rows["physics (HBV, 8 params)"]["ens"]["late_all"]:.3f}')
    print('\nruntimes and early stopping:')
    for (m, inp), v in meta.items():
        v = np.array(v)
        print(f'  {m:12s} {inp:9s} {len(v)} fits, {v[:, 0].sum() / 60:6.1f} min total, '
              f'{v[:, 0].mean():5.1f} s/fit; best epoch median {np.median(v[:, 1]):.0f} '
              f'(range {v[:, 1].min()}..{v[:, 1].max()}), epochs run median {np.median(v[:, 2]):.0f}')
    out = {k: dict(ensemble=r['ens'], per_seed=r['seeds']) for k, r in rows.items()}
    json.dump(out, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--device', default='mps')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--report', action='store_true', help='only report cached fits')
    ap.add_argument('--only', help='run only jobs whose model matches, e.g. lstm')
    a = ap.parse_args()
    if not a.report:
        todo = [j for j in jobs(a.seeds) if not os.path.exists(path(*j)) and (not a.only or j[0] == a.only)]
        # seed by seed, so stopping early leaves complete seeds; slow transformer first within a seed
        todo.sort(key=lambda j: (j[4], j[0] != 'transformer'))
        t0 = time.time()
        print(f'{len(todo)} fits to run on {a.device}, {a.workers} workers', flush=True)
        with ProcessPoolExecutor(a.workers) as ex:
            for n, (job, sec) in enumerate(ex.map(_worker, [(j, a.device, a.threads) for j in todo]), 1):
                print(f'[{n}/{len(todo)} {(time.time() - t0) / 60:5.1f} min] {job} {sec:.0f}s', flush=True)
    report(a.seeds)
