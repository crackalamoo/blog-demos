# The figures that go in the post. Run after calibrate.py, montecarlo.py and crossval.py.
# Every fit is trained on the 25 ordinary (non-drought) water years; held-out scores come
# from cross-validation. Three parameter sets are followed through the figures: A, the
# best fit, and B and C, which score almost as well but describe different basins.
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import ConnectionPatch
from camels import load, ordinary_years, drought_years, snotel_tmin, winter_mean, BLACKWOOD, GENERAL
from model import STAGES, full_model, groundwater_step, nse

OUT = 'figures'
os.makedirs(OUT, exist_ok=True)
# The water year shown in figures 2-5 (Oct 1999 - Sep 2000). An ordinary year in which
# every stage and all three sets fit well, so the differences shown come from the model,
# not from one odd year. NSE in the figures is over all ordinary years.
WY = 2000
# Figure 1 shows the raw data over three water years, including the one above.
DATA_WYS = (1999, 2001)

INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8984'
GRID, BAND, SURFACE = '#e8e7e3', '#dcdbd6', '#ffffff'
MODEL = '#e34948'
SETS = {'A': MODEL, 'B': '#2a78d6', 'C': '#1baf7a'}  # A is the model of fig 2d
# Muted colors for the inputs, kept apart from the saturated model and A/B/C colors
PRECIP, TEMP, TEMP_BAND = '#5b87b5', '#b7832f', '#f1e2c6'

plt.rcParams.update({
    'font.family': ['Helvetica Neue', 'DejaVu Sans'],
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
    'axes.titlepad': 10, 'axes.labelcolor': INK2, 'axes.edgecolor': MUTED, 'axes.linewidth': 0.8,
    'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.size': 0, 'ytick.major.size': 0,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'lines.linewidth': 1.6, 'lines.solid_capstyle': 'round',
})
W = 7.5  # inches; saved at 200 dpi -> 1500 px, shown at ~750 px


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png', dpi=200, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)


def month_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))


def half_life(k):
    """Days for a groundwater store draining at k to fall to half, with no inflow.
    Measured from the model's own one-day step, so it holds for any time-stepping."""
    kept = np.array([groundwater_step(1.0, 0.0, 0.0, ki, ki, 0.0)[0] for ki in np.atleast_1d(k)])
    hl = np.log(2) / -np.log(kept)
    return hl if np.ndim(k) else hl[0]


# ------------------------------------------------------------------ data and runs
d, _ = load(BLACKWOOD)
g, _ = load(GENERAL)
obs, gobs = d.q.values, g.q.values
ORD = ordinary_years(d)
GORD = ordinary_years(g)  # same years as Blackwood
forcing = lambda df: (df.prcp.values, df.tmean.values, df.pet.values)

mc = np.load(f'data/{BLACKWOOD}_montecarlo.npz')
names = list(mc['names'])
# A random set is a "good fit" if it scores within 0.04 of the best of the million.
THRESHOLD = mc['score'].max() - 0.04
good = mc['score'] > THRESHOLD
Xg, sg = mc['X'][good], mc['score'][good]
col = {n: j for j, n in enumerate(names)}

calib = json.load(open(f'data/{BLACKWOOD}_calibration.json'))
# A, B and C come from calibrate.py
SET_RUNS = {'A': '4: + soil', 'B': 'set B', 'C': 'set C'}
picks = {k: np.array([calib[r]['params'][n] for n in names]) for k, r in SET_RUNS.items()}


def run(df, x):
    return full_model(*forcing(df), *x)


Qb = {k: run(d, x) for k, x in picks.items()}
Qg = {k: run(g, x) for k, x in picks.items()}
band_b = np.array([run(d, x) for x in Xg])
nse_b = {k: nse(q[ORD], obs[ORD]) for k, q in Qb.items()}
nse_g = {k: nse(q[GORD], gobs[GORD]) for k, q in Qg.items()}
for k, x in picks.items():
    print(k, {n: round(float(v), 3) for n, v in zip(names, x)},
          f'NSE Blackwood {nse_b[k]:.3f}, General {nse_g[k]:.3f}')

wy = (d.wy == WY).values
dates = d.index[wy]


def ensemble(ax, x, band, sel, label=True):
    ax.fill_between(x, np.percentile(band[:, sel], 5, 0), np.percentile(band[:, sel], 95, 0),
                    color=BAND, lw=0, label=f'middle 90% of the {len(band):,} good fits' if label else None)


SET_LABELS = {'A': 'set A (best fit)', 'B': 'set B', 'C': 'set C'}


def abc(ax, x, runs, sel):
    for k, q in runs.items():
        ax.plot(x, q[sel], color=SETS[k], lw=1.6, label=SET_LABELS[k])


# ------------------------------------------------------------------ figure 1
# The problem, before any model: the inputs and the target over three water years.
y = d[(d.wy >= DATA_WYS[0]) & (d.wy <= DATA_WYS[1])]
fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(W, 5.4), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1, 0.8, 1.2], hspace=0.25))
a0.bar(y.index, y.prcp, width=1, color=PRECIP)
a0.set_ylabel('precipitation\n(mm/day)')
a1.fill_between(y.index, y.tmin, y.tmax, color=TEMP_BAND, lw=0, label='daily low to high')
a1.plot(y.index, y.tmean, color=TEMP, lw=0.8, label='daily mean')
a1.axhline(0, color=MUTED, lw=0.8)
a1.set_ylabel('temperature\n(°C)')
a1.legend(loc='upper left', fontsize=8.5, ncol=2)
a2.plot(y.index, y.q, color=INK, lw=1.4)
a2.set_ylabel('creek flow\n(mm/day)')
a2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
a2.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
a0.set_title('Blackwood Creek: weather in, creek flow out')
a2.text(0, -0.5, f'Water years {DATA_WYS[0]}–{DATA_WYS[1]} (Oct {DATA_WYS[0] - 1} – Sep {DATA_WYS[1]}).\n'
        'Precipitation and temperature are Daymet basin averages; flow is the USGS gauge.',
        transform=a2.transAxes, color=MUTED, fontsize=9)
save(fig, 'fig1')

# ------------------------------------------------------------------ figure 2a-d
# One figure per stage, on the same axes, so each can sit next to its equation.
LABELS = ['One bucket', 'Add snow', 'Add fast and slow groundwater', 'Add soil and evaporation']
sims = [fn(*[d[c].values for c in inputs], *[calib[stage]['params'][p[0]] for p in params])
        for stage, fn, inputs, params in STAGES]
ymax = 1.05 * max(obs[wy].max(), *(s[wy].max() for s in sims))
for i, ((stage, *_), label, sim) in enumerate(zip(STAGES, LABELS, sims)):
    fig, a = plt.subplots(figsize=(W, 2.9))
    a.plot(dates, obs[wy], color=INK, lw=1.3, label='observed')
    a.plot(dates, sim[wy], color=MODEL, lw=1.3, label='model')
    a.set_ylim(0, ymax)
    a.set_ylabel('flow (mm/day)')
    month_axis(a)
    a.legend(loc='upper left', fontsize=9)
    a.set_title(f'Stage {stage[0]}: {label.lower()}')
    a.text(0.98, 0.94, f'NSE {calib[stage]["nse"]:.2f}', transform=a.transAxes, ha='right', va='top',
           color=INK, fontsize=11)
    a.text(0, -0.3, f'Blackwood Creek, water year {WY}.\nNSE is scored on all ordinary (non-drought) years '
           '1981–2011; 1 is a perfect fit.', transform=a.transAxes, color=MUTED, fontsize=9)
    save(fig, f'fig2{"abcd"[i]}')

# ------------------------------------------------------------------ figure 3
# Rows in the order the post introduces them, with the post's symbols. Each row also gets a
# strip: where each set's value sits within the range sampled (half-lives on a log scale).
def linear(n):
    lo, hi = RANGES[n]
    return lambda x: (x[col[n]] - lo) / (hi - lo)


def log_half_life(n):
    lo, hi = np.log(half_life(RANGES[n][1])), np.log(half_life(RANGES[n][0]))
    return lambda x: (np.log(half_life(x[col[n]])) - lo) / (hi - lo)


RANGES = {n: (lo, hi) for n, lo, hi in STAGES[-1][3]}
C_ET_RANGE = (RANGES['LP'][0] * RANGES['FC'][0], RANGES['LP'][1] * RANGES['FC'][1])
ROWS = [(r'Melt threshold $T_{\mathrm{melt}}$ (°C)', lambda x: f'{x[col["TT"]]:.1f}', linear('TT')),
        (r'Melt factor $f_{\mathrm{melt}}$ (mm per °C per day)', lambda x: f'{x[col["CFMAX"]]:.1f}', linear('CFMAX')),
        (r'Fast store half-life, from $k_{\mathrm{fast}}$ (days)', lambda x: f'{half_life(x[col["K1"]]):.1f}',
         log_half_life('K1')),
        (r'Slow store half-life, from $k_{\mathrm{slow}}$ (days)', lambda x: f'{half_life(x[col["K2"]]):.0f}',
         log_half_life('K2')),
        (r'Leak, fast to slow $r_{\mathrm{leak}}$ (mm/day)', lambda x: f'{x[col["PERC"]]:.1f}', linear('PERC')),
        (r'Soil water capacity $C_{\mathrm{max\ soil}}$ (mm)', lambda x: f'{x[col["FC"]]:.0f}', linear('FC')),
        (r'Runoff curve shape $\beta$', lambda x: f'{x[col["BETA"]]:.1f}', linear('BETA')),
        (r'Full-rate evaporation above $C_{\mathrm{ET}}$ (mm)', lambda x: f'{x[col["LP"]] * x[col["FC"]]:.0f}',
         lambda x: (x[col['LP']] * x[col['FC']] - C_ET_RANGE[0]) / (C_ET_RANGE[1] - C_ET_RANGE[0]))]
fig = plt.figure(figsize=(W, 6.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1], hspace=0.4)
a = fig.add_subplot(gs[0])
ensemble(a, dates, band_b, wy)
abc(a, dates, Qb, wy)
a.plot(dates, obs[wy], color=INK, lw=1.3, label='observed')
a.set_ylabel('flow (mm/day)')
month_axis(a)
a.legend(loc='upper left', fontsize=9)
a.set_title('The best fit, A, and two nearly as good fits, B and C')
t = fig.add_subplot(gs[1])
t.axis('off')
cells = ([[fmt(x) for x in picks.values()] + [''] for _, fmt, _ in ROWS]
         + [[f'{nse_b[k]:.2f}' for k in picks] + ['']])
labels = [r for r, _, _ in ROWS] + ['NSE, ordinary years 1981–2011']
tab = t.table(cellText=cells, rowLabels=labels, colLabels=[f'set {k}' for k in picks] + ['within range sampled'],
              colWidths=[0.1] * 3 + [0.24], loc='center right', edges='horizontal')
tab.auto_set_font_size(False)
tab.set_fontsize(9.5)
tab.scale(1, 1.35)
for (r, c), cell in tab.get_celld().items():
    cell.set_edgecolor(GRID)
    cell.get_text().set_color(INK if r == 0 or c >= 0 else INK2)
    if r == 0:
        cell.get_text().set_fontweight('bold')
    if c == -1:
        cell.get_text().set_ha('right')
        cell.PAD = 0.03
STRIP = len(picks)  # column index of the strips
tab[0, STRIP].get_text().set_fontweight('normal')
tab[0, STRIP].get_text().set_color(INK2)
for i in range(STRIP + 1):  # a color key above each column instead of colored text
    tab[0, i].set_linewidth(0)
fig.canvas.draw()
to_fig = fig.transFigure.inverted()
for i, k in enumerate(picks):
    bb = tab[0, i].get_window_extent().transformed(to_fig)
    fig.add_artist(plt.Line2D([bb.x0 + 0.2 * bb.width, bb.x1 - 0.2 * bb.width], [bb.y1 + 0.004] * 2,
                              color=SETS[k], lw=3, solid_capstyle='butt', transform=fig.transFigure))
for r, (_, _, pos) in enumerate(ROWS, start=1):
    bb = tab[r, STRIP].get_window_extent().transformed(to_fig)
    x0, x1 = bb.x0 + 0.1 * bb.width, bb.x1 - 0.1 * bb.width
    yc = (bb.y0 + bb.y1) / 2
    fig.add_artist(plt.Line2D([x0, x1], [yc, yc], color=BAND, lw=2.5, solid_capstyle='round',
                              transform=fig.transFigure))
    for j, (k, x) in enumerate(picks.items()):  # slightly apart vertically so ties stay visible
        fig.add_artist(plt.Line2D([x0 + pos(x) * (x1 - x0)], [yc + (1 - j) * 0.18 * bb.height], marker='o',
                                  ms=5.5, color=SETS[k], markeredgecolor=SURFACE, markeredgewidth=0.8, lw=0,
                                  transform=fig.transFigure))
save(fig, 'fig3')

# ------------------------------------------------------------------ figure 3b (optional)
# CAMELS max_water_content: soil depth x porosity, the water the soil holds when saturated
soil = pd.read_csv('data/camels_soil.txt', sep=';', dtype={'gauge_id': str}).set_index('gauge_id')
soil_max = 1000 * soil.loc[BLACKWOOD, 'max_water_content']  # mm
X, score = mc['X'], mc['score']
SHOW_ABOVE = 0.5  # below this, the plot would squash the top into a strip
show = score > SHOW_ABOVE
fig, axs = plt.subplots(1, 2, figsize=(W, 3.0), sharey=True)
for a, n, xl in [(axs[0], 'TT', 'temperature below which precipitation is snow (°C)'),
                 (axs[1], 'FC', 'soil water capacity (mm)')]:
    a.scatter(X[show, col[n]], score[show], s=1.5, color=BAND, rasterized=True, lw=0, zorder=1)
    a.scatter(Xg[:, col[n]], sg, s=4, color=INK2, lw=0, zorder=3)
    a.set_xlim(*RANGES[n])  # the full range sampled
    a.set_xlabel(xl)
axs[0].set_ylabel('NSE')
# The soil map's limit, with the good fits past it tinted: they fit just as well.
fc_good = Xg[:, col['FC']]
over = (fc_good > soil_max).mean()
past = show & (X[:, col['FC']] > soil_max)
axs[1].scatter(X[past, col['FC']], score[past], s=1.5, color='#ecd5ce', rasterized=True, lw=0, zorder=1)
axs[1].scatter(fc_good[fc_good > soil_max], sg[fc_good > soil_max], s=4, color='#b5523b', lw=0, zorder=3)
axs[1].axvline(soil_max, color='#b5523b', lw=1.2, zorder=2)  # behind the good fits (zorder 3)
axs[1].text(soil_max + 8, SHOW_ABOVE + 0.008, f"water capacity\nlimit ({soil_max:.0f} mm)", color='#b5523b',
            fontsize=8.5, va='bottom', zorder=4)
fig.suptitle(f'NSE of {len(X):,} random parameter sets', x=0.012, ha='left',
             fontsize=12, fontweight='bold')
fig.text(0.012, -0.15, f'Dark points: the {good.sum():,} good fits (NSE above {THRESHOLD:.2f}). '
         f'Sets below {SHOW_ABOVE} ({100 * (1 - show.mean()):.0f}% of all) are not shown.\n'
         f'Water capacity limit: the most water the soil can hold. '
         f'Red: the {100 * over:.0f}% of dark points above it.', color=MUTED, fontsize=9)
fig.tight_layout()
save(fig, 'fig3b')

# ------------------------------------------------------------------ figure 4 and 4b
# Train/test splits (calibrate.py). Each row is one fit: its NSE on the years it was trained
# on (filled) and on the years it was tested on (open).
def train_test(rows, title, note, name):
    fig, s = plt.subplots(figsize=(W, 0.75 + 0.78 * len(rows)))
    for row, (label, train, test) in enumerate(rows):
        y_ = len(rows) - 1 - row
        s.plot([test, train], [y_, y_], color=BAND, lw=3, solid_capstyle='butt', zorder=1)
        s.plot(train, y_, marker='o', ms=9, color=INK, lw=0, zorder=2)
        s.plot(test, y_, marker='o', ms=9, color=SURFACE, markeredgecolor=INK, markeredgewidth=1.8, lw=0,
               zorder=2)
        left, right = (test, train) if test < train else (train, test)
        s.text(right + 0.012, y_, f'{right:.2f}', ha='left', va='center', color=INK2, fontsize=9.5)
        s.text(left - 0.012, y_, f'{left:.2f}', ha='right', va='center', color=INK2, fontsize=9.5)
        s.text((train + test) / 2, y_ + 0.12, f'{test - train:+.2f}'.replace('-', '−'), ha='center',
               va='bottom', color=MUTED, fontsize=9)
        if row == 0:  # label the two kinds of dot once, on the top row
            for v, which in [(train, 'training years'), (test, 'test years')]:
                s.text(v, y_ + 0.3, which, ha='center', va='bottom', color=INK, fontsize=9.5)
        print(f'{name}: {label.replace(chr(10), " ")}: train {train:.3f}, test {test:.3f}')
    s.set_yticks(range(len(rows))[::-1], [label for label, _, _ in rows])
    s.set_ylim(-0.5, len(rows) - 0.25)
    s.set_xlim(0.3, 0.8)
    s.set_xlabel('NSE')
    s.grid(axis='y', visible=False)
    s.spines['left'].set_visible(False)
    s.set_title(title)
    s.annotate(note, (0, 0), xycoords='axes fraction', xytext=(0, -42), textcoords='offset points', va='top',
               color=MUTED, fontsize=9)
    save(fig, name)


def split_nse(stage, mask):
    p = calib[stage]['params']
    return nse(run(d, [p[n] for n in names])[mask], obs[mask])


# Cross-validation (crossval.py): each fold's fit is trained on 20 ordinary years and
# tested on the other 5. For a scheme, the test score pools every year's prediction from
# the fit that didn't see it; the training score is the mean over the five fits.
cv = [r for r in json.load(open(f'data/{BLACKWOOD}_crossval.json')) if r['stage'] == STAGES[-1][0]]


def pooled(scheme):
    folds = [r for r in cv if r['scheme'] == scheme]
    sim = np.full(len(obs), np.nan)
    for r in folds:
        sim[ORD & d.wy.isin(r['test_wys']).values] = r['sim_test']
    return np.mean([r['nse_train'] for r in folds]), nse(sim[ORD], obs[ORD])


last = next(r for r in cv if r['scheme'] == 'blocked' and r['test_wys'][-1] == 2011)
train_test([('test years: every 5th year', *pooled('interleaved')),
            ('test years: 5-year blocks', *pooled('blocked')),
            (f'test years: {last["test_wys"][0]}–{last["test_wys"][-1]},\nthe last and driest block',
             last['nse_train'], last['nse_test'])],
           'NSE on the training years vs. the test years',
           'Full model, Blackwood Creek. Each fit is trained on 20 of the 25 ordinary years and tested on the other 5.\n'
           'Top two rows: five fits each, so every year is tested once. Bottom row: one of the five block fits.',
           'fig4')
DROUGHT = drought_years(d)
train_test([('trained on ordinary years,\ntested on drought years', split_nse('4: + soil', ORD),
             split_nse('4: + soil', DROUGHT)),
            ('trained on drought years,\ntested on ordinary years', split_nse('drought', DROUGHT),
             split_nse('drought', ORD))],
           'NSE on ordinary years vs. drought years',
           'Blackwood Creek. Ordinary years: 1981–1986, 1993–2011. Drought years: 1987–1992, 2012–2014.', 'fig4b')

# ------------------------------------------------------------------ figure 5
# Left: the whole year, where A, B and C overlap. Right: late summer on a log scale,
# where they come apart.
summer = wy & d.index.month.isin([7, 8, 9])
sd = d.index[summer]
fig, (a0, a1) = plt.subplots(1, 2, figsize=(W, 3.3), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.22))
for k, q in Qb.items():
    lab = f'{SET_LABELS[k]}, NSE {nse_b[k]:.2f}'
    a0.plot(dates, q[wy], color=SETS[k], lw=1.4, label=lab)
    a1.plot(sd, q[summer], color=SETS[k], lw=1.6)
a0.plot(dates, obs[wy], color=INK, lw=1.2, label='observed')
a1.plot(sd, obs[summer], color=INK, lw=1.3)
# Outline the late-summer stretch and connect it to the right panel
top = 1.25 * max(obs[summer].max(), *(q[summer].max() for q in Qb.values()))
x0_, x1_ = mdates.date2num(sd[0]), mdates.date2num(sd[-1])
a0.add_patch(plt.Rectangle((x0_, 0), x1_ - x0_, top, fill=False, ec=INK2, lw=0.9, zorder=4))
for ya, yb in [(top, 1), (0, 0)]:
    fig.add_artist(ConnectionPatch((x1_, ya), (0, yb), coordsA=a0.transData, coordsB=a1.transAxes,
                                   color=MUTED, lw=0.8))
a0.set_ylabel('flow (mm/day)')
a0.set_title('Water year 2000', fontsize=10.5)
a0.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[10, 1, 4, 7]))
a0.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
a1.set_yscale('log')
a1.set_title('July–September, log scale', fontsize=10.5)
a1.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[7, 8, 9, 10]))
a1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
a1.yaxis.tick_right()  # tick labels away from the zoom lines on the left
a1.spines['right'].set_visible(True)
a1.spines['left'].set_visible(False)
a1.set_xlim(sd[0], sd[-1] + pd.Timedelta(days=62))
ends = {k: q[summer][-1] for k, q in Qb.items()} | {'obs': obs[summer][-1]}
# value on the last day at the right edge, nudged apart (in log space) where two are close
order = sorted(ends, key=ends.get)
ys = [np.log10(ends[k]) for k in order]
for i in range(1, len(ys)):
    ys[i] = max(ys[i], ys[i - 1] + 0.09)
for k, y_ in zip(order, ys):
    a1.text(sd[-1] + pd.Timedelta(days=3), 10 ** y_, f'{ends[k]:.2f}' + (' observed' if k == 'obs' else ''),
            va='center', fontsize=8.5, color=INK if k == 'obs' else SETS[k])
for k in picks:
    print(f'fig5: set {k} on {sd[-1].date()}: {ends[k]:.3f} ({ends["obs"] / ends[k]:.1f}x below observed)')
fig.suptitle('NSE and late-summer flow of sets A, B and C', x=0.012, ha='left', fontsize=12, fontweight='bold', y=1.07)
a0.legend(loc='upper center', bbox_to_anchor=(0.9, -0.14), ncol=4, fontsize=9, handlelength=1.5, columnspacing=1.2)
fig.text(0.012, -0.17, f'Blackwood Creek, water year {WY}. NSE is over all ordinary years 1981–2011.',
         color=MUTED, fontsize=9)
save(fig, 'fig5')

# ------------------------------------------------------------------ figure 5b
# Where the best fit's NSE error comes from, month by month, vs how wrong it is each month
best = sims[-1]
months = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # water-year order
sq = (best - obs) ** 2
share = [100 * sq[ORD & (d.index.month == mo)].sum() / sq[ORD].sum() for mo in months]
rel = [100 * np.median(np.abs(best / obs - 1)[ORD & (d.index.month == mo)]) for mo in months]
LATE = [8, 9]
colors = [INK if mo in LATE else BAND for mo in months]
names_m = [pd.Timestamp(2000, mo, 1).strftime('%b') for mo in months]
fig, (e1, e2) = plt.subplots(2, 1, figsize=(W, 4.8), sharex=True, gridspec_kw=dict(hspace=0.45))
e1.bar(names_m, share, color=colors, width=0.7)
e1.axhline(100 / 12, color=MUTED, lw=1, ls='--')
e1.text(11.45, 100 / 12 + 1, 'equal share', color=MUTED, fontsize=8.5, ha='right', va='bottom')
e1.set_ylabel('% of total')
for x_, (v, mo) in enumerate(zip(share, months)):
    if mo in LATE:
        e1.text(x_, v + 0.6, f'{v:.2f}%', ha='center', va='bottom', color=INK, fontsize=8.5)
e1.set_title("Share of the score's error from each month")
e2.bar(names_m, rel, color=colors, width=0.7)
e2.set_ylabel('% of observed flow')
e2.set_title('Typical daily error in each month')
for ax in (e1, e2):
    ax.grid(axis='x', visible=False)
late_share = sum(v for v, mo in zip(share, months) if mo in LATE)
e2.text(0, -0.32, f'Set A, the best fit (NSE {calib["4: + soil"]["nse"]:.2f}), ordinary years 1981–2011. '
        f'August and September (dark): {late_share:.1f}% of the error.\n'
        'Error share: squared errors, as NSE counts them. Typical error: median of |model − observed| / observed.',
        transform=e2.transAxes, color=MUTED, fontsize=9)
save(fig, 'fig5b')

# ------------------------------------------------------------------ figure 6
coop = pd.read_csv('data/coop_USC00048758.csv', parse_dates=['DATE']).set_index('DATE')
diff = (winter_mean(snotel_tmin('848')) - winter_mean(coop.TMIN)).dropna()
before, after = diff.loc[:2003], diff.loc[2004:]
fig, a = plt.subplots(figsize=(W, 3.2))
a.plot(diff.index, diff.values, color=TEMP, lw=1.6, marker='o', ms=4, markeredgecolor=SURFACE, markeredgewidth=1)
for part in (before, after):
    a.plot([part.index[0] - 0.4, part.index[-1] + 0.4], [part.mean()] * 2, color=MUTED, lw=1)
signed = lambda v: f'{v:+.1f}'.replace('-', '−')
a.text(1997.0, before.mean() + 0.1, f'average {signed(before.mean())} °C', color=INK2, fontsize=9, va='bottom')
a.text(2008.0, after.mean() - 0.1, f'average {signed(after.mean())} °C', color=INK2, fontsize=9, va='top')
a.text(2004.3, -1.1, f'change {signed(after.mean() - before.mean())} °C', color=INK, fontsize=9.5, va='center')
a.axhline(0, color=MUTED, lw=0.8)
a.set_ylabel('difference (°C)')
a.set_title('Winter night temperature at Ward Creek minus Tahoe City')
a.text(0, -0.24, 'Mean December–March daily minimum, by water year.\n'
       'Ward Creek #3 is a SNOTEL station; Tahoe City is a NOAA cooperative station.', transform=a.transAxes, color=MUTED, fontsize=9)
save(fig, 'fig6')
print('step', f'{after.mean() - before.mean():+.2f} °C')
# The same comparison for the model's own input: Daymet over Blackwood minus Tahoe City
dm = (winter_mean(d.tmin) - winter_mean(coop.TMIN)).dropna()
print(f'Daymet minus Tahoe City: step {dm.loc[2004:].mean() - dm.loc[:2003].mean():+.2f} °C at 2004')
