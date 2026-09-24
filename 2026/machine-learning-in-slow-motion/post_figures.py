# The figures that go in the post. Run after calibrate.py and montecarlo.py.
# Three parameter sets, A, B and C, are followed through every figure: all three
# fit Blackwood Creek about equally well, but describe very different basins.
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from camels import load, calibration_years, snotel_tmin, winter_mean, BLACKWOOD, GENERAL
from model import STAGES, full_model, groundwater_step, nse

OUT = 'figures'
os.makedirs(OUT, exist_ok=True)
# The water year shown in figures 2-5 (Oct 1999 - Sep 2000). An ordinary year in which
# every stage and all three sets fit well, so the differences shown come from the model,
# not from one odd year. Fits vary a lot year to year; NSE in the figures is over all years.
WY = 2000
# Figure 1 shows the raw data over three water years, including the one above.
DATA_WYS = (1999, 2001)

INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8984'
GRID, BAND, SURFACE = '#e8e7e3', '#dcdbd6', '#ffffff'
SETS = {'A': '#2a78d6', 'B': '#eb6834', 'C': '#1baf7a'}
MODEL = '#e34948'
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
CAL = calibration_years(d)
GMASK = calibration_years(g)  # same years as Blackwood
forcing = lambda df: (df.prcp.values, df.tmean.values, df.pet.values)

mc = np.load(f'data/{BLACKWOOD}_montecarlo.npz')
names = list(mc['names'])
# A set "fits" if it scores within 0.04 of the best of the million random sets.
THRESHOLD = mc['score'].max() - 0.04
good = mc['score'] > THRESHOLD
Xg, sg = mc['X'][good], mc['score'][good]
col = {n: j for j, n in enumerate(names)}

# Picking A, B and C: among the sets that fit, rescale soil capacity FC and the slow
# store's half-life to 0-1 over their range in that group. A is the set with the
# smallest sum (shallow soil, fast-draining groundwater), B the largest (deep soil,
# slow groundwater), and C the one closest to halfway between them.
fc = Xg[:, col['FC']]
hl = half_life(Xg[:, col['K2']])
u = np.column_stack([(fc - fc.min()) / np.ptp(fc), (hl - hl.min()) / np.ptp(hl)])
ia, ib = u.sum(1).argmin(), u.sum(1).argmax()
ic = np.linalg.norm(u - (u[ia] + u[ib]) / 2, axis=1).argmin()
picks = {'A': Xg[ia], 'B': Xg[ib], 'C': Xg[ic]}


def run(df, x):
    return full_model(*forcing(df), *x)


Qb = {k: run(d, x) for k, x in picks.items()}
Qg = {k: run(g, x) for k, x in picks.items()}
band_b = np.array([run(d, x) for x in Xg])
band_g = np.array([run(g, x) for x in Xg])
nse_b = {k: nse(q[CAL], obs[CAL]) for k, q in Qb.items()}
nse_g = {k: nse(q[GMASK], gobs[GMASK]) for k, q in Qg.items()}
all_nse_b = np.array([nse(q[CAL], obs[CAL]) for q in band_b])
all_nse_g = np.array([nse(q[GMASK], gobs[GMASK]) for q in band_g])

with open(f'data/{BLACKWOOD}_abc.json', 'w') as f:
    json.dump({k: dict(params={n: float(v) for n, v in zip(names, x)},
                       nse_blackwood_cal=float(nse_b[k]), nse_general=float(nse_g[k]))
               for k, x in picks.items()}, f, indent=1)
for k, x in picks.items():
    print(k, {n: round(float(v), 3) for n, v in zip(names, x)},
          f'NSE Blackwood {nse_b[k]:.3f}, General {nse_g[k]:.3f}')

wy = (d.wy == WY).values
dates = d.index[wy]


def ensemble(ax, x, band, sel, label=True):
    ax.fill_between(x, np.percentile(band[:, sel], 5, 0), np.percentile(band[:, sel], 95, 0),
                    color=BAND, lw=0, label=f'middle 90% of the {len(band):,} good fits' if label else None)


def abc(ax, x, runs, sel):
    for k, q in runs.items():
        ax.plot(x, q[sel], color=SETS[k], lw=1.6, label=f'set {k}')


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
calib = json.load(open(f'data/{BLACKWOOD}_calibration.json'))
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
    a.text(0.98, 0.94, f'NSE {calib[stage]["nse_cal"]:.2f}', transform=a.transAxes, ha='right', va='top',
           color=INK, fontsize=11)
    a.text(0, -0.3, f'Blackwood Creek, water year {WY}.\nNSE is scored on all ordinary (non-drought) years '
           '1981–2014; 1 is a perfect fit.', transform=a.transAxes, color=MUTED, fontsize=9)
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
a.set_title('Three parameter sets that fit Blackwood Creek equally well')
t = fig.add_subplot(gs[1])
t.axis('off')
cells = [[fmt(x) for x in picks.values()] + [''] for _, fmt, _ in ROWS] + [[f'{nse_b[k]:.2f}' for k in picks] + ['']]
labels = [r for r, _, _ in ROWS] + ['NSE, ordinary years 1981–2014']
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
tab[0, 3].get_text().set_fontweight('normal')
tab[0, 3].get_text().set_color(INK2)
for i, k in enumerate(picks):  # a color key above each column instead of colored text
    cell = tab[0, i]
    cell.get_text().set_text(f'set {k}')
    cell.set_linewidth(0)
tab[0, 3].set_linewidth(0)
fig.canvas.draw()
to_fig = fig.transFigure.inverted()
for i, k in enumerate(picks):
    bb = tab[0, i].get_window_extent().transformed(to_fig)
    fig.add_artist(plt.Line2D([bb.x0 + 0.2 * bb.width, bb.x1 - 0.2 * bb.width], [bb.y1 + 0.004] * 2,
                              color=SETS[k], lw=3, solid_capstyle='butt', transform=fig.transFigure))
for r, (_, _, pos) in enumerate(ROWS, start=1):
    bb = tab[r, 3].get_window_extent().transformed(to_fig)
    x0, x1 = bb.x0 + 0.1 * bb.width, bb.x1 - 0.1 * bb.width
    yc = (bb.y0 + bb.y1) / 2
    fig.add_artist(plt.Line2D([x0, x1], [yc, yc], color=BAND, lw=2.5, solid_capstyle='round',
                              transform=fig.transFigure))
    for j, (k, x) in enumerate(picks.items()):  # A, B, C slightly apart vertically so ties stay visible
        fig.add_artist(plt.Line2D([x0 + pos(x) * (x1 - x0)], [yc + (1 - j) * 0.18 * bb.height], marker='o', ms=5.5,
                                  color=SETS[k], markeredgecolor=SURFACE, markeredgewidth=0.8, lw=0,
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
axs[0].set_ylabel('calibration NSE')
# The soil map's limit, with the good fits past it tinted: they fit just as well.
fc_good = Xg[:, col['FC']]
over = (fc_good > soil_max).mean()
past = show & (X[:, col['FC']] > soil_max)
axs[1].scatter(X[past, col['FC']], score[past], s=1.5, color='#ecd5ce', rasterized=True, lw=0, zorder=1)
axs[1].scatter(fc_good[fc_good > soil_max], sg[fc_good > soil_max], s=4, color='#b5523b', lw=0, zorder=3)
axs[1].axvline(soil_max, color='#b5523b', lw=1.2, zorder=2)  # behind the good fits (zorder 3)
axs[1].text(soil_max + 8, SHOW_ABOVE + 0.008, f"water capacity\nlimit ({soil_max:.0f} mm)", color='#b5523b',
            fontsize=8.5, va='bottom', zorder=4)
fig.suptitle(f'Calibration NSE of {len(X):,} random parameter sets', x=0.012, ha='left',
             fontsize=12, fontweight='bold')
fig.text(0.012, -0.15, f'Dark points: the {good.sum():,} good fits (NSE above {THRESHOLD:.2f}). '
         f'Sets below {SHOW_ABOVE} ({100 * (1 - show.mean()):.0f}% of all) are not shown.\n'
         f'Water capacity limit: the most water the soil can hold. '
         f'Red: the {100 * over:.0f}% of dark points above it.', color=MUTED, fontsize=9)
fig.tight_layout()
save(fig, 'fig3b')

# ------------------------------------------------------------------ figure 4
fig, s = plt.subplots(figsize=(4.6, 4.2))
for i in range(len(Xg)):
    s.plot([0, 1], [all_nse_b[i], all_nse_g[i]], color=BAND, lw=0.6, zorder=1)
for k in picks:
    s.plot([0, 1], [nse_b[k], nse_g[k]], color=SETS[k], lw=2, marker='o', ms=6,
           markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
# end labels, spread apart where sets land close together, with a leader to each point
order = sorted(picks, key=lambda k: nse_g[k])
ys = [nse_g[k] for k in order]
for i in range(1, len(ys)):
    ys[i] = max(ys[i], ys[i - 1] + 0.014)
for k, y_ in zip(order, ys):
    s.plot([1.04, 1.12], [nse_g[k], y_], color=MUTED, lw=0.6)
    s.text(1.14, y_, f'set {k}', va='center', color=INK2)
s.set_xticks([0, 1], ['Blackwood Creek\n(fitted here)', 'General Creek\n(7 km south)'])
s.set_xlim(-0.15, 1.45)
s.set_ylabel('NSE')
s.grid(axis='x', visible=False)
s.set_title('The same parameter sets\nscored on two creeks')
s.text(0, -0.22, f'Gray lines: all {good.sum():,} good fits on Blackwood.\n'
       'Both creeks scored on ordinary years 1981–2014.', transform=s.transAxes, color=MUTED, fontsize=9)
save(fig, 'fig4')
rank = pd.Series(all_nse_b).corr(pd.Series(all_nse_g), method='spearman')
print(f'General NSE of good fits {all_nse_g.min():.3f}-{all_nse_g.max():.3f}; rank correlation with Blackwood {rank:.2f}')

# ------------------------------------------------------------------ figure 5
summer = wy & d.index.month.isin([7, 8, 9])
fig, a = plt.subplots(figsize=(W, 3.6))
sd = d.index[summer]
ensemble(a, sd, band_b, summer)
abc(a, sd, Qb, summer)
a.plot(sd, obs[summer], color=INK, lw=1.3, label='observed')
a.set_yscale('log')
a.set_ylabel('flow (mm/day, log scale)')
month_axis(a)
a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), fontsize=9, ncol=5, handlelength=1.5, columnspacing=1.2)
a.set_title('Late-summer flow under the three parameter sets')
a.text(0, -0.3, f'Blackwood Creek, July–September {WY}', transform=a.transAxes, color=MUTED, fontsize=9)
save(fig, 'fig5')

# ------------------------------------------------------------------ figure 5b
# Where the best fit's NSE error comes from, month by month, vs how wrong it is each month
best = sims[-1]
months = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # water-year order
sq = (best - obs) ** 2
share = [100 * sq[CAL & (d.index.month == mo)].sum() / sq[CAL].sum() for mo in months]
rel = [100 * np.median(np.abs(best / obs - 1)[CAL & (d.index.month == mo)]) for mo in months]
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
e2.text(0, -0.32, f'Best fit (NSE {calib["4: + soil"]["nse_cal"]:.2f}), ordinary years 1981–2014. '
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
