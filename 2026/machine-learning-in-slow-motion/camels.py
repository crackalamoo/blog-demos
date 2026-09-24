# Load one CAMELS basin: Daymet basin-mean forcing + USGS streamflow, daily.
import json
import sys
import numpy as np
import pandas as pd

BLACKWOOD = '10336660'
GENERAL = '10336645'

# California's two big droughts inside the CAMELS record (water years, Oct-Sep)
DROUGHT_WYS = list(range(1987, 1993)) + [2012, 2013, 2014]


def gauge_arg(default=BLACKWOOD):
    return sys.argv[1] if len(sys.argv) > 1 else default


def calibration_years(d):
    """Mask for calibration: every full non-drought water year with observed flow.
    WY 1980 is a partial year and serves as model warm-up."""
    return (d.wy >= 1981).values & ~d.wy.isin(DROUGHT_WYS).values & d.q.notna().values


def load(gauge=BLACKWOOD):
    path = f'data/{gauge}_lump_cida_forcing_leap.txt'
    with open(path) as f:
        lat = float(f.readline())
        f.readline()  # elevation
        area = float(f.readline())  # m^2
    met = pd.read_csv(path, sep=r'\s+', skiprows=3)
    met.index = pd.to_datetime(dict(year=met.Year, month=met.Mnth, day=met.Day))
    met = met.rename(columns={'prcp(mm/day)': 'prcp', 'tmax(C)': 'tmax', 'tmin(C)': 'tmin'})
    met = met[['prcp', 'tmax', 'tmin']]

    q = pd.read_csv(f'data/{gauge}_streamflow_qc.txt', sep=r'\s+', header=None,
                    names=['gauge', 'year', 'month', 'day', 'cfs', 'flag'])
    q.index = pd.to_datetime(dict(year=q.year, month=q.month, day=q.day))
    cfs = q.cfs.where(q.cfs >= 0)
    # ft^3/s -> m^3/day -> mm/day over the basin
    met['q'] = cfs * 0.3048**3 * 86400 / area * 1000

    met['tmean'] = (met.tmax + met.tmin) / 2
    met['pet'] = hargreaves(met.index.dayofyear.values, lat, met.tmin.values, met.tmax.values)
    met['wy'] = met.index.year + (met.index.month >= 10)
    return met, dict(lat=lat, area_km2=area / 1e6)


def hargreaves(doy, lat_deg, tmin, tmax):
    """Hargreaves (1985) potential evapotranspiration, mm/day."""
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(-np.tan(phi) * np.tan(delta))
    # extraterrestrial radiation, MJ/m^2/day, then as mm/day of evaporated water
    ra = 24 * 60 / np.pi * 0.0820 * dr * (ws * np.sin(phi) * np.sin(delta)
                                          + np.cos(phi) * np.cos(delta) * np.sin(ws))
    ra_mm = 0.408 * ra
    tmean = (tmin + tmax) / 2
    return 0.0023 * ra_mm * (tmean + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0))


def snotel_tmin(sid):
    """Daily minimum temperature (°C) at one SNOTEL station."""
    e = next(e for e in json.load(open(f'data/snotel_{sid}_daily.json'))[0]['data']
             if e['stationElement']['elementCode'] == 'TMIN')
    s = pd.Series({v['date']: v.get('value') for v in e['values']}, dtype=float)
    s.index = pd.to_datetime(s.index)
    s = ((s - 32) * 5 / 9).where(lambda t: t.between(-40, 40))
    return s.loc['1990-10-01':]  # the first year of temperature is mostly 32 °F fill values


def winter_mean(s):
    """Mean over December-March of each water year, skipping winters with under 100 days of data."""
    s = s.dropna()
    s = s[s.index.month.isin([12, 1, 2, 3])]
    g = s.groupby(s.index.year + (s.index.month >= 10))
    return g.mean().where(g.count() >= 100).reindex(range(1981, 2015))
