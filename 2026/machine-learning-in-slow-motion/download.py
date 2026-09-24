# Pull CAMELS basins out of the 3.4 GB Zenodo zip using HTTP range requests.
# Usage: python download.py [gauge_id ...]
import os
import sys
import urllib.request
from remotezip import RemoteZip

GAUGES = sys.argv[1:] or ['10336660', '10336645']
RECORD = 'https://zenodo.org/api/records/15529996/files'
ZIP_URL = f'{RECORD}/basin_timeseries_v1p2_metForcing_obsFlow.zip/content'

os.makedirs('data', exist_ok=True)

with RemoteZip(ZIP_URL) as z:
    for gauge in GAUGES:
        names = [n for n in z.namelist() if gauge in n
                 and ('daymet' in n and 'lump' in n or 'streamflow' in n)]
        for n in names:
            out = os.path.join('data', os.path.basename(n))
            with open(out, 'wb') as f:
                f.write(z.read(n))
            print('wrote', out)

# Soil-map estimate of soil water capacity
urllib.request.urlretrieve(f'{RECORD}/camels_soil.txt/content', 'data/camels_soil.txt')

# Winter night temperatures at Ward Creek #3 SNOTEL (next to Blackwood Creek) and the
# NOAA cooperative-observer station at Tahoe City, which is not part of SNOTEL.
AWDB = 'https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1'
urllib.request.urlretrieve(f'{AWDB}/data?stationTriplets=848:CA:SNTL&elements=TMIN'
                           '&duration=DAILY&beginDate=1979-10-01&endDate=2015-09-30',
                           'data/snotel_848_daily.json')
urllib.request.urlretrieve('https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries'
                           '&stations=USC00048758&startDate=1979-10-01&endDate=2015-09-30'
                           '&dataTypes=TMIN&units=metric&format=csv', 'data/coop_USC00048758.csv')
