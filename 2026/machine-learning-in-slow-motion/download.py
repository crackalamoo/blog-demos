# Pull CAMELS basins out of the 3.4 GB Zenodo zip using HTTP range requests.
# Usage: python download.py [gauge_id ...]
import os
import sys
from remotezip import RemoteZip

GAUGES = sys.argv[1:] or ['10336660']
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

