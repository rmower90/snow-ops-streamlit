# hrrr_clip_aso_domains.py
import numpy as np
from datetime import datetime
import pyproj
import os
import sys
import xarray as xr
from pathlib import Path


# gets rid of a lot of annoying warnings, we know there are NaNs (might want to remove this to double check)
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(1,'/home/rossamower/bin/scripts/')

import metadata as metadata

# functions.
def pad_zero(val):
    str_val = str(val)
    if len(str_val) == 1:
        return '0'+str_val
    else:
        return str_val

# python imports.
aso_site_name = sys.argv[1]
year = sys.argv[2]
month = sys.argv[3]

# load metadata.
cfg = metadata.load_yaml(Path(f"/home/rossamower/work/aso/configs/regions/{aso_site_name}.yaml"))

# get slicing indices.
idy_min,idy_max,idx_min,idx_max = metadata.get_forcing_indices(cfg,'hrrr_insitu_buff_indexing')



#idy_min = int(sys.argv[2])
#idy_max = int(sys.argv[3])
#idx_min = int(sys.argv[4])
#idx_max = int(sys.argv[5])
#water_year = int(sys.argv[6])

print('PROCESSING')
print(aso_site_name)
print(f'idy range: {idy_min},{idy_max}')
print(f'idx range: {idx_min},{idx_max}')
print(f'month: {year}-{month}')

# wrf directory.
wrf_dir = '/home/rossamower/work/aso/data/met/WUS/hrrr/F06/'
# month dir.
month_dir = f'{wrf_dir}3_hrly/year/{year}/month/{pad_zero(month)}/'
# WRF variables.
wrf_vars = ['PREC_ACC_NC','RELH','T2','WDIR_EREL','WSPD','RAIN_ACC','SNOW_ACC']
# campaign output directory.
campaign_dir = f'/home/rossamower/work/aso/data/met/{aso_site_name}/hrrr/3_hrly/year/{year}/month/{pad_zero(month)}/'
# make directory if doesn't exist.
if not os.path.exists(campaign_dir): os.makedirs(campaign_dir)

print('SLICING VARIABLES....')
for var in wrf_vars:
    fpath = f'{month_dir}{var}.nc'
    try:
        # load dataset.
        ds = xr.load_dataset(fpath)
        try:
            # slice dataset.
            ds_slice = ds.isel({'south_north':slice(idy_min,idy_max),
                    'east_west':slice(idx_min,idx_max)})
        except:
            # slice dataset.
            ds_slice = ds.isel({'south_north':slice(idy_min,idy_max),
                    'west_east':slice(idx_min,idx_max)})
        # make directory if doesn't exist.
        ds_slice.to_netcdf(f'{campaign_dir}{var}.nc')
    except:
        print(f'Could not process, {year}-{month}; {var}')

print('FINISHED....')



