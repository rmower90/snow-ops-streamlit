# uaswe_basin_process.py
"""
    Script identifies snowmodel bounds and projection and changes input into topo_vege scripts.
"""
import xarray as xr
import os 
# import rioxarray
from pyproj import CRS, Transformer
import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
# from rasterio.enums import Resampling
from datetime import datetime, timedelta

sys.path.insert(1, '../../scripts/')

import metadata as metadata
import shape as shape
import pillow_api as pillow_api





def load_aso_metadata(aso_site_name: str,
                  config_dir: str = '/home/rossamower/work/aso/configs/',
                 ):
    cfg = metadata.load_yaml(Path(f"{config_dir}regions/{aso_site_name}.yaml"))
    elev_bin_edges_m, elev_bin_labels = metadata.get_elevation_bins(cfg)
    start_wy = int(cfg["aso_years"]["start"])
    end_wy = int(cfg["aso_years"]["end"])
    shape_fpath = cfg["data_filepaths"]["aso_shape"]
    demBin_fpath = cfg["data_filepaths"]["aso_demBin"]
    aso_spatial_fpath = cfg["data_filepaths"]["aso_spatial"]
    aso_tseries_fpath = cfg["data_filepaths"]["aso_temporal"]
    uaswe_dir = cfg["data_filepaths"]["uaswe_dir"]
    snodas_dir = cfg["data_filepaths"]["snodas_dir"]
    insitu_dir = cfg["data_filepaths"]["insitu_dir"]
    shape_crs = f'EPSG:{cfg["crs"]["epsg"]}'
    return elev_bin_labels, shape_fpath, demBin_fpath, aso_spatial_fpath, aso_tseries_fpath, uaswe_dir, snodas_dir,insitu_dir, shape_crs, cfg

def load_aso_data(aso_spatial_fpath: str,
                  demBin_fpath: str,
                  shape_fpath: str,
                  shape_crs: str,
                 ):
    aso_spatial_ds = xr.open_dataset(aso_spatial_fpath,engine ='netcdf4')
    aso_demBin_ds = xr.open_dataset(demBin_fpath,engine ='netcdf4')
    shape_geog_gdf = gpd.read_file(shape_fpath)
    shape_proj_gdf = shape_geog_gdf.to_crs(shape_crs)

    return aso_spatial_ds, aso_demBin_ds, shape_proj_gdf
                  




def generate_insitu_basin_todolist(year_str,
                                     month_str,
                                     day_str,
                                     uaswe_dir,
                                     aso_site_name):
    
    df = pd.read_csv(f'{uaswe_dir}/{aso_site_name}_insitu_daily_download_checklist.csv')
    last_date_str = df['download_date'].iloc[-1]
    current_date_str = f'{year_str}-{pad_zero(month_str)}-{pad_zero(day_str)}'

    date_range = np.arange(np.datetime64(last_date_str), np.datetime64(current_date_str) + np.timedelta64(1, "D"), np.timedelta64(1, "D"))
    lst = []
    for i in range(1,len(date_range)):
        lst.append([str(date_range[i])])
    df_todo = pd.DataFrame(lst,columns = ['date'])

    df_todo.to_csv(f'{uaswe_dir}/{aso_site_name}_insitu_daily_download_todo.csv',index = False)
    return

def generate_insitu_basin_checklist(date_str,
                                     uaswe_dir,
                                     aso_site_name):
    year_str = date_str[0:4]
    month_str = date_str[4:6]
    day_str = date_str[6:8]

    df_day = pd.DataFrame(data = [f'{year_str}-{month_str}-{day_str}'],
                      columns = ['download_date'])
    df_day['download_date'] = pd.to_datetime(df_day['download_date'])

    if not os.path.exists(f'{uaswe_dir}/{aso_site_name}_insitu_daily_download_checklist.csv'):
        df_day.to_csv(f'{uaswe_dir}/{aso_site_name}_insitu_daily_download_checklist.csv',index = False)
    else:
        df = pd.read_csv(f'{uaswe_dir}/{aso_site_name}_insitu_daily_download_checklist.csv')
        df['download_date'] = pd.to_datetime(df['download_date'])

        pd.concat([df,df_day]).to_csv(f'{uaswe_dir}/{aso_site_name}_insitu_daily_download_checklist.csv',index = False)
    
    return          

def pad_zero(val_str):
    if len(val_str) ==1:
        return '0' + val_str
    else:
        return val_str

def create_datelist(uaswe_dir,
                    aso_site_name):
    
    df = pd.read_csv(f'{uaswe_dir}/{aso_site_name}_insitu_daily_download_todo.csv')
    date_lst = []
    for index, row in df.iterrows():
        date = row['date']
        date_str = date.replace('-','')
        date_lst.append(date_str)
    return date_lst





if __name__ =="__main__":
    aso_site_name = sys.argv[1]
    year_str = sys.argv[2]
    month_str = sys.argv[3]
    day_str = sys.argv[4]

    elev_bin_labels, shape_fpath, demBin_fpath, aso_spatial_fpath, aso_tseries_fpath, uaswe_dir, snodas_dir,insitu_dir, shape_crs, cfg = load_aso_metadata(aso_site_name)

    
    aso_gdf_geog,aso_gdf_proj,aso_buff_gdf_geog,aso_buff_gdf_proj = shape.load_basin_shape(shape_fpath,
                                                                                           shape_crs,
                                                                                           aso_site_name = None,
                                                                                          buffer_size = 25000)
    
    # generate uaswe todo list.
    if os.path.exists(f'{insitu_dir}/{aso_site_name}_insitu_daily_download_checklist.csv'):
        generate_insitu_basin_todolist(year_str, month_str, day_str, insitu_dir, aso_site_name)
        # generate date list from checklist.
        date_lst = create_datelist(insitu_dir, aso_site_name)
        ds_all = xr.load_dataset(f'/home/rossamower/work/aso/data/insitu/{aso_site_name}/raw/{aso_site_name}_insitu_obs_daily_wy_2026.nc') 
    else:
        date_lst = [f'{year_str}{pad_zero(month_str)}{pad_zero(day_str)}']

    start_date = datetime(int(date_lst[0][0:4]), int(date_lst[0][4:6]), int(date_lst[0][6:8]))
    end_date = datetime(int(date_lst[-1][0:4]), int(date_lst[-1][4:6]), int(date_lst[-1][6:8]))

    # load in situ data from pillow api.
    print('DOWNLOADING CDEC PILLOW DATA...')
    obs_data_pil, cdec_locations, _, _ = pillow_api.load_pillow_api(
            aso_buff_gdf_geog, start_date, end_date, plot=False, debug=False
        )
    print('DOWNLOADING SNOTEL PILLOW DATA...')
    if aso_site_name == 'USCASJ':
        lle_polygon = pillow_api.load_lle_geography()
        
        obs_data_snot,snotel_locations = pillow_api.load_snotel_api(lle_polygon,
                                                           start_date, 
                                                           end_date,
                                                           cdec_locations.dropna(axis=0),
                                                           variable = 'SNOTEL:WTEQ_D',
                                                           plot = True)
    else:
        obs_data_snot,snotel_locations = pillow_api.load_snotel_api(aso_buff_gdf_geog,
                                                           start_date, 
                                                           end_date,
                                                           cdec_locations.dropna(axis=0),
                                                           variable = 'SNOTEL:WTEQ_D',
                                                           plot = True)
    
    print('COMBINING CDEC AND SNOTEL DATA...')
    obs_data_2, obs_locations = pillow_api.combine_basin_obs(aso_buff_gdf_geog,
                                                       cdec_locations.dropna(axis = 0),
                                                       snotel_locations,
                                                       obs_data_pil,
                                                       obs_data_snot)

    count = 0
    for pil in obs_data_2:
        if count == 0:
            ds = pil.to_dataset()
        else:
            ds[pil.name] = pil
        count += 1

    # snow pillow name information.
    name_rename = cfg['pillow_api']['snotel_rename']['name']
    network_rename = cfg['pillow_api']['snotel_rename']['network']
    id_rename = cfg['pillow_api']['snotel_rename']['id']
    all_pils = cfg['pillow_api']['pil_names'].split(',')
    print(all_pils)
    
    # snow pillow renaming.
    id_current = obs_locations[(obs_locations['name'] == name_rename) & (obs_locations['network'] == network_rename)]['id'].values[0]
    ds = ds.rename({id_current: id_rename})
    download_pillows = list(ds.data_vars.keys())
    
    # set missing downloaded vals to nan
    for pil_name in [i for i in all_pils if i not in download_pillows]:
        ds[pil_name] = xr.DataArray(np.nan * np.ones(ds['time'].shape),
                                   dims = ['time'],
                                   coords = {'time': ds['time']},
                                   name = pil_name)

    # sort dataset.
    ds = ds[sorted(ds.data_vars)]

    
    if not os.path.exists(f'/home/rossamower/work/aso/data/insitu/{aso_site_name}/raw/'):
        os.makedirs(f'/home/rossamower/work/aso/data/insitu/{aso_site_name}/raw/')
    if not os.path.exists(f'/home/rossamower/work/aso/data/insitu/{aso_site_name}/raw/{aso_site_name}_insitu_obs_daily_wy_2026.nc'):
        ds.to_netcdf(f'/home/rossamower/work/aso/data/insitu/{aso_site_name}/raw/{aso_site_name}_insitu_obs_daily_wy_2026.nc')   
    else:
        xr.concat([ds_all,ds],dim='time').to_netcdf(f'/home/rossamower/work/aso/data/insitu/{aso_site_name}/raw/{aso_site_name}_insitu_obs_daily_wy_2026.nc')   


    for date in date_lst:
        # update checklist
        generate_insitu_basin_checklist(date,f'/home/rossamower/work/aso/data/insitu/{aso_site_name}/',aso_site_name)
    





