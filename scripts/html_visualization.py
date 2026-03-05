# snowmodel_basin_process.py
"""
    Script identifies snowmodel bounds and projection and changes input into topo_vege scripts.
"""
import xarray as xr
import os 
import rioxarray
from pyproj import CRS, Transformer
import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
from rasterio.enums import Resampling
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import time
import rioxarray as rxr
import matplotlib.pyplot as plt

sys.path.insert(1, '/home/rossamower/bin/scripts/')

import metadata as metadata
import plotting as plotting
import preprocessing as preprocessing
import lm_model as lm_model
import postprocessing as postprocessing




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
    snowmodel_dir = cfg["data_filepaths"]["snowmodel_dir"]
    snodas_dir = cfg["data_filepaths"]["snodas_dir"]
    insitu_dir = cfg["data_filepaths"]["insitu_dir"]
    mlrPred_dir = cfg["data_filepaths"]["mlrPred_dir"]
    if not os.path.exists(mlrPred_dir): os.makedirs(mlrPred_dir)
    shape_crs = f'EPSG:{cfg["crs"]["epsg"]}'
    return elev_bin_labels, shape_fpath, demBin_fpath, aso_spatial_fpath, aso_tseries_fpath, snowmodel_dir, snodas_dir, insitu_dir, mlrPred_dir, uaswe_dir, shape_crs, cfg

def load_aso_data(aso_spatial_fpath: str,
                  aso_tseries_fpath: str,
                  demBin_fpath: str,
                  shape_fpath: str,
                  shape_crs: str,
                 ):
    aso_spatial_ds = xr.open_dataset(aso_spatial_fpath,engine ='netcdf4')
    aso_demBin_ds = xr.open_dataset(demBin_fpath,engine ='netcdf4')
    aso_tseries_ds = xr.open_dataset(aso_tseries_fpath,engine ='netcdf4')
    shape_geog_gdf = gpd.read_file(shape_fpath)
    shape_proj_gdf = shape_geog_gdf.to_crs(shape_crs)

    return aso_spatial_ds, aso_demBin_ds, shape_proj_gdf, aso_tseries_ds
                  







def generate_snowmodel_basin_todolist(year_str,
                                     month_str,
                                     day_str,
                                     snowmodel_dir,
                                     aso_site_name):
    
    df = pd.read_csv(f'{snowmodel_dir}/{aso_site_name}_snowmodel_daily_download_checklist.csv')
    last_date_str = df['download_date'].iloc[-1]
    current_date_str = f'{year_str}-{pad_zero(month_str)}-{pad_zero(day_str)}'

    date_range = np.arange(np.datetime64(last_date_str), np.datetime64(current_date_str) + np.timedelta64(1, "D"), np.timedelta64(1, "D"))
    lst = []
    for i in range(1,len(date_range)):
        lst.append([str(date_range[i])])
    df_todo = pd.DataFrame(lst,columns = ['date'])

    df_todo.to_csv(f'{snowmodel_dir}/{aso_site_name}_snowmodel_daily_download_todo.csv',index = False)
    return

def generate_snowmodel_basin_checklist(date_str,
                                     snowmodel_dir,
                                     aso_site_name):
    year_str = date_str[0:4]
    month_str = date_str[4:6]
    day_str = date_str[6:8]

    df_day = pd.DataFrame(data = [f'{year_str}-{month_str}-{day_str}'],
                      columns = ['download_date'])
    df_day['download_date'] = pd.to_datetime(df_day['download_date'])

    if not os.path.exists(f'{snowmodel_dir}/{aso_site_name}_snowmodel_daily_download_checklist.csv'):
        df_day.to_csv(f'{snowmodel_dir}/{aso_site_name}_snowmodel_daily_download_checklist.csv',index = False)
    else:
        df = pd.read_csv(f'{snowmodel_dir}/{aso_site_name}_snowmodel_daily_download_checklist.csv')
        df['download_date'] = pd.to_datetime(df['download_date'])

        pd.concat([df,df_day]).to_csv(f'{snowmodel_dir}/{aso_site_name}_snowmodel_daily_download_checklist.csv',index = False)
    
    return          

def pad_zero(val_str):
    if len(val_str) ==1:
        return '0' + val_str
    else:
        return val_str

def create_datelist(snowmodel_dir,
                    aso_site_name):
    
    df = pd.read_csv(f'{snowmodel_dir}/{aso_site_name}_snowmodel_daily_download_todo.csv')
    date_lst = []
    for index, row in df.iterrows():
        date = row['date']
        date_str = date.replace('-','')
        date_lst.append(date_str)
    return date_lst

def dataset_to_list(ds: xr.Dataset) -> list:
    da_list = []
    for pil in ds.data_vars:
        da = ds[pil]
        da.name = pil
        da_list.append(da)
    return da_list

# def preprocess(ds):
#     return ds.load()

def snowmodel_swe_fpaths(base_dir,water_yrs,var):
    nc_lst = []
    for wy in water_yrs:
        nc_dir = f'{base_dir}wy_{wy}/netcdf/'
        for file in os.listdir(nc_dir):
            if var in file:
                nc_lst.append(nc_dir + file)
    return sorted(nc_lst)

def timing_vars(obs_data_test_ds: xr.Dataset):
    date_str = str(obs_data_test_ds.time.values[-1])[0:10]
    year_str = date_str[0:4]
    month_str = date_str[5:7]
    day_str = date_str[8:10]
    
    if int(month_str) >= 10:
        wy_str = str(int(year_str) +1)
    else:
        wy_str = year_str
    return year_str, month_str, day_str, wy_str

def area_m2_acres(dem_bin,aso_data_1,aso_site_name,applyMask = True):
    """
    Creates dictionary of different elevation bins in acres.
    Input:
      dem_bin: xarray dataset with elevation bins.
      aso_data_1: xarray dataset of aso flights.
      applyMask: python boolean to indicate whether aso common mask is being applied.
    Output:
      area_dict: dictionary of different elevation bins in acres.
    """
    # pull out common mask from aso data.
    mask = (~aso_data_1.aso_swe[-1].isnull()).values

    area_dict = {}
    for i in range(dem_bin.shape[0]):
        if applyMask:
            arr = dem_bin[i].where(mask).values.flatten()
        else:
            arr = dem_bin[i].values.flatten()
        count = len(arr[~np.isnan(arr)])
        m2 = count * 50 * 50
        acres = m2 * 0.000247105
        if aso_site_name in ['USCATM','USCASJ']:
          if dem_bin.elev[i].values == '<7000':
              area_dict['<7k'] = acres
          elif dem_bin.elev[i].values == '7000-8000':
              area_dict['7k-8k'] = acres
          elif dem_bin.elev[i].values == '8000-9000':
              area_dict['8k-9k'] = acres
          elif dem_bin.elev[i].values == '9000-10000':
              area_dict['9k-10k'] = acres
          elif dem_bin.elev[i].values == '10000-11000':
              area_dict['10k-11k'] = acres
          elif dem_bin.elev[i].values == '11000-12000':
              area_dict['11k-12k'] = acres
          elif dem_bin.elev[i].values == '>12000':
              area_dict['>12k'] = acres
          elif dem_bin.elev[i].values == 'total':
              area_dict['Total'] = acres
        elif aso_site_name == 'USCOBR':
          if dem_bin.elev[i].values == '<9000':
              area_dict['<9k'] = acres
          elif dem_bin.elev[i].values == '9000-10000':
              area_dict['9k-10k'] = acres
          elif dem_bin.elev[i].values == '10000-11000':
              area_dict['10k-11k'] = acres
          elif dem_bin.elev[i].values == '11000-12000':
              area_dict['11k-12k'] = acres
          elif dem_bin.elev[i].values == '12000-13000':
              area_dict['12k-13k'] = acres
          elif dem_bin.elev[i].values == '>13000':
              area_dict['>13k'] = acres
          elif dem_bin.elev[i].values == 'total':
              area_dict['Total'] = acres
        elif aso_site_name == 'USCOGE':
          if dem_bin.elev[i].values == '<9000':
              area_dict['<9k'] = acres
          elif dem_bin.elev[i].values == '9000-10000':
              area_dict['9k-10k'] = acres
          elif dem_bin.elev[i].values == '10000-11000':
              area_dict['10k-11k'] = acres
          elif dem_bin.elev[i].values == '11000-12000':
              area_dict['11k-12k'] = acres
          elif dem_bin.elev[i].values == '>12000':
              area_dict['>12k'] = acres
          elif dem_bin.elev[i].values == 'total':
              area_dict['Total'] = acres
    return area_dict

def current_wy_aso(aso_site_name: str,
                   water_year: int,
                   aso_spatial_ds: xr.Dataset,
                   aso_gdf_proj: gpd.GeoDataFrame,
                   area_dict: dict,
                   ):
    aso_eval_dir = f'/home/rossamower/work/aso/data/aso/{aso_site_name}/wy_{water_year}/raw/'
    mean_swe = []
    mean_swe_m = []
    date_lst = []

    if os.path.exists(aso_eval_dir):
        for file in os.listdir(aso_eval_dir):
            print(file)
            da = rxr.open_rasterio(aso_eval_dir + file).squeeze(drop = True)
            print('Initial Shape:', da.shape)
            aso_template = aso_spatial_ds['aso_swe'][0]
            mask = (~aso_template.isnull()).values
            aso_template = aso_template.rio.set_crs(da.rio.crs)
            da_tu_domain = da.rio.clip(aso_gdf_proj.geometry).rio.reproject_match(aso_template).where(mask)
            mean_swe.append(float(da_tu_domain.mean()) * 3.28084 * area_dict['Total'])
            mean_swe_m.append(float(da_tu_domain.mean()) * 1000)
            date_lst.append(f'{file[-12:-8]}-{file[-8:-6]}-{file[-6:-4]}')
    
        mean_swe_arr = np.array(mean_swe)
        date_arr = np.array(date_lst,dtype = np.datetime64)
        return mean_swe_arr, date_arr
    else:
        return None, None
    

def get_default_settings():
    user_elevation_interval = -1
    model_num = 0   
    isMean = False
    isCombination = True
    prediction_mm_df = None
    prediction_acreFt_df = None
    prediction_pillow_df = None
    isCombination = True
    user_qa_level = 0
    QA_flag = user_qa_level + 1
    elev_band = user_elevation_interval
    return model_num,isMean,isCombination,prediction_mm_df,prediction_acreFt_df,prediction_pillow_df,user_qa_level,elev_band, QA_flag

def load_snowtrax_uaswe(aso_site_name: str,
                        water_year: int):
    snowtrax_url = 'https://snow.water.ca.gov/service/plotly/data/download?dash=fcast_resources&file=csv/wsfr_snow.csv'
    basin_dict = {'USCASJ': 'SBF',
                  'USCATM': 'TLG',
                  'USCATR': 'TRF'}
    if aso_site_name not in basin_dict.keys():
        raise ValueError(f"ASO site name {aso_site_name} not found in basin dictionary.")
    
    snowtrax_df = pd.read_csv(snowtrax_url)
    basin_df = snowtrax_df[snowtrax_df['STA_ID'] == basin_dict[aso_site_name]]
    basin_df['DATE'] = pd.to_datetime(basin_df['DATE'])
    basin_df.set_index('DATE', inplace=True)

    basin_wy = basin_df.loc[f'{water_year-1}-10-01':, 'SWANN_UA_SWE_AF']


    return basin_wy.reset_index()



if __name__ =="__main__":
    """
        INPUTS -----------------------------------------------
    """
    # user.
    aso_site_name = sys.argv[1]
    water_year = int(sys.argv[2])
    plotDir = sys.argv[3]
    # isSplit = bool(int(sys.argv[3]))
    # isAccum = bool(int(sys.argv[4]))
    # showOutput = bool(int(sys.argv[5]))
    # default settings.
    model_num,isMean,isCombination,prediction_mm_df,prediction_acreFt_df,prediction_pillow_df,user_qa_level,elev_band, QA_flag = get_default_settings()


    """
        LOAD DATA -----------------------------------------------
    """
    # load metadata information.
    elev_bin_labels, shape_fpath, demBin_fpath, aso_spatial_fpath, aso_tseries_fpath, snowmodel_dir, snodas_dir, insitu_dir, mlrPred_dir, uaswe_dir, shape_crs,cfg = load_aso_metadata(aso_site_name)

    # pillows to exclude from QA.
    exclude_pillows = cfg['pillow_api']['exclude_pillows']
    elev_bin_edges_m, elev_bin_labels = metadata.get_elevation_bins(cfg)
    start_wy = int(cfg["aso_years"]["start"])
    end_wy = int(cfg["aso_years"]["end"])

    # load spatial data.
    aso_spatial_ds, dem_bin, shape_proj_gdf, aso_tseries_ds = load_aso_data(aso_spatial_fpath,
                                                                                aso_tseries_fpath,
                                                                                demBin_fpath,
                                                                                shape_fpath,
                                                                                shape_crs)
    # area dict.
    area_dict = area_m2_acres(dem_bin.dem_bin,aso_spatial_ds,aso_site_name,applyMask = True)

    # current wy aso.
    current_swe, current_dates = current_wy_aso(aso_site_name,
                                               water_year,
                                               aso_spatial_ds,
                                               shape_proj_gdf,
                                               area_dict)

    # load insitu data.
    ## training.
    obs_data_train_ds = xr.load_dataset(f'{insitu_dir}processed/pillow_wy_1980_2025_qa1.nc')
    obs_data_train_lst = dataset_to_list(obs_data_train_ds)

    # load snowmodel data 
    sm_train_ds = xr.open_zarr(f'{insitu_dir}hrrr_correlated_train_2017_2025_dowy.zarr', consolidated=False)
    sm_test_ds = xr.open_zarr(f'{insitu_dir}hrrr_correlated_test_2026.zarr', consolidated=False)

    # testing raw.
    obs_data_test_ds_raw = xr.load_dataset(f'{insitu_dir}raw/{aso_site_name}_insitu_obs_daily_wy_2026.nc')
    # match times.
    obs_data_test_ds_raw = obs_data_test_ds_raw.sel(time = sm_test_ds.time)
    obs_data_test_lst_raw = dataset_to_list(obs_data_test_ds_raw)

    # testing qa.
    obs_data_test_ds = xr.load_dataset(f'{insitu_dir}processed/{aso_site_name}_insitu_obs_daily_wy_2026.nc')
    # match times.
    obs_data_test_ds = obs_data_test_ds.sel(time = sm_test_ds.time)
    obs_data_test_lst = dataset_to_list(obs_data_test_ds)

    # initial pillow list.
    pillows = list(obs_data_test_ds.data_vars)
    pillows = [i for i in pillows if i not in exclude_pillows]

    # get timing variables.
    year_str, month_str, day_str, wy_str = timing_vars(obs_data_test_ds)

    # load UASWE data [EARLY].
    uaswe_df = pd.read_csv(f'{uaswe_dir}mean_swe_uaswe_acreFt_wy{water_year}.csv')
    # load UASWE data [PROVISIONAL].
    uaswe_provisional_df = pd.read_csv(f'{uaswe_dir}mean_swe_uaswe_acreFt_provisional_wy{water_year}.csv')
    # load UASWE snowtrax data.
    try:
        uaswe_snowtrax_df = load_snowtrax_uaswe(aso_site_name, water_year)
    except:
        print('Unable to load Snowtrax UASWE data.')
        uaswe_snowtrax_df = None

    # load SNODAS data.
    snodas_df = pd.read_csv(f'{snodas_dir}mean_swe_snodas_acreFt_wy{water_year}.csv')

    # load SnowModel data.
    sm_df = pd.read_csv(f'{snowmodel_dir}mean_swe_snowmodel_acreFt_wy{water_year}.csv')
    # HARDCODED BIAS CORRECTION FOR TOTAL SNOWMODEL SWE!!!
    sm_df["total"] = sm_df["total"] * 0.554

    # convert date columns to datetime.
    snodas_df['Date'] = pd.to_datetime(snodas_df['Date'])
    uaswe_df['Date'] = pd.to_datetime(uaswe_df['Date'])
    sm_df['Date'] = pd.to_datetime(sm_df['Date'])
    uaswe_provisional_df['Date'] = pd.to_datetime(uaswe_provisional_df['Date'])

    seasonal_dirs = ["season","accum","melt"]
    models = ["COMMON_MASK","SNOWMODEL_IMPUTE"]

    
    for aso_stack_type in models:
        count = 0
        mlr_tables = []
        mlr_identifiers = {}
        for seasonal_dir in seasonal_dirs:
            acre_path = f'{mlrPred_dir}/{aso_stack_type}/{seasonal_dir}/acreFt/'
            mm_path = f'{mlrPred_dir}/{aso_stack_type}/{seasonal_dir}/mm/'
            pillow_path = f'{mlrPred_dir}/{aso_stack_type}/{seasonal_dir}/pillows/'

            # read data.
            prediction_acreFt_df = pd.read_csv(f'{acre_path}prediction_acreFt_wy{water_year}_combination.csv')
            prediction_mm_df = pd.read_csv(f'{mm_path}prediction_mm_wy{water_year}_combination.csv')
            prediction_pillows_df = pd.read_csv(f'{pillow_path}prediction_pillows_wy{water_year}_combination.csv')

            # convert date columns to datetime.
            prediction_mm_df['Date'] = pd.to_datetime(prediction_mm_df['Date'])
            prediction_acreFt_df['Date'] = pd.to_datetime(prediction_acreFt_df['Date'])
            prediction_pillows_df['Date'] = pd.to_datetime(prediction_pillows_df['Date'])
            
            mlr_tables.append(prediction_acreFt_df)
            mlr_identifiers[count] = [aso_stack_type, seasonal_dir, 'acreFt']
            count += 1 
            mlr_tables.append(prediction_mm_df)
            mlr_identifiers[count] = [aso_stack_type, seasonal_dir, 'mm']
            count += 1 
            mlr_tables.append(prediction_pillows_df)
            mlr_identifiers[count] = [aso_stack_type, seasonal_dir, 'pillows']
            count += 1

        print(mlr_identifiers)
        print(f'RUNNING MLR VISUALIZATION: {aso_site_name}; up to {str(obs_data_test_ds.time.values[-1])[0:10]}')

        fig,ax = plt.subplots(1,2,figsize = (16, 8))
        plotting.html_timeseries_plot(mlr_tables,
                                  mlr_identifiers,
                                  uaswe_df,
                                  uaswe_provisional_df,
                                  snodas_df,
                                  sm_df,
                                  aso_site_name,
                                  current_dates,
                                  current_swe,
                                  uaswe_snowtrax_df,
                                  plotDir,
                                  aso_stack_type,
                                  train_infer = 'predict NaNs',
                                  saveFIG = False,
                                  start_date = '2026-01-01',
                                  end_date = None,
                                  ax=ax[0]
        )
        plotting.visualize_pillow_selection_heatmap(
            mlr_tables=mlr_tables,
            mlr_identifiers=mlr_identifiers,
            start_date='2026-01-01',
            end_date=None,
            train_infer='predict NaNs',
            figsize=(16, 8),
            ax=ax[1],
        )

        plt.savefig(f'{plotDir}/{aso_site_name}_{aso_stack_type}_timeseries_plot.png', dpi=200, bbox_inches='tight')
    

    print('')



    







