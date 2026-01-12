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
    snowmodel_dir = cfg["data_filepaths"]["snowmodel_dir"]
    snodas_dir = cfg["data_filepaths"]["snodas_dir"]
    insitu_dir = cfg["data_filepaths"]["insitu_dir"]
    mlrPred_dir = cfg["data_filepaths"]["mlrPred_dir"]
    if not os.path.exists(mlrPred_dir): os.makedirs(mlrPred_dir)
    shape_crs = f'EPSG:{cfg["crs"]["epsg"]}'
    return elev_bin_labels, shape_fpath, demBin_fpath, aso_spatial_fpath, aso_tseries_fpath, snowmodel_dir, snodas_dir, insitu_dir, mlrPred_dir, shape_crs, cfg

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



if __name__ =="__main__":
    """
        INPUTS -----------------------------------------------
    """
    # user.
    aso_site_name = sys.argv[1]
    water_year = int(sys.argv[2])
    isSplit = bool(int(sys.argv[3]))
    isAccum = bool(int(sys.argv[4]))
    showOutput = bool(int(sys.argv[5]))
    # default settings.
    model_num,isMean,isCombination,prediction_mm_df,prediction_acreFt_df,prediction_pillow_df,user_qa_level,elev_band, QA_flag = get_default_settings()


    """
        LOAD DATA -----------------------------------------------
    """
    # load metadata information.
    elev_bin_labels, shape_fpath, demBin_fpath, aso_spatial_fpath, aso_tseries_fpath, snowmodel_dir, snodas_dir, insitu_dir, mlrPred_dir, shape_crs,cfg = load_aso_metadata(aso_site_name)

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

    print(f'RUNNING MLR PREDICTION: {aso_site_name}; up to {str(obs_data_test_ds.time.values[-1])[0:10]}')
    print('')

    """
        PREPROCESSING DATA -----------------------------------------------
    """
    print('PREPROCESSING DATA ...\n')
    aso_spatial_test,aso_tseries_test,aso_spatial_train,aso_tseries_train,obs_data_train,obs_data_qa = preprocessing.train_test_split(aso_spatial_ds,
                                                                                                      aso_tseries_ds,
                                                                                                      obs_data_train_lst,
                                                                                                      water_year)

    df_sum_total = preprocessing.combine_aso_insitu(obs_data_train,
                             aso_tseries_train['aso_swe'],
                             elev_idx = -1, # total = -1
                            )
    
    drop_na_df,all_pils,baseline_pils = preprocessing.generate_drop_NaNs_table(df_sum_total,
                                                                           obs_data_qa)

    # pillows = [i for i in pillows if i not in exclude_pillows]

    historic_vals_df = preprocessing.create_qa_tables(obs_data_train_lst, [], isQA=False)

    impute_dir = f'{mlrPred_dir}imputation/'
    
    obs_data_impute,pils_removed,impute_na_df = preprocessing.imputation_w_pillows(df_sum_total,
                                                                                   all_pils,
                                                                                   obs_data_qa,
                                                                                   aso_site_name,
                                                                                   water_year,
                                                                                   impute_dir,
                                                                                   obs_threshold = 0.50,
                                                                                   saveImputeCSV = True,
    )

    obs_data_impute_sm,pils_removed_sm,impute_na_df_sm = preprocessing.imputation_w_snowmodel(
                                                df_sum_total,
                                                all_pils,
                                                obs_data_qa,
                                                sm_train_ds,
                                                aso_site_name,
                                                water_year,
                                                impute_dir,
                                                obs_threshold = 0.50,
                                                saveImputeCSV = True,
                                                train_start_year = 2013,
                                                predictor_vars=("swed_best", "swed_second", "swed_third"),
                                            )
    
    """
        RUN MODEL -----------------------------------------------
    """
    print('RUNNING MLR MODEL ...\n')

    start = time.time()

    for t_idx in range(1,obs_data_test_lst[0].time.shape[0]):
        current_date_np = obs_data_test_lst[0].time[t_idx].values
        current_date = datetime(pd.to_datetime(current_date_np).year,pd.to_datetime(current_date_np).month,pd.to_datetime(current_date_np).day)
    

        current_vals_df,all_pils_QA,baseline_pils_,df_qa_table = preprocessing.process_daily_qa(
                                                                   t_idx,
                                                                   obs_data_test_lst_raw,
                                                                   obs_data_test_lst,
                                                                   baseline_pils,
                                                                   printOutput = showOutput,
                                                                              )

        try:
            summary_dict_all,df_sheet_lst_mm,df_sheet_lst_acreFt,df_sheet_pillow_lst = lm_model.run_all_mlr_models(aso_tseries_train.aso_swe,obs_data_qa,current_vals_df.reset_index(names = 'time'),
                                                    aso_site_name,all_pils,all_pils_QA,df_sum_total,baseline_pils_,start_wy,end_wy,isSplit,isAccum,
                                                    mlrPred_dir,current_date,dem_bin.dem_bin,elev_bin_labels,QA_flag=QA_flag,
                                                    modelNUM = model_num,isMean = False,saveModels = False,showOutput = showOutput,
                                                    saveValidation = False,pickledir = None,add_zeroASO = True,isCombination_ = isCombination)

            prediction_mm_df,prediction_acreFt_df,prediction_pillow_df = postprocessing.arrange_prediction_tables(df_sheet_lst_mm,
                                                                                                          df_sheet_lst_acreFt,
                                                                                                          df_sheet_pillow_lst,
                                                                                                          elev_bin_labels,
                                                                                                          prediction_mm_df,
                                                                                                          prediction_acreFt_df,
                                                                                                          prediction_pillow_df,
                                                                                                          )
            print(current_date)
        except:
            prediction_mm_df,prediction_acreFt_df,prediction_pillow_df = postprocessing.fill_NaNs_prediction_tables(df_sheet_lst_mm,
                                                                                                          df_sheet_lst_acreFt,
                                                                                                          df_sheet_pillow_lst,
                                                                                                          elev_bin_labels,
                                                                                                          current_date,
                                                                                                          prediction_mm_df,
                                                                                                          prediction_acreFt_df,
                                                                                                          prediction_pillow_df,
                                                                                                          )
            print(current_date,' COULD NOT PROCESS MLR!!')

                                             
    end = time.time()
    elapsed_min = (end - start) / 60
    print(f"Elapsed time: {elapsed_min:.2f} minutes")

    """
        OUTPUT
    """

    if isSplit:
        if isAccum:
            seasonal_dir = 'accum'
        else:
            seasonal_dir = 'melt'
    else:
        seasonal_dir = 'season'
    
    aso_stack_type = 'COMMON_MASK'

    dir_path = f'{mlrPred_dir}/{aso_stack_type}/'
    if not os.path.exists(dir_path): os.makedirs(dir_path)
    dir_path = f'{mlrPred_dir}/{aso_stack_type}/{seasonal_dir}/'
    if not os.path.exists(dir_path): os.makedirs(dir_path) 
#   dir_path = f'{mlrPred_dir}/{aso_stack_type}/{seasonal_dir}/{elev_dir}/'
#   if not os.path.exists(dir_path): os.makedirs(dir_path)
    dir_path = f'{mlrPred_dir}/{aso_stack_type}/{seasonal_dir}/'
    if not os.path.exists(dir_path): os.makedirs(dir_path)
    acre_path = f'{mlrPred_dir}/{aso_stack_type}/{seasonal_dir}/acreFt/'
    if not os.path.exists(acre_path): os.makedirs(acre_path)
    mm_path = f'{mlrPred_dir}/{aso_stack_type}/{seasonal_dir}/mm/'
    if not os.path.exists(mm_path): os.makedirs(mm_path)
    pillows_path = f'{mlrPred_dir}/{aso_stack_type}/{seasonal_dir}/pillows/'
    if not os.path.exists(pillows_path): os.makedirs(pillows_path)
    

    prediction_mm_df.to_csv(f'{mm_path}prediction_mm_wy{water_year}_combination.csv',index = False)
    prediction_acreFt_df.to_csv(f'{acre_path}prediction_acreFt_wy{water_year}_combination.csv',index = False)
    prediction_pillow_df.to_csv(f'{pillows_path}prediction_pillows_wy{water_year}_combination.csv',index = False)

    print('MLR PREDICTION COMPLETE!!!\n')


    







