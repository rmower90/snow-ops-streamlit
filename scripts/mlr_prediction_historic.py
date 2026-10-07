# mlr_prediction_historic.py
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

# -------------------------------------------------------------------------
# v5 experiment toggle: how ASO flights from the test water year are held out.
#   'wy'     -- exclude the ENTIRE test WY from training (v1-v4 leave-one-year-out)
#   'flight' -- include test WY flights in training, EXCEPT the specific flight date
#               being predicted (leave-one-flight-out)
# Set to 'wy' to revert to v1-v4 behavior.
# Only mlr_prediction_historic.py reads this; the real-time mlr_prediction.py is
# unaffected because preprocessing.train_test_split defaults to 'wy'.
# -------------------------------------------------------------------------
ASO_HOLDOUT_MODE = 'wy'

# -------------------------------------------------------------------------
# Which QA rung of the pillow ladder to train/test on.
#
# The imputation table is cached on disk keyed by (site, water year, suffix) ONLY, and the
# read-back is unconditional -- once the file exists it is never recomputed, regardless of
# whether the inputs changed. The tables under FRIANT/imputation/ were all written during
# the v1 run (2026-03-07) from qa1 data, so v2-v6 reused them three months later even
# though the qa ladder had changed the pillow record underneath: qa2 removed 36 flight-date
# cells inside 2017-2025, qa3 interpolated 4 more, qa5 zero-filled 2, and qa3/qa6 together
# changed 408 values inside the >= 2013 window that impute_pillow_prediction regresses on.
# Result: the "predict NaNs" half of v2-v6 was imputed from qa1 while the "drop NaNs" half
# correctly tracked the new data.
#
# Deriving the cache tag from the filename means changing the rung automatically re-keys
# the cache, so this cannot silently recur. One table per rung, computed once, reused after.
# -------------------------------------------------------------------------
OBS_QA_FILE = 'pillow_wy_1980_2025_qa6.nc'
OBS_QA_TAG = OBS_QA_FILE.replace('.nc', '').split('_')[-1]   # -> 'qa6'

# -------------------------------------------------------------------------
# Uncertainty ensemble.
#
# identify_best_stations already scores every combination of the candidate pillows taken
# 1..5 and keeps only the argmax. N_ENSEMBLE > 1 also predicts with the next-best
# combinations, giving a spread per (day, imputation mode, elevation band) instead of a
# single number. N_ENSEMBLE = 1 leaves the published path completely untouched.
#
# RANKED_TOP_K is how many ranked combinations to retain. Each one costs a training frame
# (see _build_frame in lm_model.py), so retaining more than N_ENSEMBLE is wasted work --
# keep them equal unless deliberately exploring deeper.
#
# Output goes to ensemble_tidy_wy{YYYY}.csv beside the prediction_* files. The
# `ensemble_` prefix matters: generateHTMLPred.js matches on a `prediction_acreFt_wy`
# prefix plus `_combination.csv`, and 3_MLR_Investigation.py globs prediction_mm_wy*.
# A different leading token keeps the new file invisible to both.
# -------------------------------------------------------------------------
N_ENSEMBLE = 10
RANKED_TOP_K = N_ENSEMBLE


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

if __name__ == "__main__":
    """
        INPUTS -----------------------------------------------
    """
    # user.
    aso_site_name = sys.argv[1]
    water_year = int(sys.argv[2])
    isSplit = bool(int(sys.argv[3]))
    isAccum = bool(int(sys.argv[4]))
    showOutput = bool(int(sys.argv[5]))
    pillowImputation_ = bool(int(sys.argv[6]))
    aso_stack_type = sys.argv[7]
    # default settings.
    model_num,isMean,isCombination,prediction_mm_df,prediction_acreFt_df,prediction_pillow_df,user_qa_level,elev_band, QA_flag = get_default_settings()

    # model types
    if isSplit:
        if isAccum:
            seasonal_dir = 'accum'
        else:
            seasonal_dir = 'melt'
    else:
        seasonal_dir = 'season'
    
    # if pillowImputation_:
    #     aso_stack_type = 'COMMON_MASK'
    # else:
    #     aso_stack_type = 'SNOWMODEL_IMPUTE'
    print(f'Basin: {aso_site_name}')
    print(f'Water Year: {water_year}')
    print(f'isSplit: {isSplit}')
    print(f'isAccum: {isAccum}')
    print(f'showOutput: {showOutput}')
    print(f'pillowImputation_: {pillowImputation_}')
    print(f'aso_stack_type: {aso_stack_type}')
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

    # obs data.
    obs_data = xr.load_dataset(f'{insitu_dir}processed/{OBS_QA_FILE}')
    # obs test.
    obs_data_test_ds = obs_data.where(obs_data.time>=np.datetime64(f'{water_year-1}-10-01'), drop = True) \
                       .where(obs_data.time<np.datetime64(f'{water_year}-10-01'), drop = True)    
    # obs train.
    obs_data_train_ds = obs_data.where(~obs_data.time.isin(obs_data_test_ds.time.values), drop = True)

    obs_data_train_lst = dataset_to_list(obs_data_train_ds)
    obs_data_test_lst = dataset_to_list(obs_data_test_ds)

    pillows = list(obs_data_test_ds.data_vars)
    pillows = [i for i in pillows if i not in exclude_pillows]
    # get timing variables.
    year_str, month_str, day_str, wy_str = timing_vars(obs_data_test_ds)
    """
        PREPROCESSING DATA -----------------------------------------------
    """
    print('PREPROCESSING DATA ...\n')

    aso_spatial_test,aso_tseries_test,aso_spatial_train,aso_tseries_train,obs_data_train,obs_data_qa = preprocessing.train_test_split(aso_spatial_ds,
                                                                                                  aso_tseries_ds,
                                                                                                  obs_data_train_lst,
                                                                                                  water_year,
                                                                                                  isFriant = True,
                                                                                                  aso_holdout_mode = ASO_HOLDOUT_MODE,
                                                                                                  )

    # v5 (flight holdout): pre-compute the set of test-WY ASO flight dates so the
    # daily loop can detect when today is one of them and exclude that single row
    # from training. For 'wy' mode this set is empty and the loop short-circuits.
    if ASO_HOLDOUT_MODE == 'flight':
        wy_start_dt = np.datetime64(f'{water_year-1}-10-01')
        wy_end_dt   = np.datetime64(f'{water_year}-10-01')
        test_wy_flight_dates = set(
            pd.Timestamp(d).normalize()
            for d in aso_tseries_ds.date.values
            if (wy_start_dt <= d < wy_end_dt)
        )
        print(f'v5 leave-one-flight-out: test-WY {water_year} flight dates = '
              + ', '.join(d.strftime("%Y-%m-%d") for d in sorted(test_wy_flight_dates)))
    else:
        test_wy_flight_dates = set()

    df_sum_total = preprocessing.combine_aso_insitu(obs_data_train,
                         aso_tseries_train['aso_swe'],
                         elev_idx = -1, # total = -1
                        )
    
    drop_na_df,all_pils,baseline_pils = preprocessing.generate_drop_NaNs_table(df_sum_total,
                                                                       obs_data_qa,
                                                                    #    obs_data_train
                                                                       )

    # pillows = [i for i in pillows if i not in exclude_pillows]

    historic_vals_df = preprocessing.create_qa_tables(obs_data_train_lst, [], isQA=False)

    impute_dir = f'{mlrPred_dir}imputation/'
    
    # NOTE (deferred cleanup -- "option b"):
    # All three return values here are discarded; nothing below reads them. What this call
    # actually does is WRITE {impute_dir}/pillow_impute_threePils_wy{water_year}.csv, which
    # lm_model.impute_pillow_prediction() then READS during training. So the writer lives in
    # preprocessing and the reader lives in lm_model, as two near-duplicate implementations
    # of the same triple-nested correlation search, and whichever creates the file first
    # determines its contents.
    #
    # Now that the reader is keyed by OBS_QA_TAG, this writer still emits the UNSUFFIXED
    # file on every run -- work whose output nothing consumes. Dropping the call is provably
    # a no-op on results (returns unused), but it changes which files exist on disk, so it
    # should be done deliberately and only after confirming the two implementations agree.
    # Verify that first, then delete this call and de-duplicate the two functions.
    obs_data_impute,pils_removed,impute_na_df = preprocessing.imputation_w_pillows(df_sum_total,
                                                                               all_pils,
                                                                               obs_data_qa,
                                                                               aso_site_name,
                                                                               water_year,
                                                                               impute_dir,
                                                                               obs_threshold = 0.50,
                                                                               saveImputeCSV = True,
    )

    # obs_data_impute_sm,pils_removed_sm,impute_na_df_sm = preprocessing.imputation_w_snowmodel(
    #                                             df_sum_total,
    #                                             all_pils,
    #                                             obs_data_qa,
    #                                             sm_train_ds,
    #                                             aso_site_name,
    #                                             water_year,
    #                                             impute_dir,
    #                                             obs_threshold = 0.50,
    #                                             saveImputeCSV = True,
    #                                             train_start_year = 2013,
    #                                             predictor_vars=("swed_best", "swed_second", "swed_third"),
    #                                         )
    """
        RUN MODEL -----------------------------------------------
    """
    print('RUNNING MLR MODEL ...\n')

    start = time.time()

    # Cache training results keyed by unique pillow availability sets.
    # Training (cross-validation + station selection) only depends on which
    # pillows are available, not on the day's actual SWE values, so we can
    # reuse training results for all days that share the same pillow set.
    training_model_cache = {}
    ensemble_rows = []

    n_timesteps = obs_data_test_lst[0].time.shape[0]
    print(f'num timesteps: {n_timesteps - 1}')
    print(f'last timestep: {obs_data_test_lst[0].time[-1].values}')

    for t_idx in range(1, n_timesteps):
        current_date_np = obs_data_test_lst[0].time[t_idx].values
        current_date = datetime(pd.to_datetime(current_date_np).year,pd.to_datetime(current_date_np).month,pd.to_datetime(current_date_np).day)

        current_vals_df,all_pils_QA,baseline_pils_,df_qa_table = preprocessing.process_daily_qa(
                                                                t_idx,
                                                                obs_data_test_lst,
                                                                obs_data_test_lst,
                                                                baseline_pils,
                                                                exclude_pillows,
                                                                printOutput = showOutput,
                                                                            )

        # guard: if 0 usable pillows today, MLR cannot train (sklearn errors on
        # shape=(_, 0) feature matrix). Skip the day -- the WY's CSV will simply
        # be missing this row, which downstream code handles as a normal data gap.
        if len(baseline_pils_) == 0 or len(all_pils_QA) == 0:
            print(f'  {current_date}  SKIPPED -- 0 usable pillows today '
                  f'(n_QA={len(all_pils_QA)}, n_baseline={len(baseline_pils_)})')
            continue

        # v5: determine if today is a test-WY flight date that must be excluded
        # from training. In 'wy' mode test_wy_flight_dates is empty so this is None.
        current_date_ts = pd.Timestamp(current_date).normalize()
        excluded_flight = current_date_ts if current_date_ts in test_wy_flight_dates else None

        # Build per-day training inputs. When excluding a flight, filter both the
        # ASO time series and the df_sum_total table to drop that specific date.
        # Do NOT dropna here: run_cross_val_selection2 has its own NaN filter
        # (missing_times) that needs the NaN rows present in df_sum_total to
        # correctly identify and exclude those flight dates from aso_tseries_2.
        if excluded_flight is not None:
            ex_date64 = np.datetime64(excluded_flight.strftime('%Y-%m-%d'))
            aso_tseries_train_day = aso_tseries_train.where(aso_tseries_train.date != ex_date64, drop=True)
            df_sum_total_day = df_sum_total[pd.to_datetime(df_sum_total['time']) != excluded_flight]
            print(f'  {current_date}  v5: excluding flight date {excluded_flight.date()} from training')
        else:
            aso_tseries_train_day = aso_tseries_train
            df_sum_total_day = df_sum_total

        # combine_aso_insitu silently drops flights whose pillow row is too sparse
        # (NaN-heavy) -- so a flight can exist in aso_tseries_train but be absent
        # from df_sum_total. Downstream, run_cross_val_selection2's missing_times
        # filter only inspects df_sum_total, so those "ghost" flights leak through
        # to summarize_data and pull raw NaNs from the pillow time series.
        # Intersect here so only flights matched in df_sum_total survive.
        # In 'wy' mode this is a no-op (combine_aso_insitu was already given the
        # post-train_test_split aso_tseries, so flights and df_sum_total agree).
        df_times_np = pd.to_datetime(df_sum_total_day['time']).values.astype('datetime64[ns]')
        pre_flights = aso_tseries_train_day.date.size
        aso_tseries_train_day = aso_tseries_train_day.where(
            aso_tseries_train_day.date.isin(df_times_np), drop=True)
        dropped_flights = pre_flights - aso_tseries_train_day.date.size
        if dropped_flights > 0:
            print(f'  {current_date}  v5: dropped {dropped_flights} flight(s) from training (not matched in df_sum_total)')

        # Cache key includes the excluded-flight date so each leave-one-flight-out
        # variant gets its own trained model. Most days have excluded_flight=None
        # and share one cache entry; only the ~5 test-WY flight days create new ones.
        cache_key = (frozenset(all_pils_QA), frozenset(baseline_pils_), excluded_flight)
        # v5: imputation cache must differ from v1-v4 cache (different df_sum_total
        # because test-WY flights are now included) AND must differ per excluded
        # flight date (each excluded flight produces a different df_sum_total).
        # In 'wy' mode the suffix is '' so production cache filenames are unchanged.
        # the qa tag keys the cache to the pillow rung being used, so switching rungs no
        # longer silently reuses the previous rung's imputation table.
        if ASO_HOLDOUT_MODE == 'flight':
            ex_tag = excluded_flight.strftime('%Y%m%d') if excluded_flight is not None else 'none'
            impute_cache_suffix_ = f'_{OBS_QA_TAG}_v5_excl{ex_tag}'
        else:
            impute_cache_suffix_ = f'_{OBS_QA_TAG}'
        if cache_key not in training_model_cache:
            print(f'  Training models for new pillow set (n_QA={len(all_pils_QA)}, n_baseline={len(baseline_pils_)})...')
            training_model_cache[cache_key] = lm_model.train_all_mlr_models(
                aso_tseries_train_day.aso_swe, obs_data_qa, aso_site_name, all_pils, all_pils_QA,
                df_sum_total_day, baseline_pils_, start_wy, end_wy, isSplit, isAccum,
                mlrPred_dir, current_date, dem_bin.dem_bin, QA_flag=QA_flag,
                modelNUM=model_num, isMean=False, showOutput=showOutput,
                saveValidation=False, isCombination_=isCombination,
                pillowImputation_=pillowImputation_, ds_snowmodel_=None,
                impute_cache_suffix=impute_cache_suffix_,
                capture_ranked=(N_ENSEMBLE > 1), ranked_top_k=RANKED_TOP_K)

        # mark where this day's ensemble rows start, so they can be annotated with the
        # availability counts below. predict_with_cached_training cannot supply these --
        # all_pils_QA / baseline_pils_ are computed out here in the daily loop.
        _ens_start = len(ensemble_rows)

        try:
            # v4 experiment: force the regression through the origin (yhat=0 when all
            # selected pillows=0). Default in the real-time mlr_prediction.py is True.
            summary_dict_all,df_sheet_lst_mm,df_sheet_lst_acreFt,df_sheet_pillow_lst = lm_model.predict_with_cached_training(
                                                    training_model_cache[cache_key],
                                                    current_vals_df.reset_index(names = 'time'),
                                                    current_date, elev_bin_labels,
                                                    fit_intercept=False,
                                                    n_ensemble=N_ENSEMBLE,
                                                    ensemble_sink=ensemble_rows,
                                                    )

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

        # Availability counts for this day, attached to whichever ensemble rows were just
        # produced. Recorded explicitly because inferring availability from the ensemble
        # output is unreliable: using "distinct pillows across the top-10" as a proxy
        # produced a spurious result -- it flagged Oct 2016 as a low-availability period
        # with 3x the spread, when in fact SWE was ~6mm there and the percentage was just a
        # near-zero denominator. Absolute spread was the smallest of the year.
        for _r in ensemble_rows[_ens_start:]:
            _r['n_QA'] = len(all_pils_QA)
            _r['n_baseline'] = len(baseline_pils_)
            _r['pillows_QA'] = ','.join(sorted(map(str, all_pils_QA)))

                                                
    end = time.time()
    elapsed_min = (end - start) / 60
    print(f"Elapsed time: {elapsed_min:.2f} minutes")

    """
        OUTPUT
    """



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

    if ensemble_rows:
        ens_df = pd.DataFrame(ensemble_rows).sort_values(
            ['date','isImpute','elev_band','rank']).reset_index(drop=True)
        ens_path = f'{dir_path}ensemble_tidy_wy{water_year}.csv'
        ens_df.to_csv(ens_path, index = False)
        n_fail = ens_df['error'].notna().sum() if 'error' in ens_df.columns else 0
        print(f'ENSEMBLE: {len(ens_df)} members written to {ens_path} ({n_fail} failed)')

    print('MLR PREDICTION COMPLETE!!!\n')