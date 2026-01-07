# steps.py
import xarray as xr
import pandas as pd
import numpy as np
import sys 
import os 
import copy


def dataset_to_list(ds: xr.Dataset) -> list:
    da_list = []
    for pil in ds.data_vars:
        da = ds[pil]
        da.name = pil
        da_list.append(da)
    return da_list

def train_test_split(aso_spatial_data: xr.Dataset,
                     aso_tseries_data: xr.Dataset,
                     insitu_data: list,
                     test_water_year: int,
                    ):
    # test dataset aso
    try:
        aso_spatial_test = aso_spatial_data.where(aso_spatial_data.date >= np.datetime64(f'{test_water_year-1}-10-01'),drop = True).where(aso_spatial_data.date < np.datetime64(f'{test_water_year}-10-01'),drop = True)
        aso_tseries_test = aso_tseries_data.where(aso_tseries_data.date >= np.datetime64(f'{test_water_year-1}-10-01'),drop = True).where(aso_tseries_data.date < np.datetime64(f'{test_water_year}-10-01'),drop = True)
    except:
        aso_spatial_test = None
        aso_tseries_test = None

    # train dataset aso
    try:
        aso_spatial_train = aso_spatial_data.where(~aso_spatial_data.date.isin(aso_spatial_test.date.values),drop = True)
        aso_tseries_train = aso_tseries_data.where(~aso_tseries_data.date.isin(aso_tseries_test.date.values),drop = True)
    except:
        aso_spatial_train = aso_spatial_data
        aso_tseries_train = aso_tseries_data

    insitu_train = []
    insitu_test = []
    insitu_qa = []
    for pil in insitu_data:
        da_train = pil.where(pil.time < np.datetime64(f'{test_water_year-1}-10-01'),drop = True)
        da_qa = pil.where(pil.time >= np.datetime64(f'2013-01-01'),drop = True).where(pil.time < np.datetime64(f'{test_water_year-1}-10-01'),drop = True)

        insitu_train.append(da_train)
        insitu_qa.append(da_qa)
    
    return aso_spatial_test,aso_tseries_test,aso_spatial_train,aso_tseries_train,insitu_train,insitu_qa

def combine_aso_insitu(insitu_data: list,
                       aso_tseries: xr.DataArray,
                       elev_idx: int = -1):
    """
    Combines ASO Mean SWE with Insitu Observations.
    Input:
      insitu_data - list of insitu dataarrays.
      aso_tseries - mean ASO SWE across elevation bins.
      elev_idx - integer representing elevation bin from aso_tseries.
    Output:
      shape_crs - a python string of shape crs.
    """

    aso_tseries.name = 'aso_mean_bins_mm'
    tabASO = aso_tseries.to_dataframe().reset_index()
    elev_bin = str(aso_tseries.elev[elev_idx].values)
    
    tabASO_n = tabASO[tabASO['elev'] == elev_bin].reset_index(drop = True)
    tabASO_n = tabASO_n.drop(columns = ['elev'])
    tabASO_n['date'] = pd.to_datetime(tabASO_n['date'])
    tabASO_n = tabASO_n.set_index('date')
    


    for date in range(0,len(insitu_data)):
        tabINS = insitu_data[date][insitu_data[date].time.isin(aso_tseries.date.values)].to_dataframe()
        if date == 0:
            comp_df = tabASO_n.merge(tabINS,left_on='date',right_on='time')
            comp_df['time'] = tabINS.index
            comp_df = comp_df.set_index('time')
        else:
            comp_df = comp_df.merge(tabINS,left_index = True, right_index = True)

    # set any negative values to zero
    comp_df[comp_df < 0] = 0.0
    
    return comp_df.reset_index()

def generate_drop_NaNs_table(df_summary_table,obs_data = None):
    """
        Selects a baseline number of features to start to statistical models based
        on each having at least one valid flight per year.
        Input:
            df_summary_table - pandas dataframe of ASO mean SWE and Insitu Obs.
            obs_data - list of insitu dataarrays.
        Output:
            preds - list of predictions for each cross-validated year.
    """
    ## convert time to datetime object.
    df_summary_table['time'] = pd.to_datetime(df_summary_table['time'])
    ## check to see if there is mismatching information with full QA'd dataset.
    if obs_data is not None:
        for row,col in df_summary_table.iterrows():
          time = col['time']
          for pil_idx in range(0,len(obs_data)):
            pil = obs_data[pil_idx].name 
            # obs value (safe)
            val_obs_arr = obs_data[pil_idx].sel({"time": time}).values
            val_obs = float(val_obs_arr) if getattr(val_obs_arr, "ndim", 0) == 0 else float(val_obs_arr.squeeze())

            # df value (safe)
            vals = df_summary_table.loc[df_summary_table["time"] == time, pil].values

            if vals.size == 0:
                val_df = np.nan
            elif vals.size == 1:
                val_df = float(vals[0])
            else:
                # If duplicates exist for a time stamp, take the first (we also warn later)
                val_df = float(vals[0])

            # Compare (treat NaNs as equal)
            if not ((np.isnan(val_obs) and np.isnan(val_df)) or (val_obs == val_df)):

            # val_obs = float(obs_data[pil_idx].sel({'time':time}))
            # val_df = float(df_summary_table[df_summary_table['time'] == time][pil].values)
            # if val_obs != val_df:
                if (np.isnan(val_obs)) and (np.isnan(val_df)):
                  pass
                else:
                    print(pil,time, 'qa data', val_obs, 'table data', val_df)
                    df_summary_table.loc[row,pil] = val_obs
                    # obs_data[pil_idx]
    ## notify user if there are duplicate dates
    if df_summary_table.time.nunique() != len(df_summary_table):
        print('DUPLICATE ROWS CONSIDER')
        sys.exit(0)

    ## create list of pillows.
    all_pils = df_summary_table.columns.to_list()
    all_pils.remove('time')
    all_pils.remove('aso_mean_bins_mm')
    ## group by year and sum to identify pillows without values in year.
    df_year = df_summary_table.groupby(df_summary_table.time.dt.year)[all_pils].sum()
    ## create list of pillows with at least one flight per year.
    pillow_w_flight_per_year = df_year.replace(0, np.nan).dropna(axis = 1,how = 'any').columns.to_list()
    pillows_cols = copy.deepcopy(pillow_w_flight_per_year)
    pillows_cols.append('time')
    slice_df = df_summary_table[pillows_cols]
    valid_time = slice_df.dropna(axis = 0, how = 'any').time.values
    slice_df = df_summary_table[df_summary_table['time'].isin(valid_time)].dropna(axis = 1, how = 'any')
    ## create baseline pillow list.
    baseline_pils = slice_df.columns.to_list()
    baseline_pils.remove('time')
    baseline_pils.remove('aso_mean_bins_mm')
    return slice_df,all_pils,baseline_pils

def create_qa_tables(obs_data, missing_stations, isQA=True):
    """
    Same as your existing helper (copied in for drop-in convenience).
    """
    count = 0
    for i in range(0, len(obs_data)):
        df = obs_data[i].to_dataframe().reset_index()
        df["time"] = df["time"].dt.date
        df = df.set_index("time")
        if count == 0:
            all_df = copy.deepcopy(df)
        else:
            all_df = pd.merge(all_df, df, how="left", on="time")
        count += 1

    all_df = all_df.reset_index()
    all_df["time"] = pd.to_datetime(all_df["time"])
    all_df = all_df.set_index("time")

    if isQA:
        qa_df = copy.deepcopy(all_df)
        for col in all_df.columns:
            qa_df[col] = 0
        for pil in missing_stations:
            qa_df[pil] = 0
            all_df[pil] = np.nan
        return all_df, qa_df
    else:
        return all_df
    

# def imputation_w_pillows(df_sum_total: pd.DataFrame,
#                          all_pils: list,
#                          obs_data_5: list,
#                          aso_site_name: str,
#                          water_year: int,
#                          impute_dir: str,
#                          obs_threshold = 0.50,
#                          saveImputeCSV = True
#                          ):
#     """
#         Fit a linear regression model
#         Input:
#             df_sum_total - summary pandas dataframe with features and labels.
#             all_pils - list of all pillow ids.
#             obs_data_5 - list of xarray datarrays for pillow observations.
#             obs_threshold - float of fraction of missing observations for pillow removal.
#             aso_site_name - python string of aso_site_name.
#         Output:
#             obs_data_5_ - new list of xarray datarrays for pillow observations with filled values.
#             pils_removed - updated list of features.
#             df_summary_impute - updated summary datatable.
#     """

#     ## find pillows with observations more than threshold.
#     drop_bool = ((df_sum_total[all_pils].isnull().sum(axis=0) / len(df_sum_total)) < obs_threshold).values
#     ## array of new pillows with observations more than threshold
#     pils_removed = np.array(all_pils)[drop_bool]
#     ## subset summary table to remove pillows with minimal observations.
#     df_dropped_pils = df_sum_total[pils_removed]
#     df_dropped_pils = df_dropped_pils.copy()
#     # ## create output fpath name.
#     # impute_df_fpath = f'./data/summary_table/{aso_site_name}/test/pillow_impute_MLR.csv'

#     ## create output fpath name.
#     if not os.path.exists(impute_dir):
#         os.makedirs(impute_dir)
        
#     impute_df_fpath = f'{impute_dir}/pillow_impute_threePils_wy{water_year}.csv'

#     # load impute table.
#     if os.path.exists(impute_df_fpath):
#         df_summary_impute = pd.read_csv(impute_df_fpath)
#     df_summary_impute['time'] = pd.to_datetime(df_summary_impute['time'])
#     # set negative values to zero.
#     df_summary_impute = df_summary_impute.set_index('time')
#     df_summary_impute[df_summary_impute < 0] = 0.0
#     df_summary_impute = df_summary_impute.reset_index()

#     ## create copy of observations.
#     obs_data_5_ = copy.deepcopy(obs_data_5)
#     ## fill observation data arrays with imputed values.
#     for pil_id in pils_removed:
#         ## create dataframe with time and pillow.
#         df_impute = df_summary_impute[['time',pil_id]]
#         ## identify pillow index.
#         pil_idx = [i for i in range(0,len(obs_data_5_)) if obs_data_5_[i].name == pil_id][0]
#         for row,col in df_impute.iterrows():
#             time = col['time']
#             val = col[pil_id]
#             ## if values are same pass, else update value
#             if (obs_data_5_[pil_idx].sel({'time':time}) == val).values == True:
#                 pass
#             else:
#                 obs_data_5_[pil_idx].loc[dict(time=time)] = val
#     return obs_data_5_,list(pils_removed),df_summary_impute


# def imputation_w_snowmodel(
#     df_sum_total: pd.DataFrame,
#     all_pils: list,
#     obs_data_5: list,
#     ds_swed: "xr.Dataset",         # expects swed_best/swed_second/swed_third with dims (time, pil)
#     aso_site_name: str,
#     water_year: int,
#     impute_dir: str,
#     obs_threshold: float = 0.50,
#     saveImputeCSV: bool = True,
#     train_start_year: int = 2013,
#     predictor_vars=("swed_best", "swed_second", "swed_third"),
# ):
#     """
#     Fit a linear regression model per pillow:
#         target = pillow SWE obs
#         predictors = 3 SnowModel grid cell SWE series (best/second/third) for that pillow

#     Input:
#         df_sum_total  - summary pandas dataframe with features and labels.
#                         Must contain columns for pillows in all_pils and at least 'time' and 'aso_mean_bins_mm'.
#         all_pils      - list of pillow ids.
#         obs_data_5    - list of xarray DataArrays for pillow observations (each .name is a pillow id).
#         ds_swed       - xarray Dataset with SnowModel SWE series at 3 grid cells per pillow:
#                         variables: swed_best/swed_second/swed_third
#                         dims: (time, pil)
#         obs_threshold - fraction missing threshold for pillow removal.
#         aso_site_name - python string of aso_site_name. (kept for symmetry)
#         water_year    - int water year
#         impute_dir    - output directory
#         saveImputeCSV - whether to write/read cached imputation CSV

#     Output:
#         obs_data_5_        - new list of xarray datarrays for pillow observations with filled values.
#         pils_removed(list) - pillows retained for imputation (same semantics as your original)
#         df_summary_impute  - updated summary table with imputed values
#     """
#     # Lazy import to keep signature "drop-in" if you paste into the same module.
#     import xarray as xr  # noqa: F401

#     # --- 1) Identify pillows with enough observations to attempt imputation ---
#     drop_bool = (
#         (df_sum_total[all_pils].isnull().sum(axis=0) / len(df_sum_total)) < obs_threshold
#     ).values
#     pils_removed = np.array(all_pils)[drop_bool].tolist()

#     impute_df_fpath = f"{impute_dir}/pillow_impute_threeSnowModelGrids_wy{water_year}.csv"


#     # --- 3) Load imputation table (cached or newly written) ---
#     if os.path.exists(impute_df_fpath):
#         df_summary_impute = pd.read_csv(impute_df_fpath)

#     df_summary_impute["time"] = pd.to_datetime(df_summary_impute["time"])
#     df_summary_impute = df_summary_impute.set_index("time")
#     df_summary_impute[df_summary_impute < 0] = 0.0
#     df_summary_impute = df_summary_impute.reset_index()

#     # --- 4) Fill obs_data_5 with imputed values (same behavior as your original) ---
#     obs_data_5_ = copy.deepcopy(obs_data_5)

#     for pil_id in pils_removed:
#         if pil_id not in df_summary_impute.columns:
#             continue

#         df_impute = df_summary_impute[["time", pil_id]]

#         # identify pillow index in obs_data_5_
#         matches = [i for i in range(len(obs_data_5_)) if obs_data_5_[i].name == pil_id]
#         if not matches:
#             continue
#         pil_idx = matches[0]

#         for _, row in df_impute.iterrows():
#             t = row["time"]
#             v = row[pil_id]

#             # only update if different (keeps your original semantics)
#             try:
#                 current = obs_data_5_[pil_idx].sel(time=t).values
#                 if np.asarray(current == v).item():
#                     continue
#             except Exception:
#                 # if selection fails for any reason, just try to assign
#                 pass

#             obs_data_5_[pil_idx].loc[dict(time=t)] = v

#     return obs_data_5_, list(pils_removed), df_summary_impute