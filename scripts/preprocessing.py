# preprocessing.py
import xarray as xr 
import numpy as np 
import geopandas as gpd 
import zarr
import pandas as pd 
import os
import copy
from typing import List, Dict, Tuple, Optional
from sklearn import linear_model    

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

def create_qa_tables_cp(obs_data, missing_stations, isQA=True):
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
            qa_df[pil] = 1
        return qa_df
    else:
        return all_df


def imputation_w_pillows(df_sum_total: pd.DataFrame,
                         all_pils: list,
                         obs_data_5: list,
                         aso_site_name: str,
                         water_year: int,
                         impute_dir: str,
                         obs_threshold = 0.50,
                         saveImputeCSV = True
                         ):
    """
        Fit a linear regression model
        Input:
            df_sum_total - summary pandas dataframe with features and labels.
            all_pils - list of all pillow ids.
            obs_data_5 - list of xarray datarrays for pillow observations.
            obs_threshold - float of fraction of missing observations for pillow removal.
            aso_site_name - python string of aso_site_name.
        Output:
            obs_data_5_ - new list of xarray datarrays for pillow observations with filled values.
            pils_removed - updated list of features.
            df_summary_impute - updated summary datatable.
    """

    ## find pillows with observations more than threshold.
    drop_bool = ((df_sum_total[all_pils].isnull().sum(axis=0) / len(df_sum_total)) < obs_threshold).values
    ## array of new pillows with observations more than threshold
    pils_removed = np.array(all_pils)[drop_bool]
    ## subset summary table to remove pillows with minimal observations.
    df_dropped_pils = df_sum_total[pils_removed]
    df_dropped_pils = df_dropped_pils.copy()
    # ## create output fpath name.
    # impute_df_fpath = f'./data/summary_table/{aso_site_name}/test/pillow_impute_MLR.csv'

    ## create output fpath name.
    if not os.path.exists(impute_dir):
        os.makedirs(impute_dir)
        
    impute_df_fpath = f'{impute_dir}/pillow_impute_threePils_wy{water_year}.csv'

    ## if table does not exist.
    if not os.path.exists(impute_df_fpath) or (saveImputeCSV == False):

        for pil in df_dropped_pils.columns:
            df_new = pd.DataFrame(df_dropped_pils[pil])
# iterate over observations.
            for row,col in df_new.iterrows():
                val = col[pil]
                # find nans
                if np.isnan(val):
                    valid_pillows = df_dropped_pils.iloc[row][~df_dropped_pils.iloc[row].isna()].index.tolist()
                    target_id = int([i for i in range(0,len(obs_data_5)) if obs_data_5[i].name == pil][0])
                    target_da = obs_data_5[target_id]

                    first_corr = {}
                    feature_list = []
                    for feat in valid_pillows:
                        feat_id = int([i for i in range(0,len(obs_data_5)) if obs_data_5[i].name == feat][0])
                        feat_da = obs_data_5[feat_id]
                        mask_target = ~target_da.isnull()
                        mask_feat = ~feat_da.isnull()
                        target_vals = target_da.where(mask_target).where(mask_feat).values
                        feat_vals = feat_da.where(mask_target).where(mask_feat).values

                        target_vals = target_vals[~np.isnan(target_vals)]
                        feat_vals = feat_vals[~np.isnan(feat_vals)]
                        r2 = np.corrcoef(target_vals, feat_vals)[0,1]**2
                        k = 1
                        n = len(feat_vals)
                        adjr2= 1 - ((1-r2) * (n-1)/(n-k-1))

                        first_corr[feat] = adjr2

                    # find max correlated pillow.
                    best_corr_pillow = max(first_corr, key=first_corr.get)
                    best_corr_adjr2 = first_corr[best_corr_pillow]
                    feature_list.append(best_corr_pillow)
            
            
                    bestfeat_id = int([i for i in range(0,len(obs_data_5)) if obs_data_5[i].name == best_corr_pillow][0])
                    # remove pillow from list.
                    valid_pillows.remove(best_corr_pillow)
                    blah = True
                    train_df = pd.merge(target_da.to_dataframe(),obs_data_5[bestfeat_id].to_dataframe(),on = 'time')
                    second_corr = {}
                    for feat in valid_pillows:
                        feat_id = int([i for i in range(0,len(obs_data_5)) if obs_data_5[i].name == feat][0])
                        train_df_ = pd.merge(train_df,obs_data_5[feat_id].to_dataframe(),on = 'time')
                        train_df_ = train_df_[train_df_.index.year >= 2013]
                        train_df_ = train_df_.dropna(axis = 0)
                        feat_cols = train_df_.columns.to_list()
                        feat_cols.remove(pil)
                        target_vals = train_df_[pil].values
                        feat_vals = train_df_[feat_cols].values
                        r2 = np.corrcoef(target_vals, feat_vals.T)[0,1]**2
                        k = len(feat_cols)
                        n = len(feat_vals)
                        adjr2= 1 - ((1-r2) * (n-1)/(n-k-1))

                        second_corr[feat] = adjr2

                    # find max correlated pillow.
                    second_corr_pillow = max(second_corr, key=second_corr.get)
                    second_corr_adjr2 = second_corr[second_corr_pillow]
                    if second_corr_adjr2 > best_corr_adjr2:
                        feature_list.append(second_corr_pillow)

                        second_id = int([i for i in range(0,len(obs_data_5)) if obs_data_5[i].name == second_corr_pillow][0])
                        # remove pillow from list.
                        valid_pillows.remove(second_corr_pillow)
                        blah = True
                        train_df = pd.merge(train_df,obs_data_5[second_id].to_dataframe(),on = 'time')
                        third_corr = {}
                        for feat in valid_pillows:
                            feat_id = int([i for i in range(0,len(obs_data_5)) if obs_data_5[i].name == feat][0])
                            train_df_ = pd.merge(train_df,obs_data_5[feat_id].to_dataframe(),on = 'time')
                            train_df_ = train_df_[train_df_.index.year >= 2013]
                            train_df_ = train_df_.dropna(axis = 0)
                            feat_cols = train_df_.columns.to_list()
                            feat_cols.remove(pil)
                            target_vals = train_df_[pil].values
                            feat_vals = train_df_[feat_cols].values
                            r2 = np.corrcoef(target_vals, feat_vals.T)[0,1]**2
                            k = len(feat_cols)
                            n = len(feat_vals)
                            adjr2= 1 - ((1-r2) * (n-1)/(n-k-1))
                    
                            third_corr[feat] = adjr2

                        # find max correlated pillow.
                        try:
                            third_corr_pillow = max(third_corr, key=third_corr.get)
                            third_corr_adjr2 = third_corr[third_corr_pillow]
                            if third_corr_adjr2 > second_corr_adjr2:
                                feature_list.append(third_corr_pillow)
                        except:
                            pass

                    # train regression
                    final_train_df_ = obs_data_5[target_id].to_dataframe()
                    for feat in feature_list:
                        feat_id = int([i for i in range(0,len(obs_data_5)) if obs_data_5[i].name == feat][0])
                        final_train_df_ = pd.merge(final_train_df_,obs_data_5[feat_id].to_dataframe(),on = 'time')

                    final_train_df = final_train_df_[final_train_df_.index.year >= 2013]
                    final_train_df = final_train_df.dropna(axis = 0)

                    lm = linear_model.LinearRegression()
                    lm.fit(final_train_df.values[:,1:],final_train_df.values[:,0])

                    res = lm.predict(df_dropped_pils.iloc[row][feature_list].values.reshape(1,-1))
                    if res < 0:
                        res = 0
                    df_dropped_pils.loc[row, pil] = res




        df_summary_impute = copy.deepcopy(df_dropped_pils)
        pillows_ = df_dropped_pils.columns.to_list()


        df_summary_impute['time'] = df_sum_total.time.values
        df_summary_impute['aso_mean_bins_mm'] = df_sum_total.aso_mean_bins_mm.values
        df_summary_impute = df_summary_impute[['time','aso_mean_bins_mm'] + pillows_]

        # ouput table.
        if saveImputeCSV:
            df_summary_impute.to_csv(impute_df_fpath,index = False)

    # load impute table.
    if os.path.exists(impute_df_fpath):
        df_summary_impute = pd.read_csv(impute_df_fpath)
    df_summary_impute['time'] = pd.to_datetime(df_summary_impute['time'])
    # set negative values to zero.
    df_summary_impute = df_summary_impute.set_index('time')
    df_summary_impute[df_summary_impute < 0] = 0.0
    df_summary_impute = df_summary_impute.reset_index()

    ## create copy of observations.
    obs_data_5_ = copy.deepcopy(obs_data_5)
    ## fill observation data arrays with imputed values.
    for pil_id in pils_removed:
        ## create dataframe with time and pillow.
        df_impute = df_summary_impute[['time',pil_id]]
        ## identify pillow index.
        pil_idx = [i for i in range(0,len(obs_data_5_)) if obs_data_5_[i].name == pil_id][0]
        for row,col in df_impute.iterrows():
            time = col['time']
            val = col[pil_id]
            ## if values are same pass, else update value
            if (obs_data_5_[pil_idx].sel({'time':time}) == val).values == True:
                pass
            else:
                obs_data_5_[pil_idx].loc[dict(time=time)] = val
    return obs_data_5_,list(pils_removed),df_summary_impute


def process_daily_qa(
                     t_idx: int, 
                     obs_data_raw: list,
                     obs_data_qa: list,
                     baseline_pils: list,
                     printOutput: bool = False,
                    ):
    """
    """ 
    # all pils
    all_pils = [i.name for i in obs_data_raw]
    # current day raw values.
    obs_data_current_day = [i.isel({'time':t_idx}) for i in obs_data_raw]
    # current day qa values.
    obs_data_qa_current_day = [i.isel({'time':t_idx}) for i in obs_data_qa]
    # missing stations.
    missing_stations = [i.name for i in obs_data_current_day if i.isnull()]
    # mission stations and failing qa.
    nan_stations = [i.name for i in obs_data_qa_current_day if i.isnull()]
    # stations failing qa.
    qa_stations = [i for i in nan_stations if i not in missing_stations]
    # create raw dataframe.
    obs_data_raw_df = create_qa_tables(obs_data_raw,missing_stations = [],isQA=False)
    # create qa dataframe.
    obs_data_qa_df = create_qa_tables_cp(obs_data_qa,missing_stations = qa_stations,isQA=True)
    # current day raw dataframe.
    obs_day_raw_df = pd.DataFrame(obs_data_raw_df.iloc[t_idx]).T
    # current day raw dataframe.
    obs_day_qa_df = pd.DataFrame(obs_data_qa_df.iloc[t_idx]).T


    # all_pils_QA
    all_pils_QA = [i for i in all_pils if i not in nan_stations]    
    baseline_pils_ = [i for i in baseline_pils if i not in nan_stations]
    
    return obs_day_raw_df,all_pils_QA,baseline_pils_,obs_day_qa_df




def imputation_w_snowmodel(
    df_sum_total: pd.DataFrame,
    all_pils: list,
    obs_data_5: list,
    ds_swed: "xr.Dataset",         # expects swed_best/swed_second/swed_third with dims (time, pil)
    aso_site_name: str,
    water_year: int,
    impute_dir: str,
    obs_threshold: float = 0.50,
    saveImputeCSV: bool = True,
    train_start_year: int = 2013,
    predictor_vars=("swed_best", "swed_second", "swed_third"),
):
    """
    Fit a linear regression model per pillow:
        target = pillow SWE obs
        predictors = 3 SnowModel grid cell SWE series (best/second/third) for that pillow

    Input:
        df_sum_total  - summary pandas dataframe with features and labels.
                        Must contain columns for pillows in all_pils and at least 'time' and 'aso_mean_bins_mm'.
        all_pils      - list of pillow ids.
        obs_data_5    - list of xarray DataArrays for pillow observations (each .name is a pillow id).
        ds_swed       - xarray Dataset with SnowModel SWE series at 3 grid cells per pillow:
                        variables: swed_best/swed_second/swed_third
                        dims: (time, pil)
        obs_threshold - fraction missing threshold for pillow removal.
        aso_site_name - python string of aso_site_name. (kept for symmetry)
        water_year    - int water year
        impute_dir    - output directory
        saveImputeCSV - whether to write/read cached imputation CSV

    Output:
        obs_data_5_        - new list of xarray datarrays for pillow observations with filled values.
        pils_removed(list) - pillows retained for imputation (same semantics as your original)
        df_summary_impute  - updated summary table with imputed values
    """
    # Lazy import to keep signature "drop-in" if you paste into the same module.
    import xarray as xr  # noqa: F401

    # --- 1) Identify pillows with enough observations to attempt imputation ---
    drop_bool = (
        (df_sum_total[all_pils].isnull().sum(axis=0) / len(df_sum_total)) < obs_threshold
    ).values
    pils_removed = np.array(all_pils)[drop_bool].tolist()

    # Subset summary table to only pillows retained
    df_dropped_pils = df_sum_total[pils_removed].copy()

    # Output CSV path
    os.makedirs(impute_dir, exist_ok=True)
    impute_df_fpath = f"{impute_dir}/pillow_impute_threeSnowModelGrids_wy{water_year}.csv"

    # --- 2) Build imputation table (or load it) ---
    need_build = (not os.path.exists(impute_df_fpath)) or (saveImputeCSV is False)

    if need_build:
        # Ensure ds has required predictors
        missing_vars = [v for v in predictor_vars if v not in ds_swed.data_vars]
        if missing_vars:
            raise ValueError(
                f"ds_swed is missing required variables: {missing_vars}. "
                f"Expected at least {predictor_vars}."
            )

        # Make sure ds_swed has the expected coords
        if "pil" not in ds_swed.coords or "time" not in ds_swed.coords:
            raise ValueError("ds_swed must have coords: 'pil' and 'time'.")

        # We'll impute each pil independently with its own 3-grid predictors
        for pil in pils_removed:
            # Skip if pillow not present in ds_swed
            if pil not in set(ds_swed["pil"].values.astype(str)):
                continue

            # Target pillow obs series (from obs_data_5 list)
            try:
                target_id = [i for i in range(len(obs_data_5)) if obs_data_5[i].name == pil][0]
            except IndexError:
                # pillow is in df_sum_total but not in obs_data_5 list
                continue

            target_da = obs_data_5[target_id]

            # Build training dataframe: merge target with predictors on time
            # target_da.to_dataframe() -> column name is pil (because DataArray.name == pil)
            df_target = target_da.to_dataframe().reset_index()  # columns: ['time', pil]
            df_pred = (
                ds_swed.sel(pil=pil)[list(predictor_vars)]
                .to_dataframe()
                .reset_index()
            )  # columns: ['time', 'pil', swed_best, swed_second, swed_third]
            # Keep only time + predictors
            keep_cols = ["time"] + list(predictor_vars)
            df_pred = df_pred[keep_cols]

            df_train = pd.merge(df_target, df_pred, on="time", how="inner")

            # Training subset and drop NA
            df_train = df_train[df_train["time"].dt.year >= train_start_year].dropna(axis=0)

            # If not enough data, skip
            if len(df_train) < 20:
                continue

            X = df_train[list(predictor_vars)].values
            y = df_train[pil].values

            # Fit linear regression
            lm = linear_model.LinearRegression()
            lm.fit(X, y)

            # Indices where df_dropped_pils is missing for this pillow
            miss_mask = df_dropped_pils[pil].isna()
            if not miss_mask.any():
                continue

            # Predictor values at those times
            times_missing = df_sum_total.loc[miss_mask, "time"].values
            df_pred_missing = (
                ds_swed.sel(pil=pil)
                .sel(time=times_missing)
                [list(predictor_vars)]
                .to_dataframe()
                .reset_index()
            )
            # Align by time back to the row order in df_sum_total[miss_mask]
            df_pred_missing = df_pred_missing.set_index("time").reindex(pd.to_datetime(times_missing))

            # Only predict where predictors are all present
            ok = ~df_pred_missing[list(predictor_vars)].isna().any(axis=1)
            if not ok.any():
                continue

            Xmiss = df_pred_missing.loc[ok, list(predictor_vars)].values
            pred = lm.predict(Xmiss)
            pred = np.clip(pred, 0, None)  # no negatives

            # Fill into df_dropped_pils using integer positions of missing rows
            # miss_mask is aligned to df_dropped_pils index (same as df_sum_total)
            miss_idx = df_dropped_pils.index[miss_mask]
            # ok is aligned to times_missing order; map ok True positions -> corresponding miss_idx
            ok_idx = miss_idx[np.where(ok.values)[0]]

            df_dropped_pils.loc[ok_idx, pil] = pred

        # Final imputed summary dataframe with same shape pattern as your original
        df_summary_impute = copy.deepcopy(df_dropped_pils)
        pillows_ = df_dropped_pils.columns.to_list()

        df_summary_impute["time"] = df_sum_total["time"].values
        df_summary_impute["aso_mean_bins_mm"] = df_sum_total["aso_mean_bins_mm"].values
        df_summary_impute = df_summary_impute[["time", "aso_mean_bins_mm"] + pillows_]

        if saveImputeCSV:
            df_summary_impute.to_csv(impute_df_fpath, index=False)

    # --- 3) Load imputation table (cached or newly written) ---
    if os.path.exists(impute_df_fpath):
        df_summary_impute = pd.read_csv(impute_df_fpath)

    df_summary_impute["time"] = pd.to_datetime(df_summary_impute["time"])
    df_summary_impute = df_summary_impute.set_index("time")
    df_summary_impute[df_summary_impute < 0] = 0.0
    df_summary_impute = df_summary_impute.reset_index()

    # --- 4) Fill obs_data_5 with imputed values (same behavior as your original) ---
    obs_data_5_ = copy.deepcopy(obs_data_5)

    for pil_id in pils_removed:
        if pil_id not in df_summary_impute.columns:
            continue

        df_impute = df_summary_impute[["time", pil_id]]

        # identify pillow index in obs_data_5_
        matches = [i for i in range(len(obs_data_5_)) if obs_data_5_[i].name == pil_id]
        if not matches:
            continue
        pil_idx = matches[0]

        for _, row in df_impute.iterrows():
            t = row["time"]
            v = row[pil_id]

            # only update if different (keeps your original semantics)
            try:
                current = obs_data_5_[pil_idx].sel(time=t).values
                if np.asarray(current == v).item():
                    continue
            except Exception:
                # if selection fails for any reason, just try to assign
                pass

            obs_data_5_[pil_idx].loc[dict(time=t)] = v

    return obs_data_5_, list(pils_removed), df_summary_impute


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
            val_obs = float(obs_data[pil_idx].sel({'time':time}))
            val_df = float(df_summary_table[df_summary_table['time'] == time][pil].values)
            if val_obs != val_df:
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