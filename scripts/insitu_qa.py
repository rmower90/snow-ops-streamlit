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

sys.path.insert(1, '/home/rossamower/work/aso/snow-ops-streamlit/scripts/')

import metadata as metadata
import qa_voting as qa_voting
import qa_snowmodel as qa_snowmodel 
import qa_static as qa_static
import plotting as plotting




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
    shape_crs = f'EPSG:{cfg["crs"]["epsg"]}'
    return elev_bin_labels, shape_fpath, demBin_fpath, aso_spatial_fpath, aso_tseries_fpath, snowmodel_dir, snodas_dir, insitu_dir, shape_crs, cfg

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
                  


def majority_flags_2of3_wide(
    df_simple: pd.DataFrame,
    methods=("static", "voting", "snowmodel"),
    require_all_methods: bool = False,
) -> pd.DataFrame:
    """
    df_simple (long) -> majority flag (wide) with index=time, columns=pillow, values 0/1.
    Assumes df_simple has columns: time, pillow, method, flag.
    """
    d = df_simple.copy()
    d["time"] = pd.to_datetime(d["time"])
    d["pillow"] = d["pillow"].astype(str).str.strip()
    d["method"] = d["method"].astype(str).str.strip()
    d["flag"] = d["flag"].astype(int)

    d = d[d["method"].isin(methods)]

    wide = d.pivot_table(
        index=["time", "pillow"],
        columns="method",
        values="flag",
        aggfunc="max",
    )

    if require_all_methods:
        # only vote where all methods exist
        wide = wide.dropna(subset=list(methods), how="any")
    else:
        # missing method => 0 vote
        wide = wide.reindex(columns=list(methods)).fillna(0)

    maj = (wide.sum(axis=1) >= 2).astype(int)
    maj_df = maj.unstack("pillow").sort_index()
    return maj_df

def apply_persistence_min_run(maj_df: pd.DataFrame, min_run: int = 3) -> pd.DataFrame:
    """
    Keep flag=1 only for runs of consecutive 1s with length >= min_run.
    Works column-wise (each pillow independently).

    maj_df: index=time (datetime), columns=pillow, values 0/1 (or bool)
    """
    if min_run <= 1:
        return maj_df.astype(int)

    out = maj_df.copy()

    for col in out.columns:
        s = out[col].fillna(0).astype(int)

        # run id increments when value changes
        run_id = (s != s.shift(1, fill_value=0)).cumsum()

        # length of each run
        run_len = s.groupby(run_id).transform("size")

        # keep only 1s that belong to long-enough runs
        kept = ((s == 1) & (run_len >= min_run)).astype(int)
        out[col] = kept

    return out



def apply_majority_to_pillow_dataset(
    ds: xr.Dataset,
    maj_flags: pd.DataFrame,
    keep_only_existing_pillows: bool = True,
) -> xr.Dataset:
    """
    Apply majority flags to a dataset where EACH pillow is a variable with dim 'time'.
    Returns dataset with SAME pillow variable names. Flagged timesteps -> NaN.
    """
    import xarray as xr
    import pandas as pd

    if "time" not in ds:
        # works for both coord or data_var, but your file has it as a coord
        if "time" not in ds.coords and "time" not in ds.dims:
            raise ValueError("Dataset must have a 'time' coordinate/dimension.")

    # maj_flags -> xr (time,pillow)
    maj_da = xr.DataArray(
        maj_flags.sort_index(),
        dims=("time", "pillow"),
        coords={"time": maj_flags.index.values, "pillow": maj_flags.columns.values},
        name="majority_flag",
    )

    out_vars = {}
    ds_vars = list(ds.data_vars)

    pillows = maj_flags.columns
    if keep_only_existing_pillows:
        pillows = [p for p in pillows if p in ds_vars]

    for pil in pillows:
        da = ds[pil]
        if "time" not in da.dims:
            continue

        # IMPORTANT: drop=True removes the scalar 'pillow' coordinate entirely
        m = maj_da.sel(pillow=pil, drop=True).sel(time=da["time"])

        da_out = da.where(m == 0)  # majority flagged -> NaN
        out_vars[pil] = da_out

    out = xr.Dataset(out_vars)
    out.attrs.update(ds.attrs)
    out.attrs["qa_method"] = "majority_2of3"
    return out



def build_majority_qa_ds(
    ds_raw: xr.Dataset,
    df_simple: pd.DataFrame,
    methods=("static", "voting", "snowmodel"),
    require_all_methods: bool = False,
    persistence_days: int = 3,          # <-- NEW (set None/0 to disable)
) -> xr.Dataset:
    """
    Convenience: df_simple -> majority flags -> (optional) persistence filter -> masked dataset.
    Returns dataset with SAME pillow variable names. Flagged timesteps -> NaN.
    """
    # majority flags from df_simple
    maj = majority_flags_2of3_wide(
        df_simple=df_simple,
        methods=methods,
        require_all_methods=require_all_methods,
    )

    # Align maj index to ds_raw time (important if df_simple missing some days)
    t_ds = pd.to_datetime(ds_raw["time"].values).astype("datetime64[ns]")
    t_ds = pd.to_datetime(t_ds).normalize()
    maj.index = pd.to_datetime(maj.index).normalize()

    # reindex to ds_raw time, missing => 0
    maj = maj.reindex(t_ds).fillna(0).astype(int)

    # persistence filter (only keep runs >= persistence_days)
    if persistence_days is not None and int(persistence_days) >= 2:
        maj = apply_persistence_min_run(maj, min_run=int(persistence_days))

    return apply_majority_to_pillow_dataset(ds_raw, maj)



def run_pillow_qa(ds_raw: xr.Dataset,
                  obs_raw_list: List[xr.DataArray],
                  obs_hist_list: List[xr.DataArray],
                  baseline_pils: List[str],
                  pil_corr: Dict[str, List[Tuple[str, float]]],
                  train_ds: Optional[xr.Dataset] = None,
                  test_ds: Optional[xr.Dataset] = None,
                  corr_rank: Optional[Dict[str, List[Tuple[str, float]]]] = None,
                  engineering_buff__: float = 1.5,
                  se_buff__: float = 1.2,
                  snow_melt_gate__: float = 50.0,
                  ):
    
    simple_parts = []
    detail_parts = []

    for t_idx in range(ds_raw.sizes["time"]):
        print(t_idx,end =' ')

        _, _, _, st_simple, st_detail = qa_static.run_old_qa(
            t_idx,
            obs_raw_list,        # WY data (raw pillows)
            obs_hist_list,       # historical QA’d pillows
            user_qa_level=1,     # keep as before
            baseline_pils=baseline_pils,
            pil_corr=pil_corr,
            engineering_buff_=engineering_buff__,
            se_buff_=se_buff__,
            printOutput=False,
            isv1=False,
            snow_present_mm=None,   # <-- NEW (optional but recommended)
            return_long=True,       # <-- REQUIRED for df_simple / df_detail
        )



        _, _, _, vt_simple, vt_detail = qa_voting.run_voting_qa(
            t_idx,
            obs_raw_list,
            obs_hist_list,
            1,
            baseline_pils,
            pil_corr,
            corr_rank=corr_rank,
            engineering_buff=engineering_buff__,
            se_buff=se_buff__,
            snow_present_mm=snow_melt_gate__,   # <-- melt gate ON
            return_long=True,
        )

        _, _, _, sm_simple, sm_detail = qa_snowmodel.run_snowmodel_qa(
                    t_idx,
                    obs_raw_list,
                    obs_hist_list,
                    1,
                    baseline_pils,
                    pil_corr,
                    train_ds=train_ds,
                    test_ds=test_ds,              # ✅ now supported
                    pred_method="seasonal",       # ✅ now supported: "base" | "seasonal" | "detrend"
                    season_period=365.25,
                    engineering_buff=engineering_buff__,
                    se_buff=se_buff__,
                    printOutput=False,
                    min_train=60,
                    snow_present_mm=snow_melt_gate__,         # optional melt gate
                    return_long=True,
        )


        simple_parts += [st_simple, vt_simple, sm_simple]
        detail_parts += [st_detail, vt_detail, sm_detail]

        if (t_idx + 1) % 25 == 0:
            print(f"{t_idx+1}/{ds_raw.sizes['time']}")

    df_simple = pd.concat(simple_parts, ignore_index=True)
    df_detail = pd.concat(detail_parts, ignore_index=True)
    return df_simple, df_detail







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




if __name__ =="__main__":
    aso_site_name = sys.argv[1]
    year_str = sys.argv[2]
    month_str = sys.argv[3]
    day_str = sys.argv[4]


    # one less day for snowmodel output.
    day_str = str(int(day_str) -1)

    # load metadata information.
    elev_bin_labels, shape_fpath, demBin_fpath, aso_spatial_fpath, aso_tseries_fpath, snowmodel_dir, snodas_dir, insitu_dir, shape_crs, cfg = load_aso_metadata(aso_site_name)

    # pillows to exclude from QA.
    exclude_pillows = cfg['pillow_api']['exclude_pillows']

    # load spatial data.
    aso_spatial_ds, aso_demBin_ds, shape_proj_gdf = load_aso_data(aso_spatial_fpath,
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

    # testing.
    obs_data_test_ds = xr.load_dataset(f'{insitu_dir}raw/{aso_site_name}_insitu_obs_daily_wy_2026.nc')
    # match times.
    obs_data_test_ds = obs_data_test_ds.sel(time = sm_test_ds.time)
    obs_data_test_lst = dataset_to_list(obs_data_test_ds)

    pillows = list(obs_data_test_ds.data_vars)
    pillows = [i for i in pillows if i not in exclude_pillows]
    print(f'RUNNING PILLOW QA FOR: {aso_site_name}; up to {str(obs_data_test_ds.time.values[-1])[0:10]}')
    print('')

    # pillows = [i for i in pillows if i not in exclude_pillows]

    historic_vals_df = qa_voting.create_qa_tables(obs_data_train_lst, [], isQA=False)
    
    corr_rank = qa_voting.build_corr_rank(
        historic_vals_df,
        use_delta=True,
        min_overlap=200,
        method="r2",
    )
    pil_corr = {p: corr_rank.get(p, []) for p in pillows}

    df_simple,df_detail = run_pillow_qa(
        obs_data_test_ds,
        obs_data_test_lst,
        obs_data_train_lst,
        pillows,
        pil_corr,
        train_ds=sm_train_ds,
        test_ds=sm_test_ds,
        corr_rank=corr_rank,
    )

    # visualize QA results for ALL pillows.
    for pil in obs_data_test_ds.data_vars:

        out = plotting.plot_pillow_qa_timeline(
        ds_raw=obs_data_test_ds,
        pillow=pil,
        df_simple=df_simple,
        ds_qa=None,
        start="2025-10-01",
        end=None,
        saveFIG = True,
        figDIR = f'{insitu_dir}qa/qa_viz_2026/'
    )
    for pil in obs_data_test_ds.data_vars:
    # static
        __ = plotting.plot_qa_method_diagnostics(
        ds_raw=obs_data_test_ds,
        df_detail=df_detail,
        pillow=pil,
        ds_qa=None,               # or None
        qa_method="static",
        pil_corr=pil_corr,
        top_k=5,
        start="2025-10-01",
        end=None,
        majority_df_simple=df_simple,   # optional
        saveFIG = True,
        figDIR = f'{insitu_dir}qa/qa_method_diagnostics_2026/'
        )

        # voting
        __ = plotting.plot_qa_method_diagnostics(
        ds_raw=obs_data_test_ds,
        df_detail=df_detail,
        pillow=pil,
        ds_qa=None, 
        qa_method="voting",
        start="2025-10-01",
        end=None,
        majority_df_simple=df_simple,
        saveFIG = True,
        figDIR = f'{insitu_dir}qa/qa_method_diagnostics_2026/'
        )

        # snowmodel (+ show best/second/third from test_ds)
        __ = plotting.plot_qa_method_diagnostics(
        ds_raw=obs_data_test_ds,
        df_detail=df_detail,
        pillow=pil,
        ds_qa=None,
        qa_method="snowmodel",
        pred_ds=sm_test_ds,               # <<< optional; draws swed_best/second/third
        start="2025-10-01",
        end=None,
        majority_df_simple=df_simple,
        saveFIG = True,
        figDIR = f'{insitu_dir}qa/qa_method_diagnostics_2026/'
        )

    df_simple.to_csv(f'{insitu_dir}qa/insitu_qa_simple_wy_2026.csv',index = False)
    df_detail.to_csv(f'{insitu_dir}qa/insitu_qa_detail_wy_2026.csv',index = False)

    # create updated majority QA dataset.
    ds_majority = build_majority_qa_ds(
        ds_raw=obs_data_test_ds,
        df_simple=df_simple,
        methods=("static","voting","snowmodel"),
        require_all_methods=False,
        persistence_days=3,   # <-- only mask if majority persists ≥ 3 consecutive days
    )



    # Example save
    ds_majority.to_netcdf(f'{insitu_dir}processed/{aso_site_name}_insitu_obs_daily_wy_2026.nc')

    print('INSITU QA COMPLETE.')





    







