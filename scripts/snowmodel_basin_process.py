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
import shutil

sys.path.insert(1, '/home/rossamower/bin/scripts/')

import metadata as metadata





def build_sm_pillow_training_ds_slice_first(
    df_corr: pd.DataFrame,
    ds_hrrr: xr.Dataset,
    obs_data_hist: List[xr.DataArray],
    wy_start: int = 2026,
    # wy_end: int = 2025,
    var_grid: str = "swed",
    resample_daily: bool = True,
    grid_units_to_mm: bool = True,
    interp_grid_time: bool = True,
    out_path: Optional[str] = None,
    out_format: str = "zarr",
) -> xr.Dataset:
    """
    Efficient version: slice relevant grid cells first (time,pil), THEN resample daily.
    Avoids resampling the full 3D grid.
    """

    t0 = np.datetime64(f"{wy_start-1}-10-01")
    # t1 = np.datetime64(f"{wy_end}-10-02")

    # only keep variable + time range
    ds = ds_hrrr[[var_grid]].sel(time=slice(t0, None))

    # Drop conflicting grid lat/lon coords/vars early (safe)
    for nm in ("xlat", "xlon", "lat", "lon"):
        if nm in ds.coords:
            ds = ds.reset_coords(nm, drop=True)
        if nm in ds.variables and nm not in ds.dims:
            ds = ds.drop_vars(nm, errors="ignore")

    def _to_int_or_nan(x):
        try:
            if pd.isna(x):
                return np.nan
            return int(x)
        except Exception:
            return np.nan

    pil_ids = df_corr["pil_id"].astype(str).tolist()

    best_idx   = np.array([_to_int_or_nan(x) for x in df_corr["best_idx"].values], dtype=float)
    best_idy   = np.array([_to_int_or_nan(x) for x in df_corr["best_idy"].values], dtype=float)
    second_idx = np.array([_to_int_or_nan(x) for x in df_corr["second_idx"].values], dtype=float)
    second_idy = np.array([_to_int_or_nan(x) for x in df_corr["second_idy"].values], dtype=float)
    third_idx  = np.array([_to_int_or_nan(x) for x in df_corr["third_idx"].values], dtype=float)
    third_idy  = np.array([_to_int_or_nan(x) for x in df_corr["third_idy"].values], dtype=float)

    good = np.isfinite(best_idx) & np.isfinite(best_idy)
    pil_good = [p for p, g in zip(pil_ids, good) if g]
    if len(pil_good) == 0:
        raise ValueError("No pillows in df_corr have finite best_idx/best_idy.")

    def _isel_points(idx_arr, idy_arr, name: str) -> xr.DataArray:
        idx = idx_arr[good].astype(int)
        idy = idy_arr[good].astype(int)

        da = ds[var_grid].isel(
            east_west=xr.DataArray(idx, dims="pil", coords={"pil": pil_good}),
            south_north=xr.DataArray(idy, dims="pil", coords={"pil": pil_good}),
        ).rename(name)

        # unit conversion after slicing (cheap)
        if grid_units_to_mm:
            da = da * 1000.0
            da.attrs["units"] = "mm"

        return da

    # Slice first (time, pil) — still sub-daily at this point
    swed_best = _isel_points(best_idx, best_idy, "swed_best")

    # For second/third: if missing, use best indices as placeholder then mask to NaN
    swed_second = _isel_points(
        np.where(np.isfinite(second_idx), second_idx, best_idx),
        np.where(np.isfinite(second_idy), second_idy, best_idy),
        "swed_second",
    )
    swed_third = _isel_points(
        np.where(np.isfinite(third_idx), third_idx, best_idx),
        np.where(np.isfinite(third_idy), third_idy, best_idy),
        "swed_third",
    )

    # Mask missing second/third
    sec_missing = ~(np.isfinite(second_idx[good]) & np.isfinite(second_idy[good]))
    thr_missing = ~(np.isfinite(third_idx[good]) & np.isfinite(third_idy[good]))
    if np.any(sec_missing):
        swed_second.loc[dict(pil=np.array(pil_good)[sec_missing])] = np.nan
    if np.any(thr_missing):
        swed_third.loc[dict(pil=np.array(pil_good)[thr_missing])] = np.nan

    # NOW resample daily on small (time,pil) arrays
    if resample_daily:
        swed_best = swed_best.resample(time="1D").mean()
        swed_second = swed_second.resample(time="1D").mean()
        swed_third = swed_third.resample(time="1D").mean()

    # Build obs (time,pil)
    obs_map = {da.name: da for da in obs_data_hist}
    obs_list = []
    for pil in pil_good:
        if pil not in obs_map:
            continue
        da = obs_map[pil].sel(time=slice(t0, None))
        if resample_daily:
            da = da.resample(time="1D").mean()
        da = da.rename("swe_obs").assign_coords(pil=pil).expand_dims("pil")
        obs_list.append(da)

    if len(obs_list) == 0:
        raise ValueError("No matching pillows found in obs_data_hist for pil_good.")

    swe_obs = xr.concat(obs_list, dim="pil").transpose("time", "pil")
    swe_obs.attrs["units"] = "mm"

    out = xr.Dataset(
        {
            "swed_best": swed_best,
            "swed_second": swed_second,
            "swed_third": swed_third,
            "swe_obs": swe_obs,
        }
    )

    # Align to obs time
    out = out.sel(time=swe_obs.time)

    # Interpolate missing grid values along time (now cheap)
    if interp_grid_time:
        for v in ("swed_best", "swed_second", "swed_third"):
            # out[v] = out[v].interpolate_na(dim="time")
            out[v] = out[v].chunk({"time": -1}).interpolate_na(dim="time")


    # Save
    if os.path.exists(out_path): 
        shutil.rmtree(out_path)
    
    if out_path is not None:
        if out_format.lower() == "zarr":
            out.to_zarr(out_path, mode="w")
        elif out_format.lower() in ("nc", "netcdf"):
            out.to_netcdf(out_path)
        else:
            raise ValueError("out_format must be 'zarr' or 'netcdf'.")

    return out

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
    return elev_bin_labels, shape_fpath, demBin_fpath, aso_spatial_fpath, aso_tseries_fpath, snowmodel_dir, snodas_dir, insitu_dir, shape_crs

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
                  
def aggregate_basin_swe_by_elev(
                              aso_swe_da: xr.DataArray,
                              aso_dem_bin_da: xr.DataArray,
                              date_str: str,
                             ) -> tuple[pd.DataFrame,pd.DataFrame]:
    """
    Aggregate ASO SWE by elevation bins.
    Input:
      aso_swe_da - xarray DataArray of ASO SWE [date,y,x]
      aso_dem_bin_da - xarray DataArray of ASO DEM bins [elev,y,x]
      elev_bin_labels - list of elevation bin labels.
    Output:
      xarray DataArray of binned ASO SWE [date,elev]
    """
    m3_acreFt = 0.000810714
    swe_list_m = []
    swe_list_acreFt = []
    swe_list_m.append(f'{date_str[0:4]}-{pad_zero(date_str[4:6])}-{pad_zero(date_str[6:8])}')
    swe_list_acreFt.append(f'{date_str[0:4]}-{pad_zero(date_str[4:6])}-{pad_zero(date_str[6:8])}')

    for elev in range(0,aso_dem_bin_da.elev.shape[0]):
        mask = ~aso_dem_bin_da[elev].isnull()
        num_grids = int(mask.sum())
        area_m2 = num_grids * 50 * 50
        # swe_basin_elev_m = float(aso_swe_da.where(mask).mean(dim=['y','x']).values)
        swe_basin_elev_m = (
                            aso_swe_da
                                .where(mask)
                                .mean(dim=['y','x'])
                                .values
                                .item()
                        )
        swe_basin_elev_acreft = swe_basin_elev_m * area_m2 * m3_acreFt
        
        swe_list_m.append(swe_basin_elev_m)
        swe_list_acreFt.append(swe_basin_elev_acreft)
    
    df_m = pd.DataFrame(data = [swe_list_m],columns = ['Date'] + list(aso_dem_bin_da.elev.values))
    df_acreFt = pd.DataFrame(data = [swe_list_acreFt],columns = ['Date'] + list(aso_dem_bin_da.elev.values))
    return df_m,df_acreFt

def match_sm_aso_spatial(aso_site_name: str,
                    aso_spatial_ds: xr.Dataset,
                    aso_dem: xr.DataArray,
                    shape_crs: str,
                    shape_geog_gdf: gpd.GeoDataFrame,
                    snowmodel_dates: list,
                    snowmodel_model: str = 'snowfall_frac_3',
                    var: str = 'swed',
                    campaign_base_dir: str = '/glade/campaign/ral/hap/rmower/aso/data/',
                    is50m: bool = True,
                    water_year_limit: list = None,
                    saveNC: bool = True,
                    isHRRR: bool = False,
                    ):
    
    """
    Loads SnowModel spatial dataset of flights and matched to aso dates.
    Input:

    Output:
    """
    if isHRRR:
        snowmodel_nc_dir = f'{campaign_base_dir}{aso_site_name}/outputs/wo_assim/hrrr/{snowmodel_model}/'
    else:
        snowmodel_nc_dir = f'{campaign_base_dir}{aso_site_name}/outputs/wo_assim/{snowmodel_model}/'

    snowmodel_dates = sorted([str(i)[0:10].replace('-','') for i in snowmodel_dates])

    if water_year_limit is not None:
        snowmodel_dates = [i for i in snowmodel_dates if i[0:4] in water_year_limit]

        # load tables.
    if not os.path.exists(f'/home/rossamower/work/aso/snowmodel/domains/{aso_site_name}/mean_swe_snowmodel_m_wy2026.csv'):
        mean_swe_df_m = None
        mean_swe_df_acreFt = None
    else:
        mean_swe_df_m = pd.read_csv(f'/home/rossamower/work/aso/snowmodel/domains/{aso_site_name}/mean_swe_snowmodel_m_wy2026.csv')
        mean_swe_df_acreFt = pd.read_csv(f'/home/rossamower/work/aso/snowmodel/domains/{aso_site_name}/mean_swe_snowmodel_acreFt_wy2026.csv')

    # water year netcdf snowmodel output.
    if os.path.exists(f'{snowmodel_nc_dir}wy_2022/netcdf/{var}.nc'):
        # load geography.
        fpath = f'{snowmodel_nc_dir}wy_2022/geography/ctrl_proj.nc'
        sm_geog_proj_ds = xr.load_dataset(fpath)
        sm_ds = match_aso_ds(aso_spatial_ds,sm_geog_proj_ds,var,shape_crs,shape_geog_gdf,snowmodel_nc_dir)
    else:
        # daily netcdf snowmodel output.
        sm_lst = []
        for date in snowmodel_dates:
            yr_str = date[0:4]
            mo_str = date[4:6]
            mo_int = int(mo_str)
            if mo_int >=10:
                wateryr_str = str(int(yr_str) +1)
            else:
                wateryr_str = yr_str
            # try:
            # load snowmodel dataset.
            try:
                fpath = f'{snowmodel_nc_dir}wy_{wateryr_str}/netcdf/sm_{var}_{date}030000.nc'
                sm_ds = xr.load_dataset(fpath)
            except:
                fpath = f'{snowmodel_nc_dir}wy_{wateryr_str}/netcdf/sm_{var}_{date}120000.nc'
                sm_ds = xr.load_dataset(fpath)

            # load geography.
            fpath = f'{snowmodel_nc_dir}wy_{wateryr_str}/geography/ctrl_proj.nc'
            sm_geog_proj_ds = xr.load_dataset(fpath)
            # match with aso.
            sm_match_da = match_aso_da(sm_ds,aso_spatial_ds,sm_geog_proj_ds,date,var,shape_crs,shape_geog_gdf)
            # output matched dataarray.
            if saveNC:
                if not os.path.exists(f'{snowmodel_nc_dir}/wy_{wateryr_str}/aso_basin/'): os.makedirs(f'{snowmodel_nc_dir}/wy_{wateryr_str}/aso_basin/')   
                sm_match_da.to_netcdf(f'{snowmodel_nc_dir}/wy_{wateryr_str}/aso_basin/sm_{var}_{date}.nc')
            # aggregate by elevation bin.
            day_df_m,day_df_acreFt = aggregate_basin_swe_by_elev(sm_match_da,aso_dem,date)
            if mean_swe_df_m is not None:
                mean_swe_df_m = pd.concat([mean_swe_df_m,day_df_m])
                mean_swe_df_acreFt = pd.concat([mean_swe_df_acreFt,day_df_acreFt])
            else:
                mean_swe_df_m = day_df_m
                mean_swe_df_acreFt = day_df_acreFt

            # update checklist
            generate_snowmodel_basin_checklist(date,f'/home/rossamower/work/aso/snowmodel/domains/{aso_site_name}/',aso_site_name)
    
    mean_swe_df_m.to_csv(f'/home/rossamower/work/aso/snowmodel/domains/{aso_site_name}/mean_swe_snowmodel_m_wy2026.csv',index = False)
    mean_swe_df_acreFt.to_csv(f'/home/rossamower/work/aso/snowmodel/domains/{aso_site_name}/mean_swe_snowmodel_acreFt_wy2026.csv',index = False)
    
    return mean_swe_df_m,mean_swe_df_acreFt



# def aggregate_basin_swe_by_elev(
#                               aso_swe_da: xr.DataArray,
#                               aso_dem_bin_da: xr.DataArray,
#                               date_str: str,
#                              ) -> tuple[pd.DataFrame,pd.DataFrame]:
#     """
#     Aggregate ASO SWE by elevation bins.
#     Input:
#       aso_swe_da - xarray DataArray of ASO SWE [date,y,x]
#       aso_dem_bin_da - xarray DataArray of ASO DEM bins [elev,y,x]
#       elev_bin_labels - list of elevation bin labels.
#     Output:
#       xarray DataArray of binned ASO SWE [date,elev]
#     """
#     m3_acreFt = 0.000810714
#     swe_list_m = []
#     swe_list_acreFt = []
#     swe_list_m.append(f'{date_str[0:4]}-{pad_zero(date_str[4:6])}-{pad_zero(date_str[6:8])}')
#     swe_list_acreFt.append(f'{date_str[0:4]}-{pad_zero(date_str[4:6])}-{pad_zero(date_str[6:8])}')

#     for elev in range(0,aso_dem_bin_da.elev.shape[0]):
#         mask = ~aso_dem_bin_da[elev].isnull()
#         num_grids = int(mask.sum())
#         area_m2 = num_grids * 50 * 50
#         swe_basin_elev_m = float(aso_swe_da.where(mask).mean(dim=['y','x']).values)
#         swe_basin_elev_acreft = swe_basin_elev_m * area_m2 * m3_acreFt
        
#         swe_list_m.append(swe_basin_elev_m)
#         swe_list_acreFt.append(swe_basin_elev_acreft)
    
#     df_m = pd.DataFrame(data = [swe_list_m],columns = ['Date'] + list(aso_dem_bin_da.elev.values))
#     df_acreFt = pd.DataFrame(data = [swe_list_acreFt],columns = ['Date'] + list(aso_dem_bin_da.elev.values))
#     return df_m,df_acreFt

def match_aso_da(sm_ds: xr.Dataset,
                 aso_ds: xr.Dataset,
                 geog_ds: xr.Dataset,
                 date_str: str,
                 var: str,
                 shape_crs: str,
                 shape_proj_gdf: gpd.GeoDataFrame,
                 ):

    # create date array.
    date_arry = np.array([f'{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}'],dtype = np.datetime64)
    # create mask based on aso date.
    mask_aso = (aso_ds['aso_swe'][0] >= 0).values
    # reshape snowmodel.
    arry = sm_ds[var].values
    # set negative values to zero.
    if var == 'swed':
        arry = arry.copy()
        arry[arry < 0] = 0.0
    # create xarray object.
    da_sm = xr.DataArray(
        data=arry[:,::-1,:],
        dims=["date","y", "x"],
        coords=dict(
            date=(["date"], date_arry),
            y=(["y"], geog_ds.XLAT[:,int(geog_ds.XLONG.shape[1]/2)].values[::-1]),
            x=(["x"], geog_ds.XLONG[int(geog_ds.XLONG.shape[0]/2),:].values),
        ),
    )
    # set name.
    if var == 'swed':
        da_sm.name = f'sm_{var[0:-1]}'
    else:
        pass
    # set crs.
    da_sm = da_sm.rio.write_crs(shape_crs)
    # clip and mask.
    da_sm = da_sm.rio.clip(shape_proj_gdf.geometry).where(mask_aso)
    return da_sm


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

    elev_bin_labels, shape_fpath, demBin_fpath, aso_spatial_fpath, aso_tseries_fpath, snowmodel_dir, snodas_dir, insitu_dir, shape_crs = load_aso_metadata(aso_site_name)

    aso_spatial_ds, aso_demBin_ds, shape_proj_gdf = load_aso_data(aso_spatial_fpath,
                                                                  demBin_fpath,
                                                                  shape_fpath,
                                                                  shape_crs)

    # generate snowmodel todo list.
    print(snowmodel_dir)
    if os.path.exists(f'{snowmodel_dir}/{aso_site_name}_snowmodel_daily_download_checklist.csv'):
        generate_snowmodel_basin_todolist(year_str, month_str, day_str, snowmodel_dir, aso_site_name)
        # generate date list from checklist.
        date_lst = create_datelist(snowmodel_dir, aso_site_name)
    else:
        date_lst = [f'{year_str}{pad_zero(month_str)}{pad_zero(day_str)}']
    # load template and dem bin.    
    aso_template = aso_spatial_ds['aso_swe'].isel(date=0)  
    aso_dem = aso_demBin_ds['dem_bin']
    print('elev_bin_labels',elev_bin_labels)
    print('shape_fpath',shape_fpath)
    print('demBin_fpath',demBin_fpath)
    print('aso_spatial_fpath',aso_spatial_fpath)
    print('snowmodel_dir',snowmodel_dir)
    print('snodas_dir',snodas_dir)
    print('shape_crs',shape_crs)
    print('date_lst',date_lst)

    # load and output snowmodel spatial dataset matched to aso.
    sm_spatial_ds = match_sm_aso_spatial(
                               aso_site_name,
                               aso_spatial_ds,
                               aso_dem,
                               shape_crs,
                               shape_proj_gdf,
                               date_lst,
                               snowmodel_model = 'snowfall_frac_3',
                               var = 'swed',
                               campaign_base_dir = '/home/rossamower/work/aso/snowmodel/domains/',
                               water_year_limit = None,
                               saveNC = True,
                               isHRRR = True,
                              )

    df_corr = pd.read_csv(f'{insitu_dir}/hrrr_correlated_grids_2017_2025_dowy.csv')
    sm_outputs = f'{snowmodel_dir}outputs/wo_assim/hrrr/snowfall_frac_3/'
    
    sm_output_flist = snowmodel_swe_fpaths(sm_outputs,[2026],'swed')
    # print('sm_output_flist',sm_output_flist)
    ds_hrrr = xr.open_mfdataset(sm_output_flist,concat_dim = 'time',combine = 'nested')
    obs_data_ds = xr.load_dataset(f'{insitu_dir}raw/{aso_site_name}_insitu_obs_daily_wy_2026.nc')
    obs_data_hist = []
    for pil in obs_data_ds.data_vars:
        da = obs_data_ds[pil].where(obs_data_ds[pil].time <= np.datetime64(f'{year_str}-{pad_zero(month_str)}-{pad_zero(day_str)}'),drop = True)
        da.name = pil
        obs_data_hist.append(da)
    
    out = build_sm_pillow_training_ds_slice_first(
                     df_corr,
                     ds_hrrr,
                     obs_data_hist,
                     wy_start = 2026,
                     # wy_end: int = 2025,
                     var_grid = "swed",
                     resample_daily = True,
                     grid_units_to_mm = True,
                     interp_grid_time = True,
                     out_path = f'{insitu_dir}hrrr_correlated_test_2026.zarr',
                     out_format= "zarr",
    )






