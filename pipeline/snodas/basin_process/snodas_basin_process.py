# uaswe_basin_process.py
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

sys.path.insert(1, '../../scripts/')

import metadata as metadata





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
    shape_crs = f'EPSG:{cfg["crs"]["epsg"]}'
    return elev_bin_labels, shape_fpath, demBin_fpath, aso_spatial_fpath, aso_tseries_fpath, uaswe_dir, snodas_dir, shape_crs

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
        swe_basin_elev_m = float(aso_swe_da.where(mask).mean(dim=['y','x']).values)
        swe_basin_elev_acreft = swe_basin_elev_m * area_m2 * m3_acreFt
        
        swe_list_m.append(swe_basin_elev_m)
        swe_list_acreFt.append(swe_basin_elev_acreft)
    
    df_m = pd.DataFrame(data = [swe_list_m],columns = ['Date'] + list(aso_dem_bin_da.elev.values))
    df_acreFt = pd.DataFrame(data = [swe_list_acreFt],columns = ['Date'] + list(aso_dem_bin_da.elev.values))
    return df_m,df_acreFt

def uaswe_ld_rpjt_clp(
                      geom_proj: gpd.GeoDataFrame,
                      date_lst: list,
                      aso_template: xr.DataArray,
                      aso_dem: xr.DataArray,
                      aso_site_name: str,
                      reproj_match: bool = True,
                      deleteCONUS: bool = False,
                      date_stability = None,
                      var = 'SWE',
                      epsg_uaswe = 'EPSG:4326',
                      is800m = True,
                      ):
    """
    snowmodel load, reproject, clip and save as tifs.
    Input:
      scratch_dir - python string of absolute file path for scratch 
                    directory to unzip SnowModel CONUS files.
      geom_proj - geopandas object for projected basin shape.
      clip_dir - python string of relative file path to create clipped SnowModel 
                 tifs.
      sm_concat_fpath - absolute file path of concatenated netcdf file.
    Output:
    """
    # constants
    scratch_dir = '/home/rossamower/work/aso/data/snodas/CONUS/nc_conus_files/wy_2026/'
    clip_dir = f'/home/rossamower/work/aso/data/snodas/{aso_site_name}/wy_2026/'
    if not os.path.exists(clip_dir): os.makedirs(clip_dir)
    
    # load tables.
    if not os.path.exists(f'/home/rossamower/work/aso/data/snodas/{aso_site_name}/mean_swe_snodas_m_wy2026.csv'):
        mean_swe_df_m = None
        mean_swe_df_acreFt = None
    else:
        mean_swe_df_m = pd.read_csv(f'/home/rossamower/work/aso/data/snodas/{aso_site_name}/mean_swe_snodas_m_wy2026.csv')
        mean_swe_df_acreFt = pd.read_csv(f'/home/rossamower/work/aso/data/snodas/{aso_site_name}/mean_swe_snodas_acreFt_wy2026.csv')
    
    # make sure template is in correct projection.
    aso_template = aso_template.rio.write_crs(geom_proj.crs)

    m3_acreFt = 0.000810714
    vals = []
    print('Processing SNODAS Data:')
    for date in date_lst:
        print(date)
        if not os.path.exists(f'{clip_dir}SNODAS_{date[0:4]}{date[4:6]}{date[6:8]}.nc'):
        #   try:
            for file in os.listdir(scratch_dir):
                if file.startswith(f'SNODAS_{date[0:4]}{date[4:6]}{date[6:8]}'):
                    ds = xr.load_dataset(f'{scratch_dir}{file}')

                    da_swed = xr.DataArray(
                        data = (ds[var].values), # already in meters,
                        dims = ["date","y","x"],
                            coords = dict(
                                date = (["date"], [np.datetime64(f'{date[0:4]}-{date[4:6]}-{date[6:8]}')]),
                                y = (["y"], ds.lat.values),
                                x = (["x"], ds.lon.values)
                            )
                        )
                    da_swed.name = 'SWE'
                    da_swed.attrs = {'SWE':'meters'}
                    ### write projection ##
                    da_swed = da_swed.rio.write_crs(epsg_uaswe)
                    ### reproject ##
                    da_swed_geog = da_swed.rio.reproject(geom_proj.crs)
                    ### clip ##
                    da_swed_clip = da_swed_geog.rio.clip(geom_proj.geometry)

                    ### Reproject and Match ###
                    if reproj_match:
                        da_swed_clip = da_swed_clip.rio.reproject_match(aso_template,resampling=Resampling.bilinear)
                        msk1 = aso_template.isnull().values
                        da_swed_clip = da_swed_clip.where(da_swed_clip >= 0.0)
                        da_swed_clip = da_swed_clip.where(~msk1)


                    da_swed_clip.to_netcdf(f'{clip_dir}SNODAS_{date[0:4]}{date[4:6]}{date[6:8]}.nc')
        #   except:
            #   pass

         # add values to table.
        da_ = xr.open_dataarray(f'{clip_dir}SNODAS_{date[0:4]}{date[4:6]}{date[6:8]}.nc')[0]
        day_df_m,day_df_acreFt = aggregate_basin_swe_by_elev(da_,aso_dem,date)
        if mean_swe_df_m is not None:
            mean_swe_df_m = pd.concat([mean_swe_df_m,day_df_m])
            mean_swe_df_acreFt = pd.concat([mean_swe_df_acreFt,day_df_acreFt])
        else:
            mean_swe_df_m = day_df_m
            mean_swe_df_acreFt = day_df_acreFt

        # update checklist
        generate_uaswe_basin_checklist(date,f'/home/rossamower/work/aso/data/snodas/{aso_site_name}/',aso_site_name)
    
    mean_swe_df_m.to_csv(f'/home/rossamower/work/aso/data/snodas/{aso_site_name}/mean_swe_snodas_m_wy2026.csv',index = False)
    mean_swe_df_acreFt.to_csv(f'/home/rossamower/work/aso/data/snodas/{aso_site_name}/mean_swe_snodas_acreFt_wy2026.csv',index = False)
    
    return mean_swe_df_m,mean_swe_df_acreFt

def generate_uaswe_basin_todolist(year_str,
                                     month_str,
                                     day_str,
                                     uaswe_dir,
                                     aso_site_name):
    
    df = pd.read_csv(f'{uaswe_dir}/{aso_site_name}_snodas_daily_download_checklist.csv')
    last_date_str = df['download_date'].iloc[-1]
    current_date_str = f'{year_str}-{pad_zero(month_str)}-{pad_zero(day_str)}'

    date_range = np.arange(np.datetime64(last_date_str), np.datetime64(current_date_str) + np.timedelta64(1, "D"), np.timedelta64(1, "D"))
    lst = []
    for i in range(1,len(date_range)):
        lst.append([str(date_range[i])])
    df_todo = pd.DataFrame(lst,columns = ['date'])

    df_todo.to_csv(f'{uaswe_dir}/{aso_site_name}_snodas_daily_download_todo.csv',index = False)
    return

def generate_uaswe_basin_checklist(date_str,
                                     uaswe_dir,
                                     aso_site_name):
    year_str = date_str[0:4]
    month_str = date_str[4:6]
    day_str = date_str[6:8]

    df_day = pd.DataFrame(data = [f'{year_str}-{month_str}-{day_str}'],
                      columns = ['download_date'])
    df_day['download_date'] = pd.to_datetime(df_day['download_date'])

    if not os.path.exists(f'{uaswe_dir}/{aso_site_name}_snodas_daily_download_checklist.csv'):
        df_day.to_csv(f'{uaswe_dir}/{aso_site_name}_snodas_daily_download_checklist.csv',index = False)
    else:
        df = pd.read_csv(f'{uaswe_dir}/{aso_site_name}_snodas_daily_download_checklist.csv')
        df['download_date'] = pd.to_datetime(df['download_date'])

        pd.concat([df,df_day]).to_csv(f'{uaswe_dir}/{aso_site_name}_snodas_daily_download_checklist.csv',index = False)
    
    return          

def pad_zero(val_str):
    if len(val_str) ==1:
        return '0' + val_str
    else:
        return val_str

def create_datelist(uaswe_dir,
                    aso_site_name):
    
    df = pd.read_csv(f'{uaswe_dir}/{aso_site_name}_snodas_daily_download_todo.csv')
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

    elev_bin_labels, shape_fpath, demBin_fpath, aso_spatial_fpath, aso_tseries_fpath, uaswe_dir, snodas_dir, shape_crs = load_aso_metadata(aso_site_name)

    aso_spatial_ds, aso_demBin_ds, shape_proj_gdf = load_aso_data(aso_spatial_fpath,
                                                                  demBin_fpath,
                                                                  shape_fpath,
                                                                  shape_crs)

    # generate uaswe todo list.
    if os.path.exists(f'{snodas_dir}{aso_site_name}_snodas_daily_download_checklist.csv'):
        generate_uaswe_basin_todolist(year_str, month_str, day_str, snodas_dir, aso_site_name)
        # generate date list from checklist.
        date_lst = create_datelist(snodas_dir, aso_site_name)
    else:
        date_lst = [f'{year_str}{pad_zero(month_str)}{pad_zero(day_str)}']
    # load template and dem bin.    
    aso_template = aso_spatial_ds['aso_swe'].isel(date=0)  
    aso_dem = aso_demBin_ds['dem_bin']

    df_uaswe = uaswe_ld_rpjt_clp(
                                 geom_proj = shape_proj_gdf,
                                 date_lst = date_lst,
                                 aso_template = aso_template,
                                 aso_dem = aso_dem,
                                 aso_site_name = aso_site_name,
                                 reproj_match = True,
                                 deleteCONUS = False,
                                 date_stability = None,
                                 var = 'SWE',
                                 epsg_uaswe = 'EPSG:4326',
                                 is800m = True,
                                 )




