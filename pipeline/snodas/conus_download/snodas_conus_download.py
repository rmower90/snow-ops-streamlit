# snowmodel_bounds.py
"""
    Script identifies snowmodel bounds and projection and changes input into topo_vege scripts.
"""
import xarray as xr
import os 
#import rioxarray as rxr
from pyproj import CRS, Transformer
import sys
import subprocess
import numpy as np
import pandas as pd
import re
from typing import Optional
import shutil

#def snodas_create_header(data_file):
#    # data_file = "us_ssmv01025SlL01T0024TTNATS2020012205DP001.dat"
#    header_file = data_file.replace(".dat", ".hdr")
#
#    # ENVI header content
#    header_content = """ENVI
#    samples = 6935
#    lines = 3351
#    bands = 1
#    header offset = 0
#    file type = ENVI Standard
#    data type = 2
#    interleave = bsq
#    byte order = 1
#    """
#
#    # Write the header file
#    with open(header_file, "w") as hdr_file:
#        hdr_file.write(header_content)
#
#    print(f"Header file '{header_file}' created successfully.")
#    return

def snodas_create_header(data_file):
    header_file = data_file.replace(".dat", ".hdr")

    header_content = (
        "ENVI\n"
        "samples = 6935\n"
        "lines   = 3351\n"
        "bands   = 1\n"
        "header offset = 0\n"
        "file type = ENVI Standard\n"
        "data type = 2\n"
        "interleave = bsq\n"
        "byte order = 1\n"
    )

    with open(header_file, "w") as hdr_file:
        hdr_file.write(header_content)

    print(f"Header file '{header_file}' created successfully.")
    return header_file



# def snodas_dat_to_nc(infile,nc_dir,download_date,gdat_translate_path):
#     outfile = f"{nc_dir}SNODAS_{download_date}.nc"
#     command = [
#         gdat_translate_path,
#         "-of", "NetCDF",
#         "-a_srs", "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
#         "-a_nodata", "-9999",
#         "-a_ullr", "-124.73333333333333", "52.87500000000000",
#         "-66.94166666666667", "24.95000000000000",
#         infile,
#         outfile
#     ]

#     # Run the command
#     try:
#         subprocess.run(command, check=True)
#         print("Command executed successfully!")
#     except subprocess.CalledProcessError as e:
#         print(f"Command failed with error: {e}")
#     return outfile

def parse_date_from_filename(path: str) -> np.datetime64:
    """
    Extract YYYYMMDD from filenames like SNODAS_20251001.nc
    """
    m = re.search(r"(\d{8})", os.path.basename(path))
    if not m:
        raise ValueError(f"Could not find YYYYMMDD in filename: {path}")
    s = m.group(1)
    return np.datetime64(f"{s[0:4]}-{s[4:6]}-{s[6:8]}")

def postprocess_snodas_daily_nc(
    nc_path: str,
    date: Optional[np.datetime64] = None,
    nodata: int = -9999,
    to_meters: bool = True,
) -> xr.Dataset:
    """
    Convert GDAL-produced SNODAS NetCDF into a consistent dataset:
      - Band1 -> SWE
      - nodata -> NaN
      - add time dimension
      - optional units mm->m
    """
    ds = xr.open_dataset(nc_path)

    if "Band1" not in ds:
        raise KeyError(f"'Band1' not found. Vars are: {list(ds.data_vars)}")

    # pick date
    if date is None:
        date = parse_date_from_filename(nc_path)

    # turn into float (so NaNs can exist)
    swe = ds["Band1"].astype("float32")

    # nodata masking
    swe = swe.where(swe != nodata)

    # unit conversion
    if to_meters:
        swe = swe / 1000.0
        units = "m"
    else:
        units = "mm"

    swe = swe.rename("SWE")
    swe.attrs.update(
        long_name="Snow Water Equivalent",
        units=units,
        source="SNODAS",
    )

    # Attach CRS if you want (GDAL sometimes writes it in different ways)
    # If your x/y are lon/lat degrees, add standard_name attrs:
    if "x" in ds.coords:
        ds["x"].attrs.update(standard_name="longitude", units="degrees_east")
    if "y" in ds.coords:
        ds["y"].attrs.update(standard_name="latitude", units="degrees_north")

    # Build final dataset with a time dimension
    # out = swe.expand_dims(time=[date]).to_dataset()
    date_ns = np.datetime64(date, "ns")
    out = swe.expand_dims(time=[date_ns]).to_dataset()

    # Carry over coords (x/y) and their attributes
    for c in ("x", "y"):
        if c in ds.coords:
            out = out.assign_coords({c: ds[c]})
            out[c].attrs.update(ds[c].attrs)

    # helpful global attrs
    out.attrs.update(
        title="SNODAS SWE (daily)",
        history=f"postprocess_snodas_daily_nc from {os.path.basename(nc_path)}",
    )
    out.to_netcdf(nc_path)

    return out

# def snodas_dat_to_nc(infile, nc_dir, download_date, gdat_translate_path):
#     outfile = f"{nc_dir}SNODAS_{download_date}.nc"

#     command = [
#         gdat_translate_path,
#         "-of", "NetCDF",
#         "-a_srs", "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
#         "-a_nodata", "-9999",
#         "-a_ullr", "-124.73333333333333", "52.87500000000000",
#         "-66.94166666666667", "24.95000000000000",
#         infile,
#         outfile,
#     ]

#     # --- CLEAN ENVIRONMENT FOR GDAL ---
#     env = os.environ.copy()
#     for k in [
#         "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
#         "GDAL_DRIVER_PATH", "GDAL_DATA", "PROJ_LIB",
#         "PYTHONPATH",
#         "LD_LIBRARY_PATH",
#     ]:
#         env.pop(k, None)

#     try:
#         subprocess.run(command, check=True, env=env)
#         print("Command executed successfully!")
#         ds = postprocess_snodas_daily_nc(outfile)
#     except subprocess.CalledProcessError as e:
#         print(f"Command failed with error: {e}")

#     return outfile

def snodas_dat_to_nc(infile, nc_dir, download_date, gdal_translate_path):
    outfile = f"{nc_dir}SNODAS_{download_date}.nc"

    gdal_translate_path = shutil.which(gdal_translate_path) or gdal_translate_path
    print("Using gdal_translate:", gdal_translate_path)
    print("Input:", os.path.abspath(infile))
    print("Output:", os.path.abspath(outfile))

    command = [
        gdal_translate_path,
        "-of", "NetCDF",
        "-a_srs", "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        "-a_nodata", "-9999",
        "-a_ullr", "-124.73333333333333", "52.87500000000000",
        "-66.94166666666667", "24.95000000000000",
        infile,
        outfile,
    ]

    print("Running:", " ".join(command))

    p = subprocess.run(command, text=True, capture_output=True)
    if p.returncode != 0:
        print("gdal_translate stdout:\n", p.stdout)
        print("gdal_translate stderr:\n", p.stderr)
        raise RuntimeError(f"gdal_translate failed with code {p.returncode}")

    postprocess_snodas_daily_nc(outfile)
    return outfile


def download_snodas_daily_data_water_year(water_year: str,
                               gdal_translate_fpath: str = "/glade/u/apps/derecho/24.12/spack/opt/spack/gdal/3.9.3/gcc/12.4.0/vydy/bin/gdal_translate",
                               base_out_dir: str = "/glade/campaign/ral/hap/rmower/aso/data/snodas/processing/snodas_download/data/"):
    """
    download snodas data specific to dates and basin data provided
    Input:
      cwd_ - python string of current working directory.

      scratch_dir - python string of absolute file path for scratch 
                    directory to unzip SnowModel CONUS files.
      geom_proj - geopandas object for projected basin shape.
      clip_dir - python string of relative file path to create clipped SnowModel 
                 tifs.
      aso_site_name - python string of aso site.
    Output:
    """
    # month dict
    month_dict = {
        '01': '01_Jan',
        '02': '02_Feb',
        '03': '03_Mar',
        '04': '04_Apr',
        '05': '05_May',
        '06': '06_Jun',
        '07': '07_Jul',
        '08': '08_Aug',
        '09': '09_Sep',
        '10': '10_Oct',
        '11': '11_Nov',
        '12': '12_Dec'
        
    }

    raw_dir = f'{base_out_dir}raw_conus_files/'
    nc_dir = f'{base_out_dir}nc_conus_files/'

    cwd_base = os.getcwd()

    # make directories if they do not exist.
    if not os.path.exists(raw_dir): os.makedirs(raw_dir)
    if not os.path.exists(nc_dir): os.makedirs(nc_dir)

    # create water year directory.
    nc_dir = f'{nc_dir}wy_{water_year}/'
    if not os.path.exists(nc_dir): os.makedirs(nc_dir)

    #get daily date range for water year.
    dates = np.arange(np.datetime64(f'{int(water_year)-1}-10-01'), 
                      np.datetime64(f'{int(water_year)}-09-30'), 
                      np.timedelta64(1, 'D'))
    

    # date string with hyphen
    date_str = [str(i)[0:10] for i in dates]
    # date string w/o hyphen
    date_no_hyphen = [i.replace('-','') for i in date_str]
    
    for download_date in range(0,len(date_no_hyphen)):
        download_directory = f'{raw_dir}{date_no_hyphen[download_date]}'
        print(date_no_hyphen[download_date])
        print(download_directory)
        if not os.path.exists(f'{nc_dir}SNODAS_{date_no_hyphen[download_date]}.nc'):
            try:
              tar_name = f'SNODAS_{date_no_hyphen[download_date]}.tar'
              # download snodas in directory.
              if not os.path.exists(download_directory):
                runcmd(f"wget -P {download_directory} https://noaadata.apps.nsidc.org/NOAA/G02158/masked/{date_no_hyphen[download_date][0:4]}/{month_dict[date_no_hyphen[download_date][4:6]]}/{tar_name}", verbose = False)
              # change directory
              os.chdir(download_directory)
              # unpack files.
              subprocess.run(["tar", "-xvf", tar_name])
              # iterate of unpacked files
              for file in os.listdir():
                  if 'us_ssmv11034' in file: # SWE
                    if '.dat.gz' in file:
                        # unzip file.
                        subprocess.run(["gzip", "-d", file])
                        # create header.
                        snodas_create_header(file[:-3])
                        # save netcdf.
                        file_to_save = snodas_dat_to_nc(file[:-3],nc_dir,date_no_hyphen[download_date],gdal_translate_fpath)
            
              # delete intermediate files.
              for file in os.listdir():
                os.remove(file)
            except:
              pass
        # change back to python directory since all path locations are relative
        os.chdir(cwd_base)

        #       # load netcdf.
        #       ds = xr.load_dataset(file_to_save)
        #       ds = ds.rio.write_crs('EPSG:4326')
        #       ds_clip = ds.rio.clip(geom_geog.geometry)
        #       # create xarray 
        #       da = xr.DataArray(
        #         data=ds_clip.Band1.values.reshape(1,ds_clip.Band1.shape[0],ds_clip.Band1.shape[1])/1000, # convert to meters
        #         dims=["date", "y", "x"],
        #         coords=dict(
        #             y=(["y"], ds_clip.lat.values),
        #             x=(["x"], ds_clip.lon.values),
        #             date=[np.datetime64(date_str[download_date])],
        #         ),
        #       )
        #       # name dataarray.
        #       da.name = 'SWE'
        #       da.attrs = {'SWE':'meters'}
        #       # name write crs.
        #       da = da.rio.write_crs('EPSG:4326')
        #       # reproject to shape.
        #       da = da.rio.reproject(proj_crs)
        #       # save to file.
        #       da.to_netcdf(f'{clip_dir}SNODAS_{date_no_hyphen[download_date]}.nc')
        #       # remove conus netcdf file.
        #       if deleteCONUS:
        #         os.remove(file_to_save)
        #     except:
        #         print(f'COULD NOT PROCESS {date_no_hyphen[download_date]}')

        # # delete download directory.
        # try:
        #   pass
        #   # os.removedirs(download_directory)
        # except:
        #   pass
    
    # change directory back to original.
    # os.chdir(cwd_)
    return 

def pad_zero(val):
   str_val = str(val)
   if len(str_val) == 1:
      return '0' + str_val 
   else:
      return str_val
   
def run_gdal_translate(cmd_list):
    env = os.environ.copy()

    # Remove conda contamination for GDAL + C++ runtime
    for k in [
        "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
        "GDAL_DRIVER_PATH", "GDAL_DATA", "PROJ_LIB",
        "PYTHONPATH",
        # "LD_LIBRARY_PATH",
    ]:
        env.pop(k, None)

    # (optional) keep a minimal PATH so your spack gdal_translate is found if you use just "gdal_translate"
    # but since you're passing an absolute path, PATH doesn't matter much.
    return subprocess.run(cmd_list, check=True, env=env)

def download_snodas_daily_data(year: str,
                               month: str,
                               day: str,
                               isMultipleDates: bool = True,
                               gdal_translate_fpath: str = "gdal_translate",
                               base_out_dir: str = "/glade/campaign/ral/hap/rmower/aso/data/snodas/processing/snodas_download/data/"):
    """
    download snodas data specific to dates and basin data provided
    Input:
      cwd_ - python string of current working directory.

      scratch_dir - python string of absolute file path for scratch 
                    directory to unzip SnowModel CONUS files.
      geom_proj - geopandas object for projected basin shape.
      clip_dir - python string of relative file path to create clipped SnowModel 
                 tifs.
      aso_site_name - python string of aso site.
    Output:
    """
    # month dict
    month_dict = {
        '01': '01_Jan',
        '02': '02_Feb',
        '03': '03_Mar',
        '04': '04_Apr',
        '05': '05_May',
        '06': '06_Jun',
        '07': '07_Jul',
        '08': '08_Aug',
        '09': '09_Sep',
        '10': '10_Oct',
        '11': '11_Nov',
        '12': '12_Dec'
        
    }

    if isMultipleDates:
       df_todo = pd.read_csv('./wus_snodas_daily_download_todo.csv')
       df_todo['date'] = pd.to_datetime(df_todo['date'])
       dates = df_todo['date'].values

    if int(month) >= 10:
       water_year = str(int(year) + 1)
    else:
       water_year = year
     
    raw_dir = f'{base_out_dir}raw_conus_files/'
    nc_dir = f'{base_out_dir}nc_conus_files/'

    cwd_base = os.getcwd()

    # make directories if they do not exist.
    if not os.path.exists(raw_dir): os.makedirs(raw_dir)
    if not os.path.exists(nc_dir): os.makedirs(nc_dir)

    # create water year directory.
    nc_dir = f'{nc_dir}wy_{water_year}/'
    if not os.path.exists(nc_dir): os.makedirs(nc_dir)

    if isMultipleDates == False:
        start = np.datetime64(f'{year}-{pad_zero(month)}-{pad_zero(day)}')
        end = start + np.timedelta64(1, 'D')
    
        dates = np.arange(start, 
                  end, 
                  np.timedelta64(1, 'D'))
    

    # date string with hyphen
    date_str = [str(i)[0:10] for i in dates]
    # date string w/o hyphen
    date_no_hyphen = [i.replace('-','') for i in date_str]
    
    for download_date in range(0,len(date_no_hyphen)):
        download_directory = f'{raw_dir}{date_no_hyphen[download_date]}'
        print(date_no_hyphen[download_date])
        print(download_directory)
        if not os.path.exists(f'{nc_dir}SNODAS_{date_no_hyphen[download_date]}.nc'):
            # try:
            tar_name = f'SNODAS_{date_no_hyphen[download_date]}.tar'
            # download snodas in directory.
            if not os.path.exists(download_directory):
              runcmd(f"wget -P {download_directory} https://noaadata.apps.nsidc.org/NOAA/G02158/masked/{date_no_hyphen[download_date][0:4]}/{month_dict[date_no_hyphen[download_date][4:6]]}/{tar_name}", verbose = False)
            # change directory
            os.chdir(download_directory)
            # unpack files.
            subprocess.run(["tar", "-xvf", tar_name])
            # iterate of unpacked files
            for file in os.listdir():
                if 'us_ssmv11034' in file: # SWE
                  if '.dat.gz' in file:
                      # unzip file.
                      subprocess.run(["gzip", "-d", file])
                      # create header.
                      snodas_create_header(file[:-3])
                      # save netcdf.
                      file_to_save = snodas_dat_to_nc(file[:-3],nc_dir,date_no_hyphen[download_date],gdal_translate_fpath)
                      # run_gdal_translate(command)
            
            # delete intermediate files.
            for file in os.listdir():
              os.remove(file)
            # add date to checklist.
            os.chdir(cwd_base)
            generate_snodas_download_checklist(date_no_hyphen[download_date])
            # except:
            #   pass
        # change back to python directory since all path locations are relative
        os.chdir(cwd_base)

    return

def check_mismatching_dates(df,conus_dir):
    date_lst = []
    for file in os.listdir(conus_dir):
        if file.endswith('.nc'):
            date = file.split('_')[1].split('.')[0]
            date_lst.append(f'{date[0:4]}-{date[4:6]}-{date[6:8]}')
    
    df_dir = pd.DataFrame(data = sorted(date_lst),
                      columns = ['download_date'])
    
    return df_dir

def generate_snodas_download_todolist(year_str,month_str,day_str,conus_dir):
    
    df = pd.read_csv('./wus_snodas_daily_download_checklist.csv')
    df_dir = check_mismatching_dates(df,conus_dir)
    if len(df) != len(df_dir):
        print('MISMATCHING DATES FOUND. UPDATING CHECKLIST.')
        df = df_dir
        df.to_csv('./wus_snodas_daily_download_checklist.csv',index = False)
    
    last_date_str = df['download_date'].iloc[-1]
    print(last_date_str)
    current_date_str = f'{year_str}-{pad_zero(month_str)}-{pad_zero(day_str)}'
    print(current_date_str)

    date_range = np.arange(np.datetime64(last_date_str), np.datetime64(current_date_str) + np.timedelta64(1, "D"), np.timedelta64(1, "D"))
    lst = []
    for i in range(1,len(date_range)):
        lst.append([str(date_range[i])])
    df_todo = pd.DataFrame(lst,columns = ['date'])

    df_todo.to_csv('./wus_snodas_daily_download_todo.csv',index = False)

def generate_snodas_download_checklist(date_str):
    year_str = date_str[0:4]
    month_str = date_str[4:6]
    day_str = date_str[6:8]

    df_day = pd.DataFrame(data = [f'{year_str}-{month_str}-{day_str}'],
                      columns = ['download_date'])
    df_day['download_date'] = pd.to_datetime(df_day['download_date'])

    if not os.path.exists('./wus_snodas_daily_download_checklist.csv'):
        df_day.to_csv('./wus_snodas_daily_download_checklist.csv',index = False)
    else:
        df = pd.read_csv('./wus_snodas_daily_download_checklist.csv')
        df['download_date'] = pd.to_datetime(df['download_date'])

        pd.concat([df,df_day]).to_csv('./wus_snodas_daily_download_checklist.csv',index = False)
    
    return


def runcmd(cmd, verbose = False, *args, **kwargs):

    print(cmd)
    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


if __name__ =="__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python snodas_conus_download.py <water year>")
    #     sys.exit(1)
    if len(sys.argv) == 2:
        water_year = sys.argv[1]
        download_snodas_daily_data_water_year(water_year)
    if len(sys.argv) == 4:
        year = sys.argv[1]
        month = sys.argv[2]
        day = sys.argv[3]
        generate_snodas_download_todolist(year,month,day,'/home/rossamower/work/aso/data/snodas/CONUS/nc_conus_files/wy_2026/')
        
        download_snodas_daily_data(year,
                                   month,
                                   day,
                                   isMultipleDates = True,
                                   base_out_dir = '/home/rossamower/work/aso/data/snodas/CONUS/')





