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


def download_uaswe_wateryear(water_year: str,output_dir: str = '../data/conus/',stability_level: str = None):
    """
    download snowmodel data corresponding to ASO dates
    Input:
      aso_dates - numpy array of dates.
      scratch_dir - python string of absolute file path for scratch 
                    directory to unzip SnowModel CONUS files.
      geom_proj - geopandas object for projected basin shape.
      clip_dir - python string of relative file path to create clipped SnowModel 
                 tifs.
      aso_site_name - python string of aso site.
    Output:
    """

    # get daily date range for water year.
    dates = np.arange(np.datetime64(f'{int(water_year)-1}-10-01'), 
                      np.datetime64(f'{int(water_year)}-09-30'), 
                      np.timedelta64(1, 'D'))
    # create water year directory.
    wy_dir = f'{output_dir}wy_{water_year}/'
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    if not os.path.exists(wy_dir): os.makedirs(wy_dir)
    # obtain a list of files that have not been downloaded.
    date_lst = [str(i)[0:10].replace('-','') for i in dates]
    date_lst_todwnld = []
    for date in date_lst:
        if f'uaswe_800m_{date}.nc' in os.listdir(wy_dir):
            pass 
        else:
            date_lst_todwnld.append(date)
    # download files.      
    if len(date_lst_todwnld) > 0:
      date_stability = uaswe_download(date_lst_todwnld,wy_dir,stability_level)
    return

def download_uaswe_daily(
                         year: str = None,
                         month: str = None,
                         day: str = None,
                         isMultipleDates: bool = True,
                         output_dir: str = '../data/conus/',
                         stability_level: str = None,
                         ):
    """
    download snowmodel data corresponding to ASO dates
    Input:
      aso_dates - numpy array of dates.
      scratch_dir - python string of absolute file path for scratch 
                    directory to unzip SnowModel CONUS files.
      geom_proj - geopandas object for projected basin shape.
      clip_dir - python string of relative file path to create clipped SnowModel 
                 tifs.
      aso_site_name - python string of aso site.
    Output:
    """

    if stability_level is None:
        download_fname = './wus_uaswe_daily_download_todo.csv'
    else:
        download_fname = f'./wus_uaswe_daily_download_{stability_level}_todo.csv'

    # get daily date range for water year.
    if isMultipleDates:
       df_todo = pd.read_csv(download_fname)
       df_todo['date'] = pd.to_datetime(df_todo['date'])
       dates = df_todo['date'].values
    else:
        start = np.datetime64(f'{year}-{pad_zero(month)}-{pad_zero(day)}')
        end = start + np.timedelta64(1, 'D')
    
        dates = np.arange(start, 
                  end, 
                  np.timedelta64(1, 'D'))
    
    # The water year must come from each DATE, not from the run date. Deriving it once
    # from argv files every downloaded day under the water year the *run* falls in, so a
    # backfill executed on or after Oct 1 wrote WY-N days into wy_(N+1) -- where
    # uaswe_basin_process.py (which reads wy_2026) never found them. On 2026-10-06 that put
    # 43 early and 38 provisional days in the wrong directory and the basin clip silently
    # produced nothing new, leaving the WY2026 series six weeks short.
    date_lst = [str(i)[0:10].replace('-','') for i in dates]
    print(date_lst)

    dates_by_wy = {}
    for date in date_lst:
        wy = str(int(date[0:4]) + 1) if int(date[4:6]) >= 10 else date[0:4]
        dates_by_wy.setdefault(wy, []).append(date)

    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # one download call per water year so a batch spanning Oct 1 still lands correctly.
    for water_year in sorted(dates_by_wy):
        wy_dir = f'{output_dir}wy_{water_year}/'
        if not os.path.exists(wy_dir): os.makedirs(wy_dir)

        date_lst_todwnld = []
        for date in dates_by_wy[water_year]:
            if f'uaswe_800m_{date}.nc' in os.listdir(wy_dir):
                pass
            else:
                date_lst_todwnld.append(date)
        # download files.
        if len(date_lst_todwnld) > 0:
          date_stability = uaswe_download(date_lst_todwnld,wy_dir,stability_level)
    return


def uaswe_download(date_lst,scratch_dir,stability_level=None):
    """
    Download uaswe data. Flexibility to specify stability level or download whatever is available.
    See https://climate.arizona.edu/data/UA_SWE/Readme.txt for more details.
    Input:
      date_lst - list of dates.
      scratch_dir - python string for download directory.
      stability_level - python string of stability level. See documentation for more information.
      sm_concat_fpath - absolute file path of concatenated netcdf file.
    Output:
    """
    if stability_level not in ['early','provisional','stable',None]:
        print('BAD STABILITY LEVEL INPUT. NEED TO BE EITHER "early", "provisional", or "stable".') 


    date_stability = {}
    # download any of the stability levels.
    if stability_level is None:
      for date in date_lst:
          WY = date[0:4]
          stableExists = os.path.exists(f'{scratch_dir}UA_SWE_Depth_800m_v1_{date}_stable.nc')
          provisionalExists = os.path.exists(f'{scratch_dir}UA_SWE_Depth_800m_v1_{date}_provisional.nc')
          earlyExists = os.path.exists(f'{scratch_dir}UA_SWE_Depth_800m_v1_{date}_early.nc')
          if (stableExists == True):
              date_stability[date] = 'stable'
          elif (provisionalExists == True): 
              date_stability[date] = 'provisional'
          elif (earlyExists == True):
              date_stability[date] = 'early'
          else:
            if not os.path.exists(f'{scratch_dir}UA_SWE_Depth_800m_v1_{date}_stable.nc'):
              if not os.path.exists(f'{scratch_dir}UA_SWE_Depth_800m_v1_{date}_provisional.nc'):
                  if not os.path.exists(f'{scratch_dir}UA_SWE_Depth_800m_v1_{date}_stable.nc'):
                    # stable download.
                    if date[4:6] in ['10','11','12']:
                        runcmd(f"wget -P {scratch_dir} https://climate.arizona.edu/data/UA_SWE/DailyData_800m/WY{int(WY)+1}/UA_SWE_Depth_800m_v1_{date}_stable.nc", verbose = False)
                    else:
                        runcmd(f"wget -P {scratch_dir} https://climate.arizona.edu/data/UA_SWE/DailyData_800m/WY{WY}/UA_SWE_Depth_800m_v1_{date}_stable.nc", verbose = False)
                    if not os.path.exists(f'{scratch_dir}UA_SWE_Depth_800m_v1_{date}_stable.nc'):
                        if date[4:6] in ['10','11','12']:
                            print('HERE')
                            runcmd(f"wget -P {scratch_dir} https://climate.arizona.edu/data/UA_SWE/DailyData_800m/WY{int(WY)+1}/UA_SWE_Depth_800m_v1_{date}_provisional.nc", verbose = False)
                        else:
                            runcmd(f"wget -P {scratch_dir} https://climate.arizona.edu/data/UA_SWE/DailyData_800m/WY{WY}/UA_SWE_Depth_800m_v1_{date}_provisional.nc", verbose = False)
                        if not os.path.exists(f'{scratch_dir}UA_SWE_Depth_800m_v1_{date}_provisional.nc'):
                            if date[4:6] in ['10','11','12']:
                                runcmd(f"wget -P {scratch_dir} https://climate.arizona.edu/data/UA_SWE/DailyData_800m/WY{int(WY)+1}/UA_SWE_Depth_800m_v1_{date}_early.nc", verbose = False)
                            else:
                                runcmd(f"wget -P {scratch_dir} https://climate.arizona.edu/data/UA_SWE/DailyData_800m/WY{WY}/UA_SWE_Depth_800m_v1_{date}_early.nc", verbose = False)
                           
                            if not os.path.exists(f'{scratch_dir}UA_SWE_Depth_800m_v1_{date}_early.nc'):
                               print(f'COULD NOT PROCESS: {date}')
                            else:
                                date_stability[date] = 'early'
                        else:
                            date_stability[date] = 'provisional'
                    else:
                        date_stability[date] = 'stable'
            generate_uaswe_download_checklist(date)
    else:
    # download specific (i.e. defined) stability levels.
        for date in date_lst:
            WY = date[0:4]
            if not os.path.exists(f'{scratch_dir}UA_SWE_Depth_800m_v1_{date}_{stability_level}.nc'):
                if date[4:6] in ['10','11','12']:
                    try:
                        runcmd(f"wget -P {scratch_dir} https://climate.arizona.edu/data/UA_SWE/DailyData_800m/WY{int(WY)+1}/UA_SWE_Depth_800m_v1_{date}_{stability_level}.nc", verbose = False)
                    except:
                        print(f'COULD NOT PROCESS: {date}')
                else:
                    try:
                        runcmd(f"wget -P {scratch_dir} https://climate.arizona.edu/data/UA_SWE/DailyData_800m/WY{WY}/UA_SWE_Depth_800m_v1_{date}_{stability_level}.nc", verbose = False)
                    except:
                        print(f'COULD NOT PROCESS: {date}')
                date_stability[date] = stability_level
                generate_uaswe_download_checklist(date,stability_level)


              
    return date_stability

def pad_zero(val):
   str_val = str(val)
   if len(str_val) == 1:
      return '0' + str_val 
   else:
      return str_val
   
def check_mismatching_dates(df,conus_dir):
    # conus_dir may be the CONUS base (holding wy_* subdirectories) or a single wy_
    # directory. Scanning one water year only would report every date outside it as
    # missing, truncating the checklist and re-downloading those days on every run.
    scan_dirs = sorted(
        os.path.join(conus_dir, d) for d in os.listdir(conus_dir)
        if d.startswith('wy_') and os.path.isdir(os.path.join(conus_dir, d))
    ) or [conus_dir]

    date_lst = []
    for d in scan_dirs:
        for file in os.listdir(d):
            if file.endswith('.nc'):
                date = file.split('_')[5]
                date_lst.append(f'{date[0:4]}-{date[4:6]}-{date[6:8]}')
    
    df_dir = pd.DataFrame(data = sorted(date_lst),
                      columns = ['download_date'])
    
    return df_dir

def generate_uaswe_download_todolist(year_str,month_str,day_str,conus_dir,stability_level = None):
    if stability_level is None:
        df = pd.read_csv('./wus_uaswe_daily_download_checklist.csv')
    else:
        df = pd.read_csv(f'./wus_uaswe_daily_download_{stability_level}_checklist.csv')
    df_dir = check_mismatching_dates(df,conus_dir)
    if len(df) != len(df_dir):
        print('MISMATCHING DATES FOUND. UPDATING CHECKLIST.')
        df = df_dir
        if stability_level is None:
            df.to_csv('./wus_uaswe_daily_download_checklist.csv',index = False)
        else:
            df.to_csv(f'./wus_uaswe_daily_download_{stability_level}_checklist.csv',index = False)
    
    last_date_str = df['download_date'].iloc[-1]
    current_date_str = f'{year_str}-{pad_zero(month_str)}-{pad_zero(day_str)}'

    date_range = np.arange(np.datetime64(last_date_str), np.datetime64(current_date_str) + np.timedelta64(1, "D"), np.timedelta64(1, "D"))
    lst = []
    for i in range(1,len(date_range)):
        lst.append([str(date_range[i])])
    df_todo = pd.DataFrame(lst,columns = ['date'])

    if stability_level is None:
        df_todo.to_csv('./wus_uaswe_daily_download_todo.csv',index = False)
    else:         
        df_todo.to_csv(f'./wus_uaswe_daily_download_{stability_level}_todo.csv',index = False)
    return

def generate_uaswe_download_checklist(date_str, stability_level = None):
    year_str = date_str[0:4]
    month_str = date_str[4:6]
    day_str = date_str[6:8]

    if stability_level is None:
        checklist_fname = './wus_uaswe_daily_download_checklist.csv'
    else:
        checklist_fname = f'./wus_uaswe_daily_download_{stability_level}_checklist.csv'

    df_day = pd.DataFrame(data = [f'{year_str}-{month_str}-{day_str}'],
                      columns = ['download_date'])
    df_day['download_date'] = pd.to_datetime(df_day['download_date'])

    if not os.path.exists(checklist_fname):
        df_day.to_csv(checklist_fname,index = False)
    else:
        df = pd.read_csv(checklist_fname)
        df['download_date'] = pd.to_datetime(df['download_date'])

        pd.concat([df,df_day]).to_csv(checklist_fname,index = False)
    
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
    if len(sys.argv) == 5:
        year = sys.argv[1]
        month = sys.argv[2]
        day = sys.argv[3]
        stability = sys.argv[4]
        if stability not in ['early','provisional','stable']:
            stability = None
            base_dir = '/home/rossamower/work/aso/data/uaswe/CONUS/'
            file_dir = '/home/rossamower/work/aso/data/uaswe/CONUS/'
        else:
            base_dir = f'/home/rossamower/work/aso/data/uaswe/CONUS/{stability}/'
            file_dir = f'/home/rossamower/work/aso/data/uaswe/CONUS/{stability}/'
        # download_uaswe_wateryear(water_year,out_dir,stability)
        # generate_uaswe_download_todolist(year,month,day,'/home/rossamower/work/aso/data/uaswe/CONUS/wy_2026/')
        generate_uaswe_download_todolist(year,month,day,file_dir,stability)
        
        download_uaswe_daily(year,
                             month,
                             day,
                             isMultipleDates=True,
                            #  output_dir = '/home/rossamower/work/aso/data/uaswe/CONUS/',
                             output_dir = base_dir,
                             stability_level=stability)




