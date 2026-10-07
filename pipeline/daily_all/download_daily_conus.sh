#!/bin/bash


# load conda environment.
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate sm_nc_process


# current days variables.
YEAR=$(date +%Y)
MONTH=$(date +%m)
DAY=$(date +%d)
#DAY="13"

# previous days variables.
PREV_YEAR=$(date -d "yesterday" +%Y)
PREV_MONTH=$(date -d "yesterday" +%m)
PREV_DAY=$(date -d "yesterday" +%d)
#PREV_DAY="12"

echo "PROCESSING...."
echo "YEAR: $PREV_YEAR"
echo "MONTH: $PREV_MONTH"
echo "DAY: $PREV_DAY"

# current working directory.
CWD="/home/rossamower/bin/daily_all/"

# download hrrr WUS
echo "Downloading HRRR -- WUS"
cd /home/rossamower/bin/hrrr/wus_download/jomey_hrrr/scripts/HRRR/
source ./casper_hrrr_daily_sub_loop.sh $PREV_YEAR $PREV_MONTH $PREV_DAY

# process hrrr WUS
echo "Processing HRRR for SnowModel -- WUS"
cd /home/rossamower/bin/hrrr/wus_process/SnowModel_Preproccess/netcdf/sm_wrf_preprocess 
conda activate sm_nc_process
python process_vars_one_month.py $PREV_YEAR $PREV_MONTH $PREV_DAY

echo "=== DEBUG BEFORE SNODAS ==="
echo "Shell: $SHELL"
echo "User: $(whoami)"
echo "PWD:  $(pwd)"
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "PATH=$PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo "--- which python / python -V ---"
which python || true
python -V || true
python -c "import sys; print('sys.executable:', sys.executable); print('sys.path[0]:', sys.path[0])"

echo "--- which gdal_translate ---"
which gdal_translate || true
gdal_translate --version || true

echo "--- ldd /opt/gdal/bin/gdal_translate (if exists) ---"
/usr/bin/ldd /opt/gdal/bin/gdal_translate 2>/dev/null | grep -i spatialite || true

echo "==========================="


# # download snodas
# echo "Downloading SNODAS -- CONUS"
cd /home/rossamower/bin/snodas/conus_download/
python snodas_conus_download.py $PREV_YEAR $PREV_MONTH $PREV_DAY

# # download uaswe
# echo "Downloading UASWE -- CONUS"
cd /home/rossamower/bin/uaswe/conus_download/
# run early
STABILITY="none"
python uaswe_conus_download.py $PREV_YEAR $PREV_MONTH $PREV_DAY $STABILITY
# run provisional 
STABILITY="provisional"
python uaswe_conus_download.py $PREV_YEAR $PREV_MONTH $PREV_DAY $STABILITY


