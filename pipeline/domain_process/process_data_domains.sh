#!/bin/bash


# load conda environment.
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate sm_nc_process

# aso basin abbreviation
ASO_SITE_NAME="USCASJ"
SM_DIRECTORY="/home/rossamower/work/aso/snowmodel/domains/${ASO_SITE_NAME}/"
SNOWMODEL_PAR="${SM_DIRECTORY}snowmodel.par"

# current days variables.
YEAR=$(date +%Y)
MONTH=$(date +%m)
DAY=$(date +%d)
# DAY="17"

# previous days variables.
PREV_YEAR=$(date -d "yesterday" +%Y)
PREV_MONTH=$(date -d "yesterday" +%m)
PREV_DAY=$(date -d "yesterday" +%d)
# PREV_DAY="16"

# datediff
PREV_DATE="${PREV_YEAR}-${PREV_MONTH}-${PREV_DAY}"
SM_START_DATE="2025-10-01"
start_sec=$(date -d "$SM_START_DATE" +%s)
end_sec=$(date -d "$PREV_DATE" +%s)
SM_TIMESTEP=$(( ((((end_sec - start_sec) / 86400) * 8)+ 7) ))

echo "PROCESSING...."
echo "YEAR: $PREV_YEAR"
echo "MONTH: $PREV_MONTH"
echo "DAY: $PREV_DAY"
echo "HRLY_TIMESTEPS: $SM_TIMESTEP"

# current working directory.
CWD="/home/rossamower/bin/domain_process/"

# Download INSITU -- Domain
echo "Downloading INSITU -- Domain" 
conda deactivate
conda activate bor # switch to bor environment for insitu download
cd /home/rossamower/bin/insitu/basin_download
python insitu_basin_download.py $ASO_SITE_NAME $PREV_YEAR $PREV_MONTH $PREV_DAY

# slice forcing data.
echo "Clipping HRRR -- Domain"
conda deactivate
conda activate sm_nc_process # switch to bor environment for insitu download
cd /home/rossamower/bin/hrrr/basin_process
python hrrr_clip_aso_domains.py $ASO_SITE_NAME $PREV_YEAR $PREV_MONTH

# change snowmodel.par iterations
max_iter_param=$(grep "max_iter =" $SNOWMODEL_PAR | grep -v !)
sed -i "s/$max_iter_param/       max_iter = $((SM_TIMESTEP))/g" $SNOWMODEL_PAR

# run snowmodel.
cd $SM_DIRECTORY
sbatch run_parallel_sm.sh $ASO_SITE_NAME $PREV_YEAR $PREV_MONTH $PREV_DAY

# Clip UASWE -- Domain
echo "Clipping UASWE -- Domain" 
cd /home/rossamower/bin/uaswe/basin_process
# run early.
STABILITY="none"
python uaswe_basin_process.py $ASO_SITE_NAME $PREV_YEAR $PREV_MONTH $PREV_DAY $STABILITY
# run provisional.
STABILITY="provisional"
python uaswe_basin_process.py $ASO_SITE_NAME $PREV_YEAR $PREV_MONTH $PREV_DAY $STABILITY

# Clip SNODAS -- Domain
echo "Clipping SNODAS -- Domain" 
cd /home/rossamower/bin/snodas/basin_process
python snodas_basin_process.py $ASO_SITE_NAME $PREV_YEAR $PREV_MONTH $PREV_DAY




