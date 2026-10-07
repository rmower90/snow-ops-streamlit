#!/bin/bash


# load conda environment.
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate bor

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


# Run insitu QA -- Domain
echo "Running INSITU QA -- Domain"  
cd /home/rossamower/bin/insitu/qa
python insitu_qa.py $ASO_SITE_NAME $PREV_YEAR $PREV_MONTH $PREV_DAY

# Run MLR
echo "Running MLR Prediction -- Domain"  
cd /home/rossamower/bin/mlr_prediction

# Ensemble size for the uncertainty band. 1 = published path only, byte-identical to the
# pre-ensemble behaviour (verified: 30,720/30,720 rank-1 values matched). 10 adds
# ensemble_tidy_wy{YYYY}.csv beside the existing prediction_* files; no downstream consumer
# reads it, so publishing is unaffected either way. Overridable from the environment.
# Cost measured on USCASJ WY2026: critical path 81 -> 92 min at N=10.
MLR_N_ENSEMBLE="${MLR_N_ENSEMBLE:-10}"
echo "MLR_N_ENSEMBLE=${MLR_N_ENSEMBLE}"
#python mlr_prediction.py $ASO_SITE_NAME 2026 0 0 0
# run season model -- pillow imputation.
sbatch --export=ALL,MLR_N_ENSEMBLE=${MLR_N_ENSEMBLE} run_prediction.sh $ASO_SITE_NAME 2026 0 0 0 1
# run accumulation model -- pillow imputation.
sbatch --export=ALL,MLR_N_ENSEMBLE=${MLR_N_ENSEMBLE} run_prediction.sh $ASO_SITE_NAME 2026 1 1 0 1
# run melt model -- pillow imputation.
sbatch --export=ALL,MLR_N_ENSEMBLE=${MLR_N_ENSEMBLE} run_prediction.sh $ASO_SITE_NAME 2026 1 0 0 1

# run season model -- SnowModel imputation.
sbatch --export=ALL,MLR_N_ENSEMBLE=${MLR_N_ENSEMBLE} run_prediction.sh $ASO_SITE_NAME 2026 0 0 0 0
# run accumulation model -- SnowModel imputation.
sbatch --export=ALL,MLR_N_ENSEMBLE=${MLR_N_ENSEMBLE} run_prediction.sh $ASO_SITE_NAME 2026 1 1 0 0
# run melt model -- SnowModel imputation.
sbatch --export=ALL,MLR_N_ENSEMBLE=${MLR_N_ENSEMBLE} run_prediction.sh $ASO_SITE_NAME 2026 1 0 0 0