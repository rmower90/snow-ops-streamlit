#!/bin/bash
#SBATCH --job-name=MLR_PREDICTION_2026_USCASJ
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mail-user=rossamower@ucar.edu
#SBATCH --mail-type=END
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err


ASO_SITE_NAME=$1
water_year=$2
isSplit=$3
isAccum=$4
showOutput=$5
pillowImputation=$6


## run aggregation script.
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate bor
echo "MLR Prediction..."
# -u keeps stdout unbuffered. without it python block-buffers ~8KB, so a job that is
# cancelled or killed loses every line it had not yet flushed -- which is why the
# saved_sub_files logs were 18 bytes (the echo above) with no python output at all.
python -u mlr_prediction.py $ASO_SITE_NAME $water_year $isSplit $isAccum $showOutput $pillowImputation

# move output and error files to mlr_prediction directory.
outdir="/home/rossamower/work/aso/data/mlr_prediction/${ASO_SITE_NAME}/saved_sub_files/"
mkdir -p $outdir

./clean_output.sh $outdir




