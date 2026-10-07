#!/bin/bash
#SBATCH --job-name=MLR_PREDICTION_HISTORIC_FRIANT
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mail-user=rossamower@ucar.edu
#SBATCH --mail-type=END
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err


ASO_SITE_NAME="FRIANT"
#water_year="2000"
#isSplit="0"
#isAccum="0"
showOutput="0"
pillowImputation="1"
# Required, no default. Previously this was a hardcoded literal, which is how v2-v6 came to
# be unreproducible: the version tag was only an output directory name while the behaviour
# lived in mutable source. Refusing to guess means a re-run cannot silently overwrite an
# existing version -- set it explicitly, e.g.
#     sbatch --export=water_year=1997,isSplit=0,isAccum=0,MLR_STACK_NAME=v8 ./run_prediction.sh
aso_stack_name="${MLR_STACK_NAME:?MLR_STACK_NAME not set -- refusing to guess an output version and risk overwriting one}"

## run aggregation script.
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate bor
echo "MLR HISTORIC Prediction..."
# -u keeps stdout unbuffered. without it python block-buffers ~8KB, so a job that is
# cancelled or killed loses every line it had not yet flushed -- which is why 262 of 290
# saved_sub_files logs were 27 bytes (the echo above) with no python output at all.
python -u mlr_prediction_historic.py $ASO_SITE_NAME $water_year $isSplit $isAccum $showOutput $pillowImputation $aso_stack_name

# move output and error files to mlr_prediction directory.
outdir="/home/rossamower/work/aso/data/mlr_prediction/${ASO_SITE_NAME}/saved_sub_files/"
mkdir -p $outdir

./clean_output.sh $outdir




