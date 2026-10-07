#!/bin/bash
#SBATCH --job-name=LOOKUP_DAILY_PREDICTION
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --mail-user=rossamower@ucar.edu
#SBATCH --mail-type=END
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# Exact validated interpreter -- see run_historic_prediction.sh for why `conda activate
# bor` is not used here: confirmed on 2026-08-27 that it does not reliably put xarray on
# PATH inside a non-interactive SLURM batch shell.
PYTHON=/home/rossamower/work/aso/conda/envs/bor/bin/python
if [ ! -x "$PYTHON" ]; then
    echo "FATAL: $PYTHON is not executable" >&2
    exit 1
fi

BASIN="${BASIN:?BASIN not set}"
WATER_YEAR="${WATER_YEAR:?WATER_YEAR not set}"
DATE="${DATE:?DATE not set (YYYY-MM-DD)}"
LOOKUP_LIB_VERSION="${LOOKUP_LIB_VERSION:?LOOKUP_LIB_VERSION not set -- refusing to guess a library version}"
LOOKUP_RUN_VERSION="${LOOKUP_RUN_VERSION:?LOOKUP_RUN_VERSION not set -- refusing to guess a run version}"
QA_FILE="${QA_FILE:-pillow_wy_1980_2025_qa6.nc}"
SCORE_RULE="${SCORE_RULE:-adj_r2}"

REPO_SCRIPTS="/home/rossamower/work/aso/snow-ops-streamlit/scripts/experiments"
cd "$REPO_SCRIPTS"

echo "Daily lookup prediction: basin=${BASIN} wy=${WATER_YEAR} date=${DATE} library=${LOOKUP_LIB_VERSION} run=${LOOKUP_RUN_VERSION}"

# See run_historic_prediction.sh for why the exit code is captured explicitly rather than
# left to `set -e` (would skip cleanup) or to clean_output.sh's own exit (always 0).
set +e
"$PYTHON" -u daily_lookup_prediction.py "$BASIN" "$WATER_YEAR" "$DATE" \
    --library-version "$LOOKUP_LIB_VERSION" --run-version "$LOOKUP_RUN_VERSION" \
    --qa-file "$QA_FILE" --score-rule "$SCORE_RULE"
PYTHON_EXIT=$?
set -e

outdir="/home/rossamower/work/aso/data/mlr_prediction/${BASIN}/saved_sub_files/"
mkdir -p "$outdir"
cd /home/rossamower/bin/mlr_lookup/daily
./clean_output.sh "$outdir"

exit $PYTHON_EXIT
