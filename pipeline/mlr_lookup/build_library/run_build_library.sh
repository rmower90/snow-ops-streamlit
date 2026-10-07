#!/bin/bash
#SBATCH --job-name=LOOKUP_BUILD_LIBRARY
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
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

# Required, no default -- the whole point of this milestone is that a library version is
# never guessed. Mirrors the MLR_STACK_NAME:? pattern already used in
# mlr_prediction/historic/run_prediction.sh.
BASIN="${BASIN:?BASIN not set}"
LOOKUP_LIB_VERSION="${LOOKUP_LIB_VERSION:?LOOKUP_LIB_VERSION not set -- refusing to guess a library version}"
QA_FILE="${QA_FILE:-pillow_wy_1980_2025_qa6.nc}"
MODE="${MODE:-season}"

# Set LOOKUP_REBUILD=1 to actually fit any missing unit (e.g. a genuinely new
# configuration). Default is verify-only: every unit must already exist on disk.
REBUILD_FLAG=""
if [ -n "${LOOKUP_REBUILD:-}" ]; then REBUILD_FLAG="--rebuild"; fi

REPO_SCRIPTS="/home/rossamower/work/aso/snow-ops-streamlit/scripts/experiments"
cd "$REPO_SCRIPTS"

echo "Building library generation ${LOOKUP_LIB_VERSION} for ${BASIN} (qa=${QA_FILE})"

# See run_historic_prediction.sh for why the exit code is captured explicitly.
set +e
"$PYTHON" -u build_library_generation.py "$BASIN" "$LOOKUP_LIB_VERSION" \
    --mode "$MODE" --qa-file "$QA_FILE" $REBUILD_FLAG
PYTHON_EXIT=$?
set -e

outdir="/home/rossamower/work/aso/data/mlr_prediction/${BASIN}/saved_sub_files/"
mkdir -p "$outdir"
cd /home/rossamower/bin/mlr_lookup/build_library
./clean_output.sh "$outdir"

exit $PYTHON_EXIT
