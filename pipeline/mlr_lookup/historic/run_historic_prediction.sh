#!/bin/bash
#SBATCH --job-name=LOOKUP_HISTORIC_PREDICTION
#SBATCH --ntasks=1
#SBATCH --time=00:45:00
#SBATCH --mail-user=rossamower@ucar.edu
#SBATCH --mail-type=END
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# Exact validated interpreter, not `conda activate bor`. Confirmed by real failure on
# 2026-08-27: `source /opt/miniforge3/etc/profile.d/conda.sh; conda activate bor` completed
# without error inside a non-interactive SLURM batch shell, but the resulting `python` did
# not have xarray on its path -- all 5 jobs submitted that way crashed with
# `ModuleNotFoundError: No module named 'xarray'` before running a single line of the
# prediction logic. Every validated run in this branch used this exact absolute path
# directly; do the same here rather than a different, differently-tested activation path.
PYTHON=/home/rossamower/work/aso/conda/envs/bor/bin/python
if [ ! -x "$PYTHON" ]; then
    echo "FATAL: $PYTHON is not executable" >&2
    exit 1
fi

# All four required, no defaults -- basin/library/run version are submission-level
# arguments (set once by submit_historic.sh, shared by every WY job in the sweep), not
# CSV columns, so an arbitrary subset of years can be resubmitted without editing the CSV.
BASIN="${BASIN:?BASIN not set}"
WATER_YEAR="${WATER_YEAR:?WATER_YEAR not set}"
LOOKUP_LIB_VERSION="${LOOKUP_LIB_VERSION:?LOOKUP_LIB_VERSION not set -- refusing to guess a library version}"
LOOKUP_RUN_VERSION="${LOOKUP_RUN_VERSION:?LOOKUP_RUN_VERSION not set -- refusing to guess a run version}"
QA_FILE="${QA_FILE:-pillow_wy_1980_2025_qa6.nc}"
SCORE_RULE="${SCORE_RULE:-adj_r2}"

REPO_SCRIPTS="/home/rossamower/work/aso/snow-ops-streamlit/scripts/experiments"
cd "$REPO_SCRIPTS"

echo "Historic lookup prediction: basin=${BASIN} wy=${WATER_YEAR} library=${LOOKUP_LIB_VERSION} run=${LOOKUP_RUN_VERSION}"

# set -e is active for the rest of this script EXCEPT around this one command. The job's
# final exit status must be PYTHON's, not clean_output.sh's -- letting `set -e` abort
# immediately on a python failure would skip clean_output.sh entirely (losing the failed
# job's logs to the shared submission directory instead of saved_sub_files/); letting
# clean_output.sh run unconditionally and become the LAST command would let its own
# (always-0) exit code silently overwrite python's. Capture explicitly, restore set -e,
# run cleanup either way, then exit with the captured code at the very end.
set +e
"$PYTHON" -u historic_lookup_prediction.py "$BASIN" "$WATER_YEAR" \
    --library-version "$LOOKUP_LIB_VERSION" --run-version "$LOOKUP_RUN_VERSION" \
    --qa-file "$QA_FILE" --score-rule "$SCORE_RULE"
PYTHON_EXIT=$?
set -e

outdir="/home/rossamower/work/aso/data/mlr_prediction/${BASIN}/saved_sub_files/"
mkdir -p "$outdir"
cd /home/rossamower/bin/mlr_lookup/historic
./clean_output.sh "$outdir"

exit $PYTHON_EXIT
