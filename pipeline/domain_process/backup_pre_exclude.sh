#!/bin/bash
# One-off snapshot helper: copies the current QA + MLR outputs under a tagged suffix
# so a follow-up run with date-ranged pillow exclusion doesn't overwrite them.
# Delete this file once the comparison runs are done.

set -euo pipefail

SITE="${1:-USCASJ}"
TAG="${2:-pre_exclude}"
INSITU="/home/rossamower/work/aso/data/insitu/${SITE}"
MLR="/home/rossamower/work/aso/data/mlr_prediction/${SITE}/models"

echo "Snapshotting ${SITE} outputs with tag __${TAG}"
echo "  INSITU: ${INSITU}"
echo "  MLR:    ${MLR}"
echo ""

copy_if_exists() {
    local src="$1"
    local dst="$2"
    if [ -e "$src" ]; then
        if [ -e "$dst" ]; then
            echo "  SKIP (dest exists): $dst"
        else
            cp -r "$src" "$dst"
            echo "  COPIED: $src -> $dst"
        fi
    else
        echo "  MISSING (skipped): $src"
    fi
}

echo "--- insitu_qa.py outputs ---"
copy_if_exists "${INSITU}/processed/${SITE}_insitu_obs_daily_wy_2026.nc" \
               "${INSITU}/processed/${SITE}_insitu_obs_daily_wy_2026__${TAG}.nc"
copy_if_exists "${INSITU}/qa/insitu_qa_simple_wy_2026.csv" \
               "${INSITU}/qa/insitu_qa_simple_wy_2026__${TAG}.csv"
copy_if_exists "${INSITU}/qa/insitu_qa_detail_wy_2026.csv" \
               "${INSITU}/qa/insitu_qa_detail_wy_2026__${TAG}.csv"
copy_if_exists "${INSITU}/qa/qa_viz_2026" \
               "${INSITU}/qa/qa_viz_2026__${TAG}"
copy_if_exists "${INSITU}/qa/qa_method_diagnostics_2026" \
               "${INSITU}/qa/qa_method_diagnostics_2026__${TAG}"

echo ""
echo "--- mlr_prediction.py outputs ---"
copy_if_exists "${MLR}/COMMON_MASK"      "${MLR}/COMMON_MASK__${TAG}"
copy_if_exists "${MLR}/SNOWMODEL_IMPUTE" "${MLR}/SNOWMODEL_IMPUTE__${TAG}"
copy_if_exists "${MLR}/imputation"       "${MLR}/imputation__${TAG}"

echo ""
echo "Done."
