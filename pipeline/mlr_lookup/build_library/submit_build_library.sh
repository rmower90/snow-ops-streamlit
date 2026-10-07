#!/usr/bin/env bash
# Usage: ./submit_build_library.sh BASIN LOOKUP_LIB_VERSION [QA_FILE] [--rebuild]
set -euo pipefail

BASIN="${1:?usage: submit_build_library.sh BASIN LOOKUP_LIB_VERSION [QA_FILE] [--rebuild]}"
LOOKUP_LIB_VERSION="${2:?usage: submit_build_library.sh BASIN LOOKUP_LIB_VERSION [QA_FILE] [--rebuild]}"
QA_FILE="${3:-pillow_wy_1980_2025_qa6.nc}"
REBUILD_ARG="${4:-}"

REBUILD_EXPORT=""
if [ "$REBUILD_ARG" = "--rebuild" ]; then REBUILD_EXPORT=",LOOKUP_REBUILD=1"; fi

cd "$(dirname "$0")"
echo "Submitting library build: basin=${BASIN} version=${LOOKUP_LIB_VERSION} qa=${QA_FILE} rebuild=${REBUILD_ARG:-no}"
sbatch --export=ALL,BASIN="${BASIN}",LOOKUP_LIB_VERSION="${LOOKUP_LIB_VERSION}",QA_FILE="${QA_FILE}"${REBUILD_EXPORT} \
    run_build_library.sh
