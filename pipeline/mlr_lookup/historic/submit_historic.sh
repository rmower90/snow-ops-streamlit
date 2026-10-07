#!/usr/bin/env bash
# Usage: ./submit_historic.sh BASIN LOOKUP_LIB_VERSION LOOKUP_RUN_VERSION [csv_file]
#
# Basin, library version, and run version are arguments to THIS script, not columns in
# the CSV -- every WY job submitted by one invocation shares them. The CSV lists only
# water_year, so an arbitrary subset (a rerun of a handful of years, say) is just a
# smaller CSV, no column bookkeeping required.
#
# Submission-loop robustness: a single sbatch call can fail in a way that is genuinely
# AMBIGUOUS. Observed for real on 2026-08-27: sbatch reported
# "Socket timed out on send/recv operation" for WY1984 and this script (then piping into a
# `while read` loop under set -e) aborted -- but the job had, in fact, already reached the
# controller and later ran to completion as job 162642. The client-side error did not mean
# "no job was created"; it meant "the client didn't get confirmation either way."
#
# Blindly retrying in that situation risks a duplicate job for the same water year. So on
# ANY non-zero sbatch exit this script does the smallest safe thing: stop immediately,
# name exactly which WY's outcome is unknown and which WYs were never attempted, and let a
# human check squeue/sacct for that WY before deciding whether to resubmit it. It does not
# try to distinguish "definitely rejected" from "maybe accepted" failure modes -- that
# reconciliation logic is not needed yet.
set -euo pipefail

BASIN="${1:?usage: submit_historic.sh BASIN LOOKUP_LIB_VERSION LOOKUP_RUN_VERSION [csv_file]}"
LOOKUP_LIB_VERSION="${2:?usage: submit_historic.sh BASIN LOOKUP_LIB_VERSION LOOKUP_RUN_VERSION [csv_file]}"
LOOKUP_RUN_VERSION="${3:?usage: submit_historic.sh BASIN LOOKUP_LIB_VERSION LOOKUP_RUN_VERSION [csv_file]}"
INPUT_CSV="${4:-loop_submission.csv}"

cd "$(dirname "$0")"

if [ ! -f "$INPUT_CSV" ]; then
    echo "no such CSV: $INPUT_CSV" >&2
    exit 1
fi

# Read the whole WY list into an array up front, rather than piping into a `while read`
# loop. A piped while-loop runs in a subshell, so any accounting (submitted_ids,
# submitted_wys) built up inside it would vanish the moment the pipe closes -- exactly the
# information this script needs to still have on hand if a later WY's submission fails.
mapfile -t water_years < <(tail -n +2 "$INPUT_CSV" | sed 's/[[:space:]]*$//' | grep -v '^$')

submitted_ids=()
submitted_wys=()

for i in "${!water_years[@]}"; do
    water_year="${water_years[$i]}"
    echo "submitting WY=${water_year}  basin=${BASIN}  library=${LOOKUP_LIB_VERSION}  run=${LOOKUP_RUN_VERSION}"

    # Bracketed with set +e/-e (not left to the global set -e) so a failure here is
    # handled explicitly below rather than silently killing the script mid-loop with no
    # summary of what had already been submitted.
    set +e
    sbatch_out="$(sbatch --export=ALL,BASIN="${BASIN}",WATER_YEAR="${water_year}",LOOKUP_LIB_VERSION="${LOOKUP_LIB_VERSION}",LOOKUP_RUN_VERSION="${LOOKUP_RUN_VERSION}" \
        run_historic_prediction.sh 2>&1)"
    sbatch_rc=$?
    set -e

    if [ "$sbatch_rc" -eq 0 ]; then
        job_id="$(echo "$sbatch_out" | grep -oE '[0-9]+$' | tail -1)"
        echo "  -> submitted: job ${job_id}"
        submitted_ids+=("$job_id")
        submitted_wys+=("$water_year")
    else
        remaining=("${water_years[@]:$((i+1))}")
        echo "------------------------"
        echo "AMBIGUOUS SUBMISSION FAILURE at WY=${water_year}"
        echo "  sbatch exit code: ${sbatch_rc}"
        echo "  sbatch output: ${sbatch_out}"
        echo
        echo "  This WY's job may or may not have actually been created. Do NOT rerun this"
        echo "  script blindly -- that risks a duplicate job for WY=${water_year}. Check"
        echo "  manually first, e.g.:"
        echo "    squeue -u \$USER -n LOOKUP_HISTORIC_PREDICTION"
        echo
        echo "  Successfully submitted before this failure (${#submitted_ids[@]}):"
        echo "    WYs: ${submitted_wys[*]:-<none>}"
        echo "    job IDs: ${submitted_ids[*]:-<none>}"
        echo "  NOT YET ATTEMPTED (${#remaining[@]}): ${remaining[*]:-<none>}"
        exit 1
    fi
    sleep 1
done

echo "------------------------"
echo "submitted ${#submitted_ids[@]} of ${#water_years[@]} jobs from ${INPUT_CSV}"
echo "job IDs: ${submitted_ids[*]}"
