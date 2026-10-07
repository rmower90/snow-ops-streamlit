#!/bin/bash

outdir=$1

# Move only THIS job's logs. Six USCASJ jobs (46 for FRIANT historic) are submitted
# together and share this working directory, so an unscoped `mv *.out` grabs every
# other job's log too. Content survives the move -- mv within a filesystem keeps the
# inode and slurm writes on through its open fd -- but each later job then finds
# nothing to move and errors, and the ownership of logs becomes arbitrary.
# The `[ -e ]` guard also silences `mv: cannot stat '*.out'` when a glob matches nothing.
if [ -n "$SLURM_JOB_ID" ]; then
    patterns=("*_${SLURM_JOB_ID}.out" "*_${SLURM_JOB_ID}.err")
else
    patterns=("*.out" "*.err")
fi

for pat in "${patterns[@]}"; do
    for f in $pat; do
        [ -e "$f" ] && mv "$f" "$outdir"
    done
done
