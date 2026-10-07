#!/usr/bin/env bash
# Loop through CSV with columns: year,month,directory

INPUT_CSV="loop_submission.csv"

# Skip header and read each line
tail -n +2 "$INPUT_CSV" | while IFS=',' read -r year qa_level elev_bin isSplit isAccum; do
  echo "water_year: $year"
  echo "user_qa_level: $qa_level"
  echo "isSplit: $isSplit"
  echo "isAccum: $isAccum"
  sbatch --export=water_year=${year},isSplit=${isSplit},isAccum=${isAccum} ./run_prediction.sh 
  sleep 2

  # Example usage: call your HRRR download script
  # ./download_hrrr.sh "$year" "$month" "$directory" Google

  echo "------------------------"
done
