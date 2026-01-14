#!/bin/bash

# load conda environment.
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate bor

# basins to copy from.
declare -a BASINS=("USCASJ")

# model output dir.
model_output_dir="/home/rossamower/work/aso/data/"
streamlit_output_dir="/home/rossamower/work/aso/snow-ops-streamlit/data/basins/"
current_dir=$(pwd)
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for basin in "${BASINS[@]}"; do
    # uaswe file paths.
    uaswe_dir_in="${model_output_dir}uaswe/${basin}/"
    uaswe_dir_out="${streamlit_output_dir}${basin}/uaswe/"
    uaswe_m_in="${uaswe_dir_in}mean_swe_uaswe_m_wy2026.csv"
    uaswe_acreFt_in="${uaswe_dir_in}mean_swe_uaswe_acreFt_wy2026.csv"
    uaswe_m_out="${uaswe_dir_out}mean_swe_uaswe_m_wy2026.parquet"
    uaswe_acreFt_out="${uaswe_dir_out}mean_swe_uaswe_acreFt_wy2026.parquet"
    # uaswe convert to parquet.
    python /home/rossamower/bin/scripts/csv_to_parquet.py $uaswe_m_in $uaswe_m_out
    python /home/rossamower/bin/scripts/csv_to_parquet.py $uaswe_acreFt_in $uaswe_acreFt_out

    # snodas file paths.
    snodas_dir_in="${model_output_dir}snodas/${basin}/"
    snodas_dir_out="${streamlit_output_dir}${basin}/snodas/"
    snodas_m_in="${snodas_dir_in}mean_swe_snodas_m_wy2026.csv"
    snodas_acreFt_in="${snodas_dir_in}mean_swe_snodas_acreFt_wy2026.csv"
    snodas_m_out="${snodas_dir_out}mean_swe_snodas_m_wy2026.parquet"
    snodas_acreFt_out="${snodas_dir_out}mean_swe_snodas_acreFt_wy2026.parquet"
    # snodas convert to parquet.
    python /home/rossamower/bin/scripts/csv_to_parquet.py $snodas_m_in $snodas_m_out
    python /home/rossamower/bin/scripts/csv_to_parquet.py $snodas_acreFt_in $snodas_acreFt_out

    # insitu file paths.
    insitu_dir_proc_in="${model_output_dir}insitu/${basin}/processed/"
    insitu_dir_raw_in="${model_output_dir}insitu/${basin}/raw/"
    insitu_dir_proc_out="${streamlit_output_dir}${basin}/pillows/processed/"
    insitu_dir_raw_out="${streamlit_output_dir}${basin}/pillows/raw/"
    
    train_fname_in="${insitu_dir_proc_in}pillow_wy_1980_2025_qa1.nc"
    train_fname_out="${insitu_dir_proc_out}pillow_wy_1980_2025_qa1.zarr"

    test_qa_fname_in="${insitu_dir_proc_in}${basin}_insitu_obs_daily_wy_2026.nc"
    test_qa_fname_out="${insitu_dir_proc_out}${basin}_insitu_obs_daily_wy_2026.zarr"
    
    test_raw_fname_in="${insitu_dir_raw_in}${basin}_insitu_obs_daily_wy_2026.nc"
    test_raw_fname_out="${insitu_dir_raw_out}${basin}_insitu_obs_daily_wy_2026.zarr"

    python /home/rossamower/bin/scripts/nc_to_zarr.py $train_fname_in $train_fname_out
    python /home/rossamower/bin/scripts/nc_to_zarr.py $test_qa_fname_in $test_qa_fname_out
    python /home/rossamower/bin/scripts/nc_to_zarr.py $test_raw_fname_in $test_raw_fname_out
    echo ""
    echo "Copied insitu pillows for basin: ${basin}"
    echo ""

    # mlr models.
    mlr_dir_in="${model_output_dir}mlr_prediction/${basin}/models/COMMON_MASK/"
    mlr_dir_out="${streamlit_output_dir}${basin}/mlr_prediction/"
    declare -a models=("season" "accum" "melt")
    declare -a outputs=("mm" "acreFt" "pillows")
    for model in "${models[@]}"; do
        for output in "${outputs[@]}"; do
            model_dir_in="${mlr_dir_in}${model}/${output}/"
            model_dir_out="${mlr_dir_out}${model}/"
            model_in="${model_dir_in}prediction_${output}_wy2026_combination.csv"
            model_out="${model_dir_out}prediction_${output}_wy2026.parquet"
            # mlr convert to parquet.
            python /home/rossamower/bin/scripts/csv_to_parquet.py $model_in $model_out
            echo ""
            echo "Copied MLR model: ${model_out}"
            echo "From: ${model_in}"
        done
    done
    # generate visualizations.
    echo ""
    echo "Generating visualizations for basin: ${basin}"
    python /home/rossamower/bin/mlr_prediction/html_visualization.py \
        $basin \
        2026 \
        $current_dir

    # create html file.
    echo ""
    echo "Generating html files for basin: ${basin}"
    # node generateHTMLPred.js $basin --dropTrainInfer="drop NaNs"
    node "${script_dir}/generateHTMLPred.js" "$basin" --dropTrainInfer="drop NaNs"


    
    
done
