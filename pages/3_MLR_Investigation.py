"""
MLR Investigation Page

Purpose:
- Recreate MLR preprocessing and cross-validation for a selected
  basin / WY / seasonal / elevation band
- Inspect intermediate artifacts:
    - df_sum_total
    - drop-NaNs table
    - imputed observations
    - cross-validation selections (later)

NOTE:
This page is intentionally lightweight. All heavy lifting should live
in src/snow_ops/mlr/.
"""

import streamlit as st
from pathlib import Path
import pandas as pd

# -----------------------------------------------------------------------------
# App configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="MLR Investigation", layout="wide")
st.title("MLR Investigation")

DATA_ROOT = Path("data/basins")

# -----------------------------------------------------------------------------
# Planned imports (to be implemented in src/)
# -----------------------------------------------------------------------------
# These DO NOT exist yet – this is the target architecture.

# from snow_ops.paths import (
#     list_basins,
#     available_wys_for_selection,
# )

# from snow_ops.mlr.preprocess import (
#     run_preprocessing_only,
# )

# from snow_ops.mlr.run import (
#     run_crossval_selection_for_app,
# )

# from snow_ops.viz.plotly_ts import (
#     plot_mlr_reproduction,
# )

# -----------------------------------------------------------------------------
# UI controls (mirrors TimeSeries Comparison page)
# -----------------------------------------------------------------------------
top_left, top_mid, top_right = st.columns([1.2, 1.2, 1.4])

with top_left:
    band = st.selectbox(
        "Elevation Band",
        ["total", "<7k", "7k-8k", "8k-9k", "9k-10k", "10k-11k", "11k-12k", ">12k"],
        index=0,
    )

with top_mid:
    seasonal_choice = st.selectbox(
        "Seasonal",
        ["season", "accumulation", "melt"],
        index=0,
    )

with top_right:
    units_choice = st.selectbox(
        "Units",
        ["milimeters"],   # default for now
        index=0,
        disabled=True,    # intentionally locked for v1
    )

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Basin")
    basin = st.selectbox(
        "Basin",
        ["USCASJ"],  # placeholder; will be dynamic
        index=0,
    )

    st.divider()

    st.header("Water Year")
    wy = st.selectbox(
        "Water Year",
        [2026],  # placeholder; will be auto-detected
        index=0,
    )

    st.divider()

    run_button = st.button("Run MLR Investigation")

# -----------------------------------------------------------------------------
# Execution placeholder
# -----------------------------------------------------------------------------
if run_button:
    st.info(
        "MLR preprocessing and cross-validation will run here.\n\n"
        "This will call:\n"
        "  - run_preprocessing_only(...)\n"
        "  - run_crossval_selection_for_app(...)\n\n"
        "Outputs will be visualized below."
    )

    # -------------------------------------------------------------------------
    # FUTURE (not yet implemented)
    # -------------------------------------------------------------------------
    #
    # with st.spinner("Running preprocessing..."):
    #     preprocess_out = run_preprocessing_only(
    #         basin=basin,
    #         water_year=wy,
    #         seasonal=seasonal_choice,
    #         band=band,
    #     )
    #
    # st.subheader("Preprocessing summary")
    # st.write(preprocess_out.keys())
    #
    # with st.spinner("Running cross-validation..."):
    #     cv_out = run_crossval_selection_for_app(
    #         preprocess_out,
    #         model_type="drop_nans",
    #     )
    #
    # plot_mlr_reproduction(cv_out)

# -----------------------------------------------------------------------------
# Page notes (for now)
# -----------------------------------------------------------------------------
with st.expander("Notes on planned functionality", expanded=False):
    st.markdown(
        """
        **This page will eventually allow you to:**
        - Re-run MLR preprocessing for a specific basin / WY / band
        - Inspect `df_sum_total` and NaN handling
        - Reproduce cross-validation station selection
        - Compare predicted SWE vs ASO SWE
        - Visualize residuals and coefficients

        **Design principle:**
        - Streamlit page = UI + orchestration only
        - All heavy computation lives in `src/snow_ops/mlr/`
        """
    )
