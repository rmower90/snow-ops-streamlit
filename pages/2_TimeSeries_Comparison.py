import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="SWE Time Series Compare", layout="wide")
st.title("USCASJ – SWE Time Series Comparison (WY2026)")

DATA_ROOT = Path("data/basins/USCASJ")

PATH_MLR = DATA_ROOT / "mlr_prediction/season/prediction_mm_wy2026_combination.csv"
PATH_UASWE = DATA_ROOT / "uaswe/mean_swe_uaswe_m_wy2026.csv"
PATH_SNODAS = DATA_ROOT / "snodas/mean_swe_snodas_m_wy2026.csv"

# -----------------------------
# Helpers
# -----------------------------
def load_uaswe_or_snodas(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"{path} is missing a 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df

def load_mlr(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"{path} is missing a 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df

def normalize_band_name(name: str) -> str:
    """Make band labels consistent across datasets."""
    s = str(name).strip()
    s = s.replace("<7000", "<7k").replace(">12000", ">12k")
    s = s.replace("7000-8000", "7k-8k").replace("8000-9000", "8k-9k")
    s = s.replace("9000-10000", "9k-10k").replace("10000-11000", "10k-11k")
    s = s.replace("11000-12000", "11k-12k")
    s = s.replace("total", "total")
    return s

# UASWE/SNODAS columns (meters)
GRID_BAND_COLS = ["<7000", "7000-8000", "8000-9000", "9000-10000", "10000-11000", "11000-12000", ">12000", "total"]

# -----------------------------
# Load
# -----------------------------
with st.spinner("Loading CSVs..."):
    df_uaswe = load_uaswe_or_snodas(PATH_UASWE)
    df_snodas = load_uaswe_or_snodas(PATH_SNODAS)
    df_mlr_raw = load_mlr(PATH_MLR)

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Series selection")

    band_options = ["total", "<7k", "7k-8k", "8k-9k", "9k-10k", "10k-11k", "11k-12k", ">12k"]
    band = st.selectbox("Elevation band", band_options, index=0)

    st.divider()
    st.header("MLR filters")

    basins = sorted(df_mlr_raw["Basin"].dropna().unique().tolist()) if "Basin" in df_mlr_raw.columns else ["USCASJ"]
    basin = st.selectbox("Basin (MLR)", basins, index=0)

    model_types = sorted(df_mlr_raw["Model Type"].dropna().unique().tolist()) if "Model Type" in df_mlr_raw.columns else []
    model_type = st.selectbox("Model Type", model_types) if model_types else None

    qa_vals = sorted(df_mlr_raw["Prediction QA"].dropna().unique().tolist()) if "Prediction QA" in df_mlr_raw.columns else []
    qa_choice = st.selectbox("Prediction QA", qa_vals) if qa_vals else None

    nan_vals = sorted(df_mlr_raw["Training Infer NaNs"].dropna().unique().tolist()) if "Training Infer NaNs" in df_mlr_raw.columns else []
    nan_choice = st.selectbox("Training Infer NaNs", nan_vals) if nan_vals else None

    st.divider()
    st.header("Plot options")
    show_table = st.checkbox("Show merged table", value=False)

# -----------------------------
# Get UASWE/SNODAS series for selected band (convert meters -> mm)
# -----------------------------
grid_col_map = {normalize_band_name(c): c for c in GRID_BAND_COLS if c in df_uaswe.columns}
if band not in grid_col_map:
    st.error(
        f"Band '{band}' not found in UASWE/SNODAS columns. "
        f"Available (normalized): {sorted(grid_col_map.keys())}"
    )
    st.stop()

uaswe_series = (df_uaswe[grid_col_map[band]] * 1000.0).rename("UASWE (mm)")
snodas_series = (df_snodas[grid_col_map[band]] * 1000.0).rename("SNODAS (mm)")

# -----------------------------
# Filter MLR and pick series for selected band (already mm)
# -----------------------------
df_mlr = df_mlr_raw.copy()

if "Basin" in df_mlr.columns:
    df_mlr = df_mlr[df_mlr["Basin"] == basin]

if model_type is not None:
    df_mlr = df_mlr[df_mlr["Model Type"] == model_type]
if qa_choice is not None:
    df_mlr = df_mlr[df_mlr["Prediction QA"] == qa_choice]
if nan_choice is not None:
    df_mlr = df_mlr[df_mlr["Training Infer NaNs"] == nan_choice]

mlr_col = "Basin" if band == "total" else band  # MLR uses Basin for total/basin mean
if mlr_col not in df_mlr.columns:
    st.error(
        f"MLR column '{mlr_col}' not found. "
        f"Check the CSV headers or adjust mapping for 'total'."
    )
    st.stop()

mlr_series = (
    df_mlr[["Date", mlr_col]]
    .dropna(subset=["Date"])
    .set_index("Date")[mlr_col]
    .sort_index()
    .rename("MLR (mm)")
)

# -----------------------------
# Align + plot (all in mm)
# -----------------------------
df_plot = pd.concat([mlr_series, uaswe_series, snodas_series], axis=1)

st.subheader("Basin Mean SWE (mm)")
st.line_chart(df_plot, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("MLR points", int(df_plot["MLR (mm)"].count()))
c2.metric("UASWE points", int(df_plot["UASWE (mm)"].count()))
c3.metric("SNODAS points", int(df_plot["SNODAS (mm)"].count()))

st.caption(
    f"Band: {band} | MLR filters: Basin={basin}"
    + (f", Model Type={model_type}" if model_type else "")
    + (f", Prediction QA={qa_choice}" if qa_choice else "")
    + (f", Training Infer NaNs={nan_choice}" if nan_choice else "")
    + " | Units: mm (UASWE/SNODAS converted from meters)"
)

if show_table:
    st.subheader("Merged data")
    st.dataframe(df_plot, use_container_width=True)
