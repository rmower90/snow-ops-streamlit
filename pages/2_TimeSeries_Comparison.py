import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="SWE Time Series Compare", layout="wide")
st.title("SWE Time Series Comparison (WY2026)")

DATA_ROOT = Path("data/basins")

# -----------------------------
# Helpers
# -----------------------------
def list_basins(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")])

def load_uaswe_or_snodas_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "Date" not in df.columns:
        raise ValueError(f"{path} is missing a 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").set_index("Date")

def load_mlr_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError(f"{path} is missing a 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date")

def normalize_band_name(name: str) -> str:
    s = str(name).strip()
    s = s.replace("<7000", "<7k").replace(">12000", ">12k")
    s = s.replace("7000-8000", "7k-8k").replace("8000-9000", "8k-9k")
    s = s.replace("9000-10000", "9k-10k").replace("10000-11000", "10k-11k")
    s = s.replace("11000-12000", "11k-12k")
    s = s.replace("total", "total")
    return s

GRID_BAND_COLS = [
    "<7000", "7000-8000", "8000-9000",
    "9000-10000", "10000-11000", "11000-12000",
    ">12000", "total"
]

# -----------------------------
# Basin selector (top of page)
# -----------------------------
basins = list_basins(DATA_ROOT)
if not basins:
    st.error(f"No basins found under: {DATA_ROOT.resolve()}")
    st.stop()

default_basin = "USCASJ" if "USCASJ" in basins else basins[0]
basin = st.selectbox("Basin", basins, index=basins.index(default_basin))

basin_root = DATA_ROOT / basin

# Paths for WY2026 (you can generalize WY next)
PATH_MLR = basin_root / "mlr_prediction/season/prediction_mm_wy2026_combination.csv"
PATH_UASWE = basin_root / "uaswe/mean_swe_uaswe_m_wy2026.parquet"
PATH_SNODAS = basin_root / "snodas/mean_swe_snodas_m_wy2026.parquet"

# -----------------------------
# Load data
# -----------------------------
missing = [p for p in [PATH_MLR, PATH_UASWE, PATH_SNODAS] if not p.exists()]
if missing:
    st.error(
        "Missing expected files for this basin:\n\n"
        + "\n".join([f"- {p.as_posix()}" for p in missing])
        + "\n\nCreate these files (or adjust naming) to enable the comparison page."
    )
    st.stop()

with st.spinner(f"Loading data for {basin}..."):
    df_uaswe = load_uaswe_or_snodas_parquet(PATH_UASWE)
    df_snodas = load_uaswe_or_snodas_parquet(PATH_SNODAS)
    df_mlr_raw = load_mlr_csv(PATH_MLR)

# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Series selection")
    band = st.selectbox(
        "Elevation band",
        ["total", "<7k", "7k-8k", "8k-9k", "9k-10k", "10k-11k", "11k-12k", ">12k"],
        index=0,
    )

    st.divider()
    st.header("MLR filter")
    if "Training Infer NaNs" not in df_mlr_raw.columns:
        st.error("MLR CSV missing 'Training Infer NaNs' column.")
        st.stop()

    nan_vals = sorted(df_mlr_raw["Training Infer NaNs"].dropna().unique().tolist())
    if not nan_vals:
        st.error("No valid values found in 'Training Infer NaNs'.")
        st.stop()

    nan_choice = st.selectbox("Training Infer NaNs", nan_vals)

    st.divider()
    show_table = st.checkbox("Show merged table", value=False)

# -----------------------------
# UASWE / SNODAS (meters → mm)
# -----------------------------
grid_col_map = {normalize_band_name(c): c for c in GRID_BAND_COLS if c in df_uaswe.columns}

if band not in grid_col_map:
    st.error(
        f"Band '{band}' not found in UASWE/SNODAS columns. "
        f"Available: {sorted(grid_col_map.keys())}"
    )
    st.stop()

uaswe_series = (df_uaswe[grid_col_map[band]] * 1000.0).rename("UASWE (mm)")
snodas_series = (df_snodas[grid_col_map[band]] * 1000.0).rename("SNODAS (mm)")

# -----------------------------
# MLR (already mm)
# -----------------------------
df_mlr = df_mlr_raw[df_mlr_raw["Training Infer NaNs"] == nan_choice]

mlr_col = "Basin" if band == "total" else band
if mlr_col not in df_mlr.columns:
    st.error(f"MLR column '{mlr_col}' not found.")
    st.stop()

mlr_series = (
    df_mlr[["Date", mlr_col]]
    .set_index("Date")[mlr_col]
    .sort_index()
    .rename("MLR (mm)")
)

# -----------------------------
# Combine + plot
# -----------------------------
df_plot = pd.concat([mlr_series, uaswe_series, snodas_series], axis=1)

st.subheader(f"{basin} – Basin Mean SWE (mm)")
st.line_chart(df_plot, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("MLR points", int(df_plot["MLR (mm)"].count()))
c2.metric("UASWE points", int(df_plot["UASWE (mm)"].count()))
c3.metric("SNODAS points", int(df_plot["SNODAS (mm)"].count()))

st.caption(
    f"Basin: {basin} | Band: {band} | Training Infer NaNs: {nan_choice} | "
    "Units: mm (UASWE/SNODAS converted from meters)"
)

if show_table:
    st.subheader("Merged data")
    st.dataframe(df_plot, use_container_width=True)
