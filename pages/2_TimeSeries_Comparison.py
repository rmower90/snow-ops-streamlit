import re
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(page_title="SWE Time Series Compare", layout="wide")
st.title("SWE Time Series Comparison")

DATA_ROOT = Path("data/basins")

# -----------------------------
# Helpers
# -----------------------------
def list_basins(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")])

@st.cache_data(show_spinner=False)
def load_timeseries_parquet(path_str: str) -> pd.DataFrame:
    """Load a parquet with Date column + band columns."""
    path = Path(path_str)
    df = pd.read_parquet(path)
    if "Date" not in df.columns:
        raise ValueError(f"{path} is missing a 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").set_index("Date")

@st.cache_data(show_spinner=False)
def load_mlr_parquet(path_str: str) -> pd.DataFrame:
    """Load MLR predictions parquet with Date + Training Infer NaNs + bands + Basin."""
    path = Path(path_str)
    df = pd.read_parquet(path)
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

def pick_drop_predict_values(values: list) -> tuple:
    """
    Best-effort: find two categories corresponding to 'drop' and 'predict' in the
    Training Infer NaNs column. If not found, return first two unique values.
    """
    vals = [v for v in values if pd.notna(v)]
    if not vals:
        raise ValueError("No values found in 'Training Infer NaNs'.")

    lower = {v: str(v).lower() for v in vals}
    drop = None
    pred = None
    for v, s in lower.items():
        if "drop" in s:
            drop = v
        if "pred" in s or "infer" in s or "fill" in s:
            pred = v

    uniq = []
    for v in vals:
        if v not in uniq:
            uniq.append(v)

    if drop is None or pred is None or drop == pred:
        if len(uniq) >= 2:
            return uniq[0], uniq[1]
        return uniq[0], uniq[0]

    return drop, pred

def common_window_from_series(series_list: list[pd.Series]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Intersection window across all series based on non-NaN coverage."""
    mins, maxs = [], []
    for s in series_list:
        s2 = s.dropna()
        if len(s2) == 0:
            continue
        mins.append(s2.index.min())
        maxs.append(s2.index.max())
    if not mins or not maxs:
        raise ValueError("Cannot compute common date window (one or more series is empty).")
    return max(mins), min(maxs)

def extract_wy_from_name(filename: str) -> int | None:
    """
    Extract water year from names containing 'wyYYYY' (case-insensitive).
    """
    m = re.search(r"wy(\d{4})", filename, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

def available_wys_for_selection(
    basin_root: Path,
    seasonal_dir: str,
    mlr_suffix: str,
    units_choice: str,
    cfg: dict,
) -> list[int]:
    """
    Find WYs available for this basin/seasonal/units combination by scanning:
      - MLR prediction_{suffix}_wyYYYY.parquet in mlr_prediction/{seasonal_dir}/
      - UASWE mean_swe_uaswe_{m|acreFt}_wyYYYY.parquet
      - SNODAS mean_swe_snodas_{m|acreFt}_wyYYYY.parquet

    Returns intersection set (WYs present in all three sources), sorted.
    """
    # MLR files
    mlr_dir = basin_root / "mlr_prediction" / seasonal_dir
    mlr_wys = set()
    if mlr_dir.exists():
        pat = re.compile(rf"^prediction_{re.escape(mlr_suffix)}_wy\d{{4}}\.parquet$", re.IGNORECASE)
        for p in mlr_dir.glob("*.parquet"):
            if pat.match(p.name):
                wy = extract_wy_from_name(p.name)
                if wy:
                    mlr_wys.add(wy)

    # UASWE/SNODAS files depend on units config
    uaswe_pat = (
        re.compile(r"^mean_swe_uaswe_m_wy\d{4}\.parquet$", re.IGNORECASE)
        if cfg["uaswe_kind"] == "depth_m"
        else re.compile(r"^mean_swe_uaswe_acreFt_wy\d{4}\.parquet$", re.IGNORECASE)
    )
    snodas_pat = (
        re.compile(r"^mean_swe_snodas_m_wy\d{4}\.parquet$", re.IGNORECASE)
        if cfg["snodas_kind"] == "depth_m"
        else re.compile(r"^mean_swe_snodas_acreFt_wy\d{4}\.parquet$", re.IGNORECASE)
    )

    uaswe_dir = basin_root / "uaswe"
    snodas_dir = basin_root / "snodas"

    uaswe_wys = set()
    if uaswe_dir.exists():
        for p in uaswe_dir.glob("*.parquet"):
            if uaswe_pat.match(p.name):
                wy = extract_wy_from_name(p.name)
                if wy:
                    uaswe_wys.add(wy)

    snodas_wys = set()
    if snodas_dir.exists():
        for p in snodas_dir.glob("*.parquet"):
            if snodas_pat.match(p.name):
                wy = extract_wy_from_name(p.name)
                if wy:
                    snodas_wys.add(wy)

    common = mlr_wys & uaswe_wys & snodas_wys
    return sorted(common)

# -----------------------------
# Constants / maps
# -----------------------------
GRID_BAND_COLS = [
    "<7000", "7000-8000", "8000-9000",
    "9000-10000", "10000-11000", "11000-12000",
    ">12000", "total"
]

SEASONAL_DIR_MAP = {
    "season": "season",
    "accumulation": "accum",
    "melt": "melt",
}

UNITS_CFG = {
    "milimeters": {
        "mlr_suffix": "mm",
        "y_label": "SWE [mm]",
        "uaswe_kind": "depth_m",   # read *_m_ and convert to mm
        "snodas_kind": "depth_m",
        "other_scale": 1000.0,     # m -> mm
        "other_note": "UASWE/SNODAS converted from meters → mm.",
    },
    "thousand acre-ft": {
        "mlr_suffix": "acreFt",    # MLR file in thousand acre-ft (per your note)
        "y_label": "SWE [Thousand Acre-Ft]",
        "uaswe_kind": "acreft",    # read *_acreFt_ then /1000 to thousand acre-ft
        "snodas_kind": "acreft",
        "other_scale": 1.0 / 1000.0,  # acre-ft -> thousand acre-ft
        "other_note": "UASWE/SNODAS converted from acre-ft → thousand acre-ft.",
    },
}

# -----------------------------
# Top controls
# -----------------------------
top_left, top_mid, top_right = st.columns([1.2, 1.2, 1.4])
with top_left:
    band = st.selectbox(
        "Elevation Band",
        ["total", "<7k", "7k-8k", "8k-9k", "9k-10k", "10k-11k", "11k-12k", ">12k"],
        index=0,
    )
with top_mid:
    seasonal_choice = st.selectbox("Seasonal", ["season", "accumulation", "melt"], index=0)
with top_right:
    units_choice = st.selectbox("Units", ["milimeters", "thousand acre-ft"], index=0)

# -----------------------------
# Sidebar controls
# -----------------------------
basins = list_basins(DATA_ROOT)
if not basins:
    st.error(f"No basins found under: {DATA_ROOT.resolve()}")
    st.stop()

with st.sidebar:
    st.header("Basin")
    default_basin = "USCASJ" if "USCASJ" in basins else basins[0]
    basin = st.selectbox("Basin", basins, index=basins.index(default_basin))
    st.divider()
    show_table = st.checkbox("Show merged table", value=False)

basin_root = DATA_ROOT / basin
seasonal_dir = SEASONAL_DIR_MAP[seasonal_choice]
cfg = UNITS_CFG[units_choice]

# -----------------------------
# WY dropdown (auto-detect)
# -----------------------------
wys = available_wys_for_selection(
    basin_root=basin_root,
    seasonal_dir=seasonal_dir,
    mlr_suffix=cfg["mlr_suffix"],
    units_choice=units_choice,
    cfg=cfg,
)

if not wys:
    st.error(
        f"No common WYs found for {basin} / {seasonal_choice} / {units_choice}.\n\n"
        "Expected patterns:\n"
        f"- {basin_root}/mlr_prediction/{seasonal_dir}/prediction_{cfg['mlr_suffix']}_wyYYYY.parquet\n"
        f"- {basin_root}/uaswe/mean_swe_uaswe_{'m' if cfg['uaswe_kind']=='depth_m' else 'acreFt'}_wyYYYY.parquet\n"
        f"- {basin_root}/snodas/mean_swe_snodas_{'m' if cfg['snodas_kind']=='depth_m' else 'acreFt'}_wyYYYY.parquet\n"
    )
    st.stop()

default_wy = max(wys)
wy = st.selectbox("Water Year", wys, index=wys.index(default_wy))

# -----------------------------
# Resolve filepaths for selected WY
# -----------------------------
PATH_MLR = basin_root / f"mlr_prediction/{seasonal_dir}/prediction_{cfg['mlr_suffix']}_wy{wy}.parquet"

PATH_UASWE = basin_root / (
    f"uaswe/mean_swe_uaswe_m_wy{wy}.parquet"
    if cfg["uaswe_kind"] == "depth_m"
    else f"uaswe/mean_swe_uaswe_acreFt_wy{wy}.parquet"
)
PATH_SNODAS = basin_root / (
    f"snodas/mean_swe_snodas_m_wy{wy}.parquet"
    if cfg["snodas_kind"] == "depth_m"
    else f"snodas/mean_swe_snodas_acreFt_wy{wy}.parquet"
)

missing = [p for p in [PATH_MLR, PATH_UASWE, PATH_SNODAS] if not p.exists()]
if missing:
    st.error(
        "Missing expected files for this selection:\n\n"
        + "\n".join([f"- {p.as_posix()}" for p in missing])
    )
    st.stop()

# -----------------------------
# Load data (cached)
# -----------------------------
with st.spinner(f"Loading data for {basin} (WY{wy}, {seasonal_choice}, {units_choice})..."):
    df_mlr_raw = load_mlr_parquet(str(PATH_MLR))
    df_uaswe = load_timeseries_parquet(str(PATH_UASWE))
    df_snodas = load_timeseries_parquet(str(PATH_SNODAS))

# -----------------------------
# Resolve columns
# -----------------------------
grid_col_map = {normalize_band_name(c): c for c in GRID_BAND_COLS if c in df_uaswe.columns}
if band not in grid_col_map:
    st.error(
        f"Band '{band}' not found in UASWE/SNODAS columns. "
        f"Available: {sorted(grid_col_map.keys())}"
    )
    st.stop()

mlr_col = "Basin" if band == "total" else band
if mlr_col not in df_mlr_raw.columns:
    st.error(f"MLR column '{mlr_col}' not found in MLR parquet.")
    st.stop()

if "Training Infer NaNs" not in df_mlr_raw.columns:
    st.error("MLR parquet is missing required column: 'Training Infer NaNs'")
    st.stop()

# -----------------------------
# Build series (MLR drop vs predict)
# -----------------------------
unique_nan_modes = df_mlr_raw["Training Infer NaNs"].dropna().unique().tolist()
drop_mode, pred_mode = pick_drop_predict_values(unique_nan_modes)

s_mlr_drop = (
    df_mlr_raw[df_mlr_raw["Training Infer NaNs"] == drop_mode][["Date", mlr_col]]
    .set_index("Date")[mlr_col]
    .sort_index()
    .rename("drop NaNs")
)
s_mlr_pred = (
    df_mlr_raw[df_mlr_raw["Training Infer NaNs"] == pred_mode][["Date", mlr_col]]
    .set_index("Date")[mlr_col]
    .sort_index()
    .rename("predict NaNs")
)

# Other models
s_uaswe = df_uaswe[grid_col_map[band]].copy().rename("UASWE")
s_snodas = df_snodas[grid_col_map[band]].copy().rename("SNODAS")

# Apply unit conversions for other models only (MLR already in chosen display units)
s_uaswe = s_uaswe * cfg["other_scale"]
s_snodas = s_snodas * cfg["other_scale"]

# -----------------------------
# Align by common time coverage (intersection window)
# -----------------------------
t_min, t_max = common_window_from_series([s_mlr_drop, s_mlr_pred, s_uaswe, s_snodas])
s_mlr_drop = s_mlr_drop.loc[t_min:t_max]
s_mlr_pred = s_mlr_pred.loc[t_min:t_max]
s_uaswe = s_uaswe.loc[t_min:t_max]
s_snodas = s_snodas.loc[t_min:t_max]

st.caption(f"Plot window (intersection): {t_min.date()} to {t_max.date()}")

# -----------------------------
# Plotly (interactive) with grouped legends + linestyles
# -----------------------------
fig = go.Figure()

# MLR group
fig.add_trace(
    go.Scatter(
        x=s_mlr_drop.index, y=s_mlr_drop.values,
        mode="lines",
        name="drop NaNs",
        legendgroup="MLR",
        legendgrouptitle_text="MLR Prediction",
        line=dict(dash="solid"),
        hovertemplate="Date=%{x|%Y-%m-%d}<br>Value=%{y:.2f}<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=s_mlr_pred.index, y=s_mlr_pred.values,
        mode="lines",
        name="predict NaNs",
        legendgroup="MLR",
        line=dict(dash="dash"),
        hovertemplate="Date=%{x|%Y-%m-%d}<br>Value=%{y:.2f}<extra></extra>",
    )
)

# Other models group
fig.add_trace(
    go.Scatter(
        x=s_uaswe.index, y=s_uaswe.values,
        mode="lines",
        name="UASWE",
        legendgroup="OTHER",
        legendgrouptitle_text="Other Models",
        line=dict(dash="dot"),
        hovertemplate="Date=%{x|%Y-%m-%d}<br>Value=%{y:.2f}<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=s_snodas.index, y=s_snodas.values,
        mode="lines",
        name="SNODAS",
        legendgroup="OTHER",
        line=dict(dash="dashdot"),
        hovertemplate="Date=%{x|%Y-%m-%d}<br>Value=%{y:.2f}<extra></extra>",
    )
)

fig.update_layout(
    title=dict(
        text=f"{basin} – WY{wy} – {seasonal_choice} – {band}",
        x=0.5,
        xanchor="center",
        y=0.98,
        yanchor="top",
        font=dict(size=20),
    ),
    xaxis_title="Date",
    yaxis_title=cfg["y_label"],
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="top",
        y=0.92,
        xanchor="center",
        x=0.5,
        title_text=None,
    ),
    margin=dict(l=20, r=20, t=140, b=20),
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Optional merged table
# -----------------------------
if show_table:
    df_plot = pd.concat(
        [
            s_mlr_drop.rename(f"MLR drop NaNs ({cfg['y_label']})"),
            s_mlr_pred.rename(f"MLR predict NaNs ({cfg['y_label']})"),
            s_uaswe.rename(f"UASWE ({cfg['y_label']})"),
            s_snodas.rename(f"SNODAS ({cfg['y_label']})"),
        ],
        axis=1,
    )
    st.subheader("Merged data")
    st.dataframe(df_plot, use_container_width=True)
