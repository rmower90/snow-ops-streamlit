import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone

st.set_page_config(
    page_title="Snow Ops",
    page_icon="❄️",
    layout="wide",
)


st.title("Snow Ops — Proof of Concept")

st.caption("Goal: validate layout + data loading + plots before deploying to hydro-c1-web.")

# Example “status” banner like you’ll want in production
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
st.info(f"POC running. Data last updated: {now}")

# Sidebar controls (you’ll keep this pattern)
with st.sidebar:
    st.header("Controls")
    basin = st.selectbox("Basin", ["USCASJ", "USCATM", "USCOBR", "USCOGE"])
    wy = st.selectbox("Water Year", [2023, 2024, 2025])
    n = st.slider("Mock station count", 10, 200, 50)

# Mock data stand-in (replace with real pillow/basin data later)
rng = np.random.default_rng(0)
df = pd.DataFrame({
    "station": [f"S{i:03d}" for i in range(n)],
    "swe_mm": rng.normal(500, 150, size=n).clip(0),
    "qc_flag": rng.integers(0, 2, size=n),
})

c1, c2, c3 = st.columns(3)
c1.metric("Basin", basin)
c2.metric("Water Year", wy)
c3.metric("Stations", len(df))

st.subheader("Station table")
st.dataframe(df, use_container_width=True)

st.subheader("SWE distribution")
st.bar_chart(df["swe_mm"].value_counts(bins=20).sort_index())
