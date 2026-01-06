"""
MLR preprocessing utilities.

This module will eventually contain all preprocessing logic needed
to reproduce MLR training and inference runs for the Streamlit app.

For now, this is a scaffold.
"""

from __future__ import annotations

from typing import Dict, Any


def run_preprocessing_only(
    basin: str,
    water_year: int,
    seasonal: str,
    band: str,
    *,
    config_dir: str | None = None,
    exclude_pillows: list[str] | None = None,
) -> Dict[str, Any]:
    """
    Run MLR preprocessing steps up to feature construction / imputation.

    Parameters
    ----------
    basin : str
        Basin ID (e.g., "USCASJ")
    water_year : int
        Water year to target (e.g., 2026)
    seasonal : str
        "season", "accumulation", or "melt"
    band : str
        Elevation band label ("total", "7k-8k", etc.)
    config_dir : str, optional
        Path to basin config directory
    exclude_pillows : list[str], optional
        Pillow IDs to exclude

    Returns
    -------
    dict
        Dictionary of intermediate preprocessing artifacts
        (df_sum_total, drop_na_df, obs_data, etc.)
    """
    raise NotImplementedError(
        "MLR preprocessing not wired yet. "
        "This function will be implemented incrementally."
    )
