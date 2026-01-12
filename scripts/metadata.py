# metadata.py
"""
    get station metadata.
"""


from __future__ import annotations
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import xarray as xr

FT_TO_M = 0.3048

def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)

def get_forcing_indices(cfg_region: dict, forcing_str: str):
    """Pull indices based on forcing dataset"""
    idy_min = int(cfg_region[forcing_str]['idy_min'])
    idy_max = int(cfg_region[forcing_str]['idy_max'])
    idx_min = int(cfg_region[forcing_str]['idx_min'])
    idx_max = int(cfg_region[forcing_str]['idx_max'])
    return idy_min,idy_max,idx_min,idx_max

def bins_to_si(edges: list[float | str], unit: str) -> np.ndarray:
    """Convert edges (possibly with '-inf'/'inf') to meters."""
    arr = np.array(edges, dtype=float)
    if unit.lower() in ("ft", "feet"):
        arr = arr * FT_TO_M
    elif unit.lower() in ("m", "meter", "meters"):
        pass
    else:
        raise ValueError(f"Unknown elevation unit: {unit}")
    return arr

def get_elevation_bins(cfg_region: dict, to_unit: str = "m") -> tuple[np.ndarray, list[str]]:
    """Return (edges_in_meters, labels)."""
    bins = cfg_region["elevation"]["bins"]
    labels: list[str] = bins["labels"]
    edges = bins["edges"]
    unit = cfg_region["elevation"].get("unit", "m")
    edges_m = bins_to_si(edges, unit)
    return edges_m, labels

def cut_elevation(elev_da: xr.DataArray, edges_m: np.ndarray, labels: list[str]) -> xr.DataArray:
    """
    Bin an elevation DataArray (in meters) into categorical labels.
    Returns a DataArray of dtype 'category' aligned with elev_da.
    """
    # xr.apply_ufunc + np.digitize keeps lazy dask friendliness
    idx = xr.apply_ufunc(
        np.digitize, elev_da, edges_m, input_core_dims=[[], ["edge"]], kwargs={"right": False},
        dask="parallelized", output_dtypes=[int]
    )
    # np.digitize returns bin indices in 1..len(edges)-1; map to labels (0..len(labels)-1)
    label_index = idx - 1
    label_index = label_index.clip(0, len(labels)-1)

    # Attach labels via pandas Categorical for convenience
    cats = pd.Categorical.from_codes(label_index.values.ravel(), categories=labels, ordered=True)
    labeled = xr.DataArray(
        cats.reshape(label_index.shape),
        coords=elev_da.coords, dims=elev_da.dims, name=f"{elev_da.name}_bin" if elev_da.name else "elevation_bin"
    )
    return labeled

def bin_counts(elev_da_m: xr.DataArray, edges_m: np.ndarray, labels: list[str], dims: list[str] | None = None) -> pd.Series:
    """
    Count pixels (or points) per bin. If dims is None, count all elements.
    Returns a pandas Series indexed by bin label (ordered).
    """
    binned = cut_elevation(elev_da_m, edges_m, labels)
    # convert to DataFrame for groupby
    df = binned.to_series()
    counts = df.value_counts(sort=False).reindex(labels, fill_value=0)
    return counts




if __name__ =="__main__":
    if len(sys.argv) < 2:
        print("Usage: python directories.py <aso_site_name>")
        sys.exit(1)
    # get inputs.
    aso_site_name = sys.argv[1]
    # load geography.
    shape_geog_gdf = load_campaign_directories(aso_site_name)






