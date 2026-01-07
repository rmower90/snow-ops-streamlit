from __future__ import annotations

from pathlib import Path
import yaml
import xarray as xr


def load_cfg_for_basin(basin: str, config_dir: str | Path) -> dict:
    config_dir = Path(config_dir)
    cfg_path = config_dir / f"{basin}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing basin config: {cfg_path}")
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


def load_aso_metadata(basin: str, config_dir: str | Path) -> tuple[dict, dict]:
    """
    Reads basin YAML and returns (cfg, paths) with keys expected by preprocess.py.
    Matches your YAML keys: aso_spatial, aso_temporal, aso_demBin, aso_shape.
    """
    cfg = load_cfg_for_basin(basin, config_dir)
    fp = cfg.get("data_filepaths", {})

    required = ["aso_spatial", "aso_temporal"]
    missing = [k for k in required if not fp.get(k)]
    if missing:
        raise KeyError(f"Missing required YAML keys in data_filepaths: {missing}")

    paths = {
        "aso_spatial_fpath": fp["aso_spatial"],
        "aso_tseries_fpath": fp["aso_temporal"],
        "demBin_fpath": fp.get("aso_demBin"),
        "shape_fpath": fp.get("aso_shape"),
        "snowmodel_dir": fp.get("snowmodel_dir"),
        "snodas_dir": fp.get("snodas_dir"),
        "insitu_dir": fp.get("insitu_dir"),
        "mlrPred_dir": fp.get("mlrPred_dir"),
        "shape_crs": f"EPSG:{cfg['crs']['epsg']}" if cfg.get("crs", {}).get("epsg") else None,
    }
    return cfg, paths


def load_aso_data(
    aso_spatial_fpath: str,
    aso_tseries_fpath: str,
    demBin_fpath: str | None,
    shape_fpath: str | None,
    shape_crs=None,
) -> tuple[xr.Dataset, xr.Dataset | None, object, xr.Dataset]:
    """
    Minimal for investigation page:
      - always load ASO spatial + tseries
      - demBin is OPTIONAL (can be broken locally; not needed for line~286)
      - shape is OPTIONAL (geopandas not required yet)
    """
    aso_spatial_ds = xr.load_dataset(aso_spatial_fpath)
    aso_tseries_ds = xr.load_dataset(aso_tseries_fpath)

    dem_bin_ds = None
    if demBin_fpath:
        p = Path(demBin_fpath)
        if p.exists():
            try:
                dem_bin_ds = xr.load_dataset(str(p))
            except Exception as e:
                # Don’t block preprocessing for now
                dem_bin_ds = None
                # attach a helpful attribute for debugging (optional)
                aso_spatial_ds.attrs["demBin_load_error"] = f"{type(e).__name__}: {e}"
        else:
            dem_bin_ds = None

    shape_proj_gdf = None
    _ = shape_fpath, shape_crs

    return aso_spatial_ds, dem_bin_ds, shape_proj_gdf, aso_tseries_ds
