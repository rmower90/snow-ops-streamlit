"""
sensor_ab_download.py

Stage 0 of the CDEC sensor A/B investigation.

Background: load_pillow_api() historically defaulted to sensor_priority=(82, 3) --
SWE_ADJ preferred, raw SWE filling its gaps ELEMENT-WISE, so a single station's series
could silently mix the two sensors across time. A CDEC data specialist advised that
sensor 3 (raw SWE) is the series that actually receives QA/QC, so it -- not 82 -- should
be the one we ingest.

This script re-downloads the full water year into a parallel data root so the change can
be evaluated without touching production. It is a non-destructive, range-parameterized
clone of insitu_basin_download.py's main block: same buffered geometry, same SNOTEL leg,
same combine/rename/pad/sort tail. Differences:
  - date range comes from the frozen production baseline (or --start/--end), not the
    daily checklist
  - sensor_priority is exposed per arm
  - writes to --out-dir under a per-arm filename
  - performs NO checklist writes, so re-running is idempotent

Three arms make the comparison interpretable. A straight prod-vs-sensor-3 diff would
confound two separate changes, because production was built incrementally (each day
fetched ~24h later) while a bulk refetch pulls values CDEC has since revised -- and
revision is precisely what "sensor 3 is the QA/QC'd one" means:

    PROD82  frozen copy of production        what the pipeline actually consumed
    BULK82  sensor_priority=(82,)            BULK82 - PROD82 = revision/latency effect
    TEST    sensor_priority=(3,)             TEST   - BULK82 = pure sensor effect

Both fetched arms are PURE (single sensor, no fallback), so any blended policy --
(82,3), (3,82) -- can be reconstructed offline without re-hitting CDEC.

Usage (conda env: bor):
    python sensor_ab_download.py USCASJ
    python sensor_ab_download.py USCASJ --arms TEST
    python sensor_ab_download.py USCASJ --start 2025-10-01 --end 2026-08-03
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(1, str(Path(__file__).resolve().parent.parent))

import metadata as metadata
import shape as shape
import pillow_api as pillow_api


# arm name -> sensor_priority passed to load_pillow_api.
# both are single-sensor so the arms stay pure and blends remain derivable offline.
ARMS = {
    "BULK82": (82,),
    "TEST": (3,),
}

DEFAULT_OUT_DIR = "/home/rossamower/work/aso/data_test/insitu/USCASJ/raw/"
BASELINE_SUFFIX = "PROD82"
# production applies this buffer before asking CDEC which stations are in range.
BUFFER_SIZE_M = 25000


def load_cfg(aso_site_name, config_dir="/home/rossamower/work/aso/configs/"):
    return metadata.load_yaml(Path(f"{config_dir}regions/{aso_site_name}.yaml"))


def out_path(out_dir, aso_site_name, arm):
    return Path(out_dir) / f"{aso_site_name}_insitu_obs_daily_wy_2026_{arm}.nc"


def resolve_date_range(args, out_dir, aso_site_name):
    """
    Derive the fetch window from the frozen production baseline so every arm lands on an
    identical time axis by construction. Explicit --start/--end override.
    """
    if args.start and args.end:
        return pd.Timestamp(args.start), pd.Timestamp(args.end)

    baseline = out_path(out_dir, aso_site_name, BASELINE_SUFFIX)
    if not baseline.exists():
        raise SystemExit(
            f"No frozen baseline at {baseline} and no --start/--end given.\n"
            f"Either copy production there first, or pass the range explicitly."
        )
    with xr.open_dataset(baseline) as ds:
        times = pd.to_datetime(ds["time"].values)
    start = pd.Timestamp(times.min()) if args.start is None else pd.Timestamp(args.start)
    end = pd.Timestamp(times.max()) if args.end is None else pd.Timestamp(args.end)
    return start, end


def fetch_snotel(cfg, aso_site_name, buff_geog, cdec_locations, start_date, end_date):
    """
    SNOTEL leg. Sensor-independent -- CDEC sensor 82/3 does not apply -- so it is fetched
    once and reused across arms. That also makes LLE a free control: it should be
    identical in every arm, and if it is not, the harness is wrong.
    """
    if aso_site_name == "USCASJ":
        # production hardcodes this branch: LLE sits outside the buffered basin, so the
        # SNOTEL query uses a bespoke polygon rather than the basin geometry.
        geog = pillow_api.load_lle_geography()
    else:
        geog = buff_geog

    return pillow_api.load_snotel_api(
        geog,
        start_date,
        end_date,
        cdec_locations.dropna(axis=0),
        variable="SNOTEL:WTEQ_D",
        plot=False,
    )


def build_dataset(cfg, buff_geog, cdec_locations, snotel_locations, obs_data_pil, obs_data_snot):
    """
    combine/rename/pad/sort tail, mirroring insitu_basin_download.py so the output stays a
    drop-in for the rest of the pipeline.
    """
    obs_data_2, obs_locations = pillow_api.combine_basin_obs(
        buff_geog,
        cdec_locations.dropna(axis=0),
        snotel_locations,
        obs_data_pil,
        obs_data_snot,
    )

    ds = None
    for pil in obs_data_2:
        if ds is None:
            ds = pil.to_dataset()
        else:
            ds[pil.name] = pil

    # SNOTEL station -> basin pillow code (e.g. LEAVITT LAKE -> LLE).
    name_rename = cfg["pillow_api"]["snotel_rename"]["name"]
    network_rename = cfg["pillow_api"]["snotel_rename"]["network"]
    id_rename = cfg["pillow_api"]["snotel_rename"]["id"]
    match = obs_locations[
        (obs_locations["name"] == name_rename) & (obs_locations["network"] == network_rename)
    ]["id"]
    if len(match):
        ds = ds.rename({match.values[0]: id_rename})
    else:
        print(f"  WARNING: no {network_rename} station named {name_rename}; "
              f"{id_rename} will be padded with NaN")

    # pad any configured pillow the fetch did not return, so every arm carries the same
    # variable set even when the two sensors cover different station lists.
    all_pils = cfg["pillow_api"]["pil_names"].split(",")
    downloaded = list(ds.data_vars.keys())
    for pil_name in [i for i in all_pils if i not in downloaded]:
        ds[pil_name] = xr.DataArray(
            np.nan * np.ones(ds["time"].shape),
            dims=["time"],
            coords={"time": ds["time"]},
            name=pil_name,
        )

    return ds[sorted(ds.data_vars)], downloaded


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aso_site_name")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS),
                    help="which arms to fetch (default: all)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--start", default=None, help="YYYY-MM-DD; default = baseline start")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD; default = baseline end")
    ap.add_argument("--config-dir", default="/home/rossamower/work/aso/configs/")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-fetch arms whose output file already exists")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg(args.aso_site_name, args.config_dir)
    start_date, end_date = resolve_date_range(args, out_dir, args.aso_site_name)
    n_days = (end_date - start_date).days + 1

    print(f"SENSOR A/B DOWNLOAD: {args.aso_site_name}")
    print(f"  range : {start_date.date()} -> {end_date.date()} ({n_days} days)")
    print(f"  arms  : {', '.join(f'{a}={ARMS[a]}' for a in args.arms)}")
    print(f"  out   : {out_dir}\n")

    _, _, aso_buff_gdf_geog, _ = shape.load_basin_shape(
        cfg["data_filepaths"]["aso_shape"],
        f'EPSG:{cfg["crs"]["epsg"]}',
        aso_site_name=None,
        buffer_size=BUFFER_SIZE_M,
    )

    snotel_cache = None

    for arm in args.arms:
        dest = out_path(out_dir, args.aso_site_name, arm)
        if dest.exists() and not args.overwrite:
            print(f"[{arm}] exists, skipping (use --overwrite to refetch): {dest}\n")
            continue

        sensor_priority = ARMS[arm]
        print(f"[{arm}] downloading CDEC, sensor_priority={sensor_priority} ...")
        obs_data_pil, cdec_locations, skipped, _ = pillow_api.load_pillow_api(
            aso_buff_gdf_geog,
            start_date,
            end_date,
            plot=False,
            debug=False,
            sensor_priority=sensor_priority,
        )
        print(f"[{arm}] CDEC returned {len(obs_data_pil)} station(s); "
              f"{len(skipped)} skipped")

        if snotel_cache is None:
            print(f"[{arm}] downloading SNOTEL (fetched once, shared across arms) ...")
            snotel_cache = fetch_snotel(
                cfg, args.aso_site_name, aso_buff_gdf_geog,
                cdec_locations, start_date, end_date,
            )
        # deep-copy so one arm's combine/rename cannot mutate the shared SNOTEL result.
        obs_data_snot, snotel_locations = copy.deepcopy(snotel_cache)

        ds, downloaded = build_dataset(
            cfg, aso_buff_gdf_geog, cdec_locations, snotel_locations,
            obs_data_pil, obs_data_snot,
        )

        ds.attrs["sensor_priority"] = ",".join(str(s) for s in sensor_priority)
        ds.attrs["arm"] = arm
        ds.attrs["fetch_range"] = f"{start_date.date()}/{end_date.date()}"

        ds.to_netcdf(dest)
        n_padded = len(ds.data_vars) - len(downloaded)
        print(f"[{arm}] wrote {dest}")
        print(f"[{arm}] {len(ds.data_vars)} vars x {ds.sizes['time']} days "
              f"({n_padded} padded to NaN)\n")

    print("DONE.")


if __name__ == "__main__":
    main()
