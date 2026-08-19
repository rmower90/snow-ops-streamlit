"""
sensor_ab_plot.py

Stage 0 reporting for the CDEC sensor A/B investigation. Consumes the three arms written
by sensor_ab_download.py and produces the artifacts that decide whether the downstream
pipeline work (stage 1: zarr -> QA -> MLR) is worth running.

    PROD82  frozen production baseline   what the pipeline actually consumed
    BULK82  bulk refetch of sensor 82    BULK82 - PROD82 = revision / latency effect
    TEST    bulk fetch of sensor 3       TEST   - BULK82 = pure sensor effect

Keeping those two effects apart is the whole point of the third arm: production was built
one day at a time, each day fetched ~24h later, while a bulk refetch pulls values CDEC has
since revised. A straight PROD82-vs-TEST diff cannot tell the two apart.

Outputs (to --out-dir):
    sensor_ab_timeseries.png   30-panel small multiples, three arms per pillow
    sensor_ab_delta.png        30-panel small multiples, the two deltas per pillow
    sensor_ab_summary.csv      per-pillow coverage and delta statistics

Statistics are reported twice: over all days, and over "active" days only (those outside
the manual exclusion windows in the region config). Active days are the ones that matter
-- everything inside an exclusion window is dropped downstream regardless of sensor.

Usage (conda env: bor):
    python sensor_ab_plot.py USCASJ
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(1, str(Path(__file__).resolve().parent.parent))

import metadata as metadata


DEFAULT_DATA_DIR = "/home/rossamower/work/aso/data_test/insitu/USCASJ/raw/"
DEFAULT_OUT_DIR = "/home/rossamower/work/aso/data_test/insitu/USCASJ/sensor_ab/"

# Categorical slots 1-3 of the validated default palette. Small multiples are an
# "all-pairs" form, where the documented series cap is three -- which is exactly what we
# need. Validated all-pairs, light mode: worst CVD dE 9.2, worst normal-vision dE 24.0.
C_PROD82 = "#2a78d6"   # slot 1, blue
C_BULK82 = "#eb6834"   # slot 2, orange
C_TEST = "#1baf7a"     # slot 3, aqua

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
BAND = "#d8d7d2"       # manual exclusion shading: neutral, never a series hue

ARMS = [
    ("PROD82", C_PROD82, "-", "PROD82  (production, sensor 82 w/ 3 fallback)"),
    ("BULK82", C_BULK82, "--", "BULK82  (refetch, sensor 82 only)"),
    ("TEST", C_TEST, "-", "TEST    (refetch, sensor 3 only)"),
]


def style_axis(ax):
    """Recessive grid and axes; series carry the ink, the frame does not."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.grid(True, color=INK_MUTED, alpha=0.22, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=6, length=2, width=0.6)
    # quarterly ticks only -- 30 narrow panels cannot carry a dense date axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(10, 1, 4, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))


def panel_label(r):
    """
    Headline for a panel. The bias is only defined where both sensors reported, so say
    which sensor is missing rather than rendering a NaN.
    """
    if r["act_n_test3"] == 0 and r["act_n_bulk82"] == 0:
        return "no active data"
    if r["act_n_test3"] == 0:
        return "sensor 3 absent"
    if r["act_n_bulk82"] == 0:
        return "sensor 82 absent"
    if r["act_both"] == 0:
        return "no overlapping days"
    return f"bias {r['sensor_bias_mm']:+.1f}mm"


def load_arms(data_dir, aso_site_name):
    arms = {}
    for arm, _, _, _ in ARMS:
        path = Path(data_dir) / f"{aso_site_name}_insitu_obs_daily_wy_2026_{arm}.nc"
        if not path.exists():
            raise SystemExit(f"Missing arm {arm}: {path}")
        arms[arm] = xr.load_dataset(path)
    return arms


def to_frames(arms):
    """
    One DataFrame per arm, index=time, columns=pillow, on the intersection of the arms'
    time axes so every comparison is like-for-like.
    """
    common = None
    for ds in arms.values():
        t = pd.DatetimeIndex(pd.to_datetime(ds["time"].values)).normalize()
        common = t if common is None else common.intersection(t)

    frames = {}
    for arm, ds in arms.items():
        df = ds.to_dataframe()
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).normalize()
        frames[arm] = df.reindex(common).sort_index()
    return frames, common


def manual_window_mask(cfg, pillows, index):
    """
    Boolean frame, True where a pillow-day sits inside a manual exclusion window from the
    region config. Those days are dropped downstream regardless of sensor, so they are
    excluded from the 'active' statistics and shaded on the plots.
    """
    excl = cfg["pillow_api"].get("exclude_pillows") or []
    ranges = cfg["pillow_api"].get("exclude_pillows_ranges") or []

    mask = pd.DataFrame(False, index=index, columns=pillows)
    for p in excl:
        if p in mask.columns:
            mask[p] = True
    for r in ranges:
        p = r["pillow"]
        if p not in mask.columns:
            continue
        lo, hi = pd.Timestamp(str(r["start"])), pd.Timestamp(str(r["end"]))
        mask.loc[(index >= lo) & (index <= hi), p] = True
    return mask


def summarize(frames, excl_mask, pillows):
    prod, bulk, test = frames["PROD82"], frames["BULK82"], frames["TEST"]
    rows = []

    for p in pillows:
        active = ~excl_mask[p]
        s_prod, s_bulk, s_test = prod[p], bulk[p], test[p]

        # coverage on active days
        a_prod = s_prod[active].notna()
        a_bulk = s_bulk[active].notna()
        a_test = s_test[active].notna()

        both = a_bulk & a_test
        only82 = a_bulk & ~a_test        # coverage strict sensor 3 would give up
        only3 = a_test & ~a_bulk         # days production ALREADY served sensor 3 silently

        # pure sensor effect, on days both sensors reported
        d_sensor = (s_test - s_bulk)[active][both]
        # revision effect, on days both the production file and the refetch have a value
        rev_pair = a_prod & a_bulk
        d_rev = (s_bulk - s_prod)[active][rev_pair]

        rows.append({
            "pillow": p,
            "manual_excl_days": int(excl_mask[p].sum()),
            "active_days": int(active.sum()),
            "n_prod82": int(s_prod.notna().sum()),
            "n_bulk82": int(s_bulk.notna().sum()),
            "n_test3": int(s_test.notna().sum()),
            "act_n_prod82": int(a_prod.sum()),
            "act_n_bulk82": int(a_bulk.sum()),
            "act_n_test3": int(a_test.sum()),
            "act_both": int(both.sum()),
            "act_only82": int(only82.sum()),
            "act_only3": int(only3.sum()),
            "sensor_bias_mm": _r(d_sensor.mean()),
            "sensor_mean_abs_mm": _r(d_sensor.abs().mean()),
            "sensor_med_abs_mm": _r(d_sensor.abs().median()),
            "sensor_max_abs_mm": _r(d_sensor.abs().max()),
            "sensor_days_gt_10mm": int((d_sensor.abs() > 10).sum()),
            "revision_mean_abs_mm": _r(d_rev.abs().mean()),
            "revision_max_abs_mm": _r(d_rev.abs().max()),
        })

    return pd.DataFrame(rows)


def _r(v, nd=2):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)


def grid_shape(n, ncols=5):
    return int(np.ceil(n / ncols)), ncols


def shade_windows(ax, excl_col, index):
    """Shade contiguous manual-exclusion runs."""
    if not excl_col.any():
        return
    grp = (excl_col != excl_col.shift(fill_value=False)).cumsum()
    for _, run in excl_col[excl_col].groupby(grp[excl_col]):
        ax.axvspan(run.index[0], run.index[-1], color=BAND, alpha=0.55, lw=0, zorder=0)


def plot_timeseries(frames, excl_mask, pillows, summary, out_path):
    nrows, ncols = grid_shape(len(pillows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.1, nrows * 2.0),
                             sharex=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    index = frames["PROD82"].index
    srow = summary.set_index("pillow")

    for ax, p in zip(axes, pillows):
        style_axis(ax)
        shade_windows(ax, excl_mask[p], index)
        for arm, color, ls, _ in ARMS:
            ax.plot(index, frames[arm][p].values, color=color, linestyle=ls,
                    linewidth=1.1, solid_capstyle="round")

        # direct label per panel: pillow code plus the sensor-effect headline. this is the
        # 'relief' the palette validator requires for the aqua slot's contrast warning.
        r = srow.loc[p]
        ax.set_title(f"{p}   {panel_label(r)}", fontsize=7.5, color=INK_PRIMARY,
                     loc="left", pad=3)
        if r["manual_excl_days"] == len(index):
            ax.text(0.5, 0.5, "manually excluded\n(full WY)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6.5, color=INK_SECONDARY)

    for ax in axes[len(pillows):]:
        ax.set_visible(False)

    handles = [plt.Line2D([], [], color=c, linestyle=ls, linewidth=1.6, label=lab)
               for _, c, ls, lab in ARMS]
    handles.append(plt.Line2D([], [], color=BAND, linewidth=8, alpha=0.55,
                              label="manual exclusion window (config)"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.008, 0.995),
               ncol=4, frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.suptitle("CDEC sensor A/B — daily SWE by pillow, WY2026 (mm)",
                 x=0.008, ha="left", y=1.0, fontsize=11, color=INK_PRIMARY)
    fig.supylabel("SWE (mm)", fontsize=8, color=INK_SECONDARY)
    fig.tight_layout(rect=[0.012, 0, 1, 0.955])
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_deltas(frames, excl_mask, pillows, out_path):
    nrows, ncols = grid_shape(len(pillows))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.1, nrows * 2.0),
                             sharex=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    index = frames["PROD82"].index

    d_rev_all = frames["BULK82"] - frames["PROD82"]
    d_sen_all = frames["TEST"] - frames["BULK82"]

    for ax, p in zip(axes, pillows):
        style_axis(ax)
        shade_windows(ax, excl_mask[p], index)
        ax.axhline(0, color=INK_MUTED, linewidth=0.7, zorder=1)
        ax.plot(index, d_rev_all[p].values, color=C_BULK82, linewidth=1.1,
                solid_capstyle="round")
        ax.plot(index, d_sen_all[p].values, color=C_TEST, linewidth=1.1,
                solid_capstyle="round")
        ax.set_title(p, fontsize=7.5, color=INK_PRIMARY, loc="left", pad=3)

    for ax in axes[len(pillows):]:
        ax.set_visible(False)

    handles = [
        plt.Line2D([], [], color=C_BULK82, linewidth=1.6,
                   label="revision effect   BULK82 − PROD82"),
        plt.Line2D([], [], color=C_TEST, linewidth=1.6,
                   label="sensor effect     TEST − BULK82"),
        plt.Line2D([], [], color=BAND, linewidth=8, alpha=0.55,
                   label="manual exclusion window (config)"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.008, 0.995),
               ncol=3, frameon=False, fontsize=8, labelcolor=INK_SECONDARY)
    fig.suptitle("CDEC sensor A/B — decomposed daily deltas, WY2026 (mm)",
                 x=0.008, ha="left", y=1.0, fontsize=11, color=INK_PRIMARY)
    fig.supylabel("Δ SWE (mm)", fontsize=8, color=INK_SECONDARY)
    fig.tight_layout(rect=[0.012, 0, 1, 0.955])
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aso_site_name")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--config-dir", default="/home/rossamower/work/aso/configs/")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = metadata.load_yaml(
        Path(f"{args.config_dir}regions/{args.aso_site_name}.yaml"))

    arms = load_arms(args.data_dir, args.aso_site_name)
    frames, index = to_frames(arms)
    pillows = sorted(set.intersection(*(set(f.columns) for f in frames.values())))

    print(f"SENSOR A/B REPORT: {args.aso_site_name}")
    print(f"  common range: {index.min().date()} -> {index.max().date()} "
          f"({len(index)} days), {len(pillows)} pillows\n")

    excl_mask = manual_window_mask(cfg, pillows, index)
    summary = summarize(frames, excl_mask, pillows)

    csv_path = out_dir / "sensor_ab_summary.csv"
    summary.to_csv(csv_path, index=False)

    plot_timeseries(frames, excl_mask, pillows, summary,
                    out_dir / "sensor_ab_timeseries.png")
    plot_deltas(frames, excl_mask, pillows, out_dir / "sensor_ab_delta.png")

    # console view: the columns that drive the stage-1 go/no-go.
    cols = ["pillow", "active_days", "act_n_bulk82", "act_n_test3", "act_both",
            "act_only82", "act_only3", "sensor_bias_mm", "sensor_mean_abs_mm",
            "sensor_max_abs_mm", "sensor_days_gt_10mm", "revision_mean_abs_mm"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary[cols].to_string(index=False))

    fully_excluded = summary[summary["manual_excl_days"] == len(index)]["pillow"].tolist()
    print(f"\nfully excluded by config ({len(fully_excluded)}): "
          f"{', '.join(fully_excluded) if fully_excluded else 'none'}")
    print(f"\nwrote {csv_path}")
    print(f"wrote {out_dir / 'sensor_ab_timeseries.png'}")
    print(f"wrote {out_dir / 'sensor_ab_delta.png'}")


if __name__ == "__main__":
    main()
