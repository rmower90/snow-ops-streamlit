"""
mlr_ab_harness.py

A/B instrument for changes to lm_model.py.

Every change planned for the uncertainty work touches lm_model.py, which is shared by
BOTH mlr_prediction.py (USCASJ daily, live website) and mlr_prediction_historic.py
(FRIANT v1-v6) via the /home/rossamower/bin/scripts symlink. So each change has to be
provably inert unless explicitly enabled. This harness is how that gets proved.

It calls only the two functions under change --

    lm_model.train_all_mlr_models()  ->  lm_model.predict_with_cached_training()

-- on a sampled set of timesteps, and dumps their results to a directory of its choosing.
Nothing production-owned is written, so it is safe to run against the live tree.

Two tests:

  fidelity     harness output for a date must match that date's row in the existing
               prediction_*_wy{YYYY}_combination.csv. Without this the harness could be
               self-consistently wrong and still "pass" an A/B.

  determinism  two runs on UNMODIFIED lm_model.py must be byte-identical. This is the
               gate: if A != A', no A/B comparison downstream means anything. The known
               risk is identify_best_stations' `station_lst = list(set(sum(...)))`, whose
               iteration order sets the combinations() enumeration order and therefore
               which combination wins an adjusted-R2 tie.

Usage (conda env: bor):
    # baseline, then again -- expect an empty diff
    python mlr_ab_harness.py FRIANT 1997 --label A
    python mlr_ab_harness.py FRIANT 1997 --label A2
    diff -r <out>/A <out>/A2

    # after editing lm_model.py
    python mlr_ab_harness.py FRIANT 1997 --label B
    diff -r <out>/A <out>/B

Faithfulness notes:
  - The preprocessing replay mirrors mlr_prediction_historic.py's __main__ in call order,
    importing the same preprocessing module rather than reimplementing anything.
  - imputation_w_pillows() is replayed even though its return values are discarded by
    production, because it WRITES the imputation cache that impute_pillow_prediction()
    later reads. Skipping it would change which file exists.
  - Floats are written at full double precision (%.17g) so comparison is exact, not
    rounded into agreement.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(1, '/home/rossamower/bin/scripts/')

import metadata as metadata
import preprocessing as preprocessing
import lm_model as lm_model
import postprocessing as postprocessing

DEFAULT_OUT = "/home/rossamower/work/aso/data_test/mlr_ab/"
FLOAT_FMT = "%.17g"


def load_aso_metadata(aso_site_name, config_dir='/home/rossamower/work/aso/configs/'):
    cfg = metadata.load_yaml(Path(f"{config_dir}regions/{aso_site_name}.yaml"))
    _, elev_bin_labels = metadata.get_elevation_bins(cfg)
    fp = cfg["data_filepaths"]
    return elev_bin_labels, fp, f'EPSG:{cfg["crs"]["epsg"]}', cfg


def replay_preprocessing(aso_site_name, water_year, qa_file, holdout_mode, is_friant):
    """
    Mirror mlr_prediction_historic.py __main__ up to the training call. Returns everything
    train_all_mlr_models needs, plus the test-side lists the daily loop uses.
    """
    elev_bin_labels, fp, shape_crs, cfg = load_aso_metadata(aso_site_name)
    insitu_dir, mlrPred_dir = fp["insitu_dir"], fp["mlrPred_dir"]

    aso_spatial_ds = xr.open_dataset(fp["aso_spatial"], engine='netcdf4')
    aso_tseries_ds = xr.open_dataset(fp["aso_temporal"], engine='netcdf4')
    dem_bin = xr.open_dataset(fp["aso_demBin"], engine='netcdf4')

    exclude_pillows = cfg['pillow_api'].get('exclude_pillows') or []
    start_wy = int(cfg["aso_years"]["start"])
    end_wy = int(cfg["aso_years"]["end"])

    obs_data = xr.load_dataset(f'{insitu_dir}processed/{qa_file}')
    obs_test = (obs_data.where(obs_data.time >= np.datetime64(f'{water_year-1}-10-01'), drop=True)
                        .where(obs_data.time < np.datetime64(f'{water_year}-10-01'), drop=True))
    obs_train = obs_data.where(~obs_data.time.isin(obs_test.time.values), drop=True)

    to_list = lambda ds: [ds[p].rename(p) for p in ds.data_vars]
    obs_train_lst, obs_test_lst = to_list(obs_train), to_list(obs_test)

    (aso_spatial_test, aso_tseries_test, aso_spatial_train,
     aso_tseries_train, obs_data_train, obs_data_qa) = preprocessing.train_test_split(
        aso_spatial_ds, aso_tseries_ds, obs_train_lst, water_year,
        isFriant=is_friant, aso_holdout_mode=holdout_mode)

    df_sum_total = preprocessing.combine_aso_insitu(
        obs_data_train, aso_tseries_train['aso_swe'], elev_idx=-1)

    _, all_pils, baseline_pils = preprocessing.generate_drop_NaNs_table(
        df_sum_total, obs_data_qa)

    preprocessing.create_qa_tables(obs_train_lst, [], isQA=False)

    # replayed for its side effect only: this call WRITES the imputation cache that
    # impute_pillow_prediction() reads inside training. production discards its returns.
    preprocessing.imputation_w_pillows(
        df_sum_total, all_pils, obs_data_qa, aso_site_name, water_year,
        f'{mlrPred_dir}imputation/', obs_threshold=0.50, saveImputeCSV=True)

    return dict(
        cfg=cfg, elev_bin_labels=elev_bin_labels, mlrPred_dir=mlrPred_dir,
        exclude_pillows=exclude_pillows, start_wy=start_wy, end_wy=end_wy,
        aso_tseries_train=aso_tseries_train, obs_data_qa=obs_data_qa,
        df_sum_total=df_sum_total, all_pils=all_pils, baseline_pils=baseline_pils,
        dem_bin=dem_bin, obs_test_lst=obs_test_lst, n_timesteps=obs_test_lst[0].time.size,
    )


def sample_indices(n_timesteps, n_want):
    """Evenly spaced timesteps over the WY. Index 0 is skipped, as production does."""
    if n_want >= n_timesteps - 1:
        return list(range(1, n_timesteps))
    return sorted(set(np.linspace(1, n_timesteps - 1, n_want).astype(int).tolist()))


def key_digest(all_pils_QA, baseline_pils_):
    """Stable short digest of a cache key, for cross-run comparison."""
    blob = json.dumps([sorted(all_pils_QA), sorted(baseline_pils_)])
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("aso_site_name")
    ap.add_argument("water_year", type=int)
    ap.add_argument("--label", required=True, help="subdirectory name, e.g. A / A2 / B")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--n-dates", type=int, default=10)
    ap.add_argument("--qa-file", default="pillow_wy_1980_2025_qa6.nc")
    ap.add_argument("--holdout-mode", default="wy", choices=["wy", "flight"])
    ap.add_argument("--is-friant", type=int, default=1)
    ap.add_argument("--is-split", type=int, default=0)
    ap.add_argument("--is-accum", type=int, default=0)
    ap.add_argument("--pillow-imputation", type=int, default=1)
    ap.add_argument("--fit-intercept", type=int, default=0,
                    help="historic uses 0 (force through origin); daily uses 1")
    ap.add_argument("--impute-cache-suffix", default="")
    ap.add_argument("--n-ensemble", type=int, default=None,
                    help="only pass once predict_with_cached_training accepts it")
    args = ap.parse_args()

    out = Path(args.out_dir) / f"{args.aso_site_name}_wy{args.water_year}" / args.label
    out.mkdir(parents=True, exist_ok=True)

    print(f"MLR A/B HARNESS  site={args.aso_site_name} wy={args.water_year} label={args.label}")
    P = replay_preprocessing(args.aso_site_name, args.water_year, args.qa_file,
                             args.holdout_mode, bool(args.is_friant))
    idxs = sample_indices(P["n_timesteps"], args.n_dates)
    print(f"  timesteps={P['n_timesteps']}  sampling {len(idxs)}: {idxs}")

    model_num, isMean, isCombination, *_ , user_qa_level, elev_band, QA_flag = \
        (0, False, True, None, None, None, 0, -1, 1)

    cache = {}
    cache_rows = []
    # accumulated exactly as production does, so output matches prediction_*_combination.csv
    pred_mm_df = pred_af_df = pred_pil_df = None

    for t_idx in idxs:
        t_np = P["obs_test_lst"][0].time[t_idx].values
        ts = pd.to_datetime(t_np)
        current_date = datetime(ts.year, ts.month, ts.day)

        cur_vals, all_pils_QA, baseline_pils_, _ = preprocessing.process_daily_qa(
            t_idx, P["obs_test_lst"], P["obs_test_lst"], P["baseline_pils"],
            P["exclude_pillows"], printOutput=False)

        if len(baseline_pils_) == 0 or len(all_pils_QA) == 0:
            print(f"  {current_date.date()}  SKIP (0 usable pillows)")
            continue

        ck = (frozenset(all_pils_QA), frozenset(baseline_pils_))
        digest = key_digest(all_pils_QA, baseline_pils_)
        if ck not in cache:
            cache[ck] = lm_model.train_all_mlr_models(
                P["aso_tseries_train"].aso_swe, P["obs_data_qa"], args.aso_site_name,
                P["all_pils"], all_pils_QA, P["df_sum_total"], baseline_pils_,
                P["start_wy"], P["end_wy"], bool(args.is_split), bool(args.is_accum),
                P["mlrPred_dir"], current_date, P["dem_bin"].dem_bin, QA_flag=QA_flag,
                modelNUM=model_num, isMean=False, showOutput=False,
                saveValidation=False, isCombination_=isCombination,
                pillowImputation_=bool(args.pillow_imputation), ds_snowmodel_=None,
                impute_cache_suffix=args.impute_cache_suffix)

            # training artifacts most likely to shift silently under a change.
            # stats live under 'validation_stats' (regression_results, lm_model.py:1569).
            for i, c in enumerate(cache[ck]):
                mid = c['modelID']
                sd = c['summary_dict_model'][mid]
                mf, vs = sd['model_features'], sd.get('validation_stats', {})
                cache_rows.append({
                    "key": digest, "entry": i, "modelID": mid,
                    "elev_band": mf.get('elevation_band'),
                    "isImpute": mf.get('isImpute'),
                    "num_obs": mf.get('num_obs'),
                    "selected_pils": ",".join(map(str, c['selected_pils'])),
                    "r2": vs.get('r2'),
                    "RMSE_MM": vs.get('RMSE_MM'),
                    "max_resid_absolute": vs.get('max_resid_absolute'),
                    "area_m2": c['area_m2'],
                })

        kw = {} if args.n_ensemble is None else {"n_ensemble": args.n_ensemble}
        _, mm_l, af_l, pil_l = lm_model.predict_with_cached_training(
            cache[ck], cur_vals.reset_index(names='time'), current_date,
            P["elev_bin_labels"], fit_intercept=bool(args.fit_intercept), **kw)

        # run the same postprocessing production uses. format_prediction_tables renames
        # Total->Basin, adds Band Sum, and for acreFt does /1000 then .astype(int) --
        # so acreFt is quantized to thousands and mm is the sensitive A/B target.
        pred_mm_df, pred_af_df, pred_pil_df = postprocessing.arrange_prediction_tables(
            mm_l, af_l, pil_l, P["elev_bin_labels"], pred_mm_df, pred_af_df, pred_pil_df)
        print(f"  {current_date.date()}  key={digest}  n_QA={len(all_pils_QA)}")

    for name, df in (("prediction_mm", pred_mm_df),
                     ("prediction_acreFt", pred_af_df),
                     ("prediction_pillows", pred_pil_df)):
        if df is not None:
            df.to_csv(out / f"{name}.csv", index=False, float_format=FLOAT_FMT)
    if cache_rows:
        cdf = pd.DataFrame(cache_rows).sort_values(["key", "entry"]).reset_index(drop=True)
        cdf.to_csv(out / "training.csv", index=False, float_format=FLOAT_FMT)

    (out / "run.json").write_text(json.dumps({
        "site": args.aso_site_name, "water_year": args.water_year,
        "qa_file": args.qa_file, "holdout_mode": args.holdout_mode,
        "is_split": args.is_split, "is_accum": args.is_accum,
        "pillow_imputation": args.pillow_imputation,
        "fit_intercept": args.fit_intercept,
        "impute_cache_suffix": args.impute_cache_suffix,
        "n_ensemble": args.n_ensemble,
        "sampled_indices": idxs, "distinct_cache_keys": len(cache),
    }, indent=2, sort_keys=True) + "\n")

    print(f"\n  distinct pillow sets trained: {len(cache)}")
    print(f"  wrote {out}/prediction_{{mm,acreFt,pillows}}.csv, training.csv, run.json")


if __name__ == "__main__":
    main()
