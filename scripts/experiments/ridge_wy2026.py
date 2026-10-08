"""
ridge_wy2026.py -- arm 3: ridge regression predictions for WY2026, USCASJ.

Fits one ridge model per elevation band on the historic ASO flight record
(WY2017-2025), then predicts every day of WY2026. Alpha is chosen by inner
leave-one-water-year-out within the training years only.

Counterpart to arms 1 and 2, which ran the operational OLS combination search
through mlr_prediction.py. Training data and imputation machinery are identical;
only the model differs.

NO PHASE TERM. The baseline fits three separate temporal models (season / accum /
melt) rather than one model with a phase indicator, so the counterpart to its
`season` model is ridge with melt_dates=None. This also means no accumulation/melt
assignment is needed for a water year in progress.

IMPUTATION. Ridge carries a coefficient per pillow and cannot drop a missing one,
so every eligible pillow needs a value on every predicted day. Only 4 pillows are
complete across both the 35 training flights and the 5 WY2026 flights, so
imputation is what makes the method usable at all. Both sides are filled with
preprocessing.imputation_w_pillows -- the same function arms 1 and 2 use --
called with saveImputeCSV=False so nothing is cached to disk.

Writes nothing outside its output directory.

Usage (conda env: bor):
    source /opt/miniforge3/etc/profile.d/conda.sh && conda activate bor
    /home/rossamower/work/aso/conda/envs/bor/bin/python ridge_wy2026.py
"""
import argparse, json, hashlib, subprocess, datetime, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

sys.path.insert(1, str(Path(__file__).resolve().parent.parent))
sys.path.insert(1, str(Path(__file__).resolve().parent))

import preprocessing as preprocessing
import ridge_linear as rl


def _impute(obs_ds, rows_index, label):
    """
    Fill NaNs in obs_ds at the given timestamps, using preprocessing's three-donor
    search. Builds the frame imputation_w_pillows expects: the aso_mean_bins_mm
    column is carried through but never used by the imputation itself, so NaN is
    fine for prediction rows that have no flight.
    """
    pil_list = [obs_ds[p] for p in obs_ds.data_vars]
    frame = pd.DataFrame({'time': pd.to_datetime(rows_index)})
    for p in obs_ds.data_vars:
        frame[p] = obs_ds[p].sel(time=frame['time'].values).values
    frame['aso_mean_bins_mm'] = np.nan
    all_pils = [p for p in obs_ds.data_vars]

    filled, kept, _ = preprocessing.imputation_w_pillows(
        frame, all_pils, pil_list, 'USCASJ', 2026, '/tmp/_ridge_impute_unused/',
        obs_threshold=0.50, saveImputeCSV=False,
    )
    out = xr.Dataset({da.name: da for da in filled if da.name in kept})
    n_before = int(sum(np.isfinite(frame[p].values).sum() for p in kept))
    n_after  = int(sum(np.isfinite(out[p].sel(time=frame['time'].values).values).sum() for p in kept))
    print(f'  [{label}] pillows kept {len(kept)}/{len(all_pils)}; '
          f'finite on target rows {n_before} -> {n_after}')
    return out, list(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--basin', default='USCASJ')
    ap.add_argument('--config-dir', default='/home/rossamower/work/aso/configs/')
    ap.add_argument('--train-pillows', default='processed/pillow_wy_1980_2025_qa1.nc')
    ap.add_argument('--test-pillows',  default='processed/USCASJ_insitu_obs_daily_wy_2026_s82.nc')
    ap.add_argument('--suffix', default='_s82_ridge')
    ap.add_argument('--phase', action='store_true',
                    help='add an unpenalised accumulation/melt indicator. Training phase comes '
                         'from melt_threshold.csv (per water year, per band); WY2026 phase from '
                         'the date of peak SnowModel SWE in that band -- causal, so usable in '
                         'real time. TEST ONLY: the two sides use different onset definitions.')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(f"{args.config_dir}regions/{args.basin}.yaml"))
    fp  = cfg['data_filepaths']
    labels = cfg['elevation']['bins']['labels']

    aso = xr.open_dataset(fp['aso_temporal'])['aso_swe']       # (date, elev), WY2017-2025
    train_obs = xr.open_dataset(f"{fp['insitu_dir']}{args.train_pillows}")
    test_obs  = xr.open_dataset(f"{fp['insitu_dir']}{args.test_pillows}")

    flights = pd.to_datetime(aso.date.values)
    wy26    = pd.to_datetime(test_obs.time.values)
    print(f'training flights: {len(flights)}  ({flights.min().date()}..{flights.max().date()})')
    print(f'prediction days : {len(wy26)}  ({wy26.min().date()}..{wy26.max().date()})')

    train_imp, train_keep = _impute(train_obs, flights, 'train')
    test_imp,  test_keep  = _impute(test_obs,  wy26,    'test ')

    shared = [p for p in train_keep if p in test_keep]
    print(f'  pillows usable in both: {len(shared)}')
    train_imp = train_imp[shared]
    test_imp  = test_imp[shared]

    thresh = sm_peak = None
    if args.phase:
        # melt_threshold/ sits a level ABOVE mlrPred_dir, which points at models/.
        # lm_model.py hardcodes this same path rather than deriving it from the config.
        thresh = pd.read_csv(
            f"/home/rossamower/work/aso/data/mlr_prediction/{args.basin}/melt_threshold/melt_threshold.csv")
        thresh['threshold_best'] = pd.to_datetime(thresh['threshold_best'])
        sm = pd.read_csv('/home/rossamower/work/aso/snowmodel/domains/USCASJ/mean_swe_snowmodel_m_wy2026.csv')
        sm['Date'] = pd.to_datetime(sm['Date'])
        sm_cols = dict(zip(labels, ['<7000','7000-8000','8000-9000','9000-10000',
                                    '10000-11000','11000-12000','>12000']))
        sm_peak = {lab: sm.loc[sm[sm_cols[lab]].idxmax(), 'Date'] for lab in labels}
        print('\n  WY2026 melt onset from peak SnowModel SWE:')
        for lab in labels: print(f'    {lab:<10} {sm_peak[lab].date()}')
        print()

    out = pd.DataFrame({'Date': wy26})
    diag = []
    for i, lab in enumerate(labels):
        swe = aso.isel(elev=i).rename({'date': 'date'})
        melt_dates = phase_pred = None
        if args.phase:
            # melt_threshold.csv has duplicate (water_year, elev_bin) rows -- 100 where
            # 9 years x 8 bins would be 72 -- so collapse to the first per year.
            tb = (thresh[thresh['elev_bin'] == i]
                  .groupby('water_year')['threshold_best'].first())
            fl_wy = np.where(flights.month >= 10, flights.year + 1, flights.year)
            melt_dates = np.array([d for d, w in zip(flights, fl_wy)
                                   if w in tb.index and d >= tb.loc[w]], dtype='datetime64[ns]')
            phase_pred = np.asarray(wy26 >= sm_peak[lab], dtype=float)
        model = rl.fit_ridge_band(swe, train_imp, melt_dates=melt_dates)
        yhat  = rl.predict_ridge_band(model, test_imp, wy26, phase=phase_pred)
        out[lab] = yhat
        diag.append({'band': lab, 'n_pillows': len(model['pillow_names']),
                     'alpha': model['alpha'], 'n_train': model['n_train'],
                     'finite_pred': int(np.isfinite(yhat).sum())})
        print(f"  {lab:<10} pillows={len(model['pillow_names']):>2} alpha={model['alpha']:<10.4g} "
              f"n_train={model['n_train']:>3} finite={int(np.isfinite(yhat).sum())}/{len(wy26)}")

    out_dir = Path(f"{fp['mlrPred_dir']}COMMON_MASK{args.suffix}/season/mm")
    out_dir.mkdir(parents=True, exist_ok=True)
    out.insert(1, 'Model Type', 'season')
    out.insert(2, 'Training Infer NaNs', 'predict NaNs')
    out.insert(3, 'Prediction QA', True)
    out.to_csv(out_dir / 'prediction_mm_wy2026_combination.csv', index=False)

    def _sha16(p):
        try:
            h = hashlib.sha256()
            with open(p, 'rb') as fh:
                for b in iter(lambda: fh.read(1 << 20), b''): h.update(b)
            return h.hexdigest()[:16]
        except Exception:
            return None
    def _git(*a):
        try:
            return subprocess.check_output(['git', *a], cwd=os.path.dirname(os.path.realpath(__file__)),
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return None
    manifest = {
        'written_utc': datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'basin': args.basin, 'water_year': 2026,
        'model': {'selection': 'ridge, all eligible pillows, alpha by inner LOWYO',
                  'phase_term': False, 'seasonal_dir': 'season',
                  'imputation': 'pillow (imputation_w_pillows, saveImputeCSV=False)'},
        'inputs': {k: {'path': v, 'sha256_16': _sha16(v)} for k, v in {
            'train_pillows': f"{fp['insitu_dir']}{args.train_pillows}",
            'test_pillows':  f"{fp['insitu_dir']}{args.test_pillows}",
            'aso_temporal':  fp['aso_temporal']}.items()},
        'bands': diag,
        'git': {'commit': _git('rev-parse', 'HEAD'), 'branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
                'dirty': bool(_git('status', '--porcelain'))},
    }
    (out_dir.parent / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'\nWROTE {out_dir}/prediction_mm_wy2026_combination.csv')
    print(f'WROTE {out_dir.parent}/run_manifest.json')


if __name__ == '__main__':
    main()
