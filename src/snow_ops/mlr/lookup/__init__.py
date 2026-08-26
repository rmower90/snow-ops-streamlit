"""
Pretrained MLR lookup tables for basin-mean ASO SWE.

The operational idea, in one line: fit every allowed pillow combination ONCE over the
whole historical record, store the coefficients, and at prediction time do nothing but
filter the stored models by which pillows reported today and evaluate the best eligible
one. No cross-validation, no pillow selection, and no regression fitting on the daily
path.

The property that makes it work is that the historical training matrix is built without
reference to any particular day. In the existing pathway it is not: today's QA-passing
pillow set (`all_pils_QA`) reaches into which pillows are retained, which values are
imputed and from which donors, how many ASO flights survive, and what ends up in the
on-disk imputation cache. `frame.py` severs that coupling; today's availability enters
only at query time.

Module map:

    frame.py    canonical predict-NaNs training matrix + manifest   [milestone 1]
    fit.py      enumerate combinations, fit, statistics             [not yet]
    store.py    parquet/JSON model-table persistence                [not yet]
    query.py    availability mask -> eligible -> select -> predict  [not yet]
"""
