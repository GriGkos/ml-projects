"""Проверяет полную feature matrix без обучения моделей и без записи кэшей.

Исходником feature engineering остаётся notebook: этот скрипт извлекает ровно те
ячейки, которые строят матрицу, поэтому не возникает второй расходящейся версии кода.
"""

from __future__ import annotations

import gc
from pathlib import Path
from time import perf_counter

import nbformat
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / 'notebooks' / 'home_credit_scoring.ipynb'
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'


def run() -> None:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    namespace = {
        '__name__': '__feature_matrix_validation__',
        'gc': gc,
        'np': np,
        'pd': pd,
        'Path': Path,
        'perf_counter': perf_counter,
        'DATA_DIR': DATA_DIR,
        'FAST_MODE': False,
        'FAST_NROWS': 250_000,
        'display': lambda *_args, **_kwargs: None,
    }
    # В ячейках 20, 21, 23 и 24 находятся helpers, feature builders и matrix builder.
    for index in (20, 21, 23, 24):
        exec(compile(notebook.cells[index].source, f'notebook_cell_{index}', 'exec'), namespace)

    train, target, test, categorical = namespace['build_model_matrix']()
    features = [column for column in train if column != 'SK_ID_CURR']
    checks = {
        'target_not_feature': 'TARGET' not in features,
        'equal_schema': train.columns.tolist() == test.columns.tolist(),
        'unique_train_id': train['SK_ID_CURR'].is_unique,
        'unique_test_id': test['SK_ID_CURR'].is_unique,
        'disjoint_ids': set(train['SK_ID_CURR']).isdisjoint(set(test['SK_ID_CURR'])),
        'binary_target': set(target.unique()).issubset({0, 1}),
        'no_numeric_infinity': not np.isinf(
            train[features].select_dtypes(include=np.number).to_numpy(dtype='float32', copy=True)
        ).any(),
        'categoricals_in_matrix': set(categorical).issubset(set(features)),
    }
    print(f'FULL_MATRIX train={train.shape}, test={test.shape}, categorical={len(categorical)}')
    for name, passed in checks.items():
        print(f'{name}: {passed}')
    if not all(checks.values()):
        raise RuntimeError(f'Нарушены инварианты feature matrix: {checks}')
    print('FULL_MATRIX_VALIDATION_OK')


if __name__ == '__main__':
    run()
