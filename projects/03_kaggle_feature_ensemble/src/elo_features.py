"""Feature engineering utilities for Elo Merchant Category Recommendation.

The historical transactions file is larger than memory on many laptops, so the
aggregation is deliberately performed in chunks. Every resulting feature is at
the card level and can be joined to both train and test without target leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REFERENCE_DATE = pd.Timestamp("2018-02-01")
CHUNK_SIZE = 750_000


def resolve_data_dir(project_dir: Path) -> Path:
    """Support both a normal data folder and the local worktree data junction."""
    direct = project_dir / "data"
    if (direct / "train.csv").exists():
        return direct

    linked = direct / "raw"
    if (linked / "train.csv").exists():
        return linked

    raise FileNotFoundError(
        "Could not find train.csv. Download the Kaggle files into data/."
    )


def _safe_skew(values: pd.Series) -> float:
    return values.skew() if len(values) > 2 else np.nan


def _base_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    active = pd.to_datetime(result["first_active_month"], format="%Y-%m")
    result["first_active_year"] = active.dt.year
    result["first_active_month_num"] = active.dt.month
    result["first_active_quarter"] = active.dt.quarter
    result["card_age_months"] = (
        (REFERENCE_DATE.year - active.dt.year) * 12
        + REFERENCE_DATE.month
        - active.dt.month
    )

    # The three small anonymized card features are categorical. Their
    # interactions let tree models capture combinations without target encoding.
    result["feature_1_2"] = result["feature_1"] * 10 + result["feature_2"]
    result["feature_1_3"] = result["feature_1"] * 10 + result["feature_3"]
    result["feature_2_3"] = result["feature_2"] * 10 + result["feature_3"]
    result["feature_sum"] = result[["feature_1", "feature_2", "feature_3"]].sum(axis=1)
    result = result.drop(columns="first_active_month")
    return result


def _merchant_maps(data_dir: Path) -> dict[str, pd.Series]:
    merchants = pd.read_csv(data_dir / "merchants.csv")
    merchants = merchants.drop_duplicates("merchant_id").set_index("merchant_id")

    ranges = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
    for column in ["most_recent_sales_range", "most_recent_purchases_range"]:
        merchants[column] = merchants[column].map(ranges)

    selected = [
        "numerical_1",
        "numerical_2",
        "avg_sales_lag3",
        "avg_purchases_lag3",
        "avg_sales_lag6",
        "avg_purchases_lag6",
        "avg_sales_lag12",
        "avg_purchases_lag12",
        "active_months_lag3",
        "active_months_lag6",
        "active_months_lag12",
        "most_recent_sales_range",
        "most_recent_purchases_range",
    ]
    return {column: merchants[column] for column in selected}


def _prepare_transaction_chunk(
    chunk: pd.DataFrame,
    merchant_maps: dict[str, pd.Series],
) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["purchase_date"] = pd.to_datetime(chunk["purchase_date"])
    chunk["purchase_day"] = (REFERENCE_DATE - chunk["purchase_date"]).dt.days
    chunk["month_diff"] = chunk["purchase_day"] // 30 + chunk["month_lag"]
    chunk["purchase_month"] = chunk["purchase_date"].dt.month
    chunk["purchase_weekday"] = chunk["purchase_date"].dt.weekday
    chunk["purchase_weekend"] = (chunk["purchase_weekday"] >= 5).astype("int8")
    chunk["authorized"] = (chunk["authorized_flag"] == "Y").astype("int8")
    chunk["category_1_flag"] = (chunk["category_1"] == "Y").astype("int8")
    chunk["category_2"] = chunk["category_2"].fillna(0)
    # Convert from categorical first: pandas does not allow filling a new value in a Categorical.
    chunk["category_3_code"] = (
        chunk["category_3"].astype("string").map({"A": 1, "B": 2, "C": 3}).fillna(0).astype("int8")
    )
    chunk["installments"] = chunk["installments"].replace({-1: np.nan, 999: np.nan})
    installment_denominator = chunk["installments"].where(chunk["installments"] > 0, 1)
    chunk["purchase_amount_per_installment"] = chunk["purchase_amount"] / installment_denominator

    for column, values in merchant_maps.items():
        chunk[f"merchant_{column}"] = chunk["merchant_id"].map(values)

    return chunk


def _aggregate_chunk(chunk: pd.DataFrame, prefix: str) -> pd.DataFrame:
    grouped = chunk.groupby("card_id", sort=False)

    aggregations: dict[str, list[str] | str] = {
        "purchase_amount": ["count", "sum", "mean", "std", "min", "max", "median", _safe_skew],
        "installments": ["mean", "std", "min", "max", "sum"],
        "month_lag": ["mean", "std", "min", "max", "nunique"],
        "month_diff": ["mean", "std", "min", "max"],
        "purchase_amount_per_installment": ["mean", "std", "min", "max"],
        "purchase_day": ["min", "max", "mean", "std"],
        "purchase_month": ["mean", "std", "nunique"],
        "purchase_weekday": ["mean", "std", "nunique"],
        "purchase_weekend": "mean",
        "authorized": "mean",
        "category_1_flag": "mean",
        "category_2": ["mean", "nunique"],
        "category_3_code": ["mean", "nunique"],
        "merchant_category_id": "nunique",
        "subsector_id": "nunique",
        "city_id": "nunique",
        "state_id": "nunique",
        "merchant_id": "nunique",
    }

    merchant_feature_columns = [
        name
        for name in chunk.columns
        if name.startswith("merchant_") and name not in {"merchant_id", "merchant_category_id"}
    ]
    for column in merchant_feature_columns:
        aggregations[column] = ["mean", "std"]

    result = grouped.agg(aggregations)
    result.columns = [
        f"{prefix}_{column}_{function if isinstance(function, str) else function.__name__}"
        for column, function in result.columns.to_flat_index()
    ]
    result = result.reset_index()

    # Authorized and declined transactions can have different information
    # content. Keep their volume and spend separately rather than only using
    # the overall authorization rate.
    for authorization, label in [(1, "authorized"), (0, "unauthorized")]:
        subset = chunk.loc[chunk["authorized"] == authorization]
        statistics = subset.groupby("card_id", sort=False)["purchase_amount"].agg(["count", "sum", "mean"])
        statistics.columns = [f"{prefix}_{label}_purchase_amount_{stat}" for stat in statistics.columns]
        result = result.merge(statistics, left_on="card_id", right_index=True, how="left")

    # Category-level unique counts are exact inside each chunk; for the
    # historical file they are additive approximations across chunks. They
    # remain useful diversity features and are explicitly named accordingly.
    for column in [
        "merchant_category_id",
        "subsector_id",
        "city_id",
        "state_id",
        "merchant_id",
    ]:
        original = f"{prefix}_{column}_nunique"
        if original in result:
            result = result.rename(columns={original: f"{prefix}_{column}_nunique_approx"})

    result[f"{prefix}_active_span_days"] = (
        result[f"{prefix}_purchase_day_max"] - result[f"{prefix}_purchase_day_min"]
    ).abs()
    result[f"{prefix}_transactions_per_active_day"] = (
        result[f"{prefix}_purchase_amount_count"]
        / (result[f"{prefix}_active_span_days"] + 1)
    )
    return result


def _combine_chunk_aggregates(parts: list[pd.DataFrame], prefix: str) -> pd.DataFrame:
    """Combine exact additive chunk statistics and average the remaining ratios."""
    merged = pd.concat(parts, ignore_index=True)
    numeric_columns = [column for column in merged.columns if column != "card_id"]
    additive = [
        column
        for column in numeric_columns
        if column.endswith(("_count", "_sum", "_nunique_approx"))
    ]
    minimum = [column for column in numeric_columns if column.endswith("_min")]
    maximum = [column for column in numeric_columns if column.endswith("_max")]
    remaining = [
        column
        for column in numeric_columns
        if column not in set(additive + minimum + maximum)
    ]

    grouped = merged.groupby("card_id", sort=False)
    result = pd.concat(
        [
            grouped[additive].sum(min_count=1) if additive else None,
            grouped[minimum].min() if minimum else None,
            grouped[maximum].max() if maximum else None,
            grouped[remaining].mean() if remaining else None,
        ],
        axis=1,
    )
    result = result.loc[:, ~result.columns.duplicated()].reset_index()

    count_column = f"{prefix}_purchase_amount_count"
    sum_column = f"{prefix}_purchase_amount_sum"
    if count_column in result and sum_column in result:
        result[f"{prefix}_purchase_amount_mean"] = result[sum_column] / result[count_column]

    span_column = f"{prefix}_active_span_days"
    day_min = f"{prefix}_purchase_day_min"
    day_max = f"{prefix}_purchase_day_max"
    if day_min in result and day_max in result:
        result[span_column] = (result[day_max] - result[day_min]).abs()

    if count_column in result and span_column in result:
        result[f"{prefix}_transactions_per_active_day"] = result[count_column] / (
            result[span_column] + 1
        )
    return result


def aggregate_transactions(
    file_path: Path,
    prefix: str,
    merchant_maps: dict[str, pd.Series],
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    """Read one transaction table in chunks and return one row per card."""
    parts: list[pd.DataFrame] = []
    usecols = [
        "authorized_flag",
        "card_id",
        "city_id",
        "category_1",
        "category_2",
        "category_3",
        "installments",
        "merchant_category_id",
        "merchant_id",
        "month_lag",
        "purchase_amount",
        "purchase_date",
        "state_id",
        "subsector_id",
    ]
    dtypes = {
        "authorized_flag": "category",
        "card_id": "string",
        "city_id": "int16",
        "category_1": "category",
        "category_2": "float32",
        "category_3": "category",
        "installments": "int16",
        "merchant_category_id": "int32",
        "merchant_id": "string",
        "month_lag": "int16",
        "purchase_amount": "float32",
        "state_id": "int16",
        "subsector_id": "int16",
    }
    for index, chunk in enumerate(
        pd.read_csv(file_path, usecols=usecols, dtype=dtypes, chunksize=chunk_size), start=1
    ):
        prepared = _prepare_transaction_chunk(chunk, merchant_maps)
        parts.append(_aggregate_chunk(prepared, prefix))
        print(f"{prefix}: processed chunk {index}")

    return _combine_chunk_aggregates(parts, prefix)


def _add_cross_period_features(features: pd.DataFrame) -> pd.DataFrame:
    """Create robust ratios comparing historical and evaluation windows."""
    result = features.copy()
    pairs = [
        ("purchase_amount_count", "transaction_count_ratio"),
        ("purchase_amount_sum", "purchase_amount_ratio"),
        ("merchant_id_nunique_approx", "merchant_diversity_ratio"),
    ]
    for suffix, feature_name in pairs:
        historical = result[f"hist_{suffix}"].fillna(0)
        new = result[f"new_{suffix}"].fillna(0)
        result[feature_name] = new / (historical.abs() + 1)

    result["new_minus_hist_purchase_amount_mean"] = (
        result["new_purchase_amount_mean"] - result["hist_purchase_amount_mean"]
    )
    result["new_minus_hist_authorized_rate"] = result["new_authorized_mean"] - result["hist_authorized_mean"]
    return result


def build_feature_matrices(project_dir: Path, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and cache card-level train and test matrices without target leakage."""
    artifacts_dir = project_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    train_cache = artifacts_dir / "train_features.pkl"
    test_cache = artifacts_dir / "test_features.pkl"

    if train_cache.exists() and test_cache.exists() and not force:
        return pd.read_pickle(train_cache), pd.read_pickle(test_cache)

    data_dir = resolve_data_dir(project_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    target = train.pop("target")
    train_base = _base_features(train)
    test_base = _base_features(test)

    maps = _merchant_maps(data_dir)
    historical = aggregate_transactions(
        data_dir / "historical_transactions.csv", "hist", maps
    )
    new_transactions = aggregate_transactions(
        data_dir / "new_merchant_transactions.csv", "new", maps
    )

    transaction_features = historical.merge(new_transactions, on="card_id", how="outer")
    transaction_features = _add_cross_period_features(transaction_features)
    train_features = train_base.merge(transaction_features, on="card_id", how="left")
    test_features = test_base.merge(transaction_features, on="card_id", how="left")
    train_features["target"] = target

    train_features.to_pickle(train_cache)
    test_features.to_pickle(test_cache)
    return train_features, test_features


def feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    """Compact data-quality table used in the notebook report."""
    return pd.DataFrame(
        {
            "dtype": features.dtypes.astype(str),
            "missing_rate": features.isna().mean(),
            "n_unique": features.nunique(dropna=False),
        }
    ).sort_values("missing_rate", ascending=False)
