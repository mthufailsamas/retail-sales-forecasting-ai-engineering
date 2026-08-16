"""Create the final preprocessed Store Sales train and inference tables.

This single raw-to-processed pipeline combines every source table that is
usable for the competition forecast:

1. store metadata;
2. calendar fields;
3. cutoff-aligned, last-known oil history;
4. cutoff-aligned store transaction history;
5. store-aware scheduled holiday information.

Same-day sales, transactions, and oil values are never used for a future
forecast. Oil retains the latest price published by each lag reference date and
records its age. Transaction lags remain missing when their exact historical
store-date record is absent. Model-specific sales lags, encoding, imputation,
splitting, and training remain separate later stages.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_EDA_CSV_PATH = DEFAULT_PROCESSED_DIR / "00_STORE_SALES_EDA.csv"
DEFAULT_TEST_CSV_PATH = DEFAULT_PROCESSED_DIR / "01_STORE_SALES_KAGGLE_TEST.csv"

OIL_COLUMNS = ["date", "dcoilwtico"]
TRANSACTION_COLUMNS = ["date", "store_nbr", "transactions"]
HOLIDAY_COLUMNS = [
    "date",
    "type",
    "locale",
    "locale_name",
    "description",
    "transferred",
]

CALENDAR_COLUMNS = [
    "month",
    "day_of_month",
    "day_of_week",
]
SAFE_LAGS = [16, 21, 28, 35]
OIL_LAGS = SAFE_LAGS
OIL_LAG_COLUMNS = [f"oil_lag_{lag}" for lag in OIL_LAGS]
OIL_AGE_COLUMNS = [f"oil_lag_{lag}_age_days" for lag in OIL_LAGS]
OIL_FEATURE_COLUMNS = [
    *OIL_LAG_COLUMNS,
    *OIL_AGE_COLUMNS,
]
OIL_NULLABLE_COLUMNS = [*OIL_LAG_COLUMNS, *OIL_AGE_COLUMNS]
TRANSACTION_LAGS = SAFE_LAGS
TRANSACTION_LAG_COLUMNS = [
    f"transactions_lag_{lag}" for lag in TRANSACTION_LAGS
]
TRANSACTION_FEATURE_COLUMNS = [
    *TRANSACTION_LAG_COLUMNS,
    "transactions_lag_available_count",
]
TRANSACTION_NULLABLE_COLUMNS = [*TRANSACTION_LAG_COLUMNS]
HOLIDAY_FEATURE_COLUMNS = [
    "is_holiday",
    "is_special_work_day",
    "is_holiday_transfer_source",
    "is_holiday_transfer_destination",
    "is_planned_event",
    "is_national_schedule",
    "is_regional_schedule",
    "is_local_schedule",
]
ALLOWED_MISSING_COLUMNS = OIL_NULLABLE_COLUMNS + TRANSACTION_NULLABLE_COLUMNS

TRAIN_COLUMNS = ["id", "date", "store_nbr", "family", "sales", "onpromotion"]
TEST_COLUMNS = ["id", "date", "store_nbr", "family", "onpromotion"]
STORE_COLUMNS = ["store_nbr", "city", "state", "type", "cluster"]
STORE_OUTPUT_COLUMNS = ["city", "state", "store_type", "store_cluster"]
BASE_KEY = ["date", "store_nbr", "family"]


def require_exact_columns(
    table_name: str,
    table: pd.DataFrame,
    expected_columns: list[str],
) -> None:
    """Reject a source table whose column order differs from the contract."""
    actual_columns = table.columns.tolist()
    if actual_columns != expected_columns:
        raise ValueError(
            f"{table_name} columns differ from the contract. "
            f"Expected {expected_columns}, received {actual_columns}."
        )


def read_base_table(path: Path, expected_columns: list[str]) -> pd.DataFrame:
    """Read train.csv or test.csv with explicit, memory-conscious data types."""
    table = pd.read_csv(
        path,
        dtype={
            "id": "int64",
            "store_nbr": "int16",
            "family": "string",
            "onpromotion": "int32",
        },
        parse_dates=["date"],
        low_memory=False,
    )
    require_exact_columns(path.name, table, expected_columns)
    return table


def read_store_table(path: Path) -> pd.DataFrame:
    """Read store metadata and clarify two ambiguous source column names."""
    stores = pd.read_csv(
        path,
        dtype={
            "store_nbr": "int16",
            "city": "string",
            "state": "string",
            "type": "string",
            "cluster": "int16",
        },
        low_memory=False,
    )
    require_exact_columns(path.name, stores, STORE_COLUMNS)
    return stores.rename(columns={"type": "store_type", "cluster": "store_cluster"})


def validate_base_table(table_name: str, table: pd.DataFrame) -> None:
    """Check the prediction rows that every join must preserve."""
    if table.empty:
        raise ValueError(f"{table_name} is empty.")
    if table[BASE_KEY].isna().any().any():
        raise ValueError(f"{table_name} contains missing base-key values.")
    if table.duplicated(BASE_KEY).any():
        raise ValueError(f"{table_name} is not unique on {BASE_KEY}.")
    if table["id"].isna().any() or not table["id"].is_unique:
        raise ValueError(f"{table_name} id must be present and unique.")
    if table["date"].isna().any():
        raise ValueError(f"{table_name} contains invalid dates.")


def validate_store_table(stores: pd.DataFrame) -> None:
    """Require one complete metadata row for every store number."""
    if stores.empty:
        raise ValueError("stores.csv is empty.")
    if stores["store_nbr"].isna().any():
        raise ValueError("stores.csv contains a missing store_nbr.")
    if not stores["store_nbr"].is_unique:
        raise ValueError("stores.csv must contain one row per store_nbr.")
    if stores[STORE_OUTPUT_COLUMNS].isna().any().any():
        raise ValueError("stores.csv contains missing store metadata.")


def join_store_metadata(
    table_name: str,
    base: pd.DataFrame,
    stores: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach store metadata without dropping, duplicating, or reordering rows."""
    validate_base_table(table_name, base)
    validate_store_table(stores)

    missing_store_count = int((~base["store_nbr"].isin(stores["store_nbr"])).sum())
    if missing_store_count:
        raise ValueError(
            f"{table_name} contains {missing_store_count:,} rows whose store_nbr "
            "is absent from stores.csv."
        )

    base_ids = base["id"].to_numpy(copy=True)
    joined = base.merge(
        stores,
        how="left",
        on="store_nbr",
        validate="many_to_one",
        sort=False,
    )

    if len(joined) != len(base):
        raise RuntimeError(f"{table_name} row count changed during the store join.")
    if not np.array_equal(base_ids, joined["id"].to_numpy()):
        raise RuntimeError(f"{table_name} id order changed during the store join.")
    if joined.duplicated(BASE_KEY).any():
        raise RuntimeError(f"{table_name} base key became duplicated after the join.")
    if joined[STORE_OUTPUT_COLUMNS].isna().any().any():
        raise RuntimeError(f"{table_name} has missing metadata after the store join.")

    evidence = {
        "input_rows": int(len(base)),
        "output_rows": int(len(joined)),
        "input_unique_ids": int(base["id"].nunique()),
        "output_unique_ids": int(joined["id"].nunique()),
        "base_key_duplicate_rows_after_join": int(
            joined.duplicated(BASE_KEY, keep=False).sum()
        ),
        "missing_store_metadata_cells_after_join": int(
            joined[STORE_OUTPUT_COLUMNS].isna().sum().sum()
        ),
        "id_order_preserved": True,
        "date_min": joined["date"].min().date().isoformat(),
        "date_max": joined["date"].max().date().isoformat(),
    }
    return joined, evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final preprocessed Store Sales tables."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument(
        "--eda-csv-path",
        type=Path,
        default=DEFAULT_EDA_CSV_PATH,
        help="Easy-to-find plain CSV containing the labeled preprocessed data.",
    )
    parser.add_argument(
        "--test-csv-path",
        type=Path,
        default=DEFAULT_TEST_CSV_PATH,
        help="Easy-to-find plain CSV containing Kaggle inference rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing final processed outputs after validation.",
    )
    return parser.parse_args()


def read_oil(path: Path) -> pd.DataFrame:
    oil = pd.read_csv(path, parse_dates=["date"])
    if oil.columns.tolist() != OIL_COLUMNS:
        raise ValueError("oil.csv columns differ from the accepted contract.")
    if oil["date"].isna().any() or not oil["date"].is_unique:
        raise ValueError("oil.csv dates must be valid and unique.")
    valid_prices = oil["dcoilwtico"].dropna()
    if not np.isfinite(valid_prices).all() or (valid_prices <= 0).any():
        raise ValueError("Available oil prices must be finite and positive.")
    return oil


def read_transactions(path: Path) -> pd.DataFrame:
    transactions = pd.read_csv(
        path,
        parse_dates=["date"],
        dtype={"store_nbr": "int16", "transactions": "int32"},
    )
    if transactions.columns.tolist() != TRANSACTION_COLUMNS:
        raise ValueError("transactions.csv columns differ from the contract.")
    if transactions[["date", "store_nbr", "transactions"]].isna().any().any():
        raise ValueError("transactions.csv contains missing required values.")
    if transactions.duplicated(["date", "store_nbr"]).any():
        raise ValueError("transactions.csv is not unique by date and store_nbr.")
    if (transactions["transactions"] < 0).any():
        raise ValueError("transactions.csv contains negative transaction counts.")
    return transactions


def read_holidays(path: Path) -> pd.DataFrame:
    holidays = pd.read_csv(path, parse_dates=["date"])
    if holidays.columns.tolist() != HOLIDAY_COLUMNS:
        raise ValueError("holidays_events.csv columns differ from the contract.")
    if holidays.isna().any().any():
        raise ValueError("holidays_events.csv contains missing values.")
    if holidays["date"].isna().any():
        raise ValueError("holidays_events.csv contains invalid dates.")
    holidays["transferred"] = holidays["transferred"].astype(bool)
    return holidays


def add_calendar_features(table: pd.DataFrame) -> pd.DataFrame:
    """Add values known from the calendar before any forecast is made."""
    result = table.copy()
    result["month"] = result["date"].dt.month.astype("int8")
    result["day_of_month"] = result["date"].dt.day.astype("int8")
    result["day_of_week"] = (result["date"].dt.dayofweek + 1).astype("int8")
    return result


def build_oil_lookup(
    base_dates: pd.Series, oil: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create causal last-known oil lags and source-age information."""
    lookup = pd.DataFrame(
        {"date": pd.Series(base_dates.unique()).sort_values(ignore_index=True)}
    )
    observations = (
        oil.dropna(subset=["dcoilwtico"])
        .rename(columns={"date": "oil_source_date", "dcoilwtico": "oil_value"})
        .sort_values("oil_source_date")
    )
    if observations.empty:
        raise ValueError("oil.csv has no observed price available for lag creation.")

    missing_by_lag: dict[str, int] = {}
    last_known_by_lag: dict[str, int] = {}
    maximum_age_by_lag: dict[str, int | None] = {}

    for lag in OIL_LAGS:
        reference = lookup[["date"]].copy()
        reference["oil_reference_date"] = reference["date"] - pd.Timedelta(days=lag)
        matched = pd.merge_asof(
            reference.sort_values("oil_reference_date"),
            observations,
            left_on="oil_reference_date",
            right_on="oil_source_date",
            direction="backward",
            allow_exact_matches=True,
        ).sort_values("date")

        value_column = f"oil_lag_{lag}"
        age_column = f"oil_lag_{lag}_age_days"
        lookup[value_column] = matched["oil_value"].to_numpy(dtype=np.float32)
        age_days = (
            matched["oil_reference_date"] - matched["oil_source_date"]
        ).dt.days
        lookup[age_column] = age_days.to_numpy(dtype=np.float32)

        invalid_future_match = matched["oil_source_date"].gt(
            matched["oil_reference_date"]
        ).fillna(False)
        if invalid_future_match.any():
            raise RuntimeError(f"oil_lag_{lag} used a future oil observation.")

        observed_ages = lookup[age_column].dropna()
        missing_by_lag[str(lag)] = int(lookup[value_column].isna().sum())
        last_known_by_lag[str(lag)] = int(observed_ages.gt(0).sum())
        maximum_age_by_lag[str(lag)] = (
            int(observed_ages.max()) if not observed_ages.empty else None
        )

    evidence = {
        "required_dates": int(len(lookup)),
        "lags_days": OIL_LAGS,
        "missing_values_by_lag": missing_by_lag,
        "last_known_values_by_lag": last_known_by_lag,
        "maximum_source_age_days_by_lag": maximum_age_by_lag,
        "source_date_min": oil["date"].min().date().isoformat(),
        "source_date_max": oil["date"].max().date().isoformat(),
        "future_actual_values_used": False,
    }
    return lookup[["date", *OIL_FEATURE_COLUMNS]], evidence


def build_transaction_lookup(
    base_store_dates: pd.DataFrame,
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create exact transaction lags that are safe for a 16-day forecast."""
    lookup = base_store_dates[["date", "store_nbr"]].drop_duplicates().copy()
    original_order = lookup.index

    history = transactions.set_index(["date", "store_nbr"])["transactions"]
    if not history.index.is_unique:
        raise ValueError("Transaction history must be unique by date and store.")

    for lag in TRANSACTION_LAGS:
        lookup_key = pd.MultiIndex.from_arrays(
            [
                lookup["date"] - pd.Timedelta(days=lag),
                lookup["store_nbr"],
            ],
            names=["date", "store_nbr"],
        )
        value_column = f"transactions_lag_{lag}"
        lookup[value_column] = history.reindex(lookup_key).to_numpy(dtype=np.float32)

    lookup["transactions_lag_available_count"] = (
        lookup[TRANSACTION_LAG_COLUMNS].notna().sum(axis=1).astype("int8")
    )

    lookup = lookup.loc[original_order]

    evidence = {
        "required_store_dates": int(len(lookup)),
        "lags_days": TRANSACTION_LAGS,
        "rows_with_no_available_lag": int(
            lookup["transactions_lag_available_count"].eq(0).sum()
        ),
        "missing_values_by_lag": {
            str(lag): int(lookup[f"transactions_lag_{lag}"].isna().sum())
            for lag in TRANSACTION_LAGS
        },
        "minimum_lag_days": min(TRANSACTION_LAGS),
        "future_actual_values_used": False,
    }
    return lookup[["date", "store_nbr", *TRANSACTION_FEATURE_COLUMNS]], evidence


def normalize_holidays(
    holidays: pd.DataFrame, stores: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map scheduled calendar records and exclude unplanned event hindsight."""
    store_locations = stores[["store_nbr", "city", "state"]].copy()
    events = holidays.reset_index(drop=True).copy()
    events["event_id"] = np.arange(len(events))
    events["normalized_type"] = (
        events["type"].str.lower().str.replace(" ", "_", regex=False)
    )
    events["normalized_description"] = events["description"].str.lower()
    scheduled_types = {"holiday", "additional", "bridge", "transfer", "work_day"}
    event_mask = events["normalized_type"].eq("event")
    planned_event_mask = event_mask & events["normalized_description"].str.contains(
        "dia de la madre|mundial de futbol|black friday|cyber monday",
        regex=True,
    )
    earthquake_mask = event_mask & events["normalized_description"].str.contains(
        "terremoto manabi", regex=False
    )
    unknown_event_mask = event_mask & ~planned_event_mask & ~earthquake_mask
    if unknown_event_mask.any():
        unknown_descriptions = sorted(
            events.loc[unknown_event_mask, "description"].unique().tolist()
        )
        raise ValueError(
            "holidays_events.csv contains Event descriptions without an accepted "
            f"availability classification: {unknown_descriptions}"
        )

    scheduled_mask = events["normalized_type"].isin(scheduled_types) | planned_event_mask
    scheduled = events.loc[scheduled_mask].copy()

    national = scheduled.loc[scheduled["locale"] == "National"].merge(
        store_locations[["store_nbr"]], how="cross"
    )
    regional = scheduled.loc[scheduled["locale"] == "Regional"].merge(
        store_locations[["store_nbr", "state"]],
        how="inner",
        left_on="locale_name",
        right_on="state",
        validate="many_to_many",
    )
    local = scheduled.loc[scheduled["locale"] == "Local"].merge(
        store_locations[["store_nbr", "city"]],
        how="inner",
        left_on="locale_name",
        right_on="city",
        validate="many_to_many",
    )
    mapped = pd.concat([national, regional, local], ignore_index=True)

    mapped_event_ids = set(mapped["event_id"].unique())
    unmapped_scheduled_events = int(
        (~scheduled["event_id"].isin(mapped_event_ids)).sum()
    )

    active = ~mapped["transferred"]
    normalized_type = mapped["normalized_type"]
    mapped["is_holiday"] = (
        active
        & normalized_type.isin({"holiday", "additional", "bridge", "transfer"})
    ).astype("int8")
    mapped["is_special_work_day"] = (
        active & normalized_type.eq("work_day")
    ).astype("int8")
    mapped["is_holiday_transfer_source"] = (
        normalized_type.eq("holiday") & mapped["transferred"]
    ).astype("int8")
    mapped["is_holiday_transfer_destination"] = (
        active & normalized_type.eq("transfer")
    ).astype("int8")
    mapped["is_planned_event"] = (
        active & normalized_type.eq("event")
    ).astype("int8")

    mapped["is_national_schedule"] = (
        active & (mapped["locale"] == "National")
    ).astype("int8")
    mapped["is_regional_schedule"] = (
        active & (mapped["locale"] == "Regional")
    ).astype("int8")
    mapped["is_local_schedule"] = (
        active & (mapped["locale"] == "Local")
    ).astype("int8")
    aggregations = {column: "max" for column in HOLIDAY_FEATURE_COLUMNS}
    normalized = (
        mapped.groupby(["date", "store_nbr"], as_index=False, observed=True)
        .agg(aggregations)
        .sort_values(["date", "store_nbr"])
    )

    evidence = {
        "source_calendar_rows": int(len(events)),
        "source_type_event_rows": int(event_mask.sum()),
        "scheduled_source_rows": int(len(scheduled)),
        "planned_type_event_rows": int(planned_event_mask.sum()),
        "earthquake_event_rows_excluded": int(earthquake_mask.sum()),
        "unknown_type_event_rows": int(unknown_event_mask.sum()),
        "mapped_scheduled_store_rows": int(len(mapped)),
        "unmapped_scheduled_source_rows": unmapped_scheduled_events,
        "normalized_store_date_rows": int(len(normalized)),
        "source_rows_with_transferred_true": int(events["transferred"].sum()),
        "normalized_duplicate_store_date_rows": int(
            normalized.duplicated(["date", "store_nbr"], keep=False).sum()
        ),
        "maximum_active_events_on_one_store_date": int(
            mapped.loc[active].groupby(["date", "store_nbr"], observed=True).size().max()
        ),
    }
    return normalized, evidence


def attach_many_to_one(
    table_name: str,
    base: pd.DataFrame,
    support: pd.DataFrame,
    keys: list[str],
    added_columns: list[str],
    nullable_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Attach one validated support table without changing base rows or order."""
    if support.duplicated(keys).any():
        raise ValueError(f"{table_name} support table is not unique on {keys}.")
    original_ids = base["id"].to_numpy(copy=True)
    joined = base.merge(
        support,
        on=keys,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if len(joined) != len(base):
        raise RuntimeError(f"{table_name} changed the number of base rows.")
    if not np.array_equal(original_ids, joined["id"].to_numpy()):
        raise RuntimeError(f"{table_name} changed original id order.")
    if joined.duplicated(BASE_KEY).any():
        raise RuntimeError(f"{table_name} introduced duplicate base keys.")
    nullable = set(nullable_columns or [])
    required_columns = [column for column in added_columns if column not in nullable]
    if joined[required_columns].isna().any().any():
        raise RuntimeError(f"{table_name} left missing values in required columns.")
    return joined


def attach_all_supporting_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    oil: pd.DataFrame,
    transactions: pd.DataFrame,
    holidays: pd.DataFrame,
    stores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run all non-model preprocessing while preserving the two base tables."""
    train_result = add_calendar_features(train)
    test_result = add_calendar_features(test)

    all_dates = pd.concat([train_result["date"], test_result["date"]], ignore_index=True)
    oil_lookup, oil_evidence = build_oil_lookup(all_dates, oil)
    train_result = attach_many_to_one(
        "train oil join",
        train_result,
        oil_lookup,
        ["date"],
        OIL_FEATURE_COLUMNS,
        OIL_NULLABLE_COLUMNS,
    )
    test_result = attach_many_to_one(
        "test oil join",
        test_result,
        oil_lookup,
        ["date"],
        OIL_FEATURE_COLUMNS,
        OIL_NULLABLE_COLUMNS,
    )

    all_store_dates = pd.concat(
        [
            train_result[["date", "store_nbr"]],
            test_result[["date", "store_nbr"]],
        ],
        ignore_index=True,
    )
    transaction_lookup, transaction_evidence = build_transaction_lookup(
        all_store_dates, transactions
    )
    train_result = attach_many_to_one(
        "train transaction join",
        train_result,
        transaction_lookup,
        ["date", "store_nbr"],
        TRANSACTION_FEATURE_COLUMNS,
        TRANSACTION_NULLABLE_COLUMNS,
    )
    test_result = attach_many_to_one(
        "test transaction join",
        test_result,
        transaction_lookup,
        ["date", "store_nbr"],
        TRANSACTION_FEATURE_COLUMNS,
        TRANSACTION_NULLABLE_COLUMNS,
    )

    holiday_lookup, holiday_evidence = normalize_holidays(holidays, stores)
    train_result = train_result.merge(
        holiday_lookup,
        on=["date", "store_nbr"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    test_result = test_result.merge(
        holiday_lookup,
        on=["date", "store_nbr"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    for table in [train_result, test_result]:
        table[HOLIDAY_FEATURE_COLUMNS] = (
            table[HOLIDAY_FEATURE_COLUMNS].fillna(0).astype("int16")
        )

    evidence = {
        "training_cutoff": train["date"].max().date().isoformat(),
        "forecast_horizon_days": min(SAFE_LAGS),
        "oil": oil_evidence,
        "transactions": transaction_evidence,
        "holidays": holiday_evidence,
    }
    return train_result, test_result, evidence


def validate_final_table(
    name: str,
    original: pd.DataFrame,
    processed: pd.DataFrame,
    has_target: bool,
) -> dict[str, Any]:
    """Prove that preprocessing kept the prediction population intact."""
    if len(processed) != len(original):
        raise RuntimeError(f"{name} row count changed.")
    if not np.array_equal(original["id"].to_numpy(), processed["id"].to_numpy()):
        raise RuntimeError(f"{name} id order changed.")
    if processed.duplicated(BASE_KEY).any():
        raise RuntimeError(f"{name} contains duplicate base keys.")
    missing_by_column = processed.isna().sum()
    unexpected_missing = missing_by_column.loc[
        (missing_by_column > 0)
        & (~missing_by_column.index.isin(ALLOWED_MISSING_COLUMNS))
    ]
    if not unexpected_missing.empty:
        raise RuntimeError(
            f"{name} contains unexpected missing values: "
            f"{unexpected_missing.to_dict()}"
        )
    if has_target and not np.array_equal(
        original["sales"].to_numpy(), processed["sales"].to_numpy()
    ):
        raise RuntimeError("The sales target changed during preprocessing.")
    return {
        "input_rows": int(len(original)),
        "output_rows": int(len(processed)),
        "output_columns": int(processed.shape[1]),
        "expected_missing_feature_cells": int(missing_by_column.sum()),
        "unexpected_missing_cells": int(unexpected_missing.sum()),
        "duplicate_base_key_rows": int(
            processed.duplicated(BASE_KEY, keep=False).sum()
        ),
        "id_order_preserved": True,
        "target_preserved": has_target,
    }


def write_plain_csv(table: pd.DataFrame, output_path: Path) -> None:
    """Write one processed table as an ordinary, easy-to-find CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    if temporary_path.exists():
        raise FileExistsError(
            f"Temporary EDA output already exists and was not overwritten: {temporary_path}"
        )
    try:
        table.to_csv(temporary_path, index=False, date_format="%Y-%m-%d")
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    eda_csv_path = args.eda_csv_path.resolve()
    test_csv_path = args.test_csv_path.resolve()
    project_root = PROJECT_ROOT.resolve()

    if not raw_dir.is_relative_to(project_root):
        raise ValueError("--raw-dir must stay inside the project directory.")
    if not eda_csv_path.is_relative_to(project_root):
        raise ValueError("--eda-csv-path must stay inside the project directory.")
    if not test_csv_path.is_relative_to(project_root):
        raise ValueError("--test-csv-path must stay inside the project directory.")

    source_paths = {
        "train": raw_dir / "train.csv",
        "competition_test": raw_dir / "test.csv",
        "stores": raw_dir / "stores.csv",
        "oil": raw_dir / "oil.csv",
        "transactions": raw_dir / "transactions.csv",
        "holidays": raw_dir / "holidays_events.csv",
    }
    missing_sources = [path.name for path in source_paths.values() if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError("Missing source files: " + ", ".join(missing_sources))

    output_paths = {
        "eda_csv": eda_csv_path,
        "competition_test": test_csv_path,
    }
    existing_outputs = [path.name for path in output_paths.values() if path.exists()]
    if existing_outputs and not args.overwrite:
        print("Preprocessing stopped because final outputs already exist.")
        print("Existing outputs: " + ", ".join(existing_outputs))
        print("Inspect them first, or rerun with --overwrite intentionally.")
        return 2
    raw_train = read_base_table(source_paths["train"], TRAIN_COLUMNS)
    raw_test = read_base_table(source_paths["competition_test"], TEST_COLUMNS)
    stores = read_store_table(source_paths["stores"])
    oil = read_oil(source_paths["oil"])
    transactions = read_transactions(source_paths["transactions"])
    holidays = read_holidays(source_paths["holidays"])

    train_with_stores, _ = join_store_metadata("train.csv", raw_train, stores)
    test_with_stores, _ = join_store_metadata("test.csv", raw_test, stores)
    processed_train, processed_test, _ = attach_all_supporting_data(
        train_with_stores,
        test_with_stores,
        oil,
        transactions,
        holidays,
        stores,
    )

    validate_final_table("train", raw_train, processed_train, has_target=True)
    validate_final_table(
        "competition test", raw_test, processed_test, has_target=False
    )

    write_plain_csv(processed_train, output_paths["eda_csv"])
    write_plain_csv(processed_test, output_paths["competition_test"])

    print("Complete Store Sales preprocessing: PASS")
    print(
        f"Train: {len(processed_train):,} rows x {processed_train.shape[1]} columns; "
        f"expected lag gaps: {processed_train.isna().sum().sum():,}"
    )
    print(
        f"Competition test: {len(processed_test):,} rows x {processed_test.shape[1]} columns; "
        f"expected lag gaps: {processed_test.isna().sum().sum():,}"
    )
    print(f"EDA and training CSV: {eda_csv_path}")
    print(f"Kaggle inference CSV: {test_csv_path}")
    print("Next stage is model feature engineering and chronological splitting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
