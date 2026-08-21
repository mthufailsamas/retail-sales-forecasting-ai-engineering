"""Train, save, load, and reuse the frozen Store Sales forecasting model.

Notebook 03 remains the experiment record. This module owns the stable feature,
model, artifact, and 16-day batch-inference contracts used after model selection.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost
from xgboost import XGBRegressor

from store_sales_preprocessing import (
    BASE_KEY,
    CALENDAR_COLUMNS,
    HOLIDAY_FEATURE_COLUMNS,
    OIL_AGE_COLUMNS,
    OIL_FEATURE_COLUMNS,
    OIL_LAG_COLUMNS,
    SAFE_LAGS,
    STORE_OUTPUT_COLUMNS,
    TEST_COLUMNS,
    TRAIN_COLUMNS,
    TRANSACTION_FEATURE_COLUMNS,
    TRANSACTION_LAG_COLUMNS,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "data" / "processed" / "00_STORE_SALES_EDA.csv"
DEFAULT_FUTURE_PATH = (
    PROJECT_ROOT / "data" / "processed" / "01_STORE_SALES_KAGGLE_TEST.csv"
)
DEFAULT_ARTIFACT_PATH = (
    PROJECT_ROOT / "artifacts" / "store_sales_forecast_v1.pkl"
)
DEFAULT_DEPLOYMENT_HISTORY_PATH = (
    PROJECT_ROOT / "artifacts" / "store_sales_forecast_v1_history.csv.gz"
)
DEFAULT_BATCH_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "03_STORE_SALES_BATCH_FORECAST.csv"
)

ARTIFACT_SCHEMA_VERSION = 1
MODEL_VERSION = "store-sales-forecast-v1"
FORECAST_HORIZON_DAYS = 16
RANDOM_STATE = 42
CURRENT_LIBRARY_VERSIONS = {
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
    "xgboost": xgboost.__version__,
}

SALES_LAGS = list(SAFE_LAGS)
SALES_LAG_COLUMNS = [f"sales_lag_{lag}" for lag in SALES_LAGS]
SALES_SUMMARY_COLUMNS = ["sales_lag_available_count"]
INFERENCE_HISTORY_COLUMNS = [*BASE_KEY, "sales"]

CATEGORICAL_FEATURES = [
    "store_nbr",
    "family",
    "city",
    "state",
    "store_type",
    "store_cluster",
    *CALENDAR_COLUMNS,
]
NUMERIC_FEATURES = (
    ["onpromotion"]
    + OIL_FEATURE_COLUMNS
    + TRANSACTION_FEATURE_COLUMNS
    + HOLIDAY_FEATURE_COLUMNS
    + SALES_LAG_COLUMNS
    + SALES_SUMMARY_COLUMNS
)
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

MODEL_GRIDS = {
    "Ridge Regression": {"alpha": [0.1, 1.0, 10.0]},
    "XGBoost Regression": {
        "n_estimators": [100, 300, 500],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.10, 0.30],
    },
}
LABELED_COLUMNS = (
    TRAIN_COLUMNS
    + STORE_OUTPUT_COLUMNS
    + CALENDAR_COLUMNS
    + OIL_FEATURE_COLUMNS
    + TRANSACTION_FEATURE_COLUMNS
    + HOLIDAY_FEATURE_COLUMNS
)
FUTURE_COLUMNS = (
    TEST_COLUMNS
    + STORE_OUTPUT_COLUMNS
    + CALENDAR_COLUMNS
    + OIL_FEATURE_COLUMNS
    + TRANSACTION_FEATURE_COLUMNS
    + HOLIDAY_FEATURE_COLUMNS
)

CATEGORY_COLUMNS = ["family", "city", "state", "store_type"]
INTEGER_COLUMNS = [
    "store_nbr",
    "store_cluster",
    *CALENDAR_COLUMNS,
    "transactions_lag_available_count",
    *HOLIDAY_FEATURE_COLUMNS,
]
FLOAT_COLUMNS = [
    *OIL_LAG_COLUMNS,
    *OIL_AGE_COLUMNS,
    *TRANSACTION_LAG_COLUMNS,
]
COMMON_DTYPES = {
    "id": "int32",
    "store_nbr": "int16",
    "onpromotion": "int32",
    **{column: "category" for column in CATEGORY_COLUMNS},
    **{column: "int16" for column in INTEGER_COLUMNS},
    **{column: "float32" for column in FLOAT_COLUMNS},
}


def read_processed_table(path: Path, has_target: bool) -> pd.DataFrame:
    """Read one processed interface and enforce its exact saved schema."""
    expected_columns = LABELED_COLUMNS if has_target else FUTURE_COLUMNS
    saved_columns = pd.read_csv(path, nrows=0).columns.tolist()
    if saved_columns != expected_columns:
        raise ValueError(
            f"{path.name} does not match the processed feature contract. "
            "Run notebook 01 again before training or inference."
        )

    dtypes = {**COMMON_DTYPES, **({"sales": "float32"} if has_target else {})}
    table = pd.read_csv(path, dtype=dtypes, parse_dates=["date"])
    if table.empty:
        raise ValueError(f"{path.name} is empty.")
    if table[BASE_KEY].isna().any().any() or table.duplicated(BASE_KEY).any():
        raise ValueError(f"{path.name} has an invalid {BASE_KEY} key.")
    if table["id"].isna().any() or not table["id"].is_unique:
        raise ValueError(f"{path.name} id must be present and unique.")
    if has_target:
        if table["sales"].isna().any() or (table["sales"] < 0).any():
            raise ValueError(f"{path.name} contains an invalid sales target.")
    elif "sales" in table.columns:
        raise ValueError(f"{path.name} must not contain future sales.")
    return table


def read_inference_history(
    path: Path,
    training_end: pd.Timestamp,
    *,
    chunk_size: int = 250_000,
) -> pd.DataFrame:
    """Read only the 35-day sales context required by the API runtime."""
    saved_columns = pd.read_csv(path, nrows=0).columns.tolist()
    if saved_columns not in [LABELED_COLUMNS, INFERENCE_HISTORY_COLUMNS]:
        raise ValueError(
            f"{path.name} does not match the labeled or compact history contract."
        )

    training_end = pd.Timestamp(training_end).normalize()
    history_start = training_end - pd.Timedelta(days=max(SALES_LAGS) - 1)
    use_columns = list(INFERENCE_HISTORY_COLUMNS)
    retained_chunks: list[pd.DataFrame] = []
    observed_max: pd.Timestamp | None = None
    for chunk in pd.read_csv(
        path,
        usecols=use_columns,
        dtype={
            "store_nbr": "int16",
            "family": "string",
            "sales": "float32",
        },
        parse_dates=["date"],
        chunksize=chunk_size,
    ):
        chunk_max = chunk["date"].max()
        if pd.notna(chunk_max):
            observed_max = (
                max(observed_max, chunk_max)
                if observed_max is not None
                else chunk_max
            )
        retained = chunk.loc[chunk["date"].between(history_start, training_end)]
        if not retained.empty:
            retained_chunks.append(retained)

    if observed_max != training_end:
        raise ValueError(
            "History end date differs from the artifact training cutoff. "
            "Retrain the artifact or supply its matching history."
        )
    if not retained_chunks:
        raise ValueError("No eligible sales history is available for inference.")

    history = pd.concat(retained_chunks, ignore_index=True)
    if history[BASE_KEY].isna().any().any() or history.duplicated(BASE_KEY).any():
        raise ValueError("Inference history has an invalid date-store-family key.")
    if history["sales"].isna().any() or (history["sales"] < 0).any():
        raise ValueError("Inference history contains an invalid sales target.")
    return history


def write_deployment_history(
    history: pd.DataFrame,
    training_end: pd.Timestamp,
    output_path: Path = DEFAULT_DEPLOYMENT_HISTORY_PATH,
    *,
    overwrite: bool = False,
) -> Path:
    """Write the private 35-day sales context required by the serving API."""
    missing_columns = [
        column for column in INFERENCE_HISTORY_COLUMNS if column not in history
    ]
    if missing_columns:
        raise ValueError(f"Deployment history is missing columns: {missing_columns}")

    training_end = pd.Timestamp(training_end).normalize()
    history_start = training_end - pd.Timedelta(days=max(SALES_LAGS) - 1)
    compact_history = history.loc[
        history["date"].between(history_start, training_end),
        INFERENCE_HISTORY_COLUMNS,
    ].copy()
    compact_history = compact_history.sort_values(BASE_KEY).reset_index(drop=True)
    if compact_history.empty or compact_history["date"].max() != training_end:
        raise ValueError(
            "Deployment history does not end at the artifact training cutoff."
        )
    if (
        compact_history[BASE_KEY].isna().any().any()
        or compact_history.duplicated(BASE_KEY).any()
    ):
        raise ValueError("Deployment history has an invalid date-store-family key.")
    if (
        compact_history["sales"].isna().any()
        or (compact_history["sales"] < 0).any()
    ):
        raise ValueError("Deployment history contains an invalid sales target.")

    output_path = _require_project_path(output_path, "Deployment history path")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Deployment history already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    if temporary_path.exists():
        raise FileExistsError(
            f"Temporary deployment history already exists: {temporary_path}"
        )
    try:
        compact_history.to_csv(
            temporary_path,
            index=False,
            date_format="%Y-%m-%d",
            compression="gzip",
        )
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def add_exact_sales_lags(
    target_rows: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Attach calendar-aligned sales history known before a 16-day forecast."""
    required_history = [*BASE_KEY, "sales"]
    missing_history = [column for column in required_history if column not in history]
    if missing_history:
        raise ValueError(f"Sales history is missing columns: {missing_history}")
    if history.duplicated(BASE_KEY).any():
        raise ValueError("Sales history must be unique by date, store, and family.")

    result = target_rows.copy()
    sales_lookup = history.set_index(BASE_KEY)["sales"]
    for lag in SALES_LAGS:
        lookup_key = pd.MultiIndex.from_arrays(
            [
                result["date"] - pd.Timedelta(days=lag),
                result["store_nbr"],
                result["family"],
            ],
            names=BASE_KEY,
        )
        result[f"sales_lag_{lag}"] = sales_lookup.reindex(lookup_key).to_numpy(
            dtype=np.float32
        )

    result["sales_lag_available_count"] = (
        result[SALES_LAG_COLUMNS].notna().sum(axis=1).astype("int8")
    )
    return result


def find_model_start(labeled_features: pd.DataFrame) -> pd.Timestamp:
    """Return the first date with complete required sales and oil history."""
    required_columns = SALES_LAG_COLUMNS + OIL_LAG_COLUMNS
    missing_columns = [
        column for column in required_columns if column not in labeled_features
    ]
    if missing_columns:
        raise ValueError(f"Warm-up columns are missing: {missing_columns}")
    complete_rows = labeled_features[required_columns].notna().all(axis=1)
    model_start = labeled_features.loc[complete_rows, "date"].min()
    if pd.isna(model_start):
        raise ValueError("No date has complete required sales and oil history.")
    return pd.Timestamp(model_start)


def make_feature_processor() -> ColumnTransformer:
    """Build the train-fitted encoding, imputation, and scaling pipeline."""
    categorical_pipeline = Pipeline(
        steps=[
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
            )
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ],
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def build_model(method: str, parameters: dict[str, float | int]):
    """Build one of the two accepted experiment methods."""
    if method == "Ridge Regression":
        return Ridge(
            alpha=parameters["alpha"],
            solver="lsqr",
            tol=1e-4,
            max_iter=1_000,
        )
    if method == "XGBoost Regression":
        return XGBRegressor(
            **parameters,
            objective="reg:squarederror",
            tree_method="hist",
            max_bin=256,
            subsample=1.0,
            colsample_bytree=1.0,
            min_child_weight=1.0,
            gamma=0.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model method: {method}")


def is_default_reference(method: str, parameters: dict[str, float | int]) -> bool:
    """Identify the library-default comparison inside each accepted grid."""
    if method == "Ridge Regression":
        return parameters == {"alpha": 1.0}
    return parameters == {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.30,
    }


def evaluate_forecast(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate the fixed local evaluation metrics."""
    actual_values = np.asarray(actual, dtype=np.float64)
    predicted_values = np.clip(np.asarray(predicted, dtype=np.float64), 0.0, None)
    error = predicted_values - actual_values
    denominator = actual_values.sum()
    if denominator <= 0:
        raise ValueError("Aggregate actual sales must be positive.")
    return {
        "rmsle": float(
            np.sqrt(
                np.mean(
                    (np.log1p(predicted_values) - np.log1p(actual_values)) ** 2
                )
            )
        ),
        "wape_pct": float(np.abs(error).sum() / denominator * 100),
        "signed_bias_pct": float(error.sum() / denominator * 100),
        "underforecast_pct": float(
            np.maximum(actual_values - predicted_values, 0.0).sum()
            / denominator
            * 100
        ),
        "overforecast_pct": float(
            np.maximum(predicted_values - actual_values, 0.0).sum()
            / denominator
            * 100
        ),
    }


def validate_model_features(table: pd.DataFrame, table_name: str) -> None:
    """Require the complete frozen v1 model allowlist."""
    missing_columns = [column for column in MODEL_FEATURES if column not in table]
    if missing_columns:
        raise ValueError(f"{table_name} is missing model features: {missing_columns}")


def validate_store_family_coverage(
    future_features: pd.DataFrame,
    history: pd.DataFrame,
) -> None:
    """Require the future batch to preserve the complete historical pair set."""
    pair_columns = ["store_nbr", "family"]
    missing_future = [column for column in pair_columns if column not in future_features]
    missing_history = [column for column in pair_columns if column not in history]
    if missing_future or missing_history:
        raise ValueError(
            "Store-family coverage requires store_nbr and family in both "
            "history and the future batch."
        )

    expected_pair_table = history[pair_columns].drop_duplicates().copy()
    future_pair_table = future_features[pair_columns].drop_duplicates().copy()
    for pair_table in [expected_pair_table, future_pair_table]:
        pair_table["store_nbr"] = pair_table["store_nbr"].astype("int64")
        pair_table["family"] = pair_table["family"].astype("string")

    expected_pairs = pd.MultiIndex.from_frame(expected_pair_table)
    future_pairs = pd.MultiIndex.from_frame(future_pair_table)
    missing_pairs = expected_pairs.difference(future_pairs)
    unexpected_pairs = future_pairs.difference(expected_pairs)
    if len(missing_pairs) or len(unexpected_pairs):
        raise ValueError(
            "Future store-family coverage differs from history: "
            f"{len(missing_pairs)} missing and "
            f"{len(unexpected_pairs)} unexpected pairs."
        )


def validate_forecast_window(
    future_features: pd.DataFrame,
    training_end: pd.Timestamp,
) -> None:
    """Reject a future batch that violates the fixed 16-day cutoff contract."""
    validate_model_features(future_features, "Future batch")
    if "sales" in future_features.columns:
        raise ValueError("Future batch must not contain the sales target.")
    if future_features[BASE_KEY].isna().any().any():
        raise ValueError("Future batch contains a missing base-key value.")
    if future_features.duplicated(BASE_KEY).any():
        raise ValueError("Future batch contains duplicate base keys.")
    if future_features["id"].isna().any() or not future_features["id"].is_unique:
        raise ValueError("Future batch id must be present and unique.")

    dates = pd.DatetimeIndex(future_features["date"].drop_duplicates().sort_values())
    expected_dates = pd.date_range(
        training_end + pd.Timedelta(days=1),
        periods=FORECAST_HORIZON_DAYS,
        freq="D",
    )
    if not dates.equals(expected_dates):
        raise ValueError(
            "Future batch must contain the 16 consecutive dates immediately "
            f"after {training_end.date().isoformat()}."
        )

    pair_date_counts = future_features.groupby(
        ["store_nbr", "family"], observed=True
    )["date"].nunique()
    if not pair_date_counts.eq(FORECAST_HORIZON_DAYS).all():
        raise ValueError("Every store-family pair must appear on all 16 forecast dates.")

    known_columns = CATEGORICAL_FEATURES + ["onpromotion"] + HOLIDAY_FEATURE_COLUMNS
    if future_features[known_columns].isna().any().any():
        raise ValueError("Future-known context contains missing values.")
    if not future_features["month"].astype(int).eq(future_features["date"].dt.month).all():
        raise ValueError("Future month values do not match date.")
    if not future_features["day_of_month"].astype(int).eq(
        future_features["date"].dt.day
    ).all():
        raise ValueError("Future day_of_month values do not match date.")
    expected_day_of_week = future_features["date"].dt.dayofweek + 1
    if not future_features["day_of_week"].astype(int).eq(expected_day_of_week).all():
        raise ValueError("Future day_of_week values do not match date.")


def fit_forecast_bundle(
    training_features: pd.DataFrame,
    method: str,
    parameters: dict[str, float | int],
    evaluation_reference: dict[str, Any],
) -> dict[str, Any]:
    """Fit the frozen processor and model, then return one versioned bundle."""
    validate_model_features(training_features, "Training data")
    if "sales" not in training_features:
        raise ValueError("Training data must contain sales.")
    if training_features["sales"].isna().any() or (training_features["sales"] < 0).any():
        raise ValueError("Training sales must be complete and non-negative.")
    if not evaluation_reference:
        raise ValueError("Artifact metadata requires the current-run evaluation reference.")

    processor = make_feature_processor()
    training_matrix = processor.fit_transform(
        training_features[MODEL_FEATURES]
    ).astype(np.float32)
    model = build_model(method, parameters)
    model.fit(
        training_matrix,
        np.log1p(training_features["sales"].to_numpy(dtype=np.float32)),
    )

    numeric_imputer = processor.named_transformers_["numeric"].named_steps["imputer"]
    indicator_sources = [
        NUMERIC_FEATURES[index] for index in numeric_imputer.indicator_.features_
    ]
    metadata = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "parameters": dict(parameters),
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "safe_lags": list(SALES_LAGS),
        "target_transform": "log1p",
        "prediction_floor": 0.0,
        "training_rows": int(len(training_features)),
        "training_start": training_features["date"].min().date().isoformat(),
        "training_end": training_features["date"].max().date().isoformat(),
        "model_features": list(MODEL_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "transformed_feature_count": int(training_matrix.shape[1]),
        "missing_indicator_sources": indicator_sources,
        "reference_evaluation": dict(evaluation_reference),
        "library_versions": dict(CURRENT_LIBRARY_VERSIONS),
    }
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "metadata": metadata,
        "processor": processor,
        "model": model,
    }


def validate_forecast_bundle(bundle: dict[str, Any]) -> None:
    """Reject an artifact that differs from the current executable contract."""
    required_keys = {"artifact_schema_version", "metadata", "processor", "model"}
    if set(bundle) != required_keys:
        raise ValueError("Forecast artifact has unexpected top-level fields.")
    if not callable(getattr(bundle["processor"], "transform", None)):
        raise ValueError("Forecast artifact processor has no transform interface.")
    if not callable(getattr(bundle["model"], "predict", None)):
        raise ValueError("Forecast artifact model has no predict interface.")
    if bundle["artifact_schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Forecast artifact schema version is unsupported.")
    metadata = bundle["metadata"]
    if metadata.get("model_version") != MODEL_VERSION:
        raise ValueError("Forecast artifact model version is unsupported.")
    if metadata.get("model_features") != MODEL_FEATURES:
        raise ValueError("Forecast artifact feature order differs from the v1 contract.")
    if metadata.get("forecast_horizon_days") != FORECAST_HORIZON_DAYS:
        raise ValueError("Forecast artifact horizon differs from the v1 contract.")
    if metadata.get("library_versions") != CURRENT_LIBRARY_VERSIONS:
        raise ValueError(
            "Forecast artifact library versions differ from the current environment."
        )


def predict_forecast(
    bundle: dict[str, Any],
    future_features: pd.DataFrame,
) -> pd.DataFrame:
    """Validate and predict one complete future 16-day batch."""
    validate_forecast_bundle(bundle)
    training_end = pd.Timestamp(bundle["metadata"]["training_end"])
    validate_forecast_window(future_features, training_end)

    future_matrix = bundle["processor"].transform(
        future_features[MODEL_FEATURES]
    ).astype(np.float32)
    predicted_log = bundle["model"].predict(future_matrix)
    predicted_sales = np.clip(np.expm1(predicted_log), 0.0, None)
    if len(predicted_sales) != len(future_features):
        raise RuntimeError("Prediction row count differs from the future batch.")
    if not np.isfinite(predicted_sales).all():
        raise RuntimeError("Forecast contains a non-finite value.")

    output = future_features[["id", "date", "store_nbr", "family"]].copy()
    output["forecast_sales"] = predicted_sales.astype(np.float32)
    return output


def _require_project_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(f"{label} must stay inside the project directory.")
    return resolved


def save_forecast_artifact(
    bundle: dict[str, Any],
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Write a trusted local pickle and a human-readable metadata manifest."""
    validate_forecast_bundle(bundle)
    artifact_path = _require_project_path(artifact_path, "Artifact path")
    metadata_path = artifact_path.with_suffix(".json")
    existing = [path for path in [artifact_path, metadata_path] if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Artifact output already exists: " + ", ".join(str(path) for path in existing)
        )

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_temp = artifact_path.with_name(f".{artifact_path.name}.tmp")
    metadata_temp = metadata_path.with_name(f".{metadata_path.name}.tmp")
    if artifact_temp.exists() or metadata_temp.exists():
        raise FileExistsError("A temporary artifact output already exists.")
    try:
        with artifact_temp.open("wb") as handle:
            pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        metadata_temp.write_text(
            json.dumps(bundle["metadata"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact_temp.replace(artifact_path)
        metadata_temp.replace(metadata_path)
    finally:
        if artifact_temp.exists():
            artifact_temp.unlink()
        if metadata_temp.exists():
            metadata_temp.unlink()
    return artifact_path, metadata_path


def load_forecast_artifact(
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
) -> dict[str, Any]:
    """Load a trusted project artifact; never load an untrusted pickle file."""
    artifact_path = _require_project_path(artifact_path, "Artifact path")
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Forecast artifact not found: {artifact_path}")
    with artifact_path.open("rb") as handle:
        bundle = pickle.load(handle)
    validate_forecast_bundle(bundle)
    return bundle


def write_batch_forecast(
    forecast: pd.DataFrame,
    output_path: Path = DEFAULT_BATCH_OUTPUT_PATH,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one planning-friendly row-level forecast with an atomic replace."""
    expected_columns = ["id", "date", "store_nbr", "family", "forecast_sales"]
    if forecast.columns.tolist() != expected_columns:
        raise ValueError("Batch forecast columns differ from the output contract.")
    if forecast.empty:
        raise ValueError("Batch forecast is empty.")
    forecast_values = forecast["forecast_sales"].to_numpy(dtype=np.float64)
    if not np.isfinite(forecast_values).all():
        raise ValueError("Batch forecast contains a non-finite prediction.")
    if (forecast_values < 0).any():
        raise ValueError("Batch forecast contains a negative prediction.")

    output_path = _require_project_path(output_path, "Batch output path")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Batch output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    if temporary_path.exists():
        raise FileExistsError(f"Temporary batch output already exists: {temporary_path}")
    try:
        forecast.to_csv(temporary_path, index=False, date_format="%Y-%m-%d")
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def run_batch_inference(
    artifact_path: Path,
    history_path: Path,
    future_path: Path,
    output_path: Path,
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Load processed inputs and a saved model, then write a 16-day forecast."""
    history = read_processed_table(history_path, has_target=True)
    future = read_processed_table(future_path, has_target=False)
    bundle = load_forecast_artifact(artifact_path)
    artifact_training_end = pd.Timestamp(bundle["metadata"]["training_end"])
    if history["date"].max() != artifact_training_end:
        raise ValueError(
            "History end date differs from the artifact training cutoff. "
            "Retrain the artifact or supply its matching history."
        )
    validate_store_family_coverage(future, history)
    future_features = add_exact_sales_lags(future, history)
    forecast = predict_forecast(bundle, future_features)
    write_batch_forecast(forecast, output_path, overwrite=overwrite)
    return forecast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one saved-model batch or prepare its private deployment history."
        )
    )
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--future", type=Path, default=DEFAULT_FUTURE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_BATCH_OUTPUT_PATH)
    parser.add_argument(
        "--deployment-history",
        type=Path,
        default=DEFAULT_DEPLOYMENT_HISTORY_PATH,
    )
    parser.add_argument(
        "--prepare-deployment-history",
        action="store_true",
        help="Write only the compact 35-day private history used by the API.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing generated batch output intentionally.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prepare_deployment_history:
        bundle = load_forecast_artifact(args.artifact)
        training_end = pd.Timestamp(bundle["metadata"]["training_end"])
        history = read_inference_history(args.history, training_end)
        deployment_history_path = write_deployment_history(
            history,
            training_end,
            args.deployment_history,
            overwrite=args.overwrite,
        )
        print("Private deployment history: PASS")
        print(
            f"Rows: {len(history):,}; dates: "
            f"{history['date'].min().date().isoformat()} to "
            f"{history['date'].max().date().isoformat()}"
        )
        print(f"Output: {deployment_history_path.resolve()}")
        return 0

    forecast = run_batch_inference(
        artifact_path=args.artifact,
        history_path=args.history,
        future_path=args.future,
        output_path=args.output,
        overwrite=args.overwrite,
    )
    print("Reusable 16-day batch inference: PASS")
    print(
        f"Rows: {len(forecast):,}; dates: "
        f"{forecast['date'].min().date().isoformat()} to "
        f"{forecast['date'].max().date().isoformat()}"
    )
    print(f"Output: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
