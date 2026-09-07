"""Run the fixed multi-origin historical evaluation for Store Sales V1.

The command refits the accepted V1 configuration at four historical forecast
origins, compares it with a weekly seasonal-naive reference, and writes one
private, reproducible evaluation bundle under ``artifacts/evaluation``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd

import store_sales_model
import store_sales_preprocessing
from store_sales_model import (
    CURRENT_LIBRARY_VERSIONS,
    DEFAULT_HISTORY_PATH,
    FORECAST_HORIZON_DAYS,
    MODEL_FEATURES,
    PROJECT_ROOT,
    SALES_LAGS,
    add_exact_sales_lags,
    find_model_start,
    fit_forecast_bundle,
    predict_forecast,
    read_processed_table,
    validate_forecast_window,
    validate_store_family_coverage,
)
from store_sales_preprocessing import BASE_KEY


CONTRACT_ID = "retail-history-eval-01"
DEFAULT_RUN_ID = f"{CONTRACT_ID}-v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "evaluation"
V1_MODEL_NAME = "xgboost_v1"
SEASONAL_NAIVE_NAME = "weekly_seasonal_naive"
V1_METHOD = "XGBoost Regression"
V1_PARAMETERS: dict[str, float | int] = {
    "learning_rate": 0.05,
    "max_depth": 8,
    "n_estimators": 500,
}
PREDICTION_COLUMNS = ["id", *BASE_KEY, "forecast_sales"]
ACTUAL_CONTEXT_COLUMNS = ["id", *BASE_KEY, "sales", "onpromotion", "is_holiday"]
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class EvaluationWindow:
    """One predeclared 16-day retrospective forecast window."""

    window_id: str
    forecast_origin: str
    scoring_start: str
    scoring_end: str

    @property
    def origin(self) -> pd.Timestamp:
        return pd.Timestamp(self.forecast_origin)

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.scoring_start)

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.scoring_end)


@dataclass(frozen=True)
class WindowSelectionBlock:
    """One target-free calendar block and its predeclared selection strategy."""

    window_id: str
    role: str
    candidate_start: str
    candidate_end: str
    strategy: str

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.candidate_start)

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.candidate_end)


EVALUATION_WINDOWS = (
    EvaluationWindow("W1", "2016-08-25", "2016-08-26", "2016-09-10"),
    EvaluationWindow("W2", "2016-11-24", "2016-11-25", "2016-12-10"),
    EvaluationWindow("W3", "2017-02-15", "2017-02-16", "2017-03-03"),
    EvaluationWindow("W4", "2017-06-28", "2017-06-29", "2017-07-14"),
)
WINDOW_SELECTION_SOURCE_COLUMNS = (
    "date",
    "store_nbr",
    "family",
    "onpromotion",
    "is_holiday",
    "is_planned_event",
)
WINDOW_SELECTION_BLOCKS = (
    WindowSelectionBlock(
        "W1", "typical_context", "2016-07-01", "2016-09-30", "typical"
    ),
    WindowSelectionBlock(
        "W2",
        "planned_event_and_promotion_stress",
        "2016-10-01",
        "2016-12-31",
        "planned_event_stress",
    ),
    WindowSelectionBlock(
        "W3", "holiday_stress", "2017-01-01", "2017-03-31", "holiday_stress"
    ),
    WindowSelectionBlock(
        "W4", "recent_pre_validation", "2017-04-01", "2017-07-14", "latest"
    ),
)


def validate_window_contract(
    windows: Iterable[EvaluationWindow] = EVALUATION_WINDOWS,
) -> tuple[EvaluationWindow, ...]:
    """Validate the fixed chronological window definition."""
    window_tuple = tuple(windows)
    if not window_tuple:
        raise ValueError("Historical evaluation requires at least one window.")
    if len({window.window_id for window in window_tuple}) != len(window_tuple):
        raise ValueError("Historical evaluation window IDs must be unique.")

    previous_end: pd.Timestamp | None = None
    for window in window_tuple:
        expected_dates = pd.date_range(
            window.origin + pd.Timedelta(days=1),
            periods=FORECAST_HORIZON_DAYS,
            freq="D",
        )
        if window.start != expected_dates.min() or window.end != expected_dates.max():
            raise ValueError(
                f"{window.window_id} must contain the {FORECAST_HORIZON_DAYS} "
                "dates immediately after its forecast origin."
            )
        if previous_end is not None and window.start <= previous_end:
            raise ValueError("Historical evaluation windows must not overlap.")
        previous_end = window.end
    return window_tuple


def build_window_candidates(
    labeled: pd.DataFrame,
    block: WindowSelectionBlock,
) -> pd.DataFrame:
    """Summarize complete 16-day candidates using target-free known context."""
    missing = [
        column for column in WINDOW_SELECTION_SOURCE_COLUMNS if column not in labeled
    ]
    if missing:
        raise ValueError(f"Window selection is missing source columns: {missing}")
    if block.end - block.start < pd.Timedelta(days=FORECAST_HORIZON_DAYS - 1):
        raise ValueError(f"{block.window_id} selection block is shorter than 16 days.")

    pair_count = int(labeled[["store_nbr", "family"]].drop_duplicates().shape[0])
    if pair_count <= 0:
        raise ValueError("Window selection requires at least one store-family pair.")
    daily = labeled.groupby("date", observed=True, sort=True).agg(
        rows=("family", "size"),
        promotion_rows=("onpromotion", lambda values: int(values.gt(0).sum())),
        promotion_units=("onpromotion", "sum"),
        holiday_rows=("is_holiday", "sum"),
        planned_event_rows=("is_planned_event", "sum"),
    )

    records: list[dict[str, Any]] = []
    latest_start = block.end - pd.Timedelta(days=FORECAST_HORIZON_DAYS - 1)
    for start in pd.date_range(block.start, latest_start, freq="D"):
        dates = pd.date_range(start, periods=FORECAST_HORIZON_DAYS, freq="D")
        context = daily.reindex(dates)
        if context["rows"].isna().any() or not context["rows"].eq(pair_count).all():
            continue
        total_rows = float(context["rows"].sum())
        records.append(
            {
                "start": start,
                "end": dates.max(),
                "promotion_share": float(
                    context["promotion_rows"].sum() / total_rows
                ),
                "promotion_units_per_row": float(
                    context["promotion_units"].sum() / total_rows
                ),
                "holiday_share": float(context["holiday_rows"].sum() / total_rows),
                "planned_event_share": float(
                    context["planned_event_rows"].sum() / total_rows
                ),
            }
        )
    if not records:
        raise ValueError(f"{block.window_id} has no complete 16-day candidate window.")
    return pd.DataFrame(records)


def select_window_candidate(
    candidates: pd.DataFrame,
    strategy: str,
) -> pd.Series:
    """Apply one deterministic, target-free ranking rule to candidate windows."""
    required = {
        "start",
        "end",
        "promotion_share",
        "promotion_units_per_row",
        "holiday_share",
        "planned_event_share",
    }
    if candidates.empty or not required.issubset(candidates.columns):
        raise ValueError("Window candidates do not match the selection contract.")
    ranked = candidates.copy()
    if strategy == "typical":
        measures = ["promotion_share", "holiday_share", "planned_event_share"]
        medians = ranked[measures].median()
        scales = ranked[measures].std(ddof=0).replace(0.0, 1.0)
        ranked["selection_score"] = (
            ((ranked[measures] - medians) / scales) ** 2
        ).sum(axis=1)
        ranked = ranked.sort_values(["selection_score", "start"])
    elif strategy == "planned_event_stress":
        ranked = ranked.sort_values(
            [
                "planned_event_share",
                "promotion_share",
                "promotion_units_per_row",
                "holiday_share",
                "start",
            ],
            ascending=[False, False, False, False, True],
        )
    elif strategy == "holiday_stress":
        ranked = ranked.sort_values(
            ["holiday_share", "promotion_share", "promotion_units_per_row", "start"],
            ascending=[False, False, False, True],
        )
    elif strategy == "latest":
        ranked = ranked.sort_values("start", ascending=False)
    else:
        raise ValueError(f"Unknown window selection strategy: {strategy}")
    return ranked.iloc[0]


def derive_window_selection(
    labeled: pd.DataFrame,
) -> tuple[tuple[EvaluationWindow, ...], list[dict[str, Any]]]:
    """Derive and describe the 4 frozen windows without consulting sales."""
    windows: list[EvaluationWindow] = []
    selection_records: list[dict[str, Any]] = []
    for block in WINDOW_SELECTION_BLOCKS:
        candidates = build_window_candidates(labeled, block)
        selected = select_window_candidate(candidates, block.strategy)
        start = pd.Timestamp(selected["start"])
        end = pd.Timestamp(selected["end"])
        windows.append(
            EvaluationWindow(
                block.window_id,
                (start - pd.Timedelta(days=1)).date().isoformat(),
                start.date().isoformat(),
                end.date().isoformat(),
            )
        )
        selection_records.append(
            {
                **asdict(block),
                "candidate_window_count": int(len(candidates)),
                "selected_scoring_start": start.date().isoformat(),
                "selected_scoring_end": end.date().isoformat(),
                "promotion_share": float(selected["promotion_share"]),
                "promotion_units_per_row": float(
                    selected["promotion_units_per_row"]
                ),
                "holiday_share": float(selected["holiday_share"]),
                "planned_event_share": float(selected["planned_event_share"]),
            }
        )
    return tuple(windows), selection_records


def validate_frozen_window_selection(labeled: pd.DataFrame) -> list[dict[str, Any]]:
    """Require the current target-free rules to reproduce the frozen dates."""
    derived_windows, selection_records = derive_window_selection(labeled)
    if derived_windows != EVALUATION_WINDOWS:
        raise RuntimeError(
            "Target-free window selection no longer reproduces the frozen contract."
        )
    return selection_records


def validate_run_id(run_id: str) -> str:
    """Reject output identifiers that are ambiguous or unsafe as directories."""
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "Run ID must start with a letter or digit and contain at most 64 "
            "letters, digits, dots, underscores, or hyphens."
        )
    return run_id


def require_project_path(path: Path, label: str) -> Path:
    """Keep private inputs and outputs inside the project boundary."""
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(f"{label} must stay inside the project directory.")
    return resolved


def sha256_file(path: Path, chunk_size: int = 1_048_576) -> str:
    """Hash a file without loading the complete content into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def make_json_safe(value: Any) -> Any:
    """Convert runtime configuration values into strict JSON primitives."""
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return make_json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def prediction_content_sha256(predictions: pd.DataFrame) -> str:
    """Hash a canonical, uncompressed representation of prediction rows."""
    required = ["window_id", "model", *PREDICTION_COLUMNS]
    missing = [column for column in required if column not in predictions]
    if missing:
        raise ValueError(f"Prediction digest is missing columns: {missing}")
    canonical = predictions[required].copy()
    canonical["date"] = pd.to_datetime(canonical["date"]).dt.strftime("%Y-%m-%d")
    canonical = canonical.sort_values(
        ["window_id", "model", "date", "store_nbr", "family", "id"],
        ignore_index=True,
    )
    content = canonical.to_csv(index=False, float_format="%.9g", lineterminator="\n")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def validate_evaluation_source(
    labeled: pd.DataFrame,
    windows: Iterable[EvaluationWindow] = EVALUATION_WINDOWS,
) -> None:
    """Require every scoring date, key, and store-family series."""
    if labeled.empty:
        raise ValueError("Historical evaluation source is empty.")
    if labeled[BASE_KEY].isna().any().any() or labeled.duplicated(BASE_KEY).any():
        raise ValueError("Historical evaluation source has an invalid base key.")
    if labeled["id"].isna().any() or not labeled["id"].is_unique:
        raise ValueError("Historical evaluation source IDs must be present and unique.")
    sales = labeled["sales"].to_numpy(dtype=np.float64)
    if not np.isfinite(sales).all() or (sales < 0).any():
        raise ValueError("Historical evaluation sales must be finite and non-negative.")

    for window in validate_window_contract(windows):
        scoring = labeled.loc[labeled["date"].between(window.start, window.end)]
        actual_dates = pd.DatetimeIndex(scoring["date"].drop_duplicates().sort_values())
        expected_dates = pd.date_range(
            window.start,
            periods=FORECAST_HORIZON_DAYS,
            freq="D",
        )
        if not actual_dates.equals(expected_dates):
            raise ValueError(f"{window.window_id} is missing required scoring dates.")
        counts = scoring.groupby(
            ["store_nbr", "family"], observed=True
        )["date"].nunique()
        if counts.empty or not counts.eq(FORECAST_HORIZON_DAYS).all():
            raise ValueError(
                f"{window.window_id} does not contain every pair on all scoring dates."
            )
        history = labeled.loc[labeled["date"].le(window.origin)]
        validate_store_family_coverage(scoring, history)


def make_weekly_seasonal_naive(
    future_rows: pd.DataFrame,
    history: pd.DataFrame,
    forecast_origin: pd.Timestamp,
) -> pd.DataFrame:
    """Repeat the latest eligible same-weekday sale, with a pair median fallback."""
    missing_future = [column for column in ["id", *BASE_KEY] if column not in future_rows]
    missing_history = [column for column in [*BASE_KEY, "sales"] if column not in history]
    if missing_future or missing_history:
        raise ValueError(
            "Seasonal naive input is missing required columns: "
            f"future={missing_future}, history={missing_history}."
        )
    if "sales" in future_rows:
        raise ValueError("Seasonal naive future rows must not contain sales.")
    if future_rows[BASE_KEY].isna().any().any() or future_rows.duplicated(BASE_KEY).any():
        raise ValueError("Seasonal naive future rows have an invalid base key.")
    if history[BASE_KEY].isna().any().any() or history.duplicated(BASE_KEY).any():
        raise ValueError("Seasonal naive history has an invalid base key.")

    origin = pd.Timestamp(forecast_origin).normalize()
    if not history.empty and history["date"].max() > origin:
        raise ValueError("Seasonal naive history contains data after the forecast origin.")
    history_start = origin - pd.Timedelta(days=34)
    eligible = history.loc[
        history["date"].between(history_start, origin),
        [*BASE_KEY, "sales"],
    ].copy()
    sales = eligible["sales"].to_numpy(dtype=np.float64)
    if eligible.empty or not np.isfinite(sales).all() or (sales < 0).any():
        raise ValueError("Seasonal naive has no valid eligible history.")

    eligible["day_of_week"] = eligible["date"].dt.dayofweek + 1
    latest_weekday = (
        eligible.sort_values("date")
        .groupby(["store_nbr", "family", "day_of_week"], observed=True)
        .tail(1)
        [["store_nbr", "family", "day_of_week", "sales"]]
        .rename(columns={"sales": "weekday_sales"})
    )
    pair_median = (
        eligible.groupby(["store_nbr", "family"], as_index=False, observed=True)
        ["sales"]
        .median()
        .rename(columns={"sales": "pair_median_sales"})
    )

    result = future_rows[["id", *BASE_KEY]].copy().reset_index(drop=True)
    result["_input_order"] = np.arange(len(result), dtype=np.int64)
    result["day_of_week"] = result["date"].dt.dayofweek + 1
    result = result.merge(
        latest_weekday,
        on=["store_nbr", "family", "day_of_week"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    result = result.merge(
        pair_median,
        on=["store_nbr", "family"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if result["pair_median_sales"].isna().any():
        missing_pairs = int(
            result.loc[result["pair_median_sales"].isna(), ["store_nbr", "family"]]
            .drop_duplicates()
            .shape[0]
        )
        raise ValueError(
            f"Seasonal naive has {missing_pairs} pair(s) without eligible history."
        )
    result["forecast_sales"] = result["weekday_sales"].fillna(
        result["pair_median_sales"]
    )
    result = result.sort_values("_input_order", ignore_index=True)
    return result[PREDICTION_COLUMNS]


def validate_prediction_population(
    predictions: pd.DataFrame,
    expected_rows: pd.DataFrame,
    model_name: str,
) -> None:
    """Require exact row order, keys, coverage, and valid predictions."""
    if predictions.columns.tolist() != PREDICTION_COLUMNS:
        raise ValueError(f"{model_name} prediction columns differ from the contract.")
    if len(predictions) != len(expected_rows):
        raise ValueError(f"{model_name} prediction row count differs from scoring rows.")
    if predictions.duplicated(BASE_KEY).any() or not predictions["id"].is_unique:
        raise ValueError(f"{model_name} predictions contain duplicate keys or IDs.")

    expected = expected_rows[["id", *BASE_KEY]].reset_index(drop=True)
    received = predictions[["id", *BASE_KEY]].reset_index(drop=True)
    keys_match = (
        np.array_equal(expected["id"].to_numpy(), received["id"].to_numpy())
        and np.array_equal(
            expected["date"].to_numpy(dtype="datetime64[ns]"),
            received["date"].to_numpy(dtype="datetime64[ns]"),
        )
        and np.array_equal(
            expected["store_nbr"].to_numpy(), received["store_nbr"].to_numpy()
        )
        and expected["family"].astype("string").equals(
            received["family"].astype("string")
        )
    )
    if not keys_match:
        raise ValueError(f"{model_name} prediction keys or order differ from scoring rows.")
    values = predictions["forecast_sales"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{model_name} predictions must be finite and non-negative.")


def attach_actual_context(
    predictions: pd.DataFrame,
    actual_context: pd.DataFrame,
    window: EvaluationWindow,
    model_name: str,
) -> pd.DataFrame:
    """Attach actuals only after prediction generation and population validation."""
    validate_prediction_population(predictions, actual_context, model_name)
    if actual_context.columns.tolist() != ACTUAL_CONTEXT_COLUMNS:
        raise ValueError("Actual context columns differ from the scoring contract.")
    if actual_context.duplicated(BASE_KEY).any() or not actual_context["id"].is_unique:
        raise ValueError("Actual context contains duplicate keys or IDs.")

    scored = predictions.merge(
        actual_context,
        on=["id", *BASE_KEY],
        how="left",
        validate="one_to_one",
        sort=False,
    ).rename(columns={"sales": "actual_sales"})
    if scored[["actual_sales", "onpromotion", "is_holiday"]].isna().any().any():
        raise ValueError("Scoring context did not match every prediction row.")
    scored.insert(0, "model", model_name)
    scored.insert(0, "window_id", window.window_id)
    scored.insert(0, "contract_id", CONTRACT_ID)
    scored["forecast_day"] = (scored["date"] - window.start).dt.days + 1
    scored["promotion_segment"] = np.where(
        scored["onpromotion"].gt(0), "promotion", "no_promotion"
    )
    scored["holiday_segment"] = np.where(
        scored["is_holiday"].eq(1), "holiday", "non_holiday"
    )
    scored["sales_segment"] = np.where(
        scored["actual_sales"].eq(0), "actual_zero", "positive_sales"
    )
    scored["signed_error"] = scored["forecast_sales"] - scored["actual_sales"]
    scored["absolute_error"] = scored["signed_error"].abs()
    scored["squared_log_error"] = (
        np.log1p(scored["forecast_sales"]) - np.log1p(scored["actual_sales"])
    ) ** 2
    return scored


def summarize_errors(frame: pd.DataFrame, expected_rows: int) -> dict[str, Any]:
    """Summarize one scoring population while preserving undefined percentages."""
    if frame.empty or expected_rows <= 0:
        raise ValueError("Error summary requires a non-empty expected population.")
    rows_scored = int(len(frame))
    actual = frame["actual_sales"].to_numpy(dtype=np.float64)
    predicted = frame["forecast_sales"].to_numpy(dtype=np.float64)
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Error summary contains non-finite values.")
    if (actual < 0).any() or (predicted < 0).any():
        raise ValueError("Error summary contains negative values.")

    error = predicted - actual
    actual_sales = float(actual.sum())
    predicted_sales = float(predicted.sum())
    absolute_error = float(np.abs(error).sum())
    rmsle = float(np.sqrt(np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2)))
    if actual_sales > 0:
        wape_pct: float | None = absolute_error / actual_sales * 100
        signed_bias_pct: float | None = float(error.sum()) / actual_sales * 100
        underforecast_pct: float | None = (
            float(np.maximum(actual - predicted, 0.0).sum()) / actual_sales * 100
        )
        overforecast_pct: float | None = (
            float(np.maximum(predicted - actual, 0.0).sum()) / actual_sales * 100
        )
    else:
        wape_pct = None
        signed_bias_pct = None
        underforecast_pct = None
        overforecast_pct = None

    return {
        "rows_expected": int(expected_rows),
        "rows_scored": rows_scored,
        "row_coverage_pct": rows_scored / expected_rows * 100,
        "actual_sales": actual_sales,
        "predicted_sales": predicted_sales,
        "absolute_error": absolute_error,
        "rmsle": rmsle,
        "wape_pct": wape_pct,
        "signed_bias_pct": signed_bias_pct,
        "underforecast_pct": underforecast_pct,
        "overforecast_pct": overforecast_pct,
    }


def aggregate_window_metrics(
    scored_predictions: pd.DataFrame,
    window_metrics: list[dict[str, Any]],
    model_name: str,
) -> dict[str, Any]:
    """Use equal window weight for RMSLE and pooled errors for other metrics."""
    selected_metrics = [
        record for record in window_metrics if record["model"] == model_name
    ]
    selected_rows = scored_predictions.loc[scored_predictions["model"].eq(model_name)]
    if len(selected_metrics) != len(EVALUATION_WINDOWS):
        raise ValueError(f"{model_name} does not have every required window metric.")
    expected_rows = sum(record["rows_expected"] for record in selected_metrics)
    aggregate = summarize_errors(selected_rows, expected_rows)
    aggregate["rmsle"] = float(
        np.mean([record["rmsle"] for record in selected_metrics])
    )
    return {
        "contract_id": CONTRACT_ID,
        "scope": "aggregate",
        "model": model_name,
        "window_count": len(selected_metrics),
        "rmsle_aggregation": "equal_window_mean",
        **aggregate,
    }


def build_slice_diagnostics(scored_predictions: pd.DataFrame) -> pd.DataFrame:
    """Build the predeclared day, store, family, promotion, holiday, and sales slices."""
    slice_columns = {
        "forecast_day": "forecast_day",
        "store": "store_nbr",
        "family": "family",
        "promotion": "promotion_segment",
        "holiday": "holiday_segment",
        "sales_activity": "sales_segment",
    }
    records: list[dict[str, Any]] = []
    for model_name, model_rows in scored_predictions.groupby("model", sort=False):
        total_absolute_error = float(model_rows["absolute_error"].sum())
        for slice_type, column in slice_columns.items():
            for slice_value, group in model_rows.groupby(
                column, observed=True, sort=True, dropna=False
            ):
                summary = summarize_errors(group, len(group))
                contribution = (
                    summary["absolute_error"] / total_absolute_error * 100
                    if total_absolute_error > 0
                    else None
                )
                records.append(
                    {
                        "contract_id": CONTRACT_ID,
                        "model": str(model_name),
                        "slice_type": slice_type,
                        "slice_value": str(slice_value),
                        **summary,
                        "absolute_error_contribution_pct": contribution,
                    }
                )
    return pd.DataFrame(records).reset_index(drop=True)


def evaluate_window(
    labeled: pd.DataFrame,
    labeled_features: pd.DataFrame,
    model_start: pd.Timestamp,
    window: EvaluationWindow,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fit and score both references for one origin without exposing future sales."""
    training_features = labeled_features.loc[
        labeled_features["date"].between(model_start, window.origin),
        ["date", "sales", *MODEL_FEATURES],
    ].copy()
    if training_features.empty or training_features["date"].max() != window.origin:
        raise ValueError(f"{window.window_id} training data does not end at its origin.")

    eligible_history = labeled.loc[
        labeled["date"].le(window.origin), [*BASE_KEY, "sales"]
    ].copy()
    future_base = labeled.loc[
        labeled["date"].between(window.start, window.end)
    ].drop(columns=["sales"])
    future_features = add_exact_sales_lags(future_base, eligible_history)
    validate_forecast_window(future_features, window.origin)
    validate_store_family_coverage(future_features, eligible_history)
    for lag in SALES_LAGS:
        if (future_features["date"] - pd.Timedelta(days=lag) > window.origin).any():
            raise RuntimeError(f"{window.window_id} sales_lag_{lag} crosses the origin.")

    fit_started = perf_counter()
    bundle = fit_forecast_bundle(
        training_features,
        V1_METHOD,
        V1_PARAMETERS,
        {
            "contract_id": CONTRACT_ID,
            "window_id": window.window_id,
            "evidence_scope": "Retrospective multi-origin evaluation",
        },
    )
    fit_seconds = perf_counter() - fit_started

    predict_started = perf_counter()
    model_predictions = predict_forecast(bundle, future_features)
    predict_seconds = perf_counter() - predict_started
    repeated_model_predictions = predict_forecast(bundle, future_features)
    if not model_predictions.equals(repeated_model_predictions):
        raise RuntimeError(f"{window.window_id} V1 predictions are not reproducible.")

    naive_started = perf_counter()
    naive_predictions = make_weekly_seasonal_naive(
        future_features,
        eligible_history,
        window.origin,
    )
    naive_seconds = perf_counter() - naive_started
    repeated_naive_predictions = make_weekly_seasonal_naive(
        future_features,
        eligible_history,
        window.origin,
    )
    if not naive_predictions.equals(repeated_naive_predictions):
        raise RuntimeError(f"{window.window_id} seasonal-naive predictions changed.")

    validate_prediction_population(model_predictions, future_features, V1_MODEL_NAME)
    validate_prediction_population(
        naive_predictions, future_features, SEASONAL_NAIVE_NAME
    )

    actual_context = labeled.loc[
        labeled["date"].between(window.start, window.end),
        ACTUAL_CONTEXT_COLUMNS,
    ].copy()
    model_scored = attach_actual_context(
        model_predictions, actual_context, window, V1_MODEL_NAME
    )
    naive_scored = attach_actual_context(
        naive_predictions, actual_context, window, SEASONAL_NAIVE_NAME
    )
    scored = pd.concat([model_scored, naive_scored], ignore_index=True)

    metric_records: list[dict[str, Any]] = []
    timing_by_model = {
        V1_MODEL_NAME: {
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
        },
        SEASONAL_NAIVE_NAME: {
            "fit_seconds": 0.0,
            "predict_seconds": float(naive_seconds),
        },
    }
    for model_name, group in scored.groupby("model", sort=False):
        metric_records.append(
            {
                "contract_id": CONTRACT_ID,
                "scope": "window",
                "window_id": window.window_id,
                "forecast_origin": window.forecast_origin,
                "scoring_start": window.scoring_start,
                "scoring_end": window.scoring_end,
                "model": str(model_name),
                **summarize_errors(group, len(future_features)),
                **timing_by_model[str(model_name)],
                "repeat_prediction_match": True,
            }
        )

    del bundle, repeated_model_predictions, repeated_naive_predictions
    gc.collect()
    return scored, metric_records


def write_evaluation_bundle(
    output_root: Path,
    run_id: str,
    predictions: pd.DataFrame,
    metrics_payload: dict[str, Any],
    diagnostics: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    """Publish all evaluation files together or leave no partial final run."""
    run_id = validate_run_id(run_id)
    root = require_project_path(output_root, "Evaluation output root")
    final_directory = (root / run_id).resolve()
    temporary_directory = (root / f".{run_id}.tmp").resolve()
    if final_directory.parent != root or temporary_directory.parent != root:
        raise ValueError("Evaluation run directory escaped its output root.")
    if final_directory.exists() or temporary_directory.exists():
        raise FileExistsError(
            f"Evaluation output already exists for run ID {run_id}."
        )

    root.mkdir(parents=True, exist_ok=True)
    temporary_directory.mkdir()
    try:
        predictions_path = temporary_directory / "predictions.csv.gz"
        metrics_path = temporary_directory / "metrics.json"
        diagnostics_path = temporary_directory / "diagnostics.csv.gz"
        manifest_path = temporary_directory / "manifest.json"

        predictions.to_csv(
            predictions_path,
            index=False,
            date_format="%Y-%m-%d",
            float_format="%.9g",
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
        metrics_path.write_text(
            json.dumps(metrics_payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        diagnostics.to_csv(
            diagnostics_path,
            index=False,
            float_format="%.9g",
            na_rep="",
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
        manifest["outputs"] = {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in [predictions_path, metrics_path, diagnostics_path]
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_directory.replace(final_directory)
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    return final_directory


def run_historical_evaluation(
    labeled_path: Path = DEFAULT_HISTORY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str = DEFAULT_RUN_ID,
) -> Path:
    """Execute the complete fixed S1 evaluation and write its private bundle."""
    started_at = datetime.now(timezone.utc)
    elapsed_started = perf_counter()
    windows = validate_window_contract()
    labeled_path = require_project_path(labeled_path, "Labeled input path")
    output_root = require_project_path(output_root, "Evaluation output root")
    validate_run_id(run_id)
    if not labeled_path.is_file():
        raise FileNotFoundError(f"Labeled input does not exist: {labeled_path}")

    print("Loading and validating the historical feature interface...", flush=True)
    input_sha256 = sha256_file(labeled_path)
    labeled = read_processed_table(labeled_path, has_target=True)
    validate_evaluation_source(labeled, windows)
    window_selection = validate_frozen_window_selection(labeled)
    labeled_features = add_exact_sales_lags(labeled, labeled)
    model_start = find_model_start(labeled_features)

    scored_frames: list[pd.DataFrame] = []
    window_metrics: list[dict[str, Any]] = []
    for window in windows:
        print(
            f"{window.window_id}: fitting V1 through {window.forecast_origin} "
            f"and forecasting {window.scoring_start} to {window.scoring_end}...",
            flush=True,
        )
        scored, metrics = evaluate_window(
            labeled,
            labeled_features,
            model_start,
            window,
        )
        scored_frames.append(scored)
        window_metrics.extend(metrics)
        model_metric = next(
            record for record in metrics if record["model"] == V1_MODEL_NAME
        )
        naive_metric = next(
            record for record in metrics if record["model"] == SEASONAL_NAIVE_NAME
        )
        print(
            f"{window.window_id}: V1 RMSLE={model_metric['rmsle']:.4f}; "
            f"seasonal naive RMSLE={naive_metric['rmsle']:.4f}.",
            flush=True,
        )

    predictions = pd.concat(scored_frames, ignore_index=True)
    aggregates = [
        aggregate_window_metrics(predictions, window_metrics, model_name)
        for model_name in [V1_MODEL_NAME, SEASONAL_NAIVE_NAME]
    ]
    diagnostics = build_slice_diagnostics(predictions)
    prediction_digest = prediction_content_sha256(predictions)
    completed_at = datetime.now(timezone.utc)

    metrics_payload = {
        "contract_id": CONTRACT_ID,
        "run_id": run_id,
        "window_metrics": window_metrics,
        "aggregate_metrics": aggregates,
    }
    code_paths = [
        Path(__file__).resolve(),
        Path(store_sales_model.__file__).resolve(),
        Path(store_sales_preprocessing.__file__).resolve(),
    ]
    resolved_v1_parameters = make_json_safe(
        store_sales_model.build_model(V1_METHOD, V1_PARAMETERS).get_params(deep=False)
    )
    manifest: dict[str, Any] = {
        "contract_id": CONTRACT_ID,
        "run_id": run_id,
        "created_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "elapsed_seconds": float(perf_counter() - elapsed_started),
        "evidence_scope": (
            "Controlled retrospective multi-origin evaluation; not independent "
            "future or live retail validation"
        ),
        "input": {
            "file": labeled_path.name,
            "bytes": labeled_path.stat().st_size,
            "sha256": input_sha256,
            "rows": int(len(labeled)),
            "date_min": labeled["date"].min().date().isoformat(),
            "date_max": labeled["date"].max().date().isoformat(),
        },
        "configuration": {
            "model_start": model_start.date().isoformat(),
            "windows": [asdict(window) for window in windows],
            "window_selection": {
                "method": "target_free_scenario_stratification",
                "source_columns": list(WINDOW_SELECTION_SOURCE_COLUMNS),
                "blocks": [asdict(block) for block in WINDOW_SELECTION_BLOCKS],
                "selected_context": window_selection,
            },
            "v1_method": V1_METHOD,
            "v1_selected_parameters": dict(V1_PARAMETERS),
            "v1_resolved_parameters": resolved_v1_parameters,
            "weekly_seasonal_naive": {
                "history_days": 35,
                "primary": "latest eligible sale on the same weekday",
                "fallback": "store-family median only when weekday is absent",
            },
            "actuals_attached_after_prediction": True,
            "repeat_prediction_match": True,
        },
        "prediction_content_sha256": prediction_digest,
        "code": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in code_paths
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "library_versions": dict(CURRENT_LIBRARY_VERSIONS),
        },
    }
    output_directory = write_evaluation_bundle(
        output_root,
        run_id,
        predictions,
        metrics_payload,
        diagnostics,
        manifest,
    )
    print("Historical evaluation: PASS", flush=True)
    print(f"Output: {output_directory}", flush=True)
    return output_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refit Store Sales V1 at four historical origins and compare it "
            "with the fixed weekly seasonal-naive reference."
        )
    )
    parser.add_argument("--labeled-path", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_historical_evaluation(args.labeled_path, args.output_root, args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
