"""Select and evaluate a Store Sales model over four forecast origins.

The command evaluates the frozen Ridge and XGBoost search space on four
expanding 16-day folds, freezes the lowest mean-RMSLE configuration, scores it
on the internal holdout, and writes a private candidate bundle without
replacing the accepted V1 artifact.
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
from sklearn.model_selection import ParameterGrid

import store_sales_model
import store_sales_preprocessing
from store_sales_model import (
    CURRENT_LIBRARY_VERSIONS,
    DEFAULT_FUTURE_PATH,
    DEFAULT_HISTORY_PATH,
    FORECAST_HORIZON_DAYS,
    MODEL_FEATURES,
    MODEL_GRIDS,
    PROJECT_ROOT,
    SALES_LAGS,
    add_exact_sales_lags,
    build_model,
    evaluate_forecast,
    find_model_start,
    fit_forecast_bundle,
    is_default_reference,
    make_feature_processor,
    predict_forecast,
    read_processed_table,
    save_forecast_artifact,
    validate_forecast_window,
    validate_store_family_coverage,
)
from store_sales_preprocessing import BASE_KEY


CONTRACT_ID = "retail-history-selection-01"
DEFAULT_RUN_ID = f"{CONTRACT_ID}-v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "evaluation"
DEFAULT_SAMPLE_SUBMISSION_PATH = PROJECT_ROOT / "data" / "raw" / "sample_submission.csv"
V1_MODEL_NAME = "xgboost_v1"
SELECTED_MODEL_NAME = "selected_cv_model"
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
DEVELOPMENT_END = pd.Timestamp("2017-07-30")
INTERNAL_TEST_WINDOW = EvaluationWindow(
    "internal_test", "2017-07-30", "2017-07-31", "2017-08-15"
)
FINAL_TRAINING_END = pd.Timestamp("2017-08-15")
KAGGLE_START = pd.Timestamp("2017-08-16")
KAGGLE_END = pd.Timestamp("2017-08-31")
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


def build_candidate_registry() -> list[dict[str, Any]]:
    """Expand the frozen Ridge and XGBoost grids into 30 identified candidates."""
    candidates: list[dict[str, Any]] = []
    for method, grid in MODEL_GRIDS.items():
        for run_number, parameters in enumerate(ParameterGrid(grid), start=1):
            normalized = {
                key: make_json_safe(value) for key, value in dict(parameters).items()
            }
            candidates.append(
                {
                    "run_id": f"{method.split()[0].lower()}_{run_number:02d}",
                    "method": method,
                    "parameters": normalized,
                    "parameters_json": json.dumps(normalized, sort_keys=True),
                    "default_reference": is_default_reference(method, normalized),
                    "v1_reference": (
                        method == V1_METHOD and normalized == V1_PARAMETERS
                    ),
                }
            )
    if len(candidates) != 30 or len({row["run_id"] for row in candidates}) != 30:
        raise RuntimeError("The frozen model-search contract must contain 30 runs.")
    if sum(bool(row["v1_reference"]) for row in candidates) != 1:
        raise RuntimeError("The accepted V1 configuration must appear exactly once.")
    return candidates


def aggregate_candidate_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank complete candidates by equal-fold mean RMSLE, then mean WAPE."""
    required = {
        "run_id",
        "method",
        "parameters_json",
        "default_reference",
        "v1_reference",
        "window_id",
        "rmsle",
        "wape_pct",
        "signed_bias_pct",
        "fit_seconds",
        "predict_seconds",
    }
    missing = sorted(required.difference(fold_metrics.columns))
    if fold_metrics.empty or missing:
        raise ValueError(f"Candidate fold metrics are incomplete: {missing}")
    expected_windows = {window.window_id for window in EVALUATION_WINDOWS}
    duplicate = fold_metrics.duplicated(["run_id", "window_id"]).any()
    if duplicate:
        raise ValueError("Candidate fold metrics contain duplicate run-window rows.")
    coverage = fold_metrics.groupby("run_id", observed=True)["window_id"].agg(set)
    if coverage.empty or not coverage.map(lambda values: values == expected_windows).all():
        raise ValueError("Every candidate must contain all 4 validation windows.")

    summary = (
        fold_metrics.groupby(
            [
                "run_id",
                "method",
                "parameters_json",
                "default_reference",
                "v1_reference",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            fold_count=("window_id", "size"),
            mean_rmsle=("rmsle", "mean"),
            std_rmsle=("rmsle", "std"),
            mean_wape_pct=("wape_pct", "mean"),
            mean_signed_bias_pct=("signed_bias_pct", "mean"),
            total_fit_seconds=("fit_seconds", "sum"),
            total_predict_seconds=("predict_seconds", "sum"),
        )
        .sort_values(
            ["mean_rmsle", "mean_wape_pct", "method", "run_id"],
            ignore_index=True,
        )
    )
    metric_columns = [
        "mean_rmsle",
        "std_rmsle",
        "mean_wape_pct",
        "mean_signed_bias_pct",
        "total_fit_seconds",
        "total_predict_seconds",
    ]
    if not np.isfinite(summary[metric_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("Candidate summary contains a non-finite metric.")
    summary.insert(0, "rank", np.arange(1, len(summary) + 1, dtype=np.int16))
    return summary


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


def require_available_output_paths(
    output_root: Path,
    run_id: str,
) -> tuple[Path, Path, Path]:
    """Resolve one run destination and reject existing final or temporary output."""
    run_id = validate_run_id(run_id)
    root = require_project_path(output_root, "Evaluation output root")
    final_directory = (root / run_id).resolve()
    temporary_directory = (root / f".{run_id}.tmp").resolve()
    if final_directory.parent != root or temporary_directory.parent != root:
        raise ValueError("Evaluation run directory escaped its output root.")
    if final_directory.exists() or temporary_directory.exists():
        raise FileExistsError(
            f"Evaluation output already exists for run ID {run_id}. "
            "Inspect it or choose a new --run-id; existing runs are never overwritten."
        )
    return root, final_directory, temporary_directory


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


def build_fold_inputs(
    labeled: pd.DataFrame,
    labeled_features: pd.DataFrame,
    model_start: pd.Timestamp,
    window: EvaluationWindow,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build train, forecast, history, and actual tables for one origin."""
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

    actual_context = labeled.loc[
        labeled["date"].between(window.start, window.end),
        ACTUAL_CONTEXT_COLUMNS,
    ].copy()
    expected_keys = future_features[["id", *BASE_KEY]].reset_index(drop=True)
    actual_keys = actual_context[["id", *BASE_KEY]].reset_index(drop=True)
    if not expected_keys.equals(actual_keys):
        raise RuntimeError(f"{window.window_id} feature and actual row order differs.")
    return training_features, future_features, eligible_history, actual_context


def run_candidate_search(
    labeled: pd.DataFrame,
    labeled_features: pd.DataFrame,
    model_start: pd.Timestamp,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[str, str], np.ndarray],
    dict[str, dict[str, Any]],
]:
    """Evaluate all 30 candidates on the 4 frozen validation folds."""
    candidates = build_candidate_registry()
    records: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[str, str], np.ndarray] = {}
    fold_contexts: dict[str, dict[str, Any]] = {}

    for window in EVALUATION_WINDOWS:
        print(
            f"{window.window_id}: preparing train through {window.forecast_origin} "
            f"and validation {window.scoring_start} to {window.scoring_end}...",
            flush=True,
        )
        training, future, history, actual_context = build_fold_inputs(
            labeled, labeled_features, model_start, window
        )
        processor = make_feature_processor()
        training_matrix = processor.fit_transform(
            training[MODEL_FEATURES]
        ).astype(np.float32)
        validation_matrix = processor.transform(future[MODEL_FEATURES]).astype(
            np.float32
        )
        training_target = np.log1p(training["sales"].to_numpy(dtype=np.float32))
        validation_target = actual_context["sales"].to_numpy(dtype=np.float64)

        for candidate_number, candidate in enumerate(candidates, start=1):
            model = build_model(candidate["method"], candidate["parameters"])
            fit_started = perf_counter()
            model.fit(training_matrix, training_target)
            fit_seconds = perf_counter() - fit_started

            predict_started = perf_counter()
            predicted_log = model.predict(validation_matrix)
            predicted_sales = np.clip(np.expm1(predicted_log), 0.0, None)
            predict_seconds = perf_counter() - predict_started
            repeated_log = model.predict(validation_matrix)
            repeated_sales = np.clip(np.expm1(repeated_log), 0.0, None)
            if not np.array_equal(predicted_sales, repeated_sales):
                raise RuntimeError(
                    f"{window.window_id} {candidate['run_id']} predictions changed."
                )
            if len(predicted_sales) != len(future):
                raise RuntimeError("Candidate prediction row count is invalid.")
            if not np.isfinite(predicted_sales).all() or (predicted_sales < 0).any():
                raise RuntimeError("Candidate predictions must be finite and non-negative.")

            prediction_cache[(window.window_id, candidate["run_id"])] = (
                predicted_sales.astype(np.float32, copy=True)
            )
            records.append(
                {
                    "contract_id": CONTRACT_ID,
                    "window_id": window.window_id,
                    "forecast_origin": window.forecast_origin,
                    "scoring_start": window.scoring_start,
                    "scoring_end": window.scoring_end,
                    "run_id": candidate["run_id"],
                    "method": candidate["method"],
                    "parameters_json": candidate["parameters_json"],
                    "default_reference": candidate["default_reference"],
                    "v1_reference": candidate["v1_reference"],
                    **evaluate_forecast(validation_target, predicted_sales),
                    "fit_seconds": float(fit_seconds),
                    "predict_seconds": float(predict_seconds),
                    "repeat_prediction_match": True,
                }
            )
            print(
                f"{window.window_id}: candidate {candidate_number:02d}/"
                f"{len(candidates):02d} "
                f"{candidate['run_id']} complete.",
                flush=True,
            )
            del model, predicted_log, repeated_log, repeated_sales

        naive_started = perf_counter()
        naive_predictions = make_weekly_seasonal_naive(future, history, window.origin)
        naive_predict_seconds = perf_counter() - naive_started
        validate_prediction_population(naive_predictions, future, SEASONAL_NAIVE_NAME)
        fold_contexts[window.window_id] = {
            "future": future[["id", *BASE_KEY]].copy(),
            "actual_context": actual_context,
            "naive_predictions": naive_predictions,
            "naive_predict_seconds": float(naive_predict_seconds),
        }
        del processor, training_matrix, validation_matrix, training_target
        gc.collect()

    fold_metrics = pd.DataFrame(records)
    summary = aggregate_candidate_metrics(fold_metrics)
    return fold_metrics, summary, prediction_cache, fold_contexts


def materialize_selected_fold_evidence(
    selected: pd.Series,
    fold_metrics: pd.DataFrame,
    prediction_cache: dict[tuple[str, str], np.ndarray],
    fold_contexts: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Attach actuals only for the selected candidate and seasonal benchmark."""
    selected_run_id = str(selected["run_id"])
    scored_frames: list[pd.DataFrame] = []
    metric_records: list[dict[str, Any]] = []
    for window in EVALUATION_WINDOWS:
        context = fold_contexts[window.window_id]
        future = context["future"]
        predictions = future.copy()
        predictions["forecast_sales"] = prediction_cache[
            (window.window_id, selected_run_id)
        ]
        predictions = predictions[PREDICTION_COLUMNS]
        selected_scored = attach_actual_context(
            predictions,
            context["actual_context"],
            window,
            SELECTED_MODEL_NAME,
        )
        naive_scored = attach_actual_context(
            context["naive_predictions"],
            context["actual_context"],
            window,
            SEASONAL_NAIVE_NAME,
        )
        scored_frames.extend([selected_scored, naive_scored])

        selected_timing = fold_metrics.loc[
            fold_metrics["run_id"].eq(selected_run_id)
            & fold_metrics["window_id"].eq(window.window_id)
        ].iloc[0]
        for model_name, scored, fit_seconds, predict_seconds in [
            (
                SELECTED_MODEL_NAME,
                selected_scored,
                float(selected_timing["fit_seconds"]),
                float(selected_timing["predict_seconds"]),
            ),
            (
                SEASONAL_NAIVE_NAME,
                naive_scored,
                0.0,
                float(context["naive_predict_seconds"]),
            ),
        ]:
            metric_records.append(
                {
                    "contract_id": CONTRACT_ID,
                    "scope": "window",
                    "window_id": window.window_id,
                    "forecast_origin": window.forecast_origin,
                    "scoring_start": window.scoring_start,
                    "scoring_end": window.scoring_end,
                    "model": model_name,
                    **summarize_errors(scored, len(future)),
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "repeat_prediction_match": True,
                }
            )
    return pd.concat(scored_frames, ignore_index=True), metric_records


def evaluate_window(
    labeled: pd.DataFrame,
    labeled_features: pd.DataFrame,
    model_start: pd.Timestamp,
    window: EvaluationWindow,
    method: str = V1_METHOD,
    parameters: dict[str, float | int] | None = None,
    model_name: str = V1_MODEL_NAME,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fit and score one selected configuration plus the seasonal benchmark."""
    selected_parameters = dict(V1_PARAMETERS if parameters is None else parameters)
    training_features, future_features, eligible_history, actual_context = (
        build_fold_inputs(labeled, labeled_features, model_start, window)
    )

    fit_started = perf_counter()
    bundle = fit_forecast_bundle(
        training_features,
        method,
        selected_parameters,
        {
            "contract_id": CONTRACT_ID,
            "window_id": window.window_id,
            "evidence_scope": "Selected-configuration evaluation",
        },
    )
    fit_seconds = perf_counter() - fit_started

    predict_started = perf_counter()
    model_predictions = predict_forecast(bundle, future_features)
    predict_seconds = perf_counter() - predict_started
    repeated_model_predictions = predict_forecast(bundle, future_features)
    if not model_predictions.equals(repeated_model_predictions):
        raise RuntimeError(
            f"{window.window_id} {model_name} predictions are not reproducible."
        )

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

    validate_prediction_population(model_predictions, future_features, model_name)
    validate_prediction_population(
        naive_predictions, future_features, SEASONAL_NAIVE_NAME
    )

    model_scored = attach_actual_context(
        model_predictions, actual_context, window, model_name
    )
    naive_scored = attach_actual_context(
        naive_predictions, actual_context, window, SEASONAL_NAIVE_NAME
    )
    scored = pd.concat([model_scored, naive_scored], ignore_index=True)

    metric_records: list[dict[str, Any]] = []
    timing_by_model = {
        model_name: {
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


def build_final_candidate_outputs(
    labeled: pd.DataFrame,
    labeled_features: pd.DataFrame,
    model_start: pd.Timestamp,
    selected: pd.Series,
    internal_metrics: dict[str, Any],
    future_path: Path,
    sample_submission_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Refit the selected configuration and create private Kaggle outputs."""
    if labeled["date"].max() != FINAL_TRAINING_END:
        raise ValueError("Labeled history must end on 2017-08-15.")
    future_path = require_project_path(future_path, "Kaggle input path")
    sample_submission_path = require_project_path(
        sample_submission_path, "Sample submission path"
    )
    if not future_path.is_file() or not sample_submission_path.is_file():
        raise FileNotFoundError("Kaggle input or sample submission is missing.")

    selected_parameters = json.loads(str(selected["parameters_json"]))
    final_training = labeled_features.loc[
        labeled_features["date"].between(model_start, FINAL_TRAINING_END),
        ["date", "sales", *MODEL_FEATURES],
    ].copy()
    evaluation_reference = {
        "contract_id": CONTRACT_ID,
        "selected_run_id": str(selected["run_id"]),
        "selection_fold_count": int(selected["fold_count"]),
        "selection_mean_rmsle": float(selected["mean_rmsle"]),
        "selection_mean_wape_pct": float(selected["mean_wape_pct"]),
        "internal_test_period": "2017-07-31 to 2017-08-15",
        "internal_test_rmsle": float(internal_metrics["rmsle"]),
        "internal_test_wape_pct": float(internal_metrics["wape_pct"]),
        "internal_test_signed_bias_pct": float(
            internal_metrics["signed_bias_pct"]
        ),
        "evidence_scope": "Controlled local 4-fold selection and internal test",
    }
    bundle = fit_forecast_bundle(
        final_training,
        str(selected["method"]),
        selected_parameters,
        evaluation_reference,
    )

    future = read_processed_table(future_path, has_target=False)
    future_dates = pd.DatetimeIndex(future["date"].drop_duplicates().sort_values())
    expected_dates = pd.date_range(KAGGLE_START, KAGGLE_END, freq="D")
    if not future_dates.equals(expected_dates):
        raise ValueError("Kaggle input does not contain the expected 16 dates.")
    history = labeled.loc[labeled["date"].le(FINAL_TRAINING_END)]
    future_features = add_exact_sales_lags(future, history)
    validate_store_family_coverage(future_features, history)
    forecast = predict_forecast(bundle, future_features)

    submission = pd.read_csv(sample_submission_path)
    if submission.columns.tolist() != ["id", "sales"]:
        raise ValueError("Sample submission columns differ from the Kaggle contract.")
    if len(submission) != len(forecast) or not np.array_equal(
        submission["id"].to_numpy(), forecast["id"].to_numpy()
    ):
        raise ValueError("Sample submission IDs differ from the candidate forecast.")
    submission = submission.copy()
    submission["sales"] = forecast["forecast_sales"].to_numpy(dtype=np.float32)
    return bundle, forecast, submission


def write_evaluation_bundle(
    output_root: Path,
    run_id: str,
    predictions: pd.DataFrame,
    metrics_payload: dict[str, Any],
    diagnostics: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    candidate_fold_metrics: pd.DataFrame | None = None,
    candidate_summary: pd.DataFrame | None = None,
    internal_predictions: pd.DataFrame | None = None,
    candidate_bundle: dict[str, Any] | None = None,
    kaggle_forecast: pd.DataFrame | None = None,
    kaggle_submission: pd.DataFrame | None = None,
) -> Path:
    """Publish all evaluation files together or leave no partial final run."""
    root, final_directory, temporary_directory = require_available_output_paths(
        output_root, run_id
    )

    root.mkdir(parents=True, exist_ok=True)
    temporary_created = False
    try:
        temporary_directory.mkdir()
        temporary_created = True
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
        output_paths = [predictions_path, metrics_path, diagnostics_path]
        optional_tables = [
            ("candidate_fold_metrics.csv", candidate_fold_metrics, {}),
            ("candidate_summary.csv", candidate_summary, {}),
            (
                "internal_predictions.csv.gz",
                internal_predictions,
                {"compression": {"method": "gzip", "compresslevel": 6, "mtime": 0}},
            ),
            (
                "kaggle_forecast.csv.gz",
                kaggle_forecast,
                {"compression": {"method": "gzip", "compresslevel": 6, "mtime": 0}},
            ),
            ("kaggle_submission.csv", kaggle_submission, {}),
        ]
        for filename, table, options in optional_tables:
            if table is None:
                continue
            path = temporary_directory / filename
            table.to_csv(
                path,
                index=False,
                date_format="%Y-%m-%d",
                float_format="%.9g",
                **options,
            )
            output_paths.append(path)
        if candidate_bundle is not None:
            artifact_path, metadata_path = save_forecast_artifact(
                candidate_bundle,
                temporary_directory / "candidate_model.pkl",
            )
            output_paths.extend([artifact_path, metadata_path])

        manifest["outputs"] = {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in output_paths
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_directory.replace(final_directory)
    except Exception:
        if temporary_created:
            shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    return final_directory


def run_historical_evaluation(
    labeled_path: Path = DEFAULT_HISTORY_PATH,
    future_path: Path = DEFAULT_FUTURE_PATH,
    sample_submission_path: Path = DEFAULT_SAMPLE_SUBMISSION_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str = DEFAULT_RUN_ID,
) -> Path:
    """Execute 4-fold selection, internal scoring, and private final refitting."""
    started_at = datetime.now(timezone.utc)
    elapsed_started = perf_counter()
    windows = validate_window_contract()
    validate_window_contract([*windows, INTERNAL_TEST_WINDOW])
    labeled_path = require_project_path(labeled_path, "Labeled input path")
    future_path = require_project_path(future_path, "Kaggle input path")
    sample_submission_path = require_project_path(
        sample_submission_path, "Sample submission path"
    )
    output_root = require_project_path(output_root, "Evaluation output root")
    validate_run_id(run_id)
    require_available_output_paths(output_root, run_id)
    input_paths = [labeled_path, future_path, sample_submission_path]
    missing_inputs = [str(path) for path in input_paths if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError("Required input does not exist: " + ", ".join(missing_inputs))

    print("Loading and validating the historical feature interface...", flush=True)
    input_sha256 = sha256_file(labeled_path)
    labeled = read_processed_table(labeled_path, has_target=True)
    validate_evaluation_source(labeled, [*windows, INTERNAL_TEST_WINDOW])
    window_selection = validate_frozen_window_selection(labeled)
    labeled_features = add_exact_sales_lags(labeled, labeled)
    model_start = find_model_start(labeled_features)

    print("Evaluating 30 configurations over 4 expanding validation folds...", flush=True)
    candidate_fold_metrics, candidate_summary, prediction_cache, fold_contexts = (
        run_candidate_search(labeled, labeled_features, model_start)
    )
    selected = candidate_summary.iloc[0]
    selected_parameters = json.loads(str(selected["parameters_json"]))
    predictions, window_metrics = materialize_selected_fold_evidence(
        selected,
        candidate_fold_metrics,
        prediction_cache,
        fold_contexts,
    )
    validation_aggregates = [
        aggregate_window_metrics(predictions, window_metrics, model_name)
        for model_name in [SELECTED_MODEL_NAME, SEASONAL_NAIVE_NAME]
    ]
    diagnostics = build_slice_diagnostics(predictions)
    prediction_digest = prediction_content_sha256(predictions)

    print(
        "Selected "
        f"{selected['run_id']} ({selected['method']}) with mean RMSLE "
        f"{selected['mean_rmsle']:.4f} across 4 folds.",
        flush=True,
    )
    print(
        "Scoring the frozen selection once on the 2017-07-31 to 2017-08-15 "
        "internal test...",
        flush=True,
    )
    internal_predictions, internal_window_metrics = evaluate_window(
        labeled,
        labeled_features,
        model_start,
        INTERNAL_TEST_WINDOW,
        str(selected["method"]),
        selected_parameters,
        SELECTED_MODEL_NAME,
    )
    internal_model_metric = next(
        record
        for record in internal_window_metrics
        if record["model"] == SELECTED_MODEL_NAME
    )
    print(
        f"Internal-test RMSLE={internal_model_metric['rmsle']:.4f}; "
        f"WAPE={internal_model_metric['wape_pct']:.2f}%.",
        flush=True,
    )

    del prediction_cache, fold_contexts
    gc.collect()
    print("Refitting the selected configuration through 2017-08-15...", flush=True)
    candidate_bundle, kaggle_forecast, kaggle_submission = (
        build_final_candidate_outputs(
            labeled,
            labeled_features,
            model_start,
            selected,
            internal_model_metric,
            future_path,
            sample_submission_path,
        )
    )
    completed_at = datetime.now(timezone.utc)

    v1_reference = candidate_summary.loc[candidate_summary["v1_reference"]].iloc[0]
    metrics_payload = make_json_safe(
        {
            "contract_id": CONTRACT_ID,
            "run_id": run_id,
            "selection": {
                "primary_metric": "equal_fold_mean_rmsle",
                "tie_breaker": "equal_fold_mean_wape_pct",
                "selected": selected.to_dict(),
                "accepted_v1_reference": v1_reference.to_dict(),
            },
            "validation_window_metrics": window_metrics,
            "validation_aggregate_metrics": validation_aggregates,
            "internal_test_metrics": internal_window_metrics,
        }
    )
    code_paths = [
        Path(__file__).resolve(),
        Path(store_sales_model.__file__).resolve(),
        Path(store_sales_preprocessing.__file__).resolve(),
    ]
    resolved_selected_parameters = make_json_safe(
        build_model(str(selected["method"]), selected_parameters).get_params(deep=False)
    )
    manifest: dict[str, Any] = make_json_safe(
        {
            "contract_id": CONTRACT_ID,
            "run_id": run_id,
            "created_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "elapsed_seconds": float(perf_counter() - elapsed_started),
            "evidence_scope": (
                "Controlled 4-fold model selection, one protected internal test, "
                "and one unlabeled Kaggle forecast candidate"
            ),
            "inputs": {
                "labeled_history": {
                    "file": labeled_path.name,
                    "bytes": labeled_path.stat().st_size,
                    "sha256": input_sha256,
                    "rows": int(len(labeled)),
                    "date_min": labeled["date"].min().date().isoformat(),
                    "date_max": labeled["date"].max().date().isoformat(),
                },
                "kaggle_test": {
                    "file": future_path.name,
                    "bytes": future_path.stat().st_size,
                    "sha256": sha256_file(future_path),
                },
                "sample_submission": {
                    "file": sample_submission_path.name,
                    "bytes": sample_submission_path.stat().st_size,
                    "sha256": sha256_file(sample_submission_path),
                },
            },
            "configuration": {
                "model_start": model_start.date().isoformat(),
                "validation_windows": [asdict(window) for window in windows],
                "window_selection": {
                    "method": "target_free_scenario_stratification",
                    "source_columns": list(WINDOW_SELECTION_SOURCE_COLUMNS),
                    "blocks": [asdict(block) for block in WINDOW_SELECTION_BLOCKS],
                    "selected_context": window_selection,
                },
                "search": {
                    "candidate_count": int(len(candidate_summary)),
                    "candidate_fold_fits": int(len(candidate_fold_metrics)),
                    "grids": MODEL_GRIDS,
                    "primary_metric": "equal_fold_mean_rmsle",
                    "tie_breaker": "equal_fold_mean_wape_pct",
                },
                "selected": selected.to_dict(),
                "selected_resolved_parameters": resolved_selected_parameters,
                "internal_test_window": asdict(INTERNAL_TEST_WINDOW),
                "final_training_end": FINAL_TRAINING_END.date().isoformat(),
                "kaggle_forecast_start": KAGGLE_START.date().isoformat(),
                "kaggle_forecast_end": KAGGLE_END.date().isoformat(),
                "weekly_seasonal_naive": {
                    "history_days": 35,
                    "primary": "latest eligible sale on the same weekday",
                    "fallback": "store-family median only when weekday is absent",
                },
                "actuals_attached_after_prediction": True,
                "repeat_prediction_match": True,
            },
            "validation_prediction_content_sha256": prediction_digest,
            "internal_prediction_content_sha256": prediction_content_sha256(
                internal_predictions
            ),
            "code": {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in code_paths
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "library_versions": dict(CURRENT_LIBRARY_VERSIONS),
            },
        }
    )
    output_directory = write_evaluation_bundle(
        output_root,
        run_id,
        predictions,
        metrics_payload,
        diagnostics,
        manifest,
        candidate_fold_metrics=candidate_fold_metrics,
        candidate_summary=candidate_summary,
        internal_predictions=internal_predictions,
        candidate_bundle=candidate_bundle,
        kaggle_forecast=kaggle_forecast,
        kaggle_submission=kaggle_submission,
    )
    print("4-fold selection and internal evaluation: PASS", flush=True)
    print(
        f"Private Kaggle candidate: {len(kaggle_forecast):,} rows from "
        f"{KAGGLE_START.date()} to {KAGGLE_END.date()}.",
        flush=True,
    )
    print(f"Output: {output_directory}", flush=True)
    return output_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select 1 of 30 configurations over 4 expanding validation folds, "
            "score it once on the internal test, and write a private candidate."
        )
    )
    parser.add_argument("--labeled-path", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--future-path", type=Path, default=DEFAULT_FUTURE_PATH)
    parser.add_argument(
        "--sample-submission-path",
        type=Path,
        default=DEFAULT_SAMPLE_SUBMISSION_PATH,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_historical_evaluation(
        labeled_path=args.labeled_path,
        future_path=args.future_path,
        sample_submission_path=args.sample_submission_path,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
