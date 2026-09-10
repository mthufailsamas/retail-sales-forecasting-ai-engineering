"""Evaluate the predeclared S2-01 feature across the frozen S1 search space.

The command reuses verified S1 predictions as immutable references, retrains
the same 30 candidates over the same 4 folds with one additional cutoff-safe
feature, and applies the frozen comparison gate. It never changes active V2.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from evaluate_store_sales import (
    CONTRACT_ID as S1_CONTRACT_ID,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RUN_ID as DEFAULT_S1_RUN_ID,
    EVALUATION_WINDOWS,
    SEASONAL_NAIVE_NAME,
    SELECTED_MODEL_NAME,
    aggregate_candidate_metrics,
    attach_actual_context,
    build_candidate_registry,
    build_fold_inputs,
    build_slice_diagnostics,
    make_json_safe,
    prediction_content_sha256,
    read_json_object,
    require_available_output_paths,
    require_project_path,
    sha256_file,
    summarize_errors,
    validate_evaluation_source,
    validate_frozen_window_selection,
    validate_prediction_population,
    verify_evaluation_outputs,
    write_evaluation_bundle,
)
from store_sales_model import (
    CATEGORICAL_FEATURES,
    CURRENT_LIBRARY_VERSIONS,
    DEFAULT_HISTORY_PATH,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    PROJECT_ROOT,
    add_exact_sales_lags,
    build_model,
    evaluate_forecast,
    find_model_start,
    make_feature_processor,
    read_processed_table,
)
from store_sales_preprocessing import BASE_KEY


CONTRACT_ID = "retail-s2-sales-mean-lag-16-35-01"
DEFAULT_RUN_ID = f"{CONTRACT_ID}-v1"
DEFAULT_S1_RUN_DIRECTORY = DEFAULT_OUTPUT_ROOT / DEFAULT_S1_RUN_ID
FEATURE_NAME = "sales_mean_lag_16_35"
FEATURE_LAGS = tuple(range(16, 36))
ACTIVE_MODEL_NAME = "active_v2_xgboost_18"
FEATURE_CONTROL_MODEL_NAME = "s2_xgboost_18_with_feature"
CHALLENGER_MODEL_NAME = "s2_selected_with_feature"
FEATURE_CONTROL_RUN_ID = "xgboost_18"
METHOD = "XGBoost Regression"
PARAMETERS: dict[str, float | int] = {
    "learning_rate": 0.1,
    "max_depth": 8,
    "n_estimators": 500,
}
S2_NUMERIC_FEATURES = [*NUMERIC_FEATURES, FEATURE_NAME]
S2_MODEL_FEATURES = [*MODEL_FEATURES, FEATURE_NAME]
EXPECTED_CANDIDATES = 30
MAX_FITS = EXPECTED_CANDIDATES * len(EVALUATION_WINDOWS)
MAX_ADDED_FEATURES = 1
MAX_FEATURE_BYTES = 16 * 1024 * 1024
MAX_FIT_TIME_MULTIPLIER = 2.0
MAX_MEAN_PREDICT_TIME_MULTIPLIER = 3.0


def add_sales_mean_lag_16_35(
    target_rows: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Attach one mean of available exact sales from t-16 through t-35."""
    missing_target = [column for column in BASE_KEY if column not in target_rows]
    required_history = [*BASE_KEY, "sales"]
    missing_history = [column for column in required_history if column not in history]
    if missing_target or missing_history:
        raise ValueError(
            "S2 feature input is missing required columns: "
            f"target={missing_target}, history={missing_history}."
        )
    if history.duplicated(BASE_KEY).any():
        raise ValueError(
            "S2 feature history must be unique by date, store, and family."
        )
    observed_sales = history["sales"].dropna().to_numpy(dtype=np.float64)
    if not np.isfinite(observed_sales).all() or (observed_sales < 0).any():
        raise ValueError("S2 feature history sales must be finite and non-negative.")

    result = target_rows.copy()
    sales_lookup = history.set_index(BASE_KEY)["sales"]
    total = np.zeros(len(result), dtype=np.float64)
    available = np.zeros(len(result), dtype=np.uint8)
    for lag in FEATURE_LAGS:
        lookup_key = pd.MultiIndex.from_arrays(
            [
                result["date"] - pd.Timedelta(days=lag),
                result["store_nbr"],
                result["family"],
            ],
            names=BASE_KEY,
        )
        values = sales_lookup.reindex(lookup_key).to_numpy(dtype=np.float32)
        present = ~np.isnan(values)
        total[present] += values[present]
        available[present] += 1

    feature = np.full(len(result), np.nan, dtype=np.float32)
    np.divide(total, available, out=feature, where=available > 0, casting="unsafe")
    result[FEATURE_NAME] = feature
    return result


def matrix_nbytes(matrix: Any) -> int:
    """Measure dense or CSR/CSC matrix storage used by one model input."""
    if all(hasattr(matrix, name) for name in ("data", "indices", "indptr")):
        return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    return int(matrix.nbytes)


def load_frozen_s1_reference(
    run_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Verify and load the immutable S1 validation evidence used by S2."""
    run_directory = require_project_path(run_directory, "S1 reference directory")
    if not run_directory.is_dir():
        raise FileNotFoundError(f"S1 reference directory is missing: {run_directory}")

    manifest = read_json_object(run_directory / "manifest.json", "S1 manifest")
    metrics = read_json_object(run_directory / "metrics.json", "S1 metrics")
    if manifest.get("contract_id") != S1_CONTRACT_ID:
        raise ValueError("S1 manifest uses an unexpected contract.")
    if metrics.get("contract_id") != S1_CONTRACT_ID:
        raise ValueError("S1 metrics use an unexpected contract.")
    if manifest.get("run_id") != run_directory.name:
        raise ValueError("S1 manifest run ID differs from its directory.")
    if metrics.get("run_id") != run_directory.name:
        raise ValueError("S1 metrics run ID differs from its directory.")
    verify_evaluation_outputs(run_directory, manifest)

    selection = metrics.get("selection")
    selected = selection.get("selected") if isinstance(selection, dict) else None
    if not isinstance(selected, dict) or selected.get("run_id") != "xgboost_18":
        raise ValueError("S1 reference does not select xgboost_18.")
    if selected.get("method") != METHOD:
        raise ValueError("S1 selected method differs from the S2 contract.")
    try:
        selected_parameters = json.loads(selected["parameters_json"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("S1 selected parameters are invalid.") from error
    if selected_parameters != PARAMETERS:
        raise ValueError("S1 selected parameters differ from the S2 contract.")

    predictions = pd.read_csv(run_directory / "predictions.csv.gz")
    predictions["date"] = pd.to_datetime(predictions["date"], errors="raise")
    expected_digest = manifest.get("validation_prediction_content_sha256")
    if prediction_content_sha256(predictions) != expected_digest:
        raise ValueError("S1 validation prediction content differs from its manifest.")
    expected_models = {SELECTED_MODEL_NAME, SEASONAL_NAIVE_NAME}
    if set(predictions["model"].unique()) != expected_models:
        raise ValueError("S1 validation predictions contain unexpected models.")
    expected_windows = {window.window_id for window in EVALUATION_WINDOWS}
    if set(predictions["window_id"].unique()) != expected_windows:
        raise ValueError("S1 validation predictions do not cover all frozen windows.")

    candidate_summary = pd.read_csv(run_directory / "candidate_summary.csv")
    expected_candidates = build_candidate_registry()
    expected_by_run_id = {row["run_id"]: row for row in expected_candidates}
    expected_run_ids = set(expected_by_run_id)
    if (
        len(candidate_summary) != EXPECTED_CANDIDATES
        or set(candidate_summary["run_id"]) != expected_run_ids
        or set(candidate_summary["fold_count"]) != {len(EVALUATION_WINDOWS)}
        or candidate_summary["run_id"].duplicated().any()
    ):
        raise ValueError(
            "S1 candidate summary differs from the frozen search contract."
        )
    for record in candidate_summary.to_dict("records"):
        expected = expected_by_run_id[record["run_id"]]
        try:
            parameters = json.loads(record["parameters_json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                "S1 candidate summary contains invalid parameters."
            ) from error
        if (
            record["method"] != expected["method"]
            or parameters != expected["parameters"]
        ):
            raise ValueError(
                "S1 candidate registry differs from the frozen search space."
            )
    numeric_evidence = candidate_summary[
        ["mean_rmsle", "mean_wape_pct", "total_fit_seconds", "total_predict_seconds"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric_evidence).all() or (numeric_evidence < 0).any():
        raise ValueError("S1 candidate summary contains invalid numeric evidence.")
    ranked_first = candidate_summary.sort_values("rank", ignore_index=True).iloc[0]
    if ranked_first["run_id"] != FEATURE_CONTROL_RUN_ID:
        raise ValueError("S1 candidate summary does not rank xgboost_18 first.")
    return manifest, metrics, predictions, candidate_summary


def aggregate_metrics(
    predictions: pd.DataFrame,
    window_metrics: list[dict[str, Any]],
    model_name: str,
) -> dict[str, Any]:
    """Aggregate RMSLE equally by fold and other errors over all rows."""
    selected_metrics = [row for row in window_metrics if row["model"] == model_name]
    selected_predictions = predictions.loc[predictions["model"].eq(model_name)]
    if len(selected_metrics) != len(EVALUATION_WINDOWS):
        raise ValueError(f"{model_name} does not have all 4 window metrics.")
    expected_rows = sum(int(row["rows_expected"]) for row in selected_metrics)
    aggregate = summarize_errors(selected_predictions, expected_rows)
    aggregate["rmsle"] = float(np.mean([row["rmsle"] for row in selected_metrics]))
    return {
        "contract_id": CONTRACT_ID,
        "scope": "aggregate",
        "model": model_name,
        "window_count": len(selected_metrics),
        "rmsle_aggregation": "equal_window_mean",
        **aggregate,
    }


def evaluate_s2_gate(
    window_metrics: list[dict[str, Any]],
    aggregate_by_model: dict[str, dict[str, Any]],
    resource_evidence: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen feature-effect, selected-pipeline, and resource gate."""
    window_ids = [window.window_id for window in EVALUATION_WINDOWS]
    model_names = [
        ACTIVE_MODEL_NAME,
        SEASONAL_NAIVE_NAME,
        FEATURE_CONTROL_MODEL_NAME,
        CHALLENGER_MODEL_NAME,
    ]
    expected_pairs = {
        (model_name, window_id)
        for model_name in model_names
        for window_id in window_ids
    }
    by_model_window = {
        (row["model"], row["window_id"]): row for row in window_metrics
    }
    if (
        len(window_metrics) != len(expected_pairs)
        or set(by_model_window) != expected_pairs
    ):
        raise ValueError(
            "Gate window evidence is incomplete, duplicated, or unexpected."
        )
    if set(aggregate_by_model) != set(model_names):
        raise ValueError("Gate aggregate evidence is incomplete or unexpected.")

    active = aggregate_by_model[ACTIVE_MODEL_NAME]
    seasonal = aggregate_by_model[SEASONAL_NAIVE_NAME]
    feature_control = aggregate_by_model[FEATURE_CONTROL_MODEL_NAME]
    challenger = aggregate_by_model[CHALLENGER_MODEL_NAME]
    improved_windows = [
        window_id
        for window_id in window_ids
        if by_model_window[(CHALLENGER_MODEL_NAME, window_id)]["rmsle"]
        < by_model_window[(ACTIVE_MODEL_NAME, window_id)]["rmsle"]
    ]
    active_worst_wape = max(
        by_model_window[(ACTIVE_MODEL_NAME, window_id)]["wape_pct"]
        for window_id in window_ids
    )
    challenger_worst_wape = max(
        by_model_window[(CHALLENGER_MODEL_NAME, window_id)]["wape_pct"]
        for window_id in window_ids
    )
    criteria = {
        "feature_control_mean_rmsle_lower_than_active_v2": feature_control["rmsle"]
        < active["rmsle"],
        "feature_control_pooled_wape_no_higher_than_active_v2": feature_control[
            "wape_pct"
        ]
        <= active["wape_pct"],
        "selected_mean_rmsle_lower_than_active_v2": challenger["rmsle"]
        < active["rmsle"],
        "selected_mean_rmsle_lower_than_seasonal_naive": challenger["rmsle"]
        < seasonal["rmsle"],
        "selected_pooled_wape_no_higher_than_active_v2": challenger["wape_pct"]
        <= active["wape_pct"],
        "selected_pooled_wape_no_higher_than_seasonal_naive": challenger["wape_pct"]
        <= seasonal["wape_pct"],
        "selected_rmsle_improves_in_at_least_3_windows": len(improved_windows) >= 3,
        "selected_worst_window_wape_no_higher_than_active_v2": challenger_worst_wape
        <= active_worst_wape,
        "candidate_count_complete": resource_evidence["candidate_count"]
        == EXPECTED_CANDIDATES,
        "fit_count_complete": resource_evidence["fit_count"] == MAX_FITS,
        "added_feature_count_within_budget": resource_evidence["added_feature_count"]
        == MAX_ADDED_FEATURES,
        "feature_storage_within_budget": resource_evidence["maximum_feature_bytes"]
        <= MAX_FEATURE_BYTES,
        "fit_time_within_budget": resource_evidence["search_total_fit_seconds"]
        <= resource_evidence["maximum_total_fit_seconds"],
        "prediction_time_within_budget": resource_evidence[
            "selected_mean_predict_seconds"
        ]
        <= resource_evidence["maximum_mean_predict_seconds"],
    }
    return {
        "passed": all(criteria.values()),
        "decision": (
            "eligible_for_review" if all(criteria.values()) else "retain_active_v2"
        ),
        "criteria": criteria,
        "selected_improved_rmsle_windows": improved_windows,
        "selected_improved_rmsle_window_count": len(improved_windows),
        "active_worst_window_wape_pct": active_worst_wape,
        "selected_worst_window_wape_pct": challenger_worst_wape,
    }


def run_s2_candidate_search(
    labeled: pd.DataFrame,
    labeled_features: pd.DataFrame,
    model_start: pd.Timestamp,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[str, str], np.ndarray],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    """Retrain all 30 frozen candidates with the S2 feature over 4 folds."""
    candidates = build_candidate_registry()
    records: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[str, str], np.ndarray] = {}
    fold_contexts: dict[str, dict[str, Any]] = {}
    fold_resources: list[dict[str, Any]] = []

    for window in EVALUATION_WINDOWS:
        print(
            f"{window.window_id}: preparing S2 train through "
            f"{window.forecast_origin} and validation {window.scoring_start} "
            f"to {window.scoring_end}...",
            flush=True,
        )
        training, future, _history, actual_context = build_fold_inputs(
            labeled, labeled_features, model_start, window
        )
        training[FEATURE_NAME] = labeled_features.loc[
            training.index, FEATURE_NAME
        ].to_numpy(dtype=np.float32)
        future = add_sales_mean_lag_16_35(
            future,
            labeled.loc[labeled["date"].le(window.origin)],
        )
        if (
            future["date"] - pd.Timedelta(days=min(FEATURE_LAGS)) > window.origin
        ).any():
            raise RuntimeError(
                f"{window.window_id} S2 feature crosses the forecast origin."
            )

        processor = make_feature_processor(
            categorical_features=CATEGORICAL_FEATURES,
            numeric_features=S2_NUMERIC_FEATURES,
        )
        training_matrix = processor.fit_transform(
            training[S2_MODEL_FEATURES]
        ).astype(np.float32)
        validation_matrix = processor.transform(
            future[S2_MODEL_FEATURES]
        ).astype(np.float32)
        training_target = np.log1p(training["sales"].to_numpy(dtype=np.float32))
        validation_target = actual_context["sales"].to_numpy(dtype=np.float64)

        for candidate_number, candidate in enumerate(candidates, start=1):
            model = build_model(candidate["method"], candidate["parameters"])
            fit_started = perf_counter()
            model.fit(training_matrix, training_target)
            fit_seconds = perf_counter() - fit_started
            predict_started = perf_counter()
            predicted_sales = np.clip(
                np.expm1(model.predict(validation_matrix)), 0.0, None
            )
            predict_seconds = perf_counter() - predict_started
            repeated_sales = np.clip(
                np.expm1(model.predict(validation_matrix)), 0.0, None
            )
            if not np.array_equal(predicted_sales, repeated_sales):
                raise RuntimeError(
                    f"{window.window_id} {candidate['run_id']} predictions changed."
                )
            if len(predicted_sales) != len(future):
                raise RuntimeError("Candidate prediction row count is invalid.")
            if not np.isfinite(predicted_sales).all() or (predicted_sales < 0).any():
                raise RuntimeError(
                    "Candidate predictions must be finite and non-negative."
                )

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
                f"{window.window_id}: S2 candidate {candidate_number:02d}/"
                f"{len(candidates):02d} {candidate['run_id']} complete.",
                flush=True,
            )
            del model, predicted_sales, repeated_sales

        fold_contexts[window.window_id] = {
            "future": future[["id", *BASE_KEY]].copy(),
            "actual_context": actual_context,
        }
        fold_resources.append(
            {
                "window_id": window.window_id,
                "training_rows": int(len(training)),
                "validation_rows": int(len(future)),
                "feature_bytes": int(
                    training[FEATURE_NAME].memory_usage(index=False, deep=True)
                ),
                "training_matrix_bytes": matrix_nbytes(training_matrix),
                "validation_matrix_bytes": matrix_nbytes(validation_matrix),
                "transformed_feature_count": int(training_matrix.shape[1]),
            }
        )
        del (
            processor,
            training_matrix,
            validation_matrix,
            training_target,
            validation_target,
        )
        gc.collect()

    fold_metrics = pd.DataFrame(records)
    candidate_summary = aggregate_candidate_metrics(fold_metrics)
    if len(fold_metrics) != MAX_FITS or len(candidate_summary) != EXPECTED_CANDIDATES:
        raise RuntimeError("S2 candidate search did not complete its frozen registry.")
    return (
        fold_metrics,
        candidate_summary,
        prediction_cache,
        fold_contexts,
        fold_resources,
    )


def materialize_s2_fold_evidence(
    selected_run_id: str,
    fold_metrics: pd.DataFrame,
    prediction_cache: dict[tuple[str, str], np.ndarray],
    fold_contexts: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Attach actuals only to the fixed-control and selected S2 candidates."""
    scored_frames: list[pd.DataFrame] = []
    metric_records: list[dict[str, Any]] = []
    model_runs = [
        (FEATURE_CONTROL_RUN_ID, FEATURE_CONTROL_MODEL_NAME),
        (selected_run_id, CHALLENGER_MODEL_NAME),
    ]
    for window in EVALUATION_WINDOWS:
        context = fold_contexts[window.window_id]
        future = context["future"]
        for run_id, model_name in model_runs:
            cache_key = (window.window_id, run_id)
            if cache_key not in prediction_cache:
                raise ValueError(f"S2 prediction cache is missing {cache_key}.")
            predictions = future.copy()
            predictions["forecast_sales"] = prediction_cache[cache_key]
            validate_prediction_population(predictions, future, model_name)
            scored = attach_actual_context(
                predictions,
                context["actual_context"],
                window,
                model_name,
            )
            scored["contract_id"] = CONTRACT_ID
            timing = fold_metrics.loc[
                fold_metrics["run_id"].eq(run_id)
                & fold_metrics["window_id"].eq(window.window_id)
            ]
            if len(timing) != 1:
                raise ValueError("S2 candidate timing evidence is incomplete.")
            timing_row = timing.iloc[0]
            metric_records.append(
                {
                    "contract_id": CONTRACT_ID,
                    "scope": "window",
                    "window_id": window.window_id,
                    "forecast_origin": window.forecast_origin,
                    "scoring_start": window.scoring_start,
                    "scoring_end": window.scoring_end,
                    "model": model_name,
                    "source_run_id": run_id,
                    **summarize_errors(scored, len(future)),
                    "fit_seconds": float(timing_row["fit_seconds"]),
                    "predict_seconds": float(timing_row["predict_seconds"]),
                    "repeat_prediction_match": True,
                }
            )
            scored_frames.append(scored)
    return pd.concat(scored_frames, ignore_index=True), metric_records


def run_s2_evaluation(
    labeled_path: Path = DEFAULT_HISTORY_PATH,
    s1_run_directory: Path = DEFAULT_S1_RUN_DIRECTORY,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str = DEFAULT_RUN_ID,
) -> tuple[Path, dict[str, Any]]:
    """Run S2-01 once and atomically write its private evidence bundle."""
    started_at = datetime.now(timezone.utc)
    started = perf_counter()
    require_available_output_paths(output_root, run_id)
    s1_manifest, s1_metrics, s1_predictions, s1_candidate_summary = (
        load_frozen_s1_reference(s1_run_directory)
    )
    labeled_path = require_project_path(labeled_path, "S2 labeled history")
    if not labeled_path.is_file():
        raise FileNotFoundError(f"S2 labeled history is missing: {labeled_path}")
    labeled_sha256 = sha256_file(labeled_path)
    s1_labeled_evidence = s1_manifest.get("inputs", {}).get("labeled_history", {})
    if labeled_sha256 != s1_labeled_evidence.get("sha256"):
        raise ValueError("Current labeled history differs from the frozen S1 input.")

    labeled = read_processed_table(labeled_path, has_target=True)
    validate_evaluation_source(labeled, EVALUATION_WINDOWS)
    window_selection = validate_frozen_window_selection(labeled)
    labeled_features = add_exact_sales_lags(labeled, labeled)
    labeled_features = add_sales_mean_lag_16_35(labeled_features, labeled)
    model_start = find_model_start(labeled_features)

    baseline_predictions = s1_predictions.loc[
        s1_predictions["model"].isin([SELECTED_MODEL_NAME, SEASONAL_NAIVE_NAME])
    ].copy()
    baseline_predictions["contract_id"] = CONTRACT_ID
    baseline_predictions["model"] = baseline_predictions["model"].replace(
        {SELECTED_MODEL_NAME: ACTIVE_MODEL_NAME}
    )
    baseline_window_metrics: list[dict[str, Any]] = []
    for row in s1_metrics["validation_window_metrics"]:
        record = dict(row)
        record["contract_id"] = CONTRACT_ID
        if record["model"] == SELECTED_MODEL_NAME:
            record["model"] = ACTIVE_MODEL_NAME
        baseline_window_metrics.append(record)

    (
        candidate_fold_metrics,
        candidate_summary,
        prediction_cache,
        fold_contexts,
        fold_resources,
    ) = run_s2_candidate_search(labeled, labeled_features, model_start)
    selected = candidate_summary.sort_values("rank", ignore_index=True).iloc[0]
    selected_run_id = str(selected["run_id"])
    challenger_predictions, challenger_window_metrics = materialize_s2_fold_evidence(
        selected_run_id,
        candidate_fold_metrics,
        prediction_cache,
        fold_contexts,
    )

    predictions = pd.concat(
        [baseline_predictions, challenger_predictions], ignore_index=True
    )
    window_metrics = [*baseline_window_metrics, *challenger_window_metrics]
    aggregate_rows = [
        aggregate_metrics(predictions, window_metrics, model_name)
        for model_name in [
            ACTIVE_MODEL_NAME,
            SEASONAL_NAIVE_NAME,
            FEATURE_CONTROL_MODEL_NAME,
            CHALLENGER_MODEL_NAME,
        ]
    ]
    aggregate_by_model = {row["model"]: row for row in aggregate_rows}

    s1_search_total_fit_seconds = float(
        s1_candidate_summary["total_fit_seconds"].sum()
    )
    active_predict_seconds = [
        float(row["predict_seconds"])
        for row in baseline_window_metrics
        if row["model"] == ACTIVE_MODEL_NAME
    ]
    selected_predict_seconds = [
        float(row["predict_seconds"])
        for row in challenger_window_metrics
        if row["model"] == CHALLENGER_MODEL_NAME
    ]
    resource_evidence = {
        "candidate_count": int(len(candidate_summary)),
        "expected_candidate_count": EXPECTED_CANDIDATES,
        "fit_count": int(len(candidate_fold_metrics)),
        "maximum_fit_count": MAX_FITS,
        "added_feature_count": len(S2_MODEL_FEATURES) - len(MODEL_FEATURES),
        "maximum_added_feature_count": MAX_ADDED_FEATURES,
        "maximum_feature_bytes": max(row["feature_bytes"] for row in fold_resources),
        "feature_byte_budget": MAX_FEATURE_BYTES,
        "s1_search_total_fit_seconds": s1_search_total_fit_seconds,
        "search_total_fit_seconds": float(candidate_fold_metrics["fit_seconds"].sum()),
        "maximum_total_fit_seconds": s1_search_total_fit_seconds
        * MAX_FIT_TIME_MULTIPLIER,
        "baseline_mean_predict_seconds": float(np.mean(active_predict_seconds)),
        "selected_mean_predict_seconds": float(
            np.mean(selected_predict_seconds)
        ),
        "maximum_mean_predict_seconds": float(np.mean(active_predict_seconds))
        * MAX_MEAN_PREDICT_TIME_MULTIPLIER,
        "folds": fold_resources,
    }
    gate = evaluate_s2_gate(window_metrics, aggregate_by_model, resource_evidence)
    diagnostics = build_slice_diagnostics(predictions)
    diagnostics["contract_id"] = CONTRACT_ID

    completed_at = datetime.now(timezone.utc)
    metrics_payload = make_json_safe(
        {
            "contract_id": CONTRACT_ID,
            "run_id": run_id,
            "evidence_scope": "retrospective controlled historical comparison",
            "hypothesis": {
                "feature": FEATURE_NAME,
                "lags": list(FEATURE_LAGS),
                "feature_control_run_id": FEATURE_CONTROL_RUN_ID,
                "feature_control_method": METHOD,
                "feature_control_parameters": PARAMETERS,
            },
            "selection": {
                "criterion": "equal-fold mean RMSLE",
                "tie_breaker": "mean WAPE",
                "selected": selected.to_dict(),
                "feature_control": candidate_summary.loc[
                    candidate_summary["run_id"].eq(FEATURE_CONTROL_RUN_ID)
                ].iloc[0].to_dict(),
            },
            "gate": gate,
            "validation_window_metrics": window_metrics,
            "validation_aggregate_metrics": aggregate_rows,
            "resources": resource_evidence,
        }
    )
    code_paths = [
        PROJECT_ROOT / "evaluate_store_sales_s2.py",
        PROJECT_ROOT / "evaluate_store_sales.py",
        PROJECT_ROOT / "store_sales_model.py",
        PROJECT_ROOT / "store_sales_preprocessing.py",
    ]
    manifest = make_json_safe(
        {
            "contract_id": CONTRACT_ID,
            "run_id": run_id,
            "created_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "elapsed_seconds": perf_counter() - started,
            "evidence_scope": "retrospective controlled historical comparison",
            "inputs": {
                "labeled_history": {
                    "file": labeled_path.name,
                    "bytes": labeled_path.stat().st_size,
                    "sha256": labeled_sha256,
                    "rows": len(labeled),
                },
                "s1_reference": {
                    "run_id": s1_manifest["run_id"],
                    "manifest_sha256": sha256_file(
                        s1_run_directory / "manifest.json"
                    ),
                    "prediction_content_sha256": s1_manifest[
                        "validation_prediction_content_sha256"
                    ],
                },
            },
            "configuration": {
                "feature": FEATURE_NAME,
                "feature_lags": list(FEATURE_LAGS),
                "base_model_features": MODEL_FEATURES,
                "challenger_model_features": S2_MODEL_FEATURES,
                "candidate_registry": build_candidate_registry(),
                "selection_criterion": "equal-fold mean RMSLE",
                "selection_tie_breaker": "mean WAPE",
                "selected_run_id": selected_run_id,
                "feature_control_run_id": FEATURE_CONTROL_RUN_ID,
                "windows": [asdict(window) for window in EVALUATION_WINDOWS],
                "window_selection": window_selection,
                "resource_budget": {
                    "maximum_fits": MAX_FITS,
                    "maximum_added_features": MAX_ADDED_FEATURES,
                    "maximum_feature_bytes": MAX_FEATURE_BYTES,
                    "maximum_fit_time_multiplier": MAX_FIT_TIME_MULTIPLIER,
                    "maximum_mean_predict_time_multiplier": (
                        MAX_MEAN_PREDICT_TIME_MULTIPLIER
                    ),
                },
            },
            "gate": gate,
            "code": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in code_paths
            },
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "libraries": CURRENT_LIBRARY_VERSIONS,
            },
            "prediction_content_sha256": prediction_content_sha256(predictions),
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
    )
    return output_directory, metrics_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the S2-01 feature across the frozen 30-candidate search."
    )
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument(
        "--s1-run-directory", type=Path, default=DEFAULT_S1_RUN_DIRECTORY
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_directory, metrics = run_s2_evaluation(
        labeled_path=args.history,
        s1_run_directory=args.s1_run_directory,
        output_root=args.output_root,
        run_id=args.run_id,
    )
    gate = metrics["gate"]
    aggregate = {
        row["model"]: row for row in metrics["validation_aggregate_metrics"]
    }
    print(f"Selected S2 candidate: {metrics['selection']['selected']['run_id']}")
    print(f"S2-01 gate: {'PASS' if gate['passed'] else 'RETAIN V2'}")
    print(
        "Mean RMSLE: "
        f"{aggregate[CHALLENGER_MODEL_NAME]['rmsle']:.6f} selected S2; "
        f"{aggregate[FEATURE_CONTROL_MODEL_NAME]['rmsle']:.6f} "
        "xgboost_18 with feature; "
        f"{aggregate[ACTIVE_MODEL_NAME]['rmsle']:.6f} active V2"
    )
    print(
        "Pooled WAPE: "
        f"{aggregate[CHALLENGER_MODEL_NAME]['wape_pct']:.4f}% selected S2; "
        f"{aggregate[FEATURE_CONTROL_MODEL_NAME]['wape_pct']:.4f}% "
        "xgboost_18 with feature; "
        f"{aggregate[ACTIVE_MODEL_NAME]['wape_pct']:.4f}% active V2"
    )
    print(
        "Selected S2 improved RMSLE windows: "
        f"{gate['selected_improved_rmsle_window_count']}/4"
    )
    print(f"Output: {output_directory}")


if __name__ == "__main__":
    main()
