"""Evaluate the single predeclared S2-01 feature challenger.

The command reuses the verified S1 predictions as immutable references and
fits only the unchanged active XGBoost configuration with one additional,
cutoff-safe sales-history mean. It never changes the active V2 artifact.
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
    attach_actual_context,
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
CHALLENGER_MODEL_NAME = "s2_xgboost_18_sales_mean_lag_16_35"
METHOD = "XGBoost Regression"
PARAMETERS: dict[str, float | int] = {
    "learning_rate": 0.1,
    "max_depth": 8,
    "n_estimators": 500,
}
S2_NUMERIC_FEATURES = [*NUMERIC_FEATURES, FEATURE_NAME]
S2_MODEL_FEATURES = [*MODEL_FEATURES, FEATURE_NAME]
MAX_FITS = 4
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
        raise ValueError("S2 feature history must be unique by date, store, and family.")
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
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
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
    return manifest, metrics, predictions


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
    """Apply the complete frozen S2-01 quality and resource gate."""
    window_ids = [window.window_id for window in EVALUATION_WINDOWS]
    model_names = [ACTIVE_MODEL_NAME, SEASONAL_NAIVE_NAME, CHALLENGER_MODEL_NAME]
    expected_pairs = {
        (model_name, window_id)
        for model_name in model_names
        for window_id in window_ids
    }
    by_model_window = {
        (row["model"], row["window_id"]): row for row in window_metrics
    }
    if len(window_metrics) != len(expected_pairs) or set(by_model_window) != expected_pairs:
        raise ValueError("Gate window evidence is incomplete, duplicated, or unexpected.")
    if set(aggregate_by_model) != set(model_names):
        raise ValueError("Gate aggregate evidence is incomplete or unexpected.")

    active = aggregate_by_model[ACTIVE_MODEL_NAME]
    seasonal = aggregate_by_model[SEASONAL_NAIVE_NAME]
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
        "mean_rmsle_lower_than_active_v2": challenger["rmsle"] < active["rmsle"],
        "mean_rmsle_lower_than_seasonal_naive": challenger["rmsle"]
        < seasonal["rmsle"],
        "pooled_wape_no_higher_than_active_v2": challenger["wape_pct"]
        <= active["wape_pct"],
        "pooled_wape_no_higher_than_seasonal_naive": challenger["wape_pct"]
        <= seasonal["wape_pct"],
        "rmsle_improves_in_at_least_3_windows": len(improved_windows) >= 3,
        "worst_window_wape_no_higher_than_active_v2": challenger_worst_wape
        <= active_worst_wape,
        "fit_count_within_budget": resource_evidence["fit_count"] <= MAX_FITS,
        "added_feature_count_within_budget": resource_evidence["added_feature_count"]
        <= MAX_ADDED_FEATURES,
        "feature_storage_within_budget": resource_evidence["maximum_feature_bytes"]
        <= MAX_FEATURE_BYTES,
        "fit_time_within_budget": resource_evidence["challenger_total_fit_seconds"]
        <= resource_evidence["maximum_total_fit_seconds"],
        "prediction_time_within_budget": resource_evidence[
            "challenger_mean_predict_seconds"
        ]
        <= resource_evidence["maximum_mean_predict_seconds"],
    }
    return {
        "passed": all(criteria.values()),
        "decision": "eligible_for_review" if all(criteria.values()) else "retain_active_v2",
        "criteria": criteria,
        "improved_rmsle_windows": improved_windows,
        "improved_rmsle_window_count": len(improved_windows),
        "active_worst_window_wape_pct": active_worst_wape,
        "challenger_worst_window_wape_pct": challenger_worst_wape,
    }


def run_challenger_window(
    labeled: pd.DataFrame,
    labeled_features: pd.DataFrame,
    model_start: pd.Timestamp,
    window: Any,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Fit and score the fixed S2 challenger at one frozen origin."""
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
    if (future["date"] - pd.Timedelta(days=min(FEATURE_LAGS)) > window.origin).any():
        raise RuntimeError(f"{window.window_id} S2 feature crosses the forecast origin.")

    processor = make_feature_processor(
        categorical_features=CATEGORICAL_FEATURES,
        numeric_features=S2_NUMERIC_FEATURES,
    )
    training_matrix = processor.fit_transform(training[S2_MODEL_FEATURES]).astype(
        np.float32
    )
    validation_matrix = processor.transform(future[S2_MODEL_FEATURES]).astype(np.float32)
    model = build_model(METHOD, PARAMETERS)
    target = np.log1p(training["sales"].to_numpy(dtype=np.float32))

    fit_started = perf_counter()
    model.fit(training_matrix, target)
    fit_seconds = perf_counter() - fit_started
    predict_started = perf_counter()
    predicted_sales = np.clip(np.expm1(model.predict(validation_matrix)), 0.0, None)
    predict_seconds = perf_counter() - predict_started
    repeated_sales = np.clip(np.expm1(model.predict(validation_matrix)), 0.0, None)
    if not np.array_equal(predicted_sales, repeated_sales):
        raise RuntimeError(f"{window.window_id} challenger predictions changed on repeat.")

    predictions = future[["id", *BASE_KEY]].copy()
    predictions["forecast_sales"] = predicted_sales.astype(np.float32)
    validate_prediction_population(predictions, future, CHALLENGER_MODEL_NAME)
    scored = attach_actual_context(
        predictions, actual_context, window, CHALLENGER_MODEL_NAME
    )
    scored["contract_id"] = CONTRACT_ID
    metrics = {
        "contract_id": CONTRACT_ID,
        "scope": "window",
        "window_id": window.window_id,
        "forecast_origin": window.forecast_origin,
        "scoring_start": window.scoring_start,
        "scoring_end": window.scoring_end,
        "model": CHALLENGER_MODEL_NAME,
        **summarize_errors(scored, len(future)),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "repeat_prediction_match": True,
    }
    resources = {
        "window_id": window.window_id,
        "training_rows": int(len(training)),
        "validation_rows": int(len(future)),
        "feature_bytes": int(training[FEATURE_NAME].memory_usage(index=False, deep=True)),
        "training_matrix_bytes": matrix_nbytes(training_matrix),
        "validation_matrix_bytes": matrix_nbytes(validation_matrix),
        "transformed_feature_count": int(training_matrix.shape[1]),
    }
    del (
        processor,
        model,
        training_matrix,
        validation_matrix,
        target,
        predicted_sales,
        repeated_sales,
    )
    gc.collect()
    return scored, metrics, resources


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
    s1_manifest, s1_metrics, s1_predictions = load_frozen_s1_reference(
        s1_run_directory
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

    challenger_predictions: list[pd.DataFrame] = []
    challenger_window_metrics: list[dict[str, Any]] = []
    fold_resources: list[dict[str, Any]] = []
    for window in EVALUATION_WINDOWS:
        print(f"{window.window_id}: fitting the fixed S2-01 challenger...", flush=True)
        scored, metrics, resources = run_challenger_window(
            labeled, labeled_features, model_start, window
        )
        challenger_predictions.append(scored)
        challenger_window_metrics.append(metrics)
        fold_resources.append(resources)

    predictions = pd.concat(
        [baseline_predictions, *challenger_predictions], ignore_index=True
    )
    window_metrics = [*baseline_window_metrics, *challenger_window_metrics]
    aggregate_rows = [
        aggregate_metrics(predictions, window_metrics, model_name)
        for model_name in [ACTIVE_MODEL_NAME, SEASONAL_NAIVE_NAME, CHALLENGER_MODEL_NAME]
    ]
    aggregate_by_model = {row["model"]: row for row in aggregate_rows}

    selected = s1_metrics["selection"]["selected"]
    baseline_total_fit_seconds = float(selected["total_fit_seconds"])
    active_predict_seconds = [
        float(row["predict_seconds"])
        for row in baseline_window_metrics
        if row["model"] == ACTIVE_MODEL_NAME
    ]
    challenger_predict_seconds = [
        float(row["predict_seconds"]) for row in challenger_window_metrics
    ]
    resource_evidence = {
        "fit_count": len(challenger_window_metrics),
        "maximum_fit_count": MAX_FITS,
        "added_feature_count": len(S2_MODEL_FEATURES) - len(MODEL_FEATURES),
        "maximum_added_feature_count": MAX_ADDED_FEATURES,
        "maximum_feature_bytes": max(row["feature_bytes"] for row in fold_resources),
        "feature_byte_budget": MAX_FEATURE_BYTES,
        "baseline_total_fit_seconds": baseline_total_fit_seconds,
        "challenger_total_fit_seconds": float(
            sum(row["fit_seconds"] for row in challenger_window_metrics)
        ),
        "maximum_total_fit_seconds": baseline_total_fit_seconds
        * MAX_FIT_TIME_MULTIPLIER,
        "baseline_mean_predict_seconds": float(np.mean(active_predict_seconds)),
        "challenger_mean_predict_seconds": float(
            np.mean(challenger_predict_seconds)
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
                "method": METHOD,
                "parameters": PARAMETERS,
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
                "method": METHOD,
                "parameters": PARAMETERS,
                "windows": [asdict(window) for window in EVALUATION_WINDOWS],
                "window_selection": window_selection,
                "resource_budget": {
                    "maximum_fits": MAX_FITS,
                    "maximum_added_features": MAX_ADDED_FEATURES,
                    "maximum_feature_bytes": MAX_FEATURE_BYTES,
                    "maximum_fit_time_multiplier": MAX_FIT_TIME_MULTIPLIER,
                    "maximum_mean_predict_time_multiplier": MAX_MEAN_PREDICT_TIME_MULTIPLIER,
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
    )
    return output_directory, metrics_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the fixed S2-01 sales-history mean challenger."
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
    print(f"S2-01 gate: {'PASS' if gate['passed'] else 'RETAIN V2'}")
    print(
        "Mean RMSLE: "
        f"{aggregate[CHALLENGER_MODEL_NAME]['rmsle']:.6f} challenger; "
        f"{aggregate[ACTIVE_MODEL_NAME]['rmsle']:.6f} active V2"
    )
    print(
        "Pooled WAPE: "
        f"{aggregate[CHALLENGER_MODEL_NAME]['wape_pct']:.4f}% challenger; "
        f"{aggregate[ACTIVE_MODEL_NAME]['wape_pct']:.4f}% active V2"
    )
    print(
        "Improved RMSLE windows: "
        f"{gate['improved_rmsle_window_count']}/4"
    )
    print(f"Output: {output_directory}")


if __name__ == "__main__":
    main()
