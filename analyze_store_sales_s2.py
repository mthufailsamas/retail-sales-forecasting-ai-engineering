"""Audit saved S2 predictions without retraining or changing promotion state.

The audit compares active V2 with the S2-selected challenger on identical
historical rows. It writes a separate immutable private evidence bundle and
never modifies either source evaluation or the active serving artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from evaluate_store_sales import (
    DEFAULT_OUTPUT_ROOT,
    EVALUATION_WINDOWS,
    make_json_safe,
    prediction_content_sha256,
    read_json_object,
    require_available_output_paths,
    require_project_path,
    sha256_file,
)
from evaluate_store_sales_s2 import (
    ACTIVE_MODEL_NAME,
    CHALLENGER_MODEL_NAME,
    CONTRACT_ID as S2_CONTRACT_ID,
    DEFAULT_RUN_ID as DEFAULT_S2_RUN_ID,
)
from store_sales_model import PROJECT_ROOT


CONTRACT_ID = "retail-s2-quality-audit-01"
DEFAULT_RUN_ID = f"{CONTRACT_ID}-v1"
DEFAULT_S2_RUN_DIRECTORY = DEFAULT_OUTPUT_ROOT / DEFAULT_S2_RUN_ID
EVIDENCE_SCOPE = "retrospective controlled diagnostic comparison"
COMPARISON_MODELS = (ACTIVE_MODEL_NAME, CHALLENGER_MODEL_NAME)
SOURCE_REQUIRED_OUTPUTS = {
    "candidate_fold_metrics.csv",
    "candidate_summary.csv",
    "diagnostics.csv.gz",
    "metrics.json",
    "predictions.csv.gz",
}
IDENTITY_COLUMNS = [
    "window_id",
    "id",
    "date",
    "store_nbr",
    "family",
    "actual_sales",
    "onpromotion",
    "is_holiday",
    "forecast_day",
]
REQUIRED_PREDICTION_COLUMNS = [
    "contract_id",
    "model",
    *IDENTITY_COLUMNS,
    "forecast_sales",
]
LOWER_IS_BETTER_METRICS = [
    "rmsle",
    "wape_pct",
    "mean_absolute_error",
    "median_absolute_error",
    "p90_absolute_error",
    "p95_absolute_error",
    "median_absolute_log_error",
    "p90_absolute_log_error",
    "p95_absolute_log_error",
    "maximum_absolute_log_error",
]


def verify_recorded_outputs(
    run_directory: Path,
    manifest: dict[str, Any],
) -> None:
    """Require every S2 output recorded by its manifest to remain unchanged."""
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not SOURCE_REQUIRED_OUTPUTS.issubset(outputs):
        raise ValueError("S2 manifest does not contain the required audit inputs.")
    for filename, evidence in outputs.items():
        if not isinstance(filename, str) or not isinstance(evidence, dict):
            raise ValueError("S2 manifest output evidence is malformed.")
        path = (run_directory / filename).resolve()
        if path.parent != run_directory or not path.is_file():
            raise ValueError(f"S2 output is missing or unsafe: {filename}")
        if path.stat().st_size != evidence.get("bytes"):
            raise ValueError(f"S2 output size differs from its manifest: {filename}")
        if sha256_file(path) != evidence.get("sha256"):
            raise ValueError(f"S2 output hash differs from its manifest: {filename}")


def validate_comparison_population(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return the 2-model population after proving row and context parity."""
    missing = sorted(set(REQUIRED_PREDICTION_COLUMNS).difference(predictions.columns))
    if missing:
        raise ValueError(f"S2 predictions are missing audit columns: {missing}")
    comparison = predictions.loc[
        predictions["model"].isin(COMPARISON_MODELS),
        REQUIRED_PREDICTION_COLUMNS,
    ].copy()
    if set(comparison["model"]) != set(COMPARISON_MODELS):
        raise ValueError("S2 predictions do not contain both comparison models.")
    if comparison.duplicated(["model", "window_id", "id"]).any():
        raise ValueError("S2 predictions contain duplicate model-window rows.")
    expected_windows = {window.window_id for window in EVALUATION_WINDOWS}
    if set(comparison["window_id"]) != expected_windows:
        raise ValueError("S2 predictions do not cover the frozen 4 windows.")

    reference = (
        comparison.loc[comparison["model"].eq(ACTIVE_MODEL_NAME), IDENTITY_COLUMNS]
        .sort_values(["window_id", "id"], ignore_index=True)
    )
    challenger = (
        comparison.loc[comparison["model"].eq(CHALLENGER_MODEL_NAME), IDENTITY_COLUMNS]
        .sort_values(["window_id", "id"], ignore_index=True)
    )
    if not reference.equals(challenger):
        raise ValueError("Active V2 and selected S2 do not share identical scoring rows.")

    numeric = comparison[["actual_sales", "forecast_sales"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError("Audit predictions must be finite and non-negative.")
    return comparison


def load_verified_s2_predictions(
    run_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Verify the immutable S2 bundle before exposing its predictions."""
    run_directory = require_project_path(run_directory, "S2 run directory")
    if not run_directory.is_dir():
        raise FileNotFoundError(f"S2 run directory is missing: {run_directory}")
    manifest = read_json_object(run_directory / "manifest.json", "S2 manifest")
    metrics = read_json_object(run_directory / "metrics.json", "S2 metrics")
    if (
        manifest.get("contract_id") != S2_CONTRACT_ID
        or metrics.get("contract_id") != S2_CONTRACT_ID
    ):
        raise ValueError("Quality audit source uses an unexpected S2 contract.")
    if (
        manifest.get("run_id") != run_directory.name
        or metrics.get("run_id") != run_directory.name
    ):
        raise ValueError("Quality audit source run ID is inconsistent.")
    verify_recorded_outputs(run_directory, manifest)

    predictions = pd.read_csv(
        run_directory / "predictions.csv.gz",
        parse_dates=["date"],
    )
    if prediction_content_sha256(predictions) != manifest.get(
        "prediction_content_sha256"
    ):
        raise ValueError("S2 prediction content differs from its manifest.")
    return manifest, metrics, validate_comparison_population(predictions)


def add_actual_sales_bands(identity_rows: pd.DataFrame) -> pd.DataFrame:
    """Label zero and within-series positive-demand levels for diagnosis only."""
    required = {"window_id", "id", "store_nbr", "family", "actual_sales"}
    if identity_rows.empty or not required.issubset(identity_rows.columns):
        raise ValueError("Sales-band input does not match the diagnostic contract.")
    if identity_rows.duplicated(["window_id", "id"]).any():
        raise ValueError("Sales-band input contains duplicate scoring rows.")
    result = identity_rows.copy()
    result["actual_sales_band"] = "actual_zero"
    positive_mask = result["actual_sales"].gt(0)
    positive = result.loc[positive_mask]
    grouped = positive.groupby(["store_nbr", "family"], observed=True)[
        "actual_sales"
    ]
    q25 = grouped.transform(lambda values: values.quantile(0.25))
    q75 = grouped.transform(lambda values: values.quantile(0.75))
    q90 = grouped.transform(lambda values: values.quantile(0.90))
    values = positive["actual_sales"]
    result.loc[positive.index, "actual_sales_band"] = np.select(
        [values.le(q25), values.le(q75), values.le(q90)],
        ["positive_low", "positive_typical", "positive_high"],
        default="positive_peak",
    )
    return result


def summarize_quality(frame: pd.DataFrame) -> dict[str, float | int | None]:
    """Summarize aggregate, tail, and directional point-forecast errors."""
    if frame.empty:
        raise ValueError("Quality summary requires at least 1 row.")
    actual = frame["actual_sales"].to_numpy(dtype=np.float64)
    predicted = frame["forecast_sales"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(actual).all()
        or not np.isfinite(predicted).all()
        or (actual < 0).any()
        or (predicted < 0).any()
    ):
        raise ValueError("Quality summary requires finite non-negative values.")

    signed = predicted - actual
    absolute = np.abs(signed)
    absolute_log = np.abs(np.log1p(predicted) - np.log1p(actual))
    actual_total = float(actual.sum())

    def percentile(values: np.ndarray, quantile: float) -> float:
        return float(np.quantile(values, quantile))

    return {
        "rows": int(len(frame)),
        "actual_sales": actual_total,
        "predicted_sales": float(predicted.sum()),
        "rmsle": float(np.sqrt(np.mean(absolute_log**2))),
        "wape_pct": (
            float(absolute.sum() / actual_total * 100)
            if actual_total > 0
            else None
        ),
        "signed_bias_pct": (
            float(signed.sum() / actual_total * 100)
            if actual_total > 0
            else None
        ),
        "underforecast_pct": (
            float(np.maximum(-signed, 0).sum() / actual_total * 100)
            if actual_total > 0
            else None
        ),
        "overforecast_pct": (
            float(np.maximum(signed, 0).sum() / actual_total * 100)
            if actual_total > 0
            else None
        ),
        "underforecast_row_pct": float(np.mean(signed < 0) * 100),
        "overforecast_row_pct": float(np.mean(signed > 0) * 100),
        "mean_absolute_error": float(absolute.mean()),
        "median_absolute_error": percentile(absolute, 0.50),
        "p90_absolute_error": percentile(absolute, 0.90),
        "p95_absolute_error": percentile(absolute, 0.95),
        "maximum_absolute_error": float(absolute.max()),
        "median_absolute_log_error": percentile(absolute_log, 0.50),
        "p90_absolute_log_error": percentile(absolute_log, 0.90),
        "p95_absolute_log_error": percentile(absolute_log, 0.95),
        "maximum_absolute_log_error": float(absolute_log.max()),
    }


def build_segment_metrics(comparison: pd.DataFrame) -> pd.DataFrame:
    """Measure both models over the same operational and demand slices."""
    identity = comparison.loc[
        comparison["model"].eq(ACTIVE_MODEL_NAME), IDENTITY_COLUMNS
    ].copy()
    bands = add_actual_sales_bands(identity)[
        ["window_id", "id", "actual_sales_band"]
    ]
    enriched = comparison.merge(
        bands,
        on=["window_id", "id"],
        how="left",
        validate="many_to_one",
    )
    enriched["promotion_segment"] = np.where(
        enriched["onpromotion"].gt(0), "promotion", "no_promotion"
    )
    enriched["holiday_segment"] = np.where(
        enriched["is_holiday"].eq(1), "holiday", "non_holiday"
    )
    enriched["store_family"] = (
        enriched["store_nbr"].astype(str) + " | " + enriched["family"].astype(str)
    )
    slices = [
        ("overall", None),
        ("window", "window_id"),
        ("forecast_day", "forecast_day"),
        ("promotion", "promotion_segment"),
        ("holiday", "holiday_segment"),
        ("actual_sales_band", "actual_sales_band"),
        ("store", "store_nbr"),
        ("family", "family"),
        ("store_family", "store_family"),
    ]
    records: list[dict[str, Any]] = []
    for model, model_rows in enriched.groupby("model", sort=False):
        for slice_type, column in slices:
            groups = [("all", model_rows)] if column is None else model_rows.groupby(
                column, observed=True, sort=True, dropna=False
            )
            for value, group in groups:
                records.append(
                    {
                        "contract_id": CONTRACT_ID,
                        "evidence_scope": EVIDENCE_SCOPE,
                        "model": str(model),
                        "slice_type": slice_type,
                        "slice_value": str(value),
                        **summarize_quality(group),
                    }
                )
    return pd.DataFrame(records).sort_values(
        ["slice_type", "slice_value", "model"], ignore_index=True
    )


def build_segment_comparison(segment_metrics: pd.DataFrame) -> pd.DataFrame:
    """Place active and challenger evidence side by side with signed deltas."""
    key = ["slice_type", "slice_value"]
    active = segment_metrics.loc[segment_metrics["model"].eq(ACTIVE_MODEL_NAME)].copy()
    challenger = segment_metrics.loc[
        segment_metrics["model"].eq(CHALLENGER_MODEL_NAME)
    ].copy()
    active = active.drop(columns=["contract_id", "evidence_scope", "model"])
    challenger = challenger.drop(columns=["contract_id", "evidence_scope", "model"])
    active = active.rename(
        columns={column: f"active_{column}" for column in active.columns if column not in key}
    )
    challenger = challenger.rename(
        columns={
            column: f"challenger_{column}"
            for column in challenger.columns
            if column not in key
        }
    )
    paired = active.merge(challenger, on=key, validate="one_to_one")
    if not paired["active_rows"].equals(paired["challenger_rows"]):
        raise ValueError("Segment comparison populations differ between models.")
    paired.insert(0, "evidence_scope", EVIDENCE_SCOPE)
    paired.insert(0, "contract_id", CONTRACT_ID)
    paired["rows"] = paired.pop("active_rows")
    paired = paired.drop(columns="challenger_rows")
    for metric in LOWER_IS_BETTER_METRICS:
        active_column = f"active_{metric}"
        challenger_column = f"challenger_{metric}"
        paired[f"challenger_minus_active_{metric}"] = (
            paired[challenger_column] - paired[active_column]
        )
        paired[f"challenger_improves_{metric}"] = (
            paired[challenger_column] < paired[active_column]
        ).where(paired[[active_column, challenger_column]].notna().all(axis=1))
    paired["active_absolute_bias_pct"] = paired["active_signed_bias_pct"].abs()
    paired["challenger_absolute_bias_pct"] = paired[
        "challenger_signed_bias_pct"
    ].abs()
    paired["challenger_minus_active_absolute_bias_pct"] = (
        paired["challenger_absolute_bias_pct"] - paired["active_absolute_bias_pct"]
    )
    paired["challenger_improves_absolute_bias"] = (
        paired["challenger_absolute_bias_pct"] < paired["active_absolute_bias_pct"]
    ).where(
        paired[["active_absolute_bias_pct", "challenger_absolute_bias_pct"]]
        .notna()
        .all(axis=1)
    )
    return paired


def build_quality_summary(
    segment_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    source_run_id: str,
    selected_run_id: str,
) -> dict[str, Any]:
    """Create the compact audit interpretation without making a promotion decision."""
    overall = comparison.loc[comparison["slice_type"].eq("overall")].iloc[0]
    window_rows = segment_metrics.loc[segment_metrics["slice_type"].eq("window")]
    stability: dict[str, dict[str, float]] = {}
    for model, rows in window_rows.groupby("model", observed=True):
        rmsle = rows["rmsle"].to_numpy(dtype=np.float64)
        wape = rows["wape_pct"].to_numpy(dtype=np.float64)
        stability[str(model)] = {
            "equal_fold_mean_rmsle": float(rmsle.mean()),
            "sample_std_rmsle": float(rmsle.std(ddof=1)),
            "worst_fold_rmsle": float(rmsle.max()),
            "worst_fold_wape_pct": float(wape.max()),
        }

    slice_summary: list[dict[str, Any]] = []
    for slice_type, rows in comparison.loc[
        comparison["slice_type"].ne("overall")
    ].groupby("slice_type", sort=True):
        record: dict[str, Any] = {
            "slice_type": str(slice_type),
            "slice_count": int(len(rows)),
        }
        for label, column in [
            ("rmsle", "challenger_improves_rmsle"),
            ("wape", "challenger_improves_wape_pct"),
            ("p95_absolute_log_error", "challenger_improves_p95_absolute_log_error"),
            ("absolute_bias", "challenger_improves_absolute_bias"),
        ]:
            valid = rows[column].dropna().astype(bool)
            record[f"{label}_evaluable_slice_count"] = int(len(valid))
            record[f"{label}_improved_slice_count"] = int(valid.sum())
            record[f"{label}_improved_slice_pct"] = (
                float(valid.mean() * 100) if len(valid) else None
            )
        slice_summary.append(record)

    overall_fields = {
        column: make_json_safe(value)
        for column, value in overall.to_dict().items()
        if column not in {"contract_id", "evidence_scope", "slice_type", "slice_value"}
    }
    return make_json_safe(
        {
            "contract_id": CONTRACT_ID,
            "evidence_scope": EVIDENCE_SCOPE,
            "source_s2_run_id": source_run_id,
            "selected_s2_run_id": selected_run_id,
            "comparison_models": list(COMPARISON_MODELS),
            "status": "diagnostic_complete_no_promotion_decision",
            "overall_rmsle_aggregation": "pooled across all saved prediction rows",
            "actual_sales_band_definition": {
                "scope": "each store-family across all 4 saved validation windows",
                "actual_zero": "actual sales equal 0",
                "positive_low": "positive actual sales at or below the series 25th percentile",
                "positive_typical": "above the 25th and at or below the 75th percentile",
                "positive_high": "above the 75th and at or below the 90th percentile",
                "positive_peak": "above the series 90th percentile",
                "use": "retrospective diagnostic only; never a model input",
            },
            "overall_comparison": overall_fields,
            "fold_stability": stability,
            "slice_comparison_summary": slice_summary,
        }
    )


def write_quality_audit_bundle(
    output_root: Path,
    run_id: str,
    summary: dict[str, Any],
    segment_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    manifest: dict[str, Any],
) -> Path:
    """Publish the private audit atomically and never overwrite an existing run."""
    root, final_directory, temporary_directory = require_available_output_paths(
        output_root, run_id
    )
    root.mkdir(parents=True, exist_ok=True)
    temporary_created = False
    try:
        temporary_directory.mkdir()
        temporary_created = True
        summary_path = temporary_directory / "quality_summary.json"
        metrics_path = temporary_directory / "segment_metrics.csv.gz"
        comparison_path = temporary_directory / "segment_comparison.csv.gz"
        manifest_path = temporary_directory / "manifest.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        compression = {"method": "gzip", "compresslevel": 6, "mtime": 0}
        segment_metrics.to_csv(
            metrics_path,
            index=False,
            float_format="%.9g",
            na_rep="",
            compression=compression,
        )
        comparison.to_csv(
            comparison_path,
            index=False,
            float_format="%.9g",
            na_rep="",
            compression=compression,
        )
        outputs = [summary_path, metrics_path, comparison_path]
        manifest["outputs"] = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in outputs
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


def run_quality_audit(
    s2_run_directory: Path = DEFAULT_S2_RUN_DIRECTORY,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str = DEFAULT_RUN_ID,
) -> tuple[Path, dict[str, Any]]:
    """Verify S2, audit saved predictions, and write reproducible diagnostics."""
    output_root = require_project_path(output_root, "Audit output root")
    require_available_output_paths(output_root, run_id)
    source_manifest, source_metrics, predictions = load_verified_s2_predictions(
        s2_run_directory
    )
    selection = source_metrics.get("selection")
    selected = selection.get("selected") if isinstance(selection, dict) else None
    if not isinstance(selected, dict) or not isinstance(selected.get("run_id"), str):
        raise ValueError("S2 source does not contain its selected candidate.")

    segment_metrics = build_segment_metrics(predictions)
    comparison = build_segment_comparison(segment_metrics)
    summary = build_quality_summary(
        segment_metrics,
        comparison,
        source_manifest["run_id"],
        selected["run_id"],
    )
    now = datetime.now(timezone.utc).isoformat()
    manifest = make_json_safe(
        {
            "contract_id": CONTRACT_ID,
            "run_id": run_id,
            "created_at_utc": now,
            "evidence_scope": EVIDENCE_SCOPE,
            "source": {
                "s2_contract_id": S2_CONTRACT_ID,
                "s2_run_id": source_manifest["run_id"],
                "manifest_sha256": sha256_file(s2_run_directory / "manifest.json"),
                "prediction_content_sha256": source_manifest[
                    "prediction_content_sha256"
                ],
                "selected_run_id": selected["run_id"],
            },
            "configuration": {
                "models": list(COMPARISON_MODELS),
                "slice_types": sorted(segment_metrics["slice_type"].unique()),
                "percentiles": [0.25, 0.50, 0.75, 0.90, 0.95],
                "promotion_effect": "none; diagnostic output only",
            },
            "code": {
                "analyze_store_sales_s2.py": {
                    "bytes": Path(__file__).stat().st_size,
                    "sha256": sha256_file(Path(__file__)),
                }
            },
        }
    )
    directory = write_quality_audit_bundle(
        output_root,
        run_id,
        summary,
        segment_metrics,
        comparison,
        manifest,
    )
    return directory, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit saved S2 prediction quality without retraining."
    )
    parser.add_argument(
        "--s2-run-directory", type=Path, default=DEFAULT_S2_RUN_DIRECTORY
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser.parse_args()


def main() -> None:
    directory, summary = run_quality_audit(**vars(parse_args()))
    overall = summary["overall_comparison"]
    print("S2 saved-prediction quality audit: PASS")
    print(
        "Pooled row RMSLE: "
        f"{overall['active_rmsle']:.6f} active V2; "
        f"{overall['challenger_rmsle']:.6f} selected S2"
    )
    print(
        "WAPE: "
        f"{overall['active_wape_pct']:.4f}% active V2; "
        f"{overall['challenger_wape_pct']:.4f}% selected S2"
    )
    print("Decision: diagnostic only; active V2 remains unchanged")
    print(f"Output: {directory}")


if __name__ == "__main__":
    main()
