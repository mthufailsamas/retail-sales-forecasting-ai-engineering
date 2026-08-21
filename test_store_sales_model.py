"""Contract tests for the reusable 16-day Store Sales forecast interface.

The tests use small synthetic tables. They do not read private Kaggle data,
fit a model, rerun tuning, or require a saved local artifact.
"""

from __future__ import annotations

import json
from datetime import datetime
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import UUID

from fastapi import HTTPException, status
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from app import (
    API_KEY_ENV_VAR,
    API_KEY_HEADER_NAME,
    ARTIFACT_PATH_ENV_VAR,
    HISTORY_PATH_ENV_VAR,
    REQUEST_LOGGER,
    SERVICE_METRICS,
    ForecastRuntime,
    app as api_app,
    get_runtime,
    load_runtime,
)
from store_sales_model import (
    ARTIFACT_SCHEMA_VERSION,
    CURRENT_LIBRARY_VERSIONS,
    FORECAST_HORIZON_DAYS,
    FUTURE_COLUMNS,
    INFERENCE_HISTORY_COLUMNS,
    LABELED_COLUMNS,
    MODEL_FEATURES,
    MODEL_VERSION,
    NUMERIC_FEATURES,
    PROJECT_ROOT,
    add_exact_sales_lags,
    load_forecast_artifact,
    predict_forecast,
    read_inference_history,
    save_forecast_artifact,
    validate_forecast_bundle,
    validate_forecast_window,
    validate_store_family_coverage,
    write_batch_forecast,
    write_deployment_history,
)
from store_sales_preprocessing import (
    build_oil_lookup,
    build_transaction_lookup,
    join_store_metadata,
    normalize_holidays,
    validate_final_table,
)


TRAINING_END = pd.Timestamp("2017-08-15")
TEST_API_KEY = "synthetic-test-api-key-1234567890-abcdef"


class SyntheticProcessor:
    """Return one numeric column without fitting a real preprocessing pipeline."""

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        return np.zeros((len(features), 1), dtype=np.float32)


class ConstantModel:
    """Return one fixed log-scale prediction for every input row."""

    def __init__(self, forecast_sales: float = 25.0) -> None:
        self.predicted_log = np.float32(np.log1p(forecast_sales))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), self.predicted_log, dtype=np.float32)


class NonFiniteModel:
    """Represent a broken model whose predictions must be rejected."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), np.nan, dtype=np.float32)


def make_future_batch() -> pd.DataFrame:
    """Build two complete store-family series across the required 16 dates."""
    dates = pd.date_range(
        TRAINING_END + pd.Timedelta(days=1),
        periods=FORECAST_HORIZON_DAYS,
        freq="D",
    )
    pairs = [
        {
            "store_nbr": 1,
            "family": "GROCERY I",
            "city": "Quito",
            "state": "Pichincha",
            "store_type": "A",
            "store_cluster": 1,
        },
        {
            "store_nbr": 2,
            "family": "BEVERAGES",
            "city": "Guayaquil",
            "state": "Guayas",
            "store_type": "B",
            "store_cluster": 2,
        },
    ]

    rows: list[dict[str, object]] = []
    row_id = 1
    for date in dates:
        for pair in pairs:
            row: dict[str, object] = {
                "id": row_id,
                "date": date,
                **pair,
                "month": date.month,
                "day_of_month": date.day,
                "day_of_week": date.dayofweek + 1,
            }
            row.update({column: 0.0 for column in NUMERIC_FEATURES})
            row["onpromotion"] = 3
            rows.append(row)
            row_id += 1
    return pd.DataFrame(rows)


def make_history() -> pd.DataFrame:
    """Build enough labeled history for the four accepted exact sales lags."""
    rows: list[dict[str, object]] = []
    for store_nbr, family, base_sales in [
        (1, "GROCERY I", 100.0),
        (2, "BEVERAGES", 200.0),
    ]:
        for days_before, offset in [(16, 1), (21, 2), (28, 3), (35, 4)]:
            rows.append(
                {
                    "date": TRAINING_END + pd.Timedelta(days=1-days_before),
                    "store_nbr": store_nbr,
                    "family": family,
                    "sales": base_sales + offset,
                }
            )
    return pd.DataFrame(rows)


def make_bundle(model: object | None = None) -> dict[str, object]:
    """Build the smallest valid artifact bundle required for prediction tests."""
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "metadata": {
            "model_version": MODEL_VERSION,
            "method": "Synthetic Regression",
            "model_features": list(MODEL_FEATURES),
            "forecast_horizon_days": FORECAST_HORIZON_DAYS,
            "training_end": TRAINING_END.date().isoformat(),
            "library_versions": dict(CURRENT_LIBRARY_VERSIONS),
        },
        "processor": SyntheticProcessor(),
        "model": model if model is not None else ConstantModel(),
    }


def make_api_records() -> list[dict[str, object]]:
    """Convert the synthetic future interface into JSON-compatible records."""
    future = make_future_batch()[FUTURE_COLUMNS].copy()
    future["date"] = future["date"].dt.strftime("%Y-%m-%d")
    return future.to_dict(orient="records")


def prepare_container_smoke_runtime(output_directory: Path) -> tuple[Path, Path]:
    """Write a private-data-free runtime used only by the CI container smoke."""
    artifact_path = output_directory / "store_sales_forecast_v1.pkl"
    history_path = output_directory / "store_sales_forecast_v1_history.csv.gz"
    saved_artifact, _ = save_forecast_artifact(
        make_bundle(),
        artifact_path,
        overwrite=True,
    )

    history_rows: list[dict[str, object]] = []
    history_dates = pd.date_range(
        TRAINING_END - pd.Timedelta(days=34),
        TRAINING_END,
        freq="D",
    )
    for date_offset, history_date in enumerate(history_dates):
        for store_nbr, family, base_sales in [
            (1, "GROCERY I", 100.0),
            (2, "BEVERAGES", 200.0),
        ]:
            history_rows.append(
                {
                    "date": history_date,
                    "store_nbr": store_nbr,
                    "family": family,
                    "sales": base_sales + date_offset,
                }
            )
    saved_history = write_deployment_history(
        pd.DataFrame(history_rows),
        TRAINING_END,
        history_path,
        overwrite=True,
    )
    return saved_artifact, saved_history


class PreprocessingContractTests(unittest.TestCase):
    def test_oil_lookup_uses_only_the_latest_published_value(self) -> None:
        base_dates = pd.Series([pd.Timestamp("2017-02-01")])
        oil = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2017-01-15", "2017-01-16", "2017-01-20"]
                ),
                "dcoilwtico": [70.0, np.nan, 80.0],
            }
        )

        lookup, evidence = build_oil_lookup(base_dates, oil)

        self.assertEqual(lookup.loc[0, "oil_lag_16"], 70.0)
        self.assertEqual(lookup.loc[0, "oil_lag_16_age_days"], 1.0)
        self.assertTrue(pd.isna(lookup.loc[0, "oil_lag_21"]))
        self.assertFalse(evidence["future_actual_values_used"])

    def test_transaction_lookup_keeps_exact_missing_lags(self) -> None:
        base_store_dates = pd.DataFrame(
            {
                "date": pd.to_datetime(["2017-02-01", "2017-02-01"]),
                "store_nbr": [1, 2],
            }
        )
        transactions = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2017-01-15", "2017-01-16", "2017-01-16"]
                ),
                "store_nbr": [1, 1, 2],
                "transactions": [99, 10, 20],
            }
        )

        lookup, evidence = build_transaction_lookup(
            base_store_dates,
            transactions,
        )

        store_1 = lookup.loc[lookup["store_nbr"].eq(1)].iloc[0]
        store_2 = lookup.loc[lookup["store_nbr"].eq(2)].iloc[0]
        self.assertEqual(store_1["transactions_lag_16"], 10.0)
        self.assertEqual(store_2["transactions_lag_16"], 20.0)
        self.assertTrue(pd.isna(store_1["transactions_lag_21"]))
        self.assertEqual(store_1["transactions_lag_available_count"], 1)
        self.assertFalse(evidence["future_actual_values_used"])

    def test_holiday_normalization_preserves_operating_meaning(self) -> None:
        stores = pd.DataFrame(
            {
                "store_nbr": [1],
                "city": ["Quito"],
                "state": ["Pichincha"],
            }
        )
        holidays = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2017-01-01",
                        "2017-01-02",
                        "2017-01-03",
                        "2017-01-04",
                        "2017-01-05",
                    ]
                ),
                "type": ["Holiday", "Transfer", "Work Day", "Event", "Event"],
                "locale": ["National"] * 5,
                "locale_name": ["Ecuador"] * 5,
                "description": [
                    "Founding Day",
                    "Founding Day Transfer",
                    "Recovery Work Day",
                    "Dia de la Madre",
                    "Terremoto Manabi",
                ],
                "transferred": [True, False, False, False, False],
            }
        )

        normalized, evidence = normalize_holidays(holidays, stores)
        by_date = normalized.set_index("date")

        transferred_source = by_date.loc[pd.Timestamp("2017-01-01")]
        transfer_destination = by_date.loc[pd.Timestamp("2017-01-02")]
        work_day = by_date.loc[pd.Timestamp("2017-01-03")]
        planned_event = by_date.loc[pd.Timestamp("2017-01-04")]
        self.assertEqual(transferred_source["is_holiday"], 0)
        self.assertEqual(transferred_source["is_holiday_transfer_source"], 1)
        self.assertEqual(transfer_destination["is_holiday"], 1)
        self.assertEqual(
            transfer_destination["is_holiday_transfer_destination"],
            1,
        )
        self.assertEqual(work_day["is_special_work_day"], 1)
        self.assertEqual(planned_event["is_planned_event"], 1)
        self.assertNotIn(pd.Timestamp("2017-01-05"), by_date.index)
        self.assertEqual(evidence["earthquake_event_rows_excluded"], 1)

    def test_unknown_event_description_is_rejected(self) -> None:
        stores = pd.DataFrame(
            {
                "store_nbr": [1],
                "city": ["Quito"],
                "state": ["Pichincha"],
            }
        )
        holidays = pd.DataFrame(
            {
                "date": pd.to_datetime(["2017-01-01"]),
                "type": ["Event"],
                "locale": ["National"],
                "locale_name": ["Ecuador"],
                "description": ["Unclassified Parade"],
                "transferred": [False],
            }
        )

        with self.assertRaisesRegex(ValueError, "without an accepted"):
            normalize_holidays(holidays, stores)

    def test_store_join_and_final_validation_preserve_the_base_rows(self) -> None:
        base = pd.DataFrame(
            {
                "id": [1],
                "date": pd.to_datetime(["2017-01-01"]),
                "store_nbr": [1],
                "family": ["GROCERY I"],
                "sales": [10.0],
                "onpromotion": [0],
            }
        )
        stores = pd.DataFrame(
            {
                "store_nbr": [1],
                "city": ["Quito"],
                "state": ["Pichincha"],
                "store_type": ["A"],
                "store_cluster": [1],
            }
        )

        joined, evidence = join_store_metadata("train.csv", base, stores)
        joined["oil_lag_16"] = np.nan
        validation = validate_final_table(
            "train",
            base,
            joined,
            has_target=True,
        )

        self.assertEqual(evidence["input_rows"], evidence["output_rows"])
        self.assertTrue(evidence["id_order_preserved"])
        self.assertEqual(validation["unexpected_missing_cells"], 0)
        broken = joined.copy()
        broken.loc[0, "city"] = pd.NA
        with self.assertRaisesRegex(RuntimeError, "unexpected missing"):
            validate_final_table("train", base, broken, has_target=True)


class SalesLagContractTests(unittest.TestCase):
    def test_exact_sales_lags_use_the_matching_store_and_family(self) -> None:
        future = make_future_batch()
        first_forecast_rows = future.loc[future["date"].eq(future["date"].min())]

        result = add_exact_sales_lags(first_forecast_rows, make_history())

        grocery = result.loc[result["family"] == "GROCERY I"].iloc[0]
        beverages = result.loc[result["family"] == "BEVERAGES"].iloc[0]
        self.assertEqual(grocery["sales_lag_16"], 101.0)
        self.assertEqual(grocery["sales_lag_35"], 104.0)
        self.assertEqual(beverages["sales_lag_16"], 201.0)
        self.assertEqual(beverages["sales_lag_35"], 204.0)
        self.assertEqual(grocery["sales_lag_available_count"], 4)

    def test_missing_sales_history_remains_missing(self) -> None:
        future = make_future_batch()
        first_forecast_rows = future.loc[future["date"].eq(future["date"].min())]
        history = make_history()
        history = history.loc[
            ~history["store_nbr"].eq(1) | ~history["sales"].eq(101.0)
        ]

        result = add_exact_sales_lags(first_forecast_rows, history)

        grocery = result.loc[result["family"] == "GROCERY I"].iloc[0]
        self.assertTrue(pd.isna(grocery["sales_lag_16"]))
        self.assertEqual(grocery["sales_lag_available_count"], 3)


class ForecastWindowContractTests(unittest.TestCase):
    def test_valid_forecast_window_is_accepted(self) -> None:
        validate_forecast_window(make_future_batch(), TRAINING_END)

    def test_wrong_horizon_is_rejected(self) -> None:
        future = make_future_batch()
        future = future.loc[future["date"] < future["date"].max()]

        with self.assertRaisesRegex(ValueError, "16 consecutive dates"):
            validate_forecast_window(future, TRAINING_END)

    def test_forecast_that_starts_after_the_next_day_is_rejected(self) -> None:
        future = make_future_batch()
        future["date"] += pd.Timedelta(days=1)
        future["month"] = future["date"].dt.month
        future["day_of_month"] = future["date"].dt.day
        future["day_of_week"] = future["date"].dt.dayofweek + 1

        with self.assertRaisesRegex(ValueError, "immediately after"):
            validate_forecast_window(future, TRAINING_END)

    def test_duplicate_base_key_is_rejected(self) -> None:
        future = make_future_batch()
        duplicate = future.iloc[[0]].copy()
        duplicate["id"] = future["id"].max() + 1
        future = pd.concat([future, duplicate], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "duplicate base keys"):
            validate_forecast_window(future, TRAINING_END)

    def test_duplicate_id_is_rejected(self) -> None:
        future = make_future_batch()
        future.loc[1, "id"] = future.loc[0, "id"]

        with self.assertRaisesRegex(ValueError, "id must be present and unique"):
            validate_forecast_window(future, TRAINING_END)

    def test_incomplete_pair_horizon_is_rejected(self) -> None:
        future = make_future_batch().drop(index=0).reset_index(drop=True)

        with self.assertRaisesRegex(ValueError, "all 16 forecast dates"):
            validate_forecast_window(future, TRAINING_END)

    def test_future_sales_target_is_rejected(self) -> None:
        future = make_future_batch()
        future["sales"] = 0.0

        with self.assertRaisesRegex(ValueError, "must not contain the sales target"):
            validate_forecast_window(future, TRAINING_END)

    def test_incorrect_calendar_value_is_rejected(self) -> None:
        future = make_future_batch()
        future.loc[0, "day_of_week"] = 7

        with self.assertRaisesRegex(ValueError, "day_of_week"):
            validate_forecast_window(future, TRAINING_END)

    def test_missing_model_feature_is_rejected(self) -> None:
        future = make_future_batch().drop(columns=[MODEL_FEATURES[-1]])

        with self.assertRaisesRegex(ValueError, "missing model features"):
            validate_forecast_window(future, TRAINING_END)

    def test_missing_future_known_context_is_rejected(self) -> None:
        future = make_future_batch()
        future.loc[0, "city"] = pd.NA

        with self.assertRaisesRegex(ValueError, "context contains missing"):
            validate_forecast_window(future, TRAINING_END)

    def test_entire_missing_store_family_pair_is_rejected(self) -> None:
        future = make_future_batch().query("store_nbr == 1").reset_index(drop=True)

        with self.assertRaisesRegex(ValueError, "1 missing"):
            validate_store_family_coverage(future, make_history())

    def test_unexpected_store_family_pair_is_rejected(self) -> None:
        future = make_future_batch().copy()
        extra = future.query("store_nbr == 1").copy()
        extra["id"] += future["id"].max()
        extra["store_nbr"] = 99
        extra["family"] = "NEW FAMILY"
        future = pd.concat([future, extra], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "1 unexpected"):
            validate_store_family_coverage(future, make_history())


class ArtifactAndPredictionContractTests(unittest.TestCase):
    def test_artifact_library_mismatch_is_rejected(self) -> None:
        bundle = make_bundle()
        bundle["metadata"]["library_versions"]["xgboost"] = "different-version"

        with self.assertRaisesRegex(ValueError, "library versions"):
            validate_forecast_bundle(bundle)

    def test_artifact_without_prediction_interfaces_is_rejected(self) -> None:
        bundle = make_bundle()
        bundle["processor"] = object()

        with self.assertRaisesRegex(ValueError, "transform interface"):
            validate_forecast_bundle(bundle)

        bundle = make_bundle()
        bundle["model"] = object()

        with self.assertRaisesRegex(ValueError, "predict interface"):
            validate_forecast_bundle(bundle)

    def test_same_artifact_and_input_produce_the_same_forecast(self) -> None:
        bundle = make_bundle()
        future = make_future_batch()

        first = predict_forecast(bundle, future)
        second = predict_forecast(bundle, future)

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(
            first.columns.tolist(),
            ["id", "date", "store_nbr", "family", "forecast_sales"],
        )
        np.testing.assert_allclose(first["forecast_sales"], 25.0, rtol=1e-6)

    def test_non_finite_model_prediction_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            predict_forecast(make_bundle(NonFiniteModel()), make_future_batch())

    def test_artifact_save_and_load_round_trip(self) -> None:
        with TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            artifact_path = Path(temporary_directory) / "synthetic_artifact.pkl"

            saved_artifact, metadata_path = save_forecast_artifact(
                make_bundle(),
                artifact_path,
            )
            loaded_bundle = load_forecast_artifact(saved_artifact)

            self.assertTrue(saved_artifact.is_file())
            self.assertTrue(metadata_path.is_file())
            self.assertEqual(
                loaded_bundle["metadata"]["model_version"],
                MODEL_VERSION,
            )


class InferenceHistoryContractTests(unittest.TestCase):
    def test_history_reader_keeps_only_the_required_columns(self) -> None:
        labeled = make_future_batch()[FUTURE_COLUMNS].copy()
        labeled["date"] -= pd.Timedelta(days=16)
        labeled["month"] = labeled["date"].dt.month
        labeled["day_of_month"] = labeled["date"].dt.day
        labeled["day_of_week"] = labeled["date"].dt.dayofweek + 1
        labeled["sales"] = 10.0
        labeled = labeled[LABELED_COLUMNS]

        with TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            history_path = Path(temporary_directory) / "history.csv"
            labeled.to_csv(history_path, index=False)

            result = read_inference_history(history_path, TRAINING_END)

        self.assertEqual(
            result.columns.tolist(),
            ["date", "store_nbr", "family", "sales"],
        )
        self.assertEqual(result["date"].max(), TRAINING_END)

    def test_compact_deployment_history_round_trip(self) -> None:
        history = make_history()
        history = pd.concat(
            [
                history,
                pd.DataFrame(
                    [
                        {
                            "date": TRAINING_END,
                            "store_nbr": 1,
                            "family": "GROCERY I",
                            "sales": 110.0,
                        },
                        {
                            "date": TRAINING_END,
                            "store_nbr": 2,
                            "family": "BEVERAGES",
                            "sales": 210.0,
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )

        with TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            history_path = Path(temporary_directory) / "history.csv.gz"
            written_path = write_deployment_history(
                history,
                TRAINING_END,
                history_path,
            )
            result = read_inference_history(written_path, TRAINING_END)

        self.assertEqual(result.columns.tolist(), INFERENCE_HISTORY_COLUMNS)
        pd.testing.assert_frame_equal(
            result.reset_index(drop=True),
            history.sort_values(INFERENCE_HISTORY_COLUMNS[:-1]).reset_index(
                drop=True
            ),
            check_dtype=False,
        )

    def test_api_runtime_uses_configured_private_paths(self) -> None:
        artifact_path = PROJECT_ROOT / "private" / "artifact.pkl"
        history_path = PROJECT_ROOT / "private" / "history.csv.gz"
        bundle = make_bundle()
        history = make_history()

        load_runtime.cache_clear()
        with (
            patch.dict(
                os.environ,
                {
                    ARTIFACT_PATH_ENV_VAR: str(artifact_path),
                    HISTORY_PATH_ENV_VAR: str(history_path),
                },
            ),
            patch(
                "app.load_forecast_artifact",
                return_value=bundle,
            ) as load_artifact,
            patch(
                "app.read_inference_history",
                return_value=history,
            ) as load_history,
        ):
            runtime = load_runtime()
        load_runtime.cache_clear()

        load_artifact.assert_called_once_with(artifact_path)
        load_history.assert_called_once_with(history_path, TRAINING_END)
        self.assertIs(runtime.bundle, bundle)
        self.assertIs(runtime.history, history)


class BatchOutputContractTests(unittest.TestCase):
    def test_non_finite_batch_output_is_rejected(self) -> None:
        forecast = predict_forecast(make_bundle(), make_future_batch())
        forecast.loc[0, "forecast_sales"] = np.inf

        with TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            output_path = Path(temporary_directory) / "forecast.csv"
            with self.assertRaisesRegex(ValueError, "non-finite"):
                write_batch_forecast(forecast, output_path)

    def test_negative_batch_output_is_rejected(self) -> None:
        forecast = predict_forecast(make_bundle(), make_future_batch())
        forecast.loc[0, "forecast_sales"] = -1.0

        with TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            output_path = Path(temporary_directory) / "forecast.csv"
            with self.assertRaisesRegex(ValueError, "negative prediction"):
                write_batch_forecast(forecast, output_path)

    def test_valid_batch_output_is_written(self) -> None:
        forecast = predict_forecast(make_bundle(), make_future_batch())

        with TemporaryDirectory(dir=PROJECT_ROOT) as temporary_directory:
            output_path = Path(temporary_directory) / "forecast.csv"
            written_path = write_batch_forecast(forecast, output_path)

            saved = pd.read_csv(written_path)
            self.assertEqual(len(saved), len(forecast))
            self.assertEqual(saved.columns.tolist(), forecast.columns.tolist())


class ForecastApiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment_patch = patch.dict(
            os.environ,
            {API_KEY_ENV_VAR: TEST_API_KEY},
        )
        self.environment_patch.start()
        runtime = ForecastRuntime(bundle=make_bundle(), history=make_history())
        api_app.dependency_overrides[get_runtime] = lambda: runtime
        REQUEST_LOGGER.disabled = True
        SERVICE_METRICS.reset()
        self.client = TestClient(
            api_app,
            headers={API_KEY_HEADER_NAME: TEST_API_KEY},
        )

    def tearDown(self) -> None:
        api_app.dependency_overrides.clear()
        REQUEST_LOGGER.disabled = False
        self.environment_patch.stop()

    def test_public_health_reports_the_loaded_and_secured_contract(self) -> None:
        response = TestClient(api_app).get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "ready",
                "model_version": MODEL_VERSION,
                "method": "Synthetic Regression",
                "training_end": "2017-08-15",
                "forecast_horizon_days": FORECAST_HORIZON_DAYS,
                "authentication": "configured",
            },
        )

    def test_forecast_and_metrics_require_a_valid_api_key(self) -> None:
        unauthenticated_client = TestClient(api_app)
        missing_key = unauthenticated_client.post(
            "/forecast",
            json={"records": make_api_records()},
        )
        wrong_key = unauthenticated_client.post(
            "/forecast",
            headers={API_KEY_HEADER_NAME: "wrong-key-that-is-not-authorized"},
            json={"records": make_api_records()},
        )
        protected_metrics = unauthenticated_client.get("/metrics")

        self.assertEqual(missing_key.status_code, 401)
        self.assertEqual(wrong_key.status_code, 401)
        self.assertEqual(protected_metrics.status_code, 401)
        self.assertEqual(
            missing_key.headers["WWW-Authenticate"],
            "ApiKey",
        )

        metrics = self.client.get("/metrics").json()
        self.assertEqual(metrics["responses_by_status"]["401"], 3)
        self.assertEqual(metrics["forecast"]["requests_total"], 2)
        self.assertEqual(
            metrics["forecast"]["authentication_rejections_total"],
            2,
        )
        self.assertEqual(metrics["forecast"]["rows_received_total"], 0)

    def test_missing_server_key_makes_the_service_unavailable(self) -> None:
        with patch.dict(os.environ, {API_KEY_ENV_VAR: ""}):
            health_response = self.client.get("/health")
            forecast_response = self.client.post(
                "/forecast",
                json={"records": make_api_records()},
            )
            metrics_response = self.client.get("/metrics")

        self.assertEqual(health_response.status_code, 503)
        self.assertEqual(forecast_response.status_code, 503)
        self.assertEqual(metrics_response.status_code, 503)
        self.assertEqual(
            health_response.json()["detail"],
            "API authentication is not configured.",
        )

    def test_invalid_api_key_is_not_written_to_the_request_log(self) -> None:
        submitted_key = "invalid-secret-value-that-must-not-appear-in-logs"
        unauthenticated_client = TestClient(api_app)
        REQUEST_LOGGER.disabled = False
        with self.assertLogs(REQUEST_LOGGER, level="INFO") as captured:
            response = unauthenticated_client.post(
                "/forecast",
                headers={API_KEY_HEADER_NAME: submitted_key},
                json={"records": make_api_records()},
            )
        REQUEST_LOGGER.disabled = True

        self.assertEqual(response.status_code, 401)
        self.assertEqual(len(captured.records), 1)
        message = captured.records[0].getMessage()
        self.assertNotIn(submitted_key, message)
        self.assertNotIn(API_KEY_HEADER_NAME, message)

    def test_metrics_start_with_an_empty_process_window(self) -> None:
        response = self.client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["monitoring_scope"], "current_process")
        self.assertEqual(body["requests_total"], 0)
        self.assertIsNone(body["last_completed_request_at_utc"])
        self.assertEqual(body["forecast"]["requests_total"], 0)

    def test_metrics_count_a_successful_forecast_without_private_rows(self) -> None:
        forecast_response = self.client.post(
            "/forecast",
            json={"records": make_api_records()},
        )
        metrics_response = self.client.get("/metrics")

        self.assertEqual(forecast_response.status_code, 200)
        self.assertEqual(metrics_response.status_code, 200)
        body = metrics_response.json()
        self.assertEqual(body["requests_total"], 1)
        self.assertEqual(body["requests_by_path"], {"/forecast": 1})
        self.assertEqual(body["responses_by_status"], {"200": 1})
        self.assertEqual(body["latency_ms_by_path"]["/forecast"]["request_count"], 1)
        self.assertEqual(body["forecast"]["requests_total"], 1)
        self.assertEqual(body["forecast"]["rows_received_total"], 32)
        self.assertEqual(body["forecast"]["success_total"], 1)
        self.assertNotIn("GROCERY I", json.dumps(body))
        self.assertNotIn("forecast_sales", json.dumps(body))

    def test_metrics_separate_schema_and_batch_contract_rejections(self) -> None:
        contract_response = self.client.post(
            "/forecast",
            json={"records": make_api_records()[:1]},
        )
        invalid_record = make_api_records()[0]
        invalid_record["sales"] = 10.0
        schema_response = self.client.post(
            "/forecast",
            json={"records": [invalid_record]},
        )
        metrics_response = self.client.get("/metrics")

        self.assertEqual(contract_response.status_code, 422)
        self.assertEqual(schema_response.status_code, 422)
        body = metrics_response.json()
        self.assertEqual(body["requests_total"], 2)
        self.assertEqual(body["responses_by_status"], {"422": 2})
        self.assertEqual(body["forecast"]["requests_total"], 2)
        self.assertEqual(body["forecast"]["rows_received_total"], 1)
        self.assertEqual(body["forecast"]["schema_rejections_total"], 1)
        self.assertEqual(body["forecast"]["contract_rejections_total"], 1)

    def test_health_returns_request_id_and_one_structured_log(self) -> None:
        REQUEST_LOGGER.disabled = False
        with self.assertLogs(REQUEST_LOGGER, level="INFO") as captured:
            response = self.client.get("/health")
        REQUEST_LOGGER.disabled = True

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(captured.records), 1)
        log_record = json.loads(captured.records[0].getMessage())
        self.assertEqual(
            set(log_record),
            {
                "event",
                "timestamp_utc",
                "request_id",
                "method",
                "path",
                "status_code",
                "latency_ms",
                "row_count",
            },
        )
        self.assertEqual(response.headers["X-Request-ID"], log_record["request_id"])
        UUID(hex=log_record["request_id"])
        self.assertIsNotNone(datetime.fromisoformat(log_record["timestamp_utc"]).tzinfo)
        self.assertEqual(log_record["event"], "http_request")
        self.assertEqual(log_record["method"], "GET")
        self.assertEqual(log_record["path"], "/health")
        self.assertEqual(log_record["status_code"], 200)
        self.assertEqual(log_record["row_count"], 0)
        self.assertGreaterEqual(log_record["latency_ms"], 0)

    def test_forecast_returns_one_prediction_per_input_row(self) -> None:
        response = self.client.post(
            "/forecast",
            json={"records": make_api_records()},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model_version"], MODEL_VERSION)
        self.assertEqual(body["forecast_start"], "2017-08-16")
        self.assertEqual(body["forecast_end"], "2017-08-31")
        self.assertEqual(body["row_count"], 32)
        self.assertEqual(len(body["forecasts"]), 32)

    def test_forecast_log_contains_metadata_without_private_rows(self) -> None:
        REQUEST_LOGGER.disabled = False
        with self.assertLogs(REQUEST_LOGGER, level="INFO") as captured:
            response = self.client.post(
                "/forecast",
                json={"records": make_api_records()},
            )
        REQUEST_LOGGER.disabled = True

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(captured.records), 1)
        message = captured.records[0].getMessage()
        log_record = json.loads(message)
        self.assertEqual(response.headers["X-Request-ID"], log_record["request_id"])
        self.assertEqual(log_record["method"], "POST")
        self.assertEqual(log_record["path"], "/forecast")
        self.assertEqual(log_record["status_code"], 200)
        self.assertEqual(log_record["row_count"], 32)
        self.assertNotIn("family", message)
        self.assertNotIn("forecast_sales", message)

    def test_forecast_rejects_an_incomplete_horizon(self) -> None:
        records = make_api_records()
        last_date = max(record["date"] for record in records)
        records = [record for record in records if record["date"] != last_date]

        REQUEST_LOGGER.disabled = False
        with self.assertLogs(REQUEST_LOGGER, level="INFO") as captured:
            response = self.client.post("/forecast", json={"records": records})
        REQUEST_LOGGER.disabled = True

        self.assertEqual(response.status_code, 422)
        self.assertIn("16 consecutive dates", response.json()["detail"])
        log_record = json.loads(captured.records[0].getMessage())
        self.assertEqual(log_record["status_code"], 422)
        self.assertEqual(log_record["row_count"], len(records))

    def test_forecast_rejects_an_entire_missing_pair(self) -> None:
        records = [
            record for record in make_api_records() if record["store_nbr"] == 1
        ]

        response = self.client.post("/forecast", json={"records": records})

        self.assertEqual(response.status_code, 422)
        self.assertIn("1 missing", response.json()["detail"])

    def test_request_schema_rejects_a_future_sales_target(self) -> None:
        records = make_api_records()
        records[0]["sales"] = 10.0

        REQUEST_LOGGER.disabled = False
        with self.assertLogs(REQUEST_LOGGER, level="INFO") as captured:
            response = self.client.post("/forecast", json={"records": records})
        REQUEST_LOGGER.disabled = True

        self.assertEqual(response.status_code, 422)
        log_record = json.loads(captured.records[0].getMessage())
        self.assertEqual(log_record["status_code"], 422)
        self.assertEqual(log_record["row_count"], 0)

    def test_health_returns_503_when_the_runtime_is_unavailable(self) -> None:
        def unavailable_runtime() -> ForecastRuntime:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Forecast runtime is not ready.",
            )

        api_app.dependency_overrides[get_runtime] = unavailable_runtime

        response = self.client.get("/health")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Forecast runtime is not ready.")
        metrics = self.client.get("/metrics").json()
        self.assertEqual(metrics["responses_by_status"]["503"], 1)
        self.assertEqual(metrics["requests_by_path"]["/health"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
