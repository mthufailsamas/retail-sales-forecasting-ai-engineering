"""Verify the live API against the complete private 16-day forecast batch."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from store_sales_model import (
    DEFAULT_BATCH_OUTPUT_PATH,
    DEFAULT_FUTURE_PATH,
    FORECAST_HORIZON_DAYS,
    FUTURE_COLUMNS,
    MODEL_VERSION,
)


DEFAULT_API_URL = "http://127.0.0.1:8000"
API_KEY_ENV_VAR = "RETAIL_FORECAST_API_KEY"
API_KEY_HEADER_NAME = "X-API-Key"
MIN_API_KEY_LENGTH = 32


def read_api_key() -> str:
    """Read the local verification key without accepting it as a CLI argument."""
    api_key = os.getenv(API_KEY_ENV_VAR)
    if api_key is None or len(api_key) < MIN_API_KEY_LENGTH:
        raise RuntimeError(
            f"Set {API_KEY_ENV_VAR} to at least {MIN_API_KEY_LENGTH} characters."
        )
    return api_key


def read_future_records(path: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Read the private processed future interface as JSON-safe records."""
    saved_columns = pd.read_csv(path, nrows=0).columns.tolist()
    if saved_columns != FUTURE_COLUMNS:
        raise ValueError(
            f"{path.name} does not match the processed future contract."
        )

    future = pd.read_csv(path, parse_dates=["date"])
    json_future = future.copy()
    json_future["date"] = json_future["date"].dt.strftime("%Y-%m-%d")
    records = json.loads(json_future.to_json(orient="records"))
    return future, records


def verify_forecast_response(
    payload: dict[str, object],
    future: pd.DataFrame,
    expected_batch_path: Path,
) -> None:
    """Check response identity, horizon, values, and prior batch equivalence."""
    expected_keys = {
        "model_version",
        "forecast_start",
        "forecast_end",
        "row_count",
        "forecasts",
    }
    if set(payload) != expected_keys:
        raise ValueError("API response fields differ from the forecast contract.")
    if payload["model_version"] != MODEL_VERSION:
        raise ValueError("API returned an unexpected model version.")
    if payload["row_count"] != len(future):
        raise ValueError("API response row count differs from the request.")

    forecast = pd.DataFrame(payload["forecasts"])
    expected_columns = ["id", "date", "store_nbr", "family", "forecast_sales"]
    if forecast.columns.tolist() != expected_columns:
        raise ValueError("API forecast rows differ from the output contract.")
    forecast["date"] = pd.to_datetime(forecast["date"])

    expected_dates = pd.date_range(
        future["date"].min(),
        periods=FORECAST_HORIZON_DAYS,
        freq="D",
    )
    actual_dates = pd.DatetimeIndex(forecast["date"].drop_duplicates().sort_values())
    if not actual_dates.equals(expected_dates):
        raise ValueError("API forecast dates differ from the 16-day request horizon.")

    identity_columns = ["id", "date", "store_nbr", "family"]
    pd.testing.assert_frame_equal(
        forecast[identity_columns].reset_index(drop=True),
        future[identity_columns].reset_index(drop=True),
        check_dtype=False,
    )
    forecast_values = forecast["forecast_sales"].to_numpy(dtype=np.float64)
    if not np.isfinite(forecast_values).all() or (forecast_values < 0).any():
        raise ValueError("API returned a non-finite or negative prediction.")

    if not expected_batch_path.is_file():
        raise FileNotFoundError(
            "Verified notebook batch is unavailable for response comparison: "
            f"{expected_batch_path}"
        )
    expected_batch = pd.read_csv(expected_batch_path, parse_dates=["date"])
    pd.testing.assert_frame_equal(
        forecast[identity_columns].reset_index(drop=True),
        expected_batch[identity_columns].reset_index(drop=True),
        check_dtype=False,
    )
    np.testing.assert_allclose(
        forecast_values,
        expected_batch["forecast_sales"].to_numpy(dtype=np.float64),
        rtol=1e-6,
        atol=1e-6,
    )


def verify_monitoring_response(
    payload: dict[str, object],
    successful_batch_rows: int,
) -> None:
    """Check counters after valid, invalid, and unauthenticated batches."""
    expected_keys = {
        "monitoring_scope",
        "started_at_utc",
        "last_completed_request_at_utc",
        "requests_total",
        "requests_by_path",
        "responses_by_status",
        "latency_ms_by_path",
        "forecast",
    }
    if set(payload) != expected_keys:
        raise ValueError("API metrics fields differ from the monitoring contract.")
    if payload["monitoring_scope"] != "current_process":
        raise ValueError("API metrics returned an unexpected monitoring scope.")
    if datetime.fromisoformat(str(payload["started_at_utc"])).tzinfo is None:
        raise ValueError("API metrics start time is not timezone-aware.")
    if payload["last_completed_request_at_utc"] is None:
        raise ValueError("API metrics did not record a completed request.")

    requests_by_path = payload["requests_by_path"]
    responses_by_status = payload["responses_by_status"]
    latency_by_path = payload["latency_ms_by_path"]
    forecast = payload["forecast"]
    if requests_by_path.get("/forecast", 0) < 4:
        raise ValueError("API metrics did not count all forecast verification requests.")
    if responses_by_status.get("200", 0) < 1:
        raise ValueError("API metrics did not count the successful response.")
    if responses_by_status.get("422", 0) < 2:
        raise ValueError("API metrics did not count both rejected responses.")
    if responses_by_status.get("401", 0) < 1:
        raise ValueError("API metrics did not count the authentication rejection.")
    if latency_by_path.get("/forecast", {}).get("request_count", 0) < 4:
        raise ValueError("API metrics did not record forecast latency.")
    if forecast.get("requests_total", 0) < 4:
        raise ValueError("API metrics did not count all forecast outcomes.")
    if forecast.get("rows_received_total", 0) < successful_batch_rows + 1:
        raise ValueError("API metrics did not count received forecast rows.")
    if forecast.get("success_total", 0) < 1:
        raise ValueError("API metrics did not count the successful forecast.")
    if forecast.get("contract_rejections_total", 0) < 1:
        raise ValueError("API metrics did not count the contract rejection.")
    if forecast.get("schema_rejections_total", 0) < 1:
        raise ValueError("API metrics did not count the schema rejection.")
    if forecast.get("authentication_rejections_total", 0) < 1:
        raise ValueError("API metrics did not count the authentication rejection.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send the complete processed future batch to the live API."
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--future", type=Path, default=DEFAULT_FUTURE_PATH)
    parser.add_argument(
        "--expected-batch",
        type=Path,
        default=DEFAULT_BATCH_OUTPUT_PATH,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    future, records = read_future_records(args.future)
    api_key = read_api_key()
    auth_headers = {API_KEY_HEADER_NAME: api_key}
    base_url = args.api_url.rstrip("/")

    with httpx.Client(timeout=300.0) as client:
        health_response = client.get(f"{base_url}/health")
        health_response.raise_for_status()
        if health_response.json().get("status") != "ready":
            raise RuntimeError("API health response is not ready.")

        forecast_response = client.post(
            f"{base_url}/forecast",
            headers=auth_headers,
            json={"records": records},
        )
        if forecast_response.status_code != 200:
            raise RuntimeError(
                "API forecast request failed with "
                f"HTTP {forecast_response.status_code}: {forecast_response.text}"
            )

        contract_response = client.post(
            f"{base_url}/forecast",
            headers=auth_headers,
            json={"records": [records[0]]},
        )
        if contract_response.status_code != 422:
            raise RuntimeError("API did not reject an incomplete forecast batch.")

        invalid_record = dict(records[0])
        invalid_record["sales"] = 10.0
        schema_response = client.post(
            f"{base_url}/forecast",
            headers=auth_headers,
            json={"records": [invalid_record]},
        )
        if schema_response.status_code != 422:
            raise RuntimeError("API did not reject a future sales target.")

        authentication_response = client.post(
            f"{base_url}/forecast",
            json={"records": [records[0]]},
        )
        if authentication_response.status_code != 401:
            raise RuntimeError("API did not reject a missing API key.")

        metrics_response = client.get(
            f"{base_url}/metrics",
            headers=auth_headers,
        )
        metrics_response.raise_for_status()

    payload = forecast_response.json()
    verify_forecast_response(payload, future, args.expected_batch)
    verify_monitoring_response(metrics_response.json(), len(future))
    print("Live API forecast: PASS")
    print(
        f"Rows: {payload['row_count']:,}; dates: "
        f"{payload['forecast_start']} to {payload['forecast_end']}"
    )
    print("Predictions match the verified notebook batch: PASS")
    print("Operational monitoring counters: PASS")
    print("Successful, contract-rejected, and schema-rejected batches: PASS")
    print("Protected endpoints reject missing API credentials: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
