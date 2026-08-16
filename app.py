"""HTTP interface for the verified 16-day retail forecast artifact."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import lru_cache
import json
import logging
import os
from pathlib import Path
from secrets import compare_digest
from threading import Lock
from time import perf_counter
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from store_sales_model import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_HISTORY_PATH,
    FORECAST_HORIZON_DAYS,
    FUTURE_COLUMNS,
    add_exact_sales_lags,
    load_forecast_artifact,
    predict_forecast,
    read_inference_history,
    validate_store_family_coverage,
)


REQUEST_LOGGER = logging.getLogger("retail_forecast_api")
if not REQUEST_LOGGER.handlers:
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter("%(message)s"))
    REQUEST_LOGGER.addHandler(log_handler)
REQUEST_LOGGER.setLevel(logging.INFO)
REQUEST_LOGGER.propagate = False

API_KEY_ENV_VAR = "RETAIL_FORECAST_API_KEY"
ARTIFACT_PATH_ENV_VAR = "RETAIL_FORECAST_ARTIFACT_PATH"
HISTORY_PATH_ENV_VAR = "RETAIL_FORECAST_HISTORY_PATH"
API_KEY_HEADER_NAME = "X-API-Key"
MIN_API_KEY_LENGTH = 32
API_KEY_HEADER = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


class ServiceMetrics:
    """Keep bounded, process-local counters for completed HTTP requests."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.reset()

    def reset(self) -> None:
        """Reset the current process window; used directly by synthetic tests."""
        with self._lock:
            self._started_at_utc = datetime.now(timezone.utc)
            self._last_completed_request_at_utc: datetime | None = None
            self._requests_total = 0
            self._requests_by_path: dict[str, int] = {}
            self._responses_by_status: dict[str, int] = {}
            self._latency_by_path: dict[str, dict[str, float]] = {}
            self._forecast = {
                "requests_total": 0,
                "rows_received_total": 0,
                "success_total": 0,
                "schema_rejections_total": 0,
                "contract_rejections_total": 0,
                "authentication_rejections_total": 0,
                "service_unavailable_total": 0,
                "model_errors_total": 0,
                "other_responses_total": 0,
            }

    def record(
        self,
        path: str,
        status_code: int,
        latency_ms: float,
        row_count: int,
        forecast_outcome: str | None,
    ) -> None:
        """Record aggregate metadata without retaining request or forecast rows."""
        known_path = (
            path if path in {"/health", "/forecast", "/metrics"} else "other"
        )
        status_key = str(status_code)
        with self._lock:
            self._last_completed_request_at_utc = datetime.now(timezone.utc)
            self._requests_total += 1
            self._requests_by_path[known_path] = (
                self._requests_by_path.get(known_path, 0) + 1
            )
            self._responses_by_status[status_key] = (
                self._responses_by_status.get(status_key, 0) + 1
            )

            latency = self._latency_by_path.setdefault(
                known_path,
                {"count": 0.0, "total_ms": 0.0, "maximum_ms": 0.0},
            )
            latency["count"] += 1
            latency["total_ms"] += latency_ms
            latency["maximum_ms"] = max(latency["maximum_ms"], latency_ms)

            if known_path == "/forecast":
                self._forecast["requests_total"] += 1
                self._forecast["rows_received_total"] += row_count
                outcome_key = {
                    "success": "success_total",
                    "schema_rejection": "schema_rejections_total",
                    "contract_rejection": "contract_rejections_total",
                    "authentication_rejection": (
                        "authentication_rejections_total"
                    ),
                    "service_unavailable": "service_unavailable_total",
                    "model_error": "model_errors_total",
                }.get(forecast_outcome, "other_responses_total")
                self._forecast[outcome_key] += 1

    def snapshot(self) -> dict[str, Any]:
        """Return a stable snapshot through the previous completed request."""
        with self._lock:
            latency_by_path = {}
            for path, values in self._latency_by_path.items():
                count = int(values["count"])
                latency_by_path[path] = {
                    "request_count": count,
                    "average_ms": round(values["total_ms"] / count, 3),
                    "maximum_ms": round(values["maximum_ms"], 3),
                }
            return {
                "monitoring_scope": "current_process",
                "started_at_utc": self._started_at_utc,
                "last_completed_request_at_utc": (
                    self._last_completed_request_at_utc
                ),
                "requests_total": self._requests_total,
                "requests_by_path": dict(self._requests_by_path),
                "responses_by_status": dict(self._responses_by_status),
                "latency_ms_by_path": latency_by_path,
                "forecast": dict(self._forecast),
            }


SERVICE_METRICS = ServiceMetrics()


def write_request_log(
    request: Request,
    request_id: str,
    status_code: int,
    latency_ms: float,
) -> None:
    """Write one JSON record without request rows or forecast values."""
    log_record = {
        "event": "http_request",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": status_code,
        "latency_ms": latency_ms,
        "row_count": getattr(request.state, "forecast_row_count", 0),
    }
    REQUEST_LOGGER.info(json.dumps(log_record, separators=(",", ":")))


def classify_forecast_outcome(request: Request, status_code: int) -> str | None:
    """Translate one forecast response into an input or runtime outcome."""
    if request.url.path != "/forecast":
        return None

    explicit_outcome = getattr(request.state, "forecast_outcome", None)
    if explicit_outcome is not None:
        return explicit_outcome
    if status_code == status.HTTP_422_UNPROCESSABLE_CONTENT:
        if getattr(request.state, "forecast_row_count", 0) == 0:
            return "schema_rejection"
        return "contract_rejection"
    if status_code in {
        status.HTTP_401_UNAUTHORIZED,
        status.HTTP_403_FORBIDDEN,
    }:
        return "authentication_rejection"
    if status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        return "service_unavailable"
    if status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
        return "model_error"
    if 200 <= status_code < 300:
        return "success"
    return "other_response"


def record_completed_request(
    request: Request,
    request_id: str,
    status_code: int,
    started_at: float,
) -> None:
    """Send one completed request to both logs and bounded counters."""
    latency_ms = round((perf_counter() - started_at) * 1_000, 3)
    row_count = getattr(request.state, "forecast_row_count", 0)
    SERVICE_METRICS.record(
        path=request.url.path,
        status_code=status_code,
        latency_ms=latency_ms,
        row_count=row_count,
        forecast_outcome=classify_forecast_outcome(request, status_code),
    )
    write_request_log(request, request_id, status_code, latency_ms)


class ForecastRecord(BaseModel):
    """One processed future date-store-family row accepted by the model."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=0)
    date: date
    store_nbr: int = Field(ge=1)
    family: str = Field(min_length=1)
    onpromotion: int = Field(ge=0)
    city: str = Field(min_length=1)
    state: str = Field(min_length=1)
    store_type: str = Field(min_length=1)
    store_cluster: int = Field(ge=1)
    month: int = Field(ge=1, le=12)
    day_of_month: int = Field(ge=1, le=31)
    day_of_week: int = Field(ge=1, le=7)
    oil_lag_16: float | None
    oil_lag_21: float | None
    oil_lag_28: float | None
    oil_lag_35: float | None
    oil_lag_16_age_days: float | None = Field(ge=0)
    oil_lag_21_age_days: float | None = Field(ge=0)
    oil_lag_28_age_days: float | None = Field(ge=0)
    oil_lag_35_age_days: float | None = Field(ge=0)
    transactions_lag_16: float | None = Field(ge=0)
    transactions_lag_21: float | None = Field(ge=0)
    transactions_lag_28: float | None = Field(ge=0)
    transactions_lag_35: float | None = Field(ge=0)
    transactions_lag_available_count: int = Field(ge=0, le=4)
    is_holiday: int = Field(ge=0, le=1)
    is_special_work_day: int = Field(ge=0, le=1)
    is_holiday_transfer_source: int = Field(ge=0, le=1)
    is_holiday_transfer_destination: int = Field(ge=0, le=1)
    is_planned_event: int = Field(ge=0, le=1)
    is_national_schedule: int = Field(ge=0, le=1)
    is_regional_schedule: int = Field(ge=0, le=1)
    is_local_schedule: int = Field(ge=0, le=1)


class ForecastRequest(BaseModel):
    """One complete future batch covering the fixed 16-day horizon."""

    model_config = ConfigDict(extra="forbid")

    records: list[ForecastRecord] = Field(min_length=1)


class ForecastItem(BaseModel):
    id: int
    date: date
    store_nbr: int
    family: str
    forecast_sales: float = Field(ge=0)


class ForecastResponse(BaseModel):
    model_version: str
    forecast_start: date
    forecast_end: date
    row_count: int
    forecasts: list[ForecastItem]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    method: str
    training_end: date
    forecast_horizon_days: int
    authentication: str


class PathLatencyMetrics(BaseModel):
    request_count: int
    average_ms: float
    maximum_ms: float


class ForecastMonitoringMetrics(BaseModel):
    requests_total: int
    rows_received_total: int
    success_total: int
    schema_rejections_total: int
    contract_rejections_total: int
    authentication_rejections_total: int
    service_unavailable_total: int
    model_errors_total: int
    other_responses_total: int


class MetricsResponse(BaseModel):
    monitoring_scope: str
    started_at_utc: datetime
    last_completed_request_at_utc: datetime | None
    requests_total: int
    requests_by_path: dict[str, int]
    responses_by_status: dict[str, int]
    latency_ms_by_path: dict[str, PathLatencyMetrics]
    forecast: ForecastMonitoringMetrics


@dataclass(frozen=True)
class ForecastRuntime:
    """Artifact and compact sales history reused across API requests."""

    bundle: dict[str, Any]
    history: pd.DataFrame


@lru_cache(maxsize=1)
def load_runtime() -> ForecastRuntime:
    """Load the trusted artifact and only its required 35-day sales context."""
    artifact_path = Path(os.getenv(ARTIFACT_PATH_ENV_VAR, DEFAULT_ARTIFACT_PATH))
    history_path = Path(os.getenv(HISTORY_PATH_ENV_VAR, DEFAULT_HISTORY_PATH))
    bundle = load_forecast_artifact(artifact_path)
    training_end = pd.Timestamp(bundle["metadata"]["training_end"])
    history = read_inference_history(history_path, training_end)
    return ForecastRuntime(bundle=bundle, history=history)


def get_runtime() -> ForecastRuntime:
    """Translate local runtime failures into a stable service response."""
    try:
        return load_runtime()
    except (FileNotFoundError, OSError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecast runtime is not ready.",
        ) from error


def get_configured_api_key() -> str:
    """Read the API key from process configuration without persisting it."""
    api_key = os.getenv(API_KEY_ENV_VAR)
    if api_key is None or len(api_key) < MIN_API_KEY_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured.",
        )
    return api_key


def require_api_key(
    provided_api_key: Annotated[str | None, Security(API_KEY_HEADER)],
) -> None:
    """Reject protected requests without the configured API key."""
    configured_api_key = get_configured_api_key()
    if provided_api_key is None or not compare_digest(
        provided_api_key.encode("utf-8"),
        configured_api_key.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Valid API credentials are required.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


RuntimeDependency = Annotated[ForecastRuntime, Depends(get_runtime)]
ConfiguredApiKeyDependency = Annotated[str, Depends(get_configured_api_key)]
AuthenticatedDependency = Annotated[None, Depends(require_api_key)]

app = FastAPI(
    title="Retail Sales Forecasting API",
    version="1.0.0",
    description=(
        "Scores one complete 16-day store-and-product-family batch with the "
        "versioned XGBoost forecast artifact."
    ),
)


@app.middleware("http")
async def log_http_request(request: Request, call_next):
    """Attach a request ID and record one safe structured access log."""
    request_id = uuid4().hex
    request.state.request_id = request_id
    started_at = perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        record_completed_request(request, request_id, 500, started_at)
        raise

    response.headers["X-Request-ID"] = request_id
    record_completed_request(request, request_id, response.status_code, started_at)
    return response


@app.get("/health", response_model=HealthResponse)
def health(
    _configured_api_key: ConfiguredApiKeyDependency,
    runtime: RuntimeDependency,
) -> HealthResponse:
    """Report readiness after authentication and model runtime configuration."""
    metadata = runtime.bundle["metadata"]
    return HealthResponse(
        status="ready",
        model_version=metadata["model_version"],
        method=metadata["method"],
        training_end=pd.Timestamp(metadata["training_end"]).date(),
        forecast_horizon_days=metadata["forecast_horizon_days"],
        authentication="configured",
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics(_authentication: AuthenticatedDependency) -> MetricsResponse:
    """Report process-local API counters through the last completed request."""
    return MetricsResponse(**SERVICE_METRICS.snapshot())


@app.post("/forecast", response_model=ForecastResponse)
def forecast(
    payload: ForecastRequest,
    http_request: Request,
    _authentication: AuthenticatedDependency,
    runtime: RuntimeDependency,
) -> ForecastResponse:
    """Validate and score one complete future 16-day JSON batch."""
    http_request.state.forecast_row_count = len(payload.records)
    try:
        future = pd.DataFrame(
            [record.model_dump(mode="python") for record in payload.records]
        )
        future = future[FUTURE_COLUMNS]
        future["date"] = pd.to_datetime(future["date"])
        validate_store_family_coverage(future, runtime.history)
        future_features = add_exact_sales_lags(future, runtime.history)
        result = predict_forecast(runtime.bundle, future_features)
    except ValueError as error:
        http_request.state.forecast_outcome = "contract_rejection"
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except RuntimeError as error:
        http_request.state.forecast_outcome = "model_error"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Forecast generation failed.",
        ) from error

    forecasts = [
        ForecastItem(
            id=int(row.id),
            date=pd.Timestamp(row.date).date(),
            store_nbr=int(row.store_nbr),
            family=str(row.family),
            forecast_sales=float(row.forecast_sales),
        )
        for row in result.itertuples(index=False)
    ]
    metadata = runtime.bundle["metadata"]
    http_request.state.forecast_outcome = "success"
    return ForecastResponse(
        model_version=metadata["model_version"],
        forecast_start=result["date"].min().date(),
        forecast_end=result["date"].max().date(),
        row_count=len(result),
        forecasts=forecasts,
    )
