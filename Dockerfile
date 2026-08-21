FROM python:3.12.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# XGBoost requires the GNU OpenMP runtime in the slim Python image.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && groupadd --system retail \
    && useradd --system --gid retail --no-create-home \
        --home-dir /nonexistent retail \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --requirement requirements.txt

COPY --chown=retail:retail \
    app.py store_sales_model.py store_sales_preprocessing.py ./

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=25)"

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]

FROM base AS ci-smoke

COPY --chown=retail:retail test_store_sales_model.py ./
RUN python -c \
    "from pathlib import Path; from test_store_sales_model import prepare_container_smoke_runtime; prepare_container_smoke_runtime(Path('ci_runtime'))"

ENV RETAIL_FORECAST_ARTIFACT_PATH=/app/ci_runtime/store_sales_forecast_v1.pkl \
    RETAIL_FORECAST_HISTORY_PATH=/app/ci_runtime/store_sales_forecast_v1_history.csv.gz

USER retail

FROM base AS runtime

USER retail
