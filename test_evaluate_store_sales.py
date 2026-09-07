"""Synthetic contracts for the fixed Store Sales historical evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import evaluate_store_sales as evaluator


def make_prediction_rows(
    dates: list[str],
    *,
    store_nbr: int = 1,
    family: str = "A",
    values: list[float] | None = None,
) -> pd.DataFrame:
    row_count = len(dates)
    return pd.DataFrame(
        {
            "id": np.arange(100, 100 + row_count, dtype=np.int64),
            "date": pd.to_datetime(dates),
            "store_nbr": [store_nbr] * row_count,
            "family": [family] * row_count,
            "forecast_sales": values if values is not None else [1.0] * row_count,
        }
    )


class WindowContractTests(unittest.TestCase):
    def test_fixed_windows_are_valid(self) -> None:
        windows = evaluator.validate_window_contract()

        self.assertEqual(
            [window.window_id for window in windows], ["W1", "W2", "W3", "W4"]
        )
        self.assertEqual(
            [window.scoring_start for window in windows],
            ["2016-08-26", "2016-11-25", "2017-02-16", "2017-06-29"],
        )
        self.assertTrue(
            all(len(pd.date_range(window.start, window.end)) == 16 for window in windows)
        )

    def test_selection_contract_uses_no_target_or_realized_context(self) -> None:
        self.assertNotIn("sales", evaluator.WINDOW_SELECTION_SOURCE_COLUMNS)
        self.assertEqual(
            [block.strategy for block in evaluator.WINDOW_SELECTION_BLOCKS],
            ["typical", "planned_event_stress", "holiday_stress", "latest"],
        )

    def test_candidate_strategies_are_deterministic(self) -> None:
        candidates = pd.DataFrame(
            {
                "start": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03"]
                ),
                "end": pd.to_datetime(
                    ["2024-01-16", "2024-01-17", "2024-01-18"]
                ),
                "promotion_share": [0.1, 0.2, 0.9],
                "promotion_units_per_row": [1.0, 2.0, 9.0],
                "holiday_share": [0.0, 0.1, 0.8],
                "planned_event_share": [0.1, 0.5, 0.9],
            }
        )

        expected = {
            "typical": "2024-01-02",
            "planned_event_stress": "2024-01-03",
            "holiday_stress": "2024-01-03",
            "latest": "2024-01-03",
        }
        for strategy, expected_start in expected.items():
            with self.subTest(strategy=strategy):
                selected = evaluator.select_window_candidate(candidates, strategy)
                self.assertEqual(selected["start"], pd.Timestamp(expected_start))

    def test_candidate_builder_rejects_incomplete_16_day_windows(self) -> None:
        block = evaluator.WindowSelectionBlock(
            "T1", "test", "2024-01-01", "2024-01-17", "latest"
        )
        complete = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=17),
                "store_nbr": 1,
                "family": "A",
                "onpromotion": 0,
                "is_holiday": 0,
                "is_planned_event": 0,
            }
        )

        candidates = evaluator.build_window_candidates(complete, block)

        self.assertEqual(len(candidates), 2)
        incomplete = complete.loc[complete["date"].ne(pd.Timestamp("2024-01-09"))]
        with self.assertRaisesRegex(ValueError, "no complete 16-day candidate"):
            evaluator.build_window_candidates(incomplete, block)

    def test_invalid_horizon_and_overlap_are_rejected(self) -> None:
        wrong_horizon = evaluator.EvaluationWindow(
            "W1", "2020-01-01", "2020-01-02", "2020-01-16"
        )
        with self.assertRaisesRegex(ValueError, "16 dates"):
            evaluator.validate_window_contract([wrong_horizon])

        first = evaluator.EvaluationWindow(
            "W1", "2020-01-01", "2020-01-02", "2020-01-17"
        )
        overlapping = evaluator.EvaluationWindow(
            "W2", "2020-01-16", "2020-01-17", "2020-02-01"
        )
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            evaluator.validate_window_contract([first, overlapping])

    def test_source_requires_every_scoring_date(self) -> None:
        rows: list[dict[str, object]] = []
        next_id = 1
        rows.append(
            {
                "id": next_id,
                "date": pd.Timestamp("2016-06-30"),
                "store_nbr": 1,
                "family": "A",
                "sales": 1.0,
            }
        )
        next_id += 1
        for window in evaluator.EVALUATION_WINDOWS:
            for date in pd.date_range(window.start, window.end):
                rows.append(
                    {
                        "id": next_id,
                        "date": date,
                        "store_nbr": 1,
                        "family": "A",
                        "sales": 1.0,
                    }
                )
                next_id += 1
        source = pd.DataFrame(rows)

        evaluator.validate_evaluation_source(source)
        missing_date = source.loc[source["date"].ne(pd.Timestamp("2016-09-01"))]
        with self.assertRaisesRegex(ValueError, "missing required scoring dates"):
            evaluator.validate_evaluation_source(missing_date)


class SeasonalNaiveTests(unittest.TestCase):
    def test_latest_same_weekday_is_used_and_input_order_is_preserved(self) -> None:
        history = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-04", "2024-01-06", "2024-01-10"]),
                "store_nbr": [1, 1, 1],
                "family": ["A", "A", "A"],
                "sales": [7.0, 11.0, 13.0],
            }
        )
        future = pd.DataFrame(
            {
                "id": [2, 1],
                "date": pd.to_datetime(["2024-01-13", "2024-01-11"]),
                "store_nbr": [1, 1],
                "family": ["A", "A"],
            }
        )

        result = evaluator.make_weekly_seasonal_naive(
            future, history, pd.Timestamp("2024-01-10")
        )

        self.assertEqual(result["id"].tolist(), [2, 1])
        self.assertEqual(result["forecast_sales"].tolist(), [11.0, 7.0])

    def test_pair_median_is_used_only_when_weekday_is_absent(self) -> None:
        history = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-09", "2024-01-10"]),
                "store_nbr": [1, 1],
                "family": ["A", "A"],
                "sales": [2.0, 8.0],
            }
        )
        future = pd.DataFrame(
            {
                "id": [1],
                "date": pd.to_datetime(["2024-01-11"]),
                "store_nbr": [1],
                "family": ["A"],
            }
        )

        result = evaluator.make_weekly_seasonal_naive(
            future, history, pd.Timestamp("2024-01-10")
        )

        self.assertEqual(result.loc[0, "forecast_sales"], 5.0)

    def test_future_history_and_missing_pair_are_rejected(self) -> None:
        future = pd.DataFrame(
            {
                "id": [1],
                "date": pd.to_datetime(["2024-01-11"]),
                "store_nbr": [2],
                "family": ["B"],
            }
        )
        history = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-10"]),
                "store_nbr": [1],
                "family": ["A"],
                "sales": [3.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "without eligible history"):
            evaluator.make_weekly_seasonal_naive(
                future, history, pd.Timestamp("2024-01-10")
            )

        future_history = pd.concat(
            [
                history,
                pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2024-01-11"]),
                        "store_nbr": [1],
                        "family": ["A"],
                        "sales": [4.0],
                    }
                ),
            ],
            ignore_index=True,
        )
        with self.assertRaisesRegex(ValueError, "after the forecast origin"):
            evaluator.make_weekly_seasonal_naive(
                future.assign(store_nbr=1, family="A"),
                future_history,
                pd.Timestamp("2024-01-10"),
            )


class ScoringContractTests(unittest.TestCase):
    def test_prediction_population_requires_exact_order_and_valid_values(self) -> None:
        expected = make_prediction_rows(["2024-01-11", "2024-01-12"]).drop(
            columns="forecast_sales"
        )
        predictions = make_prediction_rows(
            ["2024-01-11", "2024-01-12"], values=[0.0, 2.0]
        )

        evaluator.validate_prediction_population(predictions, expected, "model")
        with self.assertRaisesRegex(ValueError, "keys or order"):
            evaluator.validate_prediction_population(
                predictions.iloc[::-1].reset_index(drop=True), expected, "model"
            )
        for invalid in [np.nan, np.inf, -1.0]:
            changed = predictions.copy()
            changed.loc[0, "forecast_sales"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "finite and non-negative"
            ):
                evaluator.validate_prediction_population(changed, expected, "model")

    def test_actual_context_is_attached_with_scoring_segments(self) -> None:
        window = evaluator.EvaluationWindow(
            "T1", "2024-01-10", "2024-01-11", "2024-01-26"
        )
        predictions = make_prediction_rows(
            ["2024-01-11", "2024-01-12"], values=[0.0, 4.0]
        )
        actuals = predictions.drop(columns="forecast_sales").copy()
        actuals["sales"] = [0.0, 2.0]
        actuals["onpromotion"] = [0, 3]
        actuals["is_holiday"] = [0, 1]
        actuals = actuals[evaluator.ACTUAL_CONTEXT_COLUMNS]

        scored = evaluator.attach_actual_context(predictions, actuals, window, "model")

        self.assertEqual(scored["forecast_day"].tolist(), [1, 2])
        self.assertEqual(scored["promotion_segment"].tolist(), ["no_promotion", "promotion"])
        self.assertEqual(scored["holiday_segment"].tolist(), ["non_holiday", "holiday"])
        self.assertEqual(scored["sales_segment"].tolist(), ["actual_zero", "positive_sales"])

    def test_predictions_are_generated_before_actuals_are_attached(self) -> None:
        window = evaluator.EvaluationWindow(
            "T1", "2024-01-10", "2024-01-11", "2024-01-26"
        )
        history_row = {
            "id": 1,
            "date": pd.Timestamp("2024-01-10"),
            "store_nbr": 1,
            "family": "A",
            "sales": 3.0,
            "onpromotion": 0,
            "is_holiday": 0,
        }
        future_rows = [
            {
                "id": 10 + index,
                "date": date,
                "store_nbr": 1,
                "family": "A",
                "sales": float(index),
                "onpromotion": index % 2,
                "is_holiday": 0,
            }
            for index, date in enumerate(pd.date_range(window.start, window.end))
        ]
        labeled = pd.DataFrame([history_row, *future_rows])
        labeled_features = labeled.copy()
        events: list[str] = []

        def predict(_bundle: object, future: pd.DataFrame) -> pd.DataFrame:
            self.assertNotIn("sales", future.columns)
            events.append("predict")
            output = future[["id", *evaluator.BASE_KEY]].copy()
            output["forecast_sales"] = 1.0
            return output

        def naive(
            future: pd.DataFrame, _history: pd.DataFrame, _origin: pd.Timestamp
        ) -> pd.DataFrame:
            self.assertNotIn("sales", future.columns)
            events.append("predict")
            output = future[["id", *evaluator.BASE_KEY]].copy()
            output["forecast_sales"] = 2.0
            return output

        original_attach = evaluator.attach_actual_context

        def attach(*args: object, **kwargs: object) -> pd.DataFrame:
            events.append("attach")
            return original_attach(*args, **kwargs)

        with (
            patch.object(evaluator, "MODEL_FEATURES", []),
            patch.object(
                evaluator,
                "add_exact_sales_lags",
                side_effect=lambda frame, _history: frame.copy(),
            ),
            patch.object(evaluator, "validate_forecast_window"),
            patch.object(evaluator, "validate_store_family_coverage"),
            patch.object(evaluator, "fit_forecast_bundle", return_value={}),
            patch.object(evaluator, "predict_forecast", side_effect=predict),
            patch.object(evaluator, "make_weekly_seasonal_naive", side_effect=naive),
            patch.object(evaluator, "attach_actual_context", side_effect=attach),
        ):
            scored, metrics = evaluator.evaluate_window(
                labeled,
                labeled_features,
                pd.Timestamp("2024-01-10"),
                window,
            )

        self.assertEqual(events, ["predict", "predict", "predict", "predict", "attach", "attach"])
        self.assertEqual(len(scored), 32)
        self.assertEqual(len(metrics), 2)


class MetricAndOutputTests(unittest.TestCase):
    def test_runtime_configuration_is_strict_json_safe(self) -> None:
        value = {
            "missing": np.nan,
            "positive": np.inf,
            "negative": -np.inf,
            "integer": np.int64(3),
        }

        converted = evaluator.make_json_safe(value)

        self.assertEqual(
            converted,
            {
                "missing": "NaN",
                "positive": "Infinity",
                "negative": "-Infinity",
                "integer": 3,
            },
        )
        json.dumps(converted, allow_nan=False)

    def test_zero_actual_population_preserves_undefined_percentages(self) -> None:
        frame = pd.DataFrame(
            {"actual_sales": [0.0, 0.0], "forecast_sales": [1.0, 2.0]}
        )

        summary = evaluator.summarize_errors(frame, expected_rows=2)

        self.assertIsNone(summary["wape_pct"])
        self.assertIsNone(summary["signed_bias_pct"])
        self.assertIsNone(summary["underforecast_pct"])
        self.assertIsNone(summary["overforecast_pct"])
        self.assertEqual(summary["row_coverage_pct"], 100.0)

    def test_aggregate_rmsle_uses_equal_window_weight(self) -> None:
        scored = pd.DataFrame(
            {
                "model": ["model"] * 4,
                "actual_sales": [1.0, 1.0, 1.0, 1.0],
                "forecast_sales": [1.0, 1.0, 1.0, 1.0],
            }
        )
        window_metrics = [
            {"model": "model", "rmsle": value, "rows_expected": 1}
            for value in [1.0, 2.0, 3.0, 4.0]
        ]

        result = evaluator.aggregate_window_metrics(scored, window_metrics, "model")

        self.assertEqual(result["rmsle"], 2.5)
        self.assertEqual(result["rmsle_aggregation"], "equal_window_mean")
        self.assertEqual(result["window_count"], 4)

    def test_diagnostics_cover_every_predeclared_slice(self) -> None:
        scored = pd.DataFrame(
            {
                "model": ["model"] * 3,
                "forecast_day": [10, 2, 1],
                "store_nbr": [1, 1, 2],
                "family": ["A", "B", "A"],
                "promotion_segment": ["promotion", "no_promotion", "no_promotion"],
                "holiday_segment": ["holiday", "non_holiday", "non_holiday"],
                "sales_segment": ["actual_zero", "positive_sales", "positive_sales"],
                "actual_sales": [0.0, 2.0, 4.0],
                "forecast_sales": [1.0, 2.0, 3.0],
                "absolute_error": [1.0, 0.0, 1.0],
            }
        )

        diagnostics = evaluator.build_slice_diagnostics(scored)

        self.assertEqual(
            set(diagnostics["slice_type"]),
            {"forecast_day", "store", "family", "promotion", "holiday", "sales_activity"},
        )
        day_values = diagnostics.loc[
            diagnostics["slice_type"].eq("forecast_day"), "slice_value"
        ].tolist()
        self.assertEqual(day_values, ["1", "2", "10"])
        zero_wape = diagnostics.loc[
            diagnostics["slice_value"].eq("actual_zero"), "wape_pct"
        ].iloc[0]
        self.assertTrue(pd.isna(zero_wape))

    def test_prediction_digest_is_order_independent_and_content_sensitive(self) -> None:
        first = make_prediction_rows(
            ["2024-01-11", "2024-01-12"], values=[1.0, 2.0]
        )
        first.insert(0, "model", "model")
        first.insert(0, "window_id", "W1")
        second = first.iloc[::-1].reset_index(drop=True)

        self.assertEqual(
            evaluator.prediction_content_sha256(first),
            evaluator.prediction_content_sha256(second),
        )
        changed = first.copy()
        changed.loc[0, "forecast_sales"] = 9.0
        self.assertNotEqual(
            evaluator.prediction_content_sha256(first),
            evaluator.prediction_content_sha256(changed),
        )

    def test_output_bundle_is_atomic_identified_and_non_overwriting(self) -> None:
        predictions = make_prediction_rows(["2024-01-11"])
        diagnostics = pd.DataFrame({"slice_type": ["store"], "slice_value": ["1"]})
        metrics = {"contract_id": evaluator.CONTRACT_ID, "aggregate_metrics": []}
        manifest = {"contract_id": evaluator.CONTRACT_ID, "run_id": "test-run"}

        with tempfile.TemporaryDirectory(dir=evaluator.PROJECT_ROOT) as temporary_root:
            output_root = Path(temporary_root) / "evaluation"
            result = evaluator.write_evaluation_bundle(
                output_root,
                "test-run",
                predictions,
                metrics,
                diagnostics,
                manifest,
            )

            self.assertEqual(
                {path.name for path in result.iterdir()},
                {"predictions.csv.gz", "metrics.json", "diagnostics.csv.gz", "manifest.json"},
            )
            written_manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                set(written_manifest["outputs"]),
                {"predictions.csv.gz", "metrics.json", "diagnostics.csv.gz"},
            )
            with self.assertRaises(FileExistsError):
                evaluator.write_evaluation_bundle(
                    output_root,
                    "test-run",
                    predictions,
                    metrics,
                    diagnostics,
                    manifest,
                )

            with self.assertRaises(ValueError):
                evaluator.write_evaluation_bundle(
                    output_root,
                    "bad-run",
                    predictions,
                    {"invalid": float("nan")},
                    diagnostics,
                    {"run_id": "bad-run"},
                )
            self.assertFalse((output_root / ".bad-run.tmp").exists())
            self.assertFalse((output_root / "bad-run").exists())

    def test_run_ids_and_project_boundary_are_enforced(self) -> None:
        self.assertEqual(evaluator.validate_run_id("history-01.v1"), "history-01.v1")
        for invalid in ["", "../escape", "has space", "a" * 65]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                evaluator.validate_run_id(invalid)

        with tempfile.TemporaryDirectory() as outside:
            with self.assertRaisesRegex(ValueError, "inside the project"):
                evaluator.require_project_path(Path(outside), "Test path")


if __name__ == "__main__":
    unittest.main()
