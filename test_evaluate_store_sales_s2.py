"""Synthetic contracts for the isolated S2-01 evaluator."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from evaluate_store_sales import EVALUATION_WINDOWS, SEASONAL_NAIVE_NAME
from evaluate_store_sales_s2 import (
    ACTIVE_MODEL_NAME,
    CHALLENGER_MODEL_NAME,
    FEATURE_NAME,
    MAX_FEATURE_BYTES,
    add_sales_mean_lag_16_35,
    evaluate_s2_gate,
    matrix_nbytes,
)
from store_sales_model import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    make_feature_processor,
)


def make_history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for store_nbr, offset in [(1, 0.0), (2, 100.0)]:
        for day, date in enumerate(pd.date_range("2020-01-01", periods=45), start=1):
            rows.append(
                {
                    "date": date,
                    "store_nbr": store_nbr,
                    "family": "GROCERY I",
                    "sales": float(day) + offset,
                }
            )
    return pd.DataFrame(rows)


def make_gate_evidence(
    challenger_rmsle: tuple[float, float, float, float] = (0.4, 0.4, 0.4, 0.6),
    challenger_wape: tuple[float, float, float, float] = (9.0, 9.0, 9.0, 9.0),
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]], dict[str, float]]:
    records: list[dict[str, object]] = []
    for index, window in enumerate(EVALUATION_WINDOWS):
        records.extend(
            [
                {
                    "model": ACTIVE_MODEL_NAME,
                    "window_id": window.window_id,
                    "rmsle": 0.5,
                    "wape_pct": 10.0,
                },
                {
                    "model": SEASONAL_NAIVE_NAME,
                    "window_id": window.window_id,
                    "rmsle": 0.7,
                    "wape_pct": 20.0,
                },
                {
                    "model": CHALLENGER_MODEL_NAME,
                    "window_id": window.window_id,
                    "rmsle": challenger_rmsle[index],
                    "wape_pct": challenger_wape[index],
                },
            ]
        )
    aggregates = {
        ACTIVE_MODEL_NAME: {"rmsle": 0.5, "wape_pct": 10.0},
        SEASONAL_NAIVE_NAME: {"rmsle": 0.7, "wape_pct": 20.0},
        CHALLENGER_MODEL_NAME: {
            "rmsle": float(np.mean(challenger_rmsle)),
            "wape_pct": float(np.mean(challenger_wape)),
        },
    }
    resources = {
        "fit_count": 4,
        "added_feature_count": 1,
        "maximum_feature_bytes": MAX_FEATURE_BYTES,
        "challenger_total_fit_seconds": 200.0,
        "maximum_total_fit_seconds": 300.0,
        "challenger_mean_predict_seconds": 0.2,
        "maximum_mean_predict_seconds": 0.3,
    }
    return records, aggregates, resources


class SalesMeanLagFeatureTests(unittest.TestCase):
    def test_feature_uses_exact_dates_and_matching_series(self) -> None:
        history = make_history()
        target = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-02-09"), pd.Timestamp("2020-02-09")],
                "store_nbr": [1, 2],
                "family": ["GROCERY I", "GROCERY I"],
            }
        )

        result = add_sales_mean_lag_16_35(target, history)

        self.assertAlmostEqual(float(result.loc[0, FEATURE_NAME]), 14.5)
        self.assertAlmostEqual(float(result.loc[1, FEATURE_NAME]), 114.5)
        self.assertEqual(result[FEATURE_NAME].dtype, np.dtype("float32"))

    def test_feature_does_not_trust_preexisting_lag_columns(self) -> None:
        history = make_history()
        target = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-02-09")],
                "store_nbr": [1],
                "family": ["GROCERY I"],
                "sales_lag_16": [1000.0],
            }
        )

        result = add_sales_mean_lag_16_35(target, history)

        self.assertAlmostEqual(float(result.loc[0, FEATURE_NAME]), 14.5)

    def test_feature_averages_only_available_history(self) -> None:
        history = make_history()
        history = history.loc[
            ~(
                history["store_nbr"].eq(1)
                & history["date"].eq(pd.Timestamp("2020-01-05"))
            )
        ]
        target = pd.DataFrame(
            {
                "date": [pd.Timestamp("2020-02-09")],
                "store_nbr": [1],
                "family": ["GROCERY I"],
            }
        )

        result = add_sales_mean_lag_16_35(target, history)

        self.assertAlmostEqual(float(result.loc[0, FEATURE_NAME]), sum(range(6, 25)) / 19)

    def test_feature_preserves_all_missing_window(self) -> None:
        history = make_history()
        target = pd.DataFrame(
            {
                "date": [pd.Timestamp("2021-01-01")],
                "store_nbr": [1],
                "family": ["GROCERY I"],
            }
        )

        result = add_sales_mean_lag_16_35(target, history)

        self.assertTrue(pd.isna(result.loc[0, FEATURE_NAME]))

    def test_feature_rejects_negative_history(self) -> None:
        history = make_history()
        history.loc[0, "sales"] = -1.0
        target = history.head(1).drop(columns="sales")

        with self.assertRaisesRegex(ValueError, "non-negative"):
            add_sales_mean_lag_16_35(target, history)


class S2GateTests(unittest.TestCase):
    def test_complete_improvement_passes(self) -> None:
        records, aggregates, resources = make_gate_evidence()

        gate = evaluate_s2_gate(records, aggregates, resources)

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["decision"], "eligible_for_review")
        self.assertEqual(gate["improved_rmsle_window_count"], 3)

    def test_fewer_than_3_improved_windows_retains_v2(self) -> None:
        records, aggregates, resources = make_gate_evidence(
            challenger_rmsle=(0.4, 0.4, 0.6, 0.6)
        )

        gate = evaluate_s2_gate(records, aggregates, resources)

        self.assertFalse(gate["passed"])
        self.assertFalse(gate["criteria"]["rmsle_improves_in_at_least_3_windows"])
        self.assertEqual(gate["decision"], "retain_active_v2")

    def test_worst_window_regression_retains_v2(self) -> None:
        records, aggregates, resources = make_gate_evidence(
            challenger_wape=(8.0, 8.0, 8.0, 11.0)
        )

        gate = evaluate_s2_gate(records, aggregates, resources)

        self.assertFalse(gate["passed"])
        self.assertFalse(
            gate["criteria"]["worst_window_wape_no_higher_than_active_v2"]
        )

    def test_resource_overrun_retains_v2(self) -> None:
        records, aggregates, resources = make_gate_evidence()
        resources["challenger_total_fit_seconds"] = 301.0

        gate = evaluate_s2_gate(records, aggregates, resources)

        self.assertFalse(gate["passed"])
        self.assertFalse(gate["criteria"]["fit_time_within_budget"])

    def test_incomplete_window_evidence_is_rejected(self) -> None:
        records, aggregates, resources = make_gate_evidence()
        records.pop()

        with self.assertRaisesRegex(ValueError, "incomplete"):
            evaluate_s2_gate(records, aggregates, resources)

    def test_duplicate_window_evidence_is_rejected(self) -> None:
        records, aggregates, resources = make_gate_evidence()
        records.append(dict(records[0]))

        with self.assertRaisesRegex(ValueError, "duplicated"):
            evaluate_s2_gate(records, aggregates, resources)

    def test_dense_matrix_storage_is_measured(self) -> None:
        matrix = np.zeros((2, 3), dtype=np.float32)

        self.assertEqual(matrix_nbytes(matrix), 24)


class FeatureProcessorExtensionTests(unittest.TestCase):
    def test_explicit_default_contract_matches_implicit_default(self) -> None:
        implicit = make_feature_processor()
        explicit = make_feature_processor(
            categorical_features=CATEGORICAL_FEATURES,
            numeric_features=NUMERIC_FEATURES,
        )

        implicit_contract = [
            (name, columns) for name, _step, columns in implicit.transformers
        ]
        explicit_contract = [
            (name, columns) for name, _step, columns in explicit.transformers
        ]
        self.assertEqual(implicit_contract, explicit_contract)

    def test_duplicate_custom_features_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            make_feature_processor(
                categorical_features=["family"],
                numeric_features=["family"],
            )


if __name__ == "__main__":
    unittest.main()
