"""Synthetic contracts for the saved-prediction S2 quality audit."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

import analyze_store_sales_s2 as audit
from evaluate_store_sales import EVALUATION_WINDOWS
from evaluate_store_sales_s2 import ACTIVE_MODEL_NAME, CHALLENGER_MODEL_NAME


def make_comparison_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    actuals = [0.0, 10.0, 100.0, 1000.0]
    for index, (window, actual) in enumerate(
        zip(EVALUATION_WINDOWS, actuals, strict=True), start=1
    ):
        for model, forecast in [
            (ACTIVE_MODEL_NAME, actual * 1.20 + 1.0),
            (CHALLENGER_MODEL_NAME, actual * 1.05),
        ]:
            rows.append(
                {
                    "contract_id": "source-contract",
                    "model": model,
                    "window_id": window.window_id,
                    "id": index,
                    "date": pd.Timestamp(window.scoring_start),
                    "store_nbr": 1,
                    "family": "GROCERY I",
                    "actual_sales": actual,
                    "onpromotion": index % 2,
                    "is_holiday": int(index == 2),
                    "forecast_day": 1,
                    "forecast_sales": forecast,
                }
            )
    return pd.DataFrame(rows)


class QualityMetricTests(unittest.TestCase):
    def test_sales_bands_cover_zero_through_peak(self) -> None:
        identity = pd.DataFrame(
            {
                "window_id": ["W1"] * 11,
                "id": range(11),
                "store_nbr": [1] * 11,
                "family": ["A"] * 11,
                "actual_sales": range(11),
            }
        )

        result = audit.add_actual_sales_bands(identity)

        self.assertEqual(result.loc[0, "actual_sales_band"], "actual_zero")
        self.assertEqual(result.loc[1, "actual_sales_band"], "positive_low")
        self.assertEqual(result.loc[5, "actual_sales_band"], "positive_typical")
        self.assertEqual(result.loc[9, "actual_sales_band"], "positive_high")
        self.assertEqual(result.loc[10, "actual_sales_band"], "positive_peak")

    def test_quality_summary_reports_tail_and_directional_errors(self) -> None:
        frame = pd.DataFrame(
            {
                "actual_sales": [0.0, 10.0, 20.0, 30.0],
                "forecast_sales": [1.0, 8.0, 24.0, 30.0],
            }
        )

        result = audit.summarize_quality(frame)

        self.assertEqual(result["rows"], 4)
        self.assertAlmostEqual(result["signed_bias_pct"], 5.0)
        self.assertAlmostEqual(result["underforecast_pct"], 100 / 30)
        self.assertAlmostEqual(result["overforecast_pct"], 500 / 60)
        self.assertGreater(result["p95_absolute_log_error"], 0)
        self.assertGreaterEqual(
            result["maximum_absolute_log_error"], result["p95_absolute_log_error"]
        )

    def test_comparison_uses_identical_rows_and_finds_better_challenger(self) -> None:
        predictions = audit.validate_comparison_population(
            make_comparison_predictions()
        )

        metrics = audit.build_segment_metrics(predictions)
        comparison = audit.build_segment_comparison(metrics)
        overall = comparison.loc[comparison["slice_type"].eq("overall")].iloc[0]

        self.assertTrue(overall["challenger_improves_rmsle"])
        self.assertTrue(overall["challenger_improves_wape_pct"])
        self.assertTrue(overall["challenger_improves_p95_absolute_log_error"])
        self.assertIn("store_family", set(metrics["slice_type"]))

    def test_population_mismatch_is_rejected(self) -> None:
        predictions = make_comparison_predictions()
        mask = predictions["model"].eq(CHALLENGER_MODEL_NAME)
        predictions.loc[mask, "actual_sales"] += 1

        with self.assertRaisesRegex(ValueError, "identical scoring rows"):
            audit.validate_comparison_population(predictions)


class QualityEvidenceTests(unittest.TestCase):
    def test_source_output_hash_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=audit.PROJECT_ROOT) as temporary_root:
            run_directory = Path(temporary_root)
            outputs: dict[str, dict[str, object]] = {}
            for filename in sorted(audit.SOURCE_REQUIRED_OUTPUTS):
                path = run_directory / filename
                path.write_bytes(filename.encode("utf-8"))
                outputs[filename] = {
                    "bytes": path.stat().st_size,
                    "sha256": audit.sha256_file(path),
                }
            manifest = {"outputs": outputs}

            audit.verify_recorded_outputs(run_directory, manifest)
            (run_directory / "predictions.csv.gz").write_text(
                "changed", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "differs from its manifest"):
                audit.verify_recorded_outputs(run_directory, manifest)

    def test_audit_bundle_is_atomic_hashed_and_non_overwriting(self) -> None:
        predictions = audit.validate_comparison_population(
            make_comparison_predictions()
        )
        metrics = audit.build_segment_metrics(predictions)
        comparison = audit.build_segment_comparison(metrics)
        summary = {"contract_id": audit.CONTRACT_ID, "status": "complete"}
        manifest = {"contract_id": audit.CONTRACT_ID, "run_id": "audit-test"}

        with tempfile.TemporaryDirectory(dir=audit.PROJECT_ROOT) as temporary_root:
            output_root = Path(temporary_root) / "evaluation"
            result = audit.write_quality_audit_bundle(
                output_root,
                "audit-test",
                summary,
                metrics,
                comparison,
                manifest,
            )

            self.assertEqual(
                {path.name for path in result.iterdir()},
                {
                    "manifest.json",
                    "quality_summary.json",
                    "segment_comparison.csv.gz",
                    "segment_metrics.csv.gz",
                },
            )
            written = json.loads(
                (result / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(written["outputs"]),
                {
                    "quality_summary.json",
                    "segment_comparison.csv.gz",
                    "segment_metrics.csv.gz",
                },
            )
            for filename, evidence in written["outputs"].items():
                path = result / filename
                self.assertEqual(path.stat().st_size, evidence["bytes"])
                self.assertEqual(audit.sha256_file(path), evidence["sha256"])
            with self.assertRaises(FileExistsError):
                audit.write_quality_audit_bundle(
                    output_root,
                    "audit-test",
                    summary,
                    metrics,
                    comparison,
                    manifest,
                )


if __name__ == "__main__":
    unittest.main()
