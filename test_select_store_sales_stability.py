"""Synthetic contracts for retrospective stability-first selection."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

import select_store_sales_stability as stability


def make_candidate_folds() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidates = {
        "lower_mean_unstable": [0.10, 0.10, 0.10, 0.90],
        "stable": [0.31, 0.32, 0.33, 0.34],
    }
    for run_id, values in candidates.items():
        for index, rmsle in enumerate(values, start=1):
            rows.append(
                {
                    "run_id": run_id,
                    "method": "XGBoost Regression",
                    "parameters_json": json.dumps({"candidate": run_id}),
                    "window_id": f"W{index}",
                    "rmsle": rmsle,
                    "wape_pct": rmsle * 10,
                    "signed_bias_pct": (-1) ** index * rmsle,
                }
            )
    return pd.DataFrame(rows)


class StableSelectionTests(unittest.TestCase):
    def test_worst_period_precedes_lower_mean(self) -> None:
        summary = stability.rank_stable_candidates(
            make_candidate_folds(), stability.NO_FEATURE_BRANCH
        )

        winner = stability.select_branch_winner(summary)

        self.assertEqual(winner["run_id"], "stable")
        self.assertLess(winner["worst_fold_rmsle"], 0.90)

    def test_both_branches_are_required_for_overall_selection(self) -> None:
        summary = stability.rank_stable_candidates(
            make_candidate_folds(), stability.NO_FEATURE_BRANCH
        )
        winner = stability.select_branch_winner(summary)

        with self.assertRaisesRegex(ValueError, "both feature branches"):
            stability.select_overall_winner(pd.DataFrame([winner.to_dict()]))

    def test_same_rule_compares_both_branch_winners(self) -> None:
        no_feature = stability.rank_stable_candidates(
            make_candidate_folds(), stability.NO_FEATURE_BRANCH
        )
        feature_folds = make_candidate_folds()
        feature_folds["rmsle"] -= 0.05
        with_feature = stability.rank_stable_candidates(
            feature_folds, stability.WITH_FEATURE_BRANCH
        )
        winners = pd.DataFrame(
            [
                stability.select_branch_winner(no_feature).to_dict(),
                stability.select_branch_winner(with_feature).to_dict(),
            ]
        )

        selected = stability.select_overall_winner(winners)

        self.assertEqual(selected["branch"], stability.WITH_FEATURE_BRANCH)


class StabilityBundleTests(unittest.TestCase):
    def test_bundle_is_atomic_hashed_and_non_overwriting(self) -> None:
        no_feature = stability.rank_stable_candidates(
            make_candidate_folds(), stability.NO_FEATURE_BRANCH
        )
        with_feature = stability.rank_stable_candidates(
            make_candidate_folds(), stability.WITH_FEATURE_BRANCH
        )
        candidates = pd.concat([no_feature, with_feature], ignore_index=True)
        winners = pd.DataFrame(
            [
                stability.select_branch_winner(no_feature).to_dict(),
                stability.select_branch_winner(with_feature).to_dict(),
            ]
        )
        metrics = {"contract_id": stability.CONTRACT_ID}
        manifest = {"contract_id": stability.CONTRACT_ID, "run_id": "stable-test"}

        with tempfile.TemporaryDirectory(
            dir=stability.DEFAULT_OUTPUT_ROOT.parent.parent
        ) as temporary_root:
            output_root = Path(temporary_root) / "evaluation"
            result = stability.write_stability_bundle(
                output_root,
                "stable-test",
                candidates,
                winners,
                metrics,
                manifest,
            )

            written = json.loads(
                (result / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(written["outputs"]),
                {"branch_winners.csv", "candidate_stability_summary.csv", "metrics.json"},
            )
            for filename, evidence in written["outputs"].items():
                path = result / filename
                self.assertEqual(path.stat().st_size, evidence["bytes"])
                self.assertEqual(stability.sha256_file(path), evidence["sha256"])
            with self.assertRaises(FileExistsError):
                stability.write_stability_bundle(
                    output_root,
                    "stable-test",
                    candidates,
                    winners,
                    metrics,
                    manifest,
                )


if __name__ == "__main__":
    unittest.main()
