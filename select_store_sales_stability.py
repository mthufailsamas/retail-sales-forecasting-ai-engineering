"""Compare completed no-feature and feature searches with one stability rule.

This command reuses immutable candidate-fold metrics. It does not retrain,
materialize a model, or change the active serving artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Callable

import numpy as np
import pandas as pd

from analyze_store_sales_s2 import verify_recorded_outputs
from evaluate_store_sales import (
    CONTRACT_ID as S1_CONTRACT_ID,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RUN_ID as DEFAULT_S1_RUN_ID,
    EVALUATION_WINDOWS,
    build_candidate_registry,
    make_json_safe,
    read_json_object,
    require_available_output_paths,
    require_project_path,
    sha256_file,
    verify_evaluation_outputs,
)
from evaluate_store_sales_s2 import (
    CONTRACT_ID as S2_CONTRACT_ID,
    DEFAULT_RUN_ID as DEFAULT_S2_RUN_ID,
    FEATURE_NAME,
)


CONTRACT_ID = "retail-stability-selection-01"
DEFAULT_RUN_ID = f"{CONTRACT_ID}-v1"
DEFAULT_S1_RUN_DIRECTORY = DEFAULT_OUTPUT_ROOT / DEFAULT_S1_RUN_ID
DEFAULT_S2_RUN_DIRECTORY = DEFAULT_OUTPUT_ROOT / DEFAULT_S2_RUN_ID
EVIDENCE_SCOPE = "retrospective development selection from frozen fold metrics"
NO_FEATURE_BRANCH = "without_s2_feature"
WITH_FEATURE_BRANCH = "with_s2_feature"
EXPECTED_CANDIDATES = 30
EXPECTED_FITS = EXPECTED_CANDIDATES * len(EVALUATION_WINDOWS)
REQUIRED_COLUMNS = {
    "contract_id",
    "window_id",
    "run_id",
    "method",
    "parameters_json",
    "rmsle",
    "wape_pct",
    "signed_bias_pct",
    "repeat_prediction_match",
}
STABILITY_ORDER = [
    "worst_fold_rmsle",
    "mean_rmsle",
    "std_rmsle",
    "worst_fold_wape_pct",
    "mean_wape_pct",
    "worst_absolute_fold_bias_pct",
    "method",
    "run_id",
]


def validate_candidate_folds(
    folds: pd.DataFrame,
    expected_contract_id: str,
) -> pd.DataFrame:
    """Require one complete, repeatable 30-candidate by 4-window search."""
    missing = sorted(REQUIRED_COLUMNS.difference(folds.columns))
    if folds.empty or missing:
        raise ValueError(f"Candidate-fold evidence is incomplete: {missing}")
    if set(folds["contract_id"]) != {expected_contract_id}:
        raise ValueError("Candidate-fold evidence uses an unexpected contract.")
    expected_windows = {window.window_id for window in EVALUATION_WINDOWS}
    if folds.duplicated(["run_id", "window_id"]).any():
        raise ValueError("Candidate-fold evidence contains duplicate run-window rows.")
    coverage = folds.groupby("run_id", observed=True)["window_id"].agg(set)
    if (
        len(folds) != EXPECTED_FITS
        or len(coverage) != EXPECTED_CANDIDATES
        or not coverage.map(lambda values: values == expected_windows).all()
    ):
        raise ValueError("Candidate-fold evidence does not contain 30 complete candidates.")
    if not folds["repeat_prediction_match"].astype(bool).all():
        raise ValueError("Candidate-fold evidence contains a repeatability failure.")
    numeric = folds[["rmsle", "wape_pct", "signed_bias_pct"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric).all() or (numeric[:, :2] < 0).any():
        raise ValueError("Candidate-fold evidence contains invalid metric values.")

    expected_registry = {row["run_id"]: row for row in build_candidate_registry()}
    if set(coverage.index) != set(expected_registry):
        raise ValueError("Candidate-fold evidence differs from the frozen registry.")
    for run_id, rows in folds.groupby("run_id", observed=True):
        identity = rows[["method", "parameters_json"]].drop_duplicates()
        if len(identity) != 1:
            raise ValueError("Candidate identity changes between folds.")
        expected = expected_registry[str(run_id)]
        current = identity.iloc[0]
        try:
            parameters = json.loads(current["parameters_json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("Candidate parameters are not valid JSON.") from error
        if current["method"] != expected["method"] or parameters != expected["parameters"]:
            raise ValueError("Candidate-fold evidence differs from the frozen registry.")
    return folds.copy()


def load_verified_candidate_folds(
    run_directory: Path,
    expected_contract_id: str,
    output_verifier: Callable[[Path, dict[str, Any]], None],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Verify an immutable evaluation bundle before reading candidate metrics."""
    run_directory = require_project_path(run_directory, "Evaluation run directory")
    if not run_directory.is_dir():
        raise FileNotFoundError(f"Evaluation run directory is missing: {run_directory}")
    manifest = read_json_object(run_directory / "manifest.json", "Evaluation manifest")
    if manifest.get("contract_id") != expected_contract_id:
        raise ValueError("Evaluation source uses an unexpected contract.")
    if manifest.get("run_id") != run_directory.name:
        raise ValueError("Evaluation source run ID is inconsistent.")
    output_verifier(run_directory, manifest)
    folds = pd.read_csv(run_directory / "candidate_fold_metrics.csv")
    return manifest, validate_candidate_folds(folds, expected_contract_id)


def rank_stable_candidates(folds: pd.DataFrame, branch: str) -> pd.DataFrame:
    """Rank complete candidates by worst-period accuracy before average accuracy."""
    grouped = folds.groupby(
        ["run_id", "method", "parameters_json"],
        as_index=False,
        observed=True,
    )
    summary = grouped.agg(
        fold_count=("window_id", "size"),
        worst_fold_rmsle=("rmsle", "max"),
        mean_rmsle=("rmsle", "mean"),
        std_rmsle=("rmsle", "std"),
        worst_fold_wape_pct=("wape_pct", "max"),
        mean_wape_pct=("wape_pct", "mean"),
    )
    worst_bias = grouped["signed_bias_pct"].apply(
        lambda values: float(values.abs().max())
    )
    summary = summary.merge(
        worst_bias.rename(columns={"signed_bias_pct": "worst_absolute_fold_bias_pct"}),
        on=["run_id", "method", "parameters_json"],
        validate="one_to_one",
    )
    if summary["fold_count"].nunique() != 1 or summary["fold_count"].iloc[0] != len(
        EVALUATION_WINDOWS
    ):
        raise ValueError("Stable ranking requires complete 4-window candidates.")
    summary.insert(0, "branch", branch)
    summary = summary.sort_values(STABILITY_ORDER, ignore_index=True)
    summary.insert(1, "stability_rank", np.arange(1, len(summary) + 1))
    return summary


def select_branch_winner(candidate_summary: pd.DataFrame) -> pd.Series:
    """Return the single first-ranked candidate from one branch."""
    winners = candidate_summary.loc[candidate_summary["stability_rank"].eq(1)]
    if len(winners) != 1:
        raise ValueError("Each feature branch must have exactly 1 stable winner.")
    return winners.iloc[0]


def select_overall_winner(branch_winners: pd.DataFrame) -> pd.Series:
    """Compare both branch winners with the same deterministic stability order."""
    if set(branch_winners["branch"]) != {NO_FEATURE_BRANCH, WITH_FEATURE_BRANCH}:
        raise ValueError("Overall selection requires both feature branches.")
    if len(branch_winners) != 2:
        raise ValueError("Overall selection requires exactly 2 branch winners.")
    return branch_winners.sort_values(
        [*STABILITY_ORDER, "branch"], ignore_index=True
    ).iloc[0]


def write_stability_bundle(
    output_root: Path,
    run_id: str,
    candidate_summary: pd.DataFrame,
    branch_winners: pd.DataFrame,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
) -> Path:
    """Publish one atomic, non-overwriting private selection bundle."""
    root, final_directory, temporary_directory = require_available_output_paths(
        output_root, run_id
    )
    root.mkdir(parents=True, exist_ok=True)
    temporary_created = False
    try:
        temporary_directory.mkdir()
        temporary_created = True
        candidate_path = temporary_directory / "candidate_stability_summary.csv"
        winners_path = temporary_directory / "branch_winners.csv"
        metrics_path = temporary_directory / "metrics.json"
        manifest_path = temporary_directory / "manifest.json"
        candidate_summary.to_csv(candidate_path, index=False, float_format="%.9g")
        branch_winners.to_csv(winners_path, index=False, float_format="%.9g")
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        outputs = [candidate_path, winners_path, metrics_path]
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


def run_stability_selection(
    s1_run_directory: Path = DEFAULT_S1_RUN_DIRECTORY,
    s2_run_directory: Path = DEFAULT_S2_RUN_DIRECTORY,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str = DEFAULT_RUN_ID,
) -> tuple[Path, dict[str, Any]]:
    """Select stable winners from both completed searches without retraining."""
    output_root = require_project_path(output_root, "Stability output root")
    s1_run_directory = require_project_path(
        s1_run_directory, "S1 evaluation run directory"
    )
    s2_run_directory = require_project_path(
        s2_run_directory, "S2 evaluation run directory"
    )
    require_available_output_paths(output_root, run_id)
    s1_manifest, s1_folds = load_verified_candidate_folds(
        s1_run_directory,
        S1_CONTRACT_ID,
        verify_evaluation_outputs,
    )
    s2_manifest, s2_folds = load_verified_candidate_folds(
        s2_run_directory,
        S2_CONTRACT_ID,
        verify_recorded_outputs,
    )
    s1_summary = rank_stable_candidates(s1_folds, NO_FEATURE_BRANCH)
    s2_summary = rank_stable_candidates(s2_folds, WITH_FEATURE_BRANCH)
    candidate_summary = pd.concat([s1_summary, s2_summary], ignore_index=True)
    branch_winners = pd.DataFrame(
        [
            select_branch_winner(s1_summary).to_dict(),
            select_branch_winner(s2_summary).to_dict(),
        ]
    )
    overall_winner = select_overall_winner(branch_winners)
    metrics = make_json_safe(
        {
            "contract_id": CONTRACT_ID,
            "run_id": run_id,
            "evidence_scope": EVIDENCE_SCOPE,
            "selection_rule": {
                "kind": "lexicographic worst-period-first",
                "ordered_fields": STABILITY_ORDER,
                "lower_is_better": True,
            },
            "branch_winners": branch_winners.to_dict("records"),
            "development_recommendation": overall_winner.to_dict(),
            "active_model_effect": "none",
        }
    )
    created_at = datetime.now(timezone.utc).isoformat()
    code_path = Path(__file__)
    manifest = make_json_safe(
        {
            "contract_id": CONTRACT_ID,
            "run_id": run_id,
            "created_at_utc": created_at,
            "evidence_scope": EVIDENCE_SCOPE,
            "sources": {
                NO_FEATURE_BRANCH: {
                    "contract_id": S1_CONTRACT_ID,
                    "run_id": s1_manifest["run_id"],
                    "manifest_sha256": sha256_file(s1_run_directory / "manifest.json"),
                    "candidate_fold_metrics_sha256": sha256_file(
                        s1_run_directory / "candidate_fold_metrics.csv"
                    ),
                },
                WITH_FEATURE_BRANCH: {
                    "contract_id": S2_CONTRACT_ID,
                    "run_id": s2_manifest["run_id"],
                    "feature": FEATURE_NAME,
                    "manifest_sha256": sha256_file(s2_run_directory / "manifest.json"),
                    "candidate_fold_metrics_sha256": sha256_file(
                        s2_run_directory / "candidate_fold_metrics.csv"
                    ),
                },
            },
            "configuration": {
                "candidate_count_per_branch": EXPECTED_CANDIDATES,
                "fold_count_per_candidate": len(EVALUATION_WINDOWS),
                "selection_order": STABILITY_ORDER,
                "active_model_effect": "none",
            },
            "code": {
                code_path.name: {
                    "bytes": code_path.stat().st_size,
                    "sha256": sha256_file(code_path),
                }
            },
        }
    )
    directory = write_stability_bundle(
        output_root,
        run_id,
        candidate_summary,
        branch_winners,
        metrics,
        manifest,
    )
    return directory, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare frozen no-feature and feature searches by stability."
    )
    parser.add_argument("--s1-run-directory", type=Path, default=DEFAULT_S1_RUN_DIRECTORY)
    parser.add_argument("--s2-run-directory", type=Path, default=DEFAULT_S2_RUN_DIRECTORY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser.parse_args()


def main() -> None:
    directory, metrics = run_stability_selection(**vars(parse_args()))
    print("Stable no-feature versus feature selection: PASS")
    for winner in metrics["branch_winners"]:
        print(
            f"{winner['branch']}: {winner['run_id']} | "
            f"worst RMSLE {winner['worst_fold_rmsle']:.6f} | "
            f"mean RMSLE {winner['mean_rmsle']:.6f}"
        )
    selected = metrics["development_recommendation"]
    print(
        "Development recommendation: "
        f"{selected['branch']} / {selected['run_id']}"
    )
    print("Active model: unchanged")
    print(f"Output: {directory}")


if __name__ == "__main__":
    main()
