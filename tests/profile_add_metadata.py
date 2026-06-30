#!/usr/bin/env python3
"""Profile add_pandas_metadata_columns() and add_mfe_metadata_columns() on a realistic candidate matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.Metadata.mfe.Add_MFE_Metafeatures import (  # noqa: E402
    add_mfe_metadata_columns,
    enable_profiling as enable_mfe_profiling,
    get_profiling_results as get_mfe_profiling_results,
    get_profiling_wall_seconds as get_mfe_profiling_wall_seconds,
)
from src.Metadata.pandas.Add_Pandas_Metafeatures import (  # noqa: E402
    add_pandas_metadata_columns,
    enable_profiling as enable_pandas_profiling,
    get_profiling_results as get_pandas_profiling_results,
    get_profiling_wall_seconds as get_pandas_profiling_wall_seconds,
)
from src.utils.create_feature_and_featurename import (  # noqa: E402
    create_featurenames,
    extract_operation_and_original_features,
)
from src.utils.get_data import get_openml_dataset_split_and_metadata  # noqa: E402
from src.utils.get_matrix import get_matrix_core_columns  # noqa: E402

# Used in SurrogateModel_TabArena_Recursion.py and run_metadata_parallel_classification.sh
DEFAULT_DATASET_ID = 2073
DEFAULT_MODEL = "LightGBM_BAG_L1"


def build_candidate_matrix(dataset_id: int, X_train: pd.DataFrame, model: str) -> pd.DataFrame:
    """Build the same candidate structure as create_empty_core_matrix_for_dataset()."""
    columns = get_matrix_core_columns()
    comparison_result_matrix = pd.DataFrame(columns=columns)

    for feature1 in X_train.columns:
        featurename = "without - " + str(feature1)
        new_rows = pd.DataFrame(columns=columns)
        new_rows.loc[len(new_rows)] = [dataset_id, featurename, "delete", model, 0]
        comparison_result_matrix = pd.concat(
            [comparison_result_matrix, pd.DataFrame(new_rows)], ignore_index=True
        )

    featurenames = create_featurenames(X_train.columns)
    new_rows = pd.DataFrame(columns=columns)
    for featurename in featurenames:
        operator, _ = extract_operation_and_original_features(featurename)
        new_rows.loc[len(new_rows)] = [dataset_id, featurename, operator, model, 0]
    comparison_result_matrix = pd.concat(
        [comparison_result_matrix, pd.DataFrame(new_rows)], ignore_index=True
    )
    return comparison_result_matrix


def print_profiling_table(function_name: str, timers: dict[str, float], wall_seconds: float) -> None:
    total = wall_seconds if wall_seconds > 0 else sum(timers.values())
    rows = []
    for label, seconds in timers.items():
        pct = (seconds / total * 100.0) if total > 0 else 0.0
        rows.append((label, seconds, pct))

    tracked = sum(timers.values())
    untracked = max(0.0, total - tracked)
    if untracked > 1e-9:
        rows.append(("other (setup / final concat / untracked)", untracked, untracked / total * 100.0))

    rows.sort(key=lambda item: item[1], reverse=True)

    print(f"\n{'=' * 72}")
    print(f"Profiling: {function_name}")
    print(f"Wall time: {total:.3f}s | candidates loop overhead tracked: {tracked:.3f}s")
    print(f"{'=' * 72}")
    print(f"{'Poste':<42} {'Temps (s)':>12} {'% total':>10}")
    print(f"{'-' * 42} {'-' * 12} {'-' * 10}")
    for label, seconds, pct in rows:
        print(f"{label:<42} {seconds:12.3f} {pct:9.1f}%")
    print(f"{'=' * 72}\n")


def run_profile(
    label: str,
    fn,
    enable_profiling,
    get_results,
    get_wall,
    *args,
):
    enable_profiling()
    fn(*args)
    print_profiling_table(label, get_results(), get_wall())


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile metadata column enrichment functions.")
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=DEFAULT_DATASET_ID,
        help=f"OpenML task id (default: {DEFAULT_DATASET_ID})",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional cap on candidate rows (full matrix if omitted)",
    )
    parser.add_argument(
        "--skip-mfe",
        action="store_true",
        help="Skip MFE profiling (much slower than pandas)",
    )
    args = parser.parse_args()

    print(f"Loading OpenML task {args.dataset_id}...")
    X_train, y_train, _, _, dataset_metadata = get_openml_dataset_split_and_metadata(args.dataset_id)
    print(
        f"Train shape: {X_train.shape[0]} rows x {X_train.shape[1]} cols | "
        f"task_type={dataset_metadata.get('task_type', 'unknown')}"
    )

    result_matrix = build_candidate_matrix(args.dataset_id, X_train, DEFAULT_MODEL)
    if args.max_candidates is not None:
        result_matrix = result_matrix.iloc[: args.max_candidates].copy()
    print(f"Candidate matrix: {len(result_matrix)} rows")

    run_profile(
        "add_pandas_metadata_columns()",
        add_pandas_metadata_columns,
        enable_pandas_profiling,
        get_pandas_profiling_results,
        get_pandas_profiling_wall_seconds,
        dataset_metadata,
        X_train,
        result_matrix.copy(),
    )

    if not args.skip_mfe:
        run_profile(
            "add_mfe_metadata_columns()",
            add_mfe_metadata_columns,
            enable_mfe_profiling,
            get_mfe_profiling_results,
            get_mfe_profiling_wall_seconds,
            X_train,
            y_train,
            result_matrix.copy(),
        )
    else:
        print("MFE profiling skipped (--skip-mfe).")


if __name__ == "__main__":
    main()
