#!/usr/bin/env python3
"""Compute the empirical 95th percentile of kappa(M) for the main donors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EMPIRICAL_ROOT = HERE.parent
sys.path.insert(0, str(EMPIRICAL_ROOT))

import common_empirical as ce


def subgroup_condition_numbers(
    data: pd.DataFrame,
    educ: int,
    nchild: int,
) -> list[dict[str, float | int]]:
    """Construct every fold- and age-specific baseline-differenced matrix M."""

    subgroup = ce.restrict_data(data, educ, nchild, ce.QUANTILE_TOP4)
    prep = ce.prepare_arrays(subgroup)
    y = np.asarray(prep["y_array"]).ravel()
    year = np.asarray(prep["t_array"]).ravel()
    state = np.asarray(prep["g_array"]).ravel()
    age = np.asarray(prep["x_array"])
    folds = np.asarray(prep["folds"])
    groups = np.asarray(prep["groups"])
    years = np.asarray(prep["times"])
    age_grid = np.asarray(prep["age_unique"])
    bandwidth = float(prep["h_y"])
    rows: list[dict[str, float | int]] = []

    for fold in range(ce.FOLD_COUNT):
        training = folds != fold
        y_train = y[training]
        year_train = year[training]
        state_train = state[training]
        age_train = age[training]
        unit_weights = np.ones(int(training.sum()), dtype=float)
        for evaluation_age in age_grid:
            moments = np.empty((years.size, groups.size), dtype=float)
            for group_index, group in enumerate(groups):
                group_mask = state_train == group
                for year_index, evaluation_year in enumerate(years):
                    moments[year_index, group_index] = ce.local_linear(
                        age_train[group_mask],
                        y_train[group_mask] * (year_train[group_mask] == evaluation_year),
                        float(evaluation_age),
                        bandwidth,
                        unit_weights[group_mask],
                    )
            donor_pre = moments[:-1, 1:]
            matrix = donor_pre[:, :-1] - donor_pre[:, -1, None]
            singular_values = np.linalg.svd(matrix, compute_uv=False)
            rank = int(np.linalg.matrix_rank(matrix))
            kappa = np.inf
            if rank == matrix.shape[1] and singular_values[-1] > 0.0:
                kappa = float(singular_values[0] / singular_values[-1])
            rows.append(
                {
                    "educ": educ,
                    "nchild": nchild,
                    "fold": fold,
                    "age": float(evaluation_age),
                    "rank_m": rank,
                    "kappa_m": kappa,
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "generated")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = ce.load_data()
    rows = [
        row
        for educ, nchild in ce.SUBGROUPS
        for row in subgroup_condition_numbers(data, educ, nchild)
    ]
    detail = pd.DataFrame(rows)
    finite = detail.loc[np.isfinite(detail["kappa_m"]), "kappa_m"].to_numpy(dtype=float)
    result = pd.DataFrame(
        [
            {
                "configuration": "quantile_top4_main",
                "donor_states": "51|33|24|49",
                "donors": "Virginia, New Hampshire, Maryland, Utah",
                "condition_definition": "kappa(M)=sigma_max(M)/sigma_min(M)",
                "quantile": 0.95,
                "quantile_method": "inverted_cdf",
                "kappa_m_p95": float(np.quantile(finite, 0.95, method="inverted_cdf")),
                "finite_evaluations": int(finite.size),
                "rank_deficient_evaluations": int((detail["rank_m"] < 3).sum()),
                "kappa_m_median": float(np.quantile(finite, 0.50, method="inverted_cdf")),
                "kappa_m_min": float(np.min(finite)),
                "kappa_m_max": float(np.max(finite)),
            }
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ce.save_csv_atomic(result, args.output_dir / "condition_number_result.csv")
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
