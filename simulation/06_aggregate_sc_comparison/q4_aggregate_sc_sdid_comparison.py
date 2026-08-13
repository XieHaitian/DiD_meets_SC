#!/usr/bin/env python3
"""Aggregate conventional-SC and synthetic-DiD Monte Carlo comparison.

This self-contained local runner reproduces the three Quantile-top-4 DGPs
used in the 0802 bandwidth experiment at kappa(M)=13.070487184660164:

* ``sc_only``: conditional synthetic control holds and conditional parallel
  trends fails;
* ``pt_only``: conditional parallel trends holds and conditional synthetic
  control fails;
* ``pt_sc_both``: both conditional restrictions hold.

Each individual-level sample is collapsed to five state-by-period averages
(Alaska and four donors) before estimation.  The two aggregate estimators are
the conventional synthetic-control comparator and Algorithm 1 synthetic
difference-in-differences from Arkhangelsky et al. (2021).  There is no
bootstrap or confidence interval in this experiment.

Defaults use total pooled n in {2000, 4000}, 500 replications, master
seed 2026073001, and all locally visible CPU cores.  The first 500 replications
reproduce the underlying Monte Carlo samples used by the bandwidth runner.
Common random numbers are used across the three DGPs, and progress is printed
at every 10 percent milestone.
"""

from __future__ import annotations

import os


# Monte Carlo parallelism is across outer samples.  Keep numerical libraries
# single-threaded inside each spawned worker.
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable

import numpy as np


SCRIPT_VERSION = "q4_aggregate_sc_sdid_comparison_v1"
SOURCE_DGP_VERSION = "quantile_top4_kappa13_bandwidth_three_exact_regimes_v1"
SOURCE_DGP_SHA256 = (
    "ac6fe68a8b555c8ee359fcfdc77482f4576b2cb329a5b703902de4613e69ab76"
)
EXPERIMENT_PURPOSE = "aggregate_state_level_sc_and_sdid_comparison"

DGP_NAMES = ("sc_only", "pt_only", "pt_sc_both")
DGP_LABELS = {
    "sc_only": "Conditional SC only",
    "pt_only": "Conditional PT only",
    "pt_sc_both": "Conditional PT + SC",
}
EXPECTED_STATUS = {
    "sc_only": (False, True),
    "pt_only": (True, False),
    "pt_sc_both": (True, True),
}
METHOD_NAMES = ("conventional_sc", "synthetic_did")
METHOD_LABELS = {
    "conventional_sc": "Conventional SC",
    "synthetic_did": "Synthetic DiD",
}

STATE_NAMES = ("Alaska", "Maryland", "New Hampshire", "Utah", "Virginia")
DONOR_CODES = (24, 33, 49, 51)
DONOR_STATES = "Maryland|New Hampshire|Utah|Virginia"
N_GROUPS = 5
N_DONORS = 4
N_PRE = 5
N_PERIODS = 6
N_X = 101

KAPPA_M = 13.070487184660164
Q4_REFERENCE_GAMMA = math.sqrt(29.07742027873098)
Q4_POST_SCALE = 3.0
PT_ONLY_SC_VIOLATION = 0.40
Q4_EMPIRICAL_POST_CONTRAST = np.asarray(
    [0.211789512326912, 0.9801124738736031, 0.3509230190287762],
    dtype=float,
)
Q4_ATT_FINGERPRINT = 1.011932168518905
Q4_SC_AGGREGATE_BIAS_FINGERPRINTS = {
    "sc_only": 0.005755338140341,
    "pt_only": -0.178418587485002,
    "pt_sc_both": 0.006278642893324,
}

TREATMENT_LEVEL = 1.0
TREATMENT_AMPLITUDE = 0.7
SC_ETA_OMEGA = 1.0e-6
SDID_ETA_LAMBDA = 1.0e-6

DEFAULT_SAMPLE_SIZES = (2000, 4000)
DEFAULT_REPLICATIONS = 500
DEFAULT_MASTER_SEED = 2026073001
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "generated"
DEFAULT_TAG = "local_r1000"

ABS_TOL = 1.0e-8
REL_TOL = 1.0e-9


@dataclass(frozen=True)
class PopulationDesign:
    """One exact conditional DGP and its aggregate population panel."""

    name: str
    x_support: np.ndarray
    group_probabilities: np.ndarray
    donor_means: np.ndarray
    treated_untreated_means: np.ndarray
    treatment_effect: np.ndarray
    residual_variances: np.ndarray
    true_weights: np.ndarray
    att_true: float
    observed_mean_table: np.ndarray
    population_state_shares: np.ndarray
    population_aggregate_panel: np.ndarray
    pt_holds_conditionally: bool
    sc_holds_conditionally: bool
    maximum_conditional_pt_gap: float
    maximum_conditional_sc_residual: float


@dataclass(frozen=True)
class EstimateResult:
    """One aggregate estimator and its fitted weights."""

    estimate: float
    noise_level: float
    donor_weights: np.ndarray
    time_weights: np.ndarray
    unit_intercept: float
    time_intercept: float
    unit_objective: float
    time_objective: float
    unit_fit_rmse: float
    time_fit_rmse: float
    donor_active_count: int
    time_active_count: int


_WORKER_DESIGNS: tuple[PopulationDesign, ...] = ()


def available_cpu_count() -> int:
    """Return the CPU count visible to the current process."""

    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def replication_seed(master_seed: int, n: int, replication: int) -> int:
    """Return a seed stable to worker count and task ordering."""

    sequence = np.random.SeedSequence(
        [int(master_seed), int(n), int(replication)]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def null_space_rows(vector: np.ndarray) -> np.ndarray:
    """Return deterministic orthonormal rows perpendicular to a vector."""

    _, _, right_vectors = np.linalg.svd(
        vector.reshape(1, -1),
        full_matrices=True,
    )
    return right_vectors[1:, :]


def rank_completion_paths(
    factor_paths: np.ndarray,
    fixed_effect_contrasts: np.ndarray,
) -> np.ndarray:
    """Recreate the strong-scale Quantile-top-4 completion paths."""

    empirical_span = np.column_stack(
        [np.ones(N_PRE), factor_paths[:N_PRE]]
    )
    _, _, right_vectors = np.linalg.svd(
        empirical_span.T,
        full_matrices=True,
    )
    h_matrix = right_vectors.T[:, 3:5]
    if h_matrix.shape != (N_PRE, N_DONORS - 2):
        raise AssertionError("unexpected rank-completion dimension")
    c_matrix = null_space_rows(fixed_effect_contrasts)
    completion_magnitude = float(
        Q4_REFERENCE_GAMMA
        * math.sqrt(N_PRE)
        * np.linalg.norm(fixed_effect_contrasts)
    )
    donor_loadings = np.zeros((N_DONORS - 2, N_DONORS), dtype=float)
    donor_loadings[:, : N_DONORS - 1] = completion_magnitude * c_matrix
    factor_with_post = np.vstack([h_matrix, h_matrix[-1]])
    return factor_with_post @ donor_loadings


def smooth_left_null_directions(matrices: np.ndarray) -> np.ndarray:
    """Recreate the audited smooth left-null direction for the PT-only DGP."""

    candidate_anchors = np.asarray(
        [
            [1.0, -1.0, 1.0, -1.0, 1.0],
            [1.0, 0.0, -1.0, 0.0, 1.0],
            [0.0, 1.0, -1.0, 1.0, -1.0],
            [1.0, -2.0, 0.0, 2.0, -1.0],
        ],
        dtype=float,
    )
    projectors = np.empty((N_X, N_PRE, N_PRE), dtype=float)
    for index, matrix_m in enumerate(matrices):
        left_vectors = np.linalg.svd(
            matrix_m,
            full_matrices=False,
        )[0]
        projectors[index] = np.eye(N_PRE) - left_vectors @ left_vectors.T
    minimum_norms = np.asarray(
        [
            min(
                np.linalg.norm(projectors[index] @ anchor)
                for index in range(N_X)
            )
            for anchor in candidate_anchors
        ],
        dtype=float,
    )
    anchor = candidate_anchors[int(np.argmax(minimum_norms))]
    if float(np.max(minimum_norms)) <= 1.0e-8:
        raise AssertionError("no uniformly separated left-null anchor exists")
    directions = np.empty((N_X, N_PRE), dtype=float)
    for index, matrix_m in enumerate(matrices):
        projected = projectors[index] @ anchor
        directions[index] = projected / np.linalg.norm(projected)
        if np.linalg.norm(matrix_m.T @ directions[index]) > 1.0e-9:
            raise AssertionError("constructed direction is not left-null")
    return directions


def build_q4_common_population() -> dict[str, np.ndarray | float]:
    """Construct the shared Quantile-top-4 calibrated population arrays."""

    x_support = np.linspace(0.0, 1.0, N_X)
    group_logit_slopes = np.asarray(
        [
            0.0,
            0.19078507925363716,
            0.05314469516678279,
            -0.26765206397071295,
            0.4328933796158437,
        ],
        dtype=float,
    )
    logits = x_support[:, None] * group_logit_slopes[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    group_probabilities = np.exp(logits)
    group_probabilities /= group_probabilities.sum(axis=1, keepdims=True)

    donor_fixed_effects = np.asarray(
        [
            5.455770518036821,
            4.995424122579652,
            3.884004326884359,
            4.915659742119369,
        ],
        dtype=float,
    )
    time_effects = np.asarray(
        [
            4.3864265322446405,
            4.663045884584056,
            4.696736030129637,
            4.708472573779675,
            4.644734865535675,
            4.5418036135102415,
        ],
        dtype=float,
    )
    factor_paths = np.asarray(
        [
            [0.12199374500206683, -0.1265403779395305],
            [-0.0016213308292879907, -0.1975759032037497],
            [0.033187904499223174, 0.05054699062146668],
            [-1.2304141471753696, -1.534591893026852],
            [1.9322196082404353, -0.1309554039674733],
            [-0.8582667839451474, 1.889517054961726],
        ],
        dtype=float,
    )
    residual_variances = np.asarray(
        [
            2.5547584684639357,
            2.865039361132424,
            3.209739055360748,
            0.13563632812100845,
            0.16505603642973157,
            0.06763232260723523,
        ],
        dtype=float,
    )
    loading_coefficients = np.asarray(
        [
            [
                [
                    0.05504623773292663,
                    -0.1933574832800169,
                    0.14155692214292365,
                ],
                [
                    0.12958527990094934,
                    -0.4341695502123225,
                    0.31310468234986366,
                ],
            ],
            [
                [
                    -0.04569433625436106,
                    0.14013829339801145,
                    -0.10104604900514517,
                ],
                [
                    0.1598637449813732,
                    -0.446799828059552,
                    0.2945602418165082,
                ],
            ],
            [
                [
                    0.01898038602236328,
                    -0.11574333888296137,
                    0.11063163604102708,
                ],
                [
                    0.058767871781695465,
                    -0.08364617571678026,
                    0.0022284743197091717,
                ],
            ],
            [
                [
                    -0.07683253088475803,
                    0.2852992768248282,
                    -0.23724358309126015,
                ],
                [
                    0.11429532661261399,
                    -0.32459744735242685,
                    0.23522847464847624,
                ],
            ],
        ],
        dtype=float,
    )
    x_polynomial = np.column_stack(
        [np.ones(N_X), x_support, x_support * x_support]
    )
    donor_factor_loadings = np.einsum(
        "xp,dfp->xdf",
        x_polynomial,
        loading_coefficients,
    )
    fixed_effect_contrasts = (
        donor_fixed_effects[: N_DONORS - 1] - donor_fixed_effects[-1]
    )
    completion = rank_completion_paths(
        factor_paths,
        fixed_effect_contrasts,
    )
    reference_donor_means = (
        donor_fixed_effects[None, None, :]
        + time_effects[None, :, None]
        + np.einsum("tf,xdf->xtd", factor_paths, donor_factor_loadings)
        + completion[None, :, :]
    )

    donor_means = reference_donor_means.copy()
    pre_matrices = np.empty(
        (N_X, N_PRE, N_DONORS - 1),
        dtype=float,
    )
    for x_index in range(N_X):
        baseline = reference_donor_means[x_index, :, -1]
        reference_m = (
            reference_donor_means[x_index, :N_PRE, : N_DONORS - 1]
            - baseline[:N_PRE, None]
        )
        left_vectors, singular_values, right_vectors = np.linalg.svd(
            reference_m,
            full_matrices=False,
        )
        target_minimum = singular_values[0] / KAPPA_M
        adjusted_values = np.maximum(singular_values, target_minimum)
        adjusted_values[-1] = target_minimum
        target_m = (
            left_vectors @ np.diag(adjusted_values) @ right_vectors
        )
        pre_matrices[x_index] = target_m
        donor_means[x_index, :N_PRE, : N_DONORS - 1] = (
            baseline[:N_PRE, None] + target_m
        )
        post_contrast = (
            target_m[-1] + Q4_POST_SCALE * Q4_EMPIRICAL_POST_CONTRAST
        )
        donor_means[x_index, -1, : N_DONORS - 1] = (
            baseline[-1] + post_contrast
        )

    donor_indices = np.arange(1, N_DONORS + 1, dtype=float)
    weight_logits = x_support[:, None] * (
        donor_indices[None, :] - N_DONORS
    )
    weight_logits -= weight_logits.max(axis=1, keepdims=True)
    softmax_weights = np.exp(weight_logits)
    softmax_weights /= softmax_weights.sum(axis=1, keepdims=True)
    true_weights = 0.05 + 0.8 * softmax_weights
    treatment_effect = TREATMENT_LEVEL + TREATMENT_AMPLITUDE * np.sin(
        2.0 * np.pi * x_support
    )
    treated_probability = group_probabilities[:, 0]
    att_true = float(
        np.sum(treated_probability * treatment_effect)
        / np.sum(treated_probability)
    )
    return {
        "x_support": x_support,
        "group_probabilities": group_probabilities,
        "donor_means": donor_means,
        "pre_matrices": pre_matrices,
        "true_weights": true_weights,
        "treatment_effect": treatment_effect,
        "residual_variances": residual_variances,
        "att_true": att_true,
    }


def conditional_diagnostics(
    donor_means: np.ndarray,
    treated_means: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Return maximum PT gap, SC residual, and realized condition numbers."""

    maximum_pt_gap = 0.0
    maximum_sc_residual = 0.0
    condition_numbers = np.empty(N_X, dtype=float)
    for x_index in range(N_X):
        donor_path = donor_means[x_index]
        donor_pre = donor_path[:N_PRE]
        matrix_m = (
            donor_pre[:, : N_DONORS - 1] - donor_pre[:, [-1]]
        )
        target = treated_means[x_index, :N_PRE] - donor_pre[:, -1]
        free_weights = np.linalg.solve(
            matrix_m.T @ matrix_m,
            matrix_m.T @ target,
        )
        affine_weights = np.r_[
            free_weights,
            1.0 - free_weights.sum(),
        ]
        full_residual = (
            treated_means[x_index] - donor_path @ affine_weights
        )
        donor_changes = donor_path[-1] - donor_path[-2]
        treated_change = (
            treated_means[x_index, -1]
            - treated_means[x_index, -2]
        )
        maximum_pt_gap = max(
            maximum_pt_gap,
            float(np.max(np.abs(treated_change - donor_changes))),
        )
        maximum_sc_residual = max(
            maximum_sc_residual,
            float(np.max(np.abs(full_residual))),
            float(
                np.linalg.norm(
                    target - matrix_m @ free_weights
                )
            ),
        )
        singular_values = np.linalg.svd(
            matrix_m,
            compute_uv=False,
        )
        condition_numbers[x_index] = (
            singular_values[0] / singular_values[-1]
        )
    return maximum_pt_gap, maximum_sc_residual, condition_numbers


def population_aggregate_panel(
    group_probabilities: np.ndarray,
    observed_mean_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return group shares and the exact noise-free state-average panel."""

    joint_probabilities = group_probabilities / N_X
    state_shares = joint_probabilities.sum(axis=0)
    panel = np.empty((N_GROUPS, N_PERIODS), dtype=float)
    for group_index in range(N_GROUPS):
        conditional_x_weights = (
            joint_probabilities[:, group_index] / state_shares[group_index]
        )
        panel[group_index] = (
            conditional_x_weights @ observed_mean_table[:, group_index]
        )
    return state_shares, panel


def build_designs() -> tuple[PopulationDesign, ...]:
    """Build and audit all three exact conditional regimes."""

    common = build_q4_common_population()
    x_support = np.asarray(common["x_support"])
    group_probabilities = np.asarray(common["group_probabilities"])
    base_donor_means = np.asarray(common["donor_means"])
    pre_matrices = np.asarray(common["pre_matrices"])
    true_weights = np.asarray(common["true_weights"])
    treatment_effect = np.asarray(common["treatment_effect"])
    residual_variances = np.asarray(common["residual_variances"])
    att_true = float(common["att_true"])

    base_treated_means = np.einsum(
        "xd,xtd->xt",
        true_weights,
        base_donor_means,
    )
    reference_change = (
        base_donor_means[:, -1, -1]
        - base_donor_means[:, -2, -1]
    )
    left_null = smooth_left_null_directions(pre_matrices)

    designs: list[PopulationDesign] = []
    for name in DGP_NAMES:
        if name == "sc_only":
            donor_means = base_donor_means.copy()
            treated_means = base_treated_means.copy()
        else:
            donor_means = base_donor_means.copy()
            donor_means[:, -1] = (
                donor_means[:, -2] + reference_change[:, None]
            )
            synthetic_path = np.einsum(
                "xd,xtd->xt",
                true_weights,
                donor_means,
            )
            if name == "pt_sc_both":
                treated_means = synthetic_path
            else:
                treated_means = synthetic_path.copy()
                treated_means[:, :N_PRE] += (
                    PT_ONLY_SC_VIOLATION * left_null
                )
                treated_means[:, -1] += (
                    PT_ONLY_SC_VIOLATION * left_null[:, -1]
                )

        observed_mean_table = np.empty(
            (N_X, N_GROUPS, N_PERIODS),
            dtype=float,
        )
        observed_mean_table[:, 0] = treated_means
        observed_mean_table[:, 0, -1] += treatment_effect
        for donor_index in range(N_DONORS):
            observed_mean_table[:, donor_index + 1] = (
                donor_means[:, :, donor_index]
            )
        state_shares, aggregate_panel = population_aggregate_panel(
            group_probabilities,
            observed_mean_table,
        )
        pt_gap, sc_residual, condition_numbers = conditional_diagnostics(
            donor_means,
            treated_means,
        )
        realized_status = (
            pt_gap <= ABS_TOL,
            sc_residual <= ABS_TOL,
        )
        if realized_status != EXPECTED_STATUS[name]:
            raise AssertionError(
                f"{name}: conditional status {realized_status} does not "
                f"equal {EXPECTED_STATUS[name]}"
            )
        if not np.allclose(
            condition_numbers,
            KAPPA_M,
            rtol=REL_TOL,
            atol=ABS_TOL,
        ):
            raise AssertionError(f"{name}: condition-number audit failed")
        designs.append(
            PopulationDesign(
                name=name,
                x_support=x_support.copy(),
                group_probabilities=group_probabilities.copy(),
                donor_means=donor_means,
                treated_untreated_means=treated_means,
                treatment_effect=treatment_effect.copy(),
                residual_variances=residual_variances.copy(),
                true_weights=true_weights.copy(),
                att_true=att_true,
                observed_mean_table=observed_mean_table,
                population_state_shares=state_shares,
                population_aggregate_panel=aggregate_panel,
                pt_holds_conditionally=realized_status[0],
                sc_holds_conditionally=realized_status[1],
                maximum_conditional_pt_gap=pt_gap,
                maximum_conditional_sc_residual=sc_residual,
            )
        )

    if not math.isclose(
        att_true,
        Q4_ATT_FINGERPRINT,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
    ):
        raise AssertionError("ATT fingerprint changed")
    return tuple(designs)


def simplex_penalized_least_squares(
    matrix: np.ndarray,
    target: np.ndarray,
    zeta: float,
    intercept: bool,
) -> tuple[np.ndarray, float, float, int]:
    """Exactly solve a tiny penalized least-squares problem on a simplex.

    The objective is mean squared residual plus ``zeta**2 * ||weights||^2``.
    All nonempty active sets are enumerated (at most 31 here).  An SVD solve
    makes the infinitesimal-ridge time problem stable in its structural null
    direction.
    """

    matrix = np.asarray(matrix, dtype=float)
    target = np.asarray(target, dtype=float)
    if matrix.ndim != 2 or target.shape != (matrix.shape[0],):
        raise ValueError("matrix and target dimensions do not conform")
    if matrix.shape[1] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("simplex design matrix is invalid")
    if not np.all(np.isfinite(target)) or not np.isfinite(zeta) or zeta < 0.0:
        raise ValueError("simplex target or ridge is invalid")

    if intercept:
        centered_matrix = matrix - matrix.mean(axis=0, keepdims=True)
        centered_target = target - target.mean()
    else:
        centered_matrix = matrix
        centered_target = target
    row_count, column_count = centered_matrix.shape
    best_weights: np.ndarray | None = None
    best_objective = math.inf
    feasibility_tolerance = 1.0e-9

    for active_count in range(1, column_count + 1):
        for active_tuple in combinations(range(column_count), active_count):
            active = np.asarray(active_tuple, dtype=int)
            active_matrix = centered_matrix[:, active]
            base_weights = np.full(active_count, 1.0 / active_count)
            if active_count == 1:
                active_weights = base_weights
            else:
                _, _, right_vectors = np.linalg.svd(
                    np.ones((1, active_count)),
                    full_matrices=True,
                )
                tangent_basis = right_vectors[1:].T
                residual_target = (
                    centered_target - active_matrix @ base_weights
                )
                tangent_matrix = active_matrix @ tangent_basis
                left_vectors, singular_values, tangent_right = np.linalg.svd(
                    tangent_matrix,
                    full_matrices=False,
                )
                if singular_values.size:
                    shrinkage = singular_values / (
                        singular_values * singular_values
                        + row_count * zeta * zeta
                    )
                    tangent_coefficients = (
                        tangent_right.T
                        @ (shrinkage * (left_vectors.T @ residual_target))
                    )
                else:
                    tangent_coefficients = np.zeros(active_count - 1)
                active_weights = (
                    base_weights + tangent_basis @ tangent_coefficients
                )
            if float(np.min(active_weights)) < -feasibility_tolerance:
                continue
            active_weights = np.maximum(active_weights, 0.0)
            active_weights /= active_weights.sum()
            weights = np.zeros(column_count, dtype=float)
            weights[active] = active_weights
            residual = centered_matrix @ weights - centered_target
            objective = float(
                np.mean(residual * residual)
                + zeta * zeta * (weights @ weights)
            )
            if objective < best_objective - 1.0e-14:
                best_objective = objective
                best_weights = weights

    if best_weights is None:
        raise FloatingPointError("simplex solver found no feasible candidate")
    fitted_intercept = (
        float(np.mean(target - matrix @ best_weights))
        if intercept
        else 0.0
    )
    if (
        float(np.min(best_weights)) < -1.0e-10
        or not math.isclose(float(best_weights.sum()), 1.0, abs_tol=1.0e-10)
    ):
        raise AssertionError("invalid simplex solution")
    return (
        best_weights,
        best_objective,
        fitted_intercept,
        int(np.sum(best_weights > 1.0e-10)),
    )


def aggregate_noise_level(panel: np.ndarray) -> float:
    """Match the official synthdid SD of donor pre-period differences."""

    controls_pre = panel[1:, :N_PRE]
    differences = np.diff(controls_pre, axis=1).reshape(-1)
    noise_level = float(np.std(differences, ddof=1))
    if not np.isfinite(noise_level) or noise_level <= 0.0:
        raise FloatingPointError("aggregate noise level is not positive")
    return noise_level


def estimate_conventional_sc(panel: np.ndarray) -> EstimateResult:
    """Estimate conventional trajectory synthetic control on state means."""

    controls_pre = panel[1:, :N_PRE]
    controls_post = panel[1:, -1]
    treated_pre = panel[0, :N_PRE]
    treated_post = float(panel[0, -1])
    noise_level = aggregate_noise_level(panel)
    donor_weights, objective, _, active_count = (
        simplex_penalized_least_squares(
            controls_pre.T,
            treated_pre,
            SC_ETA_OMEGA * noise_level,
            intercept=False,
        )
    )
    estimate = float(treated_post - controls_post @ donor_weights)
    residual = controls_pre.T @ donor_weights - treated_pre
    return EstimateResult(
        estimate=estimate,
        noise_level=noise_level,
        donor_weights=donor_weights,
        time_weights=np.zeros(N_PRE, dtype=float),
        unit_intercept=0.0,
        time_intercept=0.0,
        unit_objective=objective,
        time_objective=0.0,
        unit_fit_rmse=float(np.sqrt(np.mean(residual * residual))),
        time_fit_rmse=np.nan,
        donor_active_count=active_count,
        time_active_count=0,
    )


def estimate_synthetic_did(panel: np.ndarray) -> EstimateResult:
    """Estimate Algorithm 1 synthetic difference-in-differences."""

    controls_pre = panel[1:, :N_PRE]
    controls_post = panel[1:, -1]
    treated_pre = panel[0, :N_PRE]
    treated_post = float(panel[0, -1])
    noise_level = aggregate_noise_level(panel)
    eta_omega = ((N_GROUPS - N_DONORS) * (N_PERIODS - N_PRE)) ** 0.25
    donor_weights, unit_objective, unit_intercept, donor_active = (
        simplex_penalized_least_squares(
            controls_pre.T,
            treated_pre,
            eta_omega * noise_level,
            intercept=True,
        )
    )
    time_weights, time_objective, time_intercept, time_active = (
        simplex_penalized_least_squares(
            controls_pre,
            controls_post,
            SDID_ETA_LAMBDA * noise_level,
            intercept=True,
        )
    )
    estimate = float(
        (treated_post - treated_pre @ time_weights)
        - donor_weights
        @ (controls_post - controls_pre @ time_weights)
    )
    matrix_form = float(
        np.r_[-donor_weights, 1.0]
        @ np.vstack([panel[1:], panel[0]])
        @ np.r_[-time_weights, 1.0]
    )
    if not math.isclose(estimate, matrix_form, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise AssertionError("synthetic-DiD contrast formulas disagree")
    unit_residual = (
        unit_intercept + controls_pre.T @ donor_weights - treated_pre
    )
    time_residual = (
        time_intercept + controls_pre @ time_weights - controls_post
    )
    return EstimateResult(
        estimate=estimate,
        noise_level=noise_level,
        donor_weights=donor_weights,
        time_weights=time_weights,
        unit_intercept=unit_intercept,
        time_intercept=time_intercept,
        unit_objective=unit_objective,
        time_objective=time_objective,
        unit_fit_rmse=float(np.sqrt(np.mean(unit_residual * unit_residual))),
        time_fit_rmse=float(np.sqrt(np.mean(time_residual * time_residual))),
        donor_active_count=donor_active,
        time_active_count=time_active,
    )


def estimate_method(panel: np.ndarray, method: str) -> EstimateResult:
    """Dispatch one aggregate estimator."""

    if method == "conventional_sc":
        return estimate_conventional_sc(panel)
    if method == "synthetic_did":
        return estimate_synthetic_did(panel)
    raise ValueError(f"unknown method {method!r}")


def population_method_rows(
    designs: tuple[PopulationDesign, ...],
) -> list[dict[str, float | int | str | bool]]:
    """Return noise-free aggregate pseudo-true estimates and weights."""

    rows: list[dict[str, float | int | str | bool]] = []
    for design in designs:
        for method in METHOD_NAMES:
            result = estimate_method(
                design.population_aggregate_panel,
                method,
            )
            row: dict[str, float | int | str | bool] = {
                "script_version": SCRIPT_VERSION,
                "source_dgp_version": SOURCE_DGP_VERSION,
                "dgp_name": design.name,
                "dgp_label": DGP_LABELS[design.name],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "att_true": design.att_true,
                "population_aggregate_estimate": result.estimate,
                "population_aggregate_bias": result.estimate - design.att_true,
                "noise_level": result.noise_level,
                "unit_intercept": result.unit_intercept,
                "time_intercept": result.time_intercept,
                "unit_objective": result.unit_objective,
                "time_objective": result.time_objective,
                "unit_fit_rmse": result.unit_fit_rmse,
                "time_fit_rmse": result.time_fit_rmse,
                "donor_active_count": result.donor_active_count,
                "time_active_count": result.time_active_count,
                "pt_holds_conditionally": design.pt_holds_conditionally,
                "sc_holds_conditionally": design.sc_holds_conditionally,
                "maximum_conditional_pt_gap": (
                    design.maximum_conditional_pt_gap
                ),
                "maximum_conditional_sc_residual": (
                    design.maximum_conditional_sc_residual
                ),
            }
            for index, state in enumerate(STATE_NAMES):
                key = state.lower().replace(" ", "_")
                row[f"population_share_{key}"] = float(
                    design.population_state_shares[index]
                )
            for index, state in enumerate(STATE_NAMES[1:]):
                key = state.lower().replace(" ", "_")
                row[f"donor_weight_{key}"] = float(
                    result.donor_weights[index]
                )
            for period in range(N_PRE):
                row[f"time_weight_pre_{period + 1}"] = float(
                    result.time_weights[period]
                )
            rows.append(row)

    for row in rows:
        if row["method"] != "conventional_sc":
            continue
        fingerprint = Q4_SC_AGGREGATE_BIAS_FINGERPRINTS[
            str(row["dgp_name"])
        ]
        if not math.isclose(
            float(row["population_aggregate_bias"]),
            fingerprint,
            rel_tol=1.0e-6,
            abs_tol=1.0e-7,
        ):
            raise AssertionError(
                f"{row['dgp_name']}: aggregate SC fingerprint changed"
            )
    return rows


def initialize_worker() -> None:
    """Build immutable DGP arrays once in every spawned process."""

    global _WORKER_DESIGNS
    _WORKER_DESIGNS = build_designs()


def aggregate_sample_panel(
    design: PopulationDesign,
    x_index: np.ndarray,
    groups: np.ndarray,
    residuals: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Collapse one individual panel to equally weighted state means."""

    outcomes = design.observed_mean_table[x_index, groups] + residuals
    panel = np.empty((N_GROUPS, N_PERIODS), dtype=float)
    for group_index in range(N_GROUPS):
        group_mask = groups == group_index
        panel[group_index] = outcomes[group_mask].mean(axis=0)
    if not np.all(np.isfinite(panel)) or np.any(counts <= 0):
        raise FloatingPointError("sample contains an empty or invalid state")
    return panel


def run_replication_task(
    task: tuple[int, int, int],
) -> list[dict[str, float | int | str | bool]]:
    """Generate one common sample and evaluate all six configurations."""

    n, replication, master_seed = task
    if not _WORKER_DESIGNS:
        initialize_worker()
    base_seed = replication_seed(master_seed, n, replication)
    sequence = np.random.SeedSequence(int(base_seed))
    data_stream, residual_stream, _, _ = sequence.spawn(4)
    data_rng = np.random.default_rng(data_stream)
    residual_rng = np.random.default_rng(residual_stream)
    reference = _WORKER_DESIGNS[0]
    x_index = data_rng.integers(0, N_X, size=n)
    probabilities = reference.group_probabilities[x_index]
    uniforms = data_rng.random(n)
    cumulative = np.cumsum(probabilities, axis=1)
    groups = np.sum(
        uniforms[:, None] > cumulative,
        axis=1,
    ).astype(np.int64)
    groups = np.minimum(groups, N_GROUPS - 1)
    standard_errors = residual_rng.normal(size=(n, N_PERIODS))
    residuals = standard_errors * np.sqrt(
        reference.residual_variances
    )[None, :]
    counts = np.bincount(groups, minlength=N_GROUPS)
    if np.any(counts <= 0):
        raise FloatingPointError("a state has zero sampled individuals")
    treated_sample_att = float(
        np.mean(reference.treatment_effect[x_index[groups == 0]])
    )

    rows: list[dict[str, float | int | str | bool]] = []
    for design in _WORKER_DESIGNS:
        panel = aggregate_sample_panel(
            design,
            x_index,
            groups,
            residuals,
            counts,
        )
        for method in METHOD_NAMES:
            result = estimate_method(panel, method)
            error = result.estimate - design.att_true
            row: dict[str, float | int | str | bool] = {
                "script_version": SCRIPT_VERSION,
                "source_dgp_version": SOURCE_DGP_VERSION,
                "experiment_purpose": EXPERIMENT_PURPOSE,
                "dgp_name": design.name,
                "dgp_label": DGP_LABELS[design.name],
                "method": method,
                "method_label": METHOD_LABELS[method],
                "n": int(n),
                "replication": int(replication),
                "master_seed": int(master_seed),
                "base_seed_hex": f"0x{base_seed:016x}",
                "att_true": design.att_true,
                "sample_treated_att": treated_sample_att,
                "estimate": result.estimate,
                "error": error,
                "squared_error": error * error,
                "noise_level": result.noise_level,
                "unit_intercept": result.unit_intercept,
                "time_intercept": result.time_intercept,
                "unit_objective": result.unit_objective,
                "time_objective": result.time_objective,
                "unit_fit_rmse": result.unit_fit_rmse,
                "time_fit_rmse": result.time_fit_rmse,
                "donor_active_count": result.donor_active_count,
                "time_active_count": result.time_active_count,
                "pt_holds_conditionally": design.pt_holds_conditionally,
                "sc_holds_conditionally": design.sc_holds_conditionally,
            }
            for index, state in enumerate(STATE_NAMES):
                key = state.lower().replace(" ", "_")
                row[f"count_{key}"] = int(counts[index])
            for index, state in enumerate(STATE_NAMES[1:]):
                key = state.lower().replace(" ", "_")
                row[f"donor_weight_{key}"] = float(
                    result.donor_weights[index]
                )
            for period in range(N_PRE):
                row[f"time_weight_pre_{period + 1}"] = float(
                    result.time_weights[period]
                )
            rows.append(row)
    return rows


def summarize_rows(
    raw_rows: list[dict[str, float | int | str | bool]],
    population_rows: list[dict[str, float | int | str | bool]],
    sample_sizes: tuple[int, ...],
    requested_replications: int,
) -> list[dict[str, float | int | str]]:
    """Compute direct Monte Carlo bias, empirical SD, and MSE."""

    population_lookup = {
        (str(row["dgp_name"]), str(row["method"])): row
        for row in population_rows
    }
    summaries: list[dict[str, float | int | str]] = []
    for n in sample_sizes:
        for dgp_name in DGP_NAMES:
            for method in METHOD_NAMES:
                selected = [
                    row
                    for row in raw_rows
                    if int(row["n"]) == n
                    and row["dgp_name"] == dgp_name
                    and row["method"] == method
                ]
                estimates = np.asarray(
                    [row["estimate"] for row in selected],
                    dtype=float,
                )
                estimates = estimates[np.isfinite(estimates)]
                target = float(population_lookup[(dgp_name, method)]["att_true"])
                errors = estimates - target
                empirical_sd = (
                    float(np.std(estimates, ddof=1))
                    if estimates.size > 1
                    else np.nan
                )
                bias = float(np.mean(errors)) if errors.size else np.nan
                mse = (
                    float(np.mean(errors * errors))
                    if errors.size
                    else np.nan
                )
                identity_mse = (
                    bias * bias
                    + (estimates.size - 1) / estimates.size * empirical_sd**2
                    if estimates.size > 1
                    else np.nan
                )
                if not math.isclose(
                    mse,
                    identity_mse,
                    rel_tol=1.0e-11,
                    abs_tol=1.0e-12,
                ):
                    raise AssertionError("MSE decomposition audit failed")
                population_row = population_lookup[(dgp_name, method)]
                summaries.append(
                    {
                        "script_version": SCRIPT_VERSION,
                        "source_dgp_version": SOURCE_DGP_VERSION,
                        "dgp_name": dgp_name,
                        "dgp_label": DGP_LABELS[dgp_name],
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "n": n,
                        "requested_replications": requested_replications,
                        "valid_replications": int(estimates.size),
                        "failed_replications": int(
                            requested_replications - estimates.size
                        ),
                        "att_true": target,
                        "population_aggregate_estimate": float(
                            population_row["population_aggregate_estimate"]
                        ),
                        "population_aggregate_bias": float(
                            population_row["population_aggregate_bias"]
                        ),
                        "mean_estimate": float(np.mean(estimates)),
                        "bias": bias,
                        "sd": empirical_sd,
                        "mse": mse,
                        "rmse": float(math.sqrt(mse)),
                        "bias_mcse": float(
                            empirical_sd / math.sqrt(estimates.size)
                        ),
                    }
                )
    return summaries


def write_csv_atomic(
    path: Path,
    rows: list[dict[str, float | int | str | bool]],
) -> None:
    """Write a complete CSV through an atomic same-directory replacement."""

    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Write JSON through an atomic same-directory replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse and validate the local-run command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        nargs="+",
        type=int,
        default=list(DEFAULT_SAMPLE_SIZES),
        help="total pooled individual sample sizes",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=DEFAULT_REPLICATIONS,
        help="outer Monte Carlo replications per sample size",
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=0,
        help="worker processes; 0 uses every visible local CPU",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=DEFAULT_MASTER_SEED,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--tag", default=DEFAULT_TAG)
    args = parser.parse_args(argv)
    sample_sizes = tuple(dict.fromkeys(args.n))
    if not sample_sizes or any(n <= 0 for n in sample_sizes):
        parser.error("all sample sizes must be positive")
    if args.replications <= 1:
        parser.error("replications must exceed one")
    if args.nprocs < 0:
        parser.error("nprocs cannot be negative")
    if not args.tag or any(character.isspace() for character in args.tag):
        parser.error("tag must be nonempty and contain no spaces")
    args.n = sample_sizes
    return args


def main(argv: Iterable[str] | None = None) -> int:
    """Run the full parallel local Monte Carlo comparison."""

    args = parse_args(argv)
    start = time.perf_counter()
    designs = build_designs()
    population_rows = population_method_rows(designs)
    requested_workers = (
        available_cpu_count() if args.nprocs == 0 else args.nprocs
    )
    tasks = [
        (n, replication, args.master_seed)
        for n in args.n
        for replication in range(args.replications)
    ]
    worker_count = max(1, min(requested_workers, len(tasks)))
    print(
        f"Running {len(tasks)} outer samples ({len(tasks) * 6} estimates) "
        f"with {worker_count} worker process(es).",
        flush=True,
    )
    raw_rows: list[dict[str, float | int | str | bool]] = []
    next_milestone = 10
    context = get_context("spawn")
    chunk_size = max(1, len(tasks) // max(worker_count * 8, 1))
    with context.Pool(
        processes=worker_count,
        initializer=initialize_worker,
    ) as pool:
        iterator = pool.imap_unordered(
            run_replication_task,
            tasks,
            chunksize=chunk_size,
        )
        for completed, rows in enumerate(iterator, start=1):
            raw_rows.extend(rows)
            completion_percentage = 100.0 * completed / len(tasks)
            while next_milestone <= completion_percentage + 1.0e-12:
                elapsed = time.perf_counter() - start
                print(
                    f"{next_milestone}% complete "
                    f"({completed}/{len(tasks)} outer samples; "
                    f"{elapsed:.1f} seconds).",
                    flush=True,
                )
                next_milestone += 10

    dgp_order = {name: index for index, name in enumerate(DGP_NAMES)}
    method_order = {name: index for index, name in enumerate(METHOD_NAMES)}
    raw_rows.sort(
        key=lambda row: (
            int(row["n"]),
            int(row["replication"]),
            dgp_order[str(row["dgp_name"])],
            method_order[str(row["method"])],
        )
    )
    summaries = summarize_rows(
        raw_rows,
        population_rows,
        args.n,
        args.replications,
    )
    expected_raw_count = len(args.n) * args.replications * 6
    if len(raw_rows) != expected_raw_count:
        raise AssertionError(
            f"received {len(raw_rows)} rows, expected {expected_raw_count}"
        )
    if any(int(row["valid_replications"]) != args.replications for row in summaries):
        raise AssertionError("one or more configurations lost replications")

    output_dir = args.output_dir.resolve()
    prefix = f"q4_aggregate_sc_sdid_{args.tag}"
    raw_path = output_dir / f"{prefix}_outer.csv"
    summary_path = output_dir / f"{prefix}_summary.csv"
    population_path = output_dir / f"{prefix}_population.csv"
    metadata_path = output_dir / f"{prefix}_metadata.json"
    write_csv_atomic(raw_path, raw_rows)
    write_csv_atomic(summary_path, summaries)
    write_csv_atomic(population_path, population_rows)
    runtime = time.perf_counter() - start
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata: dict[str, object] = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": script_hash,
        "source_dgp_version": SOURCE_DGP_VERSION,
        "source_dgp_sha256": SOURCE_DGP_SHA256,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "sample_sizes": list(args.n),
        "replications": args.replications,
        "master_seed": args.master_seed,
        "workers": worker_count,
        "runtime_seconds": runtime,
        "true_att": Q4_ATT_FINGERPRINT,
        "condition_number_m": KAPPA_M,
        "state_order": list(STATE_NAMES),
        "donor_codes": list(DONOR_CODES),
        "dgp_names": list(DGP_NAMES),
        "methods": list(METHOD_NAMES),
        "n_definition": "total pooled individuals across all five states",
        "aggregation": (
            "separate unweighted individual mean within each "
            "state-period cell"
        ),
        "target": "fixed population ATT",
        "common_random_numbers": "within each n and replication across all three DGPs",
        "noise_level": "sample SD (ddof=1) of 16 donor pre-period first differences",
        "conventional_sc": {
            "donor_weights": "simplex, no intercept, all five pre outcomes",
            "eta_omega": SC_ETA_OMEGA,
            "time_weights": "zero",
        },
        "synthetic_did": {
            "donor_weights": "simplex with free intercept",
            "eta_omega": "(N_treated*T_post)^(1/4)=1",
            "time_weights": "five-period simplex with free intercept",
            "eta_lambda": SDID_ETA_LAMBDA,
        },
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "outputs": {
            "outer": str(raw_path),
            "summary": str(summary_path),
            "population": str(population_path),
            "metadata": str(metadata_path),
        },
    }
    write_json_atomic(metadata_path, metadata)

    print("\nBias, empirical SD, and MSE relative to the fixed population ATT:")
    for row in summaries:
        print(
            f"n={int(row['n']):5d}  "
            f"{str(row['dgp_name']):11s}  "
            f"{str(row['method']):15s}  "
            f"bias={float(row['bias']): .6f}  "
            f"sd={float(row['sd']):.6f}  "
            f"mse={float(row['mse']):.6f}"
        )
    print(f"\nFinished in {runtime:.1f} seconds.")
    print(f"Summary: {summary_path}")
    print(f"Raw estimates: {raw_path}")
    print(f"Population audit: {population_path}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
