#!/usr/bin/env python3
"""Self-contained HPC runner for the Quantile-top-4 SC-only experiment.

The donor order is Maryland, New Hampshire, Utah, and Virginia, with Virginia
as the affine baseline.  There are five pre-treatment periods, one post period,
exact synthetic control, heterogeneous donor trends (so PT fails), and full
population column rank.  At every covariate value the SVD path preserves the
reference singular vectors and largest singular value, sets
``sigma_min = sigma_max / kappa_m``, and keeps the calibrated donor
post-minus-last-pre contrasts fixed.  At kappa=1 the middle singular value is
also lifted to preserve singular-value ordering; on the weak side of the grid
only the smallest singular value changes.

The benchmark ``13.070487184660164`` is the unweighted inverted-CDF p95 of the
direct condition number of the empirical 5x3 matrix
``[Maryland-Virginia, New Hampshire-Virginia, Utah-Virginia]``.  The condition
number of its Gram matrix is the square, ``170.8376352443656``.

For every (n, kappa_m, outer replication) cell the program computes the
fixed-two-fold DiD-SC estimate and B independent exponential-multiplier
re-estimates of every nuisance function and donor weight.  Folds stay fixed
within an outer dataset, and common random numbers are shared across the
kappa grid.  Compact bootstrap diagnostics are written instead of individual
roots.  Defaults are n=2,000 and n=4,000, R=500, B=500, and every CPU visible
to the allocation.
"""

from __future__ import annotations

import os


_cache_root = (
    os.environ.get("SLURM_TMPDIR")
    or os.environ.get("TMPDIR")
    or os.environ.get("TMP")
    or "."
)
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(_cache_root, "q4_sc_only_numba_cache"),
)
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)


# Parallelism is across outer Monte Carlo samples.  Numerical libraries and
# Numba must stay single-threaded inside every spawned process.
for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_variable] = "1"

import argparse
import csv
import hashlib
import json
import math
import time
import warnings
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from statistics import NormalDist

import numpy as np
from numba import NumbaPerformanceWarning, config as numba_config, njit


numba_config.DISABLE_PERFORMANCE_WARNINGS = 1
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)


DGP_VERSION = "quantile_top4_sc_only_lower_smin_fixed_folds_v1"
DONOR_CODES = (24, 33, 49, 51)
DONOR_STATES = "Maryland|New Hampshire|Utah|Virginia"
BASELINE_DONOR_CODE = 51
BASELINE_DONOR_STATE = "Virginia"
CONDITION_DEFINITION = "kappa_2(M)=sigma_max(M)/sigma_min(M)"
CALIBRATION_TARGET_KAPPA_M = 13.070487184660164
CALIBRATION_TARGET_KAPPA_MTM = CALIBRATION_TARGET_KAPPA_M**2
CALIBRATION_TARGET_SOURCE = (
    "unweighted_inverted_cdf_p95_of_3494_full_rank_empirical_matrices"
)
CALIBRATION_TARGET_DONOR_STATES = DONOR_STATES
EXPERIMENT_PURPOSE = "sc_only_coverage_by_quantile_top4_condition_number"
MASTER_SEED = 2026080201
DEFAULT_SAMPLE_SIZES = (2000, 4000)
DEFAULT_OUTER_REPS = 500
DEFAULT_BOOTSTRAPS = 500
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "generated"
DEFAULT_TAG = "production_v1"
# Direct-SVD condition grid.  Every entry is kappa(M), never kappa(M'M).
DEFAULT_KAPPA_M_GRID = (
    1.0,
    2.0,
    4.0,
    10.0,
    20.0,
    40.0,
    80.0,
    160.0,
)
# Frozen 2025-09-16 augmentation scale used by the Quantile-top-4 reference.
# This fixes the absolute singular-value scale and is not the condition grid.
REFERENCE_AUGMENTATION_GAMMA = math.sqrt(29.077419588830004)
EMPIRICAL_POST_CONTRAST = np.asarray(
    [0.211789512326912, 0.9801124738736031, 0.3509230190287762],
    dtype=float,
)
DEFAULT_POST_SCALE = 3.0
N_GROUPS = 5
N_DONORS = 4
N_PRE = 5
N_PERIODS = 6
N_X = 101
N_FOLDS = 2
TREATMENT_LEVEL = 1.0
TREATMENT_AMPLITUDE = 0.7
BANDWIDTH_COEFFICIENT = 2.5
RIDGE_LLR = 1e-6
RIDGE_SC = 1e-6
TRIM_QUANTILES = (0.0, 1.0)
WEIGHT_ESTIMATOR = (
    "ridge-stabilized unrestricted affine least squares followed by "
    "Euclidean projection of all four donor weights onto the simplex"
)


@dataclass(frozen=True)
class Design:
    """Frozen population quantities for the report DGP."""

    kappa_m: float
    x_support: np.ndarray
    group_probabilities: np.ndarray
    donor_means: np.ndarray
    treated_untreated_means: np.ndarray
    treatment_effect: np.ndarray
    true_weights: np.ndarray
    residual_variances: np.ndarray
    att_true: float
    population_trim_normalized_score_target: float
    population_kappa_m_median: float
    population_kappa_m_p95: float
    population_kappa_mtm_median: float
    population_kappa_mtm_p95: float
    population_sigma_max_median: float
    population_sigma_middle_median: float
    population_sigma_min_median: float
    population_rank_min: int
    maximum_sc_residual: float
    maximum_pt_gap: float
    weak_post_projection_median: float
    weak_post_projection_over_sigma_min_median: float
    post_scale: float


@dataclass(frozen=True)
class LatentSample:
    """One outer panel's covariate, group, and fixed fold assignment."""

    x_index: np.ndarray
    group: np.ndarray
    folds: np.ndarray
    fold_hash: str


@dataclass(frozen=True)
class OuterResult:
    """One point estimate and all requested multiplier-bootstrap estimates."""

    kappa_m: float
    replication: int
    base_seed: int
    fold_hash: str
    point_estimate: float
    bootstrap_estimates: np.ndarray
    bootstrap_errors: np.ndarray
    bootstrap_failures: dict[int, str]
    sample_kappa_m_median: float
    sample_kappa_m_p95: float
    sample_kappa_m_max: float
    sample_sigma_max_median: float
    sample_sigma_middle_median: float
    sample_sigma_min_median: float
    sample_rank_min: int
    sample_rank_deficient_share: float
    maximum_estimated_weight: float
    estimated_weight_l2_median: float
    raw_weight_l2_median: float
    simplex_projection_distance_median: float
    simplex_boundary_fraction: float
    projected_weight_error_l2_median: float
    raw_weight_error_l2_median: float
    pre_fit_l2_median: float
    pre_fit_max: float
    point_failure: str
    runtime_seconds: float


@dataclass(frozen=True)
class SampleOutput:
    """Output paths and final coverage summary for one sample size."""

    n: int
    outer_path: Path
    bootstrap_path: Path
    coverage_path: Path
    population_path: Path
    summaries: list[dict[str, float | int | str]]


_DESIGNS: dict[float, Design] | None = None
_KAPPA_M_GRID: tuple[float, ...] = ()
_POST_SCALE = DEFAULT_POST_SCALE


def _null_space_rows(vector: np.ndarray) -> np.ndarray:
    """Return deterministic orthonormal rows spanning vector-perpendicular."""

    _, _, right_vectors = np.linalg.svd(
        vector.reshape(1, -1),
        full_matrices=True,
    )
    return right_vectors[1:, :]


def _rank_completion_paths(
    factor_paths: np.ndarray,
    fixed_effect_contrasts: np.ndarray,
    relative_scale: float,
) -> tuple[np.ndarray, float]:
    """Return scale-equivariant deterministic rank-completion paths."""

    empirical_span = np.column_stack(
        [np.ones(N_PRE), factor_paths[:N_PRE]]
    )
    _, _, right_vectors = np.linalg.svd(
        empirical_span.T,
        full_matrices=True,
    )
    h_matrix = right_vectors.T[:, 3:5]
    if h_matrix.shape != (N_PRE, N_DONORS - 2):
        raise RuntimeError("unexpected rank-completion dimension")
    c_matrix = _null_space_rows(fixed_effect_contrasts)
    completion_magnitude = float(
        relative_scale
        * math.sqrt(N_PRE)
        * np.linalg.norm(fixed_effect_contrasts)
    )
    donor_loadings = np.zeros((2, N_DONORS), dtype=float)
    donor_loadings[:, : N_DONORS - 1] = (
        completion_magnitude * c_matrix
    )
    factor_with_post = np.vstack([h_matrix, h_matrix[-1]])
    return factor_with_post @ donor_loadings, completion_magnitude


def build_design(
    kappa_m: float,
    post_scale: float = DEFAULT_POST_SCALE,
) -> Design:
    """Construct the six-period, four-donor SC-only population."""

    kappa_m = float(kappa_m)
    if not np.isfinite(kappa_m) or kappa_m < 1.0:
        raise ValueError("kappa_m must be finite and at least one")
    post_scale = float(post_scale)
    if not np.isfinite(post_scale) or post_scale <= 0.0:
        raise ValueError("post_scale must be finite and strictly positive")
    x_support = np.linspace(0.0, 1.0, N_X)

    # Original all-state MNLogit coefficients, renormalized to Alaska and the
    # four report donors.  Alaska is the zero-logit reference category.
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
    # Polynomial coefficients are ordered as intercept, x, x^2.  Dimensions
    # are donor x factor x coefficient.
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
    reference_rank_completion, _ = _rank_completion_paths(
        factor_paths,
        fixed_effect_contrasts,
        REFERENCE_AUGMENTATION_GAMMA,
    )
    reference_donor_means = (
        donor_fixed_effects[None, None, :]
        + time_effects[None, :, None]
        + np.einsum(
            "tf,xdf->xtd",
            factor_paths,
            donor_factor_loadings,
        )
        + reference_rank_completion[None, :, :]
    )

    # Lower-sigma_min path used by factor_subset_dgp_weak_condition.py.  The
    # Virginia trajectory (last donor) is held fixed.  At each x, decompose
    # the calibrated 5x3 baseline-differenced donor matrix, preserve U, V and
    # sigma_max, and set sigma_min=sigma_max/kappa_m.  Values below that floor
    # are lifted, which keeps the singular values ordered at low kappa.
    donor_means = reference_donor_means.copy()
    weak_post_projections = np.empty(N_X, dtype=float)
    for x_index in range(N_X):
        baseline = reference_donor_means[x_index, :, -1]
        reference_m = (
            reference_donor_means[x_index, :N_PRE, : N_DONORS - 1]
            - baseline[:N_PRE, None]
        )
        left_vectors, singular_values_reference, right_vectors = np.linalg.svd(
            reference_m,
            full_matrices=False,
        )
        target_min = singular_values_reference[0] / kappa_m
        adjusted = np.maximum(singular_values_reference, target_min)
        adjusted[-1] = target_min
        target_m = (
            left_vectors
            @ np.diag(adjusted)
            @ right_vectors
        )
        donor_means[x_index, :N_PRE, : N_DONORS - 1] = (
            baseline[:N_PRE, None] + target_m
        )
        post_contrast = (
            target_m[-1] + post_scale * EMPIRICAL_POST_CONTRAST
        )
        donor_means[x_index, -1, : N_DONORS - 1] = (
            baseline[-1] + post_contrast
        )
        _, _, target_right_vectors = np.linalg.svd(
            target_m,
            full_matrices=False,
        )
        weak_post_projections[x_index] = abs(
            float(
                target_right_vectors[-1]
                @ (post_contrast - target_m[-1])
            )
        )

    donor_indices = np.arange(1, N_DONORS + 1, dtype=float)
    weight_logits = (
        x_support[:, None] * (donor_indices[None, :] - N_DONORS)
    )
    weight_logits -= weight_logits.max(axis=1, keepdims=True)
    softmax_weights = np.exp(weight_logits)
    softmax_weights /= softmax_weights.sum(axis=1, keepdims=True)
    true_weights = 0.2 / N_DONORS + 0.8 * softmax_weights
    treated_untreated_means = np.einsum(
        "xd,xtd->xt",
        true_weights,
        donor_means,
    )
    treatment_effect = TREATMENT_LEVEL + TREATMENT_AMPLITUDE * np.sin(
        2.0 * np.pi * x_support
    )
    treated_propensity = group_probabilities[:, 0]
    att_true = float(
        np.sum(treated_propensity * treatment_effect)
        / np.sum(treated_propensity)
    )
    trim_mask = (
        (x_support > TRIM_QUANTILES[0])
        & (x_support < TRIM_QUANTILES[1])
    )
    population_trim_normalized_score_target = float(
        np.sum(treated_propensity[trim_mask] * treatment_effect[trim_mask])
        / np.sum(treated_propensity)
        / np.mean(trim_mask)
    )

    condition_numbers = np.empty(N_X, dtype=float)
    singular_values = np.empty((N_X, N_DONORS - 1), dtype=float)
    ranks = np.empty(N_X, dtype=int)
    for x_index in range(N_X):
        donor_pre = donor_means[x_index, :N_PRE]
        matrix_m = (
            donor_pre[:, : N_DONORS - 1]
            - donor_pre[:, [-1]]
        )
        values = np.linalg.svd(matrix_m, compute_uv=False)
        singular_values[x_index] = values
        ranks[x_index] = np.linalg.matrix_rank(matrix_m)
        condition_numbers[x_index] = values[0] / values[-1]

    reconstructed_treated = np.einsum(
        "xd,xtd->xt",
        true_weights,
        donor_means,
    )
    maximum_sc_residual = float(
        np.max(np.abs(treated_untreated_means - reconstructed_treated))
    )
    donor_changes = donor_means[:, -1] - donor_means[:, -2]
    treated_changes = (
        treated_untreated_means[:, -1]
        - treated_untreated_means[:, -2]
    )
    maximum_pt_gap = float(
        np.max(np.abs(treated_changes[:, None] - donor_changes))
    )

    return Design(
        kappa_m=kappa_m,
        x_support=x_support,
        group_probabilities=group_probabilities,
        donor_means=donor_means,
        treated_untreated_means=treated_untreated_means,
        treatment_effect=treatment_effect,
        true_weights=true_weights,
        residual_variances=residual_variances,
        att_true=att_true,
        population_trim_normalized_score_target=(
            population_trim_normalized_score_target
        ),
        population_kappa_m_median=float(
            np.median(condition_numbers)
        ),
        population_kappa_m_p95=float(
            np.quantile(condition_numbers, 0.95)
        ),
        population_kappa_mtm_median=float(
            np.median(condition_numbers**2)
        ),
        population_kappa_mtm_p95=float(
            np.quantile(condition_numbers**2, 0.95)
        ),
        population_sigma_max_median=float(
            np.median(singular_values[:, 0])
        ),
        population_sigma_middle_median=float(
            np.median(singular_values[:, 1])
        ),
        population_sigma_min_median=float(
            np.median(singular_values[:, 2])
        ),
        population_rank_min=int(np.min(ranks)),
        maximum_sc_residual=maximum_sc_residual,
        maximum_pt_gap=maximum_pt_gap,
        weak_post_projection_median=float(
            np.median(weak_post_projections)
        ),
        weak_post_projection_over_sigma_min_median=float(
            np.median(weak_post_projections / singular_values[:, -1])
        ),
        post_scale=post_scale,
    )


@njit
def project_to_simplex(weights: np.ndarray) -> np.ndarray:
    """Euclidean projection onto nonnegative weights that sum to one."""

    ordered = np.sort(weights)[::-1]
    cumulative = np.cumsum(ordered)
    rho = np.nonzero(
        ordered * np.arange(1, len(weights) + 1)
        > cumulative - 1.0
    )[0][-1]
    threshold = (cumulative[rho] - 1.0) / (rho + 1)
    projected = weights - threshold
    for index in range(len(projected)):
        if projected[index] < 0.0:
            projected[index] = 0.0
    return projected


@njit
def affine_ridge_projected_weights(
    moments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the legacy affine solve followed by simplex projection."""

    matrix_m = (
        moments[:, 1:N_DONORS]
        - moments[:, N_DONORS][:, None]
    )
    target = moments[:, 0] - moments[:, N_DONORS]
    gram = matrix_m.T @ matrix_m
    eigenvalues = np.linalg.eigvalsh(gram)
    lambda_min = max(eigenvalues[0], 0.0)
    lambda_max = max(eigenvalues[-1], 0.0)
    condition_mtm = (
        np.inf if lambda_min <= 0.0 else lambda_max / lambda_min
    )
    ridge = RIDGE_SC
    if condition_mtm > 1.0e6:
        diagonal_scale = 0.0
        for index in range(gram.shape[0]):
            diagonal_scale = max(diagonal_scale, abs(gram[index, index]))
        ridge = RIDGE_SC * max(diagonal_scale, 1.0)
    regularized = gram + ridge * np.eye(N_DONORS - 1)
    rhs = np.zeros(N_DONORS - 1)
    for column in range(N_DONORS - 1):
        for row in range(N_PRE):
            rhs[column] += matrix_m[row, column] * target[row]
    free_weights = np.linalg.solve(regularized, rhs)
    raw_weights = np.empty(N_DONORS)
    total = 0.0
    for index in range(N_DONORS - 1):
        raw_weights[index] = free_weights[index]
        total += free_weights[index]
    raw_weights[-1] = 1.0 - total
    return project_to_simplex(raw_weights), raw_weights


def aggregate_panel_cells(
    latent: LatentSample,
    outcomes: np.ndarray,
    observation_weights: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate weighted counts and outcomes to group-covariate cells."""

    flat = latent.group * N_X + latent.x_index
    size = N_GROUPS * N_X
    counts = np.bincount(
        flat[mask],
        weights=observation_weights[mask],
        minlength=size,
    ).reshape(N_GROUPS, N_X)
    sums = np.empty((N_GROUPS, N_PERIODS, N_X), dtype=float)
    for period in range(N_PERIODS):
        sums[:, period] = np.bincount(
            flat[mask],
            weights=(
                observation_weights[mask] * outcomes[mask, period]
            ),
            minlength=size,
        ).reshape(N_GROUPS, N_X)
    return counts, sums


def local_linear_intercept(
    counts: np.ndarray,
    response_sums: np.ndarray,
    support: np.ndarray,
    evaluation: float,
    bandwidth: float,
) -> float:
    """Return a ridge-stabilized local-linear intercept."""

    centered = support - evaluation
    scaled = centered / bandwidth
    kernel = np.where(
        np.abs(scaled) <= 1.0,
        0.75 * (1.0 - scaled * scaled),
        0.0,
    )
    weighted_counts = kernel * counts
    s0 = float(weighted_counts.sum())
    s1 = float(weighted_counts @ centered)
    s2 = float(weighted_counts @ (centered * centered))
    t0 = float(kernel @ response_sums)
    t1 = float((kernel * centered) @ response_sums)
    ridge = RIDGE_LLR * max((s0 + s2) / 2.0, 1.0)
    a00 = s0 + ridge
    a11 = s2 + ridge
    determinant = max(a00 * a11 - s1 * s1, ridge)
    return float(
        (a11 * t0 - s1 * t1) / determinant
    )


def local_ratio(
    donor_counts: np.ndarray,
    treated_counts: np.ndarray,
    support: np.ndarray,
    evaluation: float,
    bandwidth: float,
) -> float:
    """Return the local-linear treated-to-donor density ratio."""

    centered = support - evaluation
    scaled = centered / bandwidth
    kernel = np.where(
        np.abs(scaled) <= 1.0,
        0.75 * (1.0 - scaled * scaled),
        0.0,
    )
    weighted_donor = kernel * donor_counts
    s0 = float(weighted_donor.sum())
    s1 = float(weighted_donor @ centered)
    s2 = float(weighted_donor @ (centered * centered))
    weighted_treated = kernel * treated_counts
    t0 = float(weighted_treated.sum())
    t1 = float(weighted_treated @ centered)
    determinant = s0 * s2 - s1 * s1
    scale = (s0 + s2) / 2.0
    adaptive_epsilon = 1e-6 * max(scale, 1.0)
    if abs(determinant) < adaptive_epsilon:
        if abs(s0) > adaptive_epsilon:
            return float(t0 / s0)
        return float(t0 / (s0 + adaptive_epsilon))
    return float((t0 * s2 - t1 * s1) / determinant)


def estimate_didsc(
    design: Design,
    latent: LatentSample,
    outcomes: np.ndarray,
    observation_weights: np.ndarray,
    collect_diagnostics: bool = False,
) -> tuple[float, dict[str, float | int]]:
    """Evaluate the fixed-fold score and optional matrix diagnostics."""

    n = outcomes.shape[0]
    observed_x = design.x_support[latent.x_index]
    trim_low, trim_high = np.quantile(observed_x, TRIM_QUANTILES)
    observed_indices = np.zeros(N_X, dtype=bool)
    observed_indices[latent.x_index] = True
    eval_indices = np.flatnonzero(
        (design.x_support > trim_low)
        & (design.x_support < trim_high)
        & observed_indices
    )
    if eval_indices.size == 0:
        raise FloatingPointError("no covariate points survive trimming")

    pi_treated = float(
        np.sum(observation_weights * (latent.group == 0)) / n
    )
    if not np.isfinite(pi_treated) or pi_treated <= 0.0:
        raise FloatingPointError("weighted treated share is invalid")

    bandwidth = BANDWIDTH_COEFFICIENT * n ** (-1.0 / 3.5)
    fold_estimates: list[float] = []
    kappas: list[float] = []
    sigmas: list[np.ndarray] = []
    ranks: list[int] = []
    maximum_weight = 0.0
    estimated_weight_l2: list[float] = []
    raw_weight_l2: list[float] = []
    projection_distances: list[float] = []
    boundary_indicators: list[float] = []
    projected_weight_errors: list[float] = []
    raw_weight_errors: list[float] = []
    pre_fit_l2: list[float] = []
    pre_fit_maximum: list[float] = []
    for fold in range(N_FOLDS):
        test = latent.folds == fold
        train = ~test
        if not np.any(test) or not np.any(train):
            raise FloatingPointError("empty cross-fitting fold")
        train_counts, train_sums = aggregate_panel_cells(
            latent,
            outcomes,
            observation_weights,
            train,
        )
        test_counts, test_sums = aggregate_panel_cells(
            latent,
            outcomes,
            observation_weights,
            test,
        )

        score_sum = 0.0
        group_x_counts = train_counts
        pooled_control_counts = train_counts[1:].sum(axis=0)
        pooled_control_change_sums = (
            train_sums[1:, -1].sum(axis=0)
            - train_sums[1:, -2].sum(axis=0)
        )
        for x_index in eval_indices:
            evaluation = float(design.x_support[x_index])
            moments = np.empty((N_PRE, N_GROUPS), dtype=float)
            for group_index in range(N_GROUPS):
                for period in range(N_PRE):
                    moments[period, group_index] = local_linear_intercept(
                        group_x_counts[group_index],
                        train_sums[group_index, period],
                        design.x_support,
                        evaluation,
                        bandwidth,
                    )
            donor_weights, raw_weights = affine_ridge_projected_weights(
                moments
            )
            if not np.all(np.isfinite(donor_weights)):
                raise FloatingPointError("nonfinite synthetic-control weights")

            if collect_diagnostics:
                matrix_m = moments[:, 1:N_DONORS] - moments[:, [-1]]
                values = np.linalg.svd(matrix_m, compute_uv=False)
                rank = int(np.linalg.matrix_rank(matrix_m))
                if rank == N_DONORS - 1 and values[-1] > 0.0:
                    kappas.append(float(values[0] / values[-1]))
                    sigmas.append(values)
                ranks.append(rank)
                maximum_weight = max(
                    maximum_weight,
                    float(np.max(np.abs(donor_weights))),
                )
                estimated_weight_l2.append(
                    float(np.linalg.norm(donor_weights))
                )
                raw_weight_l2.append(float(np.linalg.norm(raw_weights)))
                projection_distances.append(
                    float(np.linalg.norm(donor_weights - raw_weights))
                )
                boundary_indicators.append(
                    float(np.min(donor_weights) <= 1e-10)
                )
                true_weights = design.true_weights[x_index]
                projected_weight_errors.append(
                    float(np.linalg.norm(donor_weights - true_weights))
                )
                raw_weight_errors.append(
                    float(np.linalg.norm(raw_weights - true_weights))
                )
                fit_residual = (
                    moments[:, 0] - moments[:, 1:] @ donor_weights
                )
                pre_fit_l2.append(float(np.linalg.norm(fit_residual)))
                pre_fit_maximum.append(
                    float(np.max(np.abs(fit_residual)))
                )

            nuisance_change = local_linear_intercept(
                pooled_control_counts,
                pooled_control_change_sums,
                design.x_support,
                evaluation,
                bandwidth,
            )
            ratios = np.asarray(
                [
                    local_ratio(
                        group_x_counts[group_index],
                        group_x_counts[0],
                        design.x_support,
                        evaluation,
                        bandwidth,
                    )
                    for group_index in range(1, N_GROUPS)
                ],
                dtype=float,
            )
            group_coefficients = np.r_[1.0, -donor_weights * ratios]
            for group_index in range(N_GROUPS):
                cell_contribution = (
                    test_sums[group_index, -1, x_index]
                    - test_sums[group_index, -2, x_index]
                    - nuisance_change * test_counts[group_index, x_index]
                )
                score_sum += (
                    group_coefficients[group_index] * cell_contribution
                )

        trimmed_test_count = int(
            np.sum(
                test
                & (observed_x > trim_low)
                & (observed_x < trim_high)
            )
        )
        if trimmed_test_count <= 0:
            raise FloatingPointError("trimmed evaluation fold is empty")
        fold_estimates.append(
            float(score_sum / pi_treated / trimmed_test_count)
        )

    estimate = float(np.mean(fold_estimates))
    if not np.isfinite(estimate):
        raise FloatingPointError("nonfinite DiD-SC estimate")
    if not collect_diagnostics:
        return estimate, {}
    if not kappas or not sigmas:
        raise FloatingPointError("no full-rank sample matrices")
    sigma_array = np.asarray(sigmas, dtype=float)
    return estimate, {
        "sample_kappa_m_median": float(np.median(kappas)),
        "sample_kappa_m_p95": float(np.quantile(kappas, 0.95)),
        "sample_kappa_m_max": float(np.max(kappas)),
        "sample_sigma_max_median": float(np.median(sigma_array[:, 0])),
        "sample_sigma_middle_median": float(np.median(sigma_array[:, 1])),
        "sample_sigma_min_median": float(np.median(sigma_array[:, 2])),
        "sample_rank_min": int(min(ranks)),
        "sample_rank_deficient_share": float(
            np.mean(np.asarray(ranks) < N_DONORS - 1)
        ),
        "maximum_estimated_weight": float(maximum_weight),
        "estimated_weight_l2_median": float(
            np.median(estimated_weight_l2)
        ),
        "raw_weight_l2_median": float(np.median(raw_weight_l2)),
        "simplex_projection_distance_median": float(
            np.median(projection_distances)
        ),
        "simplex_boundary_fraction": float(
            np.mean(boundary_indicators)
        ),
        "projected_weight_error_l2_median": float(
            np.median(projected_weight_errors)
        ),
        "raw_weight_error_l2_median": float(
            np.median(raw_weight_errors)
        ),
        "pre_fit_l2_median": float(np.median(pre_fit_l2)),
        "pre_fit_max": float(np.max(pre_fit_maximum)),
    }


def replication_seed(master_seed: int, n: int, replication: int) -> int:
    """Return a seed stable to worker count and task ordering."""

    sequence = np.random.SeedSequence(
        [int(master_seed), int(n), int(replication)]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def generate_outer_sample(
    design: Design,
    n: int,
    base_seed: int,
) -> tuple[LatentSample, np.ndarray, np.random.Generator]:
    """Generate one panel and its independent multiplier RNG stream."""

    sequence = np.random.SeedSequence(int(base_seed))
    data_stream, residual_stream, fold_stream, multiplier_stream = (
        sequence.spawn(4)
    )
    data_rng = np.random.default_rng(data_stream)
    residual_rng = np.random.default_rng(residual_stream)
    fold_rng = np.random.default_rng(fold_stream)
    multiplier_rng = np.random.default_rng(multiplier_stream)

    x_index = data_rng.integers(0, N_X, size=n)
    probabilities = design.group_probabilities[x_index]
    uniforms = data_rng.random(n)
    cumulative = np.cumsum(probabilities, axis=1)
    group = np.sum(uniforms[:, None] > cumulative, axis=1).astype(np.int64)
    group = np.minimum(group, N_GROUPS - 1)

    outcomes = np.empty((n, N_PERIODS), dtype=float)
    treated = group == 0
    outcomes[treated] = design.treated_untreated_means[x_index[treated]]
    for donor in range(N_DONORS):
        mask = group == donor + 1
        outcomes[mask] = design.donor_means[x_index[mask], :, donor]
    outcomes += residual_rng.normal(size=(n, N_PERIODS)) * np.sqrt(
        design.residual_variances
    )[None, :]
    outcomes[treated, -1] += design.treatment_effect[x_index[treated]]

    folds = fold_rng.integers(0, N_FOLDS, size=n, dtype=np.int8)
    fold_hash = hashlib.sha256(folds.tobytes()).hexdigest()
    latent = LatentSample(
        x_index=x_index,
        group=group,
        folds=folds,
        fold_hash=fold_hash,
    )
    return latent, outcomes, multiplier_rng


def initialize_worker(
    kappa_m_grid: tuple[float, ...],
    post_scale: float,
) -> None:
    """Build all grid designs and warm Numba once in every worker."""

    global _DESIGNS, _KAPPA_M_GRID, _POST_SCALE
    _KAPPA_M_GRID = tuple(float(kappa_m) for kappa_m in kappa_m_grid)
    _POST_SCALE = float(post_scale)
    _DESIGNS = {
        kappa_m: build_design(kappa_m, _POST_SCALE)
        for kappa_m in _KAPPA_M_GRID
    }
    warmup = np.column_stack(
        [
            np.linspace(0.0, 1.0, N_PRE),
            np.linspace(0.1, 1.1, N_PRE),
            np.linspace(0.2, 1.2, N_PRE),
            np.linspace(0.3, 1.3, N_PRE),
            np.linspace(0.4, 1.4, N_PRE),
        ]
    )
    affine_ridge_projected_weights(warmup)


def run_outer_task(
    task: tuple[int, int, float, int, int],
) -> OuterResult:
    """Compute one point estimate and all requested bootstrap estimates."""

    replication, n, kappa_m, bootstraps, master_seed = task
    if _DESIGNS is None or kappa_m not in _DESIGNS:
        raise RuntimeError("worker designs were not initialized")
    design = _DESIGNS[kappa_m]
    started = time.perf_counter()
    base_seed = replication_seed(master_seed, n, replication)
    bootstrap_estimates = np.full(bootstraps, np.nan, dtype=float)
    bootstrap_errors = np.full(bootstraps, np.nan, dtype=float)
    failures: dict[int, str] = {}
    fold_hash = ""
    diagnostics: dict[str, float | int] = {}
    try:
        latent, outcomes, multiplier_rng = generate_outer_sample(
            design,
            n,
            base_seed,
        )
        fold_hash = latent.fold_hash
        point_estimate, diagnostics = estimate_didsc(
            design,
            latent,
            outcomes,
            np.ones(n, dtype=float),
            collect_diagnostics=True,
        )
    except Exception as error:
        return OuterResult(
            kappa_m=kappa_m,
            replication=replication,
            base_seed=base_seed,
            fold_hash=fold_hash,
            point_estimate=np.nan,
            bootstrap_estimates=bootstrap_estimates,
            bootstrap_errors=bootstrap_errors,
            bootstrap_failures=failures,
            sample_kappa_m_median=np.nan,
            sample_kappa_m_p95=np.nan,
            sample_kappa_m_max=np.nan,
            sample_sigma_max_median=np.nan,
            sample_sigma_middle_median=np.nan,
            sample_sigma_min_median=np.nan,
            sample_rank_min=0,
            sample_rank_deficient_share=np.nan,
            maximum_estimated_weight=np.nan,
            estimated_weight_l2_median=np.nan,
            raw_weight_l2_median=np.nan,
            simplex_projection_distance_median=np.nan,
            simplex_boundary_fraction=np.nan,
            projected_weight_error_l2_median=np.nan,
            raw_weight_error_l2_median=np.nan,
            pre_fit_l2_median=np.nan,
            pre_fit_max=np.nan,
            point_failure=f"{type(error).__name__}: {error}",
            runtime_seconds=float(time.perf_counter() - started),
        )

    for bootstrap in range(bootstraps):
        weights = multiplier_rng.exponential(1.0, size=n)
        try:
            weighted_estimate, _ = estimate_didsc(
                design,
                latent,
                outcomes,
                weights,
                collect_diagnostics=False,
            )
            bootstrap_estimates[bootstrap] = weighted_estimate
            bootstrap_errors[bootstrap] = weighted_estimate - point_estimate
        except Exception as error:
            failures[bootstrap] = f"{type(error).__name__}: {error}"

    return OuterResult(
        kappa_m=kappa_m,
        replication=replication,
        base_seed=base_seed,
        fold_hash=fold_hash,
        point_estimate=point_estimate,
        bootstrap_estimates=bootstrap_estimates,
        bootstrap_errors=bootstrap_errors,
        bootstrap_failures=failures,
        sample_kappa_m_median=float(
            diagnostics["sample_kappa_m_median"]
        ),
        sample_kappa_m_p95=float(diagnostics["sample_kappa_m_p95"]),
        sample_kappa_m_max=float(diagnostics["sample_kappa_m_max"]),
        sample_sigma_max_median=float(
            diagnostics["sample_sigma_max_median"]
        ),
        sample_sigma_middle_median=float(
            diagnostics["sample_sigma_middle_median"]
        ),
        sample_sigma_min_median=float(
            diagnostics["sample_sigma_min_median"]
        ),
        sample_rank_min=int(diagnostics["sample_rank_min"]),
        sample_rank_deficient_share=float(
            diagnostics["sample_rank_deficient_share"]
        ),
        maximum_estimated_weight=float(
            diagnostics["maximum_estimated_weight"]
        ),
        estimated_weight_l2_median=float(
            diagnostics["estimated_weight_l2_median"]
        ),
        raw_weight_l2_median=float(
            diagnostics["raw_weight_l2_median"]
        ),
        simplex_projection_distance_median=float(
            diagnostics["simplex_projection_distance_median"]
        ),
        simplex_boundary_fraction=float(
            diagnostics["simplex_boundary_fraction"]
        ),
        projected_weight_error_l2_median=float(
            diagnostics["projected_weight_error_l2_median"]
        ),
        raw_weight_error_l2_median=float(
            diagnostics["raw_weight_error_l2_median"]
        ),
        pre_fit_l2_median=float(diagnostics["pre_fit_l2_median"]),
        pre_fit_max=float(diagnostics["pre_fit_max"]),
        point_failure="",
        runtime_seconds=float(time.perf_counter() - started),
    )


def run_replication_grid_task(
    task: tuple[int, int, int, int],
) -> list[OuterResult]:
    """Evaluate every kappa_m for one common-random-number outer panel."""

    replication, n, bootstraps, master_seed = task
    return [
        run_outer_task(
            (replication, n, kappa_m, bootstraps, master_seed)
        )
        for kappa_m in _KAPPA_M_GRID
    ]


def available_cpu_count() -> int:
    """Return CPUs visible to the current interactive allocation."""

    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def population_diagnostics(design: Design) -> list[dict[str, float | int]]:
    """Return the complete 101-point population spectral and truth audit."""

    rows: list[dict[str, float | int]] = []
    for x_index, x_value in enumerate(design.x_support):
        donor_paths = design.donor_means[x_index]
        matrix_m = (
            donor_paths[:N_PRE, : N_DONORS - 1]
            - donor_paths[:N_PRE, [-1]]
        )
        _, singular_values, right_vectors = np.linalg.svd(
            matrix_m,
            full_matrices=False,
        )
        rank = int(np.linalg.matrix_rank(matrix_m))
        kappa_m_realized = float(
            singular_values[0] / singular_values[-1]
        )
        gram_eigenvalues = singular_values**2
        treated_path = design.treated_untreated_means[x_index]
        reconstructed = donor_paths @ design.true_weights[x_index]
        donor_changes = donor_paths[-1] - donor_paths[-2]
        treated_change = treated_path[-1] - treated_path[-2]
        post_change_contrast = (
            donor_changes[: N_DONORS - 1] - donor_changes[-1]
        )
        rows.append(
            {
                "x": float(x_value),
                "kappa_m_target": float(design.kappa_m),
                "kappa_m_realized": kappa_m_realized,
                "kappa_mtm_derived": float(kappa_m_realized**2),
                "sigma_max_m": float(singular_values[0]),
                "sigma_middle_m": float(singular_values[1]),
                "sigma_min_m": float(singular_values[2]),
                "lambda_max_mtm": float(gram_eigenvalues[0]),
                "lambda_middle_mtm": float(gram_eigenvalues[1]),
                "lambda_min_mtm": float(gram_eigenvalues[2]),
                "rank_m": rank,
                "matrix_rows": N_PRE,
                "matrix_columns": N_DONORS - 1,
                "free_donor_weights": N_DONORS - 1,
                "excess_pre_restrictions": N_PRE - (N_DONORS - 1),
                "sc_residual_max": float(
                    np.max(np.abs(treated_path - reconstructed))
                ),
                "pt_gap_max": float(
                    np.max(np.abs(treated_change - donor_changes))
                ),
                "true_weight_min": float(
                    np.min(design.true_weights[x_index])
                ),
                "true_weight_max": float(
                    np.max(design.true_weights[x_index])
                ),
                "true_weight_l2": float(
                    np.linalg.norm(design.true_weights[x_index])
                ),
                "weak_post_projection_abs": abs(
                    float(right_vectors[-1] @ post_change_contrast)
                ),
                "post_scale": float(design.post_scale),
                "att_true": float(design.att_true),
                "trim_score_target": float(
                    design.population_trim_normalized_score_target
                ),
            }
        )
    return rows


def validate_population_design(design: Design) -> None:
    """Fail before simulation if the intended weak-path properties do not hold."""

    rows = population_diagnostics(design)
    realized = np.asarray([row["kappa_m_realized"] for row in rows])
    ranks = np.asarray([row["rank_m"] for row in rows])
    sc_residuals = np.asarray([row["sc_residual_max"] for row in rows])
    pt_gaps = np.asarray([row["pt_gap_max"] for row in rows])
    weak_projection = np.asarray(
        [row["weak_post_projection_abs"] for row in rows]
    )
    donor_changes = design.donor_means[:, -1] - design.donor_means[:, -2]
    post_contrasts = (
        donor_changes[:, : N_DONORS - 1] - donor_changes[:, [-1]]
    )
    if not np.allclose(realized, design.kappa_m, rtol=1e-10, atol=1e-10):
        raise AssertionError("population kappa(M) does not equal its target")
    if not np.all(ranks == N_DONORS - 1):
        raise AssertionError("population donor matrix is not full rank")
    if float(np.max(sc_residuals)) > 1e-10:
        raise AssertionError("population synthetic-control restriction fails")
    if float(np.max(pt_gaps)) <= 1e-8:
        raise AssertionError("parallel trends unexpectedly holds")
    if float(np.max(weak_projection)) <= 1e-10:
        raise AssertionError("post change is irrelevant to the weak direction")
    if not np.allclose(
        post_contrasts,
        design.post_scale * EMPIRICAL_POST_CONTRAST,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError("calibrated donor post-change contrasts changed")


def write_population_diagnostics(
    path: Path,
    designs: dict[float, Design],
) -> None:
    """Write one row per target and covariate support point."""

    rows: list[dict[str, float | int | str]] = []
    for kappa_m, design in designs.items():
        validate_population_design(design)
        for row in population_diagnostics(design):
            rows.append(
                {
                    "dgp_version": DGP_VERSION,
                    "dgp_name": "sc_only",
                    "donor_codes": "|".join(str(code) for code in DONOR_CODES),
                    "donor_states": DONOR_STATES,
                    "baseline_donor_code": BASELINE_DONOR_CODE,
                    "baseline_donor_state": BASELINE_DONOR_STATE,
                    "condition_definition": CONDITION_DEFINITION,
                    "empirical_kappa_m_reference": CALIBRATION_TARGET_KAPPA_M,
                    "empirical_kappa_mtm_reference": CALIBRATION_TARGET_KAPPA_MTM,
                    "weight_estimator": WEIGHT_ESTIMATOR,
                    **row,
                }
            )
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def output_paths(prefix: Path) -> tuple[Path, Path, Path, Path]:
    """Return normalized output CSV paths."""

    prefix = prefix.expanduser().resolve()
    return (
        prefix.with_name(prefix.name + "_outer.csv"),
        prefix.with_name(prefix.name + "_bootstrap_diagnostics.csv"),
        prefix.with_name(prefix.name + "_coverage.csv"),
        prefix.with_name(prefix.name + "_population.csv"),
    )


def output_prefix_for_n(args: argparse.Namespace, n: int) -> Path:
    """Return a collision-free output prefix for one sample size."""

    shard = (
        f"_start{int(args.outer_start)}"
        if int(args.outer_start) != 0
        else ""
    )
    if args.output_prefix is None:
        return Path(args.results_dir) / (
            f"q4cond_sc_only_{args.tag}_n{n}_"
            f"R{int(args.outer_reps)}_"
            f"B{int(args.bootstraps)}{shard}"
        )
    prefix = Path(args.output_prefix)
    if len(args.n) > 1:
        prefix = prefix.with_name(f"{prefix.name}_n{n}")
    if shard:
        prefix = prefix.with_name(f"{prefix.name}{shard}")
    return prefix


def _wilson_interval(coverage: float, count: int) -> tuple[float, float]:
    """Return a 95% Wilson interval for a simulated coverage proportion."""

    if count <= 0 or not np.isfinite(coverage):
        return np.nan, np.nan
    z_value = 1.959963984540054
    denominator = 1.0 + z_value * z_value / count
    center = (
        coverage + z_value * z_value / (2.0 * count)
    ) / denominator
    half_width = (
        z_value
        * math.sqrt(
            coverage * (1.0 - coverage) / count
            + z_value * z_value / (4.0 * count * count)
        )
        / denominator
    )
    return center - half_width, center + half_width


def _coverage_mcse(coverage: float, count: int) -> float:
    """Return the naive Bernoulli Monte Carlo standard error."""

    if count <= 0 or not np.isfinite(coverage):
        return np.nan
    return float(math.sqrt(coverage * (1.0 - coverage) / count))


def summarize_coverage(
    n: int,
    attempted: int,
    replications: list[dict[str, float | int]],
    valid_bootstraps_total: int,
    requested_bootstraps_total: int,
    bootstrap_sum: float,
    bootstrap_sum_squares: float,
    bootstrap_count: int,
    design: Design,
    alpha: float,
) -> dict[str, float | int | str]:
    """Summarize dataset-conditional full-bootstrap coverage."""

    point_errors = np.asarray(
        [row["point_error"] for row in replications],
        dtype=float,
    )
    score_target_errors = np.asarray(
        [row["score_target_error"] for row in replications],
        dtype=float,
    )
    point_estimates = point_errors + design.att_true
    conditional_equal = np.asarray(
        [row["conditional_equal_cover"] for row in replications],
        dtype=float,
    )
    conditional_symmetric = np.asarray(
        [row["conditional_symmetric_cover"] for row in replications],
        dtype=float,
    )
    score_target_conditional_equal = np.asarray(
        [row["score_target_conditional_equal_cover"] for row in replications],
        dtype=float,
    )
    score_target_conditional_symmetric = np.asarray(
        [row["score_target_conditional_symmetric_cover"] for row in replications],
        dtype=float,
    )
    conditional_percentile = np.asarray(
        [row["conditional_percentile_cover"] for row in replications],
        dtype=float,
    )
    score_target_conditional_percentile = np.asarray(
        [
            row["score_target_conditional_percentile_cover"]
            for row in replications
        ],
        dtype=float,
    )
    conditional_normal = np.asarray(
        [row["conditional_normal_cover"] for row in replications],
        dtype=float,
    )
    score_target_conditional_normal = np.asarray(
        [row["score_target_conditional_normal_cover"] for row in replications],
        dtype=float,
    )
    conditional_lengths = np.asarray(
        [row["conditional_equal_length"] for row in replications],
        dtype=float,
    )

    equal_valid = np.isfinite(conditional_equal)
    symmetric_valid = np.isfinite(conditional_symmetric)
    conditional_equal_coverage = (
        float(np.mean(conditional_equal[equal_valid]))
        if np.any(equal_valid)
        else np.nan
    )
    conditional_symmetric_coverage = (
        float(np.mean(conditional_symmetric[symmetric_valid]))
        if np.any(symmetric_valid)
        else np.nan
    )
    score_equal_valid = np.isfinite(score_target_conditional_equal)
    score_symmetric_valid = np.isfinite(score_target_conditional_symmetric)
    percentile_valid = np.isfinite(conditional_percentile)
    score_percentile_valid = np.isfinite(
        score_target_conditional_percentile
    )
    normal_valid = np.isfinite(conditional_normal)
    score_normal_valid = np.isfinite(score_target_conditional_normal)
    score_target_conditional_equal_coverage = (
        float(np.mean(score_target_conditional_equal[score_equal_valid]))
        if np.any(score_equal_valid)
        else np.nan
    )
    score_target_conditional_symmetric_coverage = (
        float(
            np.mean(
                score_target_conditional_symmetric[score_symmetric_valid]
            )
        )
        if np.any(score_symmetric_valid)
        else np.nan
    )
    conditional_percentile_coverage = (
        float(np.mean(conditional_percentile[percentile_valid]))
        if np.any(percentile_valid)
        else np.nan
    )
    score_target_conditional_percentile_coverage = (
        float(
            np.mean(
                score_target_conditional_percentile[
                    score_percentile_valid
                ]
            )
        )
        if np.any(score_percentile_valid)
        else np.nan
    )
    conditional_normal_coverage = (
        float(np.mean(conditional_normal[normal_valid]))
        if np.any(normal_valid)
        else np.nan
    )
    score_target_conditional_normal_coverage = (
        float(np.mean(score_target_conditional_normal[score_normal_valid]))
        if np.any(score_normal_valid)
        else np.nan
    )
    equal_low, equal_high = _wilson_interval(
        conditional_equal_coverage,
        int(equal_valid.sum()),
    )
    score_equal_low, score_equal_high = _wilson_interval(
        score_target_conditional_equal_coverage,
        int(score_equal_valid.sum()),
    )
    nominal_coverage = 1.0 - alpha

    valid_points = np.isfinite(point_errors)
    if int(valid_points.sum()) > 1:
        empirical_sd = float(np.std(point_estimates[valid_points], ddof=1))
    else:
        empirical_sd = np.nan
    bias = (
        float(np.mean(point_errors[valid_points]))
        if np.any(valid_points)
        else np.nan
    )
    rmse = (
        float(np.sqrt(np.mean(point_errors[valid_points] ** 2)))
        if np.any(valid_points)
        else np.nan
    )
    score_target_bias = (
        float(np.mean(score_target_errors[valid_points]))
        if np.any(valid_points)
        else np.nan
    )
    score_target_rmse = (
        float(np.sqrt(np.mean(score_target_errors[valid_points] ** 2)))
        if np.any(valid_points)
        else np.nan
    )
    if bootstrap_count > 1:
        variance_numerator = (
            bootstrap_sum_squares
            - bootstrap_sum * bootstrap_sum / bootstrap_count
        )
        bootstrap_sd = float(
            math.sqrt(max(variance_numerator, 0.0) / (bootstrap_count - 1))
        )
    else:
        bootstrap_sd = np.nan
    bootstrap_sd_ratio = (
        float(bootstrap_sd / empirical_sd)
        if np.isfinite(bootstrap_sd)
        and np.isfinite(empirical_sd)
        and empirical_sd > 0.0
        else np.nan
    )

    def finite_values(key: str) -> np.ndarray:
        values = np.asarray([row[key] for row in replications], dtype=float)
        return values[np.isfinite(values)]

    sample_kappa_medians = finite_values("sample_kappa_m_median")
    sample_kappa_p95s = finite_values("sample_kappa_m_p95")
    sample_kappa_maxima = finite_values("sample_kappa_m_max")
    sample_sigma_max = finite_values("sample_sigma_max_median")
    sample_sigma_middle = finite_values("sample_sigma_middle_median")
    sample_sigma_min = finite_values("sample_sigma_min_median")
    sample_rank_minima = finite_values("sample_rank_min")
    rank_deficient_shares = finite_values("sample_rank_deficient_share")
    maximum_weights = finite_values("maximum_estimated_weight")
    estimated_weight_l2_values = finite_values(
        "estimated_weight_l2_median"
    )
    raw_weight_l2_values = finite_values("raw_weight_l2_median")
    projection_distance_values = finite_values(
        "simplex_projection_distance_median"
    )
    boundary_fraction_values = finite_values(
        "simplex_boundary_fraction"
    )
    projected_weight_error_values = finite_values(
        "projected_weight_error_l2_median"
    )
    raw_weight_error_values = finite_values(
        "raw_weight_error_l2_median"
    )
    pre_fit_l2_values = finite_values("pre_fit_l2_median")
    pre_fit_max_values = finite_values("pre_fit_max")
    within_root_sds = finite_values("bootstrap_root_sd")
    realized_condition = (
        float(np.median(sample_kappa_p95s))
        if sample_kappa_p95s.size
        else np.nan
    )
    condition_ratio = (
        float(realized_condition / CALIBRATION_TARGET_KAPPA_M)
        if np.isfinite(realized_condition)
        else np.nan
    )
    minimum_sample_rank = (
        int(np.min(sample_rank_minima))
        if sample_rank_minima.size
        else 0
    )
    maximum_rank_deficient_share = (
        float(np.max(rank_deficient_shares))
        if rank_deficient_shares.size
        else np.nan
    )
    condition_comparison_eligible = bool(
        sample_kappa_p95s.size >= math.ceil(0.99 * attempted)
        and minimum_sample_rank == N_DONORS - 1
        and maximum_rank_deficient_share == 0.0
    )
    condition_regime = (
        "at_or_below_empirical"
        if design.kappa_m <= CALIBRATION_TARGET_KAPPA_M
        else "above_empirical"
    )

    return {
        "dgp_version": DGP_VERSION,
        "dgp_name": "sc_only",
        "donor_codes": "|".join(str(code) for code in DONOR_CODES),
        "donor_states": DONOR_STATES,
        "baseline_donor_code": BASELINE_DONOR_CODE,
        "baseline_donor_state": BASELINE_DONOR_STATE,
        "condition_definition": CONDITION_DEFINITION,
        "calibration_target_source": CALIBRATION_TARGET_SOURCE,
        "calibration_target_donor_states": (
            CALIBRATION_TARGET_DONOR_STATES
        ),
        "calibration_target_kappa_m": CALIBRATION_TARGET_KAPPA_M,
        "calibration_target_kappa_mtm": CALIBRATION_TARGET_KAPPA_MTM,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "kappa_m": float(design.kappa_m),
        "kappa_mtm_derived": float(design.kappa_m**2),
        "inference_method": (
            "full_nested_exponential_multiplier_bootstrap_fixed_folds"
        ),
        "weight_estimator": WEIGHT_ESTIMATOR,
        "cross_fitting_folds": int(N_FOLDS),
        "fold_behavior": "fixed_within_outer_dataset",
        "post_scale": float(design.post_scale),
        "reference_augmentation_gamma": float(
            REFERENCE_AUGMENTATION_GAMMA
        ),
        "alpha": float(alpha),
        "confidence_level": float(1.0 - alpha),
        "total_periods": int(N_PERIODS),
        "pre_periods": int(N_PRE),
        "post_periods": 1,
        "n": int(n),
        "condition_regime": condition_regime,
        "condition_comparison_eligible": condition_comparison_eligible,
        "realized_median_of_sample_kappa_m_p95": realized_condition,
        "realized_condition_ratio_to_empirical": condition_ratio,
        "target_condition_ratio_to_empirical": float(
            design.kappa_m / CALIBRATION_TARGET_KAPPA_M
        ),
        "realized_condition_log_ratio_to_empirical": (
            float(math.log(condition_ratio))
            if np.isfinite(condition_ratio) and condition_ratio > 0.0
            else np.nan
        ),
        "condition_diagnostic_valid": int(sample_kappa_p95s.size),
        "median_sample_kappa_m_median": (
            float(np.median(sample_kappa_medians))
            if sample_kappa_medians.size
            else np.nan
        ),
        "median_sample_kappa_m_max": (
            float(np.median(sample_kappa_maxima))
            if sample_kappa_maxima.size
            else np.nan
        ),
        "median_sample_sigma_max": (
            float(np.median(sample_sigma_max))
            if sample_sigma_max.size
            else np.nan
        ),
        "median_sample_sigma_middle": (
            float(np.median(sample_sigma_middle))
            if sample_sigma_middle.size
            else np.nan
        ),
        "median_sample_sigma_min": (
            float(np.median(sample_sigma_min))
            if sample_sigma_min.size
            else np.nan
        ),
        "minimum_sample_rank": (
            minimum_sample_rank
        ),
        "maximum_sample_rank_deficient_share": (
            maximum_rank_deficient_share
        ),
        "maximum_estimated_weight_p95": (
            float(np.quantile(maximum_weights, 0.95))
            if maximum_weights.size
            else np.nan
        ),
        "estimated_weight_l2_median": (
            float(np.median(estimated_weight_l2_values))
            if estimated_weight_l2_values.size
            else np.nan
        ),
        "raw_weight_l2_median": (
            float(np.median(raw_weight_l2_values))
            if raw_weight_l2_values.size
            else np.nan
        ),
        "simplex_projection_distance_median": (
            float(np.median(projection_distance_values))
            if projection_distance_values.size
            else np.nan
        ),
        "simplex_boundary_fraction_median": (
            float(np.median(boundary_fraction_values))
            if boundary_fraction_values.size
            else np.nan
        ),
        "projected_weight_error_l2_median": (
            float(np.median(projected_weight_error_values))
            if projected_weight_error_values.size
            else np.nan
        ),
        "raw_weight_error_l2_median": (
            float(np.median(raw_weight_error_values))
            if raw_weight_error_values.size
            else np.nan
        ),
        "pre_fit_l2_median": (
            float(np.median(pre_fit_l2_values))
            if pre_fit_l2_values.size
            else np.nan
        ),
        "pre_fit_max_p95": (
            float(np.quantile(pre_fit_max_values, 0.95))
            if pre_fit_max_values.size
            else np.nan
        ),
        "attempted_outer_replications": int(attempted),
        "valid_point_estimates": int(valid_points.sum()),
        "point_failures": int(attempted - valid_points.sum()),
        "requested_bootstraps": int(requested_bootstraps_total),
        "valid_bootstraps": int(valid_bootstraps_total),
        "bootstrap_failures": int(
            requested_bootstraps_total - valid_bootstraps_total
        ),
        "att_true": float(design.att_true),
        "bias": bias,
        "score_target_bias": score_target_bias,
        "empirical_sd": empirical_sd,
        "rmse": rmse,
        "score_target_rmse": score_target_rmse,
        "pooled_bootstrap_error_sd": bootstrap_sd,
        "pooled_bootstrap_sd_over_empirical_sd": bootstrap_sd_ratio,
        "mean_within_bootstrap_root_sd": (
            float(np.mean(within_root_sds))
            if within_root_sds.size
            else np.nan
        ),
        "mean_within_root_sd_over_empirical_sd": (
            float(np.mean(within_root_sds) / empirical_sd)
            if within_root_sds.size
            and np.isfinite(empirical_sd)
            and empirical_sd > 0.0
            else np.nan
        ),
        "conditional_equal_tail_valid": int(equal_valid.sum()),
        "conditional_equal_tail_coverage": conditional_equal_coverage,
        "conditional_equal_tail_minus_nominal": (
            float(conditional_equal_coverage - (1.0 - alpha))
            if np.isfinite(conditional_equal_coverage)
            else np.nan
        ),
        "conditional_equal_tail_mcse": _coverage_mcse(
            conditional_equal_coverage,
            int(equal_valid.sum()),
        ),
        "conditional_equal_tail_wilson_low": equal_low,
        "conditional_equal_tail_wilson_high": equal_high,
        "conditional_equal_tail_nominal_inside_wilson": bool(
            np.isfinite(equal_low)
            and equal_low <= nominal_coverage <= equal_high
        ),
        "conditional_equal_tail_wilson_high_below_nominal": bool(
            np.isfinite(equal_high) and equal_high < nominal_coverage
        ),
        "conditional_equal_tail_mean_length": (
            float(np.nanmean(conditional_lengths))
            if np.any(np.isfinite(conditional_lengths))
            else np.nan
        ),
        "conditional_symmetric_valid": int(symmetric_valid.sum()),
        "conditional_symmetric_coverage": conditional_symmetric_coverage,
        "conditional_symmetric_mcse": _coverage_mcse(
            conditional_symmetric_coverage,
            int(symmetric_valid.sum()),
        ),
        "score_target_conditional_equal_tail_valid": int(
            score_equal_valid.sum()
        ),
        "score_target_conditional_equal_tail_coverage": (
            score_target_conditional_equal_coverage
        ),
        "score_target_conditional_equal_tail_minus_nominal": (
            float(
                score_target_conditional_equal_coverage
                - (1.0 - alpha)
            )
            if np.isfinite(score_target_conditional_equal_coverage)
            else np.nan
        ),
        "score_target_conditional_equal_tail_mcse": _coverage_mcse(
            score_target_conditional_equal_coverage,
            int(score_equal_valid.sum()),
        ),
        "score_target_conditional_equal_tail_wilson_low": score_equal_low,
        "score_target_conditional_equal_tail_wilson_high": score_equal_high,
        "score_target_equal_tail_nominal_inside_wilson": bool(
            np.isfinite(score_equal_low)
            and score_equal_low <= nominal_coverage <= score_equal_high
        ),
        "score_target_equal_tail_wilson_high_below_nominal": bool(
            np.isfinite(score_equal_high)
            and score_equal_high < nominal_coverage
        ),
        "score_target_conditional_symmetric_valid": int(
            score_symmetric_valid.sum()
        ),
        "score_target_conditional_symmetric_coverage": (
            score_target_conditional_symmetric_coverage
        ),
        "score_target_conditional_symmetric_mcse": _coverage_mcse(
            score_target_conditional_symmetric_coverage,
            int(score_symmetric_valid.sum()),
        ),
        "conditional_percentile_valid": int(percentile_valid.sum()),
        "conditional_percentile_coverage": conditional_percentile_coverage,
        "conditional_percentile_mcse": _coverage_mcse(
            conditional_percentile_coverage,
            int(percentile_valid.sum()),
        ),
        "score_target_conditional_percentile_valid": int(
            score_percentile_valid.sum()
        ),
        "score_target_conditional_percentile_coverage": (
            score_target_conditional_percentile_coverage
        ),
        "score_target_conditional_percentile_mcse": _coverage_mcse(
            score_target_conditional_percentile_coverage,
            int(score_percentile_valid.sum()),
        ),
        "conditional_normal_valid": int(normal_valid.sum()),
        "conditional_normal_coverage": conditional_normal_coverage,
        "conditional_normal_mcse": _coverage_mcse(
            conditional_normal_coverage,
            int(normal_valid.sum()),
        ),
        "score_target_conditional_normal_valid": int(
            score_normal_valid.sum()
        ),
        "score_target_conditional_normal_coverage": (
            score_target_conditional_normal_coverage
        ),
        "score_target_conditional_normal_mcse": _coverage_mcse(
            score_target_conditional_normal_coverage,
            int(score_normal_valid.sum()),
        ),
        "population_kappa_m_median": float(
            design.population_kappa_m_median
        ),
        "population_kappa_m_p95": float(
            design.population_kappa_m_p95
        ),
        "population_kappa_mtm_median": float(
            design.population_kappa_mtm_median
        ),
        "population_kappa_mtm_p95": float(
            design.population_kappa_mtm_p95
        ),
        "population_sigma_max_median": float(
            design.population_sigma_max_median
        ),
        "population_sigma_middle_median": float(
            design.population_sigma_middle_median
        ),
        "population_sigma_min_median": float(
            design.population_sigma_min_median
        ),
        "population_rank_min": int(design.population_rank_min),
        "population_matrix_rows": int(N_PRE),
        "population_matrix_columns": int(N_DONORS - 1),
        "free_donor_weights": int(N_DONORS - 1),
        "excess_pre_restrictions": int(N_PRE - (N_DONORS - 1)),
        "population_trim_normalized_score_target": float(
            design.population_trim_normalized_score_target
        ),
        "population_score_target_minus_att": float(
            design.population_trim_normalized_score_target - design.att_true
        ),
        "maximum_sc_residual": float(design.maximum_sc_residual),
        "maximum_pt_gap": float(design.maximum_pt_gap),
        "weak_post_projection_median": float(
            design.weak_post_projection_median
        ),
        "weak_post_projection_over_sigma_min_median": float(
            design.weak_post_projection_over_sigma_min_median
        ),
    }


def _write_summaries(
    path: Path,
    summaries: list[dict[str, float | int | str]],
) -> None:
    """Write all kappa_m-cell coverage summaries to CSV."""

    if not summaries:
        raise ValueError("cannot write an empty coverage summary")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def run_one_sample(
    args: argparse.Namespace,
    n: int,
    output_prefix: Path,
) -> SampleOutput:
    """Run one sample size and stream all replication results to CSV."""

    kappa_m_grid = tuple(float(kappa_m) for kappa_m in args.kappa_m_grid)
    jobs = available_cpu_count() if args.jobs == 0 else int(args.jobs)
    jobs = min(jobs, int(args.outer_reps))
    if jobs <= 0:
        raise ValueError("worker count must be positive")
    (
        outer_path,
        bootstrap_path,
        coverage_path,
        population_path,
    ) = output_paths(output_prefix)
    outer_path.parent.mkdir(parents=True, exist_ok=True)
    if any(
        path.exists()
        for path in (
            outer_path,
            bootstrap_path,
            coverage_path,
            population_path,
        )
    ):
        raise FileExistsError(
            "refusing to overwrite existing output; choose another --output-prefix"
        )

    designs = {
        kappa_m: build_design(kappa_m, args.post_scale)
        for kappa_m in kappa_m_grid
    }
    write_population_diagnostics(population_path, designs)
    tasks = [
        (
            replication,
            int(n),
            int(args.bootstraps),
            int(args.master_seed),
        )
        for replication in range(
            int(args.outer_start),
            int(args.outer_start) + int(args.outer_reps),
        )
    ]
    total_cells = len(tasks) * len(kappa_m_grid)
    outer_header = [
        "dgp_version",
        "dgp_name",
        "donor_codes",
        "donor_states",
        "baseline_donor_code",
        "baseline_donor_state",
        "condition_definition",
        "calibration_target_source",
        "calibration_target_donor_states",
        "calibration_target_kappa_m",
        "calibration_target_kappa_mtm",
        "experiment_purpose",
        "alpha",
        "confidence_level",
        "total_periods",
        "pre_periods",
        "post_periods",
        "n",
        "outer_replication",
        "master_seed",
        "base_seed_hex",
        "fold_hash",
        "kappa_m",
        "kappa_mtm_derived",
        "post_scale",
        "inference_method",
        "weight_estimator",
        "population_kappa_m_median",
        "population_kappa_m_p95",
        "population_sigma_max_median",
        "population_sigma_middle_median",
        "population_sigma_min_median",
        "population_rank_min",
        "sample_kappa_m_median",
        "sample_kappa_m_p95",
        "sample_kappa_m_max",
        "sample_sigma_max_median",
        "sample_sigma_middle_median",
        "sample_sigma_min_median",
        "sample_rank_min",
        "sample_rank_deficient_share",
        "maximum_estimated_weight",
        "estimated_weight_l2_median",
        "raw_weight_l2_median",
        "simplex_projection_distance_median",
        "simplex_boundary_fraction",
        "projected_weight_error_l2_median",
        "raw_weight_error_l2_median",
        "pre_fit_l2_median",
        "pre_fit_max",
        "population_target_at_or_below_empirical",
        "att_true",
        "population_trim_normalized_score_target",
        "point_estimate",
        "estimation_error",
        "estimation_error_vs_score_target",
        "conditional_q_alpha_over_2",
        "conditional_q_one_minus_alpha_over_2",
        "conditional_equal_tail_ci_low",
        "conditional_equal_tail_ci_high",
        "conditional_equal_tail_covers",
        "conditional_equal_tail_covers_score_target",
        "conditional_abs_q_one_minus_alpha",
        "conditional_symmetric_ci_low",
        "conditional_symmetric_ci_high",
        "conditional_symmetric_covers",
        "conditional_symmetric_covers_score_target",
        "conditional_percentile_ci_low",
        "conditional_percentile_ci_high",
        "conditional_percentile_covers",
        "conditional_percentile_covers_score_target",
        "bootstrap_root_sd",
        "conditional_normal_ci_low",
        "conditional_normal_ci_high",
        "conditional_normal_covers",
        "conditional_normal_covers_score_target",
        "requested_bootstraps",
        "valid_bootstraps",
        "bootstrap_failures",
        "status",
        "error_message",
        "runtime_seconds",
    ]
    bootstrap_header = [
        "dgp_version",
        "dgp_name",
        "donor_codes",
        "baseline_donor_code",
        "kappa_m",
        "kappa_mtm",
        "total_periods",
        "n",
        "outer_replication",
        "base_seed_hex",
        "fold_hash",
        "requested_bootstraps",
        "valid_bootstraps",
        "bootstrap_failures",
        "bootstrap_root_mean",
        "bootstrap_root_sd",
        "status",
        "failure_messages",
        "runtime_seconds",
    ]

    started = time.perf_counter()
    valid_bootstraps_total = {kappa_m: 0 for kappa_m in kappa_m_grid}
    requested_bootstraps_total = {kappa_m: 0 for kappa_m in kappa_m_grid}
    bootstrap_sum = {kappa_m: 0.0 for kappa_m in kappa_m_grid}
    bootstrap_sum_squares = {kappa_m: 0.0 for kappa_m in kappa_m_grid}
    bootstrap_count = {kappa_m: 0 for kappa_m in kappa_m_grid}
    cell_worker_runtime = {kappa_m: 0.0 for kappa_m in kappa_m_grid}
    replication_summaries: dict[
        float,
        list[dict[str, float | int]],
    ] = {kappa_m: [] for kappa_m in kappa_m_grid}
    next_progress_percent = 10
    with (
        outer_path.open("x", encoding="utf-8", newline="") as outer_handle,
        bootstrap_path.open(
            "x",
            encoding="utf-8",
            newline="",
        ) as bootstrap_handle,
    ):
        outer_writer = csv.writer(outer_handle)
        bootstrap_writer = csv.writer(bootstrap_handle)
        outer_writer.writerow(outer_header)
        bootstrap_writer.writerow(bootstrap_header)
        context = get_context("spawn")
        with context.Pool(
            processes=jobs,
            initializer=initialize_worker,
            initargs=(kappa_m_grid, float(args.post_scale)),
        ) as pool:
            batches = pool.imap_unordered(
                run_replication_grid_task,
                tasks,
                chunksize=1,
            )
            results = (
                result
                for result_batch in batches
                for result in result_batch
            )
            for completed, result in enumerate(results, start=1):
                kappa_m = float(result.kappa_m)
                design = designs[kappa_m]
                point_valid = math.isfinite(result.point_estimate)
                finite_bootstraps = np.isfinite(result.bootstrap_errors)
                valid_bootstraps = int(finite_bootstraps.sum())
                valid_bootstraps_total[kappa_m] += valid_bootstraps
                requested_bootstraps_total[kappa_m] += int(args.bootstraps)
                cell_worker_runtime[kappa_m] += float(result.runtime_seconds)
                finite_errors = result.bootstrap_errors[finite_bootstraps]
                if finite_errors.size:
                    bootstrap_sum[kappa_m] += float(np.sum(finite_errors))
                    bootstrap_sum_squares[kappa_m] += float(
                        finite_errors @ finite_errors
                    )
                    bootstrap_count[kappa_m] += int(finite_errors.size)

                point_error = (
                    result.point_estimate - design.att_true
                    if point_valid
                    else np.nan
                )
                score_target_error = (
                    result.point_estimate
                    - design.population_trim_normalized_score_target
                    if point_valid
                    else np.nan
                )
                conditional_q025 = np.nan
                conditional_q975 = np.nan
                equal_ci_low = np.nan
                equal_ci_high = np.nan
                equal_cover = np.nan
                absolute_q95 = np.nan
                symmetric_ci_low = np.nan
                symmetric_ci_high = np.nan
                symmetric_cover = np.nan
                score_target_equal_cover = np.nan
                score_target_symmetric_cover = np.nan
                percentile_ci_low = np.nan
                percentile_ci_high = np.nan
                percentile_cover = np.nan
                score_target_percentile_cover = np.nan
                bootstrap_root_sd = np.nan
                normal_ci_low = np.nan
                normal_ci_high = np.nan
                normal_cover = np.nan
                score_target_normal_cover = np.nan
                bootstrap_complete = (
                    point_valid
                    and valid_bootstraps == int(args.bootstraps)
                )
                if bootstrap_complete:
                    conditional_q025, conditional_q975 = np.quantile(
                        finite_errors,
                        [args.alpha / 2.0, 1.0 - args.alpha / 2.0],
                    )
                    equal_ci_low = result.point_estimate - conditional_q975
                    equal_ci_high = result.point_estimate - conditional_q025
                    equal_cover = float(
                        equal_ci_low <= design.att_true <= equal_ci_high
                    )
                    score_target_equal_cover = float(
                        equal_ci_low
                        <= design.population_trim_normalized_score_target
                        <= equal_ci_high
                    )
                    absolute_q95 = float(
                        np.quantile(np.abs(finite_errors), 1.0 - args.alpha)
                    )
                    symmetric_ci_low = result.point_estimate - absolute_q95
                    symmetric_ci_high = result.point_estimate + absolute_q95
                    symmetric_cover = float(
                        symmetric_ci_low <= design.att_true <= symmetric_ci_high
                    )
                    score_target_symmetric_cover = float(
                        symmetric_ci_low
                        <= design.population_trim_normalized_score_target
                        <= symmetric_ci_high
                    )
                    percentile_ci_low = (
                        result.point_estimate + conditional_q025
                    )
                    percentile_ci_high = (
                        result.point_estimate + conditional_q975
                    )
                    percentile_cover = float(
                        percentile_ci_low
                        <= design.att_true
                        <= percentile_ci_high
                    )
                    score_target_percentile_cover = float(
                        percentile_ci_low
                        <= design.population_trim_normalized_score_target
                        <= percentile_ci_high
                    )
                    bootstrap_root_sd = float(
                        np.std(finite_errors, ddof=1)
                    )
                    normal_critical = float(
                        NormalDist().inv_cdf(1.0 - args.alpha / 2.0)
                    )
                    normal_ci_low = (
                        result.point_estimate
                        - normal_critical * bootstrap_root_sd
                    )
                    normal_ci_high = (
                        result.point_estimate
                        + normal_critical * bootstrap_root_sd
                    )
                    normal_cover = float(
                        normal_ci_low <= design.att_true <= normal_ci_high
                    )
                    score_target_normal_cover = float(
                        normal_ci_low
                        <= design.population_trim_normalized_score_target
                        <= normal_ci_high
                    )
                replication_summaries[kappa_m].append(
                    {
                        "replication": int(result.replication),
                        "point_error": float(point_error),
                        "score_target_error": float(score_target_error),
                        "conditional_equal_cover": float(equal_cover),
                        "conditional_symmetric_cover": float(symmetric_cover),
                        "score_target_conditional_equal_cover": float(
                            score_target_equal_cover
                        ),
                        "score_target_conditional_symmetric_cover": float(
                            score_target_symmetric_cover
                        ),
                        "conditional_percentile_cover": float(
                            percentile_cover
                        ),
                        "score_target_conditional_percentile_cover": float(
                            score_target_percentile_cover
                        ),
                        "conditional_normal_cover": float(normal_cover),
                        "score_target_conditional_normal_cover": float(
                            score_target_normal_cover
                        ),
                        "conditional_percentile_length": float(
                            percentile_ci_high - percentile_ci_low
                        ),
                        "conditional_normal_length": float(
                            normal_ci_high - normal_ci_low
                        ),
                        "bootstrap_root_sd": float(bootstrap_root_sd),
                        "conditional_equal_length": float(
                            equal_ci_high - equal_ci_low
                        ),
                        "sample_kappa_m_median": float(
                            result.sample_kappa_m_median
                        ),
                        "sample_kappa_m_p95": float(
                            result.sample_kappa_m_p95
                        ),
                        "sample_kappa_m_max": float(
                            result.sample_kappa_m_max
                        ),
                        "sample_sigma_max_median": float(
                            result.sample_sigma_max_median
                        ),
                        "sample_sigma_middle_median": float(
                            result.sample_sigma_middle_median
                        ),
                        "sample_sigma_min_median": float(
                            result.sample_sigma_min_median
                        ),
                        "sample_rank_min": int(result.sample_rank_min),
                        "sample_rank_deficient_share": float(
                            result.sample_rank_deficient_share
                        ),
                        "maximum_estimated_weight": float(
                            result.maximum_estimated_weight
                        ),
                        "estimated_weight_l2_median": float(
                            result.estimated_weight_l2_median
                        ),
                        "raw_weight_l2_median": float(
                            result.raw_weight_l2_median
                        ),
                        "simplex_projection_distance_median": float(
                            result.simplex_projection_distance_median
                        ),
                        "simplex_boundary_fraction": float(
                            result.simplex_boundary_fraction
                        ),
                        "projected_weight_error_l2_median": float(
                            result.projected_weight_error_l2_median
                        ),
                        "raw_weight_error_l2_median": float(
                            result.raw_weight_error_l2_median
                        ),
                        "pre_fit_l2_median": float(
                            result.pre_fit_l2_median
                        ),
                        "pre_fit_max": float(result.pre_fit_max),
                    }
                )

                if not point_valid:
                    outer_status = "point_failed"
                    outer_error = result.point_failure
                elif not bootstrap_complete:
                    outer_status = "bootstrap_incomplete"
                    outer_error = (
                        f"{int(args.bootstraps) - valid_bootstraps} "
                        "bootstrap draw(s) failed; see diagnostics CSV"
                    )
                else:
                    outer_status = "valid"
                    outer_error = ""

                outer_writer.writerow(
                    [
                        DGP_VERSION,
                        "sc_only",
                        "|".join(str(code) for code in DONOR_CODES),
                        DONOR_STATES,
                        BASELINE_DONOR_CODE,
                        BASELINE_DONOR_STATE,
                        CONDITION_DEFINITION,
                        CALIBRATION_TARGET_SOURCE,
                        CALIBRATION_TARGET_DONOR_STATES,
                        CALIBRATION_TARGET_KAPPA_M,
                        CALIBRATION_TARGET_KAPPA_MTM,
                        EXPERIMENT_PURPOSE,
                        float(args.alpha),
                        float(1.0 - args.alpha),
                        N_PERIODS,
                        N_PRE,
                        1,
                        int(n),
                        result.replication,
                        int(args.master_seed),
                        f"0x{result.base_seed:016x}",
                        result.fold_hash,
                        kappa_m,
                        kappa_m**2,
                        design.post_scale,
                        (
                            "full_nested_exponential_multiplier_bootstrap_"
                            "fixed_folds"
                        ),
                        WEIGHT_ESTIMATOR,
                        design.population_kappa_m_median,
                        design.population_kappa_m_p95,
                        design.population_sigma_max_median,
                        design.population_sigma_middle_median,
                        design.population_sigma_min_median,
                        design.population_rank_min,
                        result.sample_kappa_m_median,
                        result.sample_kappa_m_p95,
                        result.sample_kappa_m_max,
                        result.sample_sigma_max_median,
                        result.sample_sigma_middle_median,
                        result.sample_sigma_min_median,
                        result.sample_rank_min,
                        result.sample_rank_deficient_share,
                        result.maximum_estimated_weight,
                        result.estimated_weight_l2_median,
                        result.raw_weight_l2_median,
                        result.simplex_projection_distance_median,
                        result.simplex_boundary_fraction,
                        result.projected_weight_error_l2_median,
                        result.raw_weight_error_l2_median,
                        result.pre_fit_l2_median,
                        result.pre_fit_max,
                        bool(kappa_m <= CALIBRATION_TARGET_KAPPA_M),
                        design.att_true,
                        design.population_trim_normalized_score_target,
                        result.point_estimate,
                        point_error,
                        score_target_error,
                        conditional_q025,
                        conditional_q975,
                        equal_ci_low,
                        equal_ci_high,
                        equal_cover,
                        score_target_equal_cover,
                        absolute_q95,
                        symmetric_ci_low,
                        symmetric_ci_high,
                        symmetric_cover,
                        score_target_symmetric_cover,
                        percentile_ci_low,
                        percentile_ci_high,
                        percentile_cover,
                        score_target_percentile_cover,
                        bootstrap_root_sd,
                        normal_ci_low,
                        normal_ci_high,
                        normal_cover,
                        score_target_normal_cover,
                        int(args.bootstraps),
                        valid_bootstraps,
                        int(args.bootstraps) - valid_bootstraps,
                        outer_status,
                        outer_error,
                        result.runtime_seconds,
                    ]
                )
                bootstrap_writer.writerow(
                    [
                        DGP_VERSION,
                        "sc_only",
                        "|".join(str(code) for code in DONOR_CODES),
                        BASELINE_DONOR_CODE,
                        kappa_m,
                        kappa_m**2,
                        N_PERIODS,
                        int(n),
                        result.replication,
                        f"0x{result.base_seed:016x}",
                        result.fold_hash,
                        int(args.bootstraps),
                        valid_bootstraps,
                        int(args.bootstraps) - valid_bootstraps,
                        (
                            float(np.mean(finite_errors))
                            if finite_errors.size
                            else np.nan
                        ),
                        (
                            float(np.std(finite_errors, ddof=1))
                            if finite_errors.size > 1
                            else np.nan
                        ),
                        outer_status,
                        json.dumps(
                            result.bootstrap_failures,
                            sort_keys=True,
                        ),
                        result.runtime_seconds,
                    ]
                )
                completed_percent = int(
                    math.floor(100.0 * completed / total_cells)
                )
                while completed_percent >= next_progress_percent:
                    reported_percent = next_progress_percent
                    elapsed = time.perf_counter() - started
                    valid_all = int(sum(valid_bootstraps_total.values()))
                    requested_all = int(
                        sum(requested_bootstraps_total.values())
                    )
                    provisional_parts: list[str] = []
                    for grid_kappa_m in kappa_m_grid:
                        rows = replication_summaries[grid_kappa_m]
                        conditions = np.asarray(
                            [row["sample_kappa_m_p95"] for row in rows],
                            dtype=float,
                        )
                        covers = np.asarray(
                            [
                                row[
                                    "score_target_conditional_equal_cover"
                                ]
                                for row in rows
                            ],
                            dtype=float,
                        )
                        conditions = conditions[np.isfinite(conditions)]
                        covers = covers[np.isfinite(covers)]
                        if conditions.size and covers.size:
                            provisional_parts.append(
                                f"kappa={grid_kappa_m:g}:"
                                f"k={np.median(conditions):.3f},"
                                f"cov={np.mean(covers):.3f},"
                                f"R={covers.size}"
                            )
                    print(
                        f"n={n} progress={reported_percent}% "
                        f"completed_cells={completed}/{total_cells} "
                        f"valid_bootstraps={valid_all}/{requested_all} "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                    print(
                        "provisional trim-score equal-tail: "
                        + " | ".join(provisional_parts),
                        flush=True,
                    )
                    outer_handle.flush()
                    bootstrap_handle.flush()
                    next_progress_percent = reported_percent + 10

    wall_runtime = float(time.perf_counter() - started)
    summaries: list[dict[str, float | int | str]] = []
    for kappa_m in kappa_m_grid:
        summary = summarize_coverage(
            n=n,
            attempted=int(args.outer_reps),
            replications=replication_summaries[kappa_m],
            valid_bootstraps_total=valid_bootstraps_total[kappa_m],
            requested_bootstraps_total=requested_bootstraps_total[kappa_m],
            bootstrap_sum=bootstrap_sum[kappa_m],
            bootstrap_sum_squares=bootstrap_sum_squares[kappa_m],
            bootstrap_count=bootstrap_count[kappa_m],
            design=designs[kappa_m],
            alpha=float(args.alpha),
        )
        summary["outer_start"] = int(args.outer_start)
        summary["outer_stop_exclusive"] = int(
            args.outer_start + args.outer_reps
        )
        summary["master_seed"] = int(args.master_seed)
        summary["cell_worker_runtime_seconds"] = float(
            cell_worker_runtime[kappa_m]
        )
        summary["sample_grid_wall_runtime_seconds"] = wall_runtime
        summaries.append(summary)
    _write_summaries(coverage_path, summaries)
    return SampleOutput(
        n=n,
        outer_path=outer_path,
        bootstrap_path=bootstrap_path,
        coverage_path=coverage_path,
        population_path=population_path,
        summaries=summaries,
    )


def run(args: argparse.Namespace) -> list[SampleOutput]:
    """Run every requested sample size sequentially."""

    planned = [
        (int(n), output_prefix_for_n(args, int(n)))
        for n in args.n
    ]
    existing_paths = [
        path
        for _, prefix in planned
        for path in output_paths(prefix)
        if path.exists()
    ]
    if existing_paths:
        formatted = "\n".join(f"  {path}" for path in existing_paths)
        raise FileExistsError(
            "refusing to overwrite existing output:\n"
            f"{formatted}\n"
            "choose another --output-prefix or move the existing files"
        )

    jobs = available_cpu_count() if args.jobs == 0 else int(args.jobs)
    jobs = min(jobs, int(args.outer_reps))
    outputs: list[SampleOutput] = []
    for index, (n, prefix) in enumerate(planned, start=1):
        print(
            f"Starting sample size {index}/{len(planned)}: "
            f"n={n}, R={args.outer_reps}, B={args.bootstraps}, "
            f"kappa_m_points={len(args.kappa_m_grid)}, workers={jobs}",
            flush=True,
        )
        output = run_one_sample(args, n, prefix)
        outputs.append(output)
        print(f"Completed n={n} outer CSV: {output.outer_path}", flush=True)
        print(
            f"Completed n={n} bootstrap diagnostics: {output.bootstrap_path}",
            flush=True,
        )
        print(
            f"Completed n={n} coverage CSV: {output.coverage_path}",
            flush=True,
        )
        print(
            f"Completed n={n} population CSV: {output.population_path}",
            flush=True,
        )
    return outputs


def print_final_coverage(outputs: list[SampleOutput]) -> None:
    """Print condition and coverage results after all cells finish."""

    print("\nFinal coverage-by-condition results", flush=True)
    header = (
        "n  kappa_m  sample_kappa_p95  sample_p95/empirical  "
        "target_regime  valid/R  "
        "trim_equal_tail(MCSE)  nominal_in_Wilson  "
        "Wilson_high_below_nominal  full_ATT_equal_tail(MCSE)  "
        "trim_symmetric"
    )
    print(header, flush=True)
    for output in outputs:
        for summary in output.summaries:
            print(
                f"{output.n}  {float(summary['kappa_m']):g}  "
                f"{float(summary['realized_median_of_sample_kappa_m_p95']):.6f}  "
                f"{float(summary['realized_condition_ratio_to_empirical']):.4f}  "
                f"{summary['condition_regime']}  "
                f"{int(summary['score_target_conditional_equal_tail_valid'])}/"
                f"{int(summary['attempted_outer_replications'])}  "
                f"{float(summary['score_target_conditional_equal_tail_coverage']):.6f}"
                f"({float(summary['score_target_conditional_equal_tail_mcse']):.6f})  "
                f"{summary['score_target_equal_tail_nominal_inside_wilson']}  "
                f"{summary['score_target_equal_tail_wilson_high_below_nominal']}  "
                f"{float(summary['conditional_equal_tail_coverage']):.6f}"
                f"({float(summary['conditional_equal_tail_mcse']):.6f})  "
                f"{float(summary['score_target_conditional_symmetric_coverage']):.6f}",
                flush=True,
            )


def audit_population_grid(args: argparse.Namespace) -> dict[str, object]:
    """Validate the embedded Quantile-top-4 calibration before simulation."""

    if DONOR_CODES != (24, 33, 49, 51):
        raise AssertionError("donor order must be Maryland, NH, Utah, Virginia")
    if BASELINE_DONOR_CODE != DONOR_CODES[-1]:
        raise AssertionError("Virginia must be the affine baseline donor")
    if not math.isclose(
        CALIBRATION_TARGET_KAPPA_MTM,
        CALIBRATION_TARGET_KAPPA_M**2,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("direct and Gram benchmark conditions disagree")

    reference = build_design(
        CALIBRATION_TARGET_KAPPA_M,
        float(args.post_scale),
    )
    validate_population_design(reference)
    reference_rows = population_diagnostics(reference)
    reference_singular = np.asarray(
        [
            [
                row["sigma_max_m"],
                row["sigma_middle_m"],
                row["sigma_min_m"],
            ]
            for row in reference_rows
        ],
        dtype=float,
    )
    quantiles = np.asarray([0.0, 0.05, 0.5, 0.95, 1.0])
    reference_spectral_quantiles = np.quantile(
        reference_singular,
        quantiles,
        axis=0,
    )
    expected_sigma_max = np.asarray(
        [14.073915, 14.073920, 14.074084, 14.077238, 14.078930]
    )
    expected_sigma_min = np.asarray(
        [1.076770, 1.076771, 1.076783, 1.077025, 1.077154]
    )
    if not np.allclose(
        reference_spectral_quantiles[:, 0],
        expected_sigma_max,
        rtol=0.0,
        atol=7e-7,
    ):
        raise AssertionError("Quantile-top-4 strong singular scale changed")
    if not np.allclose(
        reference_spectral_quantiles[:, 2],
        expected_sigma_min,
        rtol=0.0,
        atol=7e-7,
    ):
        raise AssertionError("Quantile-top-4 weak singular scale changed")
    if not math.isclose(reference.att_true, 1.011932, abs_tol=5e-7):
        raise AssertionError("Quantile-top-4 ATT fingerprint changed")

    probability_summary = np.vstack(
        [
            np.min(reference.group_probabilities, axis=0),
            np.mean(reference.group_probabilities, axis=0),
            np.max(reference.group_probabilities, axis=0),
        ]
    )
    expected_probability_summary = np.asarray(
        [
            [0.179, 0.200, 0.189, 0.137, 0.200],
            [0.190, 0.209, 0.195, 0.167, 0.237],
            [0.200, 0.217, 0.200, 0.200, 0.277],
        ]
    )
    if not np.allclose(
        probability_summary,
        expected_probability_summary,
        rtol=0.0,
        atol=5.1e-4,
    ):
        raise AssertionError("Quantile-top-4 group-probability fingerprint changed")

    reference_middle = reference_singular[:, 1]
    grid_rows: list[dict[str, object]] = []
    for kappa_m in args.kappa_m_grid:
        design = build_design(float(kappa_m), float(args.post_scale))
        validate_population_design(design)
        diagnostics = population_diagnostics(design)
        realized = np.asarray(
            [row["kappa_m_realized"] for row in diagnostics],
            dtype=float,
        )
        sigma_middle = np.asarray(
            [row["sigma_middle_m"] for row in diagnostics],
            dtype=float,
        )
        sigma_min = np.asarray(
            [row["sigma_min_m"] for row in diagnostics],
            dtype=float,
        )
        grid_rows.append(
            {
                "kappa_m_target": float(kappa_m),
                "kappa_mtm_target": float(kappa_m**2),
                "max_condition_error": float(
                    np.max(np.abs(realized - float(kappa_m)))
                ),
                "lambda_min_mtm_min": float(np.min(sigma_min**2)),
                "middle_singular_lifted_points": int(
                    np.sum(sigma_middle > reference_middle + 1e-9)
                ),
            }
        )

    return {
        "status": "PASS",
        "dgp_name": "sc_only",
        "donor_codes": list(DONOR_CODES),
        "donor_states": DONOR_STATES.split("|"),
        "baseline_donor_code": BASELINE_DONOR_CODE,
        "condition_definition": CONDITION_DEFINITION,
        "empirical_kappa_m_reference": CALIBRATION_TARGET_KAPPA_M,
        "empirical_kappa_mtm_reference": CALIBRATION_TARGET_KAPPA_MTM,
        "empirical_reference_provenance": CALIBRATION_TARGET_SOURCE,
        "att_true": float(reference.att_true),
        "post_scale": float(args.post_scale),
        "reference_spectral_quantiles": {
            "probabilities": quantiles.tolist(),
            "sigma_max": reference_spectral_quantiles[:, 0].tolist(),
            "sigma_middle": reference_spectral_quantiles[:, 1].tolist(),
            "sigma_min": reference_spectral_quantiles[:, 2].tolist(),
        },
        "grid": grid_rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse optional overrides; defaults are the requested HPC run."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=list(DEFAULT_SAMPLE_SIZES),
        metavar="N",
        help="sample sizes run sequentially (default: 2000 4000)",
    )
    parser.add_argument("--outer-start", type=int, default=0)
    parser.add_argument("--outer-reps", type=int, default=DEFAULT_OUTER_REPS)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument(
        "--kappa_m-grid",
        "--kappa-m-grid",
        "--kappa-m",
        type=float,
        nargs="+",
        default=list(DEFAULT_KAPPA_M_GRID),
        metavar="KAPPA_M",
        help=(
            "direct kappa(M) targets; defaults are the 11-point grid in the "
            "Quantile-top-4 report"
        ),
    )
    parser.add_argument(
        "--post-scale",
        type=float,
        default=DEFAULT_POST_SCALE,
        help=(
            "multiplier on the observed 2002--2003 donor-change contrast "
            "(default: 3, matching the supplied Quantile-top-4 report; "
            "use 1 for the unamplified empirical contrast)"
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="workers; 0 uses CPUs visible through process affinity",
    )
    parser.add_argument("--master-seed", type=int, default=MASTER_SEED)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="directory for generated CSV files",
    )
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help="short label included in automatically generated filenames",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help=(
            "optional prefix; with multiple n values, _n<N> is appended; "
            "otherwise filenames are generated automatically"
        ),
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="validate every requested population cell and exit",
    )
    args = parser.parse_args(argv)
    if int(args.outer_reps) <= 0:
        parser.error("--outer-reps must be positive")
    if int(args.bootstraps) < 2:
        parser.error("--bootstraps must be at least two")
    if any(int(n) <= 0 for n in args.n):
        parser.error("--n values must be positive")
    if len(set(args.n)) != len(args.n):
        parser.error("--n values must not contain duplicates")
    if args.outer_start < 0:
        parser.error("--outer-start must be nonnegative")
    if args.master_seed < 0:
        parser.error("--master-seed must be nonnegative")
    if args.jobs < 0:
        parser.error("--jobs must be nonnegative")
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must lie strictly between zero and one")
    if not np.isfinite(args.post_scale) or args.post_scale <= 0.0:
        parser.error("--post-scale must be finite and strictly positive")
    if not args.kappa_m_grid:
        parser.error("--kappa_m-grid must contain at least one value")
    if any(
        not np.isfinite(kappa_m) or float(kappa_m) < 1.0
        for kappa_m in args.kappa_m_grid
    ):
        parser.error(
            "--kappa_m-grid values must be finite and at least one"
        )
    if len(set(float(kappa_m) for kappa_m in args.kappa_m_grid)) != len(
        args.kappa_m_grid
    ):
        parser.error("--kappa_m-grid values must not contain duplicates")
    if not args.tag or any(character.isspace() for character in args.tag):
        parser.error("--tag must be nonempty and contain no whitespace")
    args.kappa_m_grid = sorted(float(kappa_m) for kappa_m in args.kappa_m_grid)
    return args


def main() -> None:
    args = parse_args()
    audit = audit_population_grid(args)
    print(json.dumps({"population_audit": audit}, indent=2), flush=True)
    if args.audit_only:
        return
    outputs = run(args)
    print_final_coverage(outputs)


if __name__ == "__main__":
    main()
