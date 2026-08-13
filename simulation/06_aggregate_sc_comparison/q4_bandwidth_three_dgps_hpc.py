#!/usr/bin/env python3
"""Self-contained Quantile-top-4 bandwidth experiment for HPC.

The only population regimes are SC only, PT only, and joint PT+SC.  All use
Maryland, New Hampshire, Utah, and Virginia (Virginia is the affine baseline),
the direct condition benchmark kappa(M)=13.070487184660164, five pre-periods,
one post-period, and a strong absolute smallest singular value.  No local or
fixed misspecification DGP is selectable.

The experiment changes only the local-linear smoothing bandwidth
``h = c*n^(-2/7)`` over ``c = 1, 1.5, 2, 2.5, 3, 3.5``.  For every
sample size, DGP, bandwidth, and outer replication it computes the fixed-fold
DiD-SC estimate and a full exponential-multiplier bootstrap that re-estimates
all nuisance functions and synthetic-control weights.  Common random numbers
are shared across bandwidths and DGPs.

Defaults are n=2,000 and 4,000, R=500, B=500, and all CPUs visible to
the allocation.  Four compact CSV files are written per sample size, with
progress flushed every 10 percent.  Only NumPy and Numba are required.
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
    os.path.join(_cache_root, "q4_bandwidth_three_dgps_numba_cache"),
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
import sys
import time
import warnings
from dataclasses import dataclass, fields
from multiprocessing import get_context
from pathlib import Path
from statistics import NormalDist
from functools import lru_cache
from typing import Iterable

import numpy as np
from numba import NumbaPerformanceWarning, config as numba_config, njit


numba_config.DISABLE_PERFORMANCE_WARNINGS = 1
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*encountered in matmul")


DGP_NAMES = ("sc_only", "pt_only", "pt_sc_both")
EXPECTED_STATUS = {
    "sc_only": (False, True),
    "pt_only": (True, False),
    "pt_sc_both": (True, True),
}
DGP_VERSION = "quantile_top4_kappa13_bandwidth_three_exact_regimes_v1"
DONOR_CODES = (24, 33, 49, 51)
DONOR_STATES = "Maryland|New Hampshire|Utah|Virginia"
CONDITION_DEFINITION = "kappa(M)=sigma_max(M)/sigma_min(M)"
EXPERIMENT_PURPOSE = "quantile_top4_bandwidth_three_exact_pt_sc_regimes"
INFERENCE_METHOD = "full_nested_exponential_multiplier_bootstrap"

KAPPA_M = 13.070487184660164
KAPPA_M_FIXED = KAPPA_M
Q4_REFERENCE_GAMMA = math.sqrt(29.07742027873098)
Q4_POST_SCALE = 3.0
PT_ONLY_SC_VIOLATION = 0.40
Q4_EMPIRICAL_POST_CONTRAST = np.asarray(
    [0.211789512326912, 0.9801124738736031, 0.3509230190287762],
    dtype=float,
)
Q4_ATT_FINGERPRINT = 1.011932168518905
Q4_SIGMA_FINGERPRINT = np.asarray(
    [14.074084253556677, 14.073902974813999, 1.0767834476800806],
    dtype=float,
)

PARAMETER_MODE = "bandwidth_grid"
BANDWIDTH_RATE_EXPONENT = -2.0 / 7.0
BANDWIDTH_KERNEL = "epanechnikov_compact_support"
BANDWIDTH_FORMULA = "h=bandwidth_coefficient*n^(-2/7)"
DEFAULT_BANDWIDTH_COEFFICIENTS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
DEFAULT_SAMPLE_SIZES = (2000, 4000)
DEFAULT_OUTER_REPS = 500
DEFAULT_BOOTSTRAPS = 500
DEFAULT_MASTER_SEED = 2026073001
RANK_COMPLETION_RELATIVE_SCALE = Q4_REFERENCE_GAMMA

TARGET_NAMES = ("att", "population_estimand", "trim_score_target")
INTERVAL_METHODS = ("basic", "symmetric", "percentile", "normal")
ABS_TOL = 1.0e-8
REL_TOL = 1.0e-9
N_GROUPS = 5
N_DONORS = 4
N_PRE = 5
N_PERIODS = 6
N_X = 101
N_FOLDS = 2
TREATMENT_LEVEL = 1.0
TREATMENT_AMPLITUDE = 0.7
# Backward-compatible fallback for the embedded estimator core. Every
# bandwidth-experiment design supplies its own explicit coefficient.
BASELINE_BANDWIDTH_COEFFICIENT = 2.5
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


_DESIGNS: dict[float, Design] = {}


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

    bandwidth_coefficient = float(
        getattr(
            design,
            "bandwidth_coefficient",
            BASELINE_BANDWIDTH_COEFFICIENT,
        )
    )
    bandwidth = bandwidth_coefficient * n ** (-2.0 / 7.0)
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


def available_cpu_count() -> int:
    """Return CPUs visible to the current interactive allocation."""

    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


@dataclass(frozen=True)
class NewDesign(Design):
    """The core estimator fields plus auditable DGP metadata."""

    dgp_name: str
    config_id: str
    parameter_mode: str
    tuning_value: float
    rho: float
    eta: float
    sigma_delta_ref: float
    pre_sc_gap_multiplier: float
    pt_holds_by_construction: bool
    sc_holds_by_construction: bool
    population_estimand: float
    population_estimand_minus_att: float
    maximum_pre_affine_sc_residual_l2: float
    median_pre_affine_sc_residual_l2: float
    maximum_post_sc_residual_at_affine_weights: float
    maximum_full_sc_residual_at_affine_weights: float
    maximum_pseudo_weight_error_l2: float


_WORKER_CONFIG_COUNT = 0


def parameter_slug(value: float) -> str:
    """Return a filesystem-safe, deterministic representation of a number."""

    value = float(value)
    if value == 0.0:
        return "z0"
    sign = "m" if value < 0.0 else "p"
    magnitude = (
        format(abs(value), ".12g")
        .replace(".", "p")
        .replace("+", "p")
        .replace("-", "m")
    )
    return sign + magnitude


def affine_population_diagnostics(
    donor_means: np.ndarray,
    treated_means: np.ndarray,
    true_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute unrestricted affine PT/SC diagnostics at every covariate."""

    condition_numbers = np.empty(N_X, dtype=float)
    singular_values = np.empty(
        (N_X, N_DONORS - 1),
        dtype=float,
    )
    ranks = np.empty(N_X, dtype=int)
    affine_weights = np.empty((N_X, N_DONORS), dtype=float)
    pre_residual_l2 = np.empty(N_X, dtype=float)
    post_residual = np.empty(N_X, dtype=float)
    full_residual_max = np.empty(N_X, dtype=float)
    pt_gap = np.empty(N_X, dtype=float)
    pseudo_weight_error = np.empty(N_X, dtype=float)
    change_gap = np.empty(N_X, dtype=float)
    weak_post_projection = np.empty(N_X, dtype=float)

    for x_index in range(N_X):
        donor_path = donor_means[x_index]
        donor_pre = donor_path[: N_PRE]
        matrix_m = donor_pre[:, : N_DONORS - 1] - donor_pre[:, [-1]]
        target = treated_means[x_index, : N_PRE] - donor_pre[:, -1]
        gram = matrix_m.T @ matrix_m
        free_weights = np.linalg.solve(gram, matrix_m.T @ target)
        weights = np.r_[free_weights, 1.0 - free_weights.sum()]
        pre_residual = target - matrix_m @ free_weights
        full_residual = treated_means[x_index] - donor_path @ weights
        donor_change = donor_path[-1] - donor_path[-2]
        treated_change = (
            treated_means[x_index, -1] - treated_means[x_index, -2]
        )
        _, values, right_vectors = np.linalg.svd(
            matrix_m,
            full_matrices=False,
        )

        condition_numbers[x_index] = values[0] / values[-1]
        singular_values[x_index] = values
        ranks[x_index] = np.linalg.matrix_rank(matrix_m)
        affine_weights[x_index] = weights
        pre_residual_l2[x_index] = np.linalg.norm(pre_residual)
        post_residual[x_index] = full_residual[-1]
        full_residual_max[x_index] = np.max(np.abs(full_residual))
        pt_gap[x_index] = np.max(np.abs(treated_change - donor_change))
        pseudo_weight_error[x_index] = np.linalg.norm(
            weights - true_weights[x_index]
        )
        change_gap[x_index] = treated_change - donor_change @ weights
        donor_change_contrast = (
            donor_change[: N_DONORS - 1] - donor_change[-1]
        )
        weak_post_projection[x_index] = abs(
            float(right_vectors[-1] @ donor_change_contrast)
        )

    return {
        "condition_numbers": condition_numbers,
        "singular_values": singular_values,
        "ranks": ranks,
        "affine_weights": affine_weights,
        "pre_residual_l2": pre_residual_l2,
        "post_residual": post_residual,
        "full_residual_max": full_residual_max,
        "pt_gap": pt_gap,
        "pseudo_weight_error": pseudo_weight_error,
        "change_gap": change_gap,
        "weak_post_projection": weak_post_projection,
    }


def population_diagnostics(
    design: NewDesign,
) -> list[dict[str, float | int | str | bool]]:
    """Return the required 101-point spectral and assumption audit."""

    diagnostics = affine_population_diagnostics(
        design.donor_means,
        design.treated_untreated_means,
        design.true_weights,
    )
    rows: list[dict[str, float | int | str | bool]] = []
    for x_index, x_value in enumerate(design.x_support):
        values = diagnostics["singular_values"][x_index]
        weights = diagnostics["affine_weights"][x_index]
        rows.append(
            {
                "dgp_version": DGP_VERSION,
                "donor_states": DONOR_STATES,
                "condition_definition": CONDITION_DEFINITION,
                "experiment_purpose": EXPERIMENT_PURPOSE,
                "dgp_name": design.dgp_name,
                "config_id": design.config_id,
                "parameter_mode": design.parameter_mode,
                "tuning_value": design.tuning_value,
                "rho": design.rho,
                "eta": design.eta,
                "x": float(x_value),
                "kappa_m_target": KAPPA_M_FIXED,
                "kappa_m_realized": float(
                    diagnostics["condition_numbers"][x_index]
                ),
                "kappa_mtm_derived": float(
                    diagnostics["condition_numbers"][x_index] ** 2
                ),
                "sigma_max_m": float(values[0]),
                "sigma_middle_m": float(values[1]),
                "sigma_min_m": float(values[2]),
                "rank_m": int(diagnostics["ranks"][x_index]),
                "post_scale": design.post_scale,
                "pre_sc_gap_multiplier": design.pre_sc_gap_multiplier,
                "sigma_delta_ref": design.sigma_delta_ref,
                "pt_gap_max_at_x": float(
                    diagnostics["pt_gap"][x_index]
                ),
                "pre_affine_sc_residual_l2": float(
                    diagnostics["pre_residual_l2"][x_index]
                ),
                "post_sc_residual_at_affine_weights": float(
                    diagnostics["post_residual"][x_index]
                ),
                "full_sc_residual_max_at_affine_weights": float(
                    diagnostics["full_residual_max"][x_index]
                ),
                "affine_pseudo_weight_error_l2": float(
                    diagnostics["pseudo_weight_error"][x_index]
                ),
                "affine_weight_min": float(np.min(weights)),
                "affine_weight_max": float(np.max(weights)),
                "true_weight_min": float(
                    np.min(design.true_weights[x_index])
                ),
                "true_weight_max": float(
                    np.max(design.true_weights[x_index])
                ),
                "change_gap_at_affine_weights": float(
                    diagnostics["change_gap"][x_index]
                ),
                "att_true": design.att_true,
                "population_estimand": design.population_estimand,
                "population_estimand_minus_att": (
                    design.population_estimand_minus_att
                ),
                "population_trim_normalized_score_target": (
                    design.population_trim_normalized_score_target
                ),
                "pt_holds": bool(
                    diagnostics["pt_gap"][x_index] <= ABS_TOL
                ),
                "sc_holds": bool(
                    diagnostics["pre_residual_l2"][x_index] <= ABS_TOL
                    and diagnostics["full_residual_max"][x_index] <= ABS_TOL
                ),
            }
        )
    return rows


def write_population_diagnostics(
    path: Path,
    designs: tuple[NewDesign, ...],
) -> None:
    """Write one validated row per DGP configuration and covariate point."""

    rows = [
        row
        for design in designs
        for row in population_diagnostics(design)
    ]
    if not rows:
        raise ValueError("cannot write empty population diagnostics")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson_interval(coverage: float, count: int) -> tuple[float, float]:
    """Return a 95% Wilson interval for a simulated coverage proportion."""

    if count <= 0 or not np.isfinite(coverage):
        return np.nan, np.nan
    z_value = 1.959963984540054
    denominator = 1.0 + z_value * z_value / count
    center = (
        coverage + z_value * z_value / (2.0 * count)
    ) / denominator
    radius = (
        z_value
        * math.sqrt(
            coverage * (1.0 - coverage) / count
            + z_value * z_value / (4.0 * count * count)
        )
        / denominator
    )
    return center - radius, center + radius


def coverage_mcse(coverage: float, count: int) -> float:
    """Return the naive Bernoulli Monte Carlo standard error."""

    if count <= 0 or not np.isfinite(coverage):
        return np.nan
    return float(math.sqrt(coverage * (1.0 - coverage) / count))


def interval_fieldnames() -> list[str]:
    """Return the interval columns shared by outer and summary logic."""

    fields: list[str] = []
    for method in INTERVAL_METHODS:
        fields.extend(
            [
                f"{method}_ci_low",
                f"{method}_ci_high",
                f"{method}_ci_length",
            ]
        )
        fields.extend(
            f"{method}_covers_{target}" for target in TARGET_NAMES
        )
    return fields


def outer_fieldnames() -> list[str]:
    """Return the stable outer-replication CSV schema."""

    return [
        "dgp_version",
        "donor_states",
        "experiment_purpose",
        "dgp_name",
        "config_id",
        "parameter_mode",
        "tuning_value",
        "rho",
        "eta",
        "post_scale",
        "pre_sc_gap_multiplier",
        "sigma_delta_ref",
        "pt_holds_by_construction",
        "sc_holds_by_construction",
        "condition_definition",
        "kappa_m",
        "kappa_mtm_derived",
        "inference_method",
        "weight_estimator",
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
        "att_true",
        "population_estimand",
        "population_estimand_minus_att",
        "population_trim_normalized_score_target",
        "trim_score_target_minus_population_estimand",
        "point_estimate",
        "estimation_error_vs_att",
        "estimation_error_vs_population_estimand",
        "estimation_error_vs_trim_score_target",
        "conditional_q_alpha_over_2",
        "conditional_q_one_minus_alpha_over_2",
        "conditional_abs_q_one_minus_alpha",
        "bootstrap_root_mean",
        "bootstrap_root_sd",
        *interval_fieldnames(),
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
        "requested_bootstraps",
        "valid_bootstraps",
        "bootstrap_failures",
        "status",
        "error_message",
        "runtime_seconds",
    ]


def bootstrap_fieldnames() -> list[str]:
    """Return the raw nested-bootstrap CSV schema."""

    return [
        "dgp_version",
        "dgp_name",
        "config_id",
        "parameter_mode",
        "tuning_value",
        "rho",
        "eta",
        "post_scale",
        "pre_sc_gap_multiplier",
        "sigma_delta_ref",
        "total_periods",
        "n",
        "outer_replication",
        "master_seed",
        "base_seed_hex",
        "bootstrap_replication",
        "bootstrap_estimate",
        "bootstrap_error",
        "status",
        "error_message",
    ]


def bootstrap_metadata(
    design: NewDesign,
    args: argparse.Namespace,
    n: int,
    result: OuterResult,
) -> dict[str, float | int | str]:
    """Return design and seed metadata shared by all bootstrap rows."""

    return {
        "dgp_version": DGP_VERSION,
        "dgp_name": design.dgp_name,
        "config_id": design.config_id,
        "parameter_mode": design.parameter_mode,
        "tuning_value": design.tuning_value,
        "rho": design.rho,
        "eta": design.eta,
        "post_scale": design.post_scale,
        "pre_sc_gap_multiplier": design.pre_sc_gap_multiplier,
        "sigma_delta_ref": design.sigma_delta_ref,
        "total_periods": N_PERIODS,
        "n": int(n),
        "outer_replication": result.replication,
        "master_seed": int(args.master_seed),
        "base_seed_hex": f"0x{result.base_seed:016x}",
    }


def target_values(design: NewDesign) -> dict[str, float]:
    """Return all population targets that must remain distinct."""

    return {
        "att": float(design.att_true),
        "population_estimand": float(design.population_estimand),
        "trim_score_target": float(
            design.population_trim_normalized_score_target
        ),
    }


def replication_row(
    result: OuterResult,
    design: NewDesign,
    n: int,
    args: argparse.Namespace,
) -> tuple[dict[str, float | int | str | bool], np.ndarray]:
    """Build one outer row and all four conditional confidence intervals."""

    point_valid = math.isfinite(result.point_estimate)
    finite_mask = np.isfinite(result.bootstrap_errors)
    finite_errors = result.bootstrap_errors[finite_mask]
    valid_bootstraps = int(finite_errors.size)
    bootstrap_complete = (
        point_valid and valid_bootstraps == int(args.bootstraps)
    )
    targets = target_values(design)
    row: dict[str, float | int | str | bool] = {
        "dgp_version": DGP_VERSION,
        "donor_states": DONOR_STATES,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "dgp_name": design.dgp_name,
        "config_id": design.config_id,
        "parameter_mode": design.parameter_mode,
        "tuning_value": design.tuning_value,
        "rho": design.rho,
        "eta": design.eta,
        "post_scale": design.post_scale,
        "pre_sc_gap_multiplier": design.pre_sc_gap_multiplier,
        "sigma_delta_ref": design.sigma_delta_ref,
        "pt_holds_by_construction": design.pt_holds_by_construction,
        "sc_holds_by_construction": design.sc_holds_by_construction,
        "condition_definition": CONDITION_DEFINITION,
        "kappa_m": design.kappa_m,
        "kappa_mtm_derived": design.kappa_m**2,
        "inference_method": INFERENCE_METHOD,
        "weight_estimator": WEIGHT_ESTIMATOR,
        "alpha": float(args.alpha),
        "confidence_level": float(1.0 - args.alpha),
        "total_periods": N_PERIODS,
        "pre_periods": N_PRE,
        "post_periods": 1,
        "n": int(n),
        "outer_replication": int(result.replication),
        "master_seed": int(args.master_seed),
        "base_seed_hex": f"0x{result.base_seed:016x}",
        "fold_hash": result.fold_hash,
        "att_true": targets["att"],
        "population_estimand": targets["population_estimand"],
        "population_estimand_minus_att": (
            design.population_estimand_minus_att
        ),
        "population_trim_normalized_score_target": (
            targets["trim_score_target"]
        ),
        "trim_score_target_minus_population_estimand": (
            targets["trim_score_target"]
            - targets["population_estimand"]
        ),
        "point_estimate": float(result.point_estimate),
        "estimation_error_vs_att": (
            float(result.point_estimate - targets["att"])
            if point_valid
            else np.nan
        ),
        "estimation_error_vs_population_estimand": (
            float(result.point_estimate - targets["population_estimand"])
            if point_valid
            else np.nan
        ),
        "estimation_error_vs_trim_score_target": (
            float(result.point_estimate - targets["trim_score_target"])
            if point_valid
            else np.nan
        ),
        "conditional_q_alpha_over_2": np.nan,
        "conditional_q_one_minus_alpha_over_2": np.nan,
        "conditional_abs_q_one_minus_alpha": np.nan,
        "bootstrap_root_mean": np.nan,
        "bootstrap_root_sd": np.nan,
        "population_kappa_m_median": design.population_kappa_m_median,
        "population_kappa_m_p95": design.population_kappa_m_p95,
        "population_sigma_max_median": (
            design.population_sigma_max_median
        ),
        "population_sigma_middle_median": (
            design.population_sigma_middle_median
        ),
        "population_sigma_min_median": (
            design.population_sigma_min_median
        ),
        "population_rank_min": design.population_rank_min,
        "sample_kappa_m_median": result.sample_kappa_m_median,
        "sample_kappa_m_p95": result.sample_kappa_m_p95,
        "sample_kappa_m_max": result.sample_kappa_m_max,
        "sample_sigma_max_median": result.sample_sigma_max_median,
        "sample_sigma_middle_median": result.sample_sigma_middle_median,
        "sample_sigma_min_median": result.sample_sigma_min_median,
        "sample_rank_min": result.sample_rank_min,
        "sample_rank_deficient_share": result.sample_rank_deficient_share,
        "maximum_estimated_weight": result.maximum_estimated_weight,
        "estimated_weight_l2_median": result.estimated_weight_l2_median,
        "raw_weight_l2_median": result.raw_weight_l2_median,
        "simplex_projection_distance_median": (
            result.simplex_projection_distance_median
        ),
        "simplex_boundary_fraction": result.simplex_boundary_fraction,
        "projected_weight_error_l2_median": (
            result.projected_weight_error_l2_median
        ),
        "raw_weight_error_l2_median": (
            result.raw_weight_error_l2_median
        ),
        "pre_fit_l2_median": result.pre_fit_l2_median,
        "pre_fit_max": result.pre_fit_max,
        "requested_bootstraps": int(args.bootstraps),
        "valid_bootstraps": valid_bootstraps,
        "bootstrap_failures": int(args.bootstraps) - valid_bootstraps,
        "status": "",
        "error_message": "",
        "runtime_seconds": result.runtime_seconds,
    }
    for field in interval_fieldnames():
        row[field] = np.nan

    if bootstrap_complete:
        q_low, q_high = np.quantile(
            finite_errors,
            [args.alpha / 2.0, 1.0 - args.alpha / 2.0],
        )
        absolute_q = float(
            np.quantile(np.abs(finite_errors), 1.0 - args.alpha)
        )
        root_sd = float(np.std(finite_errors, ddof=1))
        normal_critical = float(
            NormalDist().inv_cdf(1.0 - args.alpha / 2.0)
        )
        intervals = {
            "basic": (
                result.point_estimate - q_high,
                result.point_estimate - q_low,
            ),
            "symmetric": (
                result.point_estimate - absolute_q,
                result.point_estimate + absolute_q,
            ),
            "percentile": (
                result.point_estimate + q_low,
                result.point_estimate + q_high,
            ),
            "normal": (
                result.point_estimate - normal_critical * root_sd,
                result.point_estimate + normal_critical * root_sd,
            ),
        }
        row["conditional_q_alpha_over_2"] = float(q_low)
        row["conditional_q_one_minus_alpha_over_2"] = float(q_high)
        row["conditional_abs_q_one_minus_alpha"] = absolute_q
        row["bootstrap_root_mean"] = float(np.mean(finite_errors))
        row["bootstrap_root_sd"] = root_sd
        for method, (lower, upper) in intervals.items():
            row[f"{method}_ci_low"] = float(lower)
            row[f"{method}_ci_high"] = float(upper)
            row[f"{method}_ci_length"] = float(upper - lower)
            for target, target_value in targets.items():
                row[f"{method}_covers_{target}"] = float(
                    lower <= target_value <= upper
                )

    if not point_valid:
        row["status"] = "point_failed"
        row["error_message"] = result.point_failure
    elif not bootstrap_complete:
        row["status"] = "bootstrap_incomplete"
        row["error_message"] = (
            f"{int(args.bootstraps) - valid_bootstraps} bootstrap draw(s) "
            "failed; see bootstrap CSV"
        )
    else:
        row["status"] = "valid"
        row["error_message"] = ""
    return row, finite_errors


def finite_column(
    rows: list[dict[str, float | int | str | bool]],
    key: str,
) -> np.ndarray:
    """Extract finite floating-point values from outer rows."""

    values = np.asarray([row[key] for row in rows], dtype=float)
    return values[np.isfinite(values)]


def summarize_configuration(
    n: int,
    attempted: int,
    rows: list[dict[str, float | int | str | bool]],
    design: NewDesign,
    valid_bootstraps_total: int,
    requested_bootstraps_total: int,
    root_sum: float,
    root_sum_squares: float,
    root_count: int,
    worker_runtime: float,
    wall_runtime: float,
    args: argparse.Namespace,
) -> dict[str, float | int | str | bool]:
    """Summarize one DGP configuration for all population targets."""

    points = finite_column(rows, "point_estimate")
    within_root_sds = finite_column(rows, "bootstrap_root_sd")
    empirical_sd = (
        float(np.std(points, ddof=1)) if points.size > 1 else np.nan
    )
    if root_count > 1:
        numerator = root_sum_squares - root_sum * root_sum / root_count
        pooled_root_sd = math.sqrt(max(numerator, 0.0) / (root_count - 1))
    else:
        pooled_root_sd = np.nan
    summary: dict[str, float | int | str | bool] = {
        "dgp_version": DGP_VERSION,
        "donor_states": DONOR_STATES,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "dgp_name": design.dgp_name,
        "config_id": design.config_id,
        "parameter_mode": design.parameter_mode,
        "tuning_value": design.tuning_value,
        "rho": design.rho,
        "eta": design.eta,
        "post_scale": design.post_scale,
        "pre_sc_gap_multiplier": design.pre_sc_gap_multiplier,
        "sigma_delta_ref": design.sigma_delta_ref,
        "pt_holds_by_construction": design.pt_holds_by_construction,
        "sc_holds_by_construction": design.sc_holds_by_construction,
        "condition_definition": CONDITION_DEFINITION,
        "kappa_m": design.kappa_m,
        "kappa_mtm_derived": design.kappa_m**2,
        "inference_method": INFERENCE_METHOD,
        "weight_estimator": WEIGHT_ESTIMATOR,
        "alpha": float(args.alpha),
        "confidence_level": float(1.0 - args.alpha),
        "total_periods": N_PERIODS,
        "pre_periods": N_PRE,
        "post_periods": 1,
        "n": int(n),
        "attempted_outer_replications": int(attempted),
        "valid_point_estimates": int(points.size),
        "point_failures": int(attempted - points.size),
        "requested_bootstraps": int(requested_bootstraps_total),
        "valid_bootstraps": int(valid_bootstraps_total),
        "bootstrap_failures": int(
            requested_bootstraps_total - valid_bootstraps_total
        ),
        "att_true": design.att_true,
        "population_estimand": design.population_estimand,
        "population_estimand_minus_att": (
            design.population_estimand_minus_att
        ),
        "population_trim_normalized_score_target": (
            design.population_trim_normalized_score_target
        ),
        "trim_score_target_minus_population_estimand": (
            design.population_trim_normalized_score_target
            - design.population_estimand
        ),
        "mean_point_estimate": (
            float(np.mean(points)) if points.size else np.nan
        ),
        "empirical_sd": empirical_sd,
        "pooled_bootstrap_root_sd": pooled_root_sd,
        "pooled_bootstrap_root_sd_over_empirical_sd": (
            float(pooled_root_sd / empirical_sd)
            if np.isfinite(pooled_root_sd)
            and np.isfinite(empirical_sd)
            and empirical_sd > 0.0
            else np.nan
        ),
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
        "population_kappa_m_median": design.population_kappa_m_median,
        "population_kappa_m_p95": design.population_kappa_m_p95,
        "population_sigma_max_median": (
            design.population_sigma_max_median
        ),
        "population_sigma_middle_median": (
            design.population_sigma_middle_median
        ),
        "population_sigma_min_median": (
            design.population_sigma_min_median
        ),
        "population_rank_min": design.population_rank_min,
        "maximum_pre_affine_sc_residual_l2": (
            design.maximum_pre_affine_sc_residual_l2
        ),
        "median_pre_affine_sc_residual_l2": (
            design.median_pre_affine_sc_residual_l2
        ),
        "maximum_post_sc_residual_at_affine_weights": (
            design.maximum_post_sc_residual_at_affine_weights
        ),
        "maximum_full_sc_residual_at_affine_weights": (
            design.maximum_full_sc_residual_at_affine_weights
        ),
        "maximum_pseudo_weight_error_l2": (
            design.maximum_pseudo_weight_error_l2
        ),
        "median_sample_kappa_m_median": (
            float(np.median(finite_column(rows, "sample_kappa_m_median")))
            if finite_column(rows, "sample_kappa_m_median").size
            else np.nan
        ),
        "median_sample_kappa_m_p95": (
            float(np.median(finite_column(rows, "sample_kappa_m_p95")))
            if finite_column(rows, "sample_kappa_m_p95").size
            else np.nan
        ),
        "minimum_sample_rank": (
            int(np.min(finite_column(rows, "sample_rank_min")))
            if finite_column(rows, "sample_rank_min").size
            else 0
        ),
        "maximum_sample_rank_deficient_share": (
            float(
                np.max(
                    finite_column(rows, "sample_rank_deficient_share")
                )
            )
            if finite_column(rows, "sample_rank_deficient_share").size
            else np.nan
        ),
        "simplex_boundary_fraction_median": (
            float(
                np.median(
                    finite_column(rows, "simplex_boundary_fraction")
                )
            )
            if finite_column(rows, "simplex_boundary_fraction").size
            else np.nan
        ),
        "projected_weight_error_l2_median": (
            float(
                np.median(
                    finite_column(
                        rows,
                        "projected_weight_error_l2_median",
                    )
                )
            )
            if finite_column(
                rows,
                "projected_weight_error_l2_median",
            ).size
            else np.nan
        ),
        "raw_weight_error_l2_median": (
            float(
                np.median(
                    finite_column(rows, "raw_weight_error_l2_median")
                )
            )
            if finite_column(rows, "raw_weight_error_l2_median").size
            else np.nan
        ),
        "master_seed": int(args.master_seed),
        "cell_worker_runtime_seconds": float(worker_runtime),
        "sample_grid_wall_runtime_seconds": float(wall_runtime),
    }

    targets = target_values(design)
    for target, target_value in targets.items():
        errors = points - target_value
        summary[f"bias_vs_{target}"] = (
            float(np.mean(errors)) if errors.size else np.nan
        )
        summary[f"rmse_vs_{target}"] = (
            float(np.sqrt(np.mean(errors * errors)))
            if errors.size
            else np.nan
        )

    nominal = 1.0 - float(args.alpha)
    for method in INTERVAL_METHODS:
        lengths = finite_column(rows, f"{method}_ci_length")
        summary[f"{method}_mean_length"] = (
            float(np.mean(lengths)) if lengths.size else np.nan
        )
        summary[f"{method}_median_length"] = (
            float(np.median(lengths)) if lengths.size else np.nan
        )
        for target in TARGET_NAMES:
            covers = finite_column(rows, f"{method}_covers_{target}")
            coverage = (
                float(np.mean(covers)) if covers.size else np.nan
            )
            low, high = wilson_interval(coverage, int(covers.size))
            prefix = f"{method}_{target}"
            summary[f"{prefix}_valid"] = int(covers.size)
            summary[f"{prefix}_coverage"] = coverage
            summary[f"{prefix}_minus_nominal"] = (
                float(coverage - nominal)
                if np.isfinite(coverage)
                else np.nan
            )
            summary[f"{prefix}_mcse"] = coverage_mcse(
                coverage,
                int(covers.size),
            )
            summary[f"{prefix}_wilson_low"] = low
            summary[f"{prefix}_wilson_high"] = high
            summary[f"{prefix}_nominal_inside_wilson"] = bool(
                np.isfinite(low) and low <= nominal <= high
            )
    return summary


def provisional_coverage_text(
    rows_by_config: dict[int, list[dict[str, float | int | str | bool]]],
    designs: tuple[NewDesign, ...],
) -> str:
    """Return compact ATT/population/trim basic coverage at a progress point."""

    parts: list[str] = []
    for config_index, design in enumerate(designs):
        rows = rows_by_config[config_index]
        att = finite_column(rows, "basic_covers_att")
        population = finite_column(
            rows,
            "basic_covers_population_estimand",
        )
        trim = finite_column(rows, "basic_covers_trim_score_target")
        if not att.size or not population.size or not trim.size:
            continue
        parts.append(
            f"{design.config_id}="
            f"{np.mean(att):.3f}/{np.mean(population):.3f}/"
            f"{np.mean(trim):.3f}(R={trim.size})"
        )
    return " | ".join(parts)


# Preserve the generic CSV builders before adding bandwidth metadata.
_BANDWIDTH_POPULATION_DIAGNOSTICS = population_diagnostics
_BANDWIDTH_OUTER_FIELDNAMES = outer_fieldnames
_BANDWIDTH_BOOTSTRAP_FIELDNAMES = bootstrap_fieldnames
_BANDWIDTH_BOOTSTRAP_METADATA = bootstrap_metadata
_BANDWIDTH_REPLICATION_ROW = replication_row
_BANDWIDTH_SUMMARIZE_CONFIGURATION = summarize_configuration


BANDWIDTH_METADATA_FIELDS = (
    "dgp_source",
    "rank_completion_relative_scale",
    "bandwidth_coefficient",
    "bandwidth_rate_exponent",
    "effective_bandwidth",
    "bandwidth_kernel",
    "bandwidth_formula",
)


@dataclass(frozen=True)
class BandwidthConfig:
    """One DGP and bandwidth coefficient at a specified sample size."""

    name: str
    n: int
    bandwidth_coefficient: float

    @property
    def config_id(self) -> str:
        return (
            f"{self.name}__hcoef_"
            f"{parameter_slug(self.bandwidth_coefficient)}"
        )

    @property
    def effective_bandwidth(self) -> float:
        return float(
            self.bandwidth_coefficient
            * self.n ** BANDWIDTH_RATE_EXPONENT
        )


@dataclass(frozen=True)
class BandwidthDesign(NewDesign):
    """A validated population design paired with one estimator bandwidth."""

    population_n: int
    dgp_source: str
    bandwidth_coefficient: float
    bandwidth_rate_exponent: float
    effective_bandwidth: float
    bandwidth_kernel: str
    bandwidth_formula: str


def bandwidth_metadata(
    design: BandwidthDesign,
) -> dict[str, float | int | str]:
    """Return bandwidth and construction provenance for all CSV outputs."""

    return {
        "dgp_source": design.dgp_source,
        "rank_completion_relative_scale": (
            RANK_COMPLETION_RELATIVE_SCALE
        ),
        "bandwidth_coefficient": design.bandwidth_coefficient,
        "bandwidth_rate_exponent": design.bandwidth_rate_exponent,
        "effective_bandwidth": design.effective_bandwidth,
        "bandwidth_kernel": design.bandwidth_kernel,
        "bandwidth_formula": design.bandwidth_formula,
    }


def population_diagnostics(
    design: BandwidthDesign,
) -> list[dict[str, float | int | str | bool]]:
    """Add bandwidth metadata to the 101-point population audit."""

    rows = _BANDWIDTH_POPULATION_DIAGNOSTICS(design)
    metadata = bandwidth_metadata(design)
    for row in rows:
        row.update(
            {
                "n": design.population_n,
                **metadata,
            }
        )
    return rows


def insert_after(
    original: list[str],
    marker: str,
    extra: list[str],
) -> list[str]:
    """Insert fields once after a required schema marker."""

    result = list(original)
    position = result.index(marker) + 1
    result[position:position] = extra
    return result


def outer_fieldnames() -> list[str]:
    """Return outer fields with explicit bandwidth provenance."""

    return insert_after(
        _BANDWIDTH_OUTER_FIELDNAMES(),
        "n",
        list(BANDWIDTH_METADATA_FIELDS),
    )


def bootstrap_fieldnames() -> list[str]:
    """Return bootstrap fields with explicit bandwidth provenance."""

    return insert_after(
        _BANDWIDTH_BOOTSTRAP_FIELDNAMES(),
        "n",
        list(BANDWIDTH_METADATA_FIELDS),
    )


def bootstrap_metadata(
    design: BandwidthDesign,
    args: argparse.Namespace,
    n: int,
    result: OuterResult,
) -> dict[str, float | int | str]:
    """Add bandwidth metadata to every raw-bootstrap row."""

    metadata = _BANDWIDTH_BOOTSTRAP_METADATA(
        design,
        args,
        n,
        result,
    )
    metadata.update(bandwidth_metadata(design))
    return metadata


def replication_row(
    result: OuterResult,
    design: BandwidthDesign,
    n: int,
    args: argparse.Namespace,
) -> tuple[dict[str, float | int | str | bool], np.ndarray]:
    """Add bandwidth metadata to every outer-replication row."""

    row, finite_errors = _BANDWIDTH_REPLICATION_ROW(
        result,
        design,
        n,
        args,
    )
    row.update(bandwidth_metadata(design))
    return row, finite_errors


def summarize_configuration(
    *,
    n: int,
    attempted: int,
    rows: list[dict[str, float | int | str | bool]],
    design: BandwidthDesign,
    valid_bootstraps_total: int,
    requested_bootstraps_total: int,
    root_sum: float,
    root_sum_squares: float,
    root_count: int,
    worker_runtime: float,
    wall_runtime: float,
    args: argparse.Namespace,
) -> dict[str, float | int | str | bool]:
    """Add bandwidth metadata to one coverage-summary row."""

    summary = _BANDWIDTH_SUMMARIZE_CONFIGURATION(
        n=n,
        attempted=attempted,
        rows=rows,
        design=design,
        valid_bootstraps_total=valid_bootstraps_total,
        requested_bootstraps_total=requested_bootstraps_total,
        root_sum=root_sum,
        root_sum_squares=root_sum_squares,
        root_count=root_count,
        worker_runtime=worker_runtime,
        wall_runtime=wall_runtime,
        args=args,
    )
    summary.update(bandwidth_metadata(design))
    return summary


# The standalone engine above and this Quantile-top-4 layer share one module.
engine = sys.modules[__name__]


RESULTS_DIR = Path(__file__).resolve().parent / "generated"
_WORKER_DGP_GROUPS: tuple[tuple[int, ...], ...] = ()
_Q4_SC_BASE: engine.Design | None = None
_Q4_COMMON: tuple[engine.Design, np.ndarray, np.ndarray] | None = None


def _q4_null_space_rows(vector: np.ndarray) -> np.ndarray:
    """Return deterministic orthonormal rows spanning vector-perpendicular."""

    _, _, right_vectors = np.linalg.svd(
        vector.reshape(1, -1),
        full_matrices=True,
    )
    return right_vectors[1:, :]


def _q4_rank_completion_paths(
    factor_paths: np.ndarray,
    fixed_effect_contrasts: np.ndarray,
) -> np.ndarray:
    """Recreate the strong-scale Quantile-top-4 completion paths."""

    empirical_span = np.column_stack(
        [np.ones(engine.N_PRE), factor_paths[: engine.N_PRE]]
    )
    _, _, right_vectors = np.linalg.svd(
        empirical_span.T,
        full_matrices=True,
    )
    h_matrix = right_vectors.T[:, 3:5]
    if h_matrix.shape != (engine.N_PRE, engine.N_DONORS - 2):
        raise RuntimeError("unexpected rank-completion dimension")
    c_matrix = _q4_null_space_rows(fixed_effect_contrasts)
    completion_magnitude = float(
        Q4_REFERENCE_GAMMA
        * math.sqrt(engine.N_PRE)
        * np.linalg.norm(fixed_effect_contrasts)
    )
    donor_loadings = np.zeros((2, engine.N_DONORS), dtype=float)
    donor_loadings[:, : engine.N_DONORS - 1] = (
        completion_magnitude * c_matrix
    )
    factor_with_post = np.vstack([h_matrix, h_matrix[-1]])
    return factor_with_post @ donor_loadings


def _build_q4_sc_base() -> engine.Design:
    """Build and cache the audited Quantile-top-4 SC-only population."""

    global _Q4_SC_BASE
    if _Q4_SC_BASE is not None:
        return _Q4_SC_BASE

    x_support = np.linspace(0.0, 1.0, engine.N_X)
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
                [0.05504623773292663, -0.1933574832800169, 0.14155692214292365],
                [0.12958527990094934, -0.4341695502123225, 0.31310468234986366],
            ],
            [
                [-0.04569433625436106, 0.14013829339801145, -0.10104604900514517],
                [0.1598637449813732, -0.446799828059552, 0.2945602418165082],
            ],
            [
                [0.01898038602236328, -0.11574333888296137, 0.11063163604102708],
                [0.058767871781695465, -0.08364617571678026, 0.0022284743197091717],
            ],
            [
                [-0.07683253088475803, 0.2852992768248282, -0.23724358309126015],
                [0.11429532661261399, -0.32459744735242685, 0.23522847464847624],
            ],
        ],
        dtype=float,
    )
    x_polynomial = np.column_stack(
        [np.ones(engine.N_X), x_support, x_support * x_support]
    )
    donor_factor_loadings = np.einsum(
        "xp,dfp->xdf",
        x_polynomial,
        loading_coefficients,
    )
    fixed_effect_contrasts = (
        donor_fixed_effects[: engine.N_DONORS - 1]
        - donor_fixed_effects[-1]
    )
    rank_completion = _q4_rank_completion_paths(
        factor_paths,
        fixed_effect_contrasts,
    )
    reference_donor_means = (
        donor_fixed_effects[None, None, :]
        + time_effects[None, :, None]
        + np.einsum("tf,xdf->xtd", factor_paths, donor_factor_loadings)
        + rank_completion[None, :, :]
    )

    donor_means = reference_donor_means.copy()
    weak_post_projections = np.empty(engine.N_X, dtype=float)
    for x_index in range(engine.N_X):
        baseline = reference_donor_means[x_index, :, -1]
        reference_m = (
            reference_donor_means[
                x_index,
                : engine.N_PRE,
                : engine.N_DONORS - 1,
            ]
            - baseline[: engine.N_PRE, None]
        )
        left_vectors, singular_reference, right_vectors = np.linalg.svd(
            reference_m,
            full_matrices=False,
        )
        target_min = singular_reference[0] / KAPPA_M
        adjusted = np.maximum(singular_reference, target_min)
        adjusted[-1] = target_min
        target_m = left_vectors @ np.diag(adjusted) @ right_vectors
        donor_means[
            x_index,
            : engine.N_PRE,
            : engine.N_DONORS - 1,
        ] = baseline[: engine.N_PRE, None] + target_m
        post_contrast = (
            target_m[-1] + Q4_POST_SCALE * Q4_EMPIRICAL_POST_CONTRAST
        )
        donor_means[x_index, -1, : engine.N_DONORS - 1] = (
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

    donor_indices = np.arange(1, engine.N_DONORS + 1, dtype=float)
    weight_logits = (
        x_support[:, None]
        * (donor_indices[None, :] - engine.N_DONORS)
    )
    weight_logits -= weight_logits.max(axis=1, keepdims=True)
    softmax_weights = np.exp(weight_logits)
    softmax_weights /= softmax_weights.sum(axis=1, keepdims=True)
    true_weights = 0.2 / engine.N_DONORS + 0.8 * softmax_weights
    treated_means = np.einsum("xd,xtd->xt", true_weights, donor_means)
    treatment_effect = engine.TREATMENT_LEVEL + engine.TREATMENT_AMPLITUDE * np.sin(
        2.0 * np.pi * x_support
    )
    treated_probability = group_probabilities[:, 0]
    att_true = float(
        np.sum(treated_probability * treatment_effect)
        / np.sum(treated_probability)
    )
    trim_mask = (
        (x_support > engine.TRIM_QUANTILES[0])
        & (x_support < engine.TRIM_QUANTILES[1])
    )
    trim_target = float(
        np.sum(treated_probability[trim_mask] * treatment_effect[trim_mask])
        / np.sum(treated_probability)
        / np.mean(trim_mask)
    )
    diagnostics = engine.affine_population_diagnostics(
        donor_means,
        treated_means,
        true_weights,
    )
    singular = diagnostics["singular_values"]
    condition = diagnostics["condition_numbers"]
    true_residual = treated_means - np.einsum(
        "xd,xtd->xt",
        true_weights,
        donor_means,
    )
    base = engine.Design(
        kappa_m=KAPPA_M,
        x_support=x_support,
        group_probabilities=group_probabilities,
        donor_means=donor_means,
        treated_untreated_means=treated_means,
        treatment_effect=treatment_effect,
        true_weights=true_weights,
        residual_variances=residual_variances,
        att_true=att_true,
        population_trim_normalized_score_target=trim_target,
        population_kappa_m_median=float(np.median(condition)),
        population_kappa_m_p95=float(np.quantile(condition, 0.95)),
        population_kappa_mtm_median=float(np.median(condition**2)),
        population_kappa_mtm_p95=float(np.quantile(condition**2, 0.95)),
        population_sigma_max_median=float(np.median(singular[:, 0])),
        population_sigma_middle_median=float(np.median(singular[:, 1])),
        population_sigma_min_median=float(np.median(singular[:, -1])),
        population_rank_min=int(np.min(diagnostics["ranks"])),
        maximum_sc_residual=float(np.max(np.abs(true_residual))),
        maximum_pt_gap=float(np.max(diagnostics["pt_gap"])),
        weak_post_projection_median=float(
            np.median(weak_post_projections)
        ),
        weak_post_projection_over_sigma_min_median=float(
            np.median(weak_post_projections / singular[:, -1])
        ),
        post_scale=Q4_POST_SCALE,
    )
    if not math.isclose(
        base.att_true,
        Q4_ATT_FINGERPRINT,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
    ):
        raise AssertionError("Quantile-top-4 ATT fingerprint changed")
    realized_sigma = np.asarray(
        [
            base.population_sigma_max_median,
            base.population_sigma_middle_median,
            base.population_sigma_min_median,
        ]
    )
    if not np.allclose(
        realized_sigma,
        Q4_SIGMA_FINGERPRINT,
        rtol=REL_TOL,
        atol=ABS_TOL,
    ):
        raise AssertionError("Quantile-top-4 singular-value scale changed")
    _Q4_SC_BASE = base
    return base


def _q4_smooth_left_null_directions(
    matrices: np.ndarray,
) -> np.ndarray:
    """Choose the audited smooth unit direction orthogonal to col(M(x))."""

    candidate_anchors = np.asarray(
        [
            [1.0, -1.0, 1.0, -1.0, 1.0],
            [1.0, 0.0, -1.0, 0.0, 1.0],
            [0.0, 1.0, -1.0, 1.0, -1.0],
            [1.0, -2.0, 0.0, 2.0, -1.0],
        ],
        dtype=float,
    )
    projectors = np.empty(
        (matrices.shape[0], engine.N_PRE, engine.N_PRE),
        dtype=float,
    )
    for index, matrix_m in enumerate(matrices):
        left = np.linalg.svd(matrix_m, full_matrices=False)[0]
        projectors[index] = np.eye(engine.N_PRE) - left @ left.T
    minimum_norms = np.asarray(
        [
            min(
                np.linalg.norm(projectors[index] @ anchor)
                for index in range(matrices.shape[0])
            )
            for anchor in candidate_anchors
        ]
    )
    anchor = candidate_anchors[int(np.argmax(minimum_norms))]
    if float(np.max(minimum_norms)) <= 1.0e-8:
        raise RuntimeError("no uniformly separated left-null anchor exists")
    directions = np.empty((matrices.shape[0], engine.N_PRE), dtype=float)
    for index, matrix_m in enumerate(matrices):
        projected = projectors[index] @ anchor
        directions[index] = projected / np.linalg.norm(projected)
        if np.linalg.norm(matrix_m.T @ directions[index]) > 1.0e-9:
            raise AssertionError("Q4 direction is not in the left null space")
    return directions


def _q4_common_components(
) -> tuple[engine.Design, np.ndarray, np.ndarray]:
    """Return the Q4 base, common donor change, and smooth left-null path."""

    global _Q4_COMMON
    if _Q4_COMMON is not None:
        return _Q4_COMMON
    base = _build_q4_sc_base()
    reference_change = (
        base.donor_means[:, -1, -1]
        - base.donor_means[:, -2, -1]
    )
    pre_matrices = np.empty(
        (engine.N_X, engine.N_PRE, engine.N_DONORS - 1),
        dtype=float,
    )
    for x_index in range(engine.N_X):
        donor_pre = base.donor_means[x_index, : engine.N_PRE]
        pre_matrices[x_index] = (
            donor_pre[:, : engine.N_DONORS - 1]
            - donor_pre[:, [-1]]
        )
    left_null = _q4_smooth_left_null_directions(pre_matrices)
    _Q4_COMMON = (base, reference_change, left_null)
    return _Q4_COMMON


@lru_cache(maxsize=None)
def _build_q4_regime(name: str) -> engine.NewDesign:
    """Construct one of the three exact PT/SC assumption regimes."""

    if name not in DGP_NAMES:
        raise ValueError(f"unknown DGP {name!r}")
    base, reference_change, left_null = _q4_common_components()
    if name == "sc_only":
        donor_means = base.donor_means.copy()
        treated_means = base.treated_untreated_means.copy()
    else:
        donor_means = base.donor_means.copy()
        donor_means[:, -1] = (
            donor_means[:, -2] + reference_change[:, None]
        )
        synthetic = np.einsum(
            "xd,xtd->xt",
            base.true_weights,
            donor_means,
        )
        if name == "pt_sc_both":
            treated_means = synthetic
        else:
            treated_means = synthetic.copy()
            treated_means[:, : engine.N_PRE] += (
                PT_ONLY_SC_VIOLATION * left_null
            )
            treated_means[:, -1] += (
                PT_ONLY_SC_VIOLATION * left_null[:, -1]
            )

    diagnostics = engine.affine_population_diagnostics(
        donor_means,
        treated_means,
        base.true_weights,
    )
    singular = diagnostics["singular_values"]
    condition = diagnostics["condition_numbers"]
    treated_probability = base.group_probabilities[:, 0]
    change_gap = diagnostics["change_gap"]
    population_bias = float(
        np.sum(treated_probability * change_gap)
        / np.sum(treated_probability)
    )
    population_estimand = float(base.att_true + population_bias)
    trim_mask = (
        (base.x_support > engine.TRIM_QUANTILES[0])
        & (base.x_support < engine.TRIM_QUANTILES[1])
    )
    trim_target = float(
        np.sum(
            treated_probability[trim_mask]
            * (
                base.treatment_effect[trim_mask]
                + change_gap[trim_mask]
            )
        )
        / np.sum(treated_probability)
        / np.mean(trim_mask)
    )
    true_residual = treated_means - np.einsum(
        "xd,xtd->xt",
        base.true_weights,
        donor_means,
    )
    donor_change = donor_means[:, -1] - donor_means[:, -2]
    treated_change = treated_means[:, -1] - treated_means[:, -2]
    changes = np.column_stack([treated_change, donor_change])
    joint_probability = base.group_probabilities / engine.N_X
    noise_change_variance = (
        base.residual_variances[-2] + base.residual_variances[-1]
    )
    mean_change = float(np.sum(joint_probability * changes))
    second_moment = float(
        np.sum(
            joint_probability
            * (changes * changes + noise_change_variance)
        )
    )
    sigma_delta_ref = math.sqrt(
        max(second_moment - mean_change * mean_change, 0.0)
    )
    status = EXPECTED_STATUS[name]
    design = engine.NewDesign(
        kappa_m=KAPPA_M,
        x_support=base.x_support.copy(),
        group_probabilities=base.group_probabilities.copy(),
        donor_means=donor_means,
        treated_untreated_means=treated_means,
        treatment_effect=base.treatment_effect.copy(),
        true_weights=base.true_weights.copy(),
        residual_variances=base.residual_variances.copy(),
        att_true=float(base.att_true),
        population_trim_normalized_score_target=trim_target,
        population_kappa_m_median=float(np.median(condition)),
        population_kappa_m_p95=float(np.quantile(condition, 0.95)),
        population_kappa_mtm_median=float(np.median(condition**2)),
        population_kappa_mtm_p95=float(np.quantile(condition**2, 0.95)),
        population_sigma_max_median=float(np.median(singular[:, 0])),
        population_sigma_middle_median=float(np.median(singular[:, 1])),
        population_sigma_min_median=float(np.median(singular[:, -1])),
        population_rank_min=int(np.min(diagnostics["ranks"])),
        maximum_sc_residual=float(np.max(np.abs(true_residual))),
        maximum_pt_gap=float(np.max(diagnostics["pt_gap"])),
        weak_post_projection_median=float(
            np.median(diagnostics["weak_post_projection"])
        ),
        weak_post_projection_over_sigma_min_median=float(
            np.median(
                diagnostics["weak_post_projection"] / singular[:, -1]
            )
        ),
        post_scale=Q4_POST_SCALE,
        dgp_name=name,
        config_id=name,
        parameter_mode="bandwidth_grid",
        tuning_value=0.0,
        rho=0.0,
        eta=0.0,
        sigma_delta_ref=sigma_delta_ref,
        pre_sc_gap_multiplier=(
            PT_ONLY_SC_VIOLATION if name == "pt_only" else 0.0
        ),
        pt_holds_by_construction=status[0],
        sc_holds_by_construction=status[1],
        population_estimand=population_estimand,
        population_estimand_minus_att=population_bias,
        maximum_pre_affine_sc_residual_l2=float(
            np.max(diagnostics["pre_residual_l2"])
        ),
        median_pre_affine_sc_residual_l2=float(
            np.median(diagnostics["pre_residual_l2"])
        ),
        maximum_post_sc_residual_at_affine_weights=float(
            np.max(np.abs(diagnostics["post_residual"]))
        ),
        maximum_full_sc_residual_at_affine_weights=float(
            np.max(diagnostics["full_residual_max"])
        ),
        maximum_pseudo_weight_error_l2=float(
            np.max(diagnostics["pseudo_weight_error"])
        ),
    )
    return design


def build_configs(
    args: argparse.Namespace,
    n: int,
) -> tuple[engine.BandwidthConfig, ...]:
    selected = list(DGP_NAMES) if args.dgp == ["all"] else list(args.dgp)
    return tuple(
        engine.BandwidthConfig(
            name=name,
            n=int(n),
            bandwidth_coefficient=float(coefficient),
        )
        for name in selected
        for coefficient in args.bandwidth_coefficients
    )


def build_design(config: engine.BandwidthConfig) -> engine.BandwidthDesign:
    """Attach one bandwidth to one embedded Q4 population regime."""

    if config.name not in DGP_NAMES:
        raise ValueError(f"unknown DGP {config.name!r}")
    if config.n <= 0 or config.bandwidth_coefficient <= 0.0:
        raise ValueError("sample size and bandwidth coefficient must be positive")
    regime = _build_q4_regime(config.name)
    parent = {
        field.name: getattr(regime, field.name)
        for field in fields(engine.NewDesign)
    }
    parent["config_id"] = config.config_id
    design = engine.BandwidthDesign(
        **parent,
        population_n=int(config.n),
        dgp_source="quantile_top4_empirical_calibration_2025_09_16",
        bandwidth_coefficient=float(config.bandwidth_coefficient),
        bandwidth_rate_exponent=engine.BANDWIDTH_RATE_EXPONENT,
        effective_bandwidth=float(config.effective_bandwidth),
        bandwidth_kernel=engine.BANDWIDTH_KERNEL,
        bandwidth_formula=engine.BANDWIDTH_FORMULA,
    )
    validate_population_design(design)
    return design


def validate_population_design(design: engine.BandwidthDesign) -> None:
    """Audit Q4 geometry, exact assumption status, target, and bandwidth."""

    diagnostics = engine.affine_population_diagnostics(
        design.donor_means,
        design.treated_untreated_means,
        design.true_weights,
    )
    if not np.allclose(
        diagnostics["condition_numbers"], KAPPA_M, rtol=REL_TOL, atol=ABS_TOL
    ):
        raise AssertionError(f"{design.config_id}: kappa(M) mismatch")
    if not np.all(diagnostics["ranks"] == engine.N_DONORS - 1):
        raise AssertionError(f"{design.config_id}: rank deficiency")
    singular = diagnostics["singular_values"]
    if float(np.min(singular[:, -1] ** 2)) < 1.0 - ABS_TOL:
        raise AssertionError(f"{design.config_id}: lambda_min(M'M) is not strong")

    realized = (
        bool(np.max(diagnostics["pt_gap"]) <= ABS_TOL),
        bool(
            np.max(diagnostics["pre_residual_l2"]) <= ABS_TOL
            and np.max(diagnostics["full_residual_max"]) <= ABS_TOL
        ),
    )
    expected = EXPECTED_STATUS[design.dgp_name]
    if realized != expected:
        raise AssertionError(
            f"{design.config_id}: PT/SC status {realized}, expected {expected}"
        )
    if (
        design.pt_holds_by_construction,
        design.sc_holds_by_construction,
    ) != expected:
        raise AssertionError(f"{design.config_id}: stored status mismatch")

    expected_h = design.bandwidth_coefficient * design.population_n ** (
        -2.0 / 7.0
    )
    if not math.isclose(
        design.effective_bandwidth,
        expected_h,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
    ):
        raise AssertionError(f"{design.config_id}: bandwidth mismatch")
    if float(np.max(diagnostics["pseudo_weight_error"])) > ABS_TOL:
        raise AssertionError(f"{design.config_id}: pseudo-weight mismatch")
    if abs(design.population_estimand_minus_att) > ABS_TOL:
        raise AssertionError(f"{design.config_id}: exact DGP target mismatch")
    if not math.isclose(
        design.population_estimand,
        design.att_true,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
    ):
        raise AssertionError(f"{design.config_id}: target is not ATT")

    if design.dgp_name == "sc_only":
        donor_contrast = (
            design.donor_means[:, -1, : engine.N_DONORS - 1]
            - design.donor_means[:, -1, -1][:, None]
            - design.donor_means[:, -2, : engine.N_DONORS - 1]
            + design.donor_means[:, -2, -1][:, None]
        )
        expected = Q4_POST_SCALE * Q4_EMPIRICAL_POST_CONTRAST
        if not np.allclose(
            donor_contrast,
            expected[None, :],
            rtol=REL_TOL,
            atol=ABS_TOL,
        ):
            raise AssertionError("SC-only empirical post contrast changed")
    elif design.dgp_name == "pt_only":
        if not np.allclose(
            diagnostics["pre_residual_l2"],
            PT_ONLY_SC_VIOLATION,
            rtol=REL_TOL,
            atol=ABS_TOL,
        ):
            raise AssertionError("PT-only pre-period SC gap changed")
    elif design.dgp_name != "pt_sc_both":
        raise AssertionError("a non-exact DGP entered the active experiment")

    if tuple(DGP_NAMES) != ("sc_only", "pt_only", "pt_sc_both"):
        raise AssertionError("active DGP set is not the requested exact trio")


def initialize_worker(configs: tuple[engine.BandwidthConfig, ...]) -> None:
    global _WORKER_DGP_GROUPS
    engine._WORKER_CONFIG_COUNT = len(configs)
    engine._DESIGNS = {
        float(index): build_design(config)
        for index, config in enumerate(configs)
    }
    warmup = np.column_stack(
        [np.linspace(0.1 * column, 1.0 + 0.1 * column, engine.N_PRE)
         for column in range(engine.N_DONORS + 1)]
    )
    engine.affine_ridge_projected_weights(warmup)
    _WORKER_DGP_GROUPS = tuple(
        tuple(index for index, config in enumerate(configs) if config.name == name)
        for name in DGP_NAMES
        if any(config.name == name for config in configs)
    )


@engine.njit(cache=True, inline="always")
def _fast_local_linear_intercept(
    counts: np.ndarray,
    response_sums: np.ndarray,
    support: np.ndarray,
    evaluation: float,
    bandwidth: float,
) -> float:
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    t0 = 0.0
    t1 = 0.0
    for index in range(support.size):
        centered = support[index] - evaluation
        scaled = centered / bandwidth
        if abs(scaled) <= 1.0:
            kernel = 0.75 * (1.0 - scaled * scaled)
            weighted_count = kernel * counts[index]
            s0 += weighted_count
            s1 += weighted_count * centered
            s2 += weighted_count * centered * centered
            t0 += kernel * response_sums[index]
            t1 += kernel * centered * response_sums[index]
    ridge = engine.RIDGE_LLR * max((s0 + s2) / 2.0, 1.0)
    a00 = s0 + ridge
    a11 = s2 + ridge
    determinant = max(a00 * a11 - s1 * s1, ridge)
    return (a11 * t0 - s1 * t1) / determinant


@engine.njit(cache=True, inline="always")
def _fast_local_ratio(
    donor_counts: np.ndarray,
    treated_counts: np.ndarray,
    support: np.ndarray,
    evaluation: float,
    bandwidth: float,
) -> float:
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    t0 = 0.0
    t1 = 0.0
    for index in range(support.size):
        centered = support[index] - evaluation
        scaled = centered / bandwidth
        if abs(scaled) <= 1.0:
            kernel = 0.75 * (1.0 - scaled * scaled)
            weighted_donor = kernel * donor_counts[index]
            s0 += weighted_donor
            s1 += weighted_donor * centered
            s2 += weighted_donor * centered * centered
            weighted_treated = kernel * treated_counts[index]
            t0 += weighted_treated
            t1 += weighted_treated * centered
    determinant = s0 * s2 - s1 * s1
    adaptive_epsilon = 1.0e-6 * max((s0 + s2) / 2.0, 1.0)
    if abs(determinant) < adaptive_epsilon:
        if abs(s0) > adaptive_epsilon:
            return t0 / s0
        return t0 / (s0 + adaptive_epsilon)
    return (t0 * s2 - t1 * s1) / determinant


@engine.njit(cache=True)
def fast_estimate_didsc(
    x_index: np.ndarray,
    group: np.ndarray,
    folds: np.ndarray,
    outcomes: np.ndarray,
    observation_weights: np.ndarray,
    support: np.ndarray,
    eval_indices: np.ndarray,
    trim_low: float,
    trim_high: float,
    bandwidth: float,
) -> float:
    """Numba-equivalent implementation of the legacy score without diagnostics."""

    n = outcomes.shape[0]
    pi_numerator = 0.0
    for observation in range(n):
        if group[observation] == 0:
            pi_numerator += observation_weights[observation]
    pi_treated = pi_numerator / n
    if not np.isfinite(pi_treated) or pi_treated <= 0.0:
        raise ValueError("weighted treated share is invalid")

    fold_total = 0.0
    for fold in range(engine.N_FOLDS):
        train_counts = np.zeros((engine.N_GROUPS, engine.N_X))
        test_counts = np.zeros((engine.N_GROUPS, engine.N_X))
        train_sums = np.zeros(
            (engine.N_GROUPS, engine.N_PERIODS, engine.N_X)
        )
        test_sums = np.zeros(
            (engine.N_GROUPS, engine.N_PERIODS, engine.N_X)
        )
        trimmed_test_count = 0
        for observation in range(n):
            group_index = group[observation]
            covariate_index = x_index[observation]
            weight = observation_weights[observation]
            if folds[observation] == fold:
                test_counts[group_index, covariate_index] += weight
                for period in range(engine.N_PERIODS):
                    test_sums[group_index, period, covariate_index] += (
                        weight * outcomes[observation, period]
                    )
                observed_x = support[covariate_index]
                if observed_x > trim_low and observed_x < trim_high:
                    trimmed_test_count += 1
            else:
                train_counts[group_index, covariate_index] += weight
                for period in range(engine.N_PERIODS):
                    train_sums[group_index, period, covariate_index] += (
                        weight * outcomes[observation, period]
                    )
        if trimmed_test_count <= 0:
            raise ValueError("trimmed evaluation fold is empty")

        pooled_control_counts = np.zeros(engine.N_X)
        pooled_control_change_sums = np.zeros(engine.N_X)
        for donor_group in range(1, engine.N_GROUPS):
            for covariate_index in range(engine.N_X):
                pooled_control_counts[covariate_index] += train_counts[
                    donor_group, covariate_index
                ]
                pooled_control_change_sums[covariate_index] += (
                    train_sums[donor_group, -1, covariate_index]
                    - train_sums[donor_group, -2, covariate_index]
                )

        score_sum = 0.0
        for evaluation_position in range(eval_indices.size):
            covariate_index = eval_indices[evaluation_position]
            evaluation = support[covariate_index]
            moments = np.empty((engine.N_PRE, engine.N_GROUPS))
            for group_index in range(engine.N_GROUPS):
                for period in range(engine.N_PRE):
                    moments[period, group_index] = _fast_local_linear_intercept(
                        train_counts[group_index],
                        train_sums[group_index, period],
                        support,
                        evaluation,
                        bandwidth,
                    )
            donor_weights, _ = engine.affine_ridge_projected_weights(moments)
            nuisance_change = _fast_local_linear_intercept(
                pooled_control_counts,
                pooled_control_change_sums,
                support,
                evaluation,
                bandwidth,
            )
            for group_index in range(engine.N_GROUPS):
                if group_index == 0:
                    coefficient = 1.0
                else:
                    ratio = _fast_local_ratio(
                        train_counts[group_index],
                        train_counts[0],
                        support,
                        evaluation,
                        bandwidth,
                    )
                    coefficient = -donor_weights[group_index - 1] * ratio
                cell_contribution = (
                    test_sums[group_index, -1, covariate_index]
                    - test_sums[group_index, -2, covariate_index]
                    - nuisance_change * test_counts[group_index, covariate_index]
                )
                score_sum += coefficient * cell_contribution
        fold_total += score_sum / pi_treated / trimmed_test_count
    return fold_total / engine.N_FOLDS


def evaluation_geometry(
    design: engine.BandwidthDesign,
    latent: engine.LatentSample,
) -> tuple[np.ndarray, float, float]:
    observed_x = design.x_support[latent.x_index]
    trim_low, trim_high = np.quantile(observed_x, engine.TRIM_QUANTILES)
    observed = np.zeros(engine.N_X, dtype=bool)
    observed[latent.x_index] = True
    eval_indices = np.flatnonzero(
        (design.x_support > trim_low)
        & (design.x_support < trim_high)
        & observed
    ).astype(np.int64)
    return eval_indices, float(trim_low), float(trim_high)


def project_rows_to_simplex(raw_weights: np.ndarray) -> np.ndarray:
    """Vectorized Euclidean simplex projection, row by row."""

    ordered = np.sort(raw_weights, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1)
    multipliers = np.arange(1, raw_weights.shape[1] + 1, dtype=float)
    active = ordered * multipliers[None, :] > cumulative - 1.0
    rho = np.sum(active, axis=1) - 1
    rows = np.arange(raw_weights.shape[0])
    threshold = (cumulative[rows, rho] - 1.0) / (rho + 1.0)
    return np.maximum(raw_weights - threshold[:, None], 0.0)


def vectorized_estimate_from_aggregates(
    *,
    train_counts: np.ndarray,
    train_sums: np.ndarray,
    test_counts: np.ndarray,
    test_sums: np.ndarray,
    support: np.ndarray,
    eval_indices: np.ndarray,
    bandwidth: float,
    pi_treated: float,
    trimmed_test_count: int,
) -> float:
    """Evaluate one fold/bandwidth using batched local-linear algebra."""

    evaluations = support[eval_indices]
    centered = support[None, :] - evaluations[:, None]
    scaled = centered / bandwidth
    kernel = np.where(
        np.abs(scaled) <= 1.0,
        0.75 * (1.0 - scaled * scaled),
        0.0,
    )
    kernel_centered = kernel * centered
    kernel_centered_sq = kernel_centered * centered

    s0 = train_counts @ kernel.T
    s1 = train_counts @ kernel_centered.T
    s2 = train_counts @ kernel_centered_sq.T
    flat_sums = train_sums.reshape(engine.N_GROUPS * engine.N_PERIODS, engine.N_X)
    t0 = (flat_sums @ kernel.T).reshape(
        engine.N_GROUPS, engine.N_PERIODS, eval_indices.size
    )
    t1 = (flat_sums @ kernel_centered.T).reshape(
        engine.N_GROUPS, engine.N_PERIODS, eval_indices.size
    )
    ridge = engine.RIDGE_LLR * np.maximum((s0 + s2) / 2.0, 1.0)
    a00 = s0 + ridge
    a11 = s2 + ridge
    determinant = np.maximum(a00 * a11 - s1 * s1, ridge)
    intercepts = (
        a11[:, None, :] * t0 - s1[:, None, :] * t1
    ) / determinant[:, None, :]
    moments = np.transpose(
        intercepts[:, : engine.N_PRE, :],
        (2, 1, 0),
    )

    matrix_m = moments[:, :, 1 : engine.N_DONORS] - moments[:, :, [-1]]
    target = moments[:, :, 0] - moments[:, :, -1]
    gram = np.einsum("etj,etk->ejk", matrix_m, matrix_m)
    eigenvalues = np.linalg.eigvalsh(gram)
    lambda_min = np.maximum(eigenvalues[:, 0], 0.0)
    lambda_max = np.maximum(eigenvalues[:, -1], 0.0)
    condition_mtm = np.divide(
        lambda_max,
        lambda_min,
        out=np.full_like(lambda_max, np.inf),
        where=lambda_min > 0.0,
    )
    diagonal_scale = np.max(np.abs(np.diagonal(gram, axis1=1, axis2=2)), axis=1)
    ridge_sc = np.where(
        condition_mtm > 1.0e6,
        engine.RIDGE_SC * np.maximum(diagonal_scale, 1.0),
        engine.RIDGE_SC,
    )
    regularized = gram + ridge_sc[:, None, None] * np.eye(
        engine.N_DONORS - 1
    )[None, :, :]
    rhs = np.einsum("etj,et->ej", matrix_m, target)
    free_weights = np.linalg.solve(regularized, rhs[:, :, None])[:, :, 0]
    raw_weights = np.column_stack(
        [free_weights, 1.0 - np.sum(free_weights, axis=1)]
    )
    donor_weights = project_rows_to_simplex(raw_weights)

    pooled_counts = np.sum(train_counts[1:], axis=0)
    pooled_change_sums = np.sum(
        train_sums[1:, -1] - train_sums[1:, -2],
        axis=0,
    )
    nuisance_s0 = kernel @ pooled_counts
    nuisance_s1 = kernel_centered @ pooled_counts
    nuisance_s2 = kernel_centered_sq @ pooled_counts
    nuisance_t0 = kernel @ pooled_change_sums
    nuisance_t1 = kernel_centered @ pooled_change_sums
    nuisance_ridge = engine.RIDGE_LLR * np.maximum(
        (nuisance_s0 + nuisance_s2) / 2.0,
        1.0,
    )
    nuisance_a00 = nuisance_s0 + nuisance_ridge
    nuisance_a11 = nuisance_s2 + nuisance_ridge
    nuisance_det = np.maximum(
        nuisance_a00 * nuisance_a11 - nuisance_s1 * nuisance_s1,
        nuisance_ridge,
    )
    nuisance_change = (
        nuisance_a11 * nuisance_t0 - nuisance_s1 * nuisance_t1
    ) / nuisance_det

    treated_t0 = kernel @ train_counts[0]
    treated_t1 = kernel_centered @ train_counts[0]
    ratios = np.empty((eval_indices.size, engine.N_DONORS))
    for donor in range(engine.N_DONORS):
        donor_group = donor + 1
        ratio_det = (
            s0[donor_group] * s2[donor_group]
            - s1[donor_group] * s1[donor_group]
        )
        adaptive = 1.0e-6 * np.maximum(
            (s0[donor_group] + s2[donor_group]) / 2.0,
            1.0,
        )
        regular = np.abs(ratio_det) >= adaptive
        values = np.empty(eval_indices.size)
        values[regular] = (
            treated_t0[regular] * s2[donor_group, regular]
            - treated_t1[regular] * s1[donor_group, regular]
        ) / ratio_det[regular]
        fallback_regular = (~regular) & (np.abs(s0[donor_group]) > adaptive)
        values[fallback_regular] = (
            treated_t0[fallback_regular] / s0[donor_group, fallback_regular]
        )
        fallback_ridged = (~regular) & (~fallback_regular)
        values[fallback_ridged] = treated_t0[fallback_ridged] / (
            s0[donor_group, fallback_ridged] + adaptive[fallback_ridged]
        )
        ratios[:, donor] = values

    coefficients = np.column_stack(
        [np.ones(eval_indices.size), -donor_weights * ratios]
    )
    contributions = (
        test_sums[:, -1, eval_indices].T
        - test_sums[:, -2, eval_indices].T
        - nuisance_change[:, None] * test_counts[:, eval_indices].T
    )
    score_sum = float(np.sum(coefficients * contributions))
    return score_sum / pi_treated / trimmed_test_count


def vectorized_estimate_grid(
    *,
    latent: engine.LatentSample,
    outcomes: np.ndarray,
    observation_weights: np.ndarray,
    support: np.ndarray,
    eval_indices: np.ndarray,
    trim_low: float,
    trim_high: float,
    bandwidths: np.ndarray,
) -> np.ndarray:
    """Reuse each fold's cell aggregation across all bandwidths."""

    n = outcomes.shape[0]
    pi_treated = float(
        np.sum(observation_weights * (latent.group == 0)) / n
    )
    if not np.isfinite(pi_treated) or pi_treated <= 0.0:
        raise FloatingPointError("weighted treated share is invalid")
    estimates = np.zeros(bandwidths.size)
    observed_x = support[latent.x_index]
    for fold in range(engine.N_FOLDS):
        test = latent.folds == fold
        train = ~test
        train_counts, train_sums = engine.aggregate_panel_cells(
            latent, outcomes, observation_weights, train
        )
        test_counts, test_sums = engine.aggregate_panel_cells(
            latent, outcomes, observation_weights, test
        )
        trimmed_test_count = int(
            np.sum(test & (observed_x > trim_low) & (observed_x < trim_high))
        )
        if trimmed_test_count <= 0:
            raise FloatingPointError("trimmed evaluation fold is empty")
        for index, bandwidth in enumerate(bandwidths):
            estimates[index] += vectorized_estimate_from_aggregates(
                train_counts=train_counts,
                train_sums=train_sums,
                test_counts=test_counts,
                test_sums=test_sums,
                support=support,
                eval_indices=eval_indices,
                bandwidth=float(bandwidth),
                pi_treated=pi_treated,
                trimmed_test_count=trimmed_test_count,
            )
    return estimates / engine.N_FOLDS


def failed_outer_result(
    *,
    config_index: float,
    replication: int,
    base_seed: int,
    fold_hash: str,
    bootstraps: int,
    bootstrap_estimates: np.ndarray,
    bootstrap_errors: np.ndarray,
    failures: dict[int, str],
    error: Exception,
    started: float,
) -> engine.OuterResult:
    return engine.OuterResult(
        kappa_m=config_index,
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


def run_outer_task_fast(
    task: tuple[int, int, float, int, int],
) -> engine.OuterResult:
    replication, n, config_index, bootstraps, master_seed = task
    design = engine._DESIGNS[config_index]
    started = time.perf_counter()
    base_seed = engine.replication_seed(master_seed, n, replication)
    bootstrap_estimates = np.full(bootstraps, np.nan)
    bootstrap_errors = np.full(bootstraps, np.nan)
    failures: dict[int, str] = {}
    fold_hash = ""
    try:
        latent, outcomes, multiplier_rng = engine.generate_outer_sample(
            design, n, base_seed
        )
        fold_hash = latent.fold_hash
        point_estimate, diagnostics = engine.estimate_didsc(
            design,
            latent,
            outcomes,
            np.ones(n),
            collect_diagnostics=True,
        )
        eval_indices, trim_low, trim_high = evaluation_geometry(design, latent)
        bandwidth = design.bandwidth_coefficient * n ** (-2.0 / 7.0)
    except Exception as error:
        return failed_outer_result(
            config_index=config_index,
            replication=replication,
            base_seed=base_seed,
            fold_hash=fold_hash,
            bootstraps=bootstraps,
            bootstrap_estimates=bootstrap_estimates,
            bootstrap_errors=bootstrap_errors,
            failures=failures,
            error=error,
            started=started,
        )

    for bootstrap in range(bootstraps):
        weights = multiplier_rng.exponential(1.0, size=n)
        try:
            estimate = fast_estimate_didsc(
                latent.x_index,
                latent.group,
                latent.folds,
                outcomes,
                weights,
                design.x_support,
                eval_indices,
                trim_low,
                trim_high,
                bandwidth,
            )
            bootstrap_estimates[bootstrap] = estimate
            bootstrap_errors[bootstrap] = estimate - point_estimate
        except Exception as error:
            failures[bootstrap] = f"{type(error).__name__}: {error}"

    return engine.OuterResult(
        kappa_m=config_index,
        replication=replication,
        base_seed=base_seed,
        fold_hash=fold_hash,
        point_estimate=point_estimate,
        bootstrap_estimates=bootstrap_estimates,
        bootstrap_errors=bootstrap_errors,
        bootstrap_failures=failures,
        sample_kappa_m_median=float(diagnostics["sample_kappa_m_median"]),
        sample_kappa_m_p95=float(diagnostics["sample_kappa_m_p95"]),
        sample_kappa_m_max=float(diagnostics["sample_kappa_m_max"]),
        sample_sigma_max_median=float(diagnostics["sample_sigma_max_median"]),
        sample_sigma_middle_median=float(
            diagnostics["sample_sigma_middle_median"]
        ),
        sample_sigma_min_median=float(diagnostics["sample_sigma_min_median"]),
        sample_rank_min=int(diagnostics["sample_rank_min"]),
        sample_rank_deficient_share=float(
            diagnostics["sample_rank_deficient_share"]
        ),
        maximum_estimated_weight=float(diagnostics["maximum_estimated_weight"]),
        estimated_weight_l2_median=float(
            diagnostics["estimated_weight_l2_median"]
        ),
        raw_weight_l2_median=float(diagnostics["raw_weight_l2_median"]),
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


def run_replication_task_fast(
    task: tuple[int, int, int, int],
) -> list[engine.OuterResult]:
    replication, n, bootstraps, master_seed = task
    return [
        run_outer_task_fast(
            (replication, n, float(index), bootstraps, master_seed)
        )
        for index in range(engine._WORKER_CONFIG_COUNT)
    ]


def successful_outer_result(
    *,
    config_index: int,
    replication: int,
    base_seed: int,
    fold_hash: str,
    point_estimate: float,
    bootstrap_estimates: np.ndarray,
    bootstrap_errors: np.ndarray,
    bootstrap_failures: dict[int, str],
    diagnostics: dict[str, float | int],
    runtime_seconds: float,
) -> engine.OuterResult:
    return engine.OuterResult(
        kappa_m=float(config_index),
        replication=replication,
        base_seed=base_seed,
        fold_hash=fold_hash,
        point_estimate=point_estimate,
        bootstrap_estimates=bootstrap_estimates,
        bootstrap_errors=bootstrap_errors,
        bootstrap_failures=bootstrap_failures,
        sample_kappa_m_median=float(diagnostics["sample_kappa_m_median"]),
        sample_kappa_m_p95=float(diagnostics["sample_kappa_m_p95"]),
        sample_kappa_m_max=float(diagnostics["sample_kappa_m_max"]),
        sample_sigma_max_median=float(diagnostics["sample_sigma_max_median"]),
        sample_sigma_middle_median=float(
            diagnostics["sample_sigma_middle_median"]
        ),
        sample_sigma_min_median=float(diagnostics["sample_sigma_min_median"]),
        sample_rank_min=int(diagnostics["sample_rank_min"]),
        sample_rank_deficient_share=float(
            diagnostics["sample_rank_deficient_share"]
        ),
        maximum_estimated_weight=float(diagnostics["maximum_estimated_weight"]),
        estimated_weight_l2_median=float(
            diagnostics["estimated_weight_l2_median"]
        ),
        raw_weight_l2_median=float(diagnostics["raw_weight_l2_median"]),
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
        runtime_seconds=runtime_seconds,
    )


def run_replication_task_vectorized(
    task: tuple[int, int, int, int],
) -> list[engine.OuterResult]:
    """Evaluate all bandwidths jointly within each DGP and outer sample."""

    replication, n, bootstraps, master_seed = task
    base_seed = engine.replication_seed(master_seed, n, replication)
    results: list[engine.OuterResult | None] = [
        None for _ in range(engine._WORKER_CONFIG_COUNT)
    ]
    for config_indices in _WORKER_DGP_GROUPS:
        group_started = time.perf_counter()
        first_design = engine._DESIGNS[float(config_indices[0])]
        empty_estimates = np.full(bootstraps, np.nan)
        empty_errors = np.full(bootstraps, np.nan)
        try:
            latent, outcomes, multiplier_rng = engine.generate_outer_sample(
                first_design, n, base_seed
            )
            eval_indices, trim_low, trim_high = evaluation_geometry(
                first_design, latent
            )
            points = np.empty(len(config_indices))
            diagnostics_by_config: list[dict[str, float | int]] = []
            bandwidths = np.empty(len(config_indices))
            for position, config_index in enumerate(config_indices):
                design = engine._DESIGNS[float(config_index)]
                points[position], diagnostics = engine.estimate_didsc(
                    design,
                    latent,
                    outcomes,
                    np.ones(n),
                    collect_diagnostics=True,
                )
                diagnostics_by_config.append(diagnostics)
                bandwidths[position] = (
                    design.bandwidth_coefficient * n ** (-2.0 / 7.0)
                )
        except Exception as error:
            for config_index in config_indices:
                results[config_index] = failed_outer_result(
                    config_index=float(config_index),
                    replication=replication,
                    base_seed=base_seed,
                    fold_hash="",
                    bootstraps=bootstraps,
                    bootstrap_estimates=empty_estimates.copy(),
                    bootstrap_errors=empty_errors.copy(),
                    failures={},
                    error=error,
                    started=group_started,
                )
            continue

        bootstrap_matrix = np.full((len(config_indices), bootstraps), np.nan)
        error_matrix = np.full((len(config_indices), bootstraps), np.nan)
        failures_by_config: list[dict[int, str]] = [
            {} for _ in config_indices
        ]
        for bootstrap in range(bootstraps):
            weights = multiplier_rng.exponential(1.0, size=n)
            try:
                estimates = vectorized_estimate_grid(
                    latent=latent,
                    outcomes=outcomes,
                    observation_weights=weights,
                    support=first_design.x_support,
                    eval_indices=eval_indices,
                    trim_low=trim_low,
                    trim_high=trim_high,
                    bandwidths=bandwidths,
                )
                bootstrap_matrix[:, bootstrap] = estimates
                error_matrix[:, bootstrap] = estimates - points
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                for failures in failures_by_config:
                    failures[bootstrap] = message

        group_runtime = float(time.perf_counter() - group_started)
        runtime_per_config = group_runtime / len(config_indices)
        for position, config_index in enumerate(config_indices):
            results[config_index] = successful_outer_result(
                config_index=config_index,
                replication=replication,
                base_seed=base_seed,
                fold_hash=latent.fold_hash,
                point_estimate=float(points[position]),
                bootstrap_estimates=bootstrap_matrix[position],
                bootstrap_errors=error_matrix[position],
                bootstrap_failures=failures_by_config[position],
                diagnostics=diagnostics_by_config[position],
                runtime_seconds=runtime_per_config,
            )
    if any(result is None for result in results):
        raise AssertionError("worker did not populate every configuration")
    return [result for result in results if result is not None]


def validate_fast_estimator() -> float:
    """Compare accelerated and legacy scores on fixed random weighted samples."""

    base_seed = engine.replication_seed(2026073001, 2000, 987654)
    maximum_error = 0.0
    for dgp_index, dgp_name in enumerate(DGP_NAMES):
        for coefficient in (1.0, 3.5):
            config = engine.BandwidthConfig(dgp_name, 2000, coefficient)
            design = build_design(config)
            latent, outcomes, multiplier_rng = engine.generate_outer_sample(
                design, 2000, base_seed + dgp_index
            )
            eval_indices, trim_low, trim_high = evaluation_geometry(design, latent)
            for draw in range(2):
                weights = (
                    np.ones(2000)
                    if draw == 0
                    else multiplier_rng.exponential(1.0, size=2000)
                )
                legacy, _ = engine.estimate_didsc(
                    design, latent, outcomes, weights, collect_diagnostics=False
                )
                accelerated = fast_estimate_didsc(
                    latent.x_index,
                    latent.group,
                    latent.folds,
                    outcomes,
                    weights,
                    design.x_support,
                    eval_indices,
                    trim_low,
                    trim_high,
                    config.effective_bandwidth,
                )
                maximum_error = max(
                    maximum_error,
                    abs(legacy - accelerated),
                )
    if maximum_error > 2.0e-10:
        raise AssertionError(
            f"accelerated estimator differs from legacy by {maximum_error:.3e}"
        )
    return maximum_error


def validate_vectorized_estimator() -> float:
    """Validate the production grid evaluator against every legacy score."""

    base_seed = engine.replication_seed(2026073001, 2000, 123456)
    coefficients = (1.0, 3.5)
    maximum_error = 0.0
    for dgp_index, dgp_name in enumerate(DGP_NAMES):
        designs = [
            build_design(engine.BandwidthConfig(dgp_name, 2000, coefficient))
            for coefficient in coefficients
        ]
        latent, outcomes, multiplier_rng = engine.generate_outer_sample(
            designs[0], 2000, base_seed + dgp_index
        )
        eval_indices, trim_low, trim_high = evaluation_geometry(designs[0], latent)
        bandwidths = np.asarray(
            [design.effective_bandwidth for design in designs],
            dtype=float,
        )
        for draw in range(2):
            weights = (
                np.ones(2000)
                if draw == 0
                else multiplier_rng.exponential(1.0, size=2000)
            )
            accelerated = vectorized_estimate_grid(
                latent=latent,
                outcomes=outcomes,
                observation_weights=weights,
                support=designs[0].x_support,
                eval_indices=eval_indices,
                trim_low=trim_low,
                trim_high=trim_high,
                bandwidths=bandwidths,
            )
            for index, design in enumerate(designs):
                legacy, _ = engine.estimate_didsc(
                    design,
                    latent,
                    outcomes,
                    weights,
                    collect_diagnostics=False,
                )
                maximum_error = max(
                    maximum_error,
                    abs(legacy - accelerated[index]),
                )
    if maximum_error > 2.0e-10:
        raise AssertionError(
            f"vectorized estimator differs from legacy by {maximum_error:.3e}"
        )
    return maximum_error


def output_paths(prefix: Path) -> tuple[Path, Path, Path, Path]:
    prefix = prefix.expanduser().resolve()
    return (
        prefix.with_name(prefix.name + "_outer.csv"),
        prefix.with_name(prefix.name + "_bootstrap_diagnostics.csv"),
        prefix.with_name(prefix.name + "_coverage.csv"),
        prefix.with_name(prefix.name + "_population.csv"),
    )


BOOTSTRAP_DIAGNOSTIC_FIELDS = (
    "dgp_name",
    "config_id",
    "n",
    "outer_replication",
    "fold_hash",
    "requested_bootstraps",
    "valid_bootstraps",
    "bootstrap_failures",
    "bootstrap_failure_messages",
    "bootstrap_root_mean",
    "bootstrap_root_sd",
    "runtime_seconds",
)


def run_one_sample(
    args: argparse.Namespace,
    n: int,
    prefix: Path,
) -> engine.SampleOutput:
    configs = build_configs(args, n)
    designs = tuple(build_design(config) for config in configs)
    jobs = min(
        engine.available_cpu_count() if args.jobs == 0 else int(args.jobs),
        int(args.outer_reps),
    )
    if jobs <= 0:
        raise ValueError("worker count must be positive")

    outer_path, bootstrap_path, coverage_path, population_path = output_paths(prefix)
    outer_path.parent.mkdir(parents=True, exist_ok=True)
    existing = [
        path
        for path in (outer_path, bootstrap_path, coverage_path, population_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"refusing to overwrite: {existing}")
    engine.write_population_diagnostics(population_path, designs)

    tasks = [
        (replication, n, args.bootstraps, args.master_seed)
        for replication in range(args.outer_start, args.outer_start + args.outer_reps)
    ]
    count = len(configs)
    rows_by_config: dict[int, list[dict[str, object]]] = {
        index: [] for index in range(count)
    }
    valid_total = {index: 0 for index in range(count)}
    requested_total = {index: 0 for index in range(count)}
    root_sum = {index: 0.0 for index in range(count)}
    root_squares = {index: 0.0 for index in range(count)}
    root_count = {index: 0 for index in range(count)}
    worker_runtime = {index: 0.0 for index in range(count)}
    started = time.perf_counter()
    total_cells = len(tasks) * count
    completed = 0
    next_progress = 10

    with (
        outer_path.open("x", encoding="utf-8", newline="") as outer_handle,
        bootstrap_path.open("x", encoding="utf-8", newline="") as bootstrap_handle,
    ):
        outer_writer = csv.DictWriter(outer_handle, fieldnames=engine.outer_fieldnames())
        diagnostic_writer = csv.DictWriter(
            bootstrap_handle,
            fieldnames=list(BOOTSTRAP_DIAGNOSTIC_FIELDS),
        )
        outer_writer.writeheader()
        diagnostic_writer.writeheader()
        context = get_context("spawn")
        with context.Pool(
            processes=jobs,
            initializer=initialize_worker,
            initargs=(configs,),
        ) as pool:
            for batch in pool.imap_unordered(
                run_replication_task_vectorized,
                tasks,
                chunksize=1,
            ):
                if len({result.base_seed for result in batch}) != 1:
                    raise AssertionError("common-random-number seed audit failed")
                hashes = {result.fold_hash for result in batch if result.fold_hash}
                if len(hashes) > 1:
                    raise AssertionError("fixed-fold common-random-number audit failed")
                for result in batch:
                    index = int(round(float(result.kappa_m)))
                    design = designs[index]
                    row, finite_errors = engine.replication_row(
                        result,
                        design,
                        n,
                        args,
                    )
                    rows_by_config[index].append(row)
                    outer_writer.writerow(row)
                    valid = int(finite_errors.size)
                    valid_total[index] += valid
                    requested_total[index] += args.bootstraps
                    root_sum[index] += float(np.sum(finite_errors))
                    root_squares[index] += float(finite_errors @ finite_errors)
                    root_count[index] += valid
                    worker_runtime[index] += float(result.runtime_seconds)
                    diagnostic_writer.writerow(
                        {
                            "dgp_name": design.dgp_name,
                            "config_id": design.config_id,
                            "n": n,
                            "outer_replication": result.replication,
                            "fold_hash": result.fold_hash,
                            "requested_bootstraps": args.bootstraps,
                            "valid_bootstraps": valid,
                            "bootstrap_failures": args.bootstraps - valid,
                            "bootstrap_failure_messages": json.dumps(
                                result.bootstrap_failures,
                                sort_keys=True,
                            ),
                            "bootstrap_root_mean": row["bootstrap_root_mean"],
                            "bootstrap_root_sd": row["bootstrap_root_sd"],
                            "runtime_seconds": result.runtime_seconds,
                        }
                    )
                    completed += 1
                    percent = int(math.floor(100.0 * completed / total_cells))
                    while percent >= next_progress:
                        elapsed = time.perf_counter() - started
                        print(
                            f"n={n} progress={next_progress}% cells="
                            f"{completed}/{total_cells} elapsed={elapsed:.1f}s",
                            flush=True,
                        )
                        print(
                            "provisional basic coverage (ATT/population/trim): "
                            + engine.provisional_coverage_text(rows_by_config, designs),
                            flush=True,
                        )
                        outer_handle.flush()
                        bootstrap_handle.flush()
                        next_progress += 10

    wall_runtime = float(time.perf_counter() - started)
    summaries = [
        engine.summarize_configuration(
            n=n,
            attempted=args.outer_reps,
            rows=rows_by_config[index],
            design=design,
            valid_bootstraps_total=valid_total[index],
            requested_bootstraps_total=requested_total[index],
            root_sum=root_sum[index],
            root_sum_squares=root_squares[index],
            root_count=root_count[index],
            worker_runtime=worker_runtime[index],
            wall_runtime=wall_runtime,
            args=args,
        )
        for index, design in enumerate(designs)
    ]
    with coverage_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return engine.SampleOutput(
        n=n,
        outer_path=outer_path,
        bootstrap_path=bootstrap_path,
        coverage_path=coverage_path,
        population_path=population_path,
        summaries=summaries,
    )


def print_final_coverage(outputs: list[SampleOutput]) -> None:
    """Print final coverage for all interval methods and targets."""

    print("\nFinal coverage results", flush=True)
    print(
        "Each target row reports basic/symmetric/percentile/normal coverage.",
        flush=True,
    )
    for output in outputs:
        for summary in output.summaries:
            print(
                f"{output.n}  {summary['config_id']}  "
                f"{int(summary['basic_trim_score_target_valid'])}/"
                f"{int(summary['attempted_outer_replications'])}",
                flush=True,
            )
            for target in TARGET_NAMES:
                values = "/".join(
                    f"{float(summary[f'{method}_{target}_coverage']):.6f}"
                    for method in INTERVAL_METHODS
                )
                print(f"  {target}: {values}", flush=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dgp",
        nargs="+",
        choices=("all", *DGP_NAMES),
        default=["all"],
    )
    parser.add_argument("--n", type=int, nargs="+", default=list(DEFAULT_SAMPLE_SIZES))
    parser.add_argument("--outer-start", type=int, default=0)
    parser.add_argument("--outer-reps", type=int, default=500)
    parser.add_argument("--bootstraps", type=int, default=500)
    parser.add_argument(
        "--bandwidth-coefficients",
        "--h-grid",
        dest="bandwidth_coefficients",
        type=float,
        nargs="+",
        default=list(DEFAULT_BANDWIDTH_COEFFICIENTS),
    )
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--master-seed", type=int, default=2026073001)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--tag", type=str, default="production_v1")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if "all" in args.dgp and args.dgp != ["all"]:
        parser.error("all cannot be combined with named DGPs")
    if len(args.dgp) != len(set(args.dgp)):
        parser.error("DGP names must not contain duplicates")
    if args.outer_start < 0 or args.outer_reps <= 0 or args.bootstraps < 2:
        parser.error("invalid replication counts")
    if args.jobs < 0 or not 0.0 < args.alpha < 1.0:
        parser.error("invalid jobs or alpha")
    if args.master_seed < 0:
        parser.error("master seed must be nonnegative")
    if any(n <= 0 for n in args.n):
        parser.error("sample sizes must be positive")
    if len(args.n) != len(set(args.n)):
        parser.error("sample sizes must not contain duplicates")
    if any(c <= 0.0 or not np.isfinite(c) for c in args.bandwidth_coefficients):
        parser.error("bandwidth coefficients must be finite and positive")
    if len(args.bandwidth_coefficients) != len(
        set(float(value) for value in args.bandwidth_coefficients)
    ):
        parser.error("bandwidth coefficients must not contain duplicates")
    if not args.tag.strip():
        parser.error("tag must not be empty")
    args.parameter_mode = "bandwidth_grid"
    args.post_scale = Q4_POST_SCALE
    args.pre_sc_gap_multiplier = PT_ONLY_SC_VIOLATION
    return args


def audit(args: argparse.Namespace) -> None:
    print(
        "n config PT SC kappa(M) lambda_min_med hcoef h pop_bias",
        flush=True,
    )
    for n in args.n:
        for config in build_configs(args, n):
            design = build_design(config)
            lambda_min = design.population_sigma_min_median**2
            print(
                f"{n} {design.config_id} "
                f"{design.pt_holds_by_construction} "
                f"{design.sc_holds_by_construction} "
                f"{design.population_kappa_m_median:.6f} "
                f"{lambda_min:.6f} "
                f"{design.bandwidth_coefficient:.2f} "
                f"{design.effective_bandwidth:.6f} "
                f"{design.population_estimand_minus_att:.6f}",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    audit(args)
    if args.audit_only:
        return
    equivalence_error = max(
        validate_fast_estimator(),
        validate_vectorized_estimator(),
    )
    print(
        "Accelerated/legacy estimator maximum absolute difference: "
        f"{equivalence_error:.3e}",
        flush=True,
    )
    outputs: list[engine.SampleOutput] = []
    for n in args.n:
        shard = f"_start{args.outer_start}" if args.outer_start else ""
        prefix = args.output_dir / (
            f"q4bw_{args.tag}_n{n}_R{args.outer_reps}_B{args.bootstraps}"
            f"{shard}"
        )
        print(
            f"Starting n={n}, R={args.outer_reps}, B={args.bootstraps}, "
            f"cells={len(build_configs(args, n))}, jobs={args.jobs or 'auto'}",
            flush=True,
        )
        outputs.append(run_one_sample(args, n, prefix))
    engine.print_final_coverage(outputs)


if __name__ == "__main__":
    main()
