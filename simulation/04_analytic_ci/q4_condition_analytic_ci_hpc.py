#!/usr/bin/env python3
"""Feasible analytic CIs over the Quantile-top-4 condition-number path.

This self-contained HPC runner uses the SC-only population construction from
the manuscript's SC-only calibrated design and varies the direct
condition number kappa(M) over 1, 2, 4, 10, 20, 40, 80, and 160.  Synthetic
control holds exactly and parallel trends fails at every grid point.

For each population, the program computes two cross-fitted normal-Wald
intervals: the feasible PT influence-function interval and the feasible SC
influence-function interval from equations (A.11)--(A.13) of the paper.  The
PT interval is intentionally reported even though PT fails in this DGP.  It is
therefore a diagnostic interval, not a theoretically valid interval.  The SC
interval is theoretically valid for every fixed, full-rank population matrix.

The influence-function calculation requires a smooth synthetic-weight map.
Accordingly, weights are unrestricted affine least-squares weights with no
simplex projection and no SC ridge.  This differs from the multiplier-
bootstrap condition-number runner, whose point estimator uses ridge
stabilization and simplex projection.  Local-linear nuisance fits retain only
a tiny numerical ridge.  There is no trimming.  The point-estimator bandwidth
is always ``h_theta = 2.5*n**(-2/7)``.  Separate command-line options can set
the nuisance bandwidth used only inside the analytic standard error; its
default is ``n**(-1/5)``, as in the reported results. Two fixed cross-fitting
folds are used.

Outer samples, Gaussian residuals, and fold assignments are common across the
kappa grid.  Atomic checkpoint CSVs are refreshed at every 10 percent
milestone by default and can be resumed after preemption with ``--resume``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_BLAS_THREAD_COUNT = os.environ.get("Q4_CONDITION_ANALYTIC_BLAS_THREADS", "1")
if not _BLAS_THREAD_COUNT.isdigit() or int(_BLAS_THREAD_COUNT) < 1:
    raise ValueError("Q4_CONDITION_ANALYTIC_BLAS_THREADS must be positive")
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = _BLAS_THREAD_COUNT

import numpy as np


DGP_VERSION = "quantile_top4_sc_only_lower_smin_fixed_folds_v1"
EXPERIMENT_VERSION = "q4_condition_analytic_ci_paper_if_no_simplex_v1"
EXPERIMENT_PURPOSE = "feasible_pt_sc_analytic_ci_over_condition_number"
DONOR_CODES = (24, 33, 49, 51)
DONOR_STATES = "Maryland|New Hampshire|Utah|Virginia"
BASELINE_DONOR = "Virginia"
CONDITION_DEFINITION = "kappa(M)=sigma_max(M)/sigma_min(M)"
WEIGHT_ESTIMATOR = "unrestricted_affine_least_squares_no_ridge_no_simplex"
INFERENCE_DESCRIPTION = "normal_wald_ci_from_paper_influence_function"

N_GROUPS = 5
N_DONORS = 4
N_PRE = 5
N_PERIODS = 6
N_X = 101
N_FOLDS = 2

DEFAULT_KAPPA_GRID = (1.0, 2.0, 4.0, 10.0, 20.0, 40.0, 80.0, 160.0)
DEFAULT_SAMPLE_SIZES = (2000, 4000)
DEFAULT_REPLICATIONS = 500
DEFAULT_MASTER_SEED = 2026080201
DEFAULT_MILESTONE_PERCENT = 10

REFERENCE_AUGMENTATION_GAMMA = math.sqrt(29.077419588830004)
EMPIRICAL_KAPPA_REFERENCE = 13.070487184660164
EMPIRICAL_POST_CONTRAST = np.asarray(
    [0.211789512326912, 0.9801124738736031, 0.3509230190287762],
    dtype=float,
)
POST_SCALE = 3.0
TREATMENT_LEVEL = 1.0
TREATMENT_AMPLITUDE = 0.7
ATT_FINGERPRINT = 1.011932168518905

BANDWIDTH_COEFFICIENT = 2.5
BANDWIDTH_EXPONENT = -2.0 / 7.0
DEFAULT_VARIANCE_BANDWIDTH_COEFFICIENT = 1.0
DEFAULT_VARIANCE_BANDWIDTH_EXPONENT = -1.0 / 5.0
RIDGE_LLR = 1.0e-6
Z_975 = 1.959963984540054
ALPHA = 0.05

INTERVAL_METHODS = ("feasible_pt", "feasible_sc")
METHOD_BASIS = {"feasible_pt": "PT", "feasible_sc": "SC"}
ABS_TOL = 1.0e-10
REL_TOL = 1.0e-10


OUTER_FIELDS = (
    "experiment_version",
    "dgp_version",
    "experiment_purpose",
    "dgp_name",
    "pt_holds_by_construction",
    "sc_holds_by_construction",
    "donor_codes",
    "donor_states",
    "baseline_donor",
    "condition_definition",
    "target_kappa_m",
    "empirical_kappa_m_reference",
    "population_kappa_m_median",
    "population_kappa_m_p95",
    "population_sigma_max_median",
    "population_sigma_middle_median",
    "population_sigma_min_median",
    "population_rank_min",
    "weight_estimator",
    "inference_description",
    "n",
    "outer_replication",
    "master_seed",
    "base_seed_hex",
    "fold_hash",
    "latent_hash",
    "folds",
    "bandwidth_coefficient",
    "bandwidth_exponent",
    "effective_bandwidth",
    "variance_bandwidth_coefficient",
    "variance_bandwidth_exponent",
    "effective_variance_bandwidth",
    "local_linear_ridge",
    "sc_weight_ridge",
    "simplex_projection",
    "trimming",
    "alpha",
    "att_true",
    "point_estimate",
    "estimation_error",
    "sample_treated_share",
    "population_treated_share",
    "feasible_probabilities_recovered_from_direct_ratios",
    "feasible_ratio_minimum",
    "feasible_ratio_maximum",
    "feasible_kappa_median",
    "feasible_kappa_p95",
    "feasible_kappa_maximum",
    "feasible_sigma_min_median",
    "feasible_weight_minimum",
    "feasible_weight_maximum",
    "feasible_weight_l2_median",
    "feasible_psi1_sd",
    "feasible_psi2_sd",
    "feasible_cov_pt_psi1",
    "feasible_cov_pt_psi2",
    "feasible_cov_psi1_psi2",
    "sc_variance_uses_total_observation_influence",
    "interval_method",
    "assumption_basis",
    "theoretically_valid_in_dgp",
    "validity_reason",
    "analytic_se",
    "ci_low",
    "ci_high",
    "ci_length",
    "covers_att",
    "if_mean",
    "if_sd",
    "if_q025",
    "if_q975",
    "standardized_error",
    "status",
    "error_message",
    "runtime_seconds",
)


@dataclass(frozen=True)
class Design:
    """Population quantities for one SC-only condition-number DGP."""

    kappa_m: float
    x_support: np.ndarray
    group_probabilities: np.ndarray
    donor_means: np.ndarray
    treated_untreated_means: np.ndarray
    treatment_effect: np.ndarray
    true_weights: np.ndarray
    residual_variances: np.ndarray
    att_true: float
    population_kappa_m_median: float
    population_kappa_m_p95: float
    population_sigma_max_median: float
    population_sigma_middle_median: float
    population_sigma_min_median: float
    population_rank_min: int
    maximum_sc_residual: float
    maximum_pt_gap: float


@dataclass(frozen=True)
class NuisanceBundle:
    """Cross-fitted nuisance paths evaluated on the 101-point support."""

    p: np.ndarray
    ratio: np.ndarray
    m_pre: np.ndarray
    group_change: np.ndarray
    pooled_change: np.ndarray
    weights: np.ndarray
    matrix_m: np.ndarray
    target_m1: np.ndarray
    gram_inverse: np.ndarray
    free_weights: np.ndarray
    ratio_minimum: float
    ratio_maximum: float
    kappa_median: float
    kappa_p95: float
    kappa_maximum: float
    sigma_min_median: float


_DESIGN_CACHE: dict[tuple[float, ...], dict[float, Design]] = {}
_KERNEL_CACHE: dict[tuple[int, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


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
) -> np.ndarray:
    """Recreate the condition experiment's deterministic rank completion."""

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
        REFERENCE_AUGMENTATION_GAMMA
        * math.sqrt(N_PRE)
        * np.linalg.norm(fixed_effect_contrasts)
    )
    donor_loadings = np.zeros((N_DONORS - 2, N_DONORS), dtype=float)
    donor_loadings[:, : N_DONORS - 1] = completion_magnitude * c_matrix
    factor_with_post = np.vstack([h_matrix, h_matrix[-1]])
    return factor_with_post @ donor_loadings


def build_design(kappa_m: float) -> Design:
    """Construct one exact-SC population at the requested direct kappa(M)."""

    kappa_m = float(kappa_m)
    if not np.isfinite(kappa_m) or kappa_m < 1.0:
        raise ValueError("kappa_m must be finite and at least one")
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
    rank_completion = _rank_completion_paths(
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
        target_minimum = singular_values[0] / kappa_m
        adjusted = np.maximum(singular_values, target_minimum)
        adjusted[-1] = target_minimum
        target_m = left_vectors @ np.diag(adjusted) @ right_vectors
        donor_means[x_index, :N_PRE, : N_DONORS - 1] = (
            baseline[:N_PRE, None] + target_m
        )
        post_contrast = target_m[-1] + POST_SCALE * EMPIRICAL_POST_CONTRAST
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
    true_weights = 0.2 / N_DONORS + 0.8 * softmax_weights
    treated_untreated_means = np.einsum(
        "xd,xtd->xt",
        true_weights,
        donor_means,
    )
    treatment_effect = TREATMENT_LEVEL + TREATMENT_AMPLITUDE * np.sin(
        2.0 * np.pi * x_support
    )
    treated_probability = group_probabilities[:, 0]
    att_true = float(
        np.sum(treated_probability * treatment_effect)
        / np.sum(treated_probability)
    )

    matrix_m = (
        donor_means[:, :N_PRE, : N_DONORS - 1]
        - donor_means[:, :N_PRE, [-1]]
    )
    singular_values = np.linalg.svd(matrix_m, compute_uv=False)
    condition_numbers = singular_values[:, 0] / singular_values[:, -1]
    ranks = np.asarray(
        [np.linalg.matrix_rank(matrix_m[index]) for index in range(N_X)]
    )
    reconstructed = np.einsum("xd,xtd->xt", true_weights, donor_means)
    maximum_sc_residual = float(
        np.max(np.abs(treated_untreated_means - reconstructed))
    )
    donor_changes = donor_means[:, -1] - donor_means[:, -2]
    treated_change = (
        treated_untreated_means[:, -1]
        - treated_untreated_means[:, -2]
    )
    maximum_pt_gap = float(
        np.max(np.abs(treated_change[:, None] - donor_changes))
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
        population_kappa_m_median=float(np.median(condition_numbers)),
        population_kappa_m_p95=float(np.quantile(condition_numbers, 0.95)),
        population_sigma_max_median=float(np.median(singular_values[:, 0])),
        population_sigma_middle_median=float(np.median(singular_values[:, 1])),
        population_sigma_min_median=float(np.median(singular_values[:, 2])),
        population_rank_min=int(np.min(ranks)),
        maximum_sc_residual=maximum_sc_residual,
        maximum_pt_gap=maximum_pt_gap,
    )


def _population_pre_objects(
    design: Design,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return population pre means and the unrestricted affine weight map."""

    m_pre = np.empty((N_X, N_GROUPS, N_PRE), dtype=float)
    m_pre[:, 0] = design.treated_untreated_means[:, :N_PRE]
    m_pre[:, 1:] = np.transpose(design.donor_means[:, :N_PRE], (0, 2, 1))
    matrix_m = np.transpose(
        m_pre[:, 1:N_DONORS, :] - m_pre[:, [-1], :],
        (0, 2, 1),
    )
    target_m1 = m_pre[:, 0] - m_pre[:, -1]
    gram_inverse = np.empty((N_X, N_DONORS - 1, N_DONORS - 1))
    free_weights = np.empty((N_X, N_DONORS - 1))
    for x_index in range(N_X):
        gram = matrix_m[x_index].T @ matrix_m[x_index]
        gram_inverse[x_index] = np.linalg.inv(gram)
        free_weights[x_index] = (
            gram_inverse[x_index]
            @ matrix_m[x_index].T
            @ target_m1[x_index]
        )
    weights = np.column_stack(
        [free_weights, 1.0 - free_weights.sum(axis=1)]
    )
    return m_pre, matrix_m, target_m1, gram_inverse, free_weights, weights


def validate_population_design(design: Design) -> None:
    """Fail before simulation if the requested SC-only DGP has changed."""

    (
        _,
        matrix_m,
        _,
        _,
        _,
        affine_weights,
    ) = _population_pre_objects(design)
    singular_values = np.linalg.svd(matrix_m, compute_uv=False)
    condition_numbers = singular_values[:, 0] / singular_values[:, -1]
    if not np.allclose(
        condition_numbers,
        design.kappa_m,
        rtol=REL_TOL,
        atol=ABS_TOL,
    ):
        raise AssertionError("population kappa(M) does not equal its target")
    if design.population_rank_min != N_DONORS - 1:
        raise AssertionError("population donor matrix is not full rank")
    if design.maximum_sc_residual > ABS_TOL:
        raise AssertionError("synthetic control fails in the population DGP")
    if design.maximum_pt_gap <= 1.0e-8:
        raise AssertionError("parallel trends unexpectedly holds")
    if (
        float(np.min(design.true_weights)) <= 0.0
        or not np.allclose(
            design.true_weights.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise AssertionError("population synthetic weights are not interior")
    if not np.allclose(
        affine_weights,
        design.true_weights,
        rtol=2.0e-9,
        atol=2.0e-9,
    ):
        raise AssertionError("population affine weights changed")
    donor_changes = design.donor_means[:, -1] - design.donor_means[:, -2]
    post_contrasts = (
        donor_changes[:, : N_DONORS - 1] - donor_changes[:, [-1]]
    )
    if not np.allclose(
        post_contrasts,
        POST_SCALE * EMPIRICAL_POST_CONTRAST,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise AssertionError("calibrated donor post-change contrasts changed")
    weak_projections = np.empty(N_X)
    for x_index in range(N_X):
        _, _, right_vectors = np.linalg.svd(
            matrix_m[x_index],
            full_matrices=False,
        )
        weak_projections[x_index] = abs(
            float(right_vectors[-1] @ post_contrasts[x_index])
        )
    if float(np.max(weak_projections)) <= 1.0e-10:
        raise AssertionError("post change is irrelevant to the weak direction")
    if not math.isclose(
        design.att_true,
        ATT_FINGERPRINT,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
    ):
        raise AssertionError("ATT fingerprint changed")


def get_designs(kappas: tuple[float, ...]) -> dict[float, Design]:
    """Build and cache the requested condition-number populations."""

    key = tuple(float(value) for value in kappas)
    cached = _DESIGN_CACHE.get(key)
    if cached is not None:
        return cached
    designs = {value: build_design(value) for value in key}
    for design in designs.values():
        validate_population_design(design)
    reference = designs[key[0]]
    for design in designs.values():
        if not (
            np.array_equal(
                design.group_probabilities,
                reference.group_probabilities,
            )
            and np.array_equal(
                design.residual_variances,
                reference.residual_variances,
            )
            and np.array_equal(
                design.treatment_effect,
                reference.treatment_effect,
            )
            and np.array_equal(design.true_weights, reference.true_weights)
        ):
            raise AssertionError("common-random-number primitives vary by kappa")
    _DESIGN_CACHE[key] = designs
    return designs


def _kernel_geometry(
    support: np.ndarray,
    bandwidth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Epanechnikov local-linear geometry for all evaluations."""

    key = (support.size, float(bandwidth))
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    centered = support[None, :] - support[:, None]
    scaled = centered / bandwidth
    kernel = np.where(
        np.abs(scaled) <= 1.0,
        0.75 * (1.0 - scaled * scaled),
        0.0,
    )
    geometry = (
        kernel,
        kernel * centered,
        kernel * centered * centered,
    )
    _KERNEL_CACHE[key] = geometry
    return geometry


def _local_linear_intercepts(
    counts: np.ndarray,
    response_sums: np.ndarray,
    kernel: np.ndarray,
    kernel_centered: np.ndarray,
    kernel_centered_sq: np.ndarray,
) -> np.ndarray:
    """Fit local-linear intercepts for one design count and many responses."""

    responses = np.asarray(response_sums, dtype=float)
    one_response = responses.ndim == 1
    if one_response:
        responses = responses[None, :]
    s0 = np.einsum("xy,y->x", kernel, counts)
    s1 = np.einsum("xy,y->x", kernel_centered, counts)
    s2 = np.einsum("xy,y->x", kernel_centered_sq, counts)
    t0 = np.einsum("xy,ry->xr", kernel, responses)
    t1 = np.einsum("xy,ry->xr", kernel_centered, responses)
    ridge = RIDGE_LLR * np.maximum((s0 + s2) / 2.0, 1.0)
    a00 = s0 + ridge
    a11 = s2 + ridge
    determinant = a00 * a11 - s1 * s1
    determinant = np.where(np.abs(determinant) > ridge, determinant, ridge)
    fitted = (
        a11[:, None] * t0 - s1[:, None] * t1
    ) / determinant[:, None]
    if one_response:
        return fitted[:, 0]
    return fitted


def _local_ratio_path(
    donor_counts: np.ndarray,
    treated_counts: np.ndarray,
    kernel: np.ndarray,
    kernel_centered: np.ndarray,
    kernel_centered_sq: np.ndarray,
) -> np.ndarray:
    """Estimate p1(x)/pg(x) directly by local-linear regression."""

    s0 = np.einsum("xy,y->x", kernel, donor_counts)
    s1 = np.einsum("xy,y->x", kernel_centered, donor_counts)
    s2 = np.einsum("xy,y->x", kernel_centered_sq, donor_counts)
    t0 = np.einsum("xy,y->x", kernel, treated_counts)
    t1 = np.einsum("xy,y->x", kernel_centered, treated_counts)
    determinant = s0 * s2 - s1 * s1
    adaptive = 1.0e-6 * np.maximum((s0 + s2) / 2.0, 1.0)
    result = np.empty_like(s0)
    regular = np.abs(determinant) >= adaptive
    result[regular] = (
        t0[regular] * s2[regular] - t1[regular] * s1[regular]
    ) / determinant[regular]
    fallback = ~regular
    safe_s0 = np.where(
        np.abs(s0[fallback]) > adaptive[fallback],
        s0[fallback],
        s0[fallback] + adaptive[fallback],
    )
    result[fallback] = t0[fallback] / safe_s0
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("nonfinite treated-to-donor ratio")
    return result


def aggregate_panel_cells(
    x_index: np.ndarray,
    group: np.ndarray,
    outcomes: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate training observations by group and covariate support cell."""

    flat = group[mask] * N_X + x_index[mask]
    size = N_GROUPS * N_X
    counts = np.bincount(flat, minlength=size).reshape(N_GROUPS, N_X)
    sums = np.empty((N_GROUPS, N_PERIODS, N_X), dtype=float)
    for period in range(N_PERIODS):
        sums[:, period] = np.bincount(
            flat,
            weights=outcomes[mask, period],
            minlength=size,
        ).reshape(N_GROUPS, N_X)
    return counts.astype(float), sums


def _solve_affine_paths(
    m_pre: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Solve the paper's unrestricted affine weight map at every x."""

    matrix_m = np.transpose(
        m_pre[:, 1:N_DONORS, :] - m_pre[:, [-1], :],
        (0, 2, 1),
    )
    target_m1 = m_pre[:, 0] - m_pre[:, -1]
    gram_inverse = np.empty((N_X, N_DONORS - 1, N_DONORS - 1))
    free_weights = np.empty((N_X, N_DONORS - 1))
    condition_numbers = np.empty(N_X)
    minimum_singular_values = np.empty(N_X)
    for x_index in range(N_X):
        singular_values = np.linalg.svd(matrix_m[x_index], compute_uv=False)
        if singular_values[-1] <= 0.0:
            raise np.linalg.LinAlgError("rank-deficient feasible M(x)")
        condition_numbers[x_index] = singular_values[0] / singular_values[-1]
        minimum_singular_values[x_index] = singular_values[-1]
        gram = matrix_m[x_index].T @ matrix_m[x_index]
        gram_inverse[x_index] = np.linalg.inv(gram)
        free_weights[x_index] = (
            gram_inverse[x_index]
            @ matrix_m[x_index].T
            @ target_m1[x_index]
        )
    weights = np.column_stack(
        [free_weights, 1.0 - free_weights.sum(axis=1)]
    )
    if not np.all(np.isfinite(weights)):
        raise FloatingPointError("nonfinite unrestricted affine weights")
    return (
        matrix_m,
        target_m1,
        gram_inverse,
        free_weights,
        weights,
        condition_numbers,
        minimum_singular_values,
    )


def fit_feasible_nuisances(
    design: Design,
    n: int,
    counts: np.ndarray,
    sums: np.ndarray,
    bandwidth_coefficient: float = BANDWIDTH_COEFFICIENT,
    bandwidth_exponent: float = BANDWIDTH_EXPONENT,
) -> NuisanceBundle:
    """Estimate all nuisance functions on one cross-fitting training fold."""

    bandwidth = bandwidth_coefficient * n**bandwidth_exponent
    kernel, kernel_centered, kernel_centered_sq = _kernel_geometry(
        design.x_support,
        bandwidth,
    )
    m_pre = np.empty((N_X, N_GROUPS, N_PRE), dtype=float)
    group_change = np.empty((N_X, N_GROUPS), dtype=float)
    for group_index in range(N_GROUPS):
        m_pre[:, group_index] = _local_linear_intercepts(
            counts[group_index],
            sums[group_index, :N_PRE],
            kernel,
            kernel_centered,
            kernel_centered_sq,
        )
        change_sums = sums[group_index, -1] - sums[group_index, -2]
        group_change[:, group_index] = _local_linear_intercepts(
            counts[group_index],
            change_sums,
            kernel,
            kernel_centered,
            kernel_centered_sq,
        )

    pooled_counts = counts[1:].sum(axis=0)
    pooled_change_sums = (
        sums[1:, -1].sum(axis=0) - sums[1:, -2].sum(axis=0)
    )
    pooled_change = _local_linear_intercepts(
        pooled_counts,
        pooled_change_sums,
        kernel,
        kernel_centered,
        kernel_centered_sq,
    )
    ratio = np.column_stack(
        [
            _local_ratio_path(
                counts[donor_index],
                counts[0],
                kernel,
                kernel_centered,
                kernel_centered_sq,
            )
            for donor_index in range(1, N_GROUPS)
        ]
    )
    if float(np.min(ratio)) <= 0.0:
        raise FloatingPointError("nonpositive feasible propensity-score ratio")
    inverse_ratio = 1.0 / ratio
    p_treated = 1.0 / (1.0 + inverse_ratio.sum(axis=1))
    p = np.column_stack([p_treated, p_treated[:, None] * inverse_ratio])
    if (
        float(np.min(p)) <= 0.0
        or not np.all(np.isfinite(p))
        or not np.allclose(p.sum(axis=1), 1.0, atol=1.0e-12, rtol=0.0)
    ):
        raise FloatingPointError("invalid probabilities recovered from ratios")
    (
        matrix_m,
        target_m1,
        gram_inverse,
        free_weights,
        weights,
        condition_numbers,
        minimum_singular_values,
    ) = _solve_affine_paths(m_pre)
    return NuisanceBundle(
        p=p,
        ratio=ratio,
        m_pre=m_pre,
        group_change=group_change,
        pooled_change=pooled_change,
        weights=weights,
        matrix_m=matrix_m,
        target_m1=target_m1,
        gram_inverse=gram_inverse,
        free_weights=free_weights,
        ratio_minimum=float(np.min(ratio)),
        ratio_maximum=float(np.max(ratio)),
        kappa_median=float(np.median(condition_numbers)),
        kappa_p95=float(np.quantile(condition_numbers, 0.95)),
        kappa_maximum=float(np.max(condition_numbers)),
        sigma_min_median=float(np.median(minimum_singular_values)),
    )


def influence_parts(
    bundle: NuisanceBundle,
    x_index: np.ndarray,
    group: np.ndarray,
    outcomes: np.ndarray,
    pi_treated: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the score numerator and SC corrections (A.11)--(A.12)."""

    if not np.isfinite(pi_treated) or pi_treated <= 0.0:
        raise FloatingPointError("invalid treated share")
    n = group.size
    treated = group == 0
    donor_indicators = np.column_stack(
        [group == donor_index for donor_index in range(1, N_GROUPS)]
    ).astype(float)
    weights = bundle.weights[x_index]
    ratios = bundle.ratio[x_index]
    donor_gaps = (
        bundle.group_change[x_index, 1:]
        - bundle.pooled_change[x_index, None]
    )

    score_coefficient = (
        treated.astype(float)
        - np.sum(weights * ratios * donor_indicators, axis=1)
    )
    delta_y = outcomes[:, -1] - outcomes[:, -2]
    score_numerator = score_coefficient * (
        delta_y - bundle.pooled_change[x_index]
    )

    psi1_core = np.sum(
        weights
        * (treated[:, None].astype(float) - ratios * donor_indicators)
        * donor_gaps,
        axis=1,
    )
    psi1 = -psi1_core / pi_treated

    fitted_pre = bundle.m_pre[x_index, group]
    residual_pre = outcomes[:, :N_PRE] - fitted_pre
    p_observed = bundle.p[x_index, group]
    scaled_residual = residual_pre / p_observed[:, None]
    a1 = np.zeros((n, N_PRE), dtype=float)
    matrix_a = np.zeros((n, N_PRE, N_DONORS - 1), dtype=float)
    a1[treated] = scaled_residual[treated]
    for donor_index in range(1, N_DONORS):
        mask = group == donor_index
        matrix_a[mask, :, donor_index - 1] = scaled_residual[mask]
    baseline = group == N_DONORS
    a1[baseline] = -scaled_residual[baseline]
    matrix_a[baseline] = -scaled_residual[baseline, :, None]

    matrix_m = bundle.matrix_m[x_index]
    target_m1 = bundle.target_m1[x_index]
    gram_inverse = bundle.gram_inverse[x_index]
    free_weights = bundle.free_weights[x_index]
    mt_a1 = np.einsum("nti,nt->ni", matrix_m, a1)
    p1 = np.einsum("nij,nj->ni", gram_inverse, mt_a1)
    at_m1 = np.einsum("nti,nt->ni", matrix_a, target_m1)
    p2 = np.einsum("nij,nj->ni", gram_inverse, at_m1)
    at_m = np.einsum("nti,ntj->nij", matrix_a, matrix_m)
    mt_a = np.einsum("nti,ntj->nij", matrix_m, matrix_a)
    derivative_gram = at_m + mt_a
    derivative_rhs = np.einsum("nij,nj->ni", derivative_gram, free_weights)
    p3 = -np.einsum("nij,nj->ni", gram_inverse, derivative_rhs)
    p_total = p1 + p2 + p3
    full_weight_influence = np.column_stack(
        [p_total, -p_total.sum(axis=1)]
    )
    if float(np.max(np.abs(full_weight_influence.sum(axis=1)))) > 1.0e-12:
        raise AssertionError("affine weight influence does not sum to zero")
    u = bundle.p[x_index, 0][:, None] * donor_gaps
    psi2 = -np.sum(u * full_weight_influence, axis=1) / pi_treated
    if not (
        np.all(np.isfinite(score_numerator))
        and np.all(np.isfinite(psi1))
        and np.all(np.isfinite(psi2))
    ):
        raise FloatingPointError("nonfinite influence-function component")
    return score_numerator, psi1, psi2


def build_oracle_nuisances(design: Design) -> NuisanceBundle:
    """Construct population nuisance paths for internal implementation audits."""

    (
        m_pre,
        matrix_m,
        target_m1,
        gram_inverse,
        free_weights,
        weights,
    ) = _population_pre_objects(design)
    group_change = np.empty((N_X, N_GROUPS), dtype=float)
    group_change[:, 0] = (
        design.treated_untreated_means[:, -1]
        - design.treated_untreated_means[:, -2]
    )
    group_change[:, 1:] = (
        design.donor_means[:, -1] - design.donor_means[:, -2]
    )
    donor_probabilities = design.group_probabilities[:, 1:]
    pooled_change = np.sum(
        donor_probabilities * group_change[:, 1:],
        axis=1,
    ) / donor_probabilities.sum(axis=1)
    ratio = (
        design.group_probabilities[:, [0]]
        / design.group_probabilities[:, 1:]
    )
    singular_values = np.linalg.svd(matrix_m, compute_uv=False)
    condition_numbers = singular_values[:, 0] / singular_values[:, -1]
    return NuisanceBundle(
        p=design.group_probabilities.copy(),
        ratio=ratio,
        m_pre=m_pre,
        group_change=group_change,
        pooled_change=pooled_change,
        weights=weights,
        matrix_m=matrix_m,
        target_m1=target_m1,
        gram_inverse=gram_inverse,
        free_weights=free_weights,
        ratio_minimum=float(np.min(ratio)),
        ratio_maximum=float(np.max(ratio)),
        kappa_median=float(np.median(condition_numbers)),
        kappa_p95=float(np.quantile(condition_numbers, 0.95)),
        kappa_maximum=float(np.max(condition_numbers)),
        sigma_min_median=float(np.median(singular_values[:, -1])),
    )


def replication_seed(master_seed: int, n: int, replication: int) -> int:
    """Return a seed stable to worker count and completion ordering."""

    sequence = np.random.SeedSequence(
        [int(master_seed), int(n), int(replication)]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def generate_common_sample(
    design: Design,
    n: int,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    """Draw X, group, residuals, and folds shared by the entire kappa grid."""

    sequence = np.random.SeedSequence(int(base_seed))
    data_stream, residual_stream, fold_stream, _ = sequence.spawn(4)
    data_rng = np.random.default_rng(data_stream)
    residual_rng = np.random.default_rng(residual_stream)
    fold_rng = np.random.default_rng(fold_stream)
    x_index = data_rng.integers(0, N_X, size=n)
    probabilities = design.group_probabilities[x_index]
    uniforms = data_rng.random(n)
    cumulative = np.cumsum(probabilities, axis=1)
    group = np.sum(uniforms[:, None] > cumulative, axis=1).astype(np.int64)
    group = np.minimum(group, N_GROUPS - 1)
    residuals = residual_rng.normal(size=(n, N_PERIODS)) * np.sqrt(
        design.residual_variances
    )[None, :]
    folds = fold_rng.integers(0, N_FOLDS, size=n, dtype=np.int8)
    if any(not np.any(folds == fold) for fold in range(N_FOLDS)):
        raise FloatingPointError("empty cross-fitting fold")
    fold_hash = hashlib.sha256(folds.tobytes()).hexdigest()
    latent_hash = hashlib.sha256(
        x_index.tobytes()
        + group.tobytes()
        + residuals.tobytes()
        + folds.tobytes()
    ).hexdigest()
    return x_index, group, residuals, folds, fold_hash, latent_hash


def construct_outcomes(
    design: Design,
    x_index: np.ndarray,
    group: np.ndarray,
    residuals: np.ndarray,
) -> np.ndarray:
    """Construct the observed panel under one condition-number population."""

    outcomes = np.empty_like(residuals)
    treated = group == 0
    outcomes[treated] = design.treated_untreated_means[x_index[treated]]
    for donor_index in range(N_DONORS):
        mask = group == donor_index + 1
        outcomes[mask] = design.donor_means[
            x_index[mask],
            :,
            donor_index,
        ]
    outcomes += residuals
    outcomes[treated, -1] += design.treatment_effect[x_index[treated]]
    return outcomes


def _variance_record(
    point_estimate: float,
    att_true: float,
    influence: np.ndarray,
) -> dict[str, float | bool]:
    """Return one normal-Wald interval and influence diagnostics."""

    n = influence.size
    if n <= 1 or not np.all(np.isfinite(influence)):
        raise FloatingPointError("invalid influence vector")
    if_sd = float(np.std(influence, ddof=1))
    standard_error = if_sd / math.sqrt(n)
    if not np.isfinite(standard_error) or standard_error <= 0.0:
        raise FloatingPointError("invalid analytic standard error")
    ci_low = float(point_estimate - Z_975 * standard_error)
    ci_high = float(point_estimate + Z_975 * standard_error)
    return {
        "analytic_se": standard_error,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_length": ci_high - ci_low,
        "covers_att": bool(ci_low <= att_true <= ci_high),
        "if_mean": float(np.mean(influence)),
        "if_sd": if_sd,
        "if_q025": float(np.quantile(influence, 0.025)),
        "if_q975": float(np.quantile(influence, 0.975)),
    }


def _method_is_valid(method: str) -> bool:
    """Return theoretical validity under the exact-SC, non-PT population."""

    return method == "feasible_sc"


def _method_validity_reason(method: str) -> str:
    """Describe why an interval is valid or intentionally diagnostic."""

    return "SC_holds" if method == "feasible_sc" else "invalid_PT_fails"


def evaluate_design(
    design: Design,
    n: int,
    replication: int,
    master_seed: int,
    base_seed: int,
    x_index: np.ndarray,
    group: np.ndarray,
    residuals: np.ndarray,
    folds: np.ndarray,
    fold_hash: str,
    latent_hash: str,
    variance_bandwidth_coefficient: float,
    variance_bandwidth_exponent: float,
) -> list[dict[str, Any]]:
    """Estimate the ATT and the two feasible analytic intervals."""

    started = time.perf_counter()
    outcomes = construct_outcomes(design, x_index, group, residuals)
    pi_hat = float(np.mean(group == 0))
    pi_true = float(np.mean(design.group_probabilities[:, 0]))
    if pi_hat <= 0.0:
        raise FloatingPointError("sample contains no treated observations")

    point_score_feasible = np.empty(n)
    variance_score_feasible = np.empty(n)
    psi1_feasible = np.empty(n)
    psi2_feasible = np.empty(n)
    variance_fold_bundles: list[NuisanceBundle] = []
    for fold in range(N_FOLDS):
        test = folds == fold
        train = ~test
        if not np.any(test) or not np.any(train):
            raise FloatingPointError("empty cross-fitting fold")
        counts, sums = aggregate_panel_cells(
            x_index,
            group,
            outcomes,
            train,
        )
        point_bundle = fit_feasible_nuisances(
            design,
            n,
            counts,
            sums,
            BANDWIDTH_COEFFICIENT,
            BANDWIDTH_EXPONENT,
        )
        same_bandwidth = (
            variance_bandwidth_coefficient == BANDWIDTH_COEFFICIENT
            and variance_bandwidth_exponent == BANDWIDTH_EXPONENT
        )
        variance_bundle = (
            point_bundle
            if same_bandwidth
            else fit_feasible_nuisances(
                design,
                n,
                counts,
                sums,
                variance_bandwidth_coefficient,
                variance_bandwidth_exponent,
            )
        )
        variance_fold_bundles.append(variance_bundle)
        point_score, _, _ = influence_parts(
            point_bundle,
            x_index[test],
            group[test],
            outcomes[test],
            pi_hat,
        )
        variance_score, psi1, psi2 = influence_parts(
            variance_bundle,
            x_index[test],
            group[test],
            outcomes[test],
            pi_hat,
        )
        point_score_feasible[test] = point_score
        variance_score_feasible[test] = variance_score
        psi1_feasible[test] = psi1
        psi2_feasible[test] = psi2

    point_estimate = float(np.mean(point_score_feasible) / pi_hat)
    if not np.isfinite(point_estimate):
        raise FloatingPointError("nonfinite ATT estimate")
    feasible_pt = (
        variance_score_feasible - point_estimate * (group == 0)
    ) / pi_hat
    feasible_sc = feasible_pt + psi1_feasible + psi2_feasible
    if same_bandwidth and abs(float(np.mean(feasible_pt))) > 1.0e-10:
        raise AssertionError("feasible PT influence function is not centered")

    feasible_covariance = np.cov(
        np.vstack([feasible_pt, psi1_feasible, psi2_feasible]),
        ddof=1,
    )
    common = {
        "experiment_version": EXPERIMENT_VERSION,
        "dgp_version": DGP_VERSION,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "dgp_name": "sc_only",
        "pt_holds_by_construction": False,
        "sc_holds_by_construction": True,
        "donor_codes": "|".join(str(code) for code in DONOR_CODES),
        "donor_states": DONOR_STATES,
        "baseline_donor": BASELINE_DONOR,
        "condition_definition": CONDITION_DEFINITION,
        "target_kappa_m": design.kappa_m,
        "empirical_kappa_m_reference": EMPIRICAL_KAPPA_REFERENCE,
        "population_kappa_m_median": design.population_kappa_m_median,
        "population_kappa_m_p95": design.population_kappa_m_p95,
        "population_sigma_max_median": design.population_sigma_max_median,
        "population_sigma_middle_median": design.population_sigma_middle_median,
        "population_sigma_min_median": design.population_sigma_min_median,
        "population_rank_min": design.population_rank_min,
        "weight_estimator": WEIGHT_ESTIMATOR,
        "inference_description": INFERENCE_DESCRIPTION,
        "n": n,
        "outer_replication": replication,
        "master_seed": master_seed,
        "base_seed_hex": f"0x{base_seed:016x}",
        "fold_hash": fold_hash,
        "latent_hash": latent_hash,
        "folds": N_FOLDS,
        "bandwidth_coefficient": BANDWIDTH_COEFFICIENT,
        "bandwidth_exponent": BANDWIDTH_EXPONENT,
        "effective_bandwidth": BANDWIDTH_COEFFICIENT * n ** BANDWIDTH_EXPONENT,
        "variance_bandwidth_coefficient": variance_bandwidth_coefficient,
        "variance_bandwidth_exponent": variance_bandwidth_exponent,
        "effective_variance_bandwidth": (
            variance_bandwidth_coefficient * n**variance_bandwidth_exponent
        ),
        "local_linear_ridge": RIDGE_LLR,
        "sc_weight_ridge": 0.0,
        "simplex_projection": False,
        "trimming": "none_theorem_matched",
        "alpha": ALPHA,
        "att_true": design.att_true,
        "point_estimate": point_estimate,
        "estimation_error": point_estimate - design.att_true,
        "sample_treated_share": pi_hat,
        "population_treated_share": pi_true,
        "feasible_probabilities_recovered_from_direct_ratios": True,
        "feasible_ratio_minimum": float(
            min(bundle.ratio_minimum for bundle in variance_fold_bundles)
        ),
        "feasible_ratio_maximum": float(
            max(bundle.ratio_maximum for bundle in variance_fold_bundles)
        ),
        "feasible_kappa_median": float(
            np.mean(
                [bundle.kappa_median for bundle in variance_fold_bundles]
            )
        ),
        "feasible_kappa_p95": float(
            max(bundle.kappa_p95 for bundle in variance_fold_bundles)
        ),
        "feasible_kappa_maximum": float(
            max(bundle.kappa_maximum for bundle in variance_fold_bundles)
        ),
        "feasible_sigma_min_median": float(
            np.mean(
                [
                    bundle.sigma_min_median
                    for bundle in variance_fold_bundles
                ]
            )
        ),
        "feasible_weight_minimum": float(
            min(np.min(bundle.weights) for bundle in variance_fold_bundles)
        ),
        "feasible_weight_maximum": float(
            max(np.max(bundle.weights) for bundle in variance_fold_bundles)
        ),
        "feasible_weight_l2_median": float(
            np.mean(
                [
                    np.median(np.linalg.norm(bundle.weights, axis=1))
                    for bundle in variance_fold_bundles
                ]
            )
        ),
        "feasible_psi1_sd": float(np.std(psi1_feasible, ddof=1)),
        "feasible_psi2_sd": float(np.std(psi2_feasible, ddof=1)),
        "feasible_cov_pt_psi1": float(feasible_covariance[0, 1]),
        "feasible_cov_pt_psi2": float(feasible_covariance[0, 2]),
        "feasible_cov_psi1_psi2": float(feasible_covariance[1, 2]),
        "sc_variance_uses_total_observation_influence": True,
    }
    influence_by_method = {
        "feasible_pt": feasible_pt,
        "feasible_sc": feasible_sc,
    }
    runtime = float(time.perf_counter() - started)
    rows: list[dict[str, Any]] = []
    for method in INTERVAL_METHODS:
        interval = _variance_record(
            point_estimate,
            design.att_true,
            influence_by_method[method],
        )
        standard_error = float(interval["analytic_se"])
        rows.append(
            {
                **common,
                "interval_method": method,
                "assumption_basis": METHOD_BASIS[method],
                "theoretically_valid_in_dgp": _method_is_valid(method),
                "validity_reason": _method_validity_reason(method),
                **interval,
                "standardized_error": (
                    point_estimate - design.att_true
                ) / standard_error,
                "status": "valid",
                "error_message": "",
                "runtime_seconds": runtime,
            }
        )
    return rows


def failed_design_rows(
    design: Design,
    n: int,
    replication: int,
    master_seed: int,
    base_seed: int,
    fold_hash: str,
    latent_hash: str,
    variance_bandwidth_coefficient: float,
    variance_bandwidth_exponent: float,
    error: Exception,
) -> list[dict[str, Any]]:
    """Return method-specific failure rows without dropping a grid cell."""

    message = f"{type(error).__name__}: {error}"
    return [
        {
            "experiment_version": EXPERIMENT_VERSION,
            "dgp_version": DGP_VERSION,
            "experiment_purpose": EXPERIMENT_PURPOSE,
            "dgp_name": "sc_only",
            "pt_holds_by_construction": False,
            "sc_holds_by_construction": True,
            "donor_codes": "|".join(str(code) for code in DONOR_CODES),
            "donor_states": DONOR_STATES,
            "baseline_donor": BASELINE_DONOR,
            "condition_definition": CONDITION_DEFINITION,
            "target_kappa_m": design.kappa_m,
            "empirical_kappa_m_reference": EMPIRICAL_KAPPA_REFERENCE,
            "population_kappa_m_median": design.population_kappa_m_median,
            "population_kappa_m_p95": design.population_kappa_m_p95,
            "population_sigma_max_median": design.population_sigma_max_median,
            "population_sigma_middle_median": design.population_sigma_middle_median,
            "population_sigma_min_median": design.population_sigma_min_median,
            "population_rank_min": design.population_rank_min,
            "weight_estimator": WEIGHT_ESTIMATOR,
            "inference_description": INFERENCE_DESCRIPTION,
            "n": n,
            "outer_replication": replication,
            "master_seed": master_seed,
            "base_seed_hex": f"0x{base_seed:016x}",
            "fold_hash": fold_hash,
            "latent_hash": latent_hash,
            "folds": N_FOLDS,
            "bandwidth_coefficient": BANDWIDTH_COEFFICIENT,
            "bandwidth_exponent": BANDWIDTH_EXPONENT,
            "effective_bandwidth": BANDWIDTH_COEFFICIENT * n ** BANDWIDTH_EXPONENT,
            "variance_bandwidth_coefficient": variance_bandwidth_coefficient,
            "variance_bandwidth_exponent": variance_bandwidth_exponent,
            "effective_variance_bandwidth": (
                variance_bandwidth_coefficient * n**variance_bandwidth_exponent
            ),
            "local_linear_ridge": RIDGE_LLR,
            "sc_weight_ridge": 0.0,
            "simplex_projection": False,
            "trimming": "none_theorem_matched",
            "alpha": ALPHA,
            "att_true": design.att_true,
            "interval_method": method,
            "assumption_basis": METHOD_BASIS[method],
            "theoretically_valid_in_dgp": _method_is_valid(method),
            "validity_reason": _method_validity_reason(method),
            "status": "failed",
            "error_message": message,
        }
        for method in INTERVAL_METHODS
    ]


def run_replication_task(
    task: tuple[int, int, int, tuple[float, ...], float, float],
) -> list[dict[str, Any]]:
    """Worker entry point evaluating all kappas on one common outer sample."""

    (
        replication,
        n,
        master_seed,
        kappas,
        variance_bandwidth_coefficient,
        variance_bandwidth_exponent,
    ) = task
    designs = get_designs(kappas)
    base_seed = replication_seed(master_seed, n, replication)
    reference = designs[kappas[0]]
    fold_hash = ""
    latent_hash = ""
    try:
        (
            x_index,
            group,
            residuals,
            folds,
            fold_hash,
            latent_hash,
        ) = generate_common_sample(reference, n, base_seed)
    except Exception as error:
        rows: list[dict[str, Any]] = []
        for kappa_m in kappas:
            rows.extend(
                failed_design_rows(
                    designs[kappa_m],
                    n,
                    replication,
                    master_seed,
                    base_seed,
                    fold_hash,
                    latent_hash,
                    variance_bandwidth_coefficient,
                    variance_bandwidth_exponent,
                    error,
                )
            )
        return rows

    rows = []
    for kappa_m in kappas:
        design = designs[kappa_m]
        try:
            rows.extend(
                evaluate_design(
                    design,
                    n,
                    replication,
                    master_seed,
                    base_seed,
                    x_index,
                    group,
                    residuals,
                    folds,
                    fold_hash,
                    latent_hash,
                    variance_bandwidth_coefficient,
                    variance_bandwidth_exponent,
                )
            )
        except Exception as error:
            rows.extend(
                failed_design_rows(
                    design,
                    n,
                    replication,
                    master_seed,
                    base_seed,
                    fold_hash,
                    latent_hash,
                    variance_bandwidth_coefficient,
                    variance_bandwidth_exponent,
                    error,
                )
            )
    return rows


def _directional_derivative_audit(
    bundle: NuisanceBundle,
    kappa_m: float,
) -> None:
    """Check the P1+P2+P3 derivative used in the SC influence function."""

    seed = int(2026080701 + round(1000.0 * kappa_m))
    rng = np.random.default_rng(seed)
    for x_index in (0, N_X // 2, N_X - 1):
        matrix_m = bundle.matrix_m[x_index]
        target_m1 = bundle.target_m1[x_index]
        gram_inverse = bundle.gram_inverse[x_index]
        beta = bundle.free_weights[x_index]
        matrix_a = rng.normal(size=matrix_m.shape)
        a1 = rng.normal(size=target_m1.shape)
        p1 = gram_inverse @ matrix_m.T @ a1
        p2 = gram_inverse @ matrix_a.T @ target_m1
        p3 = -gram_inverse @ (
            matrix_a.T @ matrix_m + matrix_m.T @ matrix_a
        ) @ beta
        analytic = p1 + p2 + p3
        epsilon = 1.0e-7
        plus_m = matrix_m + epsilon * matrix_a
        minus_m = matrix_m - epsilon * matrix_a
        plus_target = target_m1 + epsilon * a1
        minus_target = target_m1 - epsilon * a1
        plus = np.linalg.solve(
            plus_m.T @ plus_m,
            plus_m.T @ plus_target,
        )
        minus = np.linalg.solve(
            minus_m.T @ minus_m,
            minus_m.T @ minus_target,
        )
        numerical = (plus - minus) / (2.0 * epsilon)
        if not np.allclose(
            analytic,
            numerical,
            rtol=5.0e-4,
            atol=5.0e-5,
        ):
            raise AssertionError(
                f"weight derivative audit failed at kappa={kappa_m}, "
                f"x index={x_index}"
            )


def population_diagnostics(
    designs: dict[float, Design],
) -> list[dict[str, Any]]:
    """Return one complete population-audit row per kappa and x value."""

    rows: list[dict[str, Any]] = []
    for kappa_m, design in designs.items():
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
            realized_kappa = float(
                singular_values[0] / singular_values[-1]
            )
            treated_path = design.treated_untreated_means[x_index]
            reconstructed = donor_paths @ design.true_weights[x_index]
            donor_changes = donor_paths[-1] - donor_paths[-2]
            treated_change = treated_path[-1] - treated_path[-2]
            post_contrast = (
                donor_changes[: N_DONORS - 1] - donor_changes[-1]
            )
            rows.append(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "dgp_version": DGP_VERSION,
                    "dgp_name": "sc_only",
                    "pt_holds_by_construction": False,
                    "sc_holds_by_construction": True,
                    "donor_codes": "|".join(
                        str(code) for code in DONOR_CODES
                    ),
                    "donor_states": DONOR_STATES,
                    "baseline_donor": BASELINE_DONOR,
                    "condition_definition": CONDITION_DEFINITION,
                    "target_kappa_m": kappa_m,
                    "empirical_kappa_m_reference": EMPIRICAL_KAPPA_REFERENCE,
                    "x_index": x_index,
                    "x": float(x_value),
                    "realized_kappa_m": realized_kappa,
                    "derived_kappa_mtm": realized_kappa**2,
                    "sigma_max_m": float(singular_values[0]),
                    "sigma_middle_m": float(singular_values[1]),
                    "sigma_min_m": float(singular_values[2]),
                    "rank_m": int(np.linalg.matrix_rank(matrix_m)),
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
                    "weak_post_projection_abs": abs(
                        float(right_vectors[-1] @ post_contrast)
                    ),
                    "post_scale": POST_SCALE,
                    "att_true": design.att_true,
                }
            )
    return rows


def run_internal_audits(kappas: tuple[float, ...]) -> list[dict[str, Any]]:
    """Validate every population and the analytic weight derivative."""

    designs = get_designs(kappas)
    for kappa_m, design in designs.items():
        validate_population_design(design)
        oracle = build_oracle_nuisances(design)
        if not np.allclose(
            oracle.weights,
            design.true_weights,
            rtol=2.0e-9,
            atol=2.0e-9,
        ):
            raise AssertionError(f"oracle weights fail at kappa={kappa_m}")
        donor_gaps = (
            oracle.group_change[:, 1:]
            - oracle.pooled_change[:, None]
        )
        if float(np.max(np.abs(donor_gaps))) <= 1.0e-8:
            raise AssertionError(
                f"SC correction degenerates because PT holds at kappa={kappa_m}"
            )
        _directional_derivative_audit(oracle, kappa_m)
    return population_diagnostics(designs)


def _finite_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    """Extract finite numeric values from live or CSV-loaded rows."""

    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, float("nan")))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def _finite_indicators(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    """Extract Boolean indicators from live or CSV-loaded rows."""

    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (bool, np.bool_)):
            values.append(float(value))
            continue
        normalized = str(value).strip().lower()
        if normalized in ("true", "1", "1.0"):
            values.append(1.0)
        elif normalized in ("false", "0", "0.0"):
            values.append(0.0)
    return np.asarray(values, dtype=float)


def _row_matches_kappa(row: dict[str, Any], kappa_m: float) -> bool:
    """Return whether a live or restored row belongs to one kappa cell."""

    try:
        value = float(row.get("target_kappa_m", float("nan")))
    except (TypeError, ValueError):
        return False
    return math.isclose(value, kappa_m, rel_tol=0.0, abs_tol=1.0e-12)


def summarize_rows(
    rows: list[dict[str, Any]],
    kappas: tuple[float, ...],
    n: int,
    attempted_replications: int,
    elapsed_seconds: float,
    milestone_percent: int = 100,
) -> list[dict[str, Any]]:
    """Return kappa-by-method coverage and standard-error summaries."""

    designs = get_designs(kappas)
    summaries: list[dict[str, Any]] = []
    for kappa_m in kappas:
        design = designs[kappa_m]
        for method in INTERVAL_METHODS:
            cell = [
                row
                for row in rows
                if _row_matches_kappa(row, kappa_m)
                and row.get("interval_method") == method
                and row.get("status") == "valid"
            ]
            estimates = _finite_values(cell, "point_estimate")
            standard_errors = _finite_values(cell, "analytic_se")
            lengths = _finite_values(cell, "ci_length")
            standardized = _finite_values(cell, "standardized_error")
            covers = _finite_indicators(cell, "covers_att")
            empirical_sd = (
                float(np.std(estimates, ddof=1))
                if estimates.size > 1
                else float("nan")
            )
            mean_se = (
                float(np.mean(standard_errors))
                if standard_errors.size
                else float("nan")
            )
            coverage = (
                float(np.mean(covers)) if covers.size else float("nan")
            )
            coverage_mcse = (
                math.sqrt(coverage * (1.0 - coverage) / covers.size)
                if covers.size and np.isfinite(coverage)
                else float("nan")
            )
            summaries.append(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "dgp_version": DGP_VERSION,
                    "dgp_name": "sc_only",
                    "pt_holds_by_construction": False,
                    "sc_holds_by_construction": True,
                    "target_kappa_m": kappa_m,
                    "population_kappa_m_median": (
                        design.population_kappa_m_median
                    ),
                    "population_sigma_max_median": (
                        design.population_sigma_max_median
                    ),
                    "population_sigma_middle_median": (
                        design.population_sigma_middle_median
                    ),
                    "population_sigma_min_median": (
                        design.population_sigma_min_median
                    ),
                    "interval_method": method,
                    "assumption_basis": METHOD_BASIS[method],
                    "n": n,
                    "theoretically_valid_in_dgp": _method_is_valid(method),
                    "validity_reason": _method_validity_reason(method),
                    "milestone_percent": milestone_percent,
                    "attempted_outer_replications": attempted_replications,
                    "valid_intervals": int(standard_errors.size),
                    "failed_intervals": int(
                        attempted_replications - standard_errors.size
                    ),
                    "att_true": design.att_true,
                    "mean_point_estimate": (
                        float(np.mean(estimates))
                        if estimates.size
                        else float("nan")
                    ),
                    "bias": (
                        float(np.mean(estimates) - design.att_true)
                        if estimates.size
                        else float("nan")
                    ),
                    "empirical_sd": empirical_sd,
                    "rmse": (
                        float(
                            np.sqrt(
                                np.mean((estimates - design.att_true) ** 2)
                            )
                        )
                        if estimates.size
                        else float("nan")
                    ),
                    "mean_analytic_se": mean_se,
                    "median_analytic_se": (
                        float(np.median(standard_errors))
                        if standard_errors.size
                        else float("nan")
                    ),
                    "mean_se_over_empirical_sd": (
                        mean_se / empirical_sd
                        if np.isfinite(mean_se)
                        and np.isfinite(empirical_sd)
                        and empirical_sd > 0.0
                        else float("nan")
                    ),
                    "ci_coverage": coverage,
                    "ci_coverage_mcse": coverage_mcse,
                    "mean_ci_length": (
                        float(np.mean(lengths))
                        if lengths.size
                        else float("nan")
                    ),
                    "median_ci_length": (
                        float(np.median(lengths))
                        if lengths.size
                        else float("nan")
                    ),
                    "standardized_error_mean": (
                        float(np.mean(standardized))
                        if standardized.size
                        else float("nan")
                    ),
                    "standardized_error_sd": (
                        float(np.std(standardized, ddof=1))
                        if standardized.size > 1
                        else float("nan")
                    ),
                    "standardized_error_q025": (
                        float(np.quantile(standardized, 0.025))
                        if standardized.size
                        else float("nan")
                    ),
                    "standardized_error_q975": (
                        float(np.quantile(standardized, 0.975))
                        if standardized.size
                        else float("nan")
                    ),
                    "elapsed_wall_seconds": elapsed_seconds,
                }
            )
    return summaries


def atomic_write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...] | None = None,
) -> None:
    """Atomically replace a CSV with a deterministic schema."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames) if fieldnames is not None else []
    seen: set[str] = set(fields)
    if fieldnames is None:
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
    else:
        unexpected = sorted(
            {key for row in rows for key in row if key not in seen}
        )
        if unexpected:
            raise KeyError(f"CSV schema is missing fields: {unexpected}")
    if not fields:
        fields = ["status"]
        rows = [{"status": "empty"}]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON metadata file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a complete atomic CSV snapshot."""

    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, Any]:
    """Read one metadata JSON object."""

    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"metadata is not a JSON object: {path}")
    return payload


def validate_resume_metadata(
    metadata: dict[str, Any],
    n: int,
    args: argparse.Namespace,
    kappas: tuple[float, ...],
) -> None:
    """Reject checkpoints created by a different experiment grid."""

    expected = {
        "experiment_version": EXPERIMENT_VERSION,
        "dgp_version": DGP_VERSION,
        "kappa_grid": list(kappas),
        "n": n,
        "replications": args.replications,
        "master_seed": args.master_seed,
        "reference_augmentation_gamma": REFERENCE_AUGMENTATION_GAMMA,
        "post_scale": POST_SCALE,
        "alpha": ALPHA,
        "folds": N_FOLDS,
        "bandwidth_coefficient": BANDWIDTH_COEFFICIENT,
        "bandwidth_exponent": BANDWIDTH_EXPONENT,
        "variance_bandwidth_coefficient": (
            args.variance_bandwidth_coefficient
        ),
        "variance_bandwidth_exponent": args.variance_bandwidth_exponent,
        "interval_methods": list(INTERVAL_METHODS),
        "weight_estimator": WEIGHT_ESTIMATOR,
        "simplex_projection": False,
        "sc_weight_ridge": 0.0,
        "trimming": "none_theorem_matched",
        "checkpoint_percent": args.milestone_percent,
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"resume metadata mismatch: {mismatches}")


def validate_checkpoint_rows(
    rows: list[dict[str, Any]],
    n: int,
    replications: int,
    kappas: tuple[float, ...],
) -> set[int]:
    """Validate complete replication batches in an outer checkpoint."""

    expected_pairs = {
        (kappa_m, method)
        for kappa_m in kappas
        for method in INTERVAL_METHODS
    }
    by_replication: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        try:
            replication = int(row.get("outer_replication", ""))
            row_n = int(row.get("n", ""))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "checkpoint has an invalid replication or sample size"
            ) from error
        if row_n != n or not 0 <= replication < replications:
            raise ValueError("checkpoint row falls outside the requested grid")
        by_replication.setdefault(replication, []).append(row)
    for replication, batch in by_replication.items():
        pairs = [
            (float(row.get("target_kappa_m", "nan")), str(row.get("interval_method")))
            for row in batch
        ]
        if len(pairs) != len(expected_pairs) or set(pairs) != expected_pairs:
            raise ValueError(
                f"checkpoint replication {replication} is incomplete or duplicated"
            )
    return set(by_replication)


def checkpoint_elapsed_seconds(path: Path) -> float:
    """Recover cumulative wall time from a checkpoint summary."""

    if not path.exists():
        return 0.0
    values = _finite_values(read_csv_rows(path), "elapsed_wall_seconds")
    return float(np.max(values)) if values.size else 0.0


def available_cpu_count() -> int:
    """Return CPUs visible inside the current HPC allocation."""

    if hasattr(os, "sched_getaffinity"):
        visible = max(1, len(os.sched_getaffinity(0)))
    else:
        visible = max(1, os.cpu_count() or 1)
    slurm_value = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if slurm_value.isdigit() and int(slurm_value) > 0:
        visible = min(visible, int(slurm_value))
    return visible


def _result_prefix(n: int, replications: int, tag: str) -> str:
    """Return the collision-resistant output prefix for one sample size."""

    return f"q4_condition_analytic_ci_{tag}_n{n}_R{replications}"


def run_sample_size(
    n: int,
    args: argparse.Namespace,
    kappas: tuple[float, ...],
    population_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one sample-size grid with atomic milestone checkpoints."""

    output_dir = Path(args.output_dir).expanduser().resolve()
    prefix = _result_prefix(n, args.replications, args.tag)
    outer_path = output_dir / f"{prefix}_outer.csv"
    summary_path = output_dir / f"{prefix}_coverage.csv"
    checkpoint_outer_path = output_dir / f"{prefix}_outer_checkpoint.csv"
    checkpoint_summary_path = output_dir / f"{prefix}_coverage_checkpoint.csv"
    population_path = output_dir / f"{prefix}_population.csv"
    metadata_path = output_dir / f"{prefix}_metadata.json"
    all_paths = (
        outer_path,
        summary_path,
        checkpoint_outer_path,
        checkpoint_summary_path,
        population_path,
        metadata_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_files = (metadata_path.exists(), checkpoint_outer_path.exists())
    if args.resume and resume_files[0] != resume_files[1]:
        raise FileNotFoundError(
            "resume requires both metadata and the outer checkpoint"
        )
    resume_available = bool(args.resume and all(resume_files))
    if resume_available:
        metadata = read_json(metadata_path)
        validate_resume_metadata(metadata, n, args, kappas)
        if not population_path.exists():
            atomic_write_csv(
                population_path,
                [{**row, "n": n} for row in population_rows],
            )
        rows: list[dict[str, Any]] = read_csv_rows(checkpoint_outer_path)
        completed_replications = validate_checkpoint_rows(
            rows,
            n,
            args.replications,
            kappas,
        )
        elapsed_before = checkpoint_elapsed_seconds(checkpoint_summary_path)
        metadata["resume_count"] = int(metadata.get("resume_count", 0)) + 1
        metadata["last_nprocs"] = args.nprocs
        metadata["status"] = "running"
        atomic_write_json(metadata_path, metadata)
        print(
            f"n={n}: resuming after {len(completed_replications)}/"
            f"{args.replications} replications",
            flush=True,
        )
    else:
        if not args.overwrite and any(path.exists() for path in all_paths):
            existing = next(path for path in all_paths if path.exists())
            raise FileExistsError(
                f"refusing to overwrite {existing}; pass --overwrite or --resume"
            )
        atomic_write_csv(
            population_path,
            [{**row, "n": n} for row in population_rows],
        )
        metadata = {
            "experiment_version": EXPERIMENT_VERSION,
            "dgp_version": DGP_VERSION,
            "experiment_purpose": EXPERIMENT_PURPOSE,
            "kappa_grid": list(kappas),
            "n": n,
            "replications": args.replications,
            "master_seed": args.master_seed,
            "reference_augmentation_gamma": REFERENCE_AUGMENTATION_GAMMA,
            "post_scale": POST_SCALE,
            "tag": args.tag,
            "nprocs": args.nprocs,
            "last_nprocs": args.nprocs,
            "multiprocessing_context": "spawn",
            "blas_threads_per_process": int(_BLAS_THREAD_COUNT),
            "folds": N_FOLDS,
            "alpha": ALPHA,
            "bandwidth_coefficient": BANDWIDTH_COEFFICIENT,
            "bandwidth_exponent": BANDWIDTH_EXPONENT,
            "effective_bandwidth": (
                BANDWIDTH_COEFFICIENT * n ** BANDWIDTH_EXPONENT
            ),
            "variance_bandwidth_coefficient": (
                args.variance_bandwidth_coefficient
            ),
            "variance_bandwidth_exponent": (
                args.variance_bandwidth_exponent
            ),
            "effective_variance_bandwidth": (
                args.variance_bandwidth_coefficient
                * n**args.variance_bandwidth_exponent
            ),
            "interval_methods": list(INTERVAL_METHODS),
            "weight_estimator": WEIGHT_ESTIMATOR,
            "probability_recovery": "from_direct_local_ratio_estimates",
            "kernel": "epanechnikov_compact_support_local_linear",
            "common_random_numbers": (
                "x_group_residuals_and_folds_shared_across_kappa_grid"
            ),
            "sc_variance_construction": (
                "sample_variance_of_total_feasible_pt_plus_psi1_plus_psi2_"
                "with_se_specific_nuisance_bandwidth"
            ),
            "simplex_projection": False,
            "sc_weight_ridge": 0.0,
            "local_linear_ridge": RIDGE_LLR,
            "trimming": "none_theorem_matched",
            "checkpoint_percent": args.milestone_percent,
            "checkpoint_restartable": True,
            "resume_count": 0,
            "status": "running",
        }
        atomic_write_json(metadata_path, metadata)
        atomic_write_csv(checkpoint_outer_path, [], OUTER_FIELDS)
        atomic_write_csv(checkpoint_summary_path, [])
        rows = []
        completed_replications = set()
        elapsed_before = 0.0

    tasks = [
        (
            replication,
            n,
            args.master_seed,
            kappas,
            args.variance_bandwidth_coefficient,
            args.variance_bandwidth_exponent,
        )
        for replication in range(args.replications)
        if replication not in completed_replications
    ]
    completed = len(completed_replications)
    started = time.perf_counter()
    milestone_counts = sorted(
        {
            min(
                args.replications,
                max(
                    1,
                    math.ceil(args.replications * percent / 100.0),
                ),
            )
            for percent in range(
                args.milestone_percent,
                101,
                args.milestone_percent,
            )
        }
        | {args.replications}
    )
    milestone_index = 0
    while (
        milestone_index < len(milestone_counts)
        and milestone_counts[milestone_index] <= completed
    ):
        milestone_index += 1

    if not tasks:
        iterator: Iterable[list[dict[str, Any]]] = iter(())
        executor = None
    elif args.nprocs == 1:
        iterator = (run_replication_task(task) for task in tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=args.nprocs,
            mp_context=multiprocessing.get_context("spawn"),
        )
        futures = [executor.submit(run_replication_task, task) for task in tasks]
        iterator = (future.result() for future in as_completed(futures))
    try:
        for batch in iterator:
            rows.extend(batch)
            completed += 1
            while (
                milestone_index < len(milestone_counts)
                and completed >= milestone_counts[milestone_index]
            ):
                percent = int(round(100.0 * completed / args.replications))
                elapsed = elapsed_before + float(time.perf_counter() - started)
                ordered = sorted(
                    rows,
                    key=lambda row: (
                        int(row.get("outer_replication", -1)),
                        float(row.get("target_kappa_m", float("nan"))),
                        str(row.get("interval_method", "")),
                    ),
                )
                provisional = summarize_rows(
                    ordered,
                    kappas,
                    n,
                    completed,
                    elapsed,
                    milestone_percent=percent,
                )
                atomic_write_csv(
                    checkpoint_outer_path,
                    ordered,
                    OUTER_FIELDS,
                )
                atomic_write_csv(checkpoint_summary_path, provisional)
                print(
                    f"n={n}: {percent}% milestone "
                    f"({completed}/{args.replications}) written to CSV",
                    flush=True,
                )
                milestone_index += 1
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    elapsed = elapsed_before + float(time.perf_counter() - started)
    rows = sorted(
        rows,
        key=lambda row: (
            int(row.get("outer_replication", -1)),
            float(row.get("target_kappa_m", float("nan"))),
            str(row.get("interval_method", "")),
        ),
    )
    summaries = summarize_rows(
        rows,
        kappas,
        n,
        args.replications,
        elapsed,
        milestone_percent=100,
    )
    atomic_write_csv(outer_path, rows, OUTER_FIELDS)
    atomic_write_csv(summary_path, summaries)
    atomic_write_csv(checkpoint_outer_path, rows, OUTER_FIELDS)
    atomic_write_csv(checkpoint_summary_path, summaries)
    metadata["status"] = "complete"
    metadata["completed_replications"] = args.replications
    metadata["elapsed_wall_seconds"] = elapsed
    atomic_write_json(metadata_path, metadata)
    print(
        f"n={n}: complete in {elapsed:.1f}s; outputs written under {output_dir}",
        flush=True,
    )
    return rows, summaries


def parse_arguments() -> argparse.Namespace:
    """Parse and validate the standalone HPC command line."""

    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Run feasible PT and SC analytic-CI coverage over the SC-only "
            "Quantile-top-4 condition-number path."
        )
    )
    parser.add_argument(
        "--n",
        nargs="+",
        type=int,
        default=list(DEFAULT_SAMPLE_SIZES),
        help="sample sizes (default: 2000 4000)",
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=DEFAULT_REPLICATIONS,
        help="Monte Carlo replications per sample size (default: 500)",
    )
    parser.add_argument(
        "--kappa",
        dest="kappas",
        nargs="+",
        type=float,
        default=list(DEFAULT_KAPPA_GRID),
        help="direct condition numbers kappa(M)",
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=0,
        help="worker processes; 0 uses all CPUs visible to the allocation",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=DEFAULT_MASTER_SEED,
    )
    parser.add_argument(
        "--variance-bandwidth-coefficient",
        type=float,
        default=DEFAULT_VARIANCE_BANDWIDTH_COEFFICIENT,
        help=(
            "coefficient for nuisance fits used only in analytic standard "
            "errors; the point estimator remains fixed at coefficient 2.5"
        ),
    )
    parser.add_argument(
        "--variance-bandwidth-exponent",
        type=float,
        default=DEFAULT_VARIANCE_BANDWIDTH_EXPONENT,
        help=(
            "exponent for nuisance fits used only in analytic standard "
            "errors; use -0.2 for n^(-1/5)"
        ),
    )
    parser.add_argument(
        "--milestone-percent",
        type=int,
        default=DEFAULT_MILESTONE_PERCENT,
        help="checkpoint interval in percentage points",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "generated",
    )
    parser.add_argument("--tag", default="condition_v1")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace outputs with the same tag and dimensions",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from the latest complete milestone checkpoint",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="run population and influence-function audits without simulation",
    )
    args = parser.parse_args()
    if any(value <= 1 for value in args.n):
        parser.error("all sample sizes must exceed one")
    if len(set(args.n)) != len(args.n):
        parser.error("sample sizes must be unique")
    if args.replications <= 0:
        parser.error("--replications must be positive")
    if not args.kappas:
        parser.error("--kappa cannot be empty")
    if any(not np.isfinite(value) or value < 1.0 for value in args.kappas):
        parser.error("every --kappa value must be finite and at least one")
    if len(set(float(value) for value in args.kappas)) != len(args.kappas):
        parser.error("--kappa values must be unique")
    if args.master_seed < 0:
        parser.error("--master-seed must be nonnegative")
    if (
        not np.isfinite(args.variance_bandwidth_coefficient)
        or args.variance_bandwidth_coefficient <= 0.0
    ):
        parser.error("--variance-bandwidth-coefficient must be positive")
    if not np.isfinite(args.variance_bandwidth_exponent):
        parser.error("--variance-bandwidth-exponent must be finite")
    if args.nprocs < 0:
        parser.error("--nprocs must be nonnegative")
    if args.overwrite and args.resume:
        parser.error("--overwrite and --resume are mutually exclusive")
    if not 1 <= args.milestone_percent <= 100:
        parser.error("--milestone-percent must be between 1 and 100")
    if 100 % args.milestone_percent != 0:
        parser.error("--milestone-percent must divide 100")
    args.n = tuple(int(value) for value in args.n)
    args.kappas = tuple(sorted(float(value) for value in args.kappas))
    if args.nprocs == 0:
        args.nprocs = available_cpu_count()
    return args


def main() -> None:
    """Audit the design, run requested sample sizes, and combine summaries."""

    args = parse_arguments()
    population_rows = run_internal_audits(args.kappas)
    print(
        "Population audit passed for kappa(M) = "
        + ", ".join(f"{value:g}" for value in args.kappas),
        flush=True,
    )
    print(
        "Inference: feasible PT diagnostic and feasible SC valid interval; "
        "unrestricted affine weights, no simplex, no SC ridge, no trimming.",
        flush=True,
    )
    print(
        "Bandwidths: point estimator h="
        f"{BANDWIDTH_COEFFICIENT:g}*n^({BANDWIDTH_EXPONENT:.12g}); "
        "standard-error nuisances h="
        f"{args.variance_bandwidth_coefficient:g}*"
        f"n^({args.variance_bandwidth_exponent:.12g}).",
        flush=True,
    )
    if args.audit_only:
        return

    all_summaries: list[dict[str, Any]] = []
    for n in args.n:
        _, summaries = run_sample_size(
            n,
            args,
            args.kappas,
            population_rows,
        )
        all_summaries.extend(summaries)
    if len(args.n) > 1:
        joined_n = "_".join(str(value) for value in args.n)
        combined_path = (
            Path(args.output_dir).expanduser().resolve()
            / f"q4_condition_analytic_ci_{args.tag}_n{joined_n}_"
            f"R{args.replications}_coverage_all_n.csv"
        )
        atomic_write_csv(combined_path, all_summaries)
        print(f"Combined coverage summary written to {combined_path}", flush=True)


if __name__ == "__main__":
    main()
