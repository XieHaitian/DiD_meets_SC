#!/usr/bin/env python3
"""HPC Monte Carlo comparison of DiDSC and panel DRDiD.

The data-generating process is the calibrated quantile-top-4 design with
Maryland, New Hampshire, Utah, and Virginia as donors and
``kappa(M)=13.070487184660164``.  It starts from the exact PT+SC parent.
For local parameter ``delta`` and sample size ``n``, donor post-period means
are perturbed along the empirical donor-change direction by

    delta * sigma_DR / sqrt(n),

where ``sigma_DR`` is the oracle root-n standard deviation of the panel
DRDiD estimator at the parent.  The direction is normalized so that the
treated-X-weighted binary PT gap equals the displayed local perturbation.
The treated untreated path is recomputed as the exact synthetic combination
of the donor paths.  Consequently, SC holds exactly for every member of the
family, the pre-period matrix and its condition number never change, PT holds
at delta=0, and PT fails locally at rate n^{-1/2} otherwise.

The two estimators share the outer sample, fixed two-fold split, Epanechnikov
kernel, local-linear algorithm, pooled-control outcome-trend construction, and
treated-to-donor ratio construction.  At delta=0, DiDSC exactly reproduces the
bandwidth experiment's PT+SC, h=2.5*n^(-2/7) implementation: the same master
seed, strict realized-boundary trimming, ratio fallback, fold-specific score
normalization, and symmetric full nested Exp(1) multiplier bootstrap.  Because
the DR score is orthogonal, DRDiD instead uses the MSE-rate bandwidth
h=n^(-1/5) for both nuisances and retains the positive ratio safeguard.  It
recovers the binary odds p_1/(1-p_1) from its four donor ratios and implements
the panel DR moment in Sant'Anna and Zhao (2020), equations (2.6), (2.7), and
(3.1), with the analytic influence-function interval from equation (2.11).

Defaults are n=2,000 and 4,000; 500 outer replications; 500 bootstrap
draws; and delta in {-2,-1.5,-1,-.5,0,.5,1,1.5,2}.  Results and atomic
checkpoints are written at every 10-percent milestone.  The script is fully
self-contained and requires only NumPy.
"""

from __future__ import annotations

import os


for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np


EXPERIMENT_VERSION = (
    "q4_didsc_drdid_local_pt_exact_sc_v3_bandwidth_aligned_didsc"
)
DGP_VERSION = "quantile_top4_exact_sc_local_binary_pt_root_n_v1"
EXPERIMENT_PURPOSE = "compare_didsc_and_panel_drdid_under_local_pt_failure"
PAPER_REFERENCE = "Sant'Anna and Zhao (2020), panel DRDID equations 2.6, 2.7, 2.11, and 3.1"

DONOR_CODES = (24, 33, 49, 51)
DONOR_STATES = "Maryland|New Hampshire|Utah|Virginia"
BASELINE_DONOR = "Virginia"
KAPPA_M = 13.070487184660164
CONDITION_DEFINITION = "kappa(M)=sigma_max(M)/sigma_min(M)"
REFERENCE_GAMMA = math.sqrt(29.07742027873098)
POST_SCALE = 3.0
EMPIRICAL_POST_CONTRAST = np.asarray(
    [0.211789512326912, 0.9801124738736031, 0.3509230190287762],
    dtype=float,
)
ATT_FINGERPRINT = 1.011932168518905
SIGMA_FINGERPRINT = np.asarray(
    [14.074084253556677, 14.073902974813999, 1.0767834476800806],
    dtype=float,
)

N_GROUPS = 5
N_DONORS = 4
N_PRE = 5
N_PERIODS = 6
N_X = 101
N_FOLDS = 2

DIDSC_BANDWIDTH_COEFFICIENT = 2.5
DIDSC_BANDWIDTH_EXPONENT = -2.0 / 7.0
DRDID_BANDWIDTH_COEFFICIENT = 1.0
DRDID_BANDWIDTH_EXPONENT = -1.0 / 5.0
KERNEL_NAME = "epanechnikov_compact_support_local_linear"
TRIM_QUANTILES = (0.0, 1.0)
RIDGE_LLR = 1.0e-6
RIDGE_SC = 1.0e-6
ALPHA = 0.05
Z_975 = NormalDist().inv_cdf(1.0 - ALPHA / 2.0)

DEFAULT_SAMPLE_SIZES = (2000, 4000)
DEFAULT_OUTER_REPS = 500
DEFAULT_BOOTSTRAPS = 500
DEFAULT_DELTA_GRID = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
DEFAULT_MASTER_SEED = 2026073001
DEFAULT_BOOTSTRAP_BATCH_SIZE = 50
DEFAULT_TAG = "production_v3"
DEFAULT_TRIM_BOUNDARIES = True

METHODS = ("didsc_multiplier", "drdid_analytic")
WEIGHT_ESTIMATOR = (
    "ridge-stabilized unrestricted affine least squares followed by "
    "Euclidean simplex projection"
)
NUISANCE_DESCRIPTION = (
    "same two-fold cross-fitted Epanechnikov local-linear algorithm and "
    "direct treated-to-donor ratio construction; DiDSC reproduces the "
    "bandwidth-experiment ratio rule at h=2.5*n^(-2/7), while DRDiD keeps "
    "its positive local-constant fallback at h=n^(-1/5)"
)
DIDSC_SCORE_NORMALIZATION = (
    "bandwidth-experiment fold average: each fold score numerator divided "
    "by the full-sample multiplier-weighted treated share and the unweighted "
    "number of retained test observations"
)

ABS_TOL = 1.0e-9
REL_TOL = 1.0e-9


@dataclass(frozen=True)
class Population:
    """Frozen population arrays and audited local direction."""

    x_support: np.ndarray
    group_probabilities: np.ndarray
    donor_means_parent: np.ndarray
    treated_means_parent: np.ndarray
    treatment_effect: np.ndarray
    true_weights: np.ndarray
    residual_variances: np.ndarray
    donor_post_direction: np.ndarray
    treated_post_direction: np.ndarray
    pooled_control_post_direction: np.ndarray
    binary_pt_direction: np.ndarray
    att_true: float
    oracle_dr_rootn_sd: float
    direction_normalizer: float
    bandwidth_affine_change_gap_parent: np.ndarray
    singular_values: np.ndarray
    condition_numbers: np.ndarray


@dataclass(frozen=True)
class Sample:
    """One common outer sample used by the full local grid."""

    x_index: np.ndarray
    group: np.ndarray
    folds: np.ndarray
    outcomes_parent: np.ndarray
    post_direction: np.ndarray
    fold_hash: str
    latent_hash: str


@dataclass
class BatchEstimate:
    """Base and local-direction coefficients for both estimators."""

    didsc_base: np.ndarray
    didsc_direction: np.ndarray
    drdid_base: np.ndarray
    drdid_direction: np.ndarray
    dr_if_base: np.ndarray | None
    dr_if_direction: np.ndarray | None
    diagnostics: dict[str, float | int]


@dataclass
class FoldAggregates:
    """Weighted group-by-X sufficient statistics for one fold."""

    counts: np.ndarray
    outcome_sums: np.ndarray
    direction_sums: np.ndarray


@dataclass
class ReplicationOutput:
    """Long-format rows returned by one outer Monte Carlo task."""

    outer_rows: list[dict[str, Any]]
    bootstrap_rows: list[dict[str, Any]]


def _null_space_rows(vector: np.ndarray) -> np.ndarray:
    """Return orthonormal row vectors spanning the null space of vector."""

    _, _, right_vectors = np.linalg.svd(
        np.asarray(vector, dtype=float).reshape(1, -1),
        full_matrices=True,
    )
    return right_vectors[1:, :]


def _rank_completion_paths(
    factor_paths: np.ndarray,
    fixed_effect_contrasts: np.ndarray,
) -> np.ndarray:
    """Recreate the calibrated rank-completion component."""

    empirical_span = np.column_stack(
        [np.ones(N_PRE), factor_paths[:N_PRE]]
    )
    _, _, right_vectors = np.linalg.svd(
        empirical_span.T,
        full_matrices=True,
    )
    h_matrix = right_vectors.T[:, 3:5]
    c_matrix = _null_space_rows(fixed_effect_contrasts)
    magnitude = float(
        REFERENCE_GAMMA
        * math.sqrt(N_PRE)
        * np.linalg.norm(fixed_effect_contrasts)
    )
    donor_loadings = np.zeros((2, N_DONORS), dtype=float)
    donor_loadings[:, : N_DONORS - 1] = magnitude * c_matrix
    factor_with_post = np.vstack([h_matrix, h_matrix[-1]])
    return factor_with_post @ donor_loadings


@lru_cache(maxsize=1)
def build_population() -> Population:
    """Build the exact PT+SC parent and normalized local PT direction."""

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

    donor_sc_only = reference_donor_means.copy()
    singular_values = np.empty((N_X, N_DONORS - 1), dtype=float)
    condition_numbers = np.empty(N_X, dtype=float)
    for x_index in range(N_X):
        baseline = reference_donor_means[x_index, :, -1]
        reference_m = (
            reference_donor_means[
                x_index,
                :N_PRE,
                : N_DONORS - 1,
            ]
            - baseline[:N_PRE, None]
        )
        left_vectors, singular_reference, right_vectors = np.linalg.svd(
            reference_m,
            full_matrices=False,
        )
        target_minimum = singular_reference[0] / KAPPA_M
        adjusted = np.maximum(singular_reference, target_minimum)
        adjusted[-1] = target_minimum
        target_m = left_vectors @ np.diag(adjusted) @ right_vectors
        donor_sc_only[
            x_index,
            :N_PRE,
            : N_DONORS - 1,
        ] = baseline[:N_PRE, None] + target_m
        donor_sc_only[x_index, -1, : N_DONORS - 1] = (
            baseline[-1]
            + target_m[-1]
            + POST_SCALE * EMPIRICAL_POST_CONTRAST
        )
        values = np.linalg.svd(target_m, compute_uv=False)
        singular_values[x_index] = values
        condition_numbers[x_index] = values[0] / values[-1]

    donor_indices = np.arange(1, N_DONORS + 1, dtype=float)
    weight_logits = (
        x_support[:, None]
        * (donor_indices[None, :] - N_DONORS)
    )
    weight_logits -= weight_logits.max(axis=1, keepdims=True)
    softmax_weights = np.exp(weight_logits)
    softmax_weights /= softmax_weights.sum(axis=1, keepdims=True)
    true_weights = 0.2 / N_DONORS + 0.8 * softmax_weights

    reference_change = (
        donor_sc_only[:, -1, -1] - donor_sc_only[:, -2, -1]
    )
    donor_parent = donor_sc_only.copy()
    donor_parent[:, -1] = (
        donor_parent[:, -2] + reference_change[:, None]
    )
    treated_parent = np.einsum(
        "xd,xtd->xt",
        true_weights,
        donor_parent,
    )
    bandwidth_affine_change_gap_parent = np.empty(N_X, dtype=float)
    for x_index in range(N_X):
        donor_path = donor_parent[x_index]
        donor_pre = donor_path[:N_PRE]
        matrix_m = (
            donor_pre[:, : N_DONORS - 1]
            - donor_pre[:, [-1]]
        )
        target = treated_parent[x_index, :N_PRE] - donor_pre[:, -1]
        gram = matrix_m.T @ matrix_m
        free_weights = np.linalg.solve(gram, matrix_m.T @ target)
        affine_weights = np.r_[
            free_weights,
            1.0 - free_weights.sum(),
        ]
        donor_change = donor_path[-1] - donor_path[-2]
        treated_change = (
            treated_parent[x_index, -1]
            - treated_parent[x_index, -2]
        )
        bandwidth_affine_change_gap_parent[x_index] = (
            treated_change - donor_change @ affine_weights
        )
    treatment_effect = 1.0 + 0.7 * np.sin(2.0 * np.pi * x_support)
    treated_probability = group_probabilities[:, 0]
    att_true = float(
        np.sum(treated_probability * treatment_effect)
        / np.sum(treated_probability)
    )

    control_probabilities = (
        group_probabilities[:, 1:]
        / (1.0 - treated_probability[:, None])
    )
    raw_direction = np.r_[EMPIRICAL_POST_CONTRAST, 0.0]
    raw_binary_direction = (
        (true_weights - control_probabilities) @ raw_direction
    )
    direction_normalizer = float(
        np.sum(treated_probability * raw_binary_direction)
        / np.sum(treated_probability)
    )
    if abs(direction_normalizer) <= 1.0e-10:
        raise AssertionError("local PT direction has zero treated mean")
    donor_post_direction = raw_direction / direction_normalizer
    treated_post_direction = true_weights @ donor_post_direction
    pooled_control_post_direction = (
        control_probabilities @ donor_post_direction
    )
    binary_pt_direction = (
        treated_post_direction - pooled_control_post_direction
    )

    pi_treated = float(np.mean(treated_probability))
    change_noise_variance = float(
        residual_variances[-2] + residual_variances[-1]
    )
    oracle_variance = float(
        np.mean(
            treated_probability
            * (
                (treatment_effect - att_true) ** 2
                + change_noise_variance
            )
            / pi_treated**2
            + treated_probability**2
            * change_noise_variance
            / ((1.0 - treated_probability) * pi_treated**2)
        )
    )
    oracle_dr_rootn_sd = math.sqrt(oracle_variance)

    population = Population(
        x_support=x_support,
        group_probabilities=group_probabilities,
        donor_means_parent=donor_parent,
        treated_means_parent=treated_parent,
        treatment_effect=treatment_effect,
        true_weights=true_weights,
        residual_variances=residual_variances,
        donor_post_direction=donor_post_direction,
        treated_post_direction=treated_post_direction,
        pooled_control_post_direction=pooled_control_post_direction,
        binary_pt_direction=binary_pt_direction,
        att_true=att_true,
        oracle_dr_rootn_sd=oracle_dr_rootn_sd,
        direction_normalizer=direction_normalizer,
        bandwidth_affine_change_gap_parent=(
            bandwidth_affine_change_gap_parent
        ),
        singular_values=singular_values,
        condition_numbers=condition_numbers,
    )
    validate_population_parent(population)
    return population


def validate_population_parent(population: Population) -> None:
    """Audit all calibrated fingerprints and the normalized direction."""

    if not math.isclose(
        population.att_true,
        ATT_FINGERPRINT,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
    ):
        raise AssertionError("ATT fingerprint changed")
    if not np.allclose(
        np.median(population.singular_values, axis=0),
        SIGMA_FINGERPRINT,
        rtol=REL_TOL,
        atol=ABS_TOL,
    ):
        raise AssertionError("singular-value fingerprint changed")
    if not np.allclose(
        population.condition_numbers,
        KAPPA_M,
        rtol=REL_TOL,
        atol=ABS_TOL,
    ):
        raise AssertionError("population condition number changed")
    reconstructed = np.einsum(
        "xd,xtd->xt",
        population.true_weights,
        population.donor_means_parent,
    )
    if not np.allclose(
        reconstructed,
        population.treated_means_parent,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("PT+SC parent does not satisfy exact SC")
    donor_changes = (
        population.donor_means_parent[:, -1]
        - population.donor_means_parent[:, -2]
    )
    if float(np.max(np.ptp(donor_changes, axis=1))) > 1.0e-12:
        raise AssertionError("PT+SC parent does not satisfy binary PT")
    weighted_direction = float(
        np.sum(
            population.group_probabilities[:, 0]
            * population.binary_pt_direction
        )
        / np.sum(population.group_probabilities[:, 0])
    )
    if not math.isclose(
        weighted_direction,
        1.0,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
    ):
        raise AssertionError("local binary PT direction is not normalized")
    if not math.isclose(
        population.oracle_dr_rootn_sd,
        1.6684768427690311,
        rel_tol=2.0e-12,
        abs_tol=2.0e-12,
    ):
        raise AssertionError("oracle DRDID root-n scale changed")
    if (
        float(np.max(np.abs(population.bandwidth_affine_change_gap_parent)))
        > 1.0e-12
    ):
        raise AssertionError("bandwidth affine population gap is not numerical zero")


def local_amplitude(population: Population, n: int, delta: float) -> float:
    """Return the signed root-n local perturbation in outcome units."""

    return float(delta * population.oracle_dr_rootn_sd / math.sqrt(n))


def common_evaluation_mask(trim_boundaries: bool) -> np.ndarray:
    """Return the population support mask used for reported targets."""

    mask = np.ones(N_X, dtype=bool)
    if trim_boundaries:
        mask[0] = False
        mask[-1] = False
    return mask


def sample_evaluation_geometry(
    sample: Sample,
    trim_boundaries: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Reproduce the bandwidth experiment's realized trimming geometry.

    Its 0% and 100% quantiles are the observed sample minimum and maximum.
    Evaluation is strict between those bounds and restricted to covariate
    cells observed somewhere in the outer sample.  Retained test-fold counts
    are unweighted, including for multiplier draws.
    """

    observed_x = np.linspace(0.0, 1.0, N_X)[sample.x_index]
    observed_support = np.zeros(N_X, dtype=bool)
    observed_support[sample.x_index] = True
    if trim_boundaries:
        trim_low, trim_high = np.quantile(observed_x, TRIM_QUANTILES)
        evaluation_mask = (
            observed_support
            & (np.linspace(0.0, 1.0, N_X) > trim_low)
            & (np.linspace(0.0, 1.0, N_X) < trim_high)
        )
    else:
        trim_low = float(np.min(observed_x))
        trim_high = float(np.max(observed_x))
        evaluation_mask = observed_support
    if not np.any(evaluation_mask):
        raise FloatingPointError("no covariate points survive trimming")
    retained_observations = evaluation_mask[sample.x_index]
    retained_fold_counts = np.asarray(
        [
            np.sum(retained_observations & (sample.folds == fold))
            for fold in range(N_FOLDS)
        ],
        dtype=np.int64,
    )
    if np.any(retained_fold_counts <= 0):
        raise FloatingPointError("trimmed evaluation fold is empty")
    return (
        evaluation_mask,
        retained_observations,
        retained_fold_counts,
        float(trim_low),
        float(trim_high),
    )


def method_population_targets(
    population: Population,
    n: int,
    delta: float,
    trim_boundaries: bool,
) -> tuple[float, float, float]:
    """Return the DiDSC score target, DR target, and local DR gap.

    With trimming, the bandwidth experiment normalizes the DiDSC score by
    the full-sample treated share and the retained-observation fraction.  The
    DR score instead remains normalized by treated mass on its evaluation
    support.  These targets therefore differ slightly even at ``delta=0``.
    """

    mask = common_evaluation_mask(trim_boundaries)
    treated_probability = population.group_probabilities[:, 0]
    didsc_target = float(
        np.sum(
            treated_probability[mask]
            * (
                population.treatment_effect[mask]
                + population.bandwidth_affine_change_gap_parent[mask]
            )
        )
        / np.sum(treated_probability)
        / np.mean(mask)
    )
    amplitude = local_amplitude(population, n, delta)
    average_gap = float(
        amplitude
        * np.sum(
            treated_probability[mask]
            * population.binary_pt_direction[mask]
        )
        / np.sum(treated_probability[mask])
    )
    dr_target = float(
        np.sum(
            treated_probability[mask]
            * population.treatment_effect[mask]
        )
        / np.sum(treated_probability[mask])
        + average_gap
    )
    return didsc_target, dr_target, average_gap


def replication_seed(master_seed: int, n: int, replication: int) -> int:
    """Return a deterministic seed invariant to worker count and ordering."""

    sequence = np.random.SeedSequence(
        [int(master_seed), int(n), int(replication)]
    )
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def generate_outer_sample(
    population: Population,
    n: int,
    base_seed: int,
) -> tuple[Sample, np.random.Generator]:
    """Draw one common panel, fixed folds, and multiplier stream."""

    sequence = np.random.SeedSequence(int(base_seed))
    data_stream, residual_stream, fold_stream, multiplier_stream = (
        sequence.spawn(4)
    )
    data_rng = np.random.default_rng(data_stream)
    residual_rng = np.random.default_rng(residual_stream)
    fold_rng = np.random.default_rng(fold_stream)
    multiplier_rng = np.random.default_rng(multiplier_stream)

    x_index = data_rng.integers(0, N_X, size=n)
    probabilities = population.group_probabilities[x_index]
    uniforms = data_rng.random(n)
    group = np.sum(
        uniforms[:, None] > np.cumsum(probabilities, axis=1),
        axis=1,
    ).astype(np.int64)
    group = np.minimum(group, N_GROUPS - 1)
    residuals = (
        residual_rng.normal(size=(n, N_PERIODS))
        * np.sqrt(population.residual_variances)[None, :]
    )
    outcomes = np.empty_like(residuals)
    treated = group == 0
    outcomes[treated] = population.treated_means_parent[
        x_index[treated]
    ]
    for donor_index in range(N_DONORS):
        donor = group == donor_index + 1
        outcomes[donor] = population.donor_means_parent[
            x_index[donor],
            :,
            donor_index,
        ]
    outcomes += residuals
    outcomes[treated, -1] += population.treatment_effect[x_index[treated]]

    post_direction = np.empty(n, dtype=float)
    post_direction[treated] = population.treated_post_direction[
        x_index[treated]
    ]
    for donor_index in range(N_DONORS):
        donor = group == donor_index + 1
        post_direction[donor] = population.donor_post_direction[donor_index]

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
    return (
        Sample(
            x_index=x_index,
            group=group,
            folds=folds,
            outcomes_parent=outcomes,
            post_direction=post_direction,
            fold_hash=fold_hash,
            latent_hash=latent_hash,
        ),
        multiplier_rng,
    )


@lru_cache(maxsize=32)
def kernel_geometry(
    n: int,
    coefficient: float,
    exponent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return local-linear kernel matrices for one method and sample size."""

    support = np.linspace(0.0, 1.0, N_X)
    bandwidth = coefficient * n**exponent
    centered = support[None, :] - support[:, None]
    scaled = centered / bandwidth
    kernel = np.where(
        np.abs(scaled) <= 1.0,
        0.75 * (1.0 - scaled * scaled),
        0.0,
    )
    kernel_centered = kernel * centered
    kernel_centered_sq = kernel * centered * centered
    return kernel, kernel_centered, kernel_centered_sq, float(bandwidth)


def aggregate_fold_batch(
    sample: Sample,
    weight_draws: np.ndarray,
    fold: int,
) -> FoldAggregates:
    """Aggregate all multiplier draws on one fixed fold."""

    draws, n = weight_draws.shape
    if n != sample.group.size:
        raise ValueError("multiplier width does not equal sample size")
    mask = sample.folds == fold
    flat = sample.group[mask] * N_X + sample.x_index[mask]
    size = N_GROUPS * N_X
    counts = np.empty((draws, N_GROUPS, N_X), dtype=float)
    outcome_sums = np.empty(
        (draws, N_GROUPS, N_PERIODS, N_X),
        dtype=float,
    )
    direction_sums = np.empty((draws, N_GROUPS, N_X), dtype=float)
    fold_outcomes = sample.outcomes_parent[mask]
    fold_direction = sample.post_direction[mask]
    for draw in range(draws):
        weights = weight_draws[draw, mask]
        counts[draw] = np.bincount(
            flat,
            weights=weights,
            minlength=size,
        ).reshape(N_GROUPS, N_X)
        for period in range(N_PERIODS):
            outcome_sums[draw, :, period] = np.bincount(
                flat,
                weights=weights * fold_outcomes[:, period],
                minlength=size,
            ).reshape(N_GROUPS, N_X)
        direction_sums[draw] = np.bincount(
            flat,
            weights=weights * fold_direction,
            minlength=size,
        ).reshape(N_GROUPS, N_X)
    return FoldAggregates(
        counts=counts,
        outcome_sums=outcome_sums,
        direction_sums=direction_sums,
    )


def local_linear_batch(
    counts: np.ndarray,
    response_sums: np.ndarray,
    geometry: tuple[np.ndarray, np.ndarray, np.ndarray, float],
) -> np.ndarray:
    """Return batched ridge-stabilized local-linear intercept paths."""

    kernel, kernel_centered, kernel_centered_sq, _ = geometry
    one_response = response_sums.ndim == 2
    responses = (
        response_sums[:, None, :]
        if one_response
        else response_sums
    )
    s0 = np.einsum("my,xy->mx", counts, kernel)
    s1 = np.einsum("my,xy->mx", counts, kernel_centered)
    s2 = np.einsum("my,xy->mx", counts, kernel_centered_sq)
    t0 = np.einsum("mry,xy->mrx", responses, kernel)
    t1 = np.einsum("mry,xy->mrx", responses, kernel_centered)
    ridge = RIDGE_LLR * np.maximum((s0 + s2) / 2.0, 1.0)
    a00 = s0 + ridge
    a11 = s2 + ridge
    determinant = np.maximum(a00 * a11 - s1 * s1, ridge)
    fitted = (
        a11[:, None, :] * t0 - s1[:, None, :] * t1
    ) / determinant[:, None, :]
    if one_response:
        return fitted[:, 0]
    return fitted


def local_ratio_batch(
    denominator_counts: np.ndarray,
    numerator_counts: np.ndarray,
    geometry: tuple[np.ndarray, np.ndarray, np.ndarray, float],
    enforce_positive: bool = False,
) -> np.ndarray:
    """Estimate local-linear ratios with the requested fallback rule.

    DiDSC deliberately uses the bandwidth experiment's rule, which falls
    back only for a near-singular local-linear system.  DRDiD additionally
    repairs nonpositive or nonfinite values before constructing binary odds.
    """

    kernel, kernel_centered, kernel_centered_sq, _ = geometry
    s0 = np.einsum("my,xy->mx", denominator_counts, kernel)
    s1 = np.einsum(
        "my,xy->mx",
        denominator_counts,
        kernel_centered,
    )
    s2 = np.einsum(
        "my,xy->mx",
        denominator_counts,
        kernel_centered_sq,
    )
    t0 = np.einsum("my,xy->mx", numerator_counts, kernel)
    t1 = np.einsum(
        "my,xy->mx",
        numerator_counts,
        kernel_centered,
    )
    determinant = s0 * s2 - s1 * s1
    adaptive = 1.0e-6 * np.maximum((s0 + s2) / 2.0, 1.0)
    regular = np.abs(determinant) >= adaptive
    result = np.empty_like(s0)
    result[regular] = (
        t0[regular] * s2[regular] - t1[regular] * s1[regular]
    ) / determinant[regular]
    safe_s0 = np.where(
        np.abs(s0) > adaptive,
        s0,
        s0 + adaptive,
    )
    local_constant = t0 / safe_s0
    result[~regular] = local_constant[~regular]
    if enforce_positive:
        invalid = ~np.isfinite(result) | (result <= 0.0)
        result[invalid] = local_constant[invalid]
    return result


def project_simplex_batch(raw_weights: np.ndarray) -> np.ndarray:
    """Project the final axis of an array onto the probability simplex."""

    ordered = np.sort(raw_weights, axis=-1)[..., ::-1]
    cumulative = np.cumsum(ordered, axis=-1)
    indices = np.arange(1, N_DONORS + 1, dtype=float)
    active = ordered * indices > cumulative - 1.0
    rho = np.sum(active, axis=-1).astype(int) - 1
    rho = np.maximum(rho, 0)
    threshold = (
        np.take_along_axis(cumulative, rho[..., None], axis=-1)[..., 0]
        - 1.0
    ) / (rho + 1.0)
    return np.maximum(raw_weights - threshold[..., None], 0.0)


def synthetic_weights_batch(
    moments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve and simplex-project all local affine SC weight systems."""

    matrix_m = (
        moments[:, :, :, 1:N_DONORS]
        - moments[:, :, :, [N_DONORS]]
    )
    target = moments[:, :, :, 0] - moments[:, :, :, N_DONORS]
    gram = np.einsum("mxpi,mxpj->mxij", matrix_m, matrix_m)
    rhs = np.einsum("mxpi,mxp->mxi", matrix_m, target)
    eigenvalues = np.linalg.eigvalsh(gram)
    lambda_min = np.maximum(eigenvalues[..., 0], 0.0)
    lambda_max = np.maximum(eigenvalues[..., -1], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        condition_mtm = np.where(
            lambda_min > 0.0,
            lambda_max / lambda_min,
            np.inf,
        )
    diagonal_scale = np.max(
        np.abs(np.diagonal(gram, axis1=-2, axis2=-1)),
        axis=-1,
    )
    ridge = np.where(
        condition_mtm > 1.0e6,
        RIDGE_SC * np.maximum(diagonal_scale, 1.0),
        RIDGE_SC,
    )
    regularized = gram + ridge[..., None, None] * np.eye(N_DONORS - 1)
    free_weights = np.linalg.solve(regularized, rhs[..., None])[..., 0]
    raw_weights = np.concatenate(
        [
            free_weights,
            1.0 - free_weights.sum(axis=-1, keepdims=True),
        ],
        axis=-1,
    )
    projected = project_simplex_batch(raw_weights)
    return projected, raw_weights, matrix_m, condition_mtm


def _finite_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide rowwise and mark nonpositive or nonfinite masses invalid."""

    valid = np.isfinite(denominator) & (denominator > 1.0e-12)
    result = np.full_like(numerator, np.nan, dtype=float)
    result[valid] = numerator[valid] / denominator[valid]
    return result


def estimate_batch_components(
    population: Population,
    sample: Sample,
    weight_draws: np.ndarray,
    trim_boundaries: bool,
    collect_influence: bool = False,
    compute_drdid: bool = True,
) -> BatchEstimate:
    """Estimate DiDSC and, when requested, DRDiD coefficients.

    All rows of ``weight_draws`` are evaluated together.  The first dimension
    may contain the unit-weight point estimate, multiplier draws, or both.
    Nuisances are re-estimated for every row.  Local alternatives are not
    separately fitted because every fitted object and score is exactly affine
    in the post-period direction while the pre-period SC weights and group
    ratios are invariant to it.
    """

    weights = np.asarray(weight_draws, dtype=float)
    if weights.ndim != 2 or weights.shape[1] != sample.group.size:
        raise ValueError("weight_draws must have shape (draws,n)")
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise FloatingPointError("invalid multiplier weights")
    if collect_influence and not compute_drdid:
        raise ValueError("DRDiD influence collection requires compute_drdid")
    draws, n = weights.shape
    didsc_geometry = kernel_geometry(
        n,
        DIDSC_BANDWIDTH_COEFFICIENT,
        DIDSC_BANDWIDTH_EXPONENT,
    )
    drdid_geometry = (
        kernel_geometry(
            n,
            DRDID_BANDWIDTH_COEFFICIENT,
            DRDID_BANDWIDTH_EXPONENT,
        )
        if compute_drdid
        else None
    )
    (
        evaluation_mask,
        retained_observations,
        retained_fold_counts,
        _,
        _,
    ) = sample_evaluation_geometry(sample, trim_boundaries)
    fold_aggregates = [
        aggregate_fold_batch(sample, weights, fold)
        for fold in range(N_FOLDS)
    ]

    treated_indicator = sample.group == 0
    full_sample_treated_share = (
        np.sum(weights[:, treated_indicator], axis=1) / n
    )
    if (
        not np.all(np.isfinite(full_sample_treated_share))
        or np.any(full_sample_treated_share <= 0.0)
    ):
        raise FloatingPointError("weighted treated share is invalid")
    didsc_base = np.zeros(draws, dtype=float)
    didsc_direction = np.zeros(draws, dtype=float)
    dr_treated_mass = np.zeros(draws, dtype=float)
    dr_treated_base_numerator = np.zeros(draws, dtype=float)
    dr_treated_direction_numerator = np.zeros(draws, dtype=float)
    dr_control_base_numerator = np.zeros(draws, dtype=float)
    dr_control_direction_numerator = np.zeros(draws, dtype=float)
    dr_control_mass = np.zeros(draws, dtype=float)

    oof_m0_base = np.full(n, np.nan) if collect_influence else None
    oof_m0_direction = np.full(n, np.nan) if collect_influence else None
    oof_odds = np.full(n, np.nan) if collect_influence else None
    diagnostic_kappas: list[float] = []
    diagnostic_sigmas: list[np.ndarray] = []
    diagnostic_projected: list[np.ndarray] = []
    diagnostic_raw: list[np.ndarray] = []
    diagnostic_projection_distances: list[np.ndarray] = []
    diagnostic_didsc_ratio_minima: list[float] = []
    diagnostic_drdid_ratio_minima: list[float] = []

    for test_fold in range(N_FOLDS):
        train = fold_aggregates[1 - test_fold]
        test = fold_aggregates[test_fold]

        fitted_pre = np.empty(
            (draws, N_X, N_GROUPS, N_PRE),
            dtype=float,
        )
        for group_index in range(N_GROUPS):
            fitted = local_linear_batch(
                train.counts[:, group_index],
                train.outcome_sums[:, group_index, :N_PRE],
                didsc_geometry,
            )
            fitted_pre[:, :, group_index, :] = np.transpose(
                fitted,
                (0, 2, 1),
            )
        moments = np.transpose(fitted_pre, (0, 1, 3, 2))
        (
            donor_weights,
            raw_weights,
            matrix_m,
            _,
        ) = synthetic_weights_batch(moments)

        pooled_control_counts = train.counts[:, 1:].sum(axis=1)
        pooled_control_change_sums = (
            train.outcome_sums[:, 1:, -1].sum(axis=1)
            - train.outcome_sums[:, 1:, -2].sum(axis=1)
        )
        pooled_control_direction_sums = (
            train.direction_sums[:, 1:].sum(axis=1)
        )
        didsc_m0_base = local_linear_batch(
            pooled_control_counts,
            pooled_control_change_sums,
            didsc_geometry,
        )
        didsc_m0_direction = local_linear_batch(
            pooled_control_counts,
            pooled_control_direction_sums,
            didsc_geometry,
        )
        treated_counts = train.counts[:, 0]
        didsc_donor_ratios = np.stack(
            [
                local_ratio_batch(
                    train.counts[:, donor_index],
                    treated_counts,
                    didsc_geometry,
                )
                for donor_index in range(1, N_GROUPS)
            ],
            axis=-1,
        )
        if not np.all(
            np.isfinite(didsc_donor_ratios[:, evaluation_mask])
        ):
            raise FloatingPointError("nonfinite DiDSC treated-to-donor ratio")

        drdid_m0_base: np.ndarray | None = None
        drdid_m0_direction: np.ndarray | None = None
        drdid_donor_ratios: np.ndarray | None = None
        binary_odds: np.ndarray | None = None
        if compute_drdid:
            assert drdid_geometry is not None
            drdid_m0_base = local_linear_batch(
                pooled_control_counts,
                pooled_control_change_sums,
                drdid_geometry,
            )
            drdid_m0_direction = local_linear_batch(
                pooled_control_counts,
                pooled_control_direction_sums,
                drdid_geometry,
            )
            drdid_donor_ratios = np.stack(
                [
                    local_ratio_batch(
                        train.counts[:, donor_index],
                        treated_counts,
                        drdid_geometry,
                        enforce_positive=True,
                    )
                    for donor_index in range(1, N_GROUPS)
                ],
                axis=-1,
            )
            if (
                not np.all(np.isfinite(drdid_donor_ratios))
                or np.any(drdid_donor_ratios <= 0.0)
            ):
                raise FloatingPointError(
                    "nonpositive DRDiD treated-to-donor local ratio"
                )
            with np.errstate(divide="ignore", invalid="ignore"):
                binary_odds = 1.0 / np.sum(
                    1.0 / drdid_donor_ratios,
                    axis=-1,
                )
            if (
                not np.all(np.isfinite(binary_odds))
                or np.any(binary_odds <= 0.0)
            ):
                raise FloatingPointError("nonpositive binary propensity odds")

        base_change_sums = (
            test.outcome_sums[:, :, -1]
            - test.outcome_sums[:, :, -2]
        )
        didsc_base_residual_sums = (
            base_change_sums
            - test.counts * didsc_m0_base[:, None, :]
        )
        didsc_direction_residual_sums = (
            test.direction_sums
            - test.counts * didsc_m0_direction[:, None, :]
        )
        score_coefficients = np.empty(
            (draws, N_GROUPS, N_X),
            dtype=float,
        )
        score_coefficients[:, 0] = 1.0
        score_coefficients[:, 1:] = -np.transpose(
            donor_weights * didsc_donor_ratios,
            (0, 2, 1),
        )
        didsc_base_contributions = np.transpose(
            score_coefficients[:, :, evaluation_mask]
            * didsc_base_residual_sums[:, :, evaluation_mask],
            (0, 2, 1),
        )
        didsc_direction_contributions = np.transpose(
            score_coefficients[:, :, evaluation_mask]
            * didsc_direction_residual_sums[:, :, evaluation_mask],
            (0, 2, 1),
        )
        didsc_base_fold_numerator = np.sum(
            didsc_base_contributions,
            axis=(1, 2),
        )
        didsc_direction_fold_numerator = np.sum(
            didsc_direction_contributions,
            axis=(1, 2),
        )
        didsc_fold_denominator = (
            full_sample_treated_share
            * float(retained_fold_counts[test_fold])
            * N_FOLDS
        )
        didsc_base += _finite_divide(
            didsc_base_fold_numerator,
            didsc_fold_denominator,
        )
        didsc_direction += _finite_divide(
            didsc_direction_fold_numerator,
            didsc_fold_denominator,
        )
        dr_treated_mass += np.sum(
            test.counts[:, 0, evaluation_mask],
            axis=1,
        )

        if compute_drdid:
            assert drdid_m0_base is not None
            assert drdid_m0_direction is not None
            assert binary_odds is not None
            drdid_base_residual_sums = (
                base_change_sums
                - test.counts * drdid_m0_base[:, None, :]
            )
            drdid_direction_residual_sums = (
                test.direction_sums
                - test.counts * drdid_m0_direction[:, None, :]
            )
            dr_treated_base_numerator += np.sum(
                drdid_base_residual_sums[:, 0, evaluation_mask],
                axis=1,
            )
            dr_treated_direction_numerator += np.sum(
                drdid_direction_residual_sums[:, 0, evaluation_mask],
                axis=1,
            )
            control_base_residual = drdid_base_residual_sums[:, 1:].sum(
                axis=1
            )
            control_direction_residual = (
                drdid_direction_residual_sums[:, 1:].sum(axis=1)
            )
            control_counts = test.counts[:, 1:].sum(axis=1)
            dr_control_base_numerator += np.sum(
                binary_odds[:, evaluation_mask]
                * control_base_residual[:, evaluation_mask],
                axis=1,
            )
            dr_control_direction_numerator += np.sum(
                binary_odds[:, evaluation_mask]
                * control_direction_residual[:, evaluation_mask],
                axis=1,
            )
            dr_control_mass += np.sum(
                binary_odds[:, evaluation_mask]
                * control_counts[:, evaluation_mask],
                axis=1,
            )

        if collect_influence:
            if draws != 1:
                raise ValueError("influence collection requires one draw")
            assert drdid_m0_base is not None
            assert drdid_m0_direction is not None
            assert drdid_donor_ratios is not None
            assert binary_odds is not None
            test_observations = sample.folds == test_fold
            test_x = sample.x_index[test_observations]
            assert oof_m0_base is not None
            assert oof_m0_direction is not None
            assert oof_odds is not None
            oof_m0_base[test_observations] = drdid_m0_base[0, test_x]
            oof_m0_direction[test_observations] = (
                drdid_m0_direction[0, test_x]
            )
            oof_odds[test_observations] = binary_odds[0, test_x]
            diagnostic_didsc_ratio_minima.append(
                float(np.min(didsc_donor_ratios[:, evaluation_mask]))
            )
            diagnostic_drdid_ratio_minima.append(
                float(np.min(drdid_donor_ratios[:, evaluation_mask]))
            )
            singular = np.linalg.svd(
                matrix_m[0, evaluation_mask],
                compute_uv=False,
            )
            diagnostic_sigmas.append(singular)
            diagnostic_kappas.extend(
                (singular[:, 0] / singular[:, -1]).tolist()
            )
            diagnostic_projected.append(donor_weights[0, evaluation_mask])
            diagnostic_raw.append(raw_weights[0, evaluation_mask])
            diagnostic_projection_distances.append(
                np.linalg.norm(
                    donor_weights[0, evaluation_mask]
                    - raw_weights[0, evaluation_mask],
                    axis=1,
                )
            )

    if compute_drdid:
        dr_treated_base = _finite_divide(
            dr_treated_base_numerator,
            dr_treated_mass,
        )
        dr_treated_direction = _finite_divide(
            dr_treated_direction_numerator,
            dr_treated_mass,
        )
        dr_control_base = _finite_divide(
            dr_control_base_numerator,
            dr_control_mass,
        )
        dr_control_direction = _finite_divide(
            dr_control_direction_numerator,
            dr_control_mass,
        )
        drdid_base = dr_treated_base - dr_control_base
        drdid_direction = dr_treated_direction - dr_control_direction
    else:
        drdid_base = np.full(draws, np.nan)
        drdid_direction = np.full(draws, np.nan)

    influence_base: np.ndarray | None = None
    influence_direction: np.ndarray | None = None
    diagnostics: dict[str, float | int] = {}
    if collect_influence:
        assert oof_m0_base is not None
        assert oof_m0_direction is not None
        assert oof_odds is not None
        if not (
            np.all(np.isfinite(oof_m0_base))
            and np.all(np.isfinite(oof_m0_direction))
            and np.all(np.isfinite(oof_odds))
        ):
            raise FloatingPointError("nonfinite out-of-fold DRDID nuisance")
        observation_mask = retained_observations.astype(float)
        treated = (sample.group == 0).astype(float)
        treated_mean = float(np.mean(observation_mask * treated))
        control_score = observation_mask * (1.0 - treated) * oof_odds
        control_mean = float(np.mean(control_score))
        if treated_mean <= 0.0 or control_mean <= 0.0:
            raise FloatingPointError("invalid DRDID normalization")
        w1 = observation_mask * treated / treated_mean
        w0 = control_score / control_mean
        base_change = (
            sample.outcomes_parent[:, -1]
            - sample.outcomes_parent[:, -2]
        )
        base_residual = base_change - oof_m0_base
        direction_residual = sample.post_direction - oof_m0_direction
        influence_base = (
            (w1 - w0) * base_residual - w1 * drdid_base[0]
        )
        influence_direction = (
            (w1 - w0) * direction_residual
            - w1 * drdid_direction[0]
        )
        if abs(float(np.mean(influence_base))) > 2.0e-10:
            raise AssertionError("DRDID parent influence is not centered")
        if abs(float(np.mean(influence_direction))) > 2.0e-10:
            raise AssertionError("DRDID direction influence is not centered")

        sigma_array = np.concatenate(diagnostic_sigmas, axis=0)
        projected_array = np.concatenate(diagnostic_projected, axis=0)
        raw_array = np.concatenate(diagnostic_raw, axis=0)
        projection_array = np.concatenate(
            diagnostic_projection_distances,
            axis=0,
        )
        kappa_array = np.asarray(diagnostic_kappas, dtype=float)
        diagnostics = {
            "sample_kappa_m_median": float(np.median(kappa_array)),
            "sample_kappa_m_p95": float(np.quantile(kappa_array, 0.95)),
            "sample_kappa_m_max": float(np.max(kappa_array)),
            "sample_sigma_max_median": float(np.median(sigma_array[:, 0])),
            "sample_sigma_middle_median": float(np.median(sigma_array[:, 1])),
            "sample_sigma_min_median": float(np.median(sigma_array[:, -1])),
            "sample_rank_min": int(
                np.min(np.sum(sigma_array > 1.0e-10, axis=1))
            ),
            "maximum_estimated_weight": float(np.max(projected_array)),
            "minimum_estimated_weight": float(np.min(projected_array)),
            "estimated_weight_l2_median": float(
                np.median(np.linalg.norm(projected_array, axis=1))
            ),
            "raw_weight_l2_median": float(
                np.median(np.linalg.norm(raw_array, axis=1))
            ),
            "simplex_projection_distance_median": float(
                np.median(projection_array)
            ),
            "didsc_ratio_minimum": float(
                np.min(diagnostic_didsc_ratio_minima)
            ),
            "drdid_ratio_minimum": float(
                np.min(diagnostic_drdid_ratio_minima)
            ),
        }

    return BatchEstimate(
        didsc_base=didsc_base,
        didsc_direction=didsc_direction,
        drdid_base=drdid_base,
        drdid_direction=drdid_direction,
        dr_if_base=influence_base,
        dr_if_direction=influence_direction,
        diagnostics=diagnostics,
    )


def population_configuration_diagnostics(
    population: Population,
    n: int,
    delta: float,
) -> dict[str, Any]:
    """Audit one exact-SC local-PT population configuration."""

    amplitude = local_amplitude(population, n, delta)
    donor_means = population.donor_means_parent.copy()
    donor_means[:, -1] += (
        amplitude * population.donor_post_direction[None, :]
    )
    treated_means = np.einsum(
        "xd,xtd->xt",
        population.true_weights,
        donor_means,
    )
    sc_residual = treated_means - np.einsum(
        "xd,xtd->xt",
        population.true_weights,
        donor_means,
    )
    donor_probability = population.group_probabilities[:, 1:].copy()
    donor_probability /= donor_probability.sum(axis=1, keepdims=True)
    donor_changes = donor_means[:, -1] - donor_means[:, -2]
    treated_change = treated_means[:, -1] - treated_means[:, -2]
    pooled_change = np.sum(donor_probability * donor_changes, axis=1)
    pt_gap = treated_change - pooled_change
    treated_probability = population.group_probabilities[:, 0]
    average_gap = float(
        np.sum(treated_probability * pt_gap)
        / np.sum(treated_probability)
    )
    expected_gap = amplitude
    if float(np.max(np.abs(sc_residual))) > 1.0e-12:
        raise AssertionError("local population lost exact SC")
    if not math.isclose(
        average_gap,
        expected_gap,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
    ):
        raise AssertionError("local average PT gap has wrong normalization")
    pre_difference = (
        donor_means[:, :N_PRE] - population.donor_means_parent[:, :N_PRE]
    )
    if not np.array_equal(pre_difference, np.zeros_like(pre_difference)):
        raise AssertionError("local perturbation changed donor pre means")
    if delta == 0.0:
        if not np.array_equal(donor_means, population.donor_means_parent):
            raise AssertionError("zero local configuration changed the parent")
    elif float(np.max(np.abs(pt_gap))) <= 1.0e-12:
        raise AssertionError("nonzero local configuration does not violate PT")
    return {
        "amplitude": amplitude,
        "donor_means": donor_means,
        "treated_means": treated_means,
        "pt_gap": pt_gap,
        "average_gap": average_gap,
        "maximum_sc_residual": float(np.max(np.abs(sc_residual))),
        "maximum_absolute_pt_gap": float(np.max(np.abs(pt_gap))),
    }


def run_internal_audits(
    delta_grid: tuple[float, ...] = DEFAULT_DELTA_GRID,
) -> dict[str, float]:
    """Validate the DGP, affine reuse, batching, IF, and CI arithmetic."""

    population = build_population()
    for n in DEFAULT_SAMPLE_SIZES:
        for delta in delta_grid:
            diagnostics = population_configuration_diagnostics(
                population,
                n,
                delta,
            )
            standardized = (
                math.sqrt(n)
                * float(diagnostics["average_gap"])
                / population.oracle_dr_rootn_sd
            )
            if not math.isclose(
                standardized,
                delta,
                rel_tol=REL_TOL,
                abs_tol=ABS_TOL,
            ):
                raise AssertionError("standardized local gap is incorrect")

    n_audit = 800
    base_seed = replication_seed(2026080802, n_audit, 991)
    sample, multiplier_rng = generate_outer_sample(
        population,
        n_audit,
        base_seed,
    )
    multiplier = multiplier_rng.exponential(1.0, size=n_audit)
    joint_weights = np.vstack([np.ones(n_audit), multiplier])
    joint = estimate_batch_components(
        population,
        sample,
        joint_weights,
        trim_boundaries=DEFAULT_TRIM_BOUNDARIES,
        collect_influence=False,
    )
    maximum_batch_error = 0.0
    for draw in range(2):
        separate = estimate_batch_components(
            population,
            sample,
            joint_weights[[draw]],
            trim_boundaries=DEFAULT_TRIM_BOUNDARIES,
            collect_influence=False,
        )
        for joint_values, separate_values in (
            (joint.didsc_base, separate.didsc_base),
            (joint.didsc_direction, separate.didsc_direction),
            (joint.drdid_base, separate.drdid_base),
            (joint.drdid_direction, separate.drdid_direction),
        ):
            maximum_batch_error = max(
                maximum_batch_error,
                abs(float(joint_values[draw] - separate_values[0])),
            )
    if maximum_batch_error > 2.0e-10:
        raise AssertionError("batched and separate estimators disagree")
    didsc_only = estimate_batch_components(
        population,
        sample,
        joint_weights,
        trim_boundaries=DEFAULT_TRIM_BOUNDARIES,
        collect_influence=False,
        compute_drdid=False,
    )
    maximum_didsc_fast_path_error = float(
        max(
            np.max(np.abs(joint.didsc_base - didsc_only.didsc_base)),
            np.max(
                np.abs(joint.didsc_direction - didsc_only.didsc_direction)
            ),
        )
    )
    if maximum_didsc_fast_path_error > 2.0e-10:
        raise AssertionError("DiDSC-only bootstrap fast path changed estimates")

    outer = estimate_batch_components(
        population,
        sample,
        np.ones((1, n_audit)),
        trim_boundaries=DEFAULT_TRIM_BOUNDARIES,
        collect_influence=True,
    )
    if outer.dr_if_base is None or outer.dr_if_direction is None:
        raise AssertionError("DRDID influence audit was not produced")
    maximum_affine_error = 0.0
    for delta in (-1.25, 1.25):
        amplitude = local_amplitude(population, n_audit, delta)
        local_outcomes = sample.outcomes_parent.copy()
        local_outcomes[:, -1] += amplitude * sample.post_direction
        local_sample = Sample(
            x_index=sample.x_index,
            group=sample.group,
            folds=sample.folds,
            outcomes_parent=local_outcomes,
            post_direction=np.zeros(n_audit),
            fold_hash=sample.fold_hash,
            latent_hash=sample.latent_hash,
        )
        direct = estimate_batch_components(
            population,
            local_sample,
            np.ones((1, n_audit)),
            trim_boundaries=DEFAULT_TRIM_BOUNDARIES,
            collect_influence=True,
        )
        maximum_affine_error = max(
            maximum_affine_error,
            abs(
                float(
                    direct.didsc_base[0]
                    - (outer.didsc_base[0] + amplitude * outer.didsc_direction[0])
                )
            ),
            abs(
                float(
                    direct.drdid_base[0]
                    - (outer.drdid_base[0] + amplitude * outer.drdid_direction[0])
                )
            ),
        )
    if maximum_affine_error > 2.0e-10:
        raise AssertionError("local affine reuse differs from direct fitting")
    return {
        "maximum_batch_error": maximum_batch_error,
        "maximum_didsc_fast_path_error": maximum_didsc_fast_path_error,
        "maximum_affine_error": maximum_affine_error,
        "oracle_dr_rootn_sd": population.oracle_dr_rootn_sd,
        "direction_normalizer": population.direction_normalizer,
    }


OUTER_FIELDS = [
    "experiment_version",
    "dgp_version",
    "experiment_purpose",
    "paper_reference",
    "method",
    "interval_method",
    "n",
    "outer_replication",
    "master_seed",
    "base_seed_hex",
    "fold_hash",
    "latent_hash",
    "delta",
    "local_scale",
    "local_amplitude",
    "root_n_local_amplitude",
    "standardized_full_support_pt_violation",
    "kappa_m",
    "kappa_mtm_derived",
    "condition_definition",
    "donor_codes",
    "donor_states",
    "baseline_donor",
    "sc_holds_by_construction",
    "pt_holds_by_construction",
    "full_population_binary_pt_gap",
    "evaluation_population_binary_pt_gap",
    "minimum_binary_pt_gap",
    "maximum_binary_pt_gap",
    "maximum_absolute_binary_pt_gap",
    "att_true",
    "common_sc_population_target",
    "method_population_target",
    "population_target_minus_att",
    "folds",
    "kernel",
    "didsc_bandwidth_coefficient",
    "didsc_bandwidth_exponent",
    "didsc_effective_bandwidth",
    "drdid_bandwidth_coefficient",
    "drdid_bandwidth_exponent",
    "drdid_effective_bandwidth",
    "method_effective_bandwidth",
    "local_linear_ridge",
    "sc_weight_ridge",
    "trim_boundaries",
    "nuisance_estimation",
    "didsc_score_normalization",
    "weight_estimator",
    "drdid_odds_construction",
    "point_estimate",
    "bias_vs_att",
    "error_vs_common_sc_target",
    "error_vs_method_target",
    "estimated_se",
    "ci_low",
    "ci_high",
    "ci_length",
    "covers_att",
    "covers_common_sc_target",
    "covers_method_target",
    "bootstrap_critical_value",
    "bootstrap_root_mean",
    "bootstrap_root_sd",
    "requested_bootstraps",
    "valid_bootstraps",
    "bootstrap_failures",
    "influence_mean",
    "influence_variance",
    "sample_kappa_m_median",
    "sample_kappa_m_p95",
    "sample_kappa_m_max",
    "sample_sigma_max_median",
    "sample_sigma_middle_median",
    "sample_sigma_min_median",
    "sample_rank_min",
    "maximum_estimated_weight",
    "minimum_estimated_weight",
    "estimated_weight_l2_median",
    "raw_weight_l2_median",
    "simplex_projection_distance_median",
    "didsc_ratio_minimum",
    "drdid_ratio_minimum",
    "status",
    "error_message",
    "runtime_seconds",
]


BOOTSTRAP_FIELDS = [
    "experiment_version",
    "dgp_version",
    "n",
    "outer_replication",
    "master_seed",
    "base_seed_hex",
    "fold_hash",
    "delta",
    "local_amplitude",
    "point_estimate",
    "requested_bootstraps",
    "valid_bootstraps",
    "bootstrap_failures",
    "bootstrap_root_mean",
    "bootstrap_root_sd",
    "symmetric_critical_value",
    "symmetric_ci_low",
    "symmetric_ci_high",
    "symmetric_ci_length",
    "symmetric_covers_att",
    "failure_messages",
]


COVERAGE_FIELDS = [
    "experiment_version",
    "dgp_version",
    "method",
    "interval_method",
    "method_effective_bandwidth",
    "n",
    "delta",
    "local_scale",
    "local_amplitude",
    "standardized_full_support_pt_violation",
    "kappa_m",
    "sc_holds_by_construction",
    "pt_holds_by_construction",
    "attempted_outer_replications",
    "valid_point_estimates",
    "valid_intervals",
    "failed_intervals",
    "att_true",
    "common_sc_population_target",
    "method_population_target",
    "mean_point_estimate",
    "bias_vs_att",
    "bias_vs_method_target",
    "empirical_sd",
    "rmse_vs_att",
    "rmse_vs_method_target",
    "mean_estimated_se",
    "median_estimated_se",
    "mean_se_over_empirical_sd",
    "ci_coverage_att",
    "ci_coverage_att_mcse",
    "ci_coverage_common_sc_target",
    "ci_coverage_method_target",
    "mean_ci_length",
    "median_ci_length",
    "requested_bootstraps_total",
    "valid_bootstraps_total",
    "bootstrap_failures_total",
    "milestone_percent",
    "elapsed_wall_seconds",
]


POPULATION_FIELDS = [
    "experiment_version",
    "dgp_version",
    "n",
    "delta",
    "x_index",
    "x",
    "kappa_m",
    "kappa_mtm_derived",
    "sigma_max_m",
    "sigma_middle_m",
    "sigma_min_m",
    "rank_m",
    "sc_residual",
    "binary_pt_gap",
    "full_population_binary_pt_gap",
    "local_scale",
    "local_amplitude",
    "root_n_local_amplitude",
    "treated_probability",
    "true_weight_minimum",
    "true_weight_maximum",
    "att_true",
    "drdid_population_target",
    "didsc_population_target",
    "donor_post_direction",
]


def _configuration_common(
    population: Population,
    n: int,
    replication: int,
    master_seed: int,
    base_seed: int,
    fold_hash: str,
    latent_hash: str,
    delta: float,
    trim_boundaries: bool,
) -> dict[str, Any]:
    """Return DGP and implementation metadata shared by both methods."""

    amplitude = local_amplitude(population, n, delta)
    common_target, dr_target, evaluation_gap = method_population_targets(
        population,
        n,
        delta,
        trim_boundaries,
    )
    pt_gap = amplitude * population.binary_pt_direction
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "dgp_version": DGP_VERSION,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "paper_reference": PAPER_REFERENCE,
        "n": n,
        "outer_replication": replication,
        "master_seed": master_seed,
        "base_seed_hex": f"0x{base_seed:016x}",
        "fold_hash": fold_hash,
        "latent_hash": latent_hash,
        "delta": delta,
        "local_scale": population.oracle_dr_rootn_sd,
        "local_amplitude": amplitude,
        "root_n_local_amplitude": math.sqrt(n) * amplitude,
        "standardized_full_support_pt_violation": delta,
        "kappa_m": KAPPA_M,
        "kappa_mtm_derived": KAPPA_M**2,
        "condition_definition": CONDITION_DEFINITION,
        "donor_codes": "|".join(str(code) for code in DONOR_CODES),
        "donor_states": DONOR_STATES,
        "baseline_donor": BASELINE_DONOR,
        "sc_holds_by_construction": True,
        "pt_holds_by_construction": bool(delta == 0.0),
        "full_population_binary_pt_gap": amplitude,
        "evaluation_population_binary_pt_gap": evaluation_gap,
        "minimum_binary_pt_gap": float(np.min(pt_gap)),
        "maximum_binary_pt_gap": float(np.max(pt_gap)),
        "maximum_absolute_binary_pt_gap": float(np.max(np.abs(pt_gap))),
        "att_true": population.att_true,
        "common_sc_population_target": common_target,
        "folds": N_FOLDS,
        "kernel": KERNEL_NAME,
        "didsc_bandwidth_coefficient": DIDSC_BANDWIDTH_COEFFICIENT,
        "didsc_bandwidth_exponent": DIDSC_BANDWIDTH_EXPONENT,
        "didsc_effective_bandwidth": (
            DIDSC_BANDWIDTH_COEFFICIENT * n**DIDSC_BANDWIDTH_EXPONENT
        ),
        "drdid_bandwidth_coefficient": DRDID_BANDWIDTH_COEFFICIENT,
        "drdid_bandwidth_exponent": DRDID_BANDWIDTH_EXPONENT,
        "drdid_effective_bandwidth": (
            DRDID_BANDWIDTH_COEFFICIENT * n**DRDID_BANDWIDTH_EXPONENT
        ),
        "local_linear_ridge": RIDGE_LLR,
        "sc_weight_ridge": RIDGE_SC,
        "trim_boundaries": trim_boundaries,
        "nuisance_estimation": NUISANCE_DESCRIPTION,
        "didsc_score_normalization": DIDSC_SCORE_NORMALIZATION,
        "weight_estimator": WEIGHT_ESTIMATOR,
        "drdid_odds_construction": (
            "q_hat=1/sum_d(1/r_d_hat), using the same donor-wise "
            "r_d_hat=p_treated/p_donor construction as DiDSC but the "
            "DRDiD bandwidth"
        ),
        "_dr_target": dr_target,
    }


def _method_row(
    common: dict[str, Any],
    method: str,
    point_estimate: float,
    estimated_se: float,
    ci_low: float,
    ci_high: float,
    bootstrap_critical: float,
    bootstrap_root_mean: float,
    bootstrap_root_sd: float,
    requested_bootstraps: int,
    valid_bootstraps: int,
    influence_mean: float,
    influence_variance: float,
    diagnostics: dict[str, float | int],
    status: str,
    error_message: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    """Construct one complete method row."""

    row = {key: value for key, value in common.items() if not key.startswith("_")}
    method_target = (
        common["common_sc_population_target"]
        if method == "didsc_multiplier"
        else common["_dr_target"]
    )
    interval_method = (
        "symmetric_multiplier_bootstrap"
        if method == "didsc_multiplier"
        else "santanna_zhao_analytic_normal"
    )
    finite_point = np.isfinite(point_estimate)
    finite_interval = np.isfinite(ci_low) and np.isfinite(ci_high)
    row.update(
        {
            "method": method,
            "interval_method": interval_method,
            "method_effective_bandwidth": (
                common["didsc_effective_bandwidth"]
                if method == "didsc_multiplier"
                else common["drdid_effective_bandwidth"]
            ),
            "method_population_target": method_target,
            "population_target_minus_att": (
                method_target - float(common["att_true"])
            ),
            "point_estimate": point_estimate,
            "bias_vs_att": (
                point_estimate - float(common["att_true"])
                if finite_point
                else np.nan
            ),
            "error_vs_common_sc_target": (
                point_estimate - float(common["common_sc_population_target"])
                if finite_point
                else np.nan
            ),
            "error_vs_method_target": (
                point_estimate - method_target if finite_point else np.nan
            ),
            "estimated_se": estimated_se,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_length": ci_high - ci_low if finite_interval else np.nan,
            "covers_att": (
                bool(ci_low <= common["att_true"] <= ci_high)
                if finite_interval
                else ""
            ),
            "covers_common_sc_target": (
                bool(
                    ci_low
                    <= common["common_sc_population_target"]
                    <= ci_high
                )
                if finite_interval
                else ""
            ),
            "covers_method_target": (
                bool(ci_low <= method_target <= ci_high)
                if finite_interval
                else ""
            ),
            "bootstrap_critical_value": bootstrap_critical,
            "bootstrap_root_mean": bootstrap_root_mean,
            "bootstrap_root_sd": bootstrap_root_sd,
            "requested_bootstraps": requested_bootstraps,
            "valid_bootstraps": valid_bootstraps,
            "bootstrap_failures": requested_bootstraps - valid_bootstraps,
            "influence_mean": influence_mean,
            "influence_variance": influence_variance,
            "status": status,
            "error_message": error_message,
            "runtime_seconds": runtime_seconds,
        }
    )
    row.update(diagnostics)
    return row


def _failed_replication_output(
    population: Population,
    n: int,
    replication: int,
    bootstraps: int,
    master_seed: int,
    base_seed: int,
    delta_grid: tuple[float, ...],
    trim_boundaries: bool,
    fold_hash: str,
    latent_hash: str,
    error: Exception,
    runtime_seconds: float,
) -> ReplicationOutput:
    """Return structurally complete failed rows for one task."""

    message = f"{type(error).__name__}: {error}"
    rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for delta in delta_grid:
        common = _configuration_common(
            population,
            n,
            replication,
            master_seed,
            base_seed,
            fold_hash,
            latent_hash,
            delta,
            trim_boundaries,
        )
        for method in METHODS:
            rows.append(
                _method_row(
                    common,
                    method,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    bootstraps if method == "didsc_multiplier" else 0,
                    0,
                    np.nan,
                    np.nan,
                    {},
                    "failed",
                    message,
                    runtime_seconds / max(len(delta_grid) * len(METHODS), 1),
                )
            )
        bootstrap_rows.append(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "dgp_version": DGP_VERSION,
                "n": n,
                "outer_replication": replication,
                "master_seed": master_seed,
                "base_seed_hex": f"0x{base_seed:016x}",
                "fold_hash": fold_hash,
                "delta": delta,
                "local_amplitude": local_amplitude(population, n, delta),
                "requested_bootstraps": bootstraps,
                "valid_bootstraps": 0,
                "bootstrap_failures": bootstraps,
                "failure_messages": json.dumps([message]),
            }
        )
    return ReplicationOutput(rows, bootstrap_rows)


def run_replication_task(
    task: tuple[
        int,
        int,
        int,
        int,
        int,
        tuple[float, ...],
        bool,
    ],
) -> ReplicationOutput:
    """Run every local configuration and both methods on one outer panel."""

    (
        replication,
        n,
        bootstraps,
        bootstrap_batch_size,
        master_seed,
        delta_grid,
        trim_boundaries,
    ) = task
    started = time.perf_counter()
    population = build_population()
    base_seed = replication_seed(master_seed, n, replication)
    fold_hash = ""
    latent_hash = ""
    try:
        sample, multiplier_rng = generate_outer_sample(
            population,
            n,
            base_seed,
        )
        fold_hash = sample.fold_hash
        latent_hash = sample.latent_hash
        outer = estimate_batch_components(
            population,
            sample,
            np.ones((1, n)),
            trim_boundaries,
            collect_influence=True,
        )
        if outer.dr_if_base is None or outer.dr_if_direction is None:
            raise AssertionError("DRDID influence vectors are missing")

        bootstrap_base = np.full(bootstraps, np.nan)
        bootstrap_direction = np.full(bootstraps, np.nan)
        failure_messages: dict[int, str] = {}
        position = 0
        while position < bootstraps:
            take = min(bootstrap_batch_size, bootstraps - position)
            multiplier_weights = multiplier_rng.exponential(
                1.0,
                size=(take, n),
            )
            try:
                batch = estimate_batch_components(
                    population,
                    sample,
                    multiplier_weights,
                    trim_boundaries,
                    collect_influence=False,
                    compute_drdid=False,
                )
                bootstrap_base[position : position + take] = batch.didsc_base
                bootstrap_direction[position : position + take] = (
                    batch.didsc_direction
                )
            except Exception as batch_error:
                for local_index in range(take):
                    bootstrap_index = position + local_index
                    try:
                        single = estimate_batch_components(
                            population,
                            sample,
                            multiplier_weights[[local_index]],
                            trim_boundaries,
                            collect_influence=False,
                            compute_drdid=False,
                        )
                        bootstrap_base[bootstrap_index] = single.didsc_base[0]
                        bootstrap_direction[bootstrap_index] = (
                            single.didsc_direction[0]
                        )
                    except Exception as error:
                        failure_messages[bootstrap_index] = (
                            f"{type(error).__name__}: {error}"
                        )
                if not failure_messages:
                    failure_messages[position] = (
                        f"batch recovered after {type(batch_error).__name__}: "
                        f"{batch_error}"
                    )
            position += take

        for bootstrap_index in np.flatnonzero(~np.isfinite(bootstrap_base)):
            failure_messages.setdefault(
                int(bootstrap_index),
                "nonfinite DiDSC bootstrap base",
            )
        for bootstrap_index in np.flatnonzero(
            ~np.isfinite(bootstrap_direction)
        ):
            failure_messages.setdefault(
                int(bootstrap_index),
                "nonfinite DiDSC bootstrap local-direction coefficient",
            )

        total_runtime = float(time.perf_counter() - started)
        runtime_per_row = total_runtime / max(len(delta_grid) * len(METHODS), 1)
        outer_rows: list[dict[str, Any]] = []
        bootstrap_rows: list[dict[str, Any]] = []
        for delta in delta_grid:
            common = _configuration_common(
                population,
                n,
                replication,
                master_seed,
                base_seed,
                fold_hash,
                latent_hash,
                delta,
                trim_boundaries,
            )
            amplitude = float(common["local_amplitude"])
            if amplitude == 0.0:
                didsc_point = float(outer.didsc_base[0])
                bootstrap_estimates = bootstrap_base.copy()
            else:
                didsc_point = float(
                    outer.didsc_base[0]
                    + amplitude * outer.didsc_direction[0]
                )
                bootstrap_estimates = (
                    bootstrap_base + amplitude * bootstrap_direction
                )
            bootstrap_roots = bootstrap_estimates - didsc_point
            valid_roots = bootstrap_roots[np.isfinite(bootstrap_roots)]
            valid_count = int(valid_roots.size)
            bootstrap_complete = valid_count == bootstraps
            if bootstrap_complete:
                critical = float(
                    np.quantile(np.abs(valid_roots), 1.0 - ALPHA)
                )
                didsc_low = didsc_point - critical
                didsc_high = didsc_point + critical
                root_mean = float(np.mean(valid_roots))
                root_sd = float(np.std(valid_roots, ddof=1))
                didsc_status = "valid"
                didsc_error = ""
            else:
                critical = np.nan
                didsc_low = np.nan
                didsc_high = np.nan
                root_mean = (
                    float(np.mean(valid_roots))
                    if valid_count > 0
                    else np.nan
                )
                root_sd = (
                    float(np.std(valid_roots, ddof=1))
                    if valid_count > 1
                    else np.nan
                )
                didsc_status = "bootstrap_incomplete"
                didsc_error = (
                    f"{bootstraps - valid_count} bootstrap draw(s) failed"
                )
            outer_rows.append(
                _method_row(
                    common,
                    "didsc_multiplier",
                    didsc_point,
                    root_sd,
                    didsc_low,
                    didsc_high,
                    critical,
                    root_mean,
                    root_sd,
                    bootstraps,
                    valid_count,
                    np.nan,
                    np.nan,
                    outer.diagnostics,
                    didsc_status,
                    didsc_error,
                    runtime_per_row,
                )
            )

            influence = (
                outer.dr_if_base + amplitude * outer.dr_if_direction
            )
            influence_mean = float(np.mean(influence))
            influence_variance = float(np.mean(influence * influence))
            analytic_se = math.sqrt(influence_variance / n)
            drdid_point = float(
                outer.drdid_base[0]
                + amplitude * outer.drdid_direction[0]
            )
            drdid_low = drdid_point - Z_975 * analytic_se
            drdid_high = drdid_point + Z_975 * analytic_se
            drdid_valid = (
                np.isfinite(drdid_point)
                and np.isfinite(analytic_se)
                and analytic_se > 0.0
                and abs(influence_mean) <= 2.0e-10
            )
            outer_rows.append(
                _method_row(
                    common,
                    "drdid_analytic",
                    drdid_point,
                    analytic_se,
                    drdid_low if drdid_valid else np.nan,
                    drdid_high if drdid_valid else np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    0,
                    0,
                    influence_mean,
                    influence_variance,
                    outer.diagnostics,
                    "valid" if drdid_valid else "failed",
                    "" if drdid_valid else "invalid analytic influence interval",
                    runtime_per_row,
                )
            )
            invalid_bootstraps = np.flatnonzero(
                ~np.isfinite(bootstrap_estimates)
            )
            unique_failure_messages = sorted(
                {
                    failure_messages[int(index)]
                    for index in invalid_bootstraps
                    if int(index) in failure_messages
                }
            )
            bootstrap_rows.append(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "dgp_version": DGP_VERSION,
                    "n": n,
                    "outer_replication": replication,
                    "master_seed": master_seed,
                    "base_seed_hex": f"0x{base_seed:016x}",
                    "fold_hash": fold_hash,
                    "delta": delta,
                    "local_amplitude": amplitude,
                    "point_estimate": didsc_point,
                    "requested_bootstraps": bootstraps,
                    "valid_bootstraps": valid_count,
                    "bootstrap_failures": bootstraps - valid_count,
                    "bootstrap_root_mean": root_mean,
                    "bootstrap_root_sd": root_sd,
                    "symmetric_critical_value": critical,
                    "symmetric_ci_low": didsc_low,
                    "symmetric_ci_high": didsc_high,
                    "symmetric_ci_length": (
                        didsc_high - didsc_low
                        if bootstrap_complete
                        else np.nan
                    ),
                    "symmetric_covers_att": (
                        bool(didsc_low <= population.att_true <= didsc_high)
                        if bootstrap_complete
                        else ""
                    ),
                    "failure_messages": json.dumps(unique_failure_messages),
                }
            )
        return ReplicationOutput(outer_rows, bootstrap_rows)
    except Exception as error:
        return _failed_replication_output(
            population,
            n,
            replication,
            bootstraps,
            master_seed,
            base_seed,
            delta_grid,
            trim_boundaries,
            fold_hash,
            latent_hash,
            error,
            float(time.perf_counter() - started),
        )


def atomic_write_csv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    """Atomically replace a CSV with a complete, ordered snapshot."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: row.get(field, "")
                    for field in fieldnames
                }
            )
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON metadata file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Read a checkpoint CSV as dictionaries."""

    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in ("true", "1", "1.0"):
        return True
    if normalized in ("false", "0", "0.0"):
        return False
    return None


def _finite_column(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    values = np.asarray([_as_float(row.get(field)) for row in rows])
    return values[np.isfinite(values)]


def _indicator_column(rows: list[dict[str, Any]], field: str) -> np.ndarray:
    values = [_as_bool(row.get(field)) for row in rows]
    return np.asarray(
        [float(value) for value in values if value is not None],
        dtype=float,
    )


def summarize_outer_rows(
    rows: list[dict[str, Any]],
    n: int,
    delta_grid: tuple[float, ...],
    attempted_replications: int,
    elapsed_seconds: float,
    milestone_percent: int,
) -> list[dict[str, Any]]:
    """Create method-by-local-parameter Monte Carlo summaries."""

    summaries: list[dict[str, Any]] = []
    for delta in delta_grid:
        for method in METHODS:
            cell = [
                row
                for row in rows
                if math.isclose(
                    _as_float(row.get("delta")),
                    delta,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                and row.get("method") == method
            ]
            valid_points = [
                row
                for row in cell
                if np.isfinite(_as_float(row.get("point_estimate")))
            ]
            valid_intervals = [
                row
                for row in cell
                if row.get("status") == "valid"
                and np.isfinite(_as_float(row.get("ci_low")))
                and np.isfinite(_as_float(row.get("ci_high")))
            ]
            estimates = _finite_column(valid_points, "point_estimate")
            att_errors = _finite_column(valid_points, "bias_vs_att")
            target_errors = _finite_column(valid_points, "error_vs_method_target")
            standard_errors = _finite_column(valid_intervals, "estimated_se")
            lengths = _finite_column(valid_intervals, "ci_length")
            covers_att = _indicator_column(valid_intervals, "covers_att")
            covers_common = _indicator_column(
                valid_intervals,
                "covers_common_sc_target",
            )
            covers_target = _indicator_column(
                valid_intervals,
                "covers_method_target",
            )
            empirical_sd = (
                float(np.std(estimates, ddof=1))
                if estimates.size > 1
                else np.nan
            )
            coverage_att = (
                float(np.mean(covers_att)) if covers_att.size else np.nan
            )
            first = cell[0] if cell else {}
            requested_total = int(
                sum(
                    _as_float(row.get("requested_bootstraps"))
                    for row in cell
                    if np.isfinite(
                        _as_float(row.get("requested_bootstraps"))
                    )
                )
            )
            valid_total = int(
                sum(
                    _as_float(row.get("valid_bootstraps"))
                    for row in cell
                    if np.isfinite(_as_float(row.get("valid_bootstraps")))
                )
            )
            summaries.append(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "dgp_version": DGP_VERSION,
                    "method": method,
                    "interval_method": first.get("interval_method", ""),
                    "method_effective_bandwidth": _as_float(
                        first.get("method_effective_bandwidth")
                    ),
                    "n": n,
                    "delta": delta,
                    "local_scale": _as_float(first.get("local_scale")),
                    "local_amplitude": _as_float(
                        first.get("local_amplitude")
                    ),
                    "standardized_full_support_pt_violation": delta,
                    "kappa_m": KAPPA_M,
                    "sc_holds_by_construction": True,
                    "pt_holds_by_construction": bool(delta == 0.0),
                    "attempted_outer_replications": attempted_replications,
                    "valid_point_estimates": int(estimates.size),
                    "valid_intervals": len(valid_intervals),
                    "failed_intervals": (
                        attempted_replications - len(valid_intervals)
                    ),
                    "att_true": _as_float(first.get("att_true")),
                    "common_sc_population_target": _as_float(
                        first.get("common_sc_population_target")
                    ),
                    "method_population_target": _as_float(
                        first.get("method_population_target")
                    ),
                    "mean_point_estimate": (
                        float(np.mean(estimates))
                        if estimates.size
                        else np.nan
                    ),
                    "bias_vs_att": (
                        float(np.mean(att_errors))
                        if att_errors.size
                        else np.nan
                    ),
                    "bias_vs_method_target": (
                        float(np.mean(target_errors))
                        if target_errors.size
                        else np.nan
                    ),
                    "empirical_sd": empirical_sd,
                    "rmse_vs_att": (
                        float(np.sqrt(np.mean(att_errors**2)))
                        if att_errors.size
                        else np.nan
                    ),
                    "rmse_vs_method_target": (
                        float(np.sqrt(np.mean(target_errors**2)))
                        if target_errors.size
                        else np.nan
                    ),
                    "mean_estimated_se": (
                        float(np.mean(standard_errors))
                        if standard_errors.size
                        else np.nan
                    ),
                    "median_estimated_se": (
                        float(np.median(standard_errors))
                        if standard_errors.size
                        else np.nan
                    ),
                    "mean_se_over_empirical_sd": (
                        float(np.mean(standard_errors) / empirical_sd)
                        if standard_errors.size
                        and np.isfinite(empirical_sd)
                        and empirical_sd > 0.0
                        else np.nan
                    ),
                    "ci_coverage_att": coverage_att,
                    "ci_coverage_att_mcse": (
                        math.sqrt(
                            coverage_att
                            * (1.0 - coverage_att)
                            / covers_att.size
                        )
                        if covers_att.size
                        else np.nan
                    ),
                    "ci_coverage_common_sc_target": (
                        float(np.mean(covers_common))
                        if covers_common.size
                        else np.nan
                    ),
                    "ci_coverage_method_target": (
                        float(np.mean(covers_target))
                        if covers_target.size
                        else np.nan
                    ),
                    "mean_ci_length": (
                        float(np.mean(lengths)) if lengths.size else np.nan
                    ),
                    "median_ci_length": (
                        float(np.median(lengths)) if lengths.size else np.nan
                    ),
                    "requested_bootstraps_total": requested_total,
                    "valid_bootstraps_total": valid_total,
                    "bootstrap_failures_total": requested_total - valid_total,
                    "milestone_percent": milestone_percent,
                    "elapsed_wall_seconds": elapsed_seconds,
                }
            )
    return summaries


def population_rows(
    n: int,
    delta_grid: tuple[float, ...],
    trim_boundaries: bool,
) -> list[dict[str, Any]]:
    """Return an auditable row for every n, delta, and support point."""

    population = build_population()
    rows: list[dict[str, Any]] = []
    direction_string = "|".join(
        f"{value:.17g}" for value in population.donor_post_direction
    )
    for delta in delta_grid:
        diagnostics = population_configuration_diagnostics(
            population,
            n,
            delta,
        )
        common_target, dr_target, _ = method_population_targets(
            population,
            n,
            delta,
            trim_boundaries,
        )
        pt_gap = np.asarray(diagnostics["pt_gap"], dtype=float)
        for x_index, x_value in enumerate(population.x_support):
            singular = population.singular_values[x_index]
            rows.append(
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "dgp_version": DGP_VERSION,
                    "n": n,
                    "delta": delta,
                    "x_index": x_index,
                    "x": float(x_value),
                    "kappa_m": float(population.condition_numbers[x_index]),
                    "kappa_mtm_derived": float(
                        population.condition_numbers[x_index] ** 2
                    ),
                    "sigma_max_m": float(singular[0]),
                    "sigma_middle_m": float(singular[1]),
                    "sigma_min_m": float(singular[-1]),
                    "rank_m": int(np.sum(singular > 1.0e-10)),
                    "sc_residual": 0.0,
                    "binary_pt_gap": float(pt_gap[x_index]),
                    "full_population_binary_pt_gap": float(
                        diagnostics["average_gap"]
                    ),
                    "local_scale": population.oracle_dr_rootn_sd,
                    "local_amplitude": float(diagnostics["amplitude"]),
                    "root_n_local_amplitude": (
                        math.sqrt(n) * float(diagnostics["amplitude"])
                    ),
                    "treated_probability": float(
                        population.group_probabilities[x_index, 0]
                    ),
                    "true_weight_minimum": float(
                        np.min(population.true_weights[x_index])
                    ),
                    "true_weight_maximum": float(
                        np.max(population.true_weights[x_index])
                    ),
                    "att_true": population.att_true,
                    "drdid_population_target": dr_target,
                    "didsc_population_target": common_target,
                    "donor_post_direction": direction_string,
                }
            )
    return rows


def available_cpus() -> int:
    """Return the number of CPUs visible to the current process."""

    limits: list[int] = []
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            limits.append(max(1, int(slurm_cpus)))
        except ValueError:
            pass
    if hasattr(os, "sched_getaffinity"):
        limits.append(max(1, len(os.sched_getaffinity(0))))
    limits.append(max(1, os.cpu_count() or 1))
    return min(limits)


def source_sha256() -> str:
    """Return the exact runner-source fingerprint stored with every run."""

    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _prefix(args: argparse.Namespace, n: int) -> str:
    start_suffix = (
        f"_start{args.outer_start}" if args.outer_start != 0 else ""
    )
    return (
        f"q4_didsc_drdid_local_pt_{args.tag}_n{n}_"
        f"R{args.outer_reps}_B{args.bootstraps}{start_suffix}"
    )


def _sort_outer(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(_as_float(row.get("outer_replication"))),
            _as_float(row.get("delta")),
            str(row.get("method")),
        ),
    )


def _sort_bootstrap(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(_as_float(row.get("outer_replication"))),
            _as_float(row.get("delta")),
        ),
    )


def completed_replications_from_checkpoint(
    rows: list[dict[str, Any]],
    delta_grid: tuple[float, ...],
    outer_start: int,
    outer_reps: int,
) -> set[int]:
    """Validate complete replication batches restored from a checkpoint."""

    expected_rows = len(delta_grid) * len(METHODS)
    lower = outer_start
    upper = outer_start + outer_reps
    counts: dict[int, int] = {}
    keys: dict[int, set[tuple[float, str]]] = {}
    for row in rows:
        replication = int(_as_float(row.get("outer_replication")))
        if replication < lower or replication >= upper:
            raise ValueError("checkpoint contains an out-of-range replication")
        counts[replication] = counts.get(replication, 0) + 1
        keys.setdefault(replication, set()).add(
            (_as_float(row.get("delta")), str(row.get("method")))
        )
    completed: set[int] = set()
    expected_keys = {
        (delta, method) for delta in delta_grid for method in METHODS
    }
    for replication, count in counts.items():
        if count != expected_rows or keys[replication] != expected_keys:
            raise ValueError(
                f"checkpoint replication {replication} is incomplete"
            )
        completed.add(replication)
    return completed


def completed_bootstrap_replications_from_checkpoint(
    rows: list[dict[str, Any]],
    delta_grid: tuple[float, ...],
    outer_start: int,
    outer_reps: int,
) -> set[int]:
    """Validate and identify complete bootstrap-diagnostic replication batches."""

    lower = outer_start
    upper = outer_start + outer_reps
    counts: dict[int, int] = {}
    keys: dict[int, set[float]] = {}
    for row in rows:
        replication = int(_as_float(row.get("outer_replication")))
        if replication < lower or replication >= upper:
            raise ValueError(
                "bootstrap checkpoint contains an out-of-range replication"
            )
        counts[replication] = counts.get(replication, 0) + 1
        keys.setdefault(replication, set()).add(_as_float(row.get("delta")))
    expected_deltas = set(delta_grid)
    completed: set[int] = set()
    for replication, count in counts.items():
        if count != len(delta_grid) or keys[replication] != expected_deltas:
            raise ValueError(
                f"bootstrap checkpoint replication {replication} is incomplete"
            )
        completed.add(replication)
    return completed


def validate_resume_metadata(
    metadata: dict[str, Any],
    args: argparse.Namespace,
    n: int,
    delta_grid: tuple[float, ...],
) -> None:
    """Reject a checkpoint created under a different configuration."""

    expected = {
        "experiment_version": EXPERIMENT_VERSION,
        "n": n,
        "outer_start": args.outer_start,
        "outer_reps": args.outer_reps,
        "bootstraps": args.bootstraps,
        "master_seed": args.master_seed,
        "delta_grid": list(delta_grid),
        "trim_boundaries": args.trim_boundaries,
        "bootstrap_batch_size": args.bootstrap_batch_size,
        "tag": args.tag,
        "source_sha256": source_sha256(),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"resume metadata mismatch for {key}: "
                f"{metadata.get(key)!r} != {value!r}"
            )


def run_sample_size(
    args: argparse.Namespace,
    n: int,
    delta_grid: tuple[float, ...],
    audit_results: dict[str, float],
) -> list[dict[str, Any]]:
    """Run one sample size with atomic milestone checkpoints and resume."""

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _prefix(args, n)
    outer_path = output_dir / f"{prefix}_outer.csv"
    bootstrap_path = output_dir / f"{prefix}_bootstrap_diagnostics.csv"
    coverage_path = output_dir / f"{prefix}_coverage.csv"
    population_path = output_dir / f"{prefix}_population.csv"
    metadata_path = output_dir / f"{prefix}_metadata.json"
    outer_checkpoint = output_dir / f"{prefix}_outer_checkpoint.csv"
    bootstrap_checkpoint = (
        output_dir / f"{prefix}_bootstrap_diagnostics_checkpoint.csv"
    )
    coverage_checkpoint = output_dir / f"{prefix}_coverage_checkpoint.csv"
    all_paths = (
        outer_path,
        bootstrap_path,
        coverage_path,
        population_path,
        metadata_path,
        outer_checkpoint,
        bootstrap_checkpoint,
        coverage_checkpoint,
    )

    population_snapshot = population_rows(
        n,
        delta_grid,
        args.trim_boundaries,
    )
    resume_existing = args.resume and metadata_path.exists()
    if args.resume and not resume_existing:
        orphaned = next((path for path in all_paths if path.exists()), None)
        if orphaned is not None:
            raise FileNotFoundError(
                f"cannot resume {orphaned} without its metadata file"
            )
    if resume_existing:
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        validate_resume_metadata(metadata, args, n, delta_grid)
        outer_rows = (
            read_csv_rows(outer_checkpoint) if outer_checkpoint.exists() else []
        )
        bootstrap_rows = (
            read_csv_rows(bootstrap_checkpoint)
            if bootstrap_checkpoint.exists()
            else []
        )
        outer_completed = completed_replications_from_checkpoint(
            outer_rows,
            delta_grid,
            args.outer_start,
            args.outer_reps,
        )
        bootstrap_completed = completed_bootstrap_replications_from_checkpoint(
            bootstrap_rows,
            delta_grid,
            args.outer_start,
            args.outer_reps,
        )
        completed = outer_completed & bootstrap_completed
        if outer_completed != bootstrap_completed:
            print(
                f"n={n}: checkpoint generations differed; retaining "
                f"{len(completed)} replication(s) present in both files",
                flush=True,
            )
        outer_rows = [
            row
            for row in outer_rows
            if int(_as_float(row.get("outer_replication"))) in completed
        ]
        bootstrap_rows = [
            row
            for row in bootstrap_rows
            if int(_as_float(row.get("outer_replication"))) in completed
        ]
        metadata["status"] = "running"
        metadata["resume_count"] = int(metadata.get("resume_count", 0)) + 1
        metadata["last_nprocs"] = args.nprocs
        atomic_write_json(metadata_path, metadata)
        print(
            f"n={n}: resuming after {len(completed)}/{args.outer_reps} replications",
            flush=True,
        )
    else:
        if not args.overwrite:
            existing = next((path for path in all_paths if path.exists()), None)
            if existing is not None:
                raise FileExistsError(
                    f"refusing to overwrite {existing}; pass --overwrite or --resume"
                )
        else:
            for path in all_paths:
                if path.exists():
                    path.unlink()
        outer_rows = []
        bootstrap_rows = []
        completed = set()
        metadata = {
            "experiment_version": EXPERIMENT_VERSION,
            "dgp_version": DGP_VERSION,
            "experiment_purpose": EXPERIMENT_PURPOSE,
            "paper_reference": PAPER_REFERENCE,
            "n": n,
            "outer_start": args.outer_start,
            "outer_reps": args.outer_reps,
            "bootstraps": args.bootstraps,
            "bootstrap_batch_size": args.bootstrap_batch_size,
            "master_seed": args.master_seed,
            "delta_grid": list(delta_grid),
            "trim_boundaries": args.trim_boundaries,
            "tag": args.tag,
            "nprocs": args.nprocs,
            "last_nprocs": args.nprocs,
            "source_sha256": source_sha256(),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "argv": [sys.executable, *sys.argv],
            "multiprocessing_context": "spawn",
            "folds": N_FOLDS,
            "alpha": ALPHA,
            "kernel": KERNEL_NAME,
            "didsc_bandwidth_coefficient": DIDSC_BANDWIDTH_COEFFICIENT,
            "didsc_bandwidth_exponent": DIDSC_BANDWIDTH_EXPONENT,
            "didsc_effective_bandwidth": (
                DIDSC_BANDWIDTH_COEFFICIENT * n**DIDSC_BANDWIDTH_EXPONENT
            ),
            "drdid_bandwidth_coefficient": DRDID_BANDWIDTH_COEFFICIENT,
            "drdid_bandwidth_exponent": DRDID_BANDWIDTH_EXPONENT,
            "drdid_effective_bandwidth": (
                DRDID_BANDWIDTH_COEFFICIENT * n**DRDID_BANDWIDTH_EXPONENT
            ),
            "nuisance_estimation": NUISANCE_DESCRIPTION,
            "didsc_score_normalization": DIDSC_SCORE_NORMALIZATION,
            "drdid_panel_estimator": (
                "cross-fitted semiparametric implementation of equations "
                "2.6/2.7/3.1; MSE-rate local-linear nuisances"
            ),
            "drdid_analytic_variance": (
                "mean(feasible influence^2)/n, equation 2.11"
            ),
            "didsc_inference": "symmetric full nested Exp(1) multiplier bootstrap",
            "sc_weight_estimator": WEIGHT_ESTIMATOR,
            "local_scale": build_population().oracle_dr_rootn_sd,
            "direction_normalizer": build_population().direction_normalizer,
            "audit_results": audit_results,
            "status": "running",
            "resume_count": 0,
        }
        atomic_write_json(metadata_path, metadata)
        atomic_write_csv(
            population_path,
            POPULATION_FIELDS,
            population_snapshot,
        )
        atomic_write_csv(outer_checkpoint, OUTER_FIELDS, [])
        atomic_write_csv(bootstrap_checkpoint, BOOTSTRAP_FIELDS, [])
        atomic_write_csv(coverage_checkpoint, COVERAGE_FIELDS, [])

    if not population_path.exists():
        atomic_write_csv(
            population_path,
            POPULATION_FIELDS,
            population_snapshot,
        )

    replications = list(
        range(args.outer_start, args.outer_start + args.outer_reps)
    )
    segment_start_completed = len(completed)
    previous_elapsed = float(metadata.get("elapsed_wall_seconds", 0.0))
    remaining = [rep for rep in replications if rep not in completed]
    tasks = [
        (
            replication,
            n,
            args.bootstraps,
            args.bootstrap_batch_size,
            args.master_seed,
            delta_grid,
            args.trim_boundaries,
        )
        for replication in remaining
    ]
    started = time.perf_counter()
    milestone_counts = {
        max(1, math.ceil(args.outer_reps * percent / 100.0)): percent
        for percent in range(10, 101, 10)
    }

    def save_checkpoint(percent: int) -> None:
        elapsed = previous_elapsed + float(time.perf_counter() - started)
        ordered_outer = _sort_outer(outer_rows)
        ordered_bootstrap = _sort_bootstrap(bootstrap_rows)
        summaries = summarize_outer_rows(
            ordered_outer,
            n,
            delta_grid,
            len(completed),
            elapsed,
            percent,
        )
        atomic_write_csv(outer_checkpoint, OUTER_FIELDS, ordered_outer)
        atomic_write_csv(
            bootstrap_checkpoint,
            BOOTSTRAP_FIELDS,
            ordered_bootstrap,
        )
        atomic_write_csv(
            coverage_checkpoint,
            COVERAGE_FIELDS,
            summaries,
        )
        print(
            f"n={n}: {percent}% milestone ({len(completed)}/{args.outer_reps}) written to CSV",
            flush=True,
        )

    worker_count = min(args.nprocs, len(remaining)) if remaining else 0
    metadata.setdefault("effective_workers", worker_count)
    metadata["last_effective_workers"] = worker_count
    if worker_count == 0:
        iterator = iter(())
        pool = None
    elif worker_count == 1:
        iterator = map(run_replication_task, tasks)
        pool = None
    else:
        context = get_context("spawn")
        pool = context.Pool(processes=worker_count)
        iterator = pool.imap_unordered(run_replication_task, tasks, chunksize=1)
    try:
        for result in iterator:
            outer_rows.extend(result.outer_rows)
            bootstrap_rows.extend(result.bootstrap_rows)
            if not result.outer_rows:
                raise AssertionError("worker returned no outer rows")
            replication = int(result.outer_rows[0]["outer_replication"])
            completed.add(replication)
            completed_count = len(completed)
            if completed_count in milestone_counts:
                save_checkpoint(milestone_counts[completed_count])
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        if completed:
            interrupted_percent = min(
                100,
                int(100 * len(completed) / args.outer_reps),
            )
            try:
                save_checkpoint(interrupted_percent)
            except Exception as checkpoint_error:
                print(
                    "WARNING: could not save an interruption checkpoint: "
                    f"{checkpoint_error}",
                    flush=True,
                )
        interrupted_segment_elapsed = float(time.perf_counter() - started)
        metadata["status"] = "interrupted"
        metadata["completed_replications"] = len(completed)
        metadata["elapsed_wall_seconds"] = (
            previous_elapsed + interrupted_segment_elapsed
        )
        metadata["last_segment_elapsed_wall_seconds"] = (
            interrupted_segment_elapsed
        )
        metadata["last_nprocs"] = args.nprocs
        interrupted_history = list(metadata.get("run_history", []))
        interrupted_history.append(
            {
                "resume": bool(resume_existing),
                "interrupted": True,
                "start_completed_replications": segment_start_completed,
                "end_completed_replications": len(completed),
                "requested_workers": args.nprocs,
                "effective_workers": worker_count,
                "elapsed_wall_seconds": interrupted_segment_elapsed,
            }
        )
        metadata["run_history"] = interrupted_history
        atomic_write_json(metadata_path, metadata)
        raise
    else:
        if pool is not None:
            pool.close()
            pool.join()

    segment_elapsed = float(time.perf_counter() - started)
    elapsed = previous_elapsed + segment_elapsed
    ordered_outer = _sort_outer(outer_rows)
    ordered_bootstrap = _sort_bootstrap(bootstrap_rows)
    summaries = summarize_outer_rows(
        ordered_outer,
        n,
        delta_grid,
        len(completed),
        elapsed,
        100,
    )
    atomic_write_csv(outer_path, OUTER_FIELDS, ordered_outer)
    atomic_write_csv(
        bootstrap_path,
        BOOTSTRAP_FIELDS,
        ordered_bootstrap,
    )
    atomic_write_csv(coverage_path, COVERAGE_FIELDS, summaries)
    atomic_write_csv(outer_checkpoint, OUTER_FIELDS, ordered_outer)
    atomic_write_csv(
        bootstrap_checkpoint,
        BOOTSTRAP_FIELDS,
        ordered_bootstrap,
    )
    atomic_write_csv(
        coverage_checkpoint,
        COVERAGE_FIELDS,
        summaries,
    )
    failed_replications = sorted(
        {
            int(_as_float(row.get("outer_replication")))
            for row in ordered_outer
            if str(row.get("status")) != "valid"
        }
    )
    metadata["status"] = (
        "complete" if not failed_replications else "complete_with_failures"
    )
    metadata["completed_replications"] = len(completed)
    metadata["failed_replications"] = failed_replications
    metadata["failed_replication_count"] = len(failed_replications)
    metadata["elapsed_wall_seconds"] = elapsed
    metadata["last_segment_elapsed_wall_seconds"] = segment_elapsed
    metadata["last_nprocs"] = args.nprocs
    run_history = list(metadata.get("run_history", []))
    run_history.append(
        {
            "resume": bool(resume_existing),
            "start_completed_replications": segment_start_completed,
            "end_completed_replications": len(completed),
            "requested_workers": args.nprocs,
            "effective_workers": worker_count,
            "elapsed_wall_seconds": segment_elapsed,
        }
    )
    metadata["run_history"] = run_history
    atomic_write_json(metadata_path, metadata)
    print(
        f"n={n}: segment complete in {segment_elapsed:.1f}s "
        f"({elapsed:.1f}s cumulative); outputs written under {output_dir}",
        flush=True,
    )
    if failed_replications:
        print(
            f"n={n}: WARNING: {len(failed_replications)} replication(s) had "
            "at least one failed method or interval; see status/error columns",
            flush=True,
        )
    return summaries


def parse_arguments() -> argparse.Namespace:
    """Parse and validate the HPC command line."""

    default_output = Path(__file__).resolve().parent / "generated"
    parser = argparse.ArgumentParser(
        description=(
            "Compare DiDSC and panel DRDiD under exact SC and root-n local PT failure."
        )
    )
    parser.add_argument(
        "--n",
        type=int,
        nargs="+",
        default=list(DEFAULT_SAMPLE_SIZES),
        help="Sample sizes.",
    )
    parser.add_argument(
        "--outer-reps",
        "--replications",
        dest="outer_reps",
        type=int,
        default=DEFAULT_OUTER_REPS,
        help="Number of outer Monte Carlo replications.",
    )
    parser.add_argument(
        "--outer-start",
        type=int,
        default=0,
        help="First outer replication index.",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=DEFAULT_BOOTSTRAPS,
        help="Nested multiplier draws for DiDSC.",
    )
    parser.add_argument(
        "--bootstrap-batch-size",
        type=int,
        default=DEFAULT_BOOTSTRAP_BATCH_SIZE,
        help="Multiplier draws processed together inside each worker.",
    )
    parser.add_argument(
        "--delta-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_DELTA_GRID),
        help="Signed standardized root-n PT-violation parameters.",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=DEFAULT_MASTER_SEED,
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=available_cpus(),
        help="Outer-replication worker processes.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_output),
    )
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG)
    trimming = parser.add_mutually_exclusive_group()
    trimming.add_argument(
        "--trim-boundaries",
        dest="trim_boundaries",
        action="store_true",
        default=DEFAULT_TRIM_BOUNDARIES,
        help=(
            "Use the bandwidth experiment's strict realized-boundary trimming "
            "for both methods (default)."
        ),
    )
    trimming.add_argument(
        "--no-trim-boundaries",
        dest="trim_boundaries",
        action="store_false",
        help="Disable trimming and evaluate every observed support cell.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if any(n <= 1 for n in args.n):
        parser.error("all sample sizes must exceed one")
    if len(set(args.n)) != len(args.n):
        parser.error("sample sizes must not be repeated")
    if args.outer_reps <= 0 or args.outer_start < 0:
        parser.error("outer replication range is invalid")
    if args.bootstraps < 2:
        parser.error("--bootstraps must be at least two")
    if args.bootstrap_batch_size <= 0:
        parser.error("--bootstrap-batch-size must be positive")
    if args.nprocs <= 0:
        parser.error("--nprocs must be positive")
    if args.master_seed < 0:
        parser.error("--master-seed must be nonnegative")
    if (
        not args.tag
        or args.tag in {".", ".."}
        or ".." in args.tag
        or any(
            not (character.isalnum() or character in {"_", "-", "."})
            for character in args.tag
        )
    ):
        parser.error(
            "--tag must be a safe filename slug using letters, numbers, _, -, or ."
        )
    if not args.delta_grid:
        parser.error("--delta-grid must not be empty")
    if not all(np.isfinite(value) for value in args.delta_grid):
        parser.error("all local parameters must be finite")
    if len(set(args.delta_grid)) != len(args.delta_grid):
        parser.error("local parameters must not be repeated")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    return args


def main() -> None:
    """Audit the implementation and run every requested sample size."""

    args = parse_arguments()
    delta_grid = tuple(float(value) for value in args.delta_grid)
    audit_results = run_internal_audits(delta_grid)
    population = build_population()
    print(
        "Population audit passed: exact SC for every local parameter; "
        f"kappa(M)={KAPPA_M:.14g}; oracle DR root-n SD="
        f"{population.oracle_dr_rootn_sd:.15g}.",
        flush=True,
    )
    print(
        "Shared estimation: two folds and Epanechnikov local linear; "
        "DiDSC h=2.5*n^(-2/7), DRDiD h=n^(-1/5). "
        "Inference: DiDSC symmetric multiplier bootstrap; DRDiD analytic Wald.",
        flush=True,
    )
    if args.audit_only:
        print(
            "Audit-only mode complete: "
            + json.dumps(audit_results, sort_keys=True),
            flush=True,
        )
        return

    output_dir = Path(args.output_dir).expanduser().resolve()
    n_slug = "_".join(str(n) for n in args.n)
    combined_name = (
        f"q4_didsc_drdid_local_pt_{args.tag}_n{n_slug}_"
        f"R{args.outer_reps}_B{args.bootstraps}"
        f"{'_start' + str(args.outer_start) if args.outer_start else ''}_"
        "coverage_all_n.csv"
    )
    combined_path = output_dir / combined_name
    if args.overwrite and combined_path.exists():
        combined_path.unlink()

    all_summaries: list[dict[str, Any]] = []
    for n in args.n:
        print(
            f"Starting n={n}, R={args.outer_reps}, B={args.bootstraps}, "
            f"workers={args.nprocs}, deltas={len(delta_grid)}.",
            flush=True,
        )
        all_summaries.extend(
            run_sample_size(
                args,
                int(n),
                delta_grid,
                audit_results,
            )
        )
    atomic_write_csv(
        combined_path,
        COVERAGE_FIELDS,
        all_summaries,
    )
    print(f"Combined coverage summary written to {combined_path}", flush=True)


if __name__ == "__main__":
    main()
