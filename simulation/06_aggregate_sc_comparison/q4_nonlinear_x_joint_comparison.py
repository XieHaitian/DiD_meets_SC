#!/usr/bin/env python3
"""Compare DiD-SC, conventional SC, and SDID as X nonlinearity grows.

The population at lambda=0 is exactly the calibrated Quantile-top-4 PT+SC
DGP with kappa(M)=13.070487184660164.  Positive lambda jointly increases the
nonlinear dependence on X of (i) the conditional synthetic-control weights
and (ii) the common conditional untreated trend.  Conditional PT and
conditional SC continue to hold exactly at every lambda.

The proposed DiD-SC estimator uses the individual observations and X with the
paper's fixed two-fold cross-fitting and bandwidth 2.5*n^(-2/7).
Conventional SC and synthetic DiD receive only the 5-by-6 matrix of state-level
outcome averages.  This experiment estimates point performance only; it does
not run a bootstrap.

Defaults simulate lambda in {0.0, 0.5, 1.0, 1.5}, n in {2000, 4000}, and 500
replications using all visible local CPUs. All four lambda values are generated
in one joint run with common random numbers.
"""

from __future__ import annotations

import os


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
import platform
import sys
import time
from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
BANDWIDTH_DIR = SCRIPT_DIR.parent / "bandwidth"
for _path in (SCRIPT_DIR, BANDWIDTH_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import q4_aggregate_sc_sdid_comparison as aggregate_engine
import q4_bandwidth_three_dgps_hpc as didsc_engine


SCRIPT_VERSION = "q4_joint_nonlinear_x_method_comparison_v1"
SOURCE_DGP_VERSION = (
    "quantile_top4_kappa13_bandwidth_three_exact_regimes_v1"
)
EXPERIMENT_PURPOSE = "joint_nonlinear_x_identification_method_comparison"
NONLINEAR_BASIS = "sin(2*pi*x)+0.5*sin(4*pi*x), treated-centered-standardized"
METHODS = ("didsc", "conventional_sc", "synthetic_did")
METHOD_LABELS = {
    "didsc": "Proposed DiD-SC",
    "conventional_sc": "Conventional SC",
    "synthetic_did": "Synthetic DiD",
}
STATE_NAMES = aggregate_engine.STATE_NAMES

DEFAULT_LAMBDAS = (0.5, 1.0, 1.5)
DEFAULT_SAMPLE_SIZES = (2000, 4000)
DEFAULT_REPLICATIONS = 500
DEFAULT_MASTER_SEED = didsc_engine.DEFAULT_MASTER_SEED
DEFAULT_BANDWIDTH_COEFFICIENT = 2.5
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "generated"
DEFAULT_TAG = "joint_lambda_R500"

KAPPA_M = didsc_engine.KAPPA_M
ATT_TRUE = didsc_engine.Q4_ATT_FINGERPRINT
ABS_TOL = 1.0e-8
REL_TOL = 1.0e-9

DONOR_SCORES = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=float)

RAW_FIELDS = [
    "script_version",
    "source_dgp_version",
    "experiment_purpose",
    "lambda",
    "n",
    "replication",
    "method",
    "method_label",
    "source",
    "master_seed",
    "base_seed_hex",
    "fold_hash",
    "bandwidth_coefficient",
    "effective_bandwidth",
    "att_true",
    "sample_treated_att",
    "estimate",
    "error",
    "squared_error",
    "aggregate_noise_level",
    "unit_intercept",
    "time_intercept",
    "unit_fit_rmse",
    "time_fit_rmse",
    "donor_active_count",
    "time_active_count",
    "donor_weight_maryland",
    "donor_weight_new_hampshire",
    "donor_weight_utah",
    "donor_weight_virginia",
    "time_weight_pre_1",
    "time_weight_pre_2",
    "time_weight_pre_3",
    "time_weight_pre_4",
    "time_weight_pre_5",
    "count_alaska",
    "count_maryland",
    "count_new_hampshire",
    "count_utah",
    "count_virginia",
]


@dataclass(frozen=True)
class NonlinearDesign:
    """One lambda configuration and its audited population quantities."""

    lambda_value: float
    design: didsc_engine.NewDesign
    nonlinear_basis: np.ndarray
    weight_scale: float
    trend_scale: float
    base_common_trend: np.ndarray
    common_trend: np.ndarray
    population_state_shares: np.ndarray
    population_aggregate_panel: np.ndarray
    population_sc_estimate: float
    population_sdid_estimate: float
    weight_nonlinearity: float
    trend_nonlinearity: float


_WORKER_DESIGNS: tuple[NonlinearDesign, ...] = ()
_WORKER_BANDWIDTH_COEFFICIENT = DEFAULT_BANDWIDTH_COEFFICIENT


def available_cpu_count() -> int:
    """Return CPUs visible to the local process."""

    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def parameter_slug(value: float) -> str:
    """Return a deterministic filesystem-safe number."""

    text = format(float(value), ".12g").replace("-", "m").replace(".", "p")
    return f"lambda_{text}"


def weighted_linear_residual_rms(
    values: np.ndarray,
    x_support: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Measure genuinely nonlinear variation after a weighted affine fit."""

    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    regressors = np.column_stack([np.ones(x_support.size), x_support])
    weighted_regressors = regressors * np.sqrt(weights)[:, None]
    residual_sum = 0.0
    for column in range(values.shape[1]):
        weighted_values = values[:, column] * np.sqrt(weights)
        coefficients = np.linalg.lstsq(
            weighted_regressors,
            weighted_values,
            rcond=None,
        )[0]
        residual = values[:, column] - regressors @ coefficients
        residual_sum += float(np.sum(weights * residual * residual))
    return math.sqrt(residual_sum)


def nonlinear_basis_and_scales(
    base: didsc_engine.NewDesign,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Return q(x), the two ex-ante scales, and the base common trend."""

    x_support = base.x_support
    treated_probability = base.group_probabilities[:, 0]
    treated_weights = treated_probability / treated_probability.sum()
    raw_basis = np.sin(2.0 * np.pi * x_support) + 0.5 * np.sin(
        4.0 * np.pi * x_support
    )
    basis_mean = float(treated_weights @ raw_basis)
    basis_sd = math.sqrt(
        float(treated_weights @ ((raw_basis - basis_mean) ** 2))
    )
    nonlinear_basis = (raw_basis - basis_mean) / basis_sd
    weight_scale = math.sqrt(float(treated_weights @ (x_support**2)))
    trend_scale = math.sqrt(
        float(base.residual_variances[-2] + base.residual_variances[-1])
    )
    base_common_trend = (
        base.donor_means[:, -1, -1] - base.donor_means[:, -2, -1]
    )
    if not math.isclose(
        float(treated_weights @ nonlinear_basis),
        0.0,
        abs_tol=1.0e-12,
    ):
        raise AssertionError("nonlinear basis is not treated-centered")
    if not math.isclose(
        float(treated_weights @ (nonlinear_basis**2)),
        1.0,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise AssertionError("nonlinear basis is not treated-standardized")
    return nonlinear_basis, weight_scale, trend_scale, base_common_trend


def build_nonlinear_design(lambda_value: float) -> NonlinearDesign:
    """Construct one exact conditional PT+SC nonlinear-X population."""

    lambda_value = float(lambda_value)
    if lambda_value < 0.0 or not np.isfinite(lambda_value):
        raise ValueError("lambda must be finite and nonnegative")
    base = didsc_engine._build_q4_regime("pt_sc_both")
    (
        nonlinear_basis,
        weight_scale,
        trend_scale,
        base_common_trend,
    ) = nonlinear_basis_and_scales(base)

    donor_indices = np.arange(1, didsc_engine.N_DONORS + 1, dtype=float)
    logits = (
        base.x_support[:, None]
        * (donor_indices[None, :] - didsc_engine.N_DONORS)
        + lambda_value
        * weight_scale
        * nonlinear_basis[:, None]
        * DONOR_SCORES[None, :]
    )
    logits -= logits.max(axis=1, keepdims=True)
    softmax_weights = np.exp(logits)
    softmax_weights /= softmax_weights.sum(axis=1, keepdims=True)
    true_weights = 0.05 + 0.8 * softmax_weights

    common_trend = (
        base_common_trend
        + lambda_value * trend_scale * nonlinear_basis
    )
    donor_means = base.donor_means.copy()
    donor_means[:, -1] = donor_means[:, -2] + common_trend[:, None]
    treated_means = np.einsum(
        "xd,xtd->xt",
        true_weights,
        donor_means,
    )
    diagnostics = didsc_engine.affine_population_diagnostics(
        donor_means,
        treated_means,
        true_weights,
    )
    singular_values = diagnostics["singular_values"]
    condition_numbers = diagnostics["condition_numbers"]
    donor_changes = donor_means[:, -1] - donor_means[:, -2]
    treated_change = treated_means[:, -1] - treated_means[:, -2]
    changes = np.column_stack([treated_change, donor_changes])
    joint_probability = base.group_probabilities / didsc_engine.N_X
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
    true_residual = treated_means - np.einsum(
        "xd,xtd->xt",
        true_weights,
        donor_means,
    )
    config_id = parameter_slug(lambda_value)
    design = replace(
        base,
        kappa_m=KAPPA_M,
        donor_means=donor_means,
        treated_untreated_means=treated_means,
        true_weights=true_weights,
        population_kappa_m_median=float(np.median(condition_numbers)),
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
            np.median(singular_values[:, -1])
        ),
        population_rank_min=int(np.min(diagnostics["ranks"])),
        maximum_sc_residual=float(np.max(np.abs(true_residual))),
        maximum_pt_gap=float(np.max(diagnostics["pt_gap"])),
        dgp_name="pt_sc_joint_nonlinear_x",
        config_id=config_id,
        parameter_mode="joint_nonlinear_x_amplitude",
        tuning_value=lambda_value,
        rho=lambda_value,
        eta=lambda_value,
        sigma_delta_ref=sigma_delta_ref,
        pre_sc_gap_multiplier=0.0,
        pt_holds_by_construction=True,
        sc_holds_by_construction=True,
        population_estimand=base.att_true,
        population_estimand_minus_att=0.0,
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

    observed_mean_table = np.empty(
        (
            didsc_engine.N_X,
            didsc_engine.N_GROUPS,
            didsc_engine.N_PERIODS,
        ),
        dtype=float,
    )
    observed_mean_table[:, 0] = treated_means
    observed_mean_table[:, 0, -1] += base.treatment_effect
    for donor in range(didsc_engine.N_DONORS):
        observed_mean_table[:, donor + 1] = donor_means[:, :, donor]
    state_shares, aggregate_panel = aggregate_engine.population_aggregate_panel(
        base.group_probabilities,
        observed_mean_table,
    )
    population_sc = aggregate_engine.estimate_conventional_sc(
        aggregate_panel
    )
    population_sdid = aggregate_engine.estimate_synthetic_did(
        aggregate_panel
    )
    treated_x_weights = (
        base.group_probabilities[:, 0]
        / base.group_probabilities[:, 0].sum()
    )
    weight_nonlinearity = weighted_linear_residual_rms(
        true_weights,
        base.x_support,
        treated_x_weights,
    )
    trend_nonlinearity = weighted_linear_residual_rms(
        common_trend,
        base.x_support,
        treated_x_weights,
    )
    result = NonlinearDesign(
        lambda_value=lambda_value,
        design=design,
        nonlinear_basis=nonlinear_basis,
        weight_scale=weight_scale,
        trend_scale=trend_scale,
        base_common_trend=base_common_trend,
        common_trend=common_trend,
        population_state_shares=state_shares,
        population_aggregate_panel=aggregate_panel,
        population_sc_estimate=population_sc.estimate,
        population_sdid_estimate=population_sdid.estimate,
        weight_nonlinearity=weight_nonlinearity,
        trend_nonlinearity=trend_nonlinearity,
    )
    validate_nonlinear_design(result, base)
    return result


def validate_nonlinear_design(
    configuration: NonlinearDesign,
    base: didsc_engine.NewDesign,
) -> None:
    """Enforce every population invariant before simulation."""

    design = configuration.design
    diagnostics = didsc_engine.affine_population_diagnostics(
        design.donor_means,
        design.treated_untreated_means,
        design.true_weights,
    )
    if not np.allclose(
        design.donor_means[:, : didsc_engine.N_PRE],
        base.donor_means[:, : didsc_engine.N_PRE],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("donor pre-period calibration changed")
    if not np.allclose(
        diagnostics["condition_numbers"],
        KAPPA_M,
        rtol=REL_TOL,
        atol=ABS_TOL,
    ):
        raise AssertionError("condition-number fingerprint changed")
    if not np.all(diagnostics["ranks"] == didsc_engine.N_DONORS - 1):
        raise AssertionError("a donor pre-period matrix lost rank")
    if float(np.max(diagnostics["pt_gap"])) > 1.0e-10:
        raise AssertionError("conditional PT does not hold exactly")
    if (
        float(np.max(diagnostics["pre_residual_l2"])) > 1.0e-10
        or float(np.max(diagnostics["full_residual_max"])) > 1.0e-10
    ):
        raise AssertionError("conditional SC does not hold exactly")
    if float(np.max(diagnostics["pseudo_weight_error"])) > 1.0e-9:
        raise AssertionError("affine and designated weights disagree")
    if not np.allclose(
        design.true_weights.sum(axis=1),
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("conditional weights do not sum to one")
    if float(np.min(design.true_weights)) < 0.05 - 1.0e-12:
        raise AssertionError("conditional weight floor failed")
    if not math.isclose(
        design.att_true,
        ATT_TRUE,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL,
    ):
        raise AssertionError("ATT fingerprint changed")
    if not np.allclose(
        design.group_probabilities,
        base.group_probabilities,
        rtol=0.0,
        atol=0.0,
    ):
        raise AssertionError("state assignment changed")
    if configuration.lambda_value == 0.0:
        if not np.allclose(
            design.donor_means,
            base.donor_means,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise AssertionError("lambda=0 donor population is not the base")
        if not np.allclose(
            design.treated_untreated_means,
            base.treated_untreated_means,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise AssertionError("lambda=0 treated population is not the base")
        if not np.allclose(
            design.true_weights,
            base.true_weights,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise AssertionError("lambda=0 weights are not the base")


def population_rows(
    configurations: tuple[NonlinearDesign, ...],
) -> list[dict[str, float | int | str]]:
    """Return the population audit, including aggregate pseudo-targets."""

    rows: list[dict[str, float | int | str]] = []
    for configuration in configurations:
        design = configuration.design
        diagnostics = didsc_engine.affine_population_diagnostics(
            design.donor_means,
            design.treated_untreated_means,
            design.true_weights,
        )
        joint_probability = design.group_probabilities / didsc_engine.N_X
        q_state_means = np.asarray(
            [
                float(
                    np.sum(
                        joint_probability[:, group]
                        * configuration.nonlinear_basis
                    )
                    / configuration.population_state_shares[group]
                )
                for group in range(didsc_engine.N_GROUPS)
            ],
            dtype=float,
        )
        row: dict[str, float | int | str] = {
            "script_version": SCRIPT_VERSION,
            "source_dgp_version": SOURCE_DGP_VERSION,
            "experiment_purpose": EXPERIMENT_PURPOSE,
            "nonlinear_basis": NONLINEAR_BASIS,
            "lambda": configuration.lambda_value,
            "weight_scale": configuration.weight_scale,
            "trend_scale": configuration.trend_scale,
            "att_true": design.att_true,
            "kappa_m_min": float(np.min(diagnostics["condition_numbers"])),
            "kappa_m_median": float(
                np.median(diagnostics["condition_numbers"])
            ),
            "kappa_m_max": float(np.max(diagnostics["condition_numbers"])),
            "maximum_conditional_pt_gap": float(
                np.max(diagnostics["pt_gap"])
            ),
            "maximum_conditional_sc_residual": float(
                np.max(diagnostics["full_residual_max"])
            ),
            "minimum_true_weight": float(np.min(design.true_weights)),
            "maximum_true_weight": float(np.max(design.true_weights)),
            "weight_nonlinearity_rms": configuration.weight_nonlinearity,
            "common_trend_min": float(np.min(configuration.common_trend)),
            "common_trend_max": float(np.max(configuration.common_trend)),
            "trend_nonlinearity_rms": configuration.trend_nonlinearity,
            "population_sc_estimate": configuration.population_sc_estimate,
            "population_sc_bias": (
                configuration.population_sc_estimate - design.att_true
            ),
            "population_sdid_estimate": configuration.population_sdid_estimate,
            "population_sdid_bias": (
                configuration.population_sdid_estimate - design.att_true
            ),
        }
        for group, state in enumerate(STATE_NAMES):
            state_key = state.lower().replace(" ", "_")
            row[f"population_share_{state_key}"] = float(
                configuration.population_state_shares[group]
            )
            row[f"mean_q_given_{state_key}"] = float(q_state_means[group])
        rows.append(row)
    return rows


def initialize_worker(
    lambda_values: tuple[float, ...],
    bandwidth_coefficient: float,
) -> None:
    """Build all requested lambda populations once in every worker."""

    global _WORKER_DESIGNS, _WORKER_BANDWIDTH_COEFFICIENT
    _WORKER_DESIGNS = tuple(
        build_nonlinear_design(value) for value in lambda_values
    )
    _WORKER_BANDWIDTH_COEFFICIENT = float(bandwidth_coefficient)


def blank_raw_row() -> dict[str, float | int | str]:
    """Return a complete normalized raw-row schema."""

    return {field: np.nan for field in RAW_FIELDS}


def aggregate_method_row_values(
    result: aggregate_engine.EstimateResult,
) -> dict[str, float | int]:
    """Return normalized diagnostics from one aggregate estimator."""

    values: dict[str, float | int] = {
        "aggregate_noise_level": result.noise_level,
        "unit_intercept": result.unit_intercept,
        "time_intercept": result.time_intercept,
        "unit_fit_rmse": result.unit_fit_rmse,
        "time_fit_rmse": result.time_fit_rmse,
        "donor_active_count": result.donor_active_count,
        "time_active_count": result.time_active_count,
    }
    for donor, state in enumerate(STATE_NAMES[1:]):
        state_key = state.lower().replace(" ", "_")
        values[f"donor_weight_{state_key}"] = float(
            result.donor_weights[donor]
        )
    for period in range(didsc_engine.N_PRE):
        values[f"time_weight_pre_{period + 1}"] = float(
            result.time_weights[period]
        )
    return values


def run_replication_task(
    task: tuple[int, int, int],
) -> list[dict[str, float | int | str]]:
    """Generate one common sample and evaluate every requested lambda."""

    n, replication, master_seed = task
    if not _WORKER_DESIGNS:
        raise RuntimeError("worker populations were not initialized")
    base_seed = didsc_engine.replication_seed(master_seed, n, replication)
    reference_latent: didsc_engine.LatentSample | None = None
    sample_att = np.nan
    rows: list[dict[str, float | int | str]] = []
    for configuration in _WORKER_DESIGNS:
        design = configuration.design
        latent, outcomes, _ = didsc_engine.generate_outer_sample(
            design,
            n,
            base_seed,
        )
        if reference_latent is None:
            reference_latent = latent
            treated = latent.group == 0
            sample_att = float(
                np.mean(design.treatment_effect[latent.x_index[treated]])
            )
        else:
            if (
                latent.fold_hash != reference_latent.fold_hash
                or not np.array_equal(latent.x_index, reference_latent.x_index)
                or not np.array_equal(latent.group, reference_latent.group)
            ):
                raise AssertionError("common random numbers failed across lambda")
        eval_indices, trim_low, trim_high = didsc_engine.evaluation_geometry(
            design,
            latent,
        )
        bandwidth = (
            _WORKER_BANDWIDTH_COEFFICIENT * n ** (-2.0 / 7.0)
        )
        didsc_estimate = float(
            didsc_engine.vectorized_estimate_grid(
                latent=latent,
                outcomes=outcomes,
                observation_weights=np.ones(n),
                support=design.x_support,
                eval_indices=eval_indices,
                trim_low=trim_low,
                trim_high=trim_high,
                bandwidths=np.asarray([bandwidth]),
            )[0]
        )
        counts = np.bincount(
            latent.group,
            minlength=didsc_engine.N_GROUPS,
        )
        if np.any(counts <= 0):
            raise FloatingPointError("a state has no sampled individuals")
        panel = np.vstack(
            [
                outcomes[latent.group == group].mean(axis=0)
                for group in range(didsc_engine.N_GROUPS)
            ]
        )
        estimates: dict[str, float | aggregate_engine.EstimateResult] = {
            "didsc": didsc_estimate,
            "conventional_sc": aggregate_engine.estimate_conventional_sc(
                panel
            ),
            "synthetic_did": aggregate_engine.estimate_synthetic_did(panel),
        }
        for method in METHODS:
            result = estimates[method]
            estimate = (
                float(result)
                if method == "didsc"
                else float(result.estimate)
            )
            error = estimate - design.att_true
            row = blank_raw_row()
            row.update(
                {
                    "script_version": SCRIPT_VERSION,
                    "source_dgp_version": SOURCE_DGP_VERSION,
                    "experiment_purpose": EXPERIMENT_PURPOSE,
                    "lambda": configuration.lambda_value,
                    "n": n,
                    "replication": replication,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "source": "new_joint_nonlinear_simulation",
                    "master_seed": master_seed,
                    "base_seed_hex": f"0x{base_seed:016x}",
                    "fold_hash": latent.fold_hash,
                    "bandwidth_coefficient": (
                        _WORKER_BANDWIDTH_COEFFICIENT
                        if method == "didsc"
                        else np.nan
                    ),
                    "effective_bandwidth": (
                        bandwidth if method == "didsc" else np.nan
                    ),
                    "att_true": design.att_true,
                    "sample_treated_att": sample_att,
                    "estimate": estimate,
                    "error": error,
                    "squared_error": error * error,
                }
            )
            if method != "didsc":
                row.update(aggregate_method_row_values(result))
            for group, state in enumerate(STATE_NAMES):
                state_key = state.lower().replace(" ", "_")
                row[f"count_{state_key}"] = int(counts[group])
            rows.append(row)
    return rows


def summarize_rows(
    rows: list[dict[str, float | int | str]],
    population: list[dict[str, float | int | str]],
    sample_sizes: tuple[int, ...],
    lambda_values: tuple[float, ...],
    replications: int,
) -> list[dict[str, float | int | str]]:
    """Compute bias, SD, MSE, paired loss differences, and MSE ratios."""

    population_lookup = {
        float(row["lambda"]): row for row in population
    }
    row_lookup = {
        (
            float(row["lambda"]),
            int(row["n"]),
            int(row["replication"]),
            str(row["method"]),
        ): row
        for row in rows
    }
    summaries: list[dict[str, float | int | str]] = []
    for lambda_value in lambda_values:
        for n in sample_sizes:
            didsc_losses = np.asarray(
                [
                    float(
                        row_lookup[
                            (lambda_value, n, replication, "didsc")
                        ]["squared_error"]
                    )
                    for replication in range(replications)
                ],
                dtype=float,
            )
            didsc_mse = float(np.mean(didsc_losses))
            for method in METHODS:
                selected = [
                    row_lookup[(lambda_value, n, replication, method)]
                    for replication in range(replications)
                ]
                estimates = np.asarray(
                    [float(row["estimate"]) for row in selected],
                    dtype=float,
                )
                errors = estimates - ATT_TRUE
                losses = errors * errors
                bias = float(np.mean(errors))
                sd = float(np.std(estimates, ddof=1))
                mse = float(np.mean(losses))
                identity = bias * bias + (replications - 1) / replications * sd**2
                if not math.isclose(
                    mse,
                    identity,
                    rel_tol=1.0e-11,
                    abs_tol=1.0e-12,
                ):
                    raise AssertionError("MSE identity audit failed")
                paired_difference = losses - didsc_losses
                population_row = population_lookup[lambda_value]
                if method == "conventional_sc":
                    population_estimate = float(
                        population_row["population_sc_estimate"]
                    )
                elif method == "synthetic_did":
                    population_estimate = float(
                        population_row["population_sdid_estimate"]
                    )
                else:
                    population_estimate = ATT_TRUE
                summaries.append(
                    {
                        "script_version": SCRIPT_VERSION,
                        "source_dgp_version": SOURCE_DGP_VERSION,
                        "experiment_purpose": EXPERIMENT_PURPOSE,
                        "lambda": lambda_value,
                        "n": n,
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "replications": replications,
                        "att_true": ATT_TRUE,
                        "population_aggregate_estimate": population_estimate,
                        "population_aggregate_bias": (
                            population_estimate - ATT_TRUE
                        ),
                        "mean_estimate": float(np.mean(estimates)),
                        "bias": bias,
                        "sd": sd,
                        "mse": mse,
                        "rmse": math.sqrt(mse),
                        "bias_mcse": sd / math.sqrt(replications),
                        "mse_over_didsc": mse / didsc_mse,
                        "paired_mse_difference_vs_didsc": float(
                            np.mean(paired_difference)
                        ),
                        "paired_mse_difference_mcse": float(
                            np.std(paired_difference, ddof=1)
                            / math.sqrt(replications)
                        ),
                    }
                )
    return summaries


def write_csv_atomic(
    path: Path,
    rows: list[dict[str, float | int | str]],
    fieldnames: list[str] | None = None,
) -> None:
    """Write CSV data through a same-directory atomic replacement."""

    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    """Write JSON through a same-directory atomic replacement."""

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
        "--lambda-values",
        nargs="+",
        type=float,
        default=list(DEFAULT_LAMBDAS),
    )
    parser.add_argument(
        "--n",
        nargs="+",
        type=int,
        default=list(DEFAULT_SAMPLE_SIZES),
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=DEFAULT_REPLICATIONS,
    )
    parser.add_argument("--nprocs", type=int, default=0)
    parser.add_argument(
        "--master-seed",
        type=int,
        default=DEFAULT_MASTER_SEED,
    )
    parser.add_argument(
        "--bandwidth-coefficient",
        type=float,
        default=DEFAULT_BANDWIDTH_COEFFICIENT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--tag", default=DEFAULT_TAG)
    args = parser.parse_args(argv)
    args.lambda_values = tuple(dict.fromkeys(args.lambda_values))
    args.n = tuple(dict.fromkeys(args.n))
    if not args.lambda_values or any(value <= 0.0 for value in args.lambda_values):
        parser.error("positive lambda values must be strictly positive")
    if any(not np.isfinite(value) for value in args.lambda_values):
        parser.error("lambda values must be finite")
    if not args.n or any(n <= 0 for n in args.n):
        parser.error("sample sizes must be positive")
    if args.replications <= 1 or args.replications > 500:
        parser.error("replications must be between 2 and 500")
    if args.nprocs < 0:
        parser.error("nprocs cannot be negative")
    if args.bandwidth_coefficient <= 0.0:
        parser.error("bandwidth coefficient must be positive")
    if not args.tag or any(character.isspace() for character in args.tag):
        parser.error("tag must be nonempty and contain no spaces")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    """Run the manuscript lambda grid in one self-contained simulation."""

    args = parse_args(argv)
    started = time.perf_counter()
    population_configurations = tuple(
        build_nonlinear_design(value)
        for value in (0.0, *args.lambda_values)
    )
    population = population_rows(population_configurations)
    if not all(
        later.weight_nonlinearity >= earlier.weight_nonlinearity - 1.0e-12
        and later.trend_nonlinearity >= earlier.trend_nonlinearity - 1.0e-12
        for earlier, later in zip(
            population_configurations,
            population_configurations[1:],
        )
    ):
        raise AssertionError("stored nonlinearity measures are not monotone")

    tasks = [
        (n, replication, args.master_seed)
        for n in args.n
        for replication in range(args.replications)
    ]
    requested_workers = (
        available_cpu_count() if args.nprocs == 0 else args.nprocs
    )
    worker_count = max(1, min(requested_workers, len(tasks)))
    all_lambdas = (0.0, *args.lambda_values)
    estimate_count = len(tasks) * len(all_lambdas) * len(METHODS)
    print(
        f"Running {len(tasks)} outer samples ({estimate_count} estimates) "
        f"with {worker_count} worker process(es).",
        flush=True,
    )
    new_rows: list[dict[str, float | int | str]] = []
    next_milestone = 10
    context = get_context("spawn")
    chunk_size = max(1, len(tasks) // max(worker_count * 8, 1))
    with context.Pool(
        processes=worker_count,
        initializer=initialize_worker,
        initargs=(all_lambdas, args.bandwidth_coefficient),
    ) as pool:
        iterator = pool.imap_unordered(
            run_replication_task,
            tasks,
            chunksize=chunk_size,
        )
        for completed, task_rows in enumerate(iterator, start=1):
            new_rows.extend(task_rows)
            completion_percentage = 100.0 * completed / len(tasks)
            while next_milestone <= completion_percentage + 1.0e-12:
                elapsed = time.perf_counter() - started
                print(
                    f"{next_milestone}% complete "
                    f"({completed}/{len(tasks)} outer samples; "
                    f"{elapsed:.1f} seconds).",
                    flush=True,
                )
                next_milestone += 10

    expected_new = (
        len(args.n)
        * args.replications
        * len(all_lambdas)
        * len(METHODS)
    )
    if len(new_rows) != expected_new:
        raise AssertionError(
            f"received {len(new_rows)} new rows, expected {expected_new}"
        )
    combined_rows = new_rows
    method_order = {method: index for index, method in enumerate(METHODS)}
    combined_rows.sort(
        key=lambda row: (
            float(row["lambda"]),
            int(row["n"]),
            int(row["replication"]),
            method_order[str(row["method"])],
        )
    )
    summaries = summarize_rows(
        combined_rows,
        population,
        args.n,
        all_lambdas,
        args.replications,
    )

    output_dir = args.output_dir.resolve()
    prefix = f"q4_nonlinear_x_{args.tag}"
    outer_path = output_dir / f"{prefix}_outer.csv"
    summary_path = output_dir / f"{prefix}_summary.csv"
    population_path = output_dir / f"{prefix}_population.csv"
    metadata_path = output_dir / f"{prefix}_metadata.json"
    write_csv_atomic(outer_path, combined_rows, RAW_FIELDS)
    write_csv_atomic(summary_path, summaries)
    write_csv_atomic(population_path, population)
    runtime = time.perf_counter() - started
    metadata: dict[str, object] = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "source_dgp_version": SOURCE_DGP_VERSION,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "nonlinear_basis": NONLINEAR_BASIS,
        "joint_channel": True,
        "simulated_lambda_values": list(args.lambda_values),
        "combined_lambda_values": list(all_lambdas),
        "sample_sizes": list(args.n),
        "replications": args.replications,
        "master_seed": args.master_seed,
        "workers": worker_count,
        "runtime_seconds": runtime,
        "bandwidth_coefficient": args.bandwidth_coefficient,
        "bandwidth_formula": "h=2.5*n^(-2/7)",
        "true_att": ATT_TRUE,
        "kappa_m": KAPPA_M,
        "weight_scores": DONOR_SCORES.tolist(),
        "weight_scale": population_configurations[1].weight_scale,
        "trend_scale": population_configurations[1].trend_scale,
        "methods": list(METHODS),
        "lambda_zero_source": "generated in this joint run",
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "outputs": {
            "outer": str(outer_path),
            "summary": str(summary_path),
            "population": str(population_path),
            "metadata": str(metadata_path),
        },
    }
    write_json_atomic(metadata_path, metadata)

    print("\nBias, empirical SD, and MSE relative to the fixed ATT:")
    for row in summaries:
        print(
            f"lambda={float(row['lambda']):.2f}  "
            f"n={int(row['n']):5d}  "
            f"{str(row['method']):15s}  "
            f"bias={float(row['bias']): .6f}  "
            f"sd={float(row['sd']):.6f}  "
            f"mse={float(row['mse']):.6f}  "
            f"ratio={float(row['mse_over_didsc']):.2f}"
        )
    print(f"\nFinished in {runtime:.1f} seconds.")
    print(f"Summary: {summary_path}")
    print(f"Raw combined estimates: {outer_path}")
    print(f"Population audit: {population_path}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
