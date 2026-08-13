#!/usr/bin/env python3
"""Empirical DRDiD and distributional-SC benchmarks for Alaska.

The program implements two benchmarks for the nine education-by-children
subgroups and the quantile-top-four donor pool (VA, NH, MD, UT):

* the repeated-cross-section DRDiD score based on Sant'Anna and Zhao (2020),
  with the empirical central-support convention, two-fold cross-fitting,
  Epanechnikov local-linear nuisances, MSE-optimal bandwidths, and centered
  plug-in influence-function inference; and
* Gunsilius-style distributional synthetic control, with one simplex weight
  vector per pre-treatment year, the five vectors averaged for 2003, 1,000
  midpoint quantiles, and a full stratified household bootstrap that
  re-estimates all quantiles and weights in every draw.

The expensive distributional-SC bootstrap checkpoints each subgroup at every
10-percent milestone and can be resumed without changing its random draws.
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
    # Each subgroup is already a separate process. Force one numerical-library
    # thread per worker so user-level BLAS settings cannot oversubscribe it.
    os.environ[_thread_variable] = "1"

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from itertools import product
from multiprocessing import get_context
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

import numpy as np
import pandas as pd


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = TASK_DIR.parent / "Alaska_MW.csv"
DEFAULT_PROPOSED_RESULTS = (
    TASK_DIR.parent
    / "01_quantile_top4_didsc"
    / "quantile_top4_didsc_results.csv"
)

TREATED_STATE = 2
DONOR_CODES = (51, 33, 24, 49)
DONOR_NAMES = ("Virginia", "New Hampshire", "Maryland", "Utah")
GUNSILIUS_SELECTION_WEIGHTS = (0.11, 0.11, 0.09, 0.07)
ALL_STATES = (TREATED_STATE,) + DONOR_CODES
PRE_YEARS = (1998, 1999, 2000, 2001, 2002)
POST_YEAR = 2003
DR_PRE_YEAR = 2002
SUBGROUPS = tuple(product(range(3), range(3)))

DEFAULT_BOOTSTRAPS = 1000
DEFAULT_QUANTILES = 1000
DEFAULT_SEED = 2026081002
DEFAULT_FOLDS = 2
DEFAULT_AGE_TRIM = (0.05, 0.95)
DEFAULT_OUTCOME_BANDWIDTH = 6.25
DEFAULT_PROPENSITY_BANDWIDTH = 12.94
DEFAULT_BANDWIDTH_REFERENCE_N = 957.0
DEFAULT_PROPENSITY_CLIP = 0.01
ALPHA = 0.05
Z_975 = NormalDist().inv_cdf(1.0 - ALPHA / 2.0)
EXPERIMENT_VERSION = (
    "empirical_q4_drdid_rc_support_if_n15_dsc_full_bootstrap_v4"
)


@dataclass(frozen=True)
class RunConfig:
    """Pickle-safe configuration passed to subgroup workers."""

    data_path: str
    output_dir: str
    bootstraps: int
    quantiles: int
    seed: int
    age_trim_lower: float
    age_trim_upper: float
    outcome_bandwidth: float
    propensity_bandwidth: float
    bandwidth_reference_n: float
    propensity_clip: float
    folds: int
    requested_workers: int
    run_drdid: bool
    run_dsc: bool
    overwrite: bool


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    """Write JSON atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_data(path: Path) -> pd.DataFrame:
    """Load and validate the repeated-cross-section data."""

    data = pd.read_csv(path)
    required = {
        "state_fips",
        "year",
        "hhseq",
        "age",
        "educ",
        "contpov",
        "nchild",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Data are missing required columns: {missing}")
    if data[list(required)].isna().any().any():
        raise ValueError("Required empirical variables contain missing values")
    data = data.copy()
    if "Unnamed: 0" in data.columns:
        data["row_id"] = data["Unnamed: 0"].astype(int)
    else:
        data["row_id"] = np.arange(data.shape[0], dtype=int)
    duplicate = data.duplicated(["state_fips", "year", "hhseq"])
    if duplicate.any():
        raise ValueError("Household identifiers are not unique within state-year")
    return data


def subgroup_sample(
    data: pd.DataFrame,
    educ: int,
    nchild: int,
    trim_lower: float,
    trim_upper: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Return the raw top-four sample and its fixed central-age indicator."""

    raw = data[
        data["state_fips"].isin(ALL_STATES)
        & (data["educ"] == educ)
        & (data["nchild"] == nchild)
        & data["year"].isin(PRE_YEARS + (POST_YEAR,))
    ].copy()
    if raw.empty:
        raise ValueError(f"Empty subgroup educ={educ}, nchild={nchild}")
    age_low, age_high = raw["age"].quantile(
        [trim_lower, trim_upper]
    ).to_numpy(dtype=float)
    raw["central_age_support"] = (
        (raw["age"] > age_low) & (raw["age"] < age_high)
    )
    raw = raw.reset_index(drop=True)
    if not raw["central_age_support"].any():
        raise ValueError("Age trimming removed the entire subgroup")
    cells = raw.groupby(["state_fips", "year"], observed=True).size()
    support_cells = raw.loc[raw["central_age_support"]].groupby(
        ["state_fips", "year"],
        observed=True,
    ).size()
    expected_cells = len(ALL_STATES) * (len(PRE_YEARS) + 1)
    if (
        cells.shape[0] != expected_cells
        or support_cells.shape[0] != expected_cells
        or (cells <= 0).any()
        or (support_cells <= 0).any()
    ):
        raise ValueError(
            f"Incomplete state-year cells for educ={educ}, nchild={nchild}"
        )
    metadata: dict[str, float | int] = {
        "n_raw": int(raw.shape[0]),
        "n_central_age_support": int(raw["central_age_support"].sum()),
        "central_age_support_share": float(raw["central_age_support"].mean()),
        "age_trim_lower_quantile": float(trim_lower),
        "age_trim_upper_quantile": float(trim_upper),
        "age_trim_lower_value": float(age_low),
        "age_trim_upper_value": float(age_high),
        "minimum_state_year_n": int(cells.min()),
        "median_state_year_n": float(cells.median()),
        "maximum_state_year_n": int(cells.max()),
        "minimum_state_year_central_n": int(support_cells.min()),
        "median_state_year_central_n": float(support_cells.median()),
        "maximum_state_year_central_n": int(support_cells.max()),
    }
    return raw, metadata


def empirical_bandwidth_reference_sample_size(data: pd.DataFrame) -> float:
    """Return the median raw 2002--2003 size over the nine top-four cells."""

    selected = data[
        data["state_fips"].isin(ALL_STATES)
        & data["year"].isin((DR_PRE_YEAR, POST_YEAR))
    ]
    counts = selected.groupby(["educ", "nchild"], observed=True).size()
    if counts.shape[0] != len(SUBGROUPS) or (counts <= 0).any():
        raise ValueError("Cannot construct the nine-cell bandwidth reference n")
    return float(counts.median())


def assign_stratified_folds(
    treatment: np.ndarray,
    post: np.ndarray,
    folds: int,
    seed: int,
) -> np.ndarray:
    """Assign deterministic balanced folds within each D-by-T cell."""

    if folds < 2:
        raise ValueError("At least two cross-fitting folds are required")
    assignment = np.full(treatment.shape[0], -1, dtype=int)
    rng = np.random.default_rng(seed)
    for d_value in (0, 1):
        for t_value in (0, 1):
            indices = np.flatnonzero(
                (treatment == d_value) & (post == t_value)
            )
            if indices.size < 2 * folds:
                raise ValueError(
                    f"D={d_value}, T={t_value} is too small for {folds} folds"
                )
            indices = rng.permutation(indices)
            assignment[indices] = np.arange(indices.size) % folds
    if (assignment < 0).any():
        raise AssertionError("Some observations were not assigned to a fold")
    return assignment


def local_linear_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_evaluate: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Epanechnikov local-linear intercepts evaluated at observed ages."""

    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_evaluate = np.asarray(x_evaluate, dtype=float)
    if x_train.size == 0 or x_train.size != y_train.size:
        raise ValueError("Invalid local-linear training arrays")
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("Bandwidth must be positive")
    unique_x, inverse = np.unique(x_evaluate, return_inverse=True)
    fitted_unique = np.empty(unique_x.size, dtype=float)
    for start in range(0, unique_x.size, 128):
        stop = min(start + 128, unique_x.size)
        evaluation = unique_x[start:stop]
        centered = x_train[:, None] - evaluation[None, :]
        scaled = centered / bandwidth
        kernel = 0.75 * (1.0 - scaled * scaled)
        kernel[np.abs(scaled) > 1.0] = 0.0
        s0 = kernel.sum(axis=0)
        s1 = (kernel * centered).sum(axis=0)
        s2 = (kernel * centered * centered).sum(axis=0)
        t0 = (kernel * y_train[:, None]).sum(axis=0)
        t1 = (kernel * centered * y_train[:, None]).sum(axis=0)
        ridge = 1.0e-6 * np.maximum((s0 + s2) / 2.0, 1.0)
        a00 = s0 + ridge
        a11 = s2 + ridge
        determinant = a00 * a11 - s1 * s1
        determinant = np.maximum(determinant, ridge * ridge)
        fitted_unique[start:stop] = (
            a11 * t0 - s1 * t1
        ) / determinant
    fitted = fitted_unique[inverse]
    if not np.all(np.isfinite(fitted)):
        raise FloatingPointError("Nonfinite local-linear prediction")
    return fitted


def estimate_repeated_cross_section_drdid(
    subgroup: pd.DataFrame,
    educ: int,
    nchild: int,
    metadata: dict[str, float | int],
    config: RunConfig,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Estimate the support-weighted repeated-cross-section DRDiD score."""

    start = time.perf_counter()
    sample = subgroup[
        subgroup["year"].isin((DR_PRE_YEAR, POST_YEAR))
    ].copy().reset_index(drop=True)
    bandwidth_scale = (
        sample.shape[0] / config.bandwidth_reference_n
    ) ** (-1.0 / 5.0)
    outcome_bandwidth = config.outcome_bandwidth * bandwidth_scale
    propensity_bandwidth = config.propensity_bandwidth * bandwidth_scale
    y = sample["contpov"].to_numpy(dtype=float)
    x = sample["age"].to_numpy(dtype=float)
    central_support = sample["central_age_support"].to_numpy(dtype=bool)
    treatment = (sample["state_fips"].to_numpy() == TREATED_STATE).astype(int)
    post = (sample["year"].to_numpy() == POST_YEAR).astype(int)
    fold_seed = int(
        np.random.SeedSequence(
            [config.seed, 3101, educ, nchild]
        ).generate_state(1, dtype=np.uint32)[0]
    )
    fold = assign_stratified_folds(
        treatment,
        post,
        config.folds,
        fold_seed,
    )

    propensity_raw = np.full(sample.shape[0], np.nan)
    outcome = {
        (d_value, t_value): np.full(sample.shape[0], np.nan)
        for d_value in (0, 1)
        for t_value in (0, 1)
    }
    for test_fold in range(config.folds):
        test = fold == test_fold
        train = ~test
        propensity_raw[test] = local_linear_predict(
            x[train],
            treatment[train].astype(float),
            x[test],
            propensity_bandwidth,
        )
        for d_value in (0, 1):
            for t_value in (0, 1):
                cell = train & (treatment == d_value) & (post == t_value)
                outcome[(d_value, t_value)][test] = local_linear_predict(
                    x[cell],
                    y[cell],
                    x[test],
                    outcome_bandwidth,
                )
    if not np.all(np.isfinite(propensity_raw)):
        raise FloatingPointError("Nonfinite out-of-fold propensity estimate")
    for fitted in outcome.values():
        if not np.all(np.isfinite(fitted)):
            raise FloatingPointError("Nonfinite out-of-fold outcome estimate")

    propensity = np.clip(
        propensity_raw,
        config.propensity_clip,
        1.0 - config.propensity_clip,
    )
    odds = propensity / (1.0 - propensity)
    treated_share = float(np.mean(treatment))
    if treated_share <= 0.0 or treated_share >= 1.0:
        raise FloatingPointError("Invalid treated share")

    w_treated: dict[int, np.ndarray] = {}
    w_control: dict[int, np.ndarray] = {}
    treated_cell_denominator: dict[int, float] = {}
    control_cell_denominator: dict[int, float] = {}
    for t_value in (0, 1):
        treated_numerator = treatment * (post == t_value)
        control_numerator = odds * (1 - treatment) * (post == t_value)
        treated_cell_mass = float(np.mean(treated_numerator))
        control_weighted_cell_mass = float(np.mean(control_numerator))
        if treated_cell_mass <= 0.0 or control_weighted_cell_mass <= 0.0:
            raise FloatingPointError("Invalid normalized DRDiD weight")
        treated_cell_denominator[t_value] = treated_cell_mass
        control_cell_denominator[t_value] = control_weighted_cell_mass
        w_treated[t_value] = treated_numerator / treated_cell_mass
        w_control[t_value] = control_numerator / control_weighted_cell_mass

    m00 = outcome[(0, 0)]
    m01 = outcome[(0, 1)]
    m10 = outcome[(1, 0)]
    m11 = outcome[(1, 1)]
    treated_change = m11 - m10
    control_change = m01 - m00
    untrimmed_pseudo_outcome = (
        treatment / treated_share * (treated_change - control_change)
        + w_treated[1] * (y - m11)
        - w_treated[0] * (y - m10)
        - w_control[1] * (y - m01)
        + w_control[0] * (y - m00)
    )
    # Match the main DiD--SC empirical score convention: all nuisance fits and
    # normalizations use the raw sample, while observations outside the fixed
    # pooled central-age support make zero score contribution. For the target
    # theta_S = E[S D {Delta m_1(X)-Delta m_0(X)}] / E[D], the fixed S(X)
    # preserves conditional score orthogonality. Estimating the full-sample
    # treated share contributes -D/E[D] theta_S to the influence function.
    # The estimated treatment-by-time and reweighted-control cell denominators
    # have no additional first-order term because their residual numerators
    # have conditional mean zero.
    score_contribution = (
        central_support.astype(float) * untrimmed_pseudo_outcome
    )
    estimate = float(np.mean(score_contribution))
    untrimmed_estimate = float(np.mean(untrimmed_pseudo_outcome))
    influence = score_contribution - treatment / treated_share * estimate
    influence_mean = float(np.mean(influence))
    influence_variance = float(np.var(influence, ddof=1))
    standard_error = math.sqrt(influence_variance / sample.shape[0])
    ci_low = estimate - Z_975 * standard_error
    ci_high = estimate + Z_975 * standard_error
    elapsed = time.perf_counter() - start
    if abs(influence_mean) > 1.0e-8:
        raise AssertionError(
            f"Repeated-cross-section influence is not centered: {influence_mean}"
        )

    result = {
        "task": "02_drdid_distributional_sc",
        "method": "drdid_repeated_cross_section",
        "donor_set": "quantile_top4",
        "donors": "|".join(DONOR_NAMES),
        "donor_selection_weights": "|".join(
            f"{weight:.2f}" for weight in GUNSILIUS_SELECTION_WEIGHTS
        ),
        "educ": educ,
        "nchild": nchild,
        "pre_year": DR_PRE_YEAR,
        "post_year": POST_YEAR,
        "sample_definition": (
            "raw_2002_2003_nuisance_training_with_fixed_central_age_"
            "score_support"
        ),
        "n": int(sample.shape[0]),
        "n_central_age_support_2002_2003": int(central_support.sum()),
        "central_age_support_share_2002_2003": float(
            central_support.mean()
        ),
        "support_cutoffs_fixed_for_inference": True,
        **metadata,
        "folds": config.folds,
        "kernel": "epanechnikov_local_linear",
        "outcome_bandwidth": outcome_bandwidth,
        "propensity_bandwidth": propensity_bandwidth,
        "outcome_bandwidth_at_reference_n": config.outcome_bandwidth,
        "propensity_bandwidth_at_reference_n": config.propensity_bandwidth,
        "bandwidth_reference_n": config.bandwidth_reference_n,
        "bandwidth_scale_from_reference": bandwidth_scale,
        "bandwidth_rate": -0.2,
        "bandwidth_rule": f"h_cell=h_reference*(n_2002_2003/{config.bandwidth_reference_n:g})^(-1/5)",
        "bandwidth_interpretation": (
            "6.25 and 12.94 are realized data-driven MSE-optimal "
            f"bandwidths at the frozen median raw 2002-2003 subgroup reference n={config.bandwidth_reference_n:g}; "
            "cells are mechanically rescaled at n^(-1/5) "
            "and no DiD-SC undersmoothing is applied"
        ),
        "propensity_clip": config.propensity_clip,
        "propensity_raw_min": float(np.min(propensity_raw)),
        "propensity_raw_max": float(np.max(propensity_raw)),
        "propensity_min": float(np.min(propensity)),
        "propensity_max": float(np.max(propensity)),
        "propensity_clipped_share": float(
            np.mean(propensity != propensity_raw)
        ),
        "treated_share": treated_share,
        "central_age_support_share_2002_2003_all": float(
            central_support.mean()
        ),
        "central_age_support_treated_mass_2002_2003": float(
            np.mean(central_support * treatment)
        ),
        "central_age_support_share_among_treated_2002_2003": float(
            np.mean(central_support * treatment) / treated_share
        ),
        "pre_period_share": float(np.mean(post == 0)),
        "post_period_share": float(np.mean(post == 1)),
        "treated_pre_cell_denominator": treated_cell_denominator[0],
        "treated_post_cell_denominator": treated_cell_denominator[1],
        "control_pre_weighted_cell_denominator": control_cell_denominator[0],
        "control_post_weighted_cell_denominator": control_cell_denominator[1],
        "mean_treated_normalization_weight": float(
            np.mean(treatment / treated_share)
        ),
        "mean_treated_pre_weight": float(np.mean(w_treated[0])),
        "mean_treated_post_weight": float(np.mean(w_treated[1])),
        "mean_control_pre_weight": float(np.mean(w_control[0])),
        "mean_control_post_weight": float(np.mean(w_control[1])),
        "estimate": estimate,
        "untrimmed_diagnostic_estimate": untrimmed_estimate,
        "analytic_se": standard_error,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_length": ci_high - ci_low,
        "influence_mean": influence_mean,
        "influence_variance": influence_variance,
        "inference": (
            "centered_overlap_trimmed_plugin_analytic_influence_based_on_"
            "SantAnna_Zhao_equation_2.13"
        ),
        "point_seconds": elapsed,
        "bootstrap_seconds": 0.0,
        "total_seconds": elapsed,
        "workers": 1,
        "seed": config.seed,
        "fold_seed": fold_seed,
        "timing_status": "measured",
    }

    detail = pd.DataFrame(
        {
            "educ": educ,
            "nchild": nchild,
            "row_id": sample["row_id"].to_numpy(dtype=int),
            "state_fips": sample["state_fips"].to_numpy(dtype=int),
            "year": sample["year"].to_numpy(dtype=int),
            "fold": fold,
            "age": x,
            "outcome": y,
            "treated": treatment,
            "post": post,
            "central_age_support": central_support,
            "propensity_raw": propensity_raw,
            "propensity": propensity,
            "m00": m00,
            "m01": m01,
            "m10": m10,
            "m11": m11,
            "w_treated_pre": w_treated[0],
            "w_treated_post": w_treated[1],
            "w_control_pre": w_control[0],
            "w_control_post": w_control[1],
            "w_treated_full_denominator": treatment / treated_share,
            "support_weighted_treated_normalization": (
                central_support * treatment / treated_share
            ),
            "untrimmed_pseudo_outcome": untrimmed_pseudo_outcome,
            "score_contribution": score_contribution,
            "zeta_support_weighted": score_contribution,
            "influence_centering_term": (
                treatment / treated_share * estimate
            ),
            "influence": influence,
        }
    )
    timing = {
        "task": "02_drdid_distributional_sc",
        "method": "drdid_repeated_cross_section",
        "educ": educ,
        "nchild": nchild,
        "n": int(sample.shape[0]),
        "point_estimation_seconds": elapsed,
        "bootstrap_seconds": 0.0,
        "bootstrap_draws_requested": 0,
        "bootstrap_draws_valid": 0,
        "seconds_per_bootstrap_draw": np.nan,
        "cell_wall_seconds": elapsed,
        "worker_processes": 1,
        "timing_status": "measured",
    }
    return result, detail, timing


def empirical_quantile(
    sorted_values: np.ndarray,
    quantile_grid: np.ndarray,
    counts: np.ndarray | None = None,
) -> np.ndarray:
    """Generalized-inverse empirical quantile, optionally with counts."""

    values = np.asarray(sorted_values, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot form a quantile from an empty cell")
    if counts is None:
        thresholds = quantile_grid * values.size
        indices = np.ceil(thresholds).astype(int) - 1
    else:
        counts = np.asarray(counts, dtype=int)
        if counts.shape != values.shape or counts.sum() != values.size:
            raise ValueError("Bootstrap counts do not preserve cell size")
        cumulative = np.cumsum(counts)
        thresholds = quantile_grid * values.size
        indices = np.searchsorted(cumulative, thresholds, side="left")
    indices = np.clip(indices, 0, values.size - 1)
    return values[indices]


def simplex_least_squares(
    donor_quantiles: np.ndarray,
    treated_quantile: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Solve the four-donor simplex least-squares problem by active sets."""

    donor_quantiles = np.asarray(donor_quantiles, dtype=float)
    treated_quantile = np.asarray(treated_quantile, dtype=float)
    if donor_quantiles.ndim != 2:
        raise ValueError("Donor quantiles must be a two-dimensional array")
    if treated_quantile.ndim != 1:
        raise ValueError("Treated quantiles must be a one-dimensional array")
    if donor_quantiles.shape[0] != treated_quantile.size:
        raise ValueError("Treated and donor quantile grids do not match")
    if not np.all(np.isfinite(donor_quantiles)) or not np.all(
        np.isfinite(treated_quantile)
    ):
        raise FloatingPointError("Nonfinite input to simplex least squares")
    donor_count = donor_quantiles.shape[1]
    best_weight = np.full(donor_count, 1.0 / donor_count)
    best_objective = float("inf")
    for mask_value in range(1, 1 << donor_count):
        active = np.array(
            [(mask_value >> j) & 1 for j in range(donor_count)],
            dtype=bool,
        )
        # The simplex restriction makes ||Xw-y|| equal to
        # ||(X-y1')w||. Form the tiny Gram matrix explicitly to avoid sending
        # thousands of small products from concurrent workers through BLAS.
        design = donor_quantiles[:, active] - treated_quantile[:, None]
        active_count = design.shape[1]
        gram = np.einsum("qi,qj->ij", design, design, optimize=False)
        gram /= donor_quantiles.shape[0]
        rhs = np.zeros(active_count)
        kkt = np.zeros((active_count + 1, active_count + 1))
        kkt[:active_count, :active_count] = gram
        kkt[:active_count, active_count] = 1.0
        kkt[active_count, :active_count] = 1.0
        target = np.concatenate([rhs, [1.0]])
        solution = np.linalg.lstsq(kkt, target, rcond=None)[0][
            :active_count
        ]
        if np.any(solution < -1.0e-9):
            continue
        candidate = np.zeros(donor_count)
        candidate[active] = np.maximum(solution, 0.0)
        candidate_sum = float(candidate.sum())
        if candidate_sum <= 0.0 or not np.isfinite(candidate_sum):
            continue
        candidate /= candidate_sum
        residual = np.sum(donor_quantiles * candidate[None, :], axis=1)
        residual -= treated_quantile
        objective = float(np.mean(residual * residual))
        if not np.isfinite(objective):
            continue
        if objective < best_objective - 1.0e-14:
            best_objective = objective
            best_weight = candidate
    if not np.all(np.isfinite(best_weight)):
        raise FloatingPointError("Nonfinite distributional-SC weight")
    if np.min(best_weight) < -1.0e-10 or abs(best_weight.sum() - 1.0) > 1.0e-10:
        raise AssertionError("Distributional-SC weights violate the simplex")
    return best_weight, best_objective


def fit_distributional_sc(
    cell_quantiles: dict[tuple[int, int], np.ndarray],
) -> dict[str, Any]:
    """Fit year-specific weights, average them, and construct 2003 effects."""

    period_weights = []
    period_objectives = []
    period_rmse = []
    for year in PRE_YEARS:
        treated_quantile = cell_quantiles[(year, TREATED_STATE)]
        donor_quantiles = np.column_stack(
            [cell_quantiles[(year, donor)] for donor in DONOR_CODES]
        )
        weights, objective = simplex_least_squares(
            donor_quantiles,
            treated_quantile,
        )
        period_weights.append(weights)
        period_objectives.append(objective)
        period_rmse.append(math.sqrt(max(objective, 0.0)))
    weight_matrix = np.vstack(period_weights)
    average_weights = weight_matrix.mean(axis=0)
    if np.min(average_weights) < -1.0e-10 or abs(average_weights.sum() - 1.0) > 1.0e-10:
        raise AssertionError("Average distributional-SC weights violate simplex")
    treated_post = cell_quantiles[(POST_YEAR, TREATED_STATE)]
    donor_post = np.column_stack(
        [cell_quantiles[(POST_YEAR, donor)] for donor in DONOR_CODES]
    )
    synthetic_post = np.sum(donor_post * average_weights[None, :], axis=1)
    effect_curve = treated_post - synthetic_post
    scalar_effect = float(np.mean(effect_curve))
    return {
        "period_weights": weight_matrix,
        "average_weights": average_weights,
        "period_objectives": np.asarray(period_objectives),
        "period_rmse": np.asarray(period_rmse),
        "treated_post": treated_post,
        "synthetic_post": synthetic_post,
        "effect_curve": effect_curve,
        "scalar_effect": scalar_effect,
    }


def sorted_outcome_cells(
    subgroup: pd.DataFrame,
) -> dict[tuple[int, int], np.ndarray]:
    """Return sorted outcomes for all 30 state-year cells."""

    cells: dict[tuple[int, int], np.ndarray] = {}
    for year in PRE_YEARS + (POST_YEAR,):
        for state in ALL_STATES:
            values = subgroup.loc[
                (subgroup["year"] == year)
                & (subgroup["state_fips"] == state),
                "contpov",
            ].to_numpy(dtype=float)
            if values.size == 0:
                raise ValueError(f"Empty state-year cell state={state}, year={year}")
            cells[(year, state)] = np.sort(values)
    return cells


def checkpoint_paths(
    output_dir: Path,
    educ: int,
    nchild: int,
) -> tuple[Path, Path]:
    """Return draw and metadata checkpoint paths."""

    checkpoint_dir = output_dir / "checkpoints"
    stem = f"distributional_sc_e{educ}_c{nchild}"
    return checkpoint_dir / f"{stem}.csv", checkpoint_dir / f"{stem}.json"


def checkpoint_fingerprint(
    educ: int,
    nchild: int,
    config: RunConfig,
    data_hash: str,
) -> dict[str, Any]:
    """Configuration fields that must match before a checkpoint is resumed."""

    return {
        "experiment_version": EXPERIMENT_VERSION,
        "data_sha256": data_hash,
        "educ": educ,
        "nchild": nchild,
        "bootstraps": config.bootstraps,
        "quantiles": config.quantiles,
        "seed": config.seed,
        "age_trim_lower": config.age_trim_lower,
        "age_trim_upper": config.age_trim_upper,
        "donor_codes": list(DONOR_CODES),
        "quantile_definition": "generalized_inverse_midpoints",
        "bootstrap": "multinomial_household_within_state_year",
    }


def load_checkpoint(
    draw_path: Path,
    metadata_path: Path,
    fingerprint: dict[str, Any],
    overwrite: bool,
) -> tuple[list[dict[str, Any]], float]:
    """Load a compatible checkpoint or start an empty draw list."""

    if overwrite:
        return [], 0.0
    if not draw_path.exists() and not metadata_path.exists():
        return [], 0.0
    if not draw_path.exists() or not metadata_path.exists():
        raise ValueError(f"Incomplete checkpoint pair at {draw_path.parent}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("fingerprint") != fingerprint:
        raise ValueError(
            f"Checkpoint configuration mismatch for {draw_path.name}; "
            "use --overwrite to replace it"
        )
    frame = pd.read_csv(draw_path)
    if frame.empty:
        return [], float(metadata.get("bootstrap_seconds", 0.0))
    expected = np.arange(1, frame.shape[0] + 1)
    if not np.array_equal(frame["draw"].to_numpy(dtype=int), expected):
        raise ValueError(f"Nonconsecutive draws in {draw_path}")
    return frame.to_dict("records"), float(
        metadata.get("bootstrap_seconds", 0.0)
    )


def save_checkpoint(
    records: list[dict[str, Any]],
    bootstrap_seconds: float,
    draw_path: Path,
    metadata_path: Path,
    fingerprint: dict[str, Any],
) -> None:
    """Atomically save subgroup bootstrap progress."""

    atomic_csv(pd.DataFrame(records), draw_path)
    atomic_json(
        {
            "fingerprint": fingerprint,
            "completed_draws": len(records),
            "bootstrap_seconds": bootstrap_seconds,
            "status": (
                "complete"
                if len(records) == int(fingerprint["bootstraps"])
                else "checkpoint"
            ),
        },
        metadata_path,
    )


def estimate_distributional_sc(
    subgroup: pd.DataFrame,
    educ: int,
    nchild: int,
    metadata: dict[str, float | int],
    config: RunConfig,
    data_hash: str,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """Estimate distributional SC and its full stratified bootstrap."""

    quantile_grid = (
        np.arange(config.quantiles, dtype=float) + 0.5
    ) / config.quantiles
    cells = sorted_outcome_cells(subgroup)
    point_start = time.perf_counter()
    point_quantiles = {
        key: empirical_quantile(values, quantile_grid)
        for key, values in cells.items()
    }
    point = fit_distributional_sc(point_quantiles)
    point_seconds = time.perf_counter() - point_start

    output_dir = Path(config.output_dir)
    draw_path, metadata_path = checkpoint_paths(
        output_dir,
        educ,
        nchild,
    )
    fingerprint = checkpoint_fingerprint(
        educ,
        nchild,
        config,
        data_hash,
    )
    records, previous_seconds = load_checkpoint(
        draw_path,
        metadata_path,
        fingerprint,
        config.overwrite,
    )
    start_draw = len(records) + 1
    segment_start = time.perf_counter()
    milestone = max(1, math.ceil(config.bootstraps / 10))
    for draw in range(start_draw, config.bootstraps + 1):
        draw_start = time.perf_counter()
        try:
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [config.seed, 4202, educ, nchild, draw]
                )
            )
            bootstrap_quantiles: dict[tuple[int, int], np.ndarray] = {}
            for key, values in cells.items():
                probabilities = np.full(values.size, 1.0 / values.size)
                counts = rng.multinomial(values.size, probabilities)
                bootstrap_quantiles[key] = empirical_quantile(
                    values,
                    quantile_grid,
                    counts,
                )
            fitted = fit_distributional_sc(bootstrap_quantiles)
            scalar = float(fitted["scalar_effect"])
            root = scalar - float(point["scalar_effect"])
            sup_deviation = float(
                np.max(
                    np.abs(
                        fitted["effect_curve"] - point["effect_curve"]
                    )
                )
            )
            row: dict[str, Any] = {
                "educ": educ,
                "nchild": nchild,
                "draw": draw,
                "estimate": scalar,
                "root": root,
                "sup_abs_deviation": sup_deviation,
                "valid": True,
                "error": "",
                "draw_seconds": time.perf_counter() - draw_start,
            }
            for donor_name, weight in zip(
                DONOR_NAMES,
                fitted["average_weights"],
            ):
                row[f"average_weight_{donor_name.replace(' ', '_')}"] = float(
                    weight
                )
        except Exception as exc:  # preserve the failed draw for diagnostics
            row = {
                "educ": educ,
                "nchild": nchild,
                "draw": draw,
                "estimate": np.nan,
                "root": np.nan,
                "sup_abs_deviation": np.nan,
                "valid": False,
                "error": f"{type(exc).__name__}: {exc}",
                "draw_seconds": time.perf_counter() - draw_start,
            }
            for donor_name in DONOR_NAMES:
                row[f"average_weight_{donor_name.replace(' ', '_')}"] = np.nan
        records.append(row)
        if draw % milestone == 0 or draw == config.bootstraps:
            accumulated_seconds = (
                previous_seconds + time.perf_counter() - segment_start
            )
            save_checkpoint(
                records,
                accumulated_seconds,
                draw_path,
                metadata_path,
                fingerprint,
            )
            percent = int(round(100.0 * draw / config.bootstraps))
            print(
                f"Distributional SC educ={educ}, nchild={nchild}: "
                f"{draw}/{config.bootstraps} ({percent}%)",
                flush=True,
            )
    bootstrap_seconds = previous_seconds + time.perf_counter() - segment_start
    if start_draw > config.bootstraps:
        bootstrap_seconds = previous_seconds
        print(
            f"Distributional SC educ={educ}, nchild={nchild}: "
            f"resumed complete checkpoint ({config.bootstraps}/"
            f"{config.bootstraps})",
            flush=True,
        )
    save_checkpoint(
        records,
        bootstrap_seconds,
        draw_path,
        metadata_path,
        fingerprint,
    )

    draws = pd.DataFrame(records)
    valid = draws[draws["valid"].astype(bool)].copy()
    valid_count = int(valid.shape[0])
    if valid_count < config.bootstraps:
        raise RuntimeError(
            f"Only {valid_count}/{config.bootstraps} valid distributional-SC "
            f"draws for educ={educ}, nchild={nchild}"
        )
    estimate = float(point["scalar_effect"])
    bootstrap_se = float(valid["estimate"].std(ddof=1))
    symmetric_critical = float(
        valid["root"].abs().quantile(1.0 - ALPHA)
    )
    symmetric_low = estimate - symmetric_critical
    symmetric_high = estimate + symmetric_critical
    percentile_low, percentile_high = valid["estimate"].quantile(
        [ALPHA / 2.0, 1.0 - ALPHA / 2.0]
    ).to_numpy(dtype=float)
    uniform_critical = float(
        valid["sup_abs_deviation"].quantile(1.0 - ALPHA)
    )

    result = {
        "task": "02_drdid_distributional_sc",
        "method": "distributional_sc",
        "donor_set": "quantile_top4",
        "donors": "|".join(DONOR_NAMES),
        "donor_selection_weights": "|".join(
            f"{weight:.2f}" for weight in GUNSILIUS_SELECTION_WEIGHTS
        ),
        "educ": educ,
        "nchild": nchild,
        "pre_years": "|".join(map(str, PRE_YEARS)),
        "post_year": POST_YEAR,
        "sample_definition": "raw_1998_2003_subgroup_without_age_trimming",
        "n": int(subgroup.shape[0]),
        **metadata,
        "quantiles": config.quantiles,
        "quantile_grid": "midpoints_(j+0.5)/Q",
        "quantile_definition": "generalized_inverse_empirical_cdf",
        "weight_estimator": (
            "period_specific_simplex_least_squares_then_arithmetic_average"
        ),
        "estimate": estimate,
        "bootstrap_se": bootstrap_se,
        "symmetric_ci_low": symmetric_low,
        "symmetric_ci_high": symmetric_high,
        "symmetric_ci_length": symmetric_high - symmetric_low,
        "percentile_ci_low": float(percentile_low),
        "percentile_ci_high": float(percentile_high),
        "percentile_ci_length": float(percentile_high - percentile_low),
        "uniform_critical_value": uniform_critical,
        "pre_rmse_mean": float(np.mean(point["period_rmse"])),
        "pre_rmse_max": float(np.max(point["period_rmse"])),
        "n_boot_requested": config.bootstraps,
        "n_boot_valid": valid_count,
        "bootstrap_scheme": (
            "independent multinomial household resampling within state-year"
        ),
        "point_seconds": point_seconds,
        "bootstrap_seconds": bootstrap_seconds,
        "total_seconds": point_seconds + bootstrap_seconds,
        "seconds_per_bootstrap_draw": bootstrap_seconds / config.bootstraps,
        "workers": 1,
        "seed": config.seed,
        "timing_status": "measured",
    }

    weight_rows = []
    for year_index, year in enumerate(PRE_YEARS):
        for donor_index, (donor_code, donor_name) in enumerate(
            zip(DONOR_CODES, DONOR_NAMES)
        ):
            weight_rows.append(
                {
                    "educ": educ,
                    "nchild": nchild,
                    "row_type": "pre_year",
                    "pre_year": year,
                    "donor_code": donor_code,
                    "donor": donor_name,
                    "weight": float(
                        point["period_weights"][year_index, donor_index]
                    ),
                    "pre_rmse": float(point["period_rmse"][year_index]),
                    "objective": float(
                        point["period_objectives"][year_index]
                    ),
                }
            )
    for donor_index, (donor_code, donor_name) in enumerate(
        zip(DONOR_CODES, DONOR_NAMES)
    ):
        weight_rows.append(
            {
                "educ": educ,
                "nchild": nchild,
                "row_type": "average",
                "pre_year": np.nan,
                "donor_code": donor_code,
                "donor": donor_name,
                "weight": float(point["average_weights"][donor_index]),
                "pre_rmse": float(np.mean(point["period_rmse"])),
                "objective": float(np.mean(point["period_objectives"])),
            }
        )
    weights = pd.DataFrame(weight_rows)

    curve = pd.DataFrame(
        {
            "educ": educ,
            "nchild": nchild,
            "quantile": quantile_grid,
            "treated_post_quantile": point["treated_post"],
            "synthetic_post_quantile": point["synthetic_post"],
            "quantile_effect": point["effect_curve"],
            "uniform_ci_low": point["effect_curve"] - uniform_critical,
            "uniform_ci_high": point["effect_curve"] + uniform_critical,
            "uniform_critical_value": uniform_critical,
        }
    )
    timing = {
        "task": "02_drdid_distributional_sc",
        "method": "distributional_sc",
        "educ": educ,
        "nchild": nchild,
        "n": int(subgroup.shape[0]),
        "point_estimation_seconds": point_seconds,
        "bootstrap_seconds": bootstrap_seconds,
        "bootstrap_draws_requested": config.bootstraps,
        "bootstrap_draws_valid": valid_count,
        "seconds_per_bootstrap_draw": bootstrap_seconds / config.bootstraps,
        "cell_wall_seconds": point_seconds + bootstrap_seconds,
        "worker_processes": 1,
        "timing_status": "measured",
    }
    return result, weights, draws, curve, timing


def subgroup_worker(task: tuple[int, int, RunConfig, str]) -> dict[str, Any]:
    """Run requested methods for one subgroup in a spawned process."""

    educ, nchild, config, data_hash = task
    data = load_data(Path(config.data_path))
    subgroup, metadata = subgroup_sample(
        data,
        educ,
        nchild,
        config.age_trim_lower,
        config.age_trim_upper,
    )
    output: dict[str, Any] = {
        "educ": educ,
        "nchild": nchild,
    }
    if config.run_drdid:
        drdid, influence, drdid_timing = estimate_repeated_cross_section_drdid(
            subgroup,
            educ,
            nchild,
            metadata,
            config,
        )
        output["drdid"] = drdid
        output["influence"] = influence
        output["drdid_timing"] = drdid_timing
    if config.run_dsc:
        dsc, weights, draws, curve, dsc_timing = estimate_distributional_sc(
            subgroup,
            educ,
            nchild,
            metadata,
            config,
            data_hash,
        )
        output["dsc"] = dsc
        output["weights"] = weights
        output["draws"] = draws
        output["curve"] = curve
        output["dsc_timing"] = dsc_timing
    return output


def concatenate_frames(items: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate nonempty frames with a stable empty fallback."""

    frames = [frame for frame in items if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def proposed_long(path: Path) -> pd.DataFrame:
    """Read the compatible Task 1 symmetric-root results if available."""

    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path)
    required = {
        "educ",
        "nchild",
        "estimate",
        "bootstrap_se",
        "ci_low",
        "ci_high",
        "n_boot_requested",
        "n_boot_valid",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(
            f"Task 1 result {path} is missing columns {missing}"
        )
    subgroup_count = data[["educ", "nchild"]].drop_duplicates().shape[0]
    if data.shape[0] != len(SUBGROUPS) or subgroup_count != len(SUBGROUPS):
        print(
            f"Task 1 result {path} is still incomplete; the proposed column "
            "will be populated by a later --summarize-only run.",
            flush=True,
        )
        return pd.DataFrame()
    out = data.copy()
    out["method"] = "proposed_didsc"
    out["se"] = out["bootstrap_se"]
    out["ci_length"] = out["ci_high"] - out["ci_low"]
    out["inference"] = "symmetric_multiplier_bootstrap"
    return out


def create_method_comparison(
    drdid: pd.DataFrame,
    dsc: pd.DataFrame,
    proposed_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """Create a tidy long comparison and the publication-ready table."""

    rows = []
    proposed = proposed_long(proposed_path)
    for row in proposed.itertuples(index=False):
        rows.append(
            {
                "educ": int(row.educ),
                "nchild": int(row.nchild),
                "method": "proposed_didsc",
                "estimate": float(row.estimate),
                "se": float(row.bootstrap_se),
                "ci_low": float(row.ci_low),
                "ci_high": float(row.ci_high),
                "ci_length": float(row.ci_high - row.ci_low),
                "inference": "symmetric_multiplier_bootstrap",
                "draws": int(row.n_boot_valid),
            }
        )
    for row in drdid.itertuples(index=False):
        rows.append(
            {
                "educ": int(row.educ),
                "nchild": int(row.nchild),
                "method": "drdid_repeated_cross_section",
                "estimate": float(row.estimate),
                "se": float(row.analytic_se),
                "ci_low": float(row.ci_low),
                "ci_high": float(row.ci_high),
                "ci_length": float(row.ci_length),
                "inference": (
                    "centered_overlap_trimmed_plugin_analytic_influence"
                ),
                "draws": 0,
            }
        )
    for row in dsc.itertuples(index=False):
        rows.append(
            {
                "educ": int(row.educ),
                "nchild": int(row.nchild),
                "method": "distributional_sc",
                "estimate": float(row.estimate),
                "se": float(row.bootstrap_se),
                "ci_low": float(row.symmetric_ci_low),
                "ci_high": float(row.symmetric_ci_high),
                "ci_length": float(row.symmetric_ci_length),
                "inference": "symmetric_stratified_household_bootstrap",
                "draws": int(row.n_boot_valid),
            }
        )
    comparison = pd.DataFrame(rows).sort_values(
        ["educ", "nchild", "method"]
    )
    atomic_csv(comparison, output_dir / "method_comparison.csv")
    write_comparison_latex(comparison, output_dir / "method_comparison.tex")
    return comparison


def format_estimate_interval(row: pd.Series | None) -> str:
    """Format a scalar estimate and interval for LaTeX."""

    if row is None:
        return "--"
    return (
        f"{float(row['estimate']):.3f} "
        f"[{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}]"
    )


def write_comparison_latex(comparison: pd.DataFrame, path: Path) -> None:
    """Write the compact side-by-side empirical table."""

    lines = [
        r"\begin{table}[htbp!]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"Education & Children & Proposed DiD--SC & DRDiD & Distributional SC \\",
        r"\midrule",
    ]
    for educ, nchild in SUBGROUPS:
        cell = comparison[
            (comparison["educ"] == educ)
            & (comparison["nchild"] == nchild)
        ]
        values: dict[str, str] = {}
        for method in (
            "proposed_didsc",
            "drdid_repeated_cross_section",
            "distributional_sc",
        ):
            method_row = cell[cell["method"] == method]
            values[method] = format_estimate_interval(
                None if method_row.empty else method_row.iloc[0]
            )
        lines.append(
            f"{educ} & {nchild} & {values['proposed_didsc']} & "
            f"{values['drdid_repeated_cross_section']} & "
            f"{values['distributional_sc']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Comparison with DRDiD and distributional synthetic control}",
            r"\label{tab:empirical-method-comparison}",
            r"\caption*{\footnotesize Entries report point estimates with 95\% confidence intervals in brackets. For the two local-smoothing estimators, nuisance functions are trained on the raw subgroup and score contributions outside the pooled strict 5th--95th percentile age support are set to zero; Task 1's displayed $n$ is the raw six-year subgroup size before score trimming. The proposed DiD--SC interval is the symmetric multiplier-bootstrap interval based on 500 draws. DRDiD uses the 2002--2003 repeated cross sections, two-fold cross-fitted Epanechnikov local-linear nuisances, and a centered plug-in analytic influence interval based on Sant'Anna and Zhao (2020). Its MSE bandwidths obey $h=h_{\rm ref}(n/957)^{-1/5}$, where 957 is the median raw 2002--2003 subgroup size and $(h_{Y,\rm ref},h_{p,\rm ref})=(6.25,12.94)$. Distributional SC instead targets the raw subgroup outcome distribution, uses 1,000 midpoint quantiles, averages separately estimated 1998--2002 nonnegative-simplex weights, and reports a symmetric full-bootstrap interval based on 1,000 household resamples within state-year cells. This is the \texttt{simplex=TRUE} option in DiSCos, rather than its default affine-weight option. Cross-method differences are descriptive because the estimands and identifying restrictions differ. Alaska is treated. The donor pool is fixed ex ante from Gunsilius's quantile-weight ranking: Virginia (0.11), New Hampshire (0.11), Maryland (0.09), and Utah (0.07).}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_distributional_latex(dsc: pd.DataFrame, path: Path) -> None:
    """Write a DSC-only table including both requested scalar intervals."""

    lines = [
        r"\begin{table}[htbp!]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\begin{tabular}{ccccccc}",
        r"\toprule",
        r"Education & Children & Estimate & SE & Symmetric CI & Percentile CI & Pre-RMSPE \\",
        r"\midrule",
    ]
    for row in dsc.sort_values(["educ", "nchild"]).itertuples(index=False):
        lines.append(
            f"{int(row.educ)} & {int(row.nchild)} & {row.estimate:.3f} & "
            f"{row.bootstrap_se:.3f} & [{row.symmetric_ci_low:.3f}, "
            f"{row.symmetric_ci_high:.3f}] & [{row.percentile_ci_low:.3f}, "
            f"{row.percentile_ci_high:.3f}] & {row.pre_rmse_mean:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Distributional synthetic-control benchmark}",
            r"\label{tab:empirical-distributional-sc}",
            r"\caption*{\footnotesize Distributional SC uses each raw subgroup without age trimming because it has no local-smoothing boundary nuisance; it therefore targets the raw subgroup outcome distribution. The donor pool is fixed ex ante from Gunsilius's quantile-weight ranking: Virginia (0.11), New Hampshire (0.11), Maryland (0.09), and Utah (0.07). Separate nonnegative-simplex weights are estimated in each pre-treatment year and averaged; this is the \texttt{simplex=TRUE} option in DiSCos rather than its default affine-weight option. The scalar estimate is the mean 2003 quantile contrast over 1,000 midpoint quantiles. SE and both 95\% confidence intervals use 1,000 full household bootstrap resamples within state-year cells; all empirical quantiles and weights are re-estimated in every draw. Pre-RMSPE is the mean pre-treatment quantile-function RMSPE.}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_drdid_latex(drdid: pd.DataFrame, path: Path) -> None:
    """Write the repeated-cross-section DRDiD table."""

    lines = [
        r"\begin{table}[htbp!]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5.0pt}",
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"Education & Children & Estimate & Analytic SE & 95\% CI & $n$ \\",
        r"\midrule",
    ]
    for row in drdid.sort_values(["educ", "nchild"]).itertuples(index=False):
        lines.append(
            f"{int(row.educ)} & {int(row.nchild)} & {row.estimate:.3f} & "
            f"{row.analytic_se:.3f} & [{row.ci_low:.3f}, "
            f"{row.ci_high:.3f}] & {int(row.n)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Repeated-cross-section DRDiD benchmark}",
            r"\label{tab:empirical-drdid}",
            r"\caption*{\footnotesize The table reports the repeated-cross-section DRDiD benchmark for 2002--2003. Outcome regressions for all four treatment-by-period cells and the treatment propensity are trained on the full raw sample by two-fold cross-fitted Epanechnikov local-linear regression. Score contributions outside the pooled strict 5th--95th percentile age support are set to zero, using the full-sample normalizations, to match the proposed estimator's empirical support convention. MSE bandwidths obey $h=h_{\rm ref}(n/957)^{-1/5}$, where 957 is the median raw 2002--2003 subgroup size and $(h_{Y,\rm ref},h_{p,\rm ref})=(6.25,12.94)$. Analytic SE uses the centered support-weighted plug-in influence contribution based on equation (2.13) of Sant'Anna and Zhao (2020), modified for the fixed support-weighted target.}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_quantile_effects(curve: pd.DataFrame, output_dir: Path) -> None:
    """Plot nine distributional-SC quantile-effect curves and uniform bands."""

    mpl_cache = output_dir / "tmp" / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 3, figsize=(11.0, 8.2), sharex=True)
    for axis, (educ, nchild) in zip(axes.ravel(), SUBGROUPS):
        cell = curve[
            (curve["educ"] == educ) & (curve["nchild"] == nchild)
        ].sort_values("quantile")
        q = cell["quantile"].to_numpy(dtype=float)
        effect = cell["quantile_effect"].to_numpy(dtype=float)
        low = cell["uniform_ci_low"].to_numpy(dtype=float)
        high = cell["uniform_ci_high"].to_numpy(dtype=float)
        axis.fill_between(q, low, high, color="#9ecae1", alpha=0.55)
        axis.plot(q, effect, color="#08519c", linewidth=1.25)
        axis.axhline(0.0, color="black", linewidth=0.7, linestyle="--")
        axis.set_title(f"Education {educ}, children {nchild}", fontsize=9)
        axis.grid(alpha=0.15)
    for axis in axes[-1, :]:
        axis.set_xlabel("Quantile")
    for axis in axes[:, 0]:
        axis.set_ylabel("Quantile effect")
    figure.suptitle(
        "Distributional synthetic-control effects with 95% uniform bands",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(
        output_dir / "distributional_sc_quantile_effects.png",
        dpi=220,
        bbox_inches="tight",
    )
    figure.savefig(
        output_dir / "distributional_sc_quantile_effects.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def write_runtime_outputs(
    timing: pd.DataFrame,
    run_total_wall_seconds: float,
    effective_workers: int,
    seed: int,
    output_dir: Path,
) -> None:
    """Write detailed and summarized computation-time records."""

    timing = timing.copy()
    timing["configuration"] = "quantile_top4"
    timing["sample_size"] = timing["n"]
    timing["total_cell_seconds"] = timing["cell_wall_seconds"]
    timing["worker_seconds_sum"] = timing["total_cell_seconds"]
    timing["run_total_wall_seconds"] = run_total_wall_seconds
    timing["seed"] = seed
    timing["code_version"] = EXPERIMENT_VERSION
    timing["worker_processes"] = effective_workers
    timing["effective_parallel_workers"] = effective_workers
    timing["complete_run_wall_seconds"] = run_total_wall_seconds
    required_columns = [
        "task",
        "method",
        "configuration",
        "educ",
        "nchild",
        "sample_size",
        "point_estimation_seconds",
        "bootstrap_seconds",
        "bootstrap_draws_requested",
        "bootstrap_draws_valid",
        "seconds_per_bootstrap_draw",
        "total_cell_seconds",
        "worker_processes",
        "worker_seconds_sum",
        "run_total_wall_seconds",
        "timing_status",
        "seed",
        "code_version",
    ]
    extra_columns = [
        column for column in timing.columns if column not in required_columns
    ]
    timing = timing[required_columns + extra_columns]
    atomic_csv(timing, output_dir / "runtime_detailed.csv")
    summary = (
        timing.groupby("method", as_index=False)
        .agg(
            subgroups=("educ", "size"),
            median_n=("n", "median"),
            median_point_seconds=("point_estimation_seconds", "median"),
            median_bootstrap_seconds=("bootstrap_seconds", "median"),
            median_cell_wall_seconds=("total_cell_seconds", "median"),
            sum_worker_seconds=("worker_seconds_sum", "sum"),
            draws_requested=("bootstrap_draws_requested", "sum"),
            draws_valid=("bootstrap_draws_valid", "sum"),
        )
    )
    summary["effective_parallel_workers"] = effective_workers
    summary["run_total_wall_seconds"] = run_total_wall_seconds
    summary["complete_run_wall_seconds"] = run_total_wall_seconds
    summary["timing_status"] = "measured"
    summary["seed"] = seed
    summary["code_version"] = EXPERIMENT_VERSION
    atomic_csv(summary, output_dir / "runtime_summary.csv")


def write_provenance(
    output_dir: Path,
    data_path: Path,
    proposed_path: Path,
) -> None:
    """Document new, reused, and rejected predecessor artifacts."""

    legacy_root = TASK_DIR / "excluded_legacy_artifacts"
    proposed_complete = False
    if proposed_path.exists():
        try:
            proposed_frame = pd.read_csv(proposed_path)
            proposed_complete = (
                proposed_frame.shape[0] == len(SUBGROUPS)
                and proposed_frame[["educ", "nchild"]]
                .drop_duplicates()
                .shape[0]
                == len(SUBGROUPS)
            )
        except (OSError, ValueError, pd.errors.ParserError):
            proposed_complete = False
    source_script = Path(__file__).resolve()
    drdid_paper = TASK_DIR.parents[1] / "DRDiD" / "SantAnna_Zhao_DRDID.pdf"
    legacy_data = legacy_root / "Alaska_MW.csv"
    records = [
        {
            "artifact": "empirical_data",
            "status": "reused_identical_copy",
            "source_path": str(
                (legacy_data if legacy_data.exists() else data_path).resolve()
            ),
            "destination_path": str(data_path.resolve()),
            "reason": "Task-local Alaska_MW.csv; SHA-256 recorded in metadata",
        },
        {
            "artifact": "task_runner",
            "status": "executed",
            "source_path": str(source_script),
            "destination_path": "",
            "reason": "Self-contained Task 2 implementation",
        },
        {
            "artifact": "proposed_didsc",
            "status": "reused_from_task1" if proposed_complete else "pending_task1",
            "source_path": str(proposed_path.resolve()),
            "destination_path": str((output_dir / "method_comparison.csv").resolve()),
            "reason": "Compatible quantile-top-four symmetric-root result" if proposed_complete else "A complete nine-subgroup Task 1 result was not yet available when Task 2 ran",
        },
        {
            "artifact": "legacy_empirical_drdid",
            "status": "not_reused",
            "source_path": str(
                (
                    legacy_root
                    / "outputs/empirical_review/top4_benchmark_didsc_vs_pooled_drdid.csv"
                ).resolve()
            ),
            "destination_path": "",
            "reason": "Legacy code retained all six years in a two-period score, used undersmoothed bandwidths and multiplier inference, and did not implement the efficient repeated-cross-section influence function",
        },
        {
            "artifact": "legacy_distributional_sc",
            "status": "not_reused",
            "source_path": str(
                (
                    legacy_root
                    / "outputs/empirical_review/dsc_bootstrap_500.csv"
                ).resolve()
            ),
            "destination_path": "",
            "reason": "Legacy result used six donors, 99 quantiles, one pooled pre-period weight vector, exponential multipliers, and 500 draws",
        },
        {
            "artifact": "drdid_reference_paper",
            "status": "reference_only",
            "source_path": str(drdid_paper.resolve()),
            "destination_path": "",
            "reason": "Sant'Anna-Zhao repeated-cross-section estimator and efficient influence function, equations 2.9 and 2.13",
        },
        {
            "artifact": "gunsilius_econometrica_reference",
            "status": "reference_only",
            "source_path": "",
            "destination_path": "",
            "reference_url": "https://doi.org/10.3982/ECTA18260",
            "reason": "Official Econometrica article for Distributional Synthetic Controls",
        },
        {
            "artifact": "discos_reference_implementation",
            "status": "reference_only",
            "source_path": "",
            "destination_path": "",
            "reference_url": "https://www.davidvandijcke.com/DiSCos/",
            "reason": "Official DiSCos package site developed by David Van Dijcke, Florian Gunsilius, and Siyun He",
        },
        {
            "artifact": "drdid_repeated_cross_section",
            "status": "newly_computed",
            "source_path": str(source_script),
            "destination_path": str((output_dir / "drdid_results.csv").resolve()),
            "reason": "Required repeated-cross-section estimator with the empirical central-support convention and analytic plug-in influence inference",
        },
        {
            "artifact": "distributional_sc_full_bootstrap",
            "status": "newly_computed",
            "source_path": str(source_script),
            "destination_path": str((output_dir / "distributional_sc_results.csv").resolve()),
            "reason": "Required year-specific weights, 1,000 midpoint quantiles, and 1,000 full stratified household resamples",
        },
    ]
    for record in records:
        source_text = str(record.get("source_path", ""))
        destination_text = str(record.get("destination_path", ""))
        source = Path(source_text) if source_text else None
        destination = Path(destination_text) if destination_text else None
        record["source_sha256"] = (
            sha256_file(source)
            if source is not None and source.is_file()
            else ""
        )
        record["destination_sha256"] = (
            sha256_file(destination)
            if destination is not None and destination.is_file()
            else ""
        )
        record["reference_url"] = str(record.get("reference_url", ""))
    provenance_columns = [
        "artifact",
        "status",
        "source_path",
        "source_sha256",
        "reference_url",
        "destination_path",
        "destination_sha256",
        "reason",
    ]
    atomic_csv(
        pd.DataFrame(records)[provenance_columns],
        output_dir / "provenance.csv",
    )


def write_readme(output_dir: Path, config: RunConfig) -> None:
    """Write task-local reproducibility notes."""

    command = (
        "python3 empirical/02_drdid_distributional_sc/"
        "run_drdid_distributional_sc.py --bootstraps 1000 --quantiles "
        f"1000 --nprocs {config.requested_workers}"
    )
    text = f"""# DRDiD and distributional-SC empirical benchmarks

This folder compares the proposed DiD-SC estimates with two benchmarks for the
nine education-by-children subgroups. Alaska is treated. The donor pool is
fixed ex ante from Gunsilius's quantile-weight ranking: Virginia (0.11), New
Hampshire (0.11), Maryland (0.09), and Utah (0.07). The common age support is
the strict 5th-95th percentile range within each raw six-year subgroup. The two
local-smoothing estimators train their nuisances on all raw observations and
set evaluation-score contributions outside this fixed range to zero. Task 1's
displayed `n` is therefore the raw subgroup size before score trimming.
Distributional SC has no local-smoothing boundary nuisance and uses the raw,
untrimmed subgroup outcome distribution. Its estimand is consequently not
identical to the overlap-trimmed local-smoothing estimands.

## DRDiD

The code uses a support-weighted version of the repeated-cross-section
Sant'Anna-Zhao score for 2002-2003, not the panel score. It estimates all four treatment-by-period
outcome regressions and the treatment propensity by two-fold cross-fitted
Epanechnikov local-linear regression. The bandwidths 6.25 (outcome) and 12.94
(propensity) are the data-driven MSE-optimal reference bandwidths at
n_ref={config.bandwidth_reference_n:g}, the median raw 2002-2003 sample size
across the nine subgroups. Each cell uses
h_cell=h_reference*(n_2002_2003/n_ref)^(-1/5), without the DiD-SC
undersmoothing factor. The reported 95% interval
uses the centered support-weighted plug-in influence contribution based on
Sant'Anna and Zhao's equation (2.13). Because the fixed support indicator
changes the target from their untrimmed ATT, the code does not describe this as
a verbatim implementation of their untrimmed efficient influence function.

The fixed support indicator is denoted S(X). The estimand is the same
full-denominator score restriction used by Task 1:

```text
theta_S = E[S(X) D (Delta m_1(X) - Delta m_0(X))] / E[D].
```

The pooled empirical 5th and 95th percentile cutoffs defining S(X) are held
fixed in the analytic variance calculation, matching Task 1's support
convention; uncertainty from re-estimating those two cutoffs is not added.

Starting from the non-theta part of the repeated-cross-section orthogonal score,
the code multiplies every augmentation and residual contribution by S(X),
calling the result zeta_S. Since S(X) is a fixed function of X, conditional
mean-zero residuals remain mean zero and nuisance orthogonality is preserved.
The centered influence contribution is

```text
phi_S = zeta_S - D/E[D] * theta_S.
```

The final term accounts explicitly for estimating the full-sample treated
share. Treated-by-period and reweighted-control-by-period residual weights are
also normalized by their full-sample empirical cell masses. Their denominator
estimation has no additional first-order term because the corresponding
residual numerators have conditional mean zero. The result CSV records all four
cell denominators, the treated share, the support share and treated support
mass, and the means of every normalized weight; the observation-level file
records S(X), all normalized weights, zeta_S, and phi_S.

## Distributional synthetic control

The code uses 1,000 midpoint quantiles on the full outcome support of each raw
subgroup; it does not trim ages or truncate the quantile range.
It estimates a nonnegative-simplex weight vector separately in each year
1998-2002, averages those five vectors, and constructs Alaska's 2003
counterfactual quantile function. This corresponds to the documented
`simplex=TRUE` DiSCos option; DiSCos's default `simplex=FALSE` instead imposes
only the adding-up constraint and permits negative weights. Each of 1,000
bootstrap draws independently resamples
households within every state-year cell and recomputes all empirical quantiles,
five weight vectors, average weights, scalar mean effect, and quantile-effect
curve. The main scalar interval is symmetric in the bootstrap root; the
percentile interval and an unstudentized 95% uniform quantile-effect band are
also saved.

There is no survey-weight variable in `Alaska_MW.csv`, so all empirical
distributions and local regressions are unweighted. Each row is a unique
household within a state-year cell.

## Method references

- Gunsilius (2023), *Distributional Synthetic Controls*, Econometrica:
  https://doi.org/10.3982/ECTA18260
- Official DiSCos reference implementation and documentation:
  https://www.davidvandijcke.com/DiSCos/
- Sant'Anna and Zhao (2020), *Doubly Robust Difference-in-Differences
  Estimators*, Journal of Econometrics.

## Run

```bash
{command}
```

The program prints every 10% distributional-SC milestone, writes subgroup
checkpoints under `checkpoints/`, resumes compatible checkpoints by default,
and records measured point-estimation and bootstrap time. Use `--overwrite`
only when intentionally replacing incompatible checkpoints. Once Task 1 is
complete, `--summarize-only` refreshes the side-by-side table without rerunning
either benchmark.

## Main outputs

- `drdid_results.csv` and `drdid_influence.csv`
- `distributional_sc_results.csv`, `distributional_sc_weights.csv`, and
  `distributional_sc_bootstrap_draws.csv`
- `distributional_sc_quantile_effects.csv` and the PNG/PDF figure
- `method_comparison.csv` and `method_comparison.tex`
- `runtime_detailed.csv`, `runtime_summary.csv`, `metadata.json`, and
  `provenance.csv`
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def aggregate_existing(
    output_dir: Path,
    proposed_path: Path,
) -> None:
    """Refresh tables and figure from completed CSVs without estimation."""

    drdid_path = output_dir / "drdid_results.csv"
    dsc_path = output_dir / "distributional_sc_results.csv"
    curve_path = output_dir / "distributional_sc_quantile_effects.csv"
    if not drdid_path.exists() or not dsc_path.exists() or not curve_path.exists():
        raise FileNotFoundError(
            "--summarize-only requires completed DRDiD and DSC result CSVs"
        )
    drdid = pd.read_csv(drdid_path)
    dsc = pd.read_csv(dsc_path)
    curve = pd.read_csv(curve_path)
    create_method_comparison(drdid, dsc, proposed_path, output_dir)
    write_drdid_latex(drdid, output_dir / "drdid_table.tex")
    write_distributional_latex(
        dsc,
        output_dir / "distributional_sc_table.tex",
    )
    plot_quantile_effects(curve, output_dir)
    write_provenance(output_dir, DEFAULT_DATA, proposed_path)


def run(config: RunConfig, proposed_path: Path) -> None:
    """Run requested estimators in parallel and write all Task 2 artifacts."""

    driver_start = time.perf_counter()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(config.data_path)
    data_hash = sha256_file(data_path)
    reference_data = load_data(data_path)
    observed_reference_n = empirical_bandwidth_reference_sample_size(
        reference_data
    )
    if (
        config.bandwidth_reference_n == DEFAULT_BANDWIDTH_REFERENCE_N
        and observed_reference_n != DEFAULT_BANDWIDTH_REFERENCE_N
    ):
        raise ValueError(
            "The frozen default bandwidth reference n does not match the "
            f"input data: expected {DEFAULT_BANDWIDTH_REFERENCE_N:g}, "
            f"observed {observed_reference_n:g}. Pass an explicit "
            "--bandwidth-reference-n for a different input."
        )
    tasks = [
        (educ, nchild, config, data_hash)
        for educ, nchild in SUBGROUPS
    ]
    effective_workers = max(
        1,
        min(config.requested_workers, len(tasks)),
    )
    estimation_start = time.perf_counter()
    if effective_workers == 1:
        outputs = [subgroup_worker(task) for task in tasks]
    else:
        context = get_context("spawn")
        with context.Pool(processes=effective_workers) as pool:
            outputs = pool.map(subgroup_worker, tasks)
    estimation_wall_seconds = time.perf_counter() - estimation_start

    timing_rows = []
    if config.run_drdid:
        drdid = pd.DataFrame([item["drdid"] for item in outputs]).sort_values(
            ["educ", "nchild"]
        )
        influence = concatenate_frames(
            item["influence"] for item in outputs
        ).sort_values(["educ", "nchild", "row_id"])
        atomic_csv(drdid, output_dir / "drdid_results.csv")
        atomic_csv(influence, output_dir / "drdid_influence.csv")
        timing_rows.extend(item["drdid_timing"] for item in outputs)
    else:
        drdid = pd.read_csv(output_dir / "drdid_results.csv")

    if config.run_dsc:
        dsc = pd.DataFrame([item["dsc"] for item in outputs]).sort_values(
            ["educ", "nchild"]
        )
        weights = concatenate_frames(
            item["weights"] for item in outputs
        ).sort_values(
            ["educ", "nchild", "row_type", "pre_year", "donor_code"],
            na_position="last",
        )
        draws = concatenate_frames(
            item["draws"] for item in outputs
        ).sort_values(["educ", "nchild", "draw"])
        curve = concatenate_frames(
            item["curve"] for item in outputs
        ).sort_values(["educ", "nchild", "quantile"])
        atomic_csv(dsc, output_dir / "distributional_sc_results.csv")
        atomic_csv(weights, output_dir / "distributional_sc_weights.csv")
        atomic_csv(
            draws,
            output_dir / "distributional_sc_bootstrap_draws.csv",
        )
        atomic_csv(
            curve,
            output_dir / "distributional_sc_quantile_effects.csv",
        )
        timing_rows.extend(item["dsc_timing"] for item in outputs)
    else:
        dsc = pd.read_csv(output_dir / "distributional_sc_results.csv")

    create_method_comparison(drdid, dsc, proposed_path, output_dir)
    timing = pd.DataFrame(timing_rows)
    script_path = Path(__file__).resolve()
    metadata = {
        "experiment_version": EXPERIMENT_VERSION,
        "status": "complete",
        "data_path": str(data_path.resolve()),
        "data_sha256": data_hash,
        "source_sha256": sha256_file(script_path),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "config": asdict(config),
        "effective_workers": effective_workers,
        "parallel_estimation_wall_seconds": estimation_wall_seconds,
        "proposed_result_path": str(proposed_path.resolve()),
        "proposed_result_available": proposed_path.exists(),
        "data_weighting": "unweighted; source CSV has no survey-weight column",
        "target_sample": (
            "DiD-SC and DRDiD train on raw subgroups and zero score contributions "
            "outside fixed pooled strict 5th-95th percentile age support; "
            "distributional SC targets the raw untrimmed subgroup distribution"
        ),
        "drdid_reference": (
            "Sant'Anna and Zhao (2020), repeated-cross-section equations "
            "2.9 and 2.13"
        ),
        "drdid_bandwidth_rule": (
            f"h=h_reference*(n_2002_2003/{config.bandwidth_reference_n:g})^(-1/5); "
            f"hY_reference={config.outcome_bandwidth:g}; "
            f"hp_reference={config.propensity_bandwidth:g}"
        ),
        "drdid_bandwidth_reference_n_definition": (
            "median raw 2002-2003 sample size across the nine quantile-top4 subgroups"
        ),
        "drdid_bandwidth_reference_n_observed_in_input": observed_reference_n,
        "dsc_bootstrap": (
            "full multinomial household resampling within state-year cells"
        ),
    }
    run_total_wall_seconds = time.perf_counter() - driver_start
    if not timing.empty:
        write_runtime_outputs(
            timing,
            run_total_wall_seconds,
            effective_workers,
            config.seed,
            output_dir,
        )
    print(
        f"Task 2 completed in {run_total_wall_seconds:.1f}s with "
        f"{effective_workers} worker(s).",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse the command line."""

    cpu_count = os.cpu_count() or 1
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated-cross-section DRDiD and full-bootstrap "
            "distributional SC for the Alaska application."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=TASK_DIR / "generated")
    parser.add_argument(
        "--proposed-results",
        type=Path,
        default=DEFAULT_PROPOSED_RESULTS,
    )
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--quantiles", type=int, default=DEFAULT_QUANTILES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--nprocs",
        type=int,
        default=max(1, min(9, cpu_count - 2)),
    )
    parser.add_argument(
        "--age-trim-lower",
        type=float,
        default=DEFAULT_AGE_TRIM[0],
    )
    parser.add_argument(
        "--age-trim-upper",
        type=float,
        default=DEFAULT_AGE_TRIM[1],
    )
    parser.add_argument(
        "--outcome-bandwidth",
        type=float,
        default=DEFAULT_OUTCOME_BANDWIDTH,
    )
    parser.add_argument(
        "--propensity-bandwidth",
        type=float,
        default=DEFAULT_PROPENSITY_BANDWIDTH,
    )
    parser.add_argument(
        "--bandwidth-reference-n",
        type=float,
        default=DEFAULT_BANDWIDTH_REFERENCE_N,
    )
    parser.add_argument(
        "--propensity-clip",
        type=float,
        default=DEFAULT_PROPENSITY_CLIP,
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument(
        "--methods",
        choices=("both", "drdid", "dsc"),
        default="both",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Fail early on unsafe or incoherent run settings."""

    if not args.data.exists():
        raise FileNotFoundError(args.data)
    if args.bootstraps < 2:
        raise ValueError("At least two bootstrap draws are required")
    if args.quantiles < 20:
        raise ValueError("At least 20 quantiles are required")
    if args.nprocs < 1:
        raise ValueError("--nprocs must be positive")
    if not 0.0 <= args.age_trim_lower < args.age_trim_upper <= 1.0:
        raise ValueError("Invalid age-trim quantiles")
    if args.outcome_bandwidth <= 0.0 or args.propensity_bandwidth <= 0.0:
        raise ValueError("Bandwidths must be positive")
    if args.bandwidth_reference_n <= 0.0:
        raise ValueError("--bandwidth-reference-n must be positive")
    if not 0.0 < args.propensity_clip < 0.5:
        raise ValueError("Propensity clipping must lie in (0,0.5)")
    if args.folds != 2:
        raise ValueError("This reproducibility design fixes two folds")


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    validate_args(args)
    if args.summarize_only:
        aggregate_existing(
            args.output_dir.resolve(),
            args.proposed_results.resolve(),
        )
        return
    config = RunConfig(
        data_path=str(args.data.resolve()),
        output_dir=str(args.output_dir.resolve()),
        bootstraps=args.bootstraps,
        quantiles=args.quantiles,
        seed=args.seed,
        age_trim_lower=args.age_trim_lower,
        age_trim_upper=args.age_trim_upper,
        outcome_bandwidth=args.outcome_bandwidth,
        propensity_bandwidth=args.propensity_bandwidth,
        bandwidth_reference_n=args.bandwidth_reference_n,
        propensity_clip=args.propensity_clip,
        folds=args.folds,
        requested_workers=args.nprocs,
        run_drdid=args.methods in ("both", "drdid"),
        run_dsc=args.methods in ("both", "dsc"),
        overwrite=args.overwrite,
    )
    run(config, args.proposed_results.resolve())


if __name__ == "__main__":
    main()
