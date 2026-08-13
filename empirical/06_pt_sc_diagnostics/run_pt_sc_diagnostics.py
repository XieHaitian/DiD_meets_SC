#!/usr/bin/env python3
"""Pre-treatment PT and conditional-SC diagnostics for the Alaska application.

The primary calculations use only 1998--2002 observations from Alaska and the
quantile-top-four donor pool.  They intentionally do not inspect the 2003
outcome.  Exponential multiplier draws re-estimate all conditional means and
synthetic-control weights while holding deterministic cross-fitting folds and
the evaluation grid fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd


SCRIPT_VERSION = "task6_conditional_preperiod_v2"
SCRIPT_DIR = Path(__file__).resolve().parent
EMPIRICAL_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_DATA = EMPIRICAL_DIR / "Alaska_MW.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "generated"
CANONICAL_DATA_SHA256 = "78cd29c90c2ac88e6b0f8fa77d3d3c9d955c3583964e4bd84aaf319ed2e0affe"

TREATED_STATE = 2
PRE_YEARS = np.asarray([1998, 1999, 2000, 2001, 2002], dtype=int)
DONORS = np.asarray([51, 33, 24, 49], dtype=int)
STATES = np.concatenate(([TREATED_STATE], DONORS))
STATE_NAMES = {
    2: "Alaska",
    24: "Maryland",
    33: "New Hampshire",
    49: "Utah",
    51: "Virginia",
}
SUBGROUPS = [(educ, nchild) for educ in range(3) for nchild in range(3)]
RIDGE_LOCAL = 1.0e-6
RIDGE_SC = 1.0e-6
BOUNDARY_TOL = 0.01


@dataclass(frozen=True)
class RunConfig:
    bootstrap: int
    age_grid_size: int
    seed: int
    trim_low: float
    trim_high: float
    bandwidth_constant: float
    bandwidth_power: float
    simplex_iterations: int
    data_path_resolved: str
    data_sha256: str

    def signature(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256((SCRIPT_VERSION + payload).encode("utf-8")).hexdigest()[:16]

    def legacy_signature(self) -> str:
        """Return the pre-data-fingerprint signature for checkpoint migration."""

        payload = asdict(self)
        payload.pop("data_path_resolved")
        payload.pop("data_sha256")
        serialized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256((SCRIPT_VERSION + serialized).encode("utf-8")).hexdigest()[:16]


def state_name(state: int) -> str:
    return STATE_NAMES.get(int(state), str(int(state)))


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(path: Path) -> pd.DataFrame:
    required = {"state_fips", "year", "age", "educ", "nchild", "contpov"}
    data = pd.read_csv(path)
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    data = data[
        data["state_fips"].isin(STATES)
        & data["year"].isin(PRE_YEARS)
        & data["educ"].isin([0, 1, 2])
        & data["nchild"].isin([0, 1, 2])
    ].copy()
    data = data.dropna(subset=["state_fips", "year", "age", "educ", "nchild", "contpov"])
    for column in ["state_fips", "year", "educ", "nchild"]:
        data[column] = data[column].astype(int)
    return data.reset_index(drop=True)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    adjusted = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    if not finite.any():
        return adjusted
    finite_values = values[finite]
    order = np.argsort(finite_values)
    m = finite_values.size
    ordered_adjusted = np.maximum.accumulate((m - np.arange(m)) * finite_values[order])
    ordered_adjusted = np.minimum(ordered_adjusted, 1.0)
    local = np.empty(m, dtype=float)
    local[order] = ordered_adjusted
    adjusted[finite] = local
    return adjusted


def symmetric_bootstrap_summary(draws: np.ndarray, alpha: float = 0.05) -> dict[str, float | int]:
    draws = np.asarray(draws, dtype=float)
    estimate = float(draws[0])
    boot = draws[1:]
    boot = boot[np.isfinite(boot)]
    if boot.size < 2 or not np.isfinite(estimate):
        return {
            "estimate": estimate,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "root_critical_value": np.nan,
            "p_value_scalar": np.nan,
            "valid_bootstrap": int(boot.size),
        }
    roots = boot - estimate
    se = float(np.std(boot, ddof=1))
    critical = float(np.quantile(np.abs(roots), 1.0 - alpha))
    p_value = float((1.0 + np.sum(np.abs(roots) >= abs(estimate))) / (boot.size + 1.0))
    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - critical,
        "ci_high": estimate + critical,
        "root_critical_value": critical,
        "p_value_scalar": p_value,
        "valid_bootstrap": int(boot.size),
    }


def joint_bootstrap_test(moment_draws: np.ndarray) -> dict[str, float | int]:
    moment_draws = np.asarray(moment_draws, dtype=float)
    estimate = moment_draws[0]
    roots = moment_draws[1:] - estimate[None, :]
    finite = np.all(np.isfinite(roots), axis=1) & np.all(np.isfinite(estimate))
    roots = roots[finite]
    if roots.shape[0] < 2:
        return {"wald_stat": np.nan, "wald_rank": 0, "p_value_joint": np.nan, "valid_bootstrap_joint": int(roots.shape[0])}
    covariance = np.cov(roots, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    scale = float(np.trace(covariance)) / max(covariance.shape[0], 1)
    ridge = max(scale, 1.0e-12) * 1.0e-10
    covariance_regularized = covariance + ridge * np.eye(covariance.shape[0])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        inverse = np.linalg.pinv(covariance_regularized, rcond=1.0e-10)
        rank = int(np.linalg.matrix_rank(covariance_regularized, tol=max(ridge, 1.0e-12)))
    observed = float(estimate @ inverse @ estimate)
    bootstrap_stats = np.einsum("bi,ij,bj->b", roots, inverse, roots, optimize=True)
    p_value = float((1.0 + np.sum(bootstrap_stats >= observed)) / (roots.shape[0] + 1.0))
    return {
        "wald_stat": observed,
        "wald_rank": rank,
        "p_value_joint": p_value,
        "valid_bootstrap_joint": int(roots.shape[0]),
    }


def project_simplex_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sorted_values = np.sort(values, axis=1)[:, ::-1]
    cumulative = np.cumsum(sorted_values, axis=1) - 1.0
    indices = np.arange(1, values.shape[1] + 1, dtype=float)
    positive = sorted_values - cumulative / indices[None, :] > 0.0
    rho = np.maximum(positive.sum(axis=1) - 1, 0)
    theta = cumulative[np.arange(values.shape[0]), rho] / (rho + 1.0)
    projected = np.maximum(values - theta[:, None], 0.0)
    normalizer = projected.sum(axis=1, keepdims=True)
    return np.divide(projected, normalizer, out=np.full_like(projected, 1.0 / values.shape[1]), where=normalizer > 0.0)


def batch_simplex_least_squares(
    design: np.ndarray,
    target: np.ndarray,
    ridge: float,
    iterations: int,
) -> np.ndarray:
    """Solve many small simplex least-squares problems by active-set enumeration.

    ``iterations`` is retained in the public call signature and run metadata for
    backward-compatible checkpoints; the exact active-set algorithm does not use
    an iteration limit.
    """

    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    problems, _, n_weights = design.shape
    del iterations
    best_weights = np.full((problems, n_weights), 1.0 / n_weights, dtype=float)
    best_objective = np.full(problems, np.inf, dtype=float)

    for mask in range(1, 1 << n_weights):
        active = np.asarray([index for index in range(n_weights) if (mask >> index) & 1], dtype=int)
        selected = design[:, :, active]
        active_size = active.size
        gram = np.einsum("ptk,ptl->pkl", selected, selected, optimize=True)
        gram += ridge * np.eye(active_size)[None, :, :]
        rhs = np.einsum("ptk,pt->pk", selected, target, optimize=True)
        ones = np.ones((problems, active_size), dtype=float)
        try:
            inverse_rhs = np.linalg.solve(gram, rhs[:, :, None])[:, :, 0]
            inverse_one = np.linalg.solve(gram, ones[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(gram, rcond=1.0e-12)
            inverse_rhs = np.einsum("pkl,pl->pk", inverse, rhs, optimize=True)
            inverse_one = np.einsum("pkl,pl->pk", inverse, ones, optimize=True)
        denominator = np.sum(inverse_one, axis=1)
        multiplier = np.divide(
            np.sum(inverse_rhs, axis=1) - 1.0,
            denominator,
            out=np.zeros(problems),
            where=np.abs(denominator) > 1.0e-14,
        )
        solution = inverse_rhs - inverse_one * multiplier[:, None]
        feasible = np.all(solution >= -1.0e-8, axis=1) & np.all(np.isfinite(solution), axis=1)
        candidate = np.zeros((problems, n_weights), dtype=float)
        candidate[:, active] = np.maximum(solution, 0.0)
        normalizer = candidate.sum(axis=1, keepdims=True)
        candidate = np.divide(
            candidate,
            normalizer,
            out=np.full_like(candidate, 1.0 / n_weights),
            where=normalizer > 0.0,
        )
        residual = np.einsum("ptk,pk->pt", design, candidate, optimize=True) - target
        objective = np.sum(residual * residual, axis=1) + ridge * np.sum(candidate * candidate, axis=1)
        improve = feasible & np.isfinite(objective) & (objective < best_objective)
        best_objective[improve] = objective[improve]
        best_weights[improve] = candidate[improve]

    return best_weights


def local_linear_mu_cube(
    ages: np.ndarray,
    outcomes: np.ndarray,
    years: np.ndarray,
    states: np.ndarray,
    folds: np.ndarray,
    multipliers: np.ndarray,
    age_grid: np.ndarray,
    bandwidth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return draw-fold-state-year-age estimates of mu=E[1{T=t}Y|G,X]."""

    draws = multipliers.shape[1]
    n_folds = 2
    n_states = STATES.size
    n_years = PRE_YEARS.size
    n_age = age_grid.size
    mu = np.full((draws, n_folds, n_states, n_years, n_age), np.nan, dtype=float)
    lambdas = np.full((draws, n_folds, n_years), np.nan, dtype=float)

    for fold in range(n_folds):
        train = folds != fold
        train_weights = multipliers[train]
        train_years = years[train]
        denominator = train_weights.sum(axis=0)
        for year_index, year in enumerate(PRE_YEARS):
            numerator = train_weights[train_years == year].sum(axis=0)
            lambdas[:, fold, year_index] = np.divide(
                numerator,
                denominator,
                out=np.full(draws, np.nan),
                where=denominator > 0.0,
            )

        for state_index, state in enumerate(STATES):
            selected = train & (states == state)
            selected_ages = ages[selected]
            selected_outcomes = outcomes[selected]
            selected_years = years[selected]
            selected_weights = multipliers[selected]
            if selected_ages.size == 0:
                continue
            centered = selected_ages[None, :] - age_grid[:, None]
            u = centered / bandwidth
            kernel = 0.75 * (1.0 - u * u)
            kernel[np.abs(u) > 1.0] = 0.0
            kernel_z = kernel * centered
            kernel_z2 = kernel_z * centered
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                s0 = kernel @ selected_weights
                s1 = kernel_z @ selected_weights
                s2 = kernel_z2 @ selected_weights
            determinant = (s0 + RIDGE_LOCAL) * (s2 + RIDGE_LOCAL) - s1 * s1
            for year_index, year in enumerate(PRE_YEARS):
                response = selected_outcomes * (selected_years == year)
                weighted_response = selected_weights * response[:, None]
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    t0 = kernel @ weighted_response
                    t1 = kernel_z @ weighted_response
                numerator = (s2 + RIDGE_LOCAL) * t0 - s1 * t1
                intercept = np.divide(
                    numerator,
                    determinant,
                    out=np.zeros_like(numerator),
                    where=np.abs(determinant) > 1.0e-12,
                )
                mu[:, fold, state_index, year_index, :] = intercept.T
    return mu, lambdas


def quartile_moments(values: np.ndarray, grid_quantiles: np.ndarray) -> np.ndarray:
    """Average draw-fold-age values within four fixed treated-age quantile bins."""

    values = np.asarray(values, dtype=float)
    bins = np.minimum((grid_quantiles * 4.0).astype(int), 3)
    output = np.full((values.shape[0], 4), np.nan)
    for quartile in range(4):
        selected = bins == quartile
        output[:, quartile] = np.nanmean(values[:, :, selected], axis=(1, 2))
    return output


def outcome_scale(data: pd.DataFrame, age_low: float, age_high: float) -> float:
    selected = data[
        (data["state_fips"] == TREATED_STATE)
        & (data["age"] > age_low)
        & (data["age"] < age_high)
    ]["contpov"].to_numpy(dtype=float)
    return float(np.std(selected, ddof=1)) if selected.size > 1 else np.nan


def cell_count_columns(data: pd.DataFrame) -> dict[str, int]:
    counts = data.groupby(["state_fips", "year"]).size()
    output: dict[str, int] = {}
    for state in STATES:
        for year in PRE_YEARS:
            output[f"n_{int(state)}_{int(year)}"] = int(counts.get((int(state), int(year)), 0))
    return output


def exact_simplex_objective(design: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    residual = design @ weights - target
    return float(residual @ residual + RIDGE_SC * (weights @ weights))


def subgroup_diagnostics(
    data_records: list[dict[str, Any]],
    educ: int,
    nchild: int,
    config_dict: dict[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    config = RunConfig(**config_dict)
    data = pd.DataFrame.from_records(data_records)
    data = data[(data["educ"] == educ) & (data["nchild"] == nchild)].copy().reset_index(drop=True)
    if data.empty:
        raise ValueError(f"No observations for education={educ}, children={nchild}")

    observed_cells = set(zip(data["state_fips"], data["year"]))
    required_cells = {(int(state), int(year)) for state in STATES for year in PRE_YEARS}
    missing_cells = sorted(required_cells.difference(observed_cells))
    if missing_cells:
        raise ValueError(f"Missing state-year cells in subgroup ({educ},{nchild}): {missing_cells}")

    ages = data["age"].to_numpy(dtype=float)
    outcomes = data["contpov"].to_numpy(dtype=float)
    years = data["year"].to_numpy(dtype=int)
    states = data["state_fips"].to_numpy(dtype=int)
    n = data.shape[0]
    age_low, age_high = np.quantile(ages, [config.trim_low, config.trim_high])
    treated_ages = ages[(states == TREATED_STATE) & (ages > age_low) & (ages < age_high)]
    if treated_ages.size < config.age_grid_size:
        raise ValueError(f"Too few treated ages in subgroup ({educ},{nchild})")
    grid_quantiles = (np.arange(config.age_grid_size, dtype=float) + 0.5) / config.age_grid_size
    age_grid = np.quantile(treated_ages, grid_quantiles)
    bandwidth = float(config.bandwidth_constant * n ** config.bandwidth_power)
    scale_y = outcome_scale(data, age_low, age_high)

    rng = np.random.default_rng(config.seed + 10000 * educ + 1000 * nchild)
    folds = rng.integers(0, 2, size=n, endpoint=False)
    for state in STATES:
        for year in PRE_YEARS:
            cell = np.where((states == state) & (years == year))[0]
            if cell.size >= 2 and np.unique(folds[cell]).size < 2:
                folds[cell[0]] = 1 - folds[cell[0]]
    multipliers = np.ones((n, config.bootstrap + 1), dtype=float)
    if config.bootstrap > 0:
        multipliers[:, 1:] = rng.exponential(1.0, size=(n, config.bootstrap))

    means_start = time.perf_counter()
    mu, lambdas = local_linear_mu_cube(
        ages=ages,
        outcomes=outcomes,
        years=years,
        states=states,
        folds=folds,
        multipliers=multipliers,
        age_grid=age_grid,
        bandwidth=bandwidth,
    )
    means_seconds = time.perf_counter() - means_start
    m = np.divide(
        mu,
        lambdas[:, :, None, :, None],
        out=np.full_like(mu, np.nan),
        where=lambdas[:, :, None, :, None] > 0.0,
    )

    counts = cell_count_columns(data)
    base = {
        "educ": educ,
        "nchild": nchild,
        "n_pre": int(n),
        "n_states": int(STATES.size),
        "n_donors": int(DONORS.size),
        "years": "1998, 1999, 2000, 2001, 2002",
        "donors": ", ".join(state_name(state) for state in DONORS),
        "age_low": float(age_low),
        "age_high": float(age_high),
        "age_grid_size": int(age_grid.size),
        "bandwidth": bandwidth,
        "bandwidth_rule": f"{config.bandwidth_constant:g}*n^({config.bandwidth_power:.12g})",
        "bootstrap_requested": int(config.bootstrap),
        "seed": int(config.seed + 10000 * educ + 1000 * nchild),
        **counts,
    }

    pt_start = time.perf_counter()
    pt_records: list[dict[str, Any]] = []
    pt_draw_store: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for transition_index in range(PRE_YEARS.size - 1):
        pre_year = int(PRE_YEARS[transition_index])
        pseudo_post_year = int(PRE_YEARS[transition_index + 1])
        treated_trend = m[:, :, 0, transition_index + 1, :] - m[:, :, 0, transition_index, :]
        for donor_index, donor in enumerate(DONORS, start=1):
            donor_trend = m[:, :, donor_index, transition_index + 1, :] - m[:, :, donor_index, transition_index, :]
            gap = treated_trend - donor_trend
            scalar_draws = np.nanmean(gap, axis=(1, 2))
            moment_draws = quartile_moments(gap, grid_quantiles)
            scalar = symmetric_bootstrap_summary(scalar_draws)
            joint = joint_bootstrap_test(moment_draws)
            record = {
                **base,
                "pre_year": pre_year,
                "pseudo_post_year": pseudo_post_year,
                "donor_state": int(donor),
                "donor": state_name(int(donor)),
                **scalar,
                **joint,
            }
            for quartile in range(4):
                record[f"age_q{quartile + 1}_gap"] = float(moment_draws[0, quartile])
            pt_records.append(record)
            pt_draw_store[(transition_index, donor_index - 1)] = (scalar_draws, moment_draws)

    pt_transition_records: list[dict[str, Any]] = []
    for transition_index in range(PRE_YEARS.size - 1):
        scalar_stack = np.column_stack([pt_draw_store[(transition_index, donor)][0] for donor in range(DONORS.size)])
        equal_weight_draws = np.nanmean(scalar_stack, axis=1)
        moment_stack = np.concatenate([pt_draw_store[(transition_index, donor)][1] for donor in range(DONORS.size)], axis=1)
        record = {
            **base,
            "pre_year": int(PRE_YEARS[transition_index]),
            "pseudo_post_year": int(PRE_YEARS[transition_index + 1]),
            **symmetric_bootstrap_summary(equal_weight_draws),
            **joint_bootstrap_test(moment_stack),
        }
        pt_transition_records.append(record)
    pt_seconds = time.perf_counter() - pt_start

    sc_start = time.perf_counter()
    design_all = np.moveaxis(mu[:, :, 1:, :, :], 2, -1)
    design_all = np.moveaxis(design_all, 3, 2)
    target_all = mu[:, :, 0, :, :].transpose(0, 1, 3, 2)
    n_draws, n_folds, n_age, _, n_donors = design_all.shape
    flat_design = design_all.reshape(-1, PRE_YEARS.size, n_donors)
    flat_target = target_all.reshape(-1, PRE_YEARS.size)
    full_weights = batch_simplex_least_squares(flat_design, flat_target, RIDGE_SC, config.simplex_iterations)
    full_weights = full_weights.reshape(n_draws, n_folds, n_age, n_donors)

    loo_weights = np.empty((n_draws, n_folds, n_age, PRE_YEARS.size, n_donors), dtype=float)
    for heldout in range(PRE_YEARS.size):
        keep_years = np.arange(PRE_YEARS.size) != heldout
        fitted = batch_simplex_least_squares(
            flat_design[:, keep_years, :],
            flat_target[:, keep_years],
            RIDGE_SC,
            config.simplex_iterations,
        )
        loo_weights[:, :, :, heldout, :] = fitted.reshape(n_draws, n_folds, n_age, n_donors)

    mu_treated = mu[:, :, 0, :, :].transpose(0, 1, 3, 2)
    mu_donors = design_all
    full_synthetic_mu = np.einsum("dfatk,dfak->dfat", mu_donors, full_weights, optimize=True)
    lambda_age = lambdas[:, :, None, :]
    full_residual = np.divide(
        mu_treated - full_synthetic_mu,
        lambda_age,
        out=np.full_like(mu_treated, np.nan),
        where=lambda_age > 0.0,
    )

    loo_residual = np.empty_like(full_residual)
    for heldout in range(PRE_YEARS.size):
        donor_mu = mu_donors[:, :, :, heldout, :]
        synthetic = np.einsum("dfak,dfak->dfa", donor_mu, loo_weights[:, :, :, heldout, :], optimize=True)
        loo_residual[:, :, :, heldout] = np.divide(
            mu_treated[:, :, :, heldout] - synthetic,
            lambdas[:, :, heldout][:, :, None],
            out=np.full_like(synthetic, np.nan),
            where=lambdas[:, :, heldout][:, :, None] > 0.0,
        )

    in_rmspe_draws = np.sqrt(np.nanmean(full_residual * full_residual, axis=(1, 2, 3)))
    loo_rmspe_draws = np.sqrt(np.nanmean(loo_residual * loo_residual, axis=(1, 2, 3)))
    in_summary = symmetric_bootstrap_summary(in_rmspe_draws)
    loo_summary = symmetric_bootstrap_summary(loo_rmspe_draws)
    prospective_gap = loo_residual[:, :, :, -1]
    prospective_draws = np.nanmean(prospective_gap, axis=(1, 2))
    prospective_moments = quartile_moments(prospective_gap, grid_quantiles)
    prospective_summary = symmetric_bootstrap_summary(prospective_draws)
    prospective_joint = joint_bootstrap_test(prospective_moments)

    point_full_abs = np.abs(full_residual[0])
    point_loo_abs = np.abs(loo_residual[0])
    point_weights = full_weights[0]
    effective_donors = 1.0 / np.sum(point_weights * point_weights, axis=-1)
    average_weights = np.mean(point_weights, axis=(0, 1))
    sc_summary_record = {
        **base,
        "outcome_scale_sd_alaska_pre": scale_y,
        "in_sample_rmspe": float(in_rmspe_draws[0]),
        "in_sample_rmspe_se": float(in_summary["se"]),
        "in_sample_normalized_rmspe": float(in_rmspe_draws[0] / scale_y) if scale_y > 0 else np.nan,
        "in_sample_max_abs_residual": float(np.nanmax(point_full_abs)),
        "in_sample_p95_abs_residual": float(np.nanquantile(point_full_abs, 0.95)),
        "loo_rmspe": float(loo_rmspe_draws[0]),
        "loo_rmspe_se": float(loo_summary["se"]),
        "loo_normalized_rmspe": float(loo_rmspe_draws[0] / scale_y) if scale_y > 0 else np.nan,
        "loo_max_abs_residual": float(np.nanmax(point_loo_abs)),
        "loo_p95_abs_residual": float(np.nanquantile(point_loo_abs, 0.95)),
        "prospective_fit_years": "1998, 1999, 2000, 2001",
        "prospective_pseudo_post_year": 2002,
        "prospective_2002_gap": float(prospective_summary["estimate"]),
        "prospective_2002_se": float(prospective_summary["se"]),
        "prospective_2002_ci_low": float(prospective_summary["ci_low"]),
        "prospective_2002_ci_high": float(prospective_summary["ci_high"]),
        "prospective_2002_scalar_p": float(prospective_summary["p_value_scalar"]),
        "prospective_2002_joint_wald": float(prospective_joint["wald_stat"]),
        "prospective_2002_joint_rank": int(prospective_joint["wald_rank"]),
        "prospective_2002_joint_p_raw": float(prospective_joint["p_value_joint"]),
        "valid_bootstrap": int(min(in_summary["valid_bootstrap"], loo_summary["valid_bootstrap"], prospective_summary["valid_bootstrap"])),
        "mean_effective_donors": float(np.mean(effective_donors)),
        "median_effective_donors": float(np.median(effective_donors)),
        "weight_exact_zero_share": float(np.mean(point_weights <= 1.0e-8)),
        "weight_near_zero_share": float(np.mean(point_weights < BOUNDARY_TOL)),
        "weight_near_one_share": float(np.mean(point_weights > 1.0 - BOUNDARY_TOL)),
        "max_weight": float(np.max(point_weights)),
        "simplex_min_weight": float(np.min(full_weights)),
        "simplex_max_sum_error": float(np.max(np.abs(full_weights.sum(axis=-1) - 1.0))),
    }
    for donor_index, donor in enumerate(DONORS):
        sc_summary_record[f"average_weight_{int(donor)}"] = float(average_weights[donor_index])

    residual_records: list[dict[str, Any]] = []
    weight_records: list[dict[str, Any]] = []
    point_m = m[0]
    for fold in range(n_folds):
        for age_index, age in enumerate(age_grid):
            for year_index, year in enumerate(PRE_YEARS):
                full_w = full_weights[0, fold, age_index]
                treated_mean = point_m[fold, 0, year_index, age_index]
                donor_means = point_m[fold, 1:, year_index, age_index]
                residual_records.append(
                    {
                        **base,
                        "fold": fold,
                        "age_grid_index": age_index,
                        "age_quantile": float(grid_quantiles[age_index]),
                        "age": float(age),
                        "fit_type": "all_pre_in_sample",
                        "heldout_year": np.nan,
                        "year": int(year),
                        "treated_conditional_mean": float(treated_mean),
                        "synthetic_conditional_mean": float(donor_means @ full_w),
                        "residual": float(full_residual[0, fold, age_index, year_index]),
                    }
                )
                loo_w = loo_weights[0, fold, age_index, year_index]
                residual_records.append(
                    {
                        **base,
                        "fold": fold,
                        "age_grid_index": age_index,
                        "age_quantile": float(grid_quantiles[age_index]),
                        "age": float(age),
                        "fit_type": "leave_one_pre_year_out",
                        "heldout_year": int(year),
                        "year": int(year),
                        "treated_conditional_mean": float(treated_mean),
                        "synthetic_conditional_mean": float(donor_means @ loo_w),
                        "residual": float(loo_residual[0, fold, age_index, year_index]),
                    }
                )

            for fit_type, heldout_year, weights_for_fit in [
                ("all_pre_in_sample", np.nan, full_weights[0, fold, age_index]),
                *[
                    ("leave_one_pre_year_out", int(PRE_YEARS[heldout]), loo_weights[0, fold, age_index, heldout])
                    for heldout in range(PRE_YEARS.size)
                ],
            ]:
                for donor_index, donor in enumerate(DONORS):
                    weight = float(weights_for_fit[donor_index])
                    weight_records.append(
                        {
                            **base,
                            "fold": fold,
                            "age_grid_index": age_index,
                            "age_quantile": float(grid_quantiles[age_index]),
                            "age": float(age),
                            "fit_type": fit_type,
                            "heldout_year": heldout_year,
                            "donor_state": int(donor),
                            "donor": state_name(int(donor)),
                            "weight": weight,
                            "exact_zero": bool(weight <= 1.0e-8),
                            "near_zero": bool(weight < BOUNDARY_TOL),
                            "near_one": bool(weight > 1.0 - BOUNDARY_TOL),
                        }
                    )

    objective_checks = []
    for fold in range(n_folds):
        for age_index in range(n_age):
            index = ((0 * n_folds + fold) * n_age + age_index)
            objective_checks.append(exact_simplex_objective(flat_design[index], flat_target[index], full_weights[0, fold, age_index]))
    sc_summary_record["max_point_simplex_objective"] = float(np.max(objective_checks))
    sc_seconds = time.perf_counter() - sc_start

    timing_record = {
        "task": "06_pt_sc_diagnostics",
        "educ": educ,
        "nchild": nchild,
        "n_pre": int(n),
        "bootstrap_requested": int(config.bootstrap),
        "age_grid_size": int(age_grid.size),
        "conditional_mean_seconds": float(means_seconds),
        "pt_seconds": float(pt_seconds),
        "sc_seconds": float(sc_seconds),
        "total_subgroup_seconds": float(time.perf_counter() - start),
        "timing_status": "measured",
        "script_version": SCRIPT_VERSION,
        "config_signature": config.signature(),
    }
    return {
        "config_signature": config.signature(),
        "pt_records": pt_records,
        "pt_transition_records": pt_transition_records,
        "sc_summary_record": sc_summary_record,
        "sc_residual_records": residual_records,
        "sc_weight_records": weight_records,
        "timing_record": timing_record,
    }


def add_multiplicity_adjustments(
    pt: pd.DataFrame,
    pt_transition: pd.DataFrame,
    sc: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pt = pt.copy()
    pt["p_value_holm_within_subgroup_16"] = np.nan
    for _, index in pt.groupby(["educ", "nchild"]).groups.items():
        index = list(index)
        pt.loc[index, "p_value_holm_within_subgroup_16"] = holm_adjust(pt.loc[index, "p_value_joint"])
    pt["reject_raw_05"] = pt["p_value_joint"] < 0.05
    pt["reject_holm_05"] = pt["p_value_holm_within_subgroup_16"] < 0.05

    pt_transition = pt_transition.copy()
    pt_transition["p_value_holm_within_subgroup_4"] = np.nan
    for _, index in pt_transition.groupby(["educ", "nchild"]).groups.items():
        index = list(index)
        pt_transition.loc[index, "p_value_holm_within_subgroup_4"] = holm_adjust(pt_transition.loc[index, "p_value_joint"])
    pt_transition["reject_raw_05"] = pt_transition["p_value_joint"] < 0.05
    pt_transition["reject_holm_05"] = pt_transition["p_value_holm_within_subgroup_4"] < 0.05

    sc = sc.copy()
    sc["prospective_2002_joint_p_holm_9"] = holm_adjust(sc["prospective_2002_joint_p_raw"])
    sc["prospective_2002_reject_raw_05"] = sc["prospective_2002_joint_p_raw"] < 0.05
    sc["prospective_2002_reject_holm_05"] = sc["prospective_2002_joint_p_holm_9"] < 0.05
    return pt, pt_transition, sc


def pt_summary_table(pt: pd.DataFrame, transition: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (educ, nchild), group in pt.groupby(["educ", "nchild"], sort=True):
        transition_group = transition[(transition["educ"] == educ) & (transition["nchild"] == nchild)]
        records.append(
            {
                "educ": int(educ),
                "nchild": int(nchild),
                "donor_transition_tests": int(group.shape[0]),
                "donor_transition_raw_rejections": int(group["reject_raw_05"].sum()),
                "donor_transition_holm_rejections": int(group["reject_holm_05"].sum()),
                "transition_omnibus_tests": int(transition_group.shape[0]),
                "transition_omnibus_raw_rejections": int(transition_group["reject_raw_05"].sum()),
                "transition_omnibus_holm_rejections": int(transition_group["reject_holm_05"].sum()),
                "max_abs_average_placebo": float(group["estimate"].abs().max()),
                "min_raw_joint_p": float(group["p_value_joint"].min()),
                "min_holm_joint_p": float(group["p_value_holm_within_subgroup_16"].min()),
            }
        )
    summary = pd.DataFrame(records)
    total = {
        "educ": "All",
        "nchild": "All",
        "donor_transition_tests": int(pt.shape[0]),
        "donor_transition_raw_rejections": int(pt["reject_raw_05"].sum()),
        "donor_transition_holm_rejections": int(pt["reject_holm_05"].sum()),
        "transition_omnibus_tests": int(transition.shape[0]),
        "transition_omnibus_raw_rejections": int(transition["reject_raw_05"].sum()),
        "transition_omnibus_holm_rejections": int(transition["reject_holm_05"].sum()),
        "max_abs_average_placebo": float(pt["estimate"].abs().max()),
        "min_raw_joint_p": float(pt["p_value_joint"].min()),
        "min_holm_joint_p": float(pt["p_value_holm_within_subgroup_16"].min()),
    }
    return pd.concat([summary, pd.DataFrame([total])], ignore_index=True)


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_").replace("%", "\\%")


def write_pt_latex(summary: pd.DataFrame, bootstrap: int, path: Path) -> None:
    body = []
    for _, row in summary[summary["educ"] != "All"].iterrows():
        body.append(
            f"{int(row['educ'])} & {int(row['nchild'])} & "
            f"{int(row['donor_transition_raw_rejections'])}/16 & {int(row['donor_transition_holm_rejections'])}/16 & "
            f"{int(row['transition_omnibus_raw_rejections'])}/4 & {int(row['transition_omnibus_holm_rejections'])}/4 & "
            f"{row['max_abs_average_placebo']:.3f} \\\\"
        )
    total = summary[summary["educ"] == "All"].iloc[0]
    body.append("\\midrule")
    body.append(
        f"All & All & {int(total['donor_transition_raw_rejections'])}/{int(total['donor_transition_tests'])} & "
        f"{int(total['donor_transition_holm_rejections'])}/{int(total['donor_transition_tests'])} & "
        f"{int(total['transition_omnibus_raw_rejections'])}/{int(total['transition_omnibus_tests'])} & "
        f"{int(total['transition_omnibus_holm_rejections'])}/{int(total['transition_omnibus_tests'])} & "
        f"{total['max_abs_average_placebo']:.3f} \\\\"
    )
    latex = f"""\\begin{{table}}[htbp!]
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{2.5pt}}
\\begin{{tabular}}{{ccccccc}}
\\toprule
Education & Children & Donor raw & Donor Holm & Omnibus raw & Omnibus Holm & Max. $|\\widehat{{\\delta}}|$ \\\\
\\midrule
{chr(10).join(body)}
\\bottomrule
\\end{{tabular}}
\\caption{{Pre-treatment diagnostics for conditional parallel trends}}
\\label{{tab:empirical-pt-diagnostics}}
\\caption*{{\\footnotesize
The table reports pre-treatment placebo diagnostics using only 1998--2002 data and the exact quantile top-four donor set reported by Gunsilius: Virginia (0.11), New Hampshire (0.11), Maryland (0.09), and Utah (0.07). For each subgroup, donor-specific tests cover four donors and four adjacent pre-period transitions. Each test jointly assesses four fixed age-quartile moments of the conditional trend gap. Conditional means use the estimator-compatible construction $\\widehat{{m}}_{{g,t}}(x)=\\widehat{{\\mu}}_{{g,t}}(x)/\\widehat{{\\lambda}}_t$, where $\\widehat{{\\mu}}_{{g,t}}(x)$ smooths $Y1\\{{T=t\\}}$ within group $g$. Omnibus tests jointly assess all donors within each transition. ``Raw'' reports unadjusted 5\\% rejections and ``Holm'' applies Holm's correction within each subgroup. Max. $|\\widehat{{\\delta}}|$ is the largest absolute age-averaged donor-specific placebo estimate. Inference uses {bootstrap} exponential-multiplier draws with all conditional means and period shares re-estimated. Rejections are diagnostic evidence against stable pre-treatment trends; nonrejection does not establish the post-treatment parallel-trends condition.}}
\\end{{table}}
"""
    path.write_text(latex, encoding="utf-8")


def write_sc_latex(sc: pd.DataFrame, bootstrap: int, path: Path) -> None:
    rows = []
    for _, row in sc.sort_values(["educ", "nchild"]).iterrows():
        rows.append(
            f"{int(row['educ'])} & {int(row['nchild'])} & {row['in_sample_rmspe']:.3f} & "
            f"{row['in_sample_normalized_rmspe']:.3f} & {row['loo_rmspe']:.3f} & "
            f"{row['loo_normalized_rmspe']:.3f} & {row['prospective_2002_gap']:.3f} & "
            f"{row['prospective_2002_joint_p_raw']:.3f} & {row['prospective_2002_joint_p_holm_9']:.3f} & "
            f"{row['mean_effective_donors']:.2f} & {100.0 * row['weight_near_zero_share']:.1f} \\\\"
        )
    latex = f"""\\begin{{table}}[htbp!]
\\centering
\\tiny
\\setlength{{\\tabcolsep}}{{1.5pt}}
\\begin{{tabular}}{{ccccccccccc}}
\\toprule
Education & Children & In RMSPE & In/SD & LOO RMSPE & LOO/SD & 2002 gap & Raw $p$ & Holm $p$ & Eff. donors & Boundary (\\%) \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{Pre-treatment diagnostics for the conditional synthetic-control relation}}
\\label{{tab:empirical-sc-diagnostics}}
\\caption*{{\\footnotesize
The table evaluates the covariate-conditional synthetic-control relation using only 1998--2002 data and the exact quantile top-four donor set reported by Gunsilius: Virginia (0.11), New Hampshire (0.11), Maryland (0.09), and Utah (0.07). It uses the same lambda-normalized conditional-mean construction as the estimator: $\\widehat{{m}}_{{g,t}}(x)=\\widehat{{\\mu}}_{{g,t}}(x)/\\widehat{{\\lambda}}_t$. ``In RMSPE'' fits the same simplex weights on all five pre-treatment years. ``LOO RMSPE'' leaves each pre-treatment year out when fitting the weights and predicts that year's conditional mean. Both are expressed in outcome units; columns labeled ``/SD'' normalize by the pre-treatment Alaska outcome standard deviation. The 2002 gap is a prospective placebo fitted on 1998--2001, the earliest rank-valid pseudo-post exercise with four donors. Its raw multiplier-bootstrap $p$-value jointly assesses four age-quartile gaps, and Holm adjusts across the nine subgroups. Eff. donors is $1/\\sum_g w_g^2$, averaged over ages and folds. Boundary is the percentage of fitted weights below 0.01. All bootstrap calculations use {bootstrap} draws and re-estimate the conditional means, period shares, and weights. These are fit diagnostics, not tests that can establish the post-treatment SC assumption.}}
\\end{{table}}
"""
    path.write_text(latex, encoding="utf-8")


def copy_legacy_auxiliary() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    legacy_root = SCRIPT_DIR / "excluded_legacy_artifacts"
    pooled_path = legacy_root / "top4_pt_placebo_checks_bootstrap500.csv"
    age_path = legacy_root / "top4_pretrend_age_cell_gap_changes_bootstrap500.csv"
    provenance: list[dict[str, Any]] = []
    pooled = pd.DataFrame()
    age = pd.DataFrame()
    if pooled_path.exists():
        pooled = pd.read_csv(pooled_path)
        pooled = pooled[
            (pooled["donor_set"] == "pool_a_va_nh_md_ut")
            & (pooled["method"] == "pooled_pt_dr")
        ].copy()
        pooled["p_value_normal"] = [
            math.erfc(abs(value) / math.sqrt(2.0))
            for value in pooled["z_stat"].to_numpy(dtype=float)
        ]
        pooled["p_value_holm_within_subgroup_4"] = np.nan
        for _, index in pooled.groupby(["educ", "nchild"]).groups.items():
            index = list(index)
            pooled.loc[index, "p_value_holm_within_subgroup_4"] = holm_adjust(pooled.loc[index, "p_value_normal"])
        pooled["reject_raw_05"] = pooled["p_value_normal"] < 0.05
        pooled["reject_holm_05"] = pooled["p_value_holm_within_subgroup_4"] < 0.05
        provenance.append(
            {
                "artifact": "auxiliary_legacy_pooled_pt.csv",
                "status": "reused_filtered",
                "source": str(pooled_path.resolve()),
                "scope": "Quantile-top-four pooled-control PT-DR pre-period placebos; auxiliary because pooling does not test every donor-specific PT restriction.",
            }
        )
    if age_path.exists():
        age = pd.read_csv(age_path)
        age = age[age["donor_set"] == "pool_a_va_nh_md_ut"].copy()
        provenance.append(
            {
                "artifact": "auxiliary_legacy_age_cell_pt.csv",
                "status": "reused_filtered",
                "source": str(age_path.resolve()),
                "scope": "Quantile-top-four pre-period age-tertile gap changes; auxiliary unconditional/bin diagnostic.",
            }
        )
    return pooled, age, provenance


def write_auxiliary_latex(pooled: pd.DataFrame, age: pd.DataFrame, path: Path) -> None:
    pooled_n = int(pooled.shape[0])
    pooled_raw = int(pooled["reject_raw_05"].sum()) if not pooled.empty else 0
    pooled_holm = int(pooled["reject_holm_05"].sum()) if not pooled.empty else 0
    eligible = age[age["change_eligible"].astype(bool)] if not age.empty else age
    age_n = int(eligible.shape[0])
    age_raw = int(eligible["significant_5pct_normal"].astype(bool).sum()) if not eligible.empty else 0
    latex = f"""\\begin{{table}}[htbp!]
\\centering
\\small
\\begin{{tabular}}{{cccc}}
\\toprule
Auxiliary diagnostic & Tests & Raw 5\\% rejections & Holm rejections \\\\
\\midrule
Pooled PT--DR adjacent-year placebo & {pooled_n} & {pooled_raw} & {pooled_holm} \\\\
Age-tertile pooled gap changes & {age_n} & {age_raw} & -- \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Previously computed pre-treatment placebo checks}}
\\label{{tab:empirical-pt-legacy-auxiliary}}
\\caption*{{\\footnotesize
The table records exact reusable quantile-top-four results computed previously with 500 multiplier draws. They are auxiliary because pooling donors can conceal donor-specific trend departures and age bins do not reproduce the paper's fully conditional restriction.}}
\\end{{table}}
"""
    path.write_text(latex, encoding="utf-8")


def save_combined(
    results: list[dict[str, Any]],
    config: RunConfig,
    output_dir: Path,
) -> None:
    if not results:
        return
    pt = pd.DataFrame([row for result in results for row in result["pt_records"]])
    pt_transition = pd.DataFrame([row for result in results for row in result["pt_transition_records"]])
    sc = pd.DataFrame([result["sc_summary_record"] for result in results])
    residuals = pd.DataFrame([row for result in results for row in result["sc_residual_records"]])
    weights = pd.DataFrame([row for result in results for row in result["sc_weight_records"]])
    timing_records = []
    for result in results:
        raw_timing = result["timing_record"]
        bootstrap_seconds = float(raw_timing["pt_seconds"] + raw_timing["sc_seconds"])
        bootstrap_requested = int(raw_timing["bootstrap_requested"])
        timing_records.append(
            {
                "task": "06_pt_sc_diagnostics",
                "method": "PT and SC diagnostics",
                "configuration": (
                    f"education={int(raw_timing['educ'])}; children={int(raw_timing['nchild'])}; "
                    f"years=1998-2002; donors=VA-NH-MD-UT; age_grid={int(raw_timing['age_grid_size'])}"
                ),
                "educ": int(raw_timing["educ"]),
                "nchild": int(raw_timing["nchild"]),
                "sample_size": int(raw_timing["n_pre"]),
                "point_estimation_seconds": float(raw_timing["conditional_mean_seconds"]),
                "bootstrap_seconds": bootstrap_seconds,
                "bootstrap_draws_requested": bootstrap_requested,
                "bootstrap_draws_valid": int(result["sc_summary_record"]["valid_bootstrap"]),
                "seconds_per_bootstrap_draw": bootstrap_seconds / max(bootstrap_requested, 1),
                "total_cell_seconds": float(raw_timing["total_subgroup_seconds"]),
                "worker_processes": np.nan,
                "worker_seconds_sum": float(raw_timing["total_subgroup_seconds"]),
                "run_total_wall_seconds": np.nan,
                "timing_status": str(raw_timing["timing_status"]),
                "seed": int(config.seed + 10000 * int(raw_timing["educ"]) + 1000 * int(raw_timing["nchild"])),
                "code_version": SCRIPT_VERSION,
            }
        )
    timing = pd.DataFrame(timing_records)
    pt, pt_transition, sc = add_multiplicity_adjustments(pt, pt_transition, sc)
    summary = pt_summary_table(pt, pt_transition)

    pt.sort_values(["educ", "nchild", "pre_year", "donor_state"]).to_csv(output_dir / "pt_diagnostics_detailed.csv", index=False)
    pt_transition.sort_values(["educ", "nchild", "pre_year"]).to_csv(output_dir / "pt_transition_omnibus.csv", index=False)
    summary.to_csv(output_dir / "pt_diagnostics_summary.csv", index=False)
    sc.sort_values(["educ", "nchild"]).to_csv(output_dir / "sc_diagnostics_summary.csv", index=False)
    residuals.sort_values(["educ", "nchild", "fit_type", "heldout_year", "fold", "age_grid_index", "year"]).to_csv(
        output_dir / "sc_age_year_residuals.csv", index=False
    )
    weights.sort_values(["educ", "nchild", "fit_type", "heldout_year", "fold", "age_grid_index", "donor_state"]).to_csv(
        output_dir / "sc_weights.csv", index=False
    )
    timing.sort_values(["educ", "nchild"]).to_csv(output_dir / "runtime_detailed.csv", index=False)
    existing_full_wall = np.nan
    existing_full_jobs = np.nan
    existing_runtime_path = output_dir / "runtime_summary.csv"
    if existing_runtime_path.exists():
        try:
            existing_runtime = pd.read_csv(existing_runtime_path)
            same_config = (
                not existing_runtime.empty
                and "config_signature" in existing_runtime
                and str(existing_runtime.iloc[0]["config_signature"]) == config.signature()
            )
            if same_config and "driver_wall_seconds_full_run" in existing_runtime:
                existing_full_wall = float(existing_runtime.iloc[0]["driver_wall_seconds_full_run"])
            if same_config and "jobs_full_run" in existing_runtime:
                existing_full_jobs = float(existing_runtime.iloc[0]["jobs_full_run"])
        except (OSError, ValueError, TypeError):
            pass
    runtime_summary = pd.DataFrame(
        [
            {
                "task": "06_pt_sc_diagnostics",
                "subgroups_completed": int(timing.shape[0]),
                "bootstrap_requested": int(config.bootstrap),
                "age_grid_size": int(config.age_grid_size),
                "sum_subgroup_seconds": float(timing["total_cell_seconds"].sum()),
                "median_subgroup_seconds": float(timing["total_cell_seconds"].median()),
                "max_subgroup_seconds": float(timing["total_cell_seconds"].max()),
                "timing_status": "measured",
                "script_version": SCRIPT_VERSION,
                "config_signature": config.signature(),
                "driver_wall_seconds_full_run": existing_full_wall,
                "jobs_full_run": existing_full_jobs,
            }
        ]
    )
    runtime_summary.to_csv(output_dir / "runtime_summary.csv", index=False)


def write_provenance(
    data_path: Path,
    legacy_rows: list[dict[str, Any]],
    config: RunConfig,
    output_dir: Path,
) -> None:
    rows = [
        {
            "artifact": "pt_diagnostics_detailed.csv; pt_transition_omnibus.csv; pt_diagnostics_summary.csv",
            "status": "newly_computed",
            "source": str(data_path.resolve()),
            "scope": "Donor-specific conditional PT diagnostics using only 1998-2002 and quantile top-four donors.",
        },
        {
            "artifact": "sc_diagnostics_summary.csv; sc_age_year_residuals.csv; sc_weights.csv",
            "status": "newly_computed",
            "source": str(data_path.resolve()),
            "scope": "Conditional in-sample, leave-one-pre-year-out, and prospective-2002 SC diagnostics using only 1998-2002.",
        },
        *legacy_rows,
    ]
    provenance = pd.DataFrame(rows)
    provenance["script_version"] = SCRIPT_VERSION
    provenance["config_signature"] = config.signature()
    provenance["source_sha256"] = [
        sha256_file(Path(source)) if Path(source).is_file() else ""
        for source in provenance["source"]
    ]
    provenance["input_data_path"] = str(data_path.resolve())
    provenance["input_data_sha256"] = config.data_sha256
    provenance["task_code_path"] = str(Path(__file__).resolve())
    provenance["task_code_sha256"] = sha256_file(Path(__file__).resolve())
    provenance.to_csv(output_dir / "provenance.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output/checkpoint directory; use a separate directory for subset or smoke runs.",
    )
    parser.add_argument("--bootstrap", "--B", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--age-grid-size", type=int, default=41)
    parser.add_argument("--seed", type=int, default=2026081006)
    parser.add_argument("--trim-low", type=float, default=0.05)
    parser.add_argument("--trim-high", type=float, default=0.95)
    parser.add_argument("--bandwidth-constant", type=float, default=6.25)
    parser.add_argument("--bandwidth-power", type=float, default=1.0 / 5.0 - 1.0 / 3.5)
    parser.add_argument("--simplex-iterations", type=int, default=250, help="Deprecated compatibility option; the exact active-set solver ignores it.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--subgroups", nargs="*", default=None, help="Optional subgroup codes such as 00 12 22.")
    return parser.parse_args()


def selected_subgroups(codes: list[str] | None) -> list[tuple[int, int]]:
    if not codes:
        return SUBGROUPS
    output = []
    for code in codes:
        normalized = code.replace(",", "").replace(":", "").strip()
        if len(normalized) != 2 or not normalized.isdigit():
            raise ValueError(f"Invalid subgroup code: {code}; use codes such as 00 12 22")
        subgroup = (int(normalized[0]), int(normalized[1]))
        if subgroup not in SUBGROUPS:
            raise ValueError(f"Invalid subgroup code: {code}")
        output.append(subgroup)
    return output


def main() -> None:
    args = parse_args()
    if args.bootstrap < 19:
        raise ValueError("Use at least 19 bootstrap draws, including for smoke tests.")
    if args.age_grid_size < 9:
        raise ValueError("Use at least 9 age-grid points.")
    if not 0.0 <= args.trim_low < args.trim_high <= 1.0:
        raise ValueError("Require 0 <= trim-low < trim-high <= 1.")
    data_path = args.data.resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = RunConfig(
        bootstrap=int(args.bootstrap),
        age_grid_size=int(args.age_grid_size),
        seed=int(args.seed),
        trim_low=float(args.trim_low),
        trim_high=float(args.trim_high),
        bandwidth_constant=float(args.bandwidth_constant),
        bandwidth_power=float(args.bandwidth_power),
        simplex_iterations=int(args.simplex_iterations),
        data_path_resolved=str(data_path),
        data_sha256=sha256_file(data_path),
    )
    subgroups = selected_subgroups(args.subgroups)
    if args.subgroups and output_dir == DEFAULT_OUTPUT_DIR.resolve():
        raise ValueError(
            "Subset/smoke runs cannot write to the production directory. "
            "Pass --output-dir with a separate location."
        )
    data = load_data(data_path)
    records = data.to_dict("records")
    results: list[dict[str, Any]] = []
    pending: list[tuple[int, int]] = []
    for educ, nchild in subgroups:
        checkpoint = checkpoint_dir / f"educ_{educ}_nchild_{nchild}.json"
        if checkpoint.exists() and not args.no_resume:
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            observed_signature = payload.get("config_signature")
            if observed_signature == config.signature():
                results.append(payload)
                print(f"Reused checkpoint for education={educ}, children={nchild}", flush=True)
                continue
            legacy_is_verifiable = (
                data_path == DEFAULT_DATA.resolve()
                and config.data_sha256 == CANONICAL_DATA_SHA256
            )
            if observed_signature == config.legacy_signature() and legacy_is_verifiable:
                payload["config_signature"] = config.signature()
                payload["input_data_path"] = config.data_path_resolved
                payload["input_data_sha256"] = config.data_sha256
                atomic_write_json(checkpoint, payload)
                results.append(payload)
                print(
                    f"Verified and upgraded checkpoint for education={educ}, children={nchild}",
                    flush=True,
                )
                continue
        pending.append((educ, nchild))

    total = len(subgroups)
    completed = len(results)
    initial_pending = len(pending)
    requested_jobs = max(int(args.jobs), 1)
    effective_jobs = min(requested_jobs, initial_pending) if initial_pending else 0
    start = time.perf_counter()
    config_dict = asdict(config)
    if pending and max(int(args.jobs), 1) == 1:
        for educ, nchild in pending:
            result = subgroup_diagnostics(records, educ, nchild, config_dict)
            result["input_data_path"] = config.data_path_resolved
            result["input_data_sha256"] = config.data_sha256
            atomic_write_json(checkpoint_dir / f"educ_{educ}_nchild_{nchild}.json", result)
            results.append(result)
            completed += 1
            save_combined(results, config, output_dir)
            print(f"Completed {completed}/{total} subgroups ({100.0 * completed / total:.0f}%)", flush=True)
    elif pending:
        try:
            with ProcessPoolExecutor(max_workers=max(int(args.jobs), 1)) as executor:
                futures = {
                    executor.submit(subgroup_diagnostics, records, educ, nchild, config_dict): (educ, nchild)
                    for educ, nchild in pending
                }
                for future in as_completed(futures):
                    educ, nchild = futures[future]
                    result = future.result()
                    result["input_data_path"] = config.data_path_resolved
                    result["input_data_sha256"] = config.data_sha256
                    atomic_write_json(checkpoint_dir / f"educ_{educ}_nchild_{nchild}.json", result)
                    results.append(result)
                    completed += 1
                    save_combined(results, config, output_dir)
                    print(f"Completed {completed}/{total} subgroups ({100.0 * completed / total:.0f}%)", flush=True)
        except PermissionError as error:
            print(f"Parallel workers are unavailable ({error}); continuing serially.", flush=True)
            effective_jobs = 1
            completed_keys = {(int(result["sc_summary_record"]["educ"]), int(result["sc_summary_record"]["nchild"])) for result in results}
            for educ, nchild in pending:
                if (educ, nchild) in completed_keys:
                    continue
                result = subgroup_diagnostics(records, educ, nchild, config_dict)
                result["input_data_path"] = config.data_path_resolved
                result["input_data_sha256"] = config.data_sha256
                atomic_write_json(checkpoint_dir / f"educ_{educ}_nchild_{nchild}.json", result)
                results.append(result)
                completed += 1
                save_combined(results, config, output_dir)
                print(f"Completed {completed}/{total} subgroups ({100.0 * completed / total:.0f}%)", flush=True)

    save_combined(results, config, output_dir)
    elapsed = time.perf_counter() - start
    runtime_path = output_dir / "runtime_summary.csv"
    runtime = pd.read_csv(runtime_path)
    runtime["driver_wall_seconds_last_invocation"] = elapsed
    runtime["jobs_last_invocation"] = effective_jobs
    if initial_pending == total:
        runtime["driver_wall_seconds_full_run"] = elapsed
        runtime["jobs_full_run"] = effective_jobs
    runtime["platform"] = platform.platform()
    runtime["python_version"] = sys.version.split()[0]
    runtime.to_csv(runtime_path, index=False)
    detailed_runtime_path = output_dir / "runtime_detailed.csv"
    detailed_runtime = pd.read_csv(detailed_runtime_path)
    recorded_full_wall = float(runtime.iloc[0]["driver_wall_seconds_full_run"])
    recorded_full_jobs = float(runtime.iloc[0]["jobs_full_run"])
    full_wall = recorded_full_wall if np.isfinite(recorded_full_wall) else elapsed
    full_jobs = int(recorded_full_jobs) if np.isfinite(recorded_full_jobs) else effective_jobs
    detailed_runtime["worker_processes"] = full_jobs
    detailed_runtime["worker_seconds_sum"] = detailed_runtime["total_cell_seconds"]
    detailed_runtime["run_total_wall_seconds"] = full_wall
    detailed_runtime.to_csv(detailed_runtime_path, index=False)
    print(f"Finished Task 6 in {elapsed:.1f} seconds; outputs are in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
