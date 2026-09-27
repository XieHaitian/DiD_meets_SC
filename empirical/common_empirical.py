#!/usr/bin/env python3
"""Shared empirical DID--SC implementation for the replication package.

The estimator reproduces the current legacy empirical implementation while
providing deterministic folds, resumable multiplier bootstrap checkpoints, and
standardized timing records.  Expensive runners import this module rather than
duplicating the numerical routines.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from multiprocessing import cpu_count, get_context
from pathlib import Path
from typing import Callable

# Multiprocessing is across bootstrap draws.  Keep numerical libraries inside
# each worker single-threaded to avoid oversubscription.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from numba import njit


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "Alaska_MW.csv"
EXPECTED_DATA_SHA256 = "78cd29c90c2ac88e6b0f8fa77d3d3c9d955c3583964e4bd84aaf319ed2e0affe"

CODE_VERSION = "empirical_q4_suite_v2"
FOLD_COUNT = 2
FOLD_SEED = 123
RIDGE_LLR = 1.0e-6
RIDGE_SC = 1.0e-6
RATIO_EPS = 1.0e-6
ALPHA = 0.05

YVAR = "contpov"
TVAR = "year"
GVAR = "state_fips"
XVAR = "age"
TREATED_STATE = 2
PRE_YEARS = (1998, 1999, 2000, 2001, 2002)
POST_YEAR = 2003

STATE_NAMES = {
    2: "Alaska",
    24: "Maryland",
    26: "Michigan",
    33: "New Hampshire",
    39: "Ohio",
    49: "Utah",
    51: "Virginia",
}

QUANTILE_TOP4 = (51, 33, 24, 49)
CDF_TOP4 = (26, 39, 24, 51)
BEST_CONDITIONED4 = (24, 26, 49, 51)
QUANTILE_TOP3 = (51, 33, 24)
UNION6 = (24, 26, 33, 39, 49, 51)
SUBGROUPS = tuple((educ, nchild) for educ in range(3) for nchild in range(3))


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_data(path: Path = DATA_PATH) -> None:
    """Fail loudly if the empirical input differs from the frozen data file."""

    if not path.exists():
        raise FileNotFoundError(f"Missing empirical data: {path}")
    observed = sha256(path)
    if observed != EXPECTED_DATA_SHA256:
        raise ValueError(
            "Empirical data checksum mismatch: "
            f"expected {EXPECTED_DATA_SHA256}, observed {observed}"
        )


def state_label(state_fips: int) -> str:
    """Return a readable state label."""

    return STATE_NAMES.get(int(state_fips), str(int(state_fips)))


def donor_label(donors: tuple[int, ...] | list[int]) -> str:
    """Return a stable human-readable donor-pool label."""

    return ", ".join(state_label(state) for state in donors)


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the frozen household-level repeated cross section."""

    verify_data(path)
    data = pd.read_csv(path)
    unnamed = [column for column in data.columns if column.startswith("Unnamed:")]
    if unnamed:
        data = data.drop(columns=unnamed)
    required = {GVAR, TVAR, "hhseq", XVAR, "educ", YVAR, "nchild"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Empirical data are missing columns: {missing}")
    data = data.reset_index(drop=True)
    data["_row_id"] = np.arange(data.shape[0], dtype=np.int64)
    return data


def restrict_data(
    data: pd.DataFrame,
    educ: int,
    nchild: int,
    donors: tuple[int, ...] | list[int],
    years: tuple[int, ...] | list[int] | None = None,
) -> pd.DataFrame:
    """Select one subgroup, Alaska, and a fixed donor pool."""

    selected_years = tuple(PRE_YEARS) + (POST_YEAR,) if years is None else tuple(years)
    states = (TREATED_STATE,) + tuple(int(state) for state in donors)
    out = data[
        (data["educ"] == int(educ))
        & (data["nchild"] == int(nchild))
        & data[GVAR].isin(states)
        & data[TVAR].isin(selected_years)
    ].copy()
    columns = [YVAR, TVAR, GVAR, XVAR, "educ", "nchild", "hhseq", "_row_id"]
    out = out[columns].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"Empty subgroup educ={educ}, nchild={nchild}, donors={donors}")
    if tuple(sorted(out[TVAR].unique())) != tuple(sorted(selected_years)):
        raise ValueError("The selected subgroup does not contain every requested year")
    if tuple(sorted(out[GVAR].unique())) != tuple(sorted(states)):
        raise ValueError("The selected subgroup does not contain every requested state")
    return out


def empirical_base_bandwidths(n: int) -> tuple[float, float]:
    """Return the paper's empirical undersmoothed outcome/ratio bandwidths."""

    adjustment = float(n) ** (1.0 / 5.0 - 1.0 / 3.5)
    return 6.25 * adjustment, 12.94 * adjustment


def bandwidths_for_c(n: int, coefficient: float) -> tuple[float, float]:
    """Map simulation-style c to the empirical baseline, where c=2.5."""

    base_y, base_g = empirical_base_bandwidths(n)
    multiplier = float(coefficient) / 2.5
    return base_y * multiplier, base_g * multiplier


def prepare_arrays(
    data: pd.DataFrame,
    h_y: float | None = None,
    h_g: float | None = None,
    trim_quantiles: tuple[float, float] = (0.05, 0.95),
) -> dict[str, object]:
    """Construct the exact arrays used by the legacy empirical estimator."""

    data = data.reset_index(drop=True)
    n = int(data.shape[0])
    times = np.sort(data[TVAR].unique()).astype(np.float64)
    groups = np.sort(data[GVAR].unique()).astype(np.float64)
    if int(groups[0]) != TREATED_STATE:
        raise ValueError("Alaska must be the first sorted group")
    if times.size < 2 or groups.size < 2:
        raise ValueError("At least two periods and two groups are required")

    x_array = data[[XVAR]].to_numpy(dtype=np.float64)
    x_quantiles = data[[XVAR]].quantile(list(trim_quantiles)).to_numpy().ravel()
    age_unique = np.unique(x_array[:, 0])
    age_unique = age_unique[
        (age_unique > float(x_quantiles[0]))
        & (age_unique < float(x_quantiles[1]))
    ]
    if age_unique.size == 0:
        raise ValueError("No age evaluation points remain after trimming")

    base_y, base_g = empirical_base_bandwidths(n)
    random_state = np.random.RandomState(FOLD_SEED)
    folds = random_state.choice(FOLD_COUNT, n, replace=True).astype(np.int64)

    return {
        "data": data,
        "n": n,
        "times": times,
        "pre_t": int(times.size - 1),
        "groups": groups,
        "n_donors": int(groups.size - 1),
        "g1": (data[GVAR].to_numpy() == groups[0]).astype(np.float64),
        "lambda_t": float((data[TVAR] == times[-1]).mean()),
        "lambda_t_1": float((data[TVAR] == times[-2]).mean()),
        "y_array": data[YVAR].to_numpy(dtype=np.float64).reshape(-1, 1),
        "t_array": data[TVAR].to_numpy(dtype=np.float64).reshape(-1, 1),
        "g_array": data[GVAR].to_numpy(dtype=np.float64).reshape(-1, 1),
        "x_array": x_array,
        "x_quantiles": x_quantiles,
        "age_unique": age_unique,
        "folds": folds,
        "h_y": float(base_y if h_y is None else h_y),
        "h_g": float(base_g if h_g is None else h_g),
    }


@njit(cache=True)
def local_linear(
    x: np.ndarray,
    y: np.ndarray,
    evaluation: float,
    bandwidth: float,
    input_weights: np.ndarray,
) -> float:
    """Epanechnikov local-linear intercept with the legacy absolute ridge."""

    n = len(y)
    p = x.shape[1]
    design = np.hstack((np.ones((n, 1)), x - evaluation))
    u = (x - evaluation) / bandwidth
    kernel_weights = (
        0.75
        * (1.0 - np.sum(u**2, axis=1))
        * (np.sum(np.abs(u), axis=1) <= 1.0)
        * (p + 1.0)
        / 2.0
    )
    weights = kernel_weights * input_weights
    sqrt_weights = np.sqrt(weights)
    xw = design * sqrt_weights[:, None]
    yw = y * sqrt_weights
    gram = xw.T @ xw + np.eye(xw.shape[1]) * RIDGE_LLR
    rhs = xw.T @ yw
    coefficient = np.linalg.solve(gram, rhs)
    return float(coefficient[0])


@njit(cache=True)
def propensity_ratio_local_linear(
    x: np.ndarray,
    group: np.ndarray,
    evaluation: float,
    bandwidth: float,
    input_weights: np.ndarray,
    donor: float,
    treated: float,
) -> float:
    """Estimate P(G=treated|X=x)/P(G=donor|X=x) directly."""

    n = x.shape[0]
    centered = x[:, 0] - evaluation
    u = centered / bandwidth
    kernel = np.zeros(n)
    mask = (u >= -1.0) & (u <= 1.0)
    kernel[mask] = 0.75 * (1.0 - u[mask] * u[mask])
    weights = kernel * input_weights
    donor_indicator = (group == donor).astype(np.float64)
    treated_indicator = (group == treated).astype(np.float64)
    s0 = np.sum(weights * donor_indicator)
    s1 = np.sum(weights * donor_indicator * centered)
    s2 = np.sum(weights * donor_indicator * centered * centered)
    t0 = np.sum(weights * treated_indicator)
    t1 = np.sum(weights * treated_indicator * centered)
    determinant = s0 * s2 - s1 * s1
    scale = (s0 + s2) / 2.0
    adaptive_eps = RATIO_EPS * max(scale, 1.0)
    if abs(determinant) < adaptive_eps:
        if abs(s0) > adaptive_eps:
            return float(t0 / s0)
        return float(t0 / (s0 + adaptive_eps))
    return float((t0 * s2 - t1 * s1) / determinant)


@njit(cache=True)
def simplex_ridge_least_squares(
    design: np.ndarray,
    target: np.ndarray,
    ridge: float = RIDGE_SC,
) -> np.ndarray:
    """Solve ridge least squares exactly over every simplex face."""

    n_obs = design.shape[0]
    n_weights = design.shape[1]
    best_weights = np.ones(n_weights) / n_weights
    best_objective = 1.0e308

    for mask in range(1, 1 << n_weights):
        active_count = 0
        for column in range(n_weights):
            if (mask >> column) & 1:
                active_count += 1
        active_index = np.empty(active_count, dtype=np.int64)
        position = 0
        for column in range(n_weights):
            if (mask >> column) & 1:
                active_index[position] = column
                position += 1

        active_design = np.empty((n_obs, active_count))
        for row in range(n_obs):
            for column in range(active_count):
                active_design[row, column] = design[row, active_index[column]]
        gram = active_design.T @ active_design + np.eye(active_count) * ridge
        rhs = active_design.T @ target
        kkt = np.zeros((active_count + 1, active_count + 1))
        for row in range(active_count):
            for column in range(active_count):
                kkt[row, column] = gram[row, column]
            kkt[row, active_count] = 1.0
            kkt[active_count, row] = 1.0
        constrained_target = np.empty(active_count + 1)
        for row in range(active_count):
            constrained_target[row] = rhs[row]
        constrained_target[active_count] = 1.0
        solution = np.linalg.solve(kkt, constrained_target)

        feasible = True
        for row in range(active_count):
            if solution[row] < -1.0e-8:
                feasible = False
                break
        if not feasible:
            continue

        candidate = np.zeros(n_weights)
        total = 0.0
        for row in range(active_count):
            value = solution[row]
            if value < 0.0:
                value = 0.0
            candidate[active_index[row]] = value
            total += value
        if total <= 0.0:
            continue
        candidate /= total

        objective = 0.0
        for row in range(n_obs):
            fitted = 0.0
            for column in range(n_weights):
                fitted += design[row, column] * candidate[column]
            residual = fitted - target[row]
            objective += residual * residual
        for column in range(n_weights):
            objective += ridge * candidate[column] * candidate[column]
        if objective < best_objective:
            best_objective = objective
            best_weights = candidate
    return best_weights


def didsc_estimate(prep: dict[str, object], observation_weights: np.ndarray) -> float:
    """Evaluate the cross-fitted DID--SC score for one multiplier draw."""

    weights = np.asarray(observation_weights, dtype=np.float64).reshape(-1, 1)
    y_array = np.asarray(prep["y_array"])
    t_array = np.asarray(prep["t_array"])
    g_array = np.asarray(prep["g_array"])
    x_array = np.asarray(prep["x_array"])
    folds = np.asarray(prep["folds"])
    groups = np.asarray(prep["groups"])
    times = np.asarray(prep["times"])
    x_quantiles = np.asarray(prep["x_quantiles"])
    age_unique = np.asarray(prep["age_unique"])
    g1 = np.asarray(prep["g1"])
    n_donors = int(prep["n_donors"])
    pre_t = int(prep["pre_t"])
    w_all = weights.ravel()
    t_all = t_array.ravel()
    weight_total = w_all.sum()
    lambda_t = float((w_all * (t_all == times[-1])).sum() / weight_total)
    lambda_t_1 = float((w_all * (t_all == times[-2])).sum() / weight_total)
    h_y = float(prep["h_y"])
    h_g = float(prep["h_g"])

    pi1 = float((weights.ravel() * g1.ravel()).sum() / g1.size)
    array = np.hstack([y_array, t_array, g_array, x_array, weights])
    fold_scores = np.empty(FOLD_COUNT)
    conditional_means = np.zeros((pre_t, n_donors + 1))
    ratios = np.empty(n_donors)
    group_indicators = np.empty(n_donors + 1, dtype=float)

    for fold in range(FOLD_COUNT):
        evaluation_index = folds == fold
        training_index = ~evaluation_index
        evaluation_data = array[evaluation_index, :]
        training_data = array[training_index, :]

        y_eval = evaluation_data[:, 0]
        t_eval = evaluation_data[:, 1]
        g_eval = evaluation_data[:, 2]
        x_eval = evaluation_data[:, 3:-1]
        w_eval = evaluation_data[:, -1]
        y_train = training_data[:, 0]
        t_train = training_data[:, 1]
        g_train = training_data[:, 2]
        x_train = training_data[:, 3:-1]
        w_train = training_data[:, -1]

        controls_train = g_train != groups[0]
        ratio_by_age = np.ones((age_unique.size, n_donors))
        control_post_by_age = np.empty(age_unique.size)
        control_last_pre_by_age = np.empty(age_unique.size)
        sc_weights_by_age = np.empty((age_unique.size, n_donors))

        for age_index, age in enumerate(age_unique):
            control_post_by_age[age_index] = local_linear(
                x_train[controls_train],
                y_train[controls_train] * (t_train[controls_train] == times[-1]),
                float(age),
                h_y,
                w_train[controls_train],
            )
            control_last_pre_by_age[age_index] = local_linear(
                x_train[controls_train],
                y_train[controls_train] * (t_train[controls_train] == times[-2]),
                float(age),
                h_y,
                w_train[controls_train],
            )
            conditional_means.fill(0.0)
            ratios.fill(np.nan)
            for group_index in range(n_donors + 1):
                if group_index >= 1:
                    ratios[group_index - 1] = propensity_ratio_local_linear(
                        x_train,
                        g_train,
                        float(age),
                        h_g,
                        w_train,
                        groups[group_index],
                        groups[0],
                    )
                group_mask = g_train == groups[group_index]
                for time_index in range(pre_t):
                    year = times[time_index]
                    conditional_means[time_index, group_index] = local_linear(
                        x_train[group_mask],
                        y_train[group_mask] * (t_train[group_mask] == year),
                        float(age),
                        h_y,
                        w_train[group_mask],
                    )
            ratio_by_age[age_index, :] = ratios
            sc_weights_by_age[age_index, :] = simplex_ridge_least_squares(
                conditional_means[:, 1:],
                conditional_means[:, 0],
                RIDGE_SC,
            )

        score = 0.0
        for row in range(y_eval.size):
            age = float(x_eval[row, 0])
            if age <= x_quantiles[0] or age >= x_quantiles[1]:
                continue
            age_index = int(np.searchsorted(age_unique, age))
            group_indicators[:] = g_eval[row] == groups
            post_indicator = t_eval[row] == times[-1]
            last_pre_indicator = t_eval[row] == times[-2]
            ratio = ratio_by_age[age_index]
            sc_weights = sc_weights_by_age[age_index]
            control_post = control_post_by_age[age_index]
            control_last_pre = control_last_pre_by_age[age_index]
            score_value = (
                w_eval[row]
                * (
                    group_indicators[0]
                    - np.sum(group_indicators[1:] * sc_weights * ratio)
                )
                * (
                    y_eval[row]
                    * (
                        post_indicator / lambda_t
                        - last_pre_indicator / lambda_t_1
                    )
                    - (
                        control_post / lambda_t
                        - control_last_pre / lambda_t_1
                    )
                )
                / pi1
            )
            if np.isfinite(score_value):
                score += float(score_value)
        fold_scores[fold] = score / y_eval.size
    return float(np.mean(fold_scores))


def multiplier_matrix(n: int, draws: int, seed: int) -> np.ndarray:
    """Generate deterministic observation-level Exp(1) multipliers."""

    rng = np.random.default_rng(int(seed))
    return rng.exponential(1.0, size=(int(n), int(draws)))


_WORKER_PREP: dict[str, object] | None = None
_WORKER_MULTIPLIERS: np.ndarray | None = None


def _initialize_worker(prep: dict[str, object], multipliers: np.ndarray) -> None:
    global _WORKER_PREP, _WORKER_MULTIPLIERS
    _WORKER_PREP = prep
    _WORKER_MULTIPLIERS = multipliers


def _bootstrap_worker(draw: int) -> tuple[int, float, float]:
    if _WORKER_PREP is None or _WORKER_MULTIPLIERS is None:
        raise RuntimeError("Bootstrap worker was not initialized")
    start = time.perf_counter()
    try:
        value = didsc_estimate(_WORKER_PREP, _WORKER_MULTIPLIERS[:, draw])
    except Exception:
        value = np.nan
    return int(draw), float(value), float(time.perf_counter() - start)


def _checkpoint_metadata(metadata: dict[str, object]) -> str:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def _write_checkpoint(
    path: Path,
    metadata: dict[str, object],
    point: float,
    point_seconds: float,
    bootstrap_values: np.ndarray,
    bootstrap_seconds: float,
    worker_seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        metadata_json=np.asarray(_checkpoint_metadata(metadata)),
        point=np.asarray(point),
        point_seconds=np.asarray(point_seconds),
        bootstrap_values=np.asarray(bootstrap_values),
        bootstrap_seconds=np.asarray(bootstrap_seconds),
        worker_seconds=np.asarray(worker_seconds),
    )
    os.replace(temporary, path)


def _read_checkpoint(
    path: Path,
    metadata: dict[str, object],
    draws: int,
) -> tuple[float, float, np.ndarray, float, float]:
    if not path.exists():
        return np.nan, 0.0, np.full(draws, np.nan), 0.0, 0.0
    with np.load(path, allow_pickle=False) as stored:
        observed_metadata = str(stored["metadata_json"].item())
        if observed_metadata != _checkpoint_metadata(metadata):
            raise ValueError(f"Checkpoint specification mismatch: {path}")
        values = np.asarray(stored["bootstrap_values"], dtype=float)
        if values.shape != (draws,):
            raise ValueError(f"Checkpoint draw count mismatch: {path}")
        return (
            float(stored["point"].item()),
            float(stored["point_seconds"].item()),
            values,
            float(stored["bootstrap_seconds"].item()),
            float(stored["worker_seconds"].item()),
        )


def run_bootstrap_cell(
    prep: dict[str, object],
    multipliers: np.ndarray,
    checkpoint_path: Path,
    metadata: dict[str, object],
    jobs: int,
    label: str,
    notify: Callable[[str], None] = print,
) -> dict[str, object]:
    """Run or resume one point estimate plus a full multiplier bootstrap."""

    draws = int(multipliers.shape[1])
    if multipliers.shape[0] != int(prep["n"]):
        raise ValueError("Multiplier matrix row count does not match the sample")
    point, point_seconds, values, bootstrap_seconds, worker_seconds = _read_checkpoint(
        checkpoint_path,
        metadata,
        draws,
    )

    if not np.isfinite(point):
        start = time.perf_counter()
        point = didsc_estimate(prep, np.ones(int(prep["n"]), dtype=float))
        point_seconds += time.perf_counter() - start
        _write_checkpoint(
            checkpoint_path,
            metadata,
            point,
            point_seconds,
            values,
            bootstrap_seconds,
            worker_seconds,
        )

    missing = np.flatnonzero(~np.isfinite(values))
    completed_before = draws - int(missing.size)
    next_milestone = max(10, ((completed_before // max(draws // 10, 1)) + 1) * 10)
    if missing.size:
        start = time.perf_counter()
        context_name = "fork" if sys.platform != "win32" else "spawn"
        context = get_context(context_name)
        worker_count = max(1, min(int(jobs), int(missing.size)))
        with context.Pool(
            processes=worker_count,
            initializer=_initialize_worker,
            initargs=(prep, multipliers),
        ) as pool:
            for draw, value, seconds in pool.imap_unordered(
                _bootstrap_worker,
                missing.tolist(),
                chunksize=1,
            ):
                values[draw] = value
                worker_seconds += seconds
                completed = int(np.isfinite(values).sum())
                percent = int(np.floor(100.0 * completed / draws + 1.0e-12))
                while next_milestone <= 100 and percent >= next_milestone:
                    elapsed = time.perf_counter() - start
                    notify(
                        f"{label}: {next_milestone}% ({completed}/{draws}) "
                        f"bootstrap draws complete; segment wall time {elapsed:.1f}s"
                    )
                    sys.stdout.flush()
                    _write_checkpoint(
                        checkpoint_path,
                        metadata,
                        point,
                        point_seconds,
                        values,
                        bootstrap_seconds + elapsed,
                        worker_seconds,
                    )
                    next_milestone += 10
        bootstrap_seconds += time.perf_counter() - start
        _write_checkpoint(
            checkpoint_path,
            metadata,
            point,
            point_seconds,
            values,
            bootstrap_seconds,
            worker_seconds,
        )

    valid = values[np.isfinite(values)]
    valid_count = int(valid.size)
    if valid_count == 0:
        standard_error = critical_value = ci_low = ci_high = np.nan
    else:
        roots = valid - point
        standard_error = float(np.std(valid, ddof=1)) if valid_count >= 2 else np.nan
        critical_value = float(np.quantile(np.abs(roots), 1.0 - ALPHA))
        ci_low = float(point - critical_value)
        ci_high = float(point + critical_value)
    return {
        "estimate": float(point),
        "bootstrap_se": standard_error,
        "symmetric_critical_value": critical_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_length": float(ci_high - ci_low) if np.isfinite(ci_low + ci_high) else np.nan,
        "n_boot_requested": draws,
        "n_boot_valid": valid_count,
        "point_seconds": float(point_seconds),
        "bootstrap_seconds": float(bootstrap_seconds),
        "worker_seconds_sum": float(worker_seconds),
        "total_cell_seconds": float(point_seconds + bootstrap_seconds),
        "seconds_per_bootstrap_draw": (
            float(bootstrap_seconds / valid_count) if valid_count else np.nan
        ),
        "workers": max(1, min(int(jobs), draws)),
        "timing_status": "measured",
        "checkpoint": str(checkpoint_path),
    }


def save_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV atomically so an interrupted run preserves prior results."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def save_text_atomic(text: str, path: Path) -> None:
    """Write text atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def runtime_frame(
    results: pd.DataFrame,
    task: str,
    method: str,
    run_total_wall_seconds: float,
) -> pd.DataFrame:
    """Return standardized per-cell timing records for Task 8."""

    frame = pd.DataFrame(
        {
            "task": task,
            "method": method,
            "configuration": results.get("configuration", results.get("donor_set", "")),
            "educ": results["educ"],
            "nchild": results["nchild"],
            "sample_size": results["n"],
            "point_estimation_seconds": results["point_seconds"],
            "bootstrap_seconds": results["bootstrap_seconds"],
            "bootstrap_draws_requested": results["n_boot_requested"],
            "bootstrap_draws_valid": results["n_boot_valid"],
            "seconds_per_bootstrap_draw": results["seconds_per_bootstrap_draw"],
            "total_cell_seconds": results["total_cell_seconds"],
            "worker_processes": results["workers"],
            "worker_seconds_sum": results["worker_seconds_sum"],
            "run_total_wall_seconds": float(run_total_wall_seconds),
            "timing_status": results["timing_status"],
            "seed": results["seed"],
            "code_version": CODE_VERSION,
        }
    )
    return frame


def machine_record(jobs: int) -> dict[str, object]:
    """Return lightweight hardware/software metadata for provenance."""

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "logical_cpu_count": cpu_count(),
        "worker_processes": int(jobs),
        "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS", ""),
        "mkl_threads": os.environ.get("MKL_NUM_THREADS", ""),
        "omp_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "veclib_threads": os.environ.get("VECLIB_MAXIMUM_THREADS", ""),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "numba": __import__("numba").__version__,
        "data_sha256": EXPECTED_DATA_SHA256,
        "code_version": CODE_VERSION,
    }


def cell_seed(educ: int, nchild: int, offset: int = 0) -> int:
    """Return a stable cell seed shared across requested sensitivity values."""

    return int(FOLD_SEED + int(offset) + 10_000 * int(educ) + 1_000 * int(nchild))


def assert_valid_results(frame: pd.DataFrame, draws: int) -> None:
    """Validate the invariants shared by all DID--SC result tables."""

    numeric = ["estimate", "bootstrap_se", "ci_low", "ci_high", "ci_length"]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite estimate, standard error, or confidence interval")
    if not (frame["ci_low"] <= frame["estimate"]).all():
        raise ValueError("A confidence interval lower endpoint exceeds its estimate")
    if not (frame["estimate"] <= frame["ci_high"]).all():
        raise ValueError("A confidence interval upper endpoint is below its estimate")
    if not (frame["n_boot_valid"] == int(draws)).all():
        raise ValueError("At least one configuration has an invalid bootstrap draw")
