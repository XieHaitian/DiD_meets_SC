#!/usr/bin/env python3
"""Estimate the main Alaska DID--SC specification and multiplier intervals."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EMPIRICAL_ROOT = HERE.parent
sys.path.insert(0, str(EMPIRICAL_ROOT))

import common_empirical as ce


RESULT_PATH = HERE / "quantile_top4_didsc_results.csv"
RUNTIME_PATH = HERE / "runtime_detailed.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstraps", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=max(1, min((os.cpu_count() or 2) - 2, 8)))
    parser.add_argument("--output-dir", type=Path, default=HERE / "generated")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstraps <= 0 or args.jobs <= 0:
        raise ValueError("--bootstraps and --jobs must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    data = ce.load_data()
    rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for educ, nchild in ce.SUBGROUPS:
        subgroup = ce.restrict_data(data, educ, nchild, ce.QUANTILE_TOP4)
        prep = ce.prepare_arrays(subgroup)
        seed = ce.cell_seed(educ, nchild, offset=1_000_000)
        multipliers = ce.multiplier_matrix(int(prep["n"]), args.bootstraps, seed)
        metadata = {
            "code_version": ce.CODE_VERSION,
            "configuration": "quantile_top4",
            "donors": list(ce.QUANTILE_TOP4),
            "educ": educ,
            "nchild": nchild,
            "n": int(prep["n"]),
            "h_y": float(prep["h_y"]),
            "h_g": float(prep["h_g"]),
            "bootstraps": args.bootstraps,
            "seed": seed,
            "data_sha256": ce.EXPECTED_DATA_SHA256,
        }
        checkpoint = checkpoint_dir / f"educ{educ}_child{nchild}_B{args.bootstraps}.npz"
        label = f"quantile_top4 educ={educ} nchild={nchild}"
        summary = ce.run_bootstrap_cell(prep, multipliers, checkpoint, metadata, args.jobs, label)
        rows.append(
            {
                "educ": educ,
                "nchild": nchild,
                "n": int(prep["n"]),
                "estimate": float(summary["estimate"]),
                "bootstrap_se": float(summary["bootstrap_se"]),
                "symmetric_critical_value": float(summary["symmetric_critical_value"]),
                "ci_low": float(summary["ci_low"]),
                "ci_high": float(summary["ci_high"]),
                "ci_length": float(summary["ci_length"]),
                "n_boot_requested": int(summary["n_boot_requested"]),
                "n_boot_valid": int(summary["n_boot_valid"]),
                "point_seconds": float(summary["point_seconds"]),
                "bootstrap_seconds": float(summary["bootstrap_seconds"]),
                "seconds_per_bootstrap_draw": float(summary["seconds_per_bootstrap_draw"]),
                "total_cell_seconds": float(summary["total_cell_seconds"]),
                "workers": int(summary["workers"]),
                "worker_seconds_sum": float(summary["worker_seconds_sum"]),
                "timing_status": str(summary["timing_status"]),
                "seed": seed,
            }
        )
        pd.DataFrame(rows).sort_values(["educ", "nchild"]).to_csv(output_dir / RESULT_PATH.name, index=False)

    results = pd.DataFrame(rows).sort_values(["educ", "nchild"]).reset_index(drop=True)
    ce.assert_valid_results(results, args.bootstraps)
    run_seconds = time.perf_counter() - started
    results.to_csv(output_dir / RESULT_PATH.name, index=False)
    ce.runtime_frame(results, "01_quantile_top4_didsc", "DID--SC", run_seconds).to_csv(output_dir / RUNTIME_PATH.name, index=False)
    print(f"Completed {len(results)} cells in {run_seconds:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
