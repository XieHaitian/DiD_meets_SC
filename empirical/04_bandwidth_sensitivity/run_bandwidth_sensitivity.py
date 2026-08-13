#!/usr/bin/env python3
"""Run the five empirical bandwidth specifications reported in the appendix."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
EMPIRICAL_ROOT = HERE.parent
sys.path.insert(0, str(EMPIRICAL_ROOT))

import common_empirical as ce


COEFFICIENTS = (1.0, 1.5, 2.0, 3.0, 3.5)


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
        n = int(subgroup.shape[0])
        seed = ce.cell_seed(educ, nchild, offset=1_000_000)
        multipliers = ce.multiplier_matrix(n, args.bootstraps, seed)
        for coefficient in COEFFICIENTS:
            h_y, h_g = ce.bandwidths_for_c(n, coefficient)
            prep = ce.prepare_arrays(subgroup, h_y=h_y, h_g=h_g)
            metadata = {
                "code_version": ce.CODE_VERSION,
                "configuration": f"c={coefficient:g}",
                "donors": list(ce.QUANTILE_TOP4),
                "educ": educ,
                "nchild": nchild,
                "n": n,
                "h_y": h_y,
                "h_g": h_g,
                "bootstraps": args.bootstraps,
                "seed": seed,
                "data_sha256": ce.EXPECTED_DATA_SHA256,
            }
            token = str(coefficient).replace(".", "p")
            checkpoint = checkpoint_dir / f"c{token}_educ{educ}_child{nchild}_B{args.bootstraps}.npz"
            label = f"bandwidth c={coefficient:g} educ={educ} nchild={nchild}"
            summary = ce.run_bootstrap_cell(prep, multipliers, checkpoint, metadata, args.jobs, label)
            rows.append(
                {
                    "educ": educ,
                    "nchild": nchild,
                    "n": n,
                    "bandwidth_c": coefficient,
                    "estimate": float(summary["estimate"]),
                    "bootstrap_se": float(summary["bootstrap_se"]),
                    "ci_low": float(summary["ci_low"]),
                    "ci_high": float(summary["ci_high"]),
                    "ci_length": float(summary["ci_length"]),
                    "n_boot_requested": int(summary["n_boot_requested"]),
                    "n_boot_valid": int(summary["n_boot_valid"]),
                }
            )
            pd.DataFrame(rows).sort_values(["educ", "nchild", "bandwidth_c"]).to_csv(output_dir / "manuscript_results.csv", index=False)

    results = pd.DataFrame(rows).sort_values(["educ", "nchild", "bandwidth_c"]).reset_index(drop=True)
    ce.assert_valid_results(results, args.bootstraps)
    results.to_csv(output_dir / "manuscript_results.csv", index=False)
    print(f"Completed {len(results)} bandwidth cells in {time.perf_counter() - started:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
