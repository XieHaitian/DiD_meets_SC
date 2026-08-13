#!/usr/bin/env python3
"""Run the three alternative donor-pool specifications reported in Appendix D."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
EMPIRICAL_ROOT = HERE.parent
sys.path.insert(0, str(EMPIRICAL_ROOT))

import common_empirical as ce


POOLS = {
    "cdf_top4": ce.CDF_TOP4,
    "best_conditioned4": ce.BEST_CONDITIONED4,
    "quantile_top3": ce.QUANTILE_TOP3,
}
SELECTION_RULES = {
    "cdf_top4": "four largest Gunsilius mixture-CDF weights",
    "best_conditioned4": "minimum kappa(M) p95 among the 15 four-state subsets",
    "quantile_top3": "three largest Gunsilius quantile-curve weights",
}


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

    for pool_name, donors in POOLS.items():
        for educ, nchild in ce.SUBGROUPS:
            subgroup = ce.restrict_data(data, educ, nchild, donors)
            prep = ce.prepare_arrays(subgroup)
            seed = ce.cell_seed(educ, nchild, offset=1_000_000)
            multipliers = ce.multiplier_matrix(int(prep["n"]), args.bootstraps, seed)
            metadata = {
                "code_version": ce.CODE_VERSION,
                "configuration": pool_name,
                "donors": list(donors),
                "educ": educ,
                "nchild": nchild,
                "n": int(prep["n"]),
                "h_y": float(prep["h_y"]),
                "h_g": float(prep["h_g"]),
                "bootstraps": args.bootstraps,
                "seed": seed,
                "data_sha256": ce.EXPECTED_DATA_SHA256,
            }
            checkpoint = checkpoint_dir / f"{pool_name}_educ{educ}_child{nchild}_B{args.bootstraps}.npz"
            label = f"{pool_name} educ={educ} nchild={nchild}"
            summary = ce.run_bootstrap_cell(prep, multipliers, checkpoint, metadata, args.jobs, label)
            rows.append(
                {
                    "donor_set": pool_name,
                    "donor_states": "|".join(str(state) for state in donors),
                    "donors": ce.donor_label(donors),
                    "selection_rule": SELECTION_RULES[pool_name],
                    "educ": educ,
                    "nchild": nchild,
                    "n": int(prep["n"]),
                    "estimate": float(summary["estimate"]),
                    "bootstrap_se": float(summary["bootstrap_se"]),
                    "ci_low": float(summary["ci_low"]),
                    "ci_high": float(summary["ci_high"]),
                    "ci_length": float(summary["ci_length"]),
                    "n_boot_requested": int(summary["n_boot_requested"]),
                    "n_boot_valid": int(summary["n_boot_valid"]),
                }
            )
            pd.DataFrame(rows).sort_values(["educ", "nchild", "donor_set"]).to_csv(output_dir / "manuscript_results.csv", index=False)

    results = pd.DataFrame(rows).sort_values(["educ", "nchild", "donor_set"]).reset_index(drop=True)
    ce.assert_valid_results(results, args.bootstraps)
    results.to_csv(output_dir / "manuscript_results.csv", index=False)
    print(f"Completed {len(results)} donor-pool cells", flush=True)


if __name__ == "__main__":
    main()
