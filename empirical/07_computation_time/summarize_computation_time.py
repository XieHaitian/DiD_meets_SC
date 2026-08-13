#!/usr/bin/env python3
"""Summarize the manuscript's DID--SC runtime benchmark from cell timings."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "runtime_detailed.csv")
    parser.add_argument("--output", type=Path, default=HERE / "generated" / "manuscript_results.csv")
    args = parser.parse_args()
    detailed = pd.read_csv(args.input)
    point_seconds = pd.to_numeric(detailed["point_seconds"], errors="raise")
    wall_seconds = pd.to_numeric(detailed["total_wall_seconds"], errors="raise")
    workers = pd.to_numeric(detailed["worker_processes"], errors="raise")
    result = pd.DataFrame(
        [
            {
                "quantity": "median point-estimation time per subgroup",
                "seconds": float(point_seconds.median()),
                "subgroups": int(detailed.shape[0]),
                "bootstrap_draws_per_subgroup": 0,
                "worker_processes": 1,
            },
            {
                "quantity": "all nine subgroups with multiplier bootstrap",
                "seconds": float(wall_seconds.max()),
                "subgroups": int(detailed.shape[0]),
                "bootstrap_draws_per_subgroup": int(detailed["requested_bootstrap_draws"].min()),
                "worker_processes": int(workers.max()),
            },
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
