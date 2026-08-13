#!/usr/bin/env python3
"""Estimate the survey-year multinomial-logit composition diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "Alaska_MW.csv"
MAIN_SPECIFICATION_STATES = (2, 24, 33, 49, 51)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "generated" / "composition_diagnostic.csv")
    args = parser.parse_args()
    data = pd.read_csv(DATA_PATH)
    data = data.loc[data["state_fips"].isin(MAIN_SPECIFICATION_STATES)].copy()
    formula = "year ~ C(state_fips) + age + C(educ) + C(nchild)"
    model = smf.mnlogit(formula, data=data).fit(method="newton", disp=False, maxiter=200)
    result = pd.DataFrame(
        [
            {
                "diagnostic": "multinomial logit of survey year on state and covariates for Alaska and the four main donors",
                "state_fips": "2|24|33|49|51",
                "formula": formula,
                "observations": int(model.nobs),
                "log_likelihood": float(model.llf),
                "null_log_likelihood": float(model.llnull),
                "mcfadden_pseudo_r2": float(model.prsquared),
                "converged": bool(model.mle_retvals.get("converged", False)),
            }
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
