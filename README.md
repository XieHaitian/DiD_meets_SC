# DiD meets SC

Replication materials for **“Difference-in-Differences Meets Synthetic Control: Doubly Robust Identification and Estimation”** by Yixiao Sun, Haitian Xie, and Yuhang Zhang.

Preprint: [https://arxiv.org/abs/2503.11375](https://arxiv.org/abs/2503.11375)

---

## Contents

* `empirical/` contains the data-cleaning script, cleaned CPS data, shared estimation routines, empirical analysis programs, and CSV results reported in the manuscript.
* `simulation/` contains the Monte Carlo programs and corresponding manuscript results.
* Each numbered task directory contains the Python code required for that exercise and its relevant CSV results.

---

## Data

* **Source:** Dube (2019), *Minimum Wages and the Distribution of Family Incomes*; data are available from the [American Economic Association](https://www.aeaweb.org/articles?id=10.1257/app.20170085).
* **Raw input:** `march_regready_1996.dta`.
* **Cleaning program:** `empirical/data_cleaning.R`.
* **Cleaned output:** `empirical/Alaska_MW.csv`.

The cleaning program selects Alaska and six potential control states, retains the 1998--2003 CPS repeated cross sections used in the application, constructs the household-level variables, and normalizes age to 1998. The cleaned dataset is included in this repository. To regenerate it, download `march_regready_1996.dta`, place it in the `empirical` directory, and run `Rscript data_cleaning.R` from that directory.

The main empirical specification uses Alaska as the treated state and Virginia, New Hampshire, Maryland, and Utah as the four donors. The two additional states in the cleaned data, Michigan and Ohio, are used in donor-pool sensitivity analyses.
Note that the multiplier bootstrap normalizes repeated-cross-section time shares by the total multiplier weight.

The R data-cleaning program requires `dplyr`, `data.table`, and `haven`. The empirical and simulation programs were developed with Python 3.13 and require NumPy, pandas, Numba, and statsmodels.

---

## Directory structure

```text
DiD_meets_SC/
├── README.md
├── empirical/
│   ├── data_cleaning.R
│   ├── Alaska_MW.csv
│   ├── common_empirical.py
│   ├── 01_quantile_top4_didsc/
│   ├── 02_drdid_distributional_sc/
│   ├── 03_condition_number/
│   ├── 04_bandwidth_sensitivity/
│   ├── 05_donor_sensitivity/
│   ├── 06_pt_sc_diagnostics/
│   ├── 07_computation_time/
│   └── 08_composition_diagnostic/
└── simulation/
    ├── 01_bandwidth/
    ├── 02_condition_number/
    ├── 03_local_misspecification/
    ├── 04_analytic_ci/
    ├── 05_drdid_comparison/
    └── 06_aggregate_sc_comparison/
```

## Manuscript crosswalk

#### Empirical analysis

| Directory | Corresponding manuscript result |
|---|---|
| `empirical/01_quantile_top4_didsc/` | Proposed estimates in Table 1 |
| `empirical/02_drdid_distributional_sc/` | DRDiD and distributional SC comparisons in Table 1 |
| `empirical/03_condition_number/` | Empirical condition number reported in Section 6 |
| `empirical/04_bandwidth_sensitivity/` | Bandwidth sensitivity analysis in Appendix Table D.1 |
| `empirical/05_donor_sensitivity/` | Donor pool sensitivity analysis in Appendix Table D.2 |
| `empirical/06_pt_sc_diagnostics/` | PT and SC diagnostic results reported in Appendix D.3 |
| `empirical/07_computation_time/` | Computation-time results reported in Section 6 |
| `empirical/08_composition_diagnostic/` | Composition diagnostic for Assumption TI reported in Section 6 |

#### Simulation analysis

| Directory | Corresponding manuscript result |
|---|---|
| `simulation/01_bandwidth/` | Bandwidth experiments in Table 2 |
| `simulation/02_condition_number/` | Condition number experiments in Table 3 |
| `simulation/03_local_misspecification/` | Locally misspecified experiments in Table 4 |
| `simulation/04_analytic_ci/` | Analytic confidence intervals in Table 5 |
| `simulation/05_drdid_comparison/` | Comparison with DRDiD in Table 6 |
| `simulation/06_aggregate_sc_comparison/` | Comparison with aggregate SC and SDiD in Table 7 |

