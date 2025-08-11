# DiD\_meets\_SC 

Replication materials for **“Difference-in-Differences Meets Synthetic Control: Doubly Robust Identification and Estimation”** by Yixiao Sun, Haitian Xie, and Yuhang Zhang.
Preprint: [https://arxiv.org/abs/2503.11375](https://arxiv.org/abs/2503.11375)

---

## Contents

* `data_cleaning.R` — Cleans Dube (2019) CPS data (`march_regready_1996.dta`) and produces `Alaska_MW.csv` used in the empirical analysis and calibrated simulations.
* `empirical.py` — Reproduces results in the paper’s empirical section using `Alaska_MW.csv`.

---

## Data

* **Source:** Dube (2019), *Minimum Wages and the Distribution of Family Incomes* (see [AEA website](https://www.aeaweb.org/articles?id=10.1257/app.20170085)).
* **Input:** `march_regready_1996.dta`
* **Output:** `Alaska_MW.csv` (cleaned dataset used by `empirical.py`)

> Note: Please download the Dube (2019) data and place `march_regready_1996.dta` in the project directory (or edit paths in `data_cleaning.R`).
