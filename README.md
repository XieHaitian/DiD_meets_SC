# DiD_meets_SC

This is the replication package for the paper Difference-in-Differences Meets Synthetic Control: Doubly Robust Identification and Estimation by Yixiao Sun, Haitian Xie, and Yuhang Zhang (https://arxiv.org/abs/2503.11375).

The file data_cleaning.R cleans the "march_regready_1996.dta" file from Dube (2019) Minimum Wages and the Distribution of Family Incomes (the data can be downloaded from https://www.aeaweb.org/articles?id=10.1257/app.20170085) and returns the file Alaska_MW.csv which is subsequently used for the empirical study and calibrated simulation.

The empirical.py file produces the results in the empirical section of the paper.
