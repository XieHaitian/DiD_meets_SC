#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import time

from numba import jit  # Just in Time Compiling can speed up the program
from multiprocessing import Pool, cpu_count

# Only leave the small-dimensional objects in the global space. 
# Place the data reading and pre-precessing inside main() so that 
# the child processes do not have to perform any of these. 

# change the following education level and number of childrens to obtain effect estimates for different subpopulation
ed = 0
nc = 0

L = 2
B = 500


seeds = 123

ridge_llr = 1e-6  # this is used as a regularization parameter for local linear regression
ridge_SC = 1e-6  # this is used as a regularization parameter for obtaining the SC weights


@jit(nopython=True)                # Use jit to speed up the program
def lwlr(X, Y, x, bandwidth, weights_input):

    n = len(Y)
    p = X.shape[1]

    ### design matrix for local linear regression
    X_design = np.hstack((np.ones((n,1)), X-x))
    
    ### Compute kernel weights
    u = (X - x) / bandwidth
    #from scipy.stats import multivariate_normal  
    #kernel_weights = multivariate_normal.pdf(u, mean=np.zeros(p)) / (bandwidth**p) ### Gaussian kernel
    kernel_weights = (3/4) * (1 - np.sum((u**2), axis=1)) * (np.sum(np.abs(u),axis=1) <= 1) * (p+1)/2 ### Epanechnikov kernel
    
    weights = kernel_weights * weights_input
    sqrt_weights = np.sqrt(weights)
    Xw = X_design * sqrt_weights[:, None]
    yw = Y * sqrt_weights
    XtX = Xw.T @ Xw + np.eye(Xw.shape[1]) * ridge_llr
    Xty = Xw.T @ yw
    coef = np.linalg.solve(XtX, Xty)     # this works with jit.  Not all functions will work with jit
    return coef[0]

@jit(nopython=True)                     # A much faster algorthim for projection. Speed up this even more with JIT
def project_to_simplex(w):
    n = len(w)
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(w - theta, 0)
    return w


def bootstrap_iteration(b,
                        y_array, t_array, g_array, x_array, W_boot, folds,
                        groups, times, X_quantiles,
                        pre_T, N_G, G1, lambda_T, lambda_T_1, age_unique,
                        h_Y, h_G):
    #if b % 50 == 0:
      #  print(f"In iteration: {b}")
    
    W_boot = W_boot.reshape(-1, 1)
    pi1 = float((W_boot.ravel() * G1.ravel()).sum() / G1.size)
    arry = np.hstack([y_array, t_array, g_array, x_array, W_boot])
    # data plus the bootstrap weights
    

    fold_scores = np.empty(L)
    # preallocate memory as much as possible.

    n_age = len(age_unique)

    # Preallocate reusable buffers
    Mat_cond_exp = np.zeros((pre_T, N_G+1))   
    # conditional expectations for all group during the pretrend period
    r_1g = np.empty(N_G)
                        
    Gg = np.empty(N_G+1, dtype=float)
    # a binary array indicating the group membership. 

    for l in range(L):
        idx_l = (folds == l)
        idx_lc = ~idx_l

        data_l = arry[idx_l, :]
        data_lc = arry[idx_lc, :]

        y_l, t_l, g_l, x_l, W_l = (
            data_l[:, 0],
            data_l[:, 1],
            data_l[:, 2],
            data_l[:, 3:-1],
            data_l[:, -1]
        )

        y_lc, t_lc, g_lc, x_lc, W_lc = (
            data_lc[:, 0],
            data_lc[:, 1],
            data_lc[:, 2],
            data_lc[:, 3:-1],
            data_lc[:, -1]
        )

        n_l = len(y_l)
        score = 0
        id_Gne1 = (g_lc != groups[0])
                 # groups[0] is the treated group. 
                 # index for the individuals who is not in the treatment group

        # Preallocate the memory.  Avoid using .append for speed.
        r_1gs = np.ones((n_age, N_G))
        mu_Gne1_Ts = np.empty(n_age)
        mu_Gne1_T_1s = np.empty(n_age)
        ws = np.empty((n_age, N_G))    # SC weight for each value of age

        for j in range(n_age):
            
            Xj = age_unique[j]

            mu_Gne1_T = lwlr(x_lc[id_Gne1], y_lc[id_Gne1] * (t_lc[id_Gne1] == times[-1]), Xj, h_Y, W_lc[id_Gne1])
            mu_Gne1_Ts[j] = mu_Gne1_T

            mu_Gne1_T_1 = lwlr(x_lc[id_Gne1], y_lc[id_Gne1] * (t_lc[id_Gne1] == times[-2]), Xj, h_Y, W_lc[id_Gne1])
            mu_Gne1_T_1s[j] = mu_Gne1_T_1

            Mat_cond_exp.fill(0)
            r_1g.fill(np.nan)

            r1 = lwlr(x_lc, (g_lc == groups[0]).astype(float), Xj, h_G, W_lc)
                                  # not the same was what we propose.

            for g_num in range(N_G + 1):
                if g_num >= 1:          # control groups
                    rg = lwlr(x_lc, (g_lc == groups[g_num]).astype(float), Xj, h_G, W_lc)
                    r_1g[g_num-1] = min(r1, 1) / min(max(rg, 0.01), 1)    

                for t_num in range(pre_T):
                    t = times[t_num]
                    id_g = (g_lc == groups[g_num])
                    mu_gt = lwlr(x_lc[id_g], y_lc[id_g] * (t_lc[id_g] == t), Xj, h_Y, W_lc[id_g])
                    Mat_cond_exp[t_num, g_num] = mu_gt

            r_1gs[j,:] = r_1g     # fill in the j-th row of r_1gs. 

            M = Mat_cond_exp[:, 1:-1] - Mat_cond_exp[:, -1:]   # the M_{rc} matrix in the paper
            m1 = Mat_cond_exp[:, 0] - Mat_cond_exp[:, -1]      # the m_{1,rc} matrix in the paper

            try:
                # Regularized matrix: M.T @ M + ridge * I
                MtM = M.T @ M
                MtM = MtM +  np.eye(MtM.shape[0])*ridge_SC
                Mtm1 = M.T @ m1  
                w0 = np.linalg.solve(MtM, Mtm1)
            except np.linalg.LinAlgError:
                w0 = np.ones(M.shape[0]) / M.shape[0]

            last_w = np.array([1.0 - np.sum(w0)])
            w = np.concatenate((w0.flatten(), last_w))
            w = project_to_simplex(w)
           
            ws[j, :] = w

        for i in range(n_l):
            X1 = x_l[i:i + 1, :]
            if (X1 <= X_quantiles[0]).any() or (X1 >= X_quantiles[1]).any():
                continue
            age_index = np.searchsorted(age_unique,X1[0, 0])
            Y, G, Time, W = y_l[i], g_l[i], t_l[i], W_l[i]
            Time_T, Time_T_1 = (Time == times[-1]), (Time == times[-2])

            #Gg = np.array([G == g for g in groups], dtype=float)
            Gg[:] = (G == groups)
            r_1g = r_1gs[age_index]
            mu_Gne1_T, mu_Gne1_T_1 = mu_Gne1_Ts[age_index], mu_Gne1_T_1s[age_index]
            w = ws[age_index]
            
            
            score_value = W * (Gg[0] - np.sum(Gg[1:] * w *  r_1g)) * \
                          (Y * (Time_T / lambda_T - Time_T_1 / lambda_T_1) -
                           (mu_Gne1_T / lambda_T - mu_Gne1_T_1 / lambda_T_1)) / pi1
                         # * is the elementwise product in python

            if not np.isnan(score_value):
                    score += score_value

       
        fold_scores[l] = score / n_l
        

    return np.mean(fold_scores)


def main(ed,nc):
    # === Load data ===
    data_file = 'Alaska_MW.csv'
    data_full = pd.read_csv(data_file)

    data = data_full[(data_full['educ'] == ed) & (data_full['nchild'] == nc)].copy()

    yvar = 'contpov'
    tvar = 'year'
    gvar = 'state_fips'
    xvar = ['age']

    columns_needed = [yvar, tvar, gvar] + xvar
    data = data[columns_needed].copy()

    X_quantiles = data[xvar].quantile([0.05, 0.95]).to_numpy().flatten()

    times = np.sort(data[tvar].unique())   # NumPy array
    pre_T = len(times) - 1                 # Number of pretreatment periods
    groups = np.sort(data[gvar].unique())  # NumPy array
    N_G = len(groups) - 1                  # Number of Control Groups

    G1 = (data[gvar].to_numpy() == groups[0]).astype(float)
    lambda_T = (data[tvar] == times[-1]).mean()
    lambda_T_1 = (data[tvar] == times[-2]).mean()

    n = data.shape[0]
    
    h_Y = 6.25 * n ** (1/5 -1/2)
    h_G = 12.94 * n ** (1/5 -1/2)

    y_array = data[yvar].to_numpy().reshape(-1, 1)
    t_array = data[tvar].to_numpy().reshape(-1, 1)
    g_array = data[gvar].to_numpy().reshape(-1, 1)
    x_array = data[xvar].to_numpy().astype(np.float64)

    age_uni = np.unique(x_array[:, 0])
    age_unique = age_uni[(age_uni > X_quantiles[0]) & (age_uni < X_quantiles[1])]

    np.random.seed(seeds)
    folds = np.random.choice(L, n, replace=True)

    bootstrap_matrix = np.ones((n, B + 1))
    bootstrap_matrix[:, 1:] = np.random.exponential(1, (n, B))

    args = [
        (b,
         y_array, t_array, g_array, x_array, bootstrap_matrix[:, b],
         folds, groups, times, X_quantiles,
         pre_T, N_G, G1, lambda_T, lambda_T_1, age_unique, h_Y, h_G)
        for b in range(B + 1)
    ]


    start_time = time.time()

    with Pool(processes=max(cpu_count() - 2, 1)) as pool:
        bootstrap_values = pool.starmap(bootstrap_iteration, args)

    end_time = time.time()

    #bootstrap_values_df = pd.DataFrame(bootstrap_values, columns=["Bootstrap Value"])
    #bootstrap_values_df.to_csv(f"bootstrap_values_{ed}{nc}.csv", index=False)
    
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    #print(bootstrap_values_df)

    alpha = 0.05
    theta_hat = float(bootstrap_values[0])                  # (no reweighting)
    boots = np.asarray(bootstrap_values[1:], dtype=float)   # exclude the first
    boots = boots[np.isfinite(boots)]                       # drop NaNs if any

    # Bias-corrected percentile CI
    boot_bias = float(np.mean(boots) - theta_hat)           # bootstrap bias
    theta_hat_bc = theta_hat - boot_bias                    # bias-corrected point estimate
    q_lo_bc, q_hi_bc = np.quantile(boots - boot_bias, [alpha/2, 1 - alpha/2])

    print(f"theta_hat = {theta_hat:.3f} (bias-corrected: {theta_hat_bc:.3f})")
    print(f"{int((1-alpha)*100)}% bias-corrected percentile CI: [{q_lo_bc:.3f}, {q_hi_bc:.3f}]")
    


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Bootstrap estimator with (ed, nc) filtering.')
    parser.add_argument('--ed', type=int, default=0, help='education level (int), default 0')
    parser.add_argument('--nc', type=int, default=0, help='number of children (int), default 0')
    args = parser.parse_args()

    main(args.ed, args.nc)
