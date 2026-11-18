"""Shared simulation methods and data-generating processes.

Statistical algorithms are retained from the supplied power.ipynb.
Model 1: independent N(0, I_p).
Model 2: correlated N(0, Sigma) with 0.4 off-diagonal correlations.
Model 3: standardized chi-square(4) factors and heterogeneous factor covariances.

This file intentionally retains the original NumPy global RNG semantics.
"""

import math
import numpy as np
from scipy.stats import norm, chi2
from scipy.special import comb

def generate_Sigma0(p):
    Sigma0 = np.full((p, p), 0.4)
    np.fill_diagonal(Sigma0, 1)
    return Sigma0

def generate_samples(Sigma1, Sigma2, m, n, p):
    Z1 = norm.rvs(size=(m, p))
    Z2 = norm.rvs(size=(n, p))
    chol_Sigma1 = np.linalg.cholesky(Sigma1)
    chol_Sigma2 = np.linalg.cholesky(Sigma2)
    X1 = np.dot(Z1, chol_Sigma1.T)
    X2 = np.dot(Z2, chol_Sigma2.T)
    return X1, X2

def generate_standardized_chi2_4(size):
    chi2_rvs = chi2.rvs(df=4, size=size)
    standardized = (chi2_rvs - 4) / np.sqrt(8)
    return standardized

def generate_samples1(m, n, p):
    Z1 = generate_standardized_chi2_4(size=(m, p))  
    Z2 = generate_standardized_chi2_4(size=(n, p))  
    tmp = p // 4  
    Lambda = np.array([[1.0, 0.0], [0.0, 0.5]])  
    
    V1_col1 = np.concatenate([np.ones(tmp), np.ones(tmp), np.ones(tmp), np.ones(tmp)])
    V1_col2 = np.concatenate([np.ones(tmp), -np.ones(tmp), np.ones(tmp), -np.ones(tmp)])
    V1 = np.column_stack([V1_col1, V1_col2])
    
    V2_col1 = np.concatenate([np.ones(tmp), np.ones(tmp), -np.ones(tmp), -np.ones(tmp)])
    V2_col2 = np.concatenate([np.ones(tmp), -np.ones(tmp), -np.ones(tmp), np.ones(tmp)])
    V2 = np.column_stack([V2_col1, V2_col2])
    
    Sigma1 = V1 @ Lambda @ V1.T + np.eye(p)
    Sigma2 = V2 @ Lambda @ V2.T + np.eye(p)
    
    chol_Sigma1 = np.linalg.cholesky(Sigma1)
    chol_Sigma2 = np.linalg.cholesky(Sigma2)
    X1 = np.dot(Z1, chol_Sigma1.T)
    X2 = np.dot(Z2, chol_Sigma2.T)
    return X1, X2, Sigma1, Sigma2

def define_W(p):
    alpha_vec = np.full(p, 2 * p ** (-3 / 8))
    i_vals = np.arange(1, p + 1)  
    beta_vec = np.sqrt(2) * (p + i_vals) / p
    B_matrix = np.diag(beta_vec ** 2)
    a = alpha_vec.reshape(-1, 1)
    W = B_matrix + np.dot(a, a.T)
    return W

def compute_c(h, n1, n2, Sigma1, Sigma2, W, beta):
    A = (Sigma1 / n1) + (Sigma2 / n2)
    B = A @ W
    tr_G_n2 = np.trace(B @ B)
    # Compute the mean-shift scale.
    numerator = beta * np.sqrt(2 * tr_G_n2)
    c = np.sqrt(numerator / h)
    return c

class Algorithm:
    def __init__(self, X1: np.ndarray, X2: np.ndarray, W: np.ndarray):
        self.X1 = X1
        self.X2 = X2
        self.W = W  
        self.n1, self.p = X1.shape
        self.n2, _ = X2.shape

        # Precompute quadratic-form matrices.
        self.M11_W = np.zeros((self.n1, self.n1))
        for i in range(self.n1):
            for j in range(i+1, self.n1):
                self.M11_W[i, j] = X1[i, :] @ W @ X1[j, :]

        self.M22_W = np.zeros((self.n2, self.n2))
        for i in range(self.n2):
            for j in range(i+1, self.n2):
                self.M22_W[i, j] = X2[i, :] @ W @ X2[j, :]

        self.M12_W = np.zeros((self.n1, self.n2))
        for i in range(self.n1):
            for j in range(self.n2):
                self.M12_W[i, j] = X1[i, :] @ W @ X2[j, :]

        # Unweighted matrices.
        self.M11 = np.zeros((self.n1, self.n1))
        for i in range(self.n1):
            for j in range(i+1, self.n1):
                self.M11[i, j] = np.sum(X1[i, :] * X1[j, :])
        
        self.M12 = np.zeros((self.n1, self.n2))
        for i in range(self.n1):
            for j in range(self.n2):
                self.M12[i, j] = np.sum(X1[i, :] * X2[j, :])
        
        self.M22 = np.zeros((self.n2, self.n2))
        for i in range(self.n2):
            for j in range(i+1, self.n2):
                self.M22[i, j] = np.sum(X2[i, :] * X2[j, :])
        
        # Compute sample means.
        self.X1_bar = np.mean(X1, axis=0)  
        self.X2_bar = np.mean(X2, axis=0) 
        
        # Centered cross-products. 
        X1_centered = X1 - self.X1_bar  
        X2_centered = X2 - self.X2_bar  
        
        self.M_red11 = X1_centered @ X1_centered.T  
        self.M_red12 = X1_centered @ X2_centered.T  
        self.M_red22 = X2_centered @ X2_centered.T 

        self.M_red11_W = X1_centered @ W @ X1_centered.T
        self.M_red12_W = X1_centered @ W @ X2_centered.T
        self.M_red22_W = X2_centered @ W @ X2_centered.T

        # Paired differences of adjacent observations.
        self.tmp1 = self.n1 // 2  
        self.tmp2 = self.n2 // 2  

        # Unweighted quadratic-form matrices.
        self.M_red11_half = np.zeros((self.tmp1, self.tmp1))
        for i in range(self.tmp1):
            for j in range(i+1, self.tmp1):
                x1_diff_i = (self.X1[2*i, :] - self.X1[2*i+1, :]) / 2
                x1_diff_j = (self.X1[2*j, :] - self.X1[2*j+1, :]) / 2
                self.M_red11_half[i, j] = np.sum(x1_diff_i * x1_diff_j)

        self.M_red22_half = np.zeros((self.tmp2, self.tmp2))
        for i in range(self.tmp2):
            for j in range(i+1, self.tmp2):
                x2_diff_i = (self.X2[2*i, :] - self.X2[2*i+1, :]) / 2
                x2_diff_j = (self.X2[2*j, :] - self.X2[2*j+1, :]) / 2
                self.M_red22_half[i, j] = np.sum(x2_diff_i * x2_diff_j)

        self.M_red12_half = np.zeros((self.tmp1, self.tmp2))
        for i in range(self.tmp1):
            for j in range(self.tmp2):
                x1_diff = (self.X1[2*i, :] - self.X1[2*i+1, :]) / 2
                x2_diff = (self.X2[2*j, :] - self.X2[2*j+1, :]) / 2
                self.M_red12_half[i, j] = np.sum(x1_diff * x2_diff)

        # Weighted quadratic-form matrices.
        self.M_red11_half_W = np.zeros((self.tmp1, self.tmp1))
        for i in range(self.tmp1):
            for j in range(i+1, self.tmp1):
                x1_diff_i = (self.X1[2*i, :] - self.X1[2*i+1, :]) / 2
                x1_diff_j = (self.X1[2*j, :] - self.X1[2*j+1, :]) / 2
                self.M_red11_half_W[i, j] = x1_diff_i @ W @ x1_diff_j

        self.M_red22_half_W = np.zeros((self.tmp2, self.tmp2))
        for i in range(self.tmp2):
            for j in range(i+1, self.tmp2):
                x2_diff_i = (self.X2[2*i, :] - self.X2[2*i+1, :]) / 2
                x2_diff_j = (self.X2[2*j, :] - self.X2[2*j+1, :]) / 2
                self.M_red22_half_W[i, j] = x2_diff_i @ W @ x2_diff_j

        self.M_red12_half_W = np.zeros((self.tmp1, self.tmp2))
        for i in range(self.tmp1):
            for j in range(self.tmp2):
                x1_diff = (self.X1[2*i, :] - self.X1[2*i+1, :]) / 2
                x2_diff = (self.X2[2*j, :] - self.X2[2*j+1, :]) / 2
                self.M_red12_half_W[i, j] = x1_diff @ W @ x2_diff

    # Original weighted RI statistic.
    def RI_W(self) -> float:
        RI1 = np.sum(self.M11_W) / (self.n1 * (self.n1 - 1) / 2.0)
        RI2 = np.sum(self.M22_W) / (self.n2 * (self.n2 - 1) / 2.0)
        RI3 = np.sum(self.M12_W) / (self.n1 * self.n2 / 2.0)
        return RI1 + RI2 - RI3

    # Original TCQ statistic.
    def TCQ(self) -> float:
        component1 = np.sum(self.M11) / (self.n1 * (self.n1 - 1) / 2.0)
        component2 = np.sum(self.M22) / (self.n2 * (self.n2 - 1) / 2.0)
        component3 = np.sum(self.M12) / (self.n1 * self.n2 / 2.0)
        return component1 + component2 - component3

    # Centered RI statistic for wild bootstrap.
    def RI_W_red(self, X1_rand: np.ndarray, X2_rand: np.ndarray) -> float:
        RI1_red = 0.0
        for i in range(self.n1):
            for j in range(i+1, self.n1):
                RI1_red += self.M_red11_W[i, j] * X1_rand[i, 0] * X1_rand[j, 0]
        RI1_red /= (self.n1 * (self.n1 - 1) / 2.0)

        RI2_red = 0.0
        for i in range(self.n2):
            for j in range(i+1, self.n2):
                RI2_red += self.M_red22_W[i, j] * X2_rand[i, 0] * X2_rand[j, 0]
        RI2_red /= (self.n2 * (self.n2 - 1) / 2.0)

        RI3_red = 0.0
        for i in range(self.n1):
            for j in range(self.n2):
                RI3_red += self.M_red12_W[i, j] * X1_rand[i, 0] * X2_rand[j, 0]
        RI3_red /= (self.n1 * self.n2)

        return RI1_red + RI2_red - 2 * RI3_red

    # Centered RI statistic for empirical bootstrap.
    def RI_W_red_empirical(self, X1_rand_idx: list, X2_rand_idx: list) -> float:
        RI1_red = 0.0
        for i in range(self.n1):
            for j in range(i+1, self.n1):
                RI1_red += self.M_red11_W[X1_rand_idx[i], X1_rand_idx[j]]
        RI1_red /= (self.n1 * (self.n1 - 1) / 2.0)

        RI2_red = 0.0
        for i in range(self.n2):
            for j in range(i+1, self.n2):
                RI2_red += self.M_red22_W[X2_rand_idx[i], X2_rand_idx[j]]
        RI2_red /= (self.n2 * (self.n2 - 1) / 2.0)

        RI3_red = 0.0
        for i in range(self.n1):
            for j in range(self.n2):
                RI3_red += self.M_red12_W[X1_rand_idx[i], X2_rand_idx[j]]
        RI3_red /= (self.n1 * self.n2)

        return RI1_red + RI2_red - 2 * RI3_red

    # Paired-difference TCQ statistic, unweighted.
    def TCQ_red_half(self, X1_rand: np.ndarray, X2_rand: np.ndarray) -> float:
        component1 = 0.0
        for i in range(self.tmp1):
            for j in range(i+1, self.tmp1):
                component1 += self.M_red11_half[i, j] * X1_rand[i, 0] * X1_rand[j, 0]
        component1 /= (self.tmp1 * (self.tmp1 - 1) / 2.0)

        component2 = 0.0
        for i in range(self.tmp2):
            for j in range(i+1, self.tmp2):
                component2 += self.M_red22_half[i, j] * X2_rand[i, 0] * X2_rand[j, 0]
        component2 /= (self.tmp2 * (self.tmp2 - 1) / 2.0)

        component1_2 = 0.0
        for i in range(self.tmp1):
            for j in range(self.tmp2):
                component1_2 += self.M_red12_half[i, j] * X1_rand[i, 0] * X2_rand[j, 0]
        component1_2 /= (self.tmp1 * self.tmp2 / 2.0)

        return component1 + component2 - component1_2

    # Paired-difference RI statistic, weighted.
    def RI_W_red_half(self, X1_rand: np.ndarray, X2_rand: np.ndarray) -> float:
        component1 = 0.0
        for i in range(self.tmp1):
            for j in range(i+1, self.tmp1):
                component1 += self.M_red11_half_W[i, j] * X1_rand[i, 0] * X1_rand[j, 0]
        component1 /= (self.tmp1 * (self.tmp1 - 1) / 2.0)

        component2 = 0.0
        for i in range(self.tmp2):
            for j in range(i+1, self.tmp2):
                component2 += self.M_red22_half_W[i, j] * X2_rand[i, 0] * X2_rand[j, 0]
        component2 /= (self.tmp2 * (self.tmp2 - 1) / 2.0)

        component1_2 = 0.0
        for i in range(self.tmp1):
            for j in range(self.tmp2):
                component1_2 += self.M_red12_half_W[i, j] * X1_rand[i, 0] * X2_rand[j, 0]
        component1_2 /= (self.tmp1 * self.tmp2 / 2.0)

        return component1 + component2 - component1_2

    # Wild bootstrap.
    def run_wild_RI_W(self, randomization_times: int = 1000) -> float:
        origin_stat = self.RI_W()
        rand_stats = []
        X1_rand = np.ones((self.n1, 1))
        X2_rand = np.ones((self.n2, 1))

        for _ in range(randomization_times):
            for i in range(self.n1):
                if np.random.randint(0, 2) == 0:
                    X1_rand[i, 0] *= -1
            for i in range(self.n2):
                if np.random.randint(0, 2) == 0:
                    X2_rand[i, 0] *= -1
            rand_stat = self.RI_W_red(X1_rand, X2_rand)
            rand_stats.append(rand_stat)

        extreme_count = sum(1 for s in rand_stats if s >= origin_stat) + 1
        return extreme_count / (randomization_times + 1)

    # Empirical bootstrap.
    def run_empirical_RI_W(self, randomization_times: int = 1000) -> float:
        origin_stat = self.RI_W()
        rand_stats = []
        for _ in range(randomization_times):
            X1_rand_idx = np.random.randint(0, self.n1, size=self.n1).tolist()
            X2_rand_idx = np.random.randint(0, self.n2, size=self.n2).tolist()
            rand_stat = self.RI_W_red_empirical(X1_rand_idx, X2_rand_idx)
            rand_stats.append(rand_stat)

        extreme_count = sum(1 for s in rand_stats if s >= origin_stat) + 1
        return extreme_count / (randomization_times + 1)

    # Wang bootstrap.
    def run_algorithm_half(self, randomization_times: int = 1000) -> float:
        origin_stat = self.TCQ()
        rand_stats = []
        X1_rand = np.ones((self.tmp1, 1))
        X2_rand = np.ones((self.tmp2, 1))

        for _ in range(randomization_times):
            for i in range(self.tmp1):
                if np.random.randint(0, 2) == 0:
                    X1_rand[i, 0] *= -1
            for i in range(self.tmp2):
                if np.random.randint(0, 2) == 0:
                    X2_rand[i, 0] *= -1
            rand_stat = self.TCQ_red_half(X1_rand, X2_rand)
            rand_stats.append(rand_stat)

        extreme_count = sum(1 for s in rand_stats if s >= origin_stat) + 1
        return extreme_count / (randomization_times + 1)

    # Proposed weighted bootstrap.
    def run_new_RI_W_half(self, randomization_times: int = 1000) -> float:
        origin_stat = self.RI_W()
        rand_stats = []
        X1_rand = np.ones((self.tmp1, 1))
        X2_rand = np.ones((self.tmp2, 1))

        for _ in range(randomization_times):
            for i in range(self.tmp1):
                if np.random.randint(0, 2) == 0:
                    X1_rand[i, 0] *= -1
            for i in range(self.tmp2):
                if np.random.randint(0, 2) == 0:
                    X2_rand[i, 0] *= -1
            rand_stat = self.RI_W_red_half(X1_rand, X2_rand)
            rand_stats.append(rand_stat)

        extreme_count = sum(1 for s in rand_stats if s >= origin_stat) + 1
        return extreme_count / (randomization_times + 1)

    def _normal_cdf(self, x):
        return 0.5 * math.erfc(-x * math.sqrt(0.5))

    # Asymptotic variance and standard-error estimators.
    def _leave_out_mean_2(self, X, i, j):
        mask = np.ones(X.shape[0], dtype=bool)
        mask[[i, j]] = False
        return np.mean(X[mask], axis=0)

    def _leave_out_mean_1(self, X, i):
        mask = np.ones(X.shape[0], dtype=bool)
        mask[i] = False
        return np.mean(X[mask], axis=0)

    def _estimate_tr_WSigma1_sq(self):
        m = self.n1
        denominator = m * (m - 1)         
        W = self.W
        total = 0.0
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                X_ij = self._leave_out_mean_2(self.X1, i, j)
                xi_cent = self.X1[i] - X_ij
                xj_cent = self.X1[j] - X_ij
                term1 = np.dot(self.X1[i], W @ xj_cent)
                term2 = np.dot(self.X1[j], W @ xi_cent)
                total += term1 * term2
        return total / denominator

    def _estimate_tr_WSigma2_sq(self):
        n = self.n2
        denominator = n * (n - 1)
        W = self.W
        total = 0.0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                Y_ij = self._leave_out_mean_2(self.X2, i, j)
                yi_cent = self.X2[i] - Y_ij
                yj_cent = self.X2[j] - Y_ij
                term1 = np.dot(self.X2[i], W @ yj_cent)
                term2 = np.dot(self.X2[j], W @ yi_cent)
                total += term1 * term2
        return total / denominator

    def _estimate_tr_WSigma1WSigma2(self):
        m, n = self.n1, self.n2
        denominator = m * n
        W = self.W
        total = 0.0
        for i in range(m):
            for j in range(n):
                X_i_mean = self._leave_out_mean_1(self.X1, i)
                Y_j_mean = self._leave_out_mean_1(self.X2, j)
                xi_cent = self.X1[i] - X_i_mean
                yj_cent = self.X2[j] - Y_j_mean
                term1 = np.dot(self.X1[i], W @ yj_cent)
                term2 = np.dot(self.X2[j], W @ xi_cent)
                total += term1 * term2
        return total / denominator

    def _estimate_tr_Sigma1_sq(self):
        m = self.n1
        denominator = m * (m - 1)         
        total = 0.0
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                X_ij = self._leave_out_mean_2(self.X1, i, j)
                dot1 = np.dot(self.X1[i], self.X1[j] - X_ij)
                dot2 = np.dot(self.X1[j], self.X1[i] - X_ij)
                total += dot1 * dot2
        return total / denominator

    def _estimate_tr_Sigma2_sq(self):
        n = self.n2
        denominator = n * (n - 1)
        total = 0.0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                Y_ij = self._leave_out_mean_2(self.X2, i, j)
                dot1 = np.dot(self.X2[i], self.X2[j] - Y_ij)
                dot2 = np.dot(self.X2[j], self.X2[i] - Y_ij)
                total += dot1 * dot2
        return total / denominator

    def _estimate_tr_Sigma1Sigma2(self):
        m, n = self.n1, self.n2
        denominator = m * n
        total = 0.0
        for i in range(m):
            for j in range(n):
                X_i_mean = self._leave_out_mean_1(self.X1, i)
                Y_j_mean = self._leave_out_mean_1(self.X2, j)
                dot1 = np.dot(self.X1[i], self.X2[j] - Y_j_mean)
                dot2 = np.dot(self.X2[j], self.X1[i] - X_i_mean)
                total += dot1 * dot2
        return total / denominator

    # RID asymptotic calibration (Jiang et al., 2024).
    def run_asymptotic_RI_W(self) -> float:
        riw_stat = self.RI_W()
        tr_WSigma1_sq = self._estimate_tr_WSigma1_sq()
        tr_WSigma2_sq = self._estimate_tr_WSigma2_sq()
        tr_WSigma1WSigma2 = self._estimate_tr_WSigma1WSigma2()
        
        Cm2 = comb(self.n1, 2)
        Cn2 = comb(self.n2, 2)
        Cm1Cn1 = self.n1 * self.n2
        
        term1 = tr_WSigma1_sq / Cm2
        term2 = tr_WSigma2_sq / Cn2
        term3 = (4 * tr_WSigma1WSigma2) / Cm1Cn1
        riw_var = term1 + term2 + term3
        riw_se = math.sqrt(max(1e-10, riw_var))  
        
        z_score = riw_stat / riw_se if riw_se > 1e-8 else 0
        p_value = max(0.0, min(1.0, 1 - self._normal_cdf(z_score)))
        return p_value

    # CQ asymptotic calibration (Chen and Qin, 2010).
    def run_asymptotic_TCQ(self) -> float:
        tcq_stat = self.TCQ()
        tr_Sigma1_sq = self._estimate_tr_Sigma1_sq()
        tr_Sigma2_sq = self._estimate_tr_Sigma2_sq()
        tr_Sigma1Sigma2 = self._estimate_tr_Sigma1Sigma2()
        
        Cm2 = comb(self.n1, 2)
        Cn2 = comb(self.n2, 2)
        Cm1Cn1 = self.n1 * self.n2
        
        term1 = tr_Sigma1_sq / Cm2
        term2 = tr_Sigma2_sq / Cn2
        term3 = (4 * tr_Sigma1Sigma2) / Cm1Cn1
        tcq_var = term1 + term2 + term3
        tcq_se = math.sqrt(max(1e-10, tcq_var))
        
        z_score = tcq_stat / tcq_se if tcq_se > 1e-8 else 0
        p_value = max(0.0, min(1.0, 1 - self._normal_cdf(z_score)))
        return p_value

METHODS = (
    'RI_W', 'TCQ', 'New Bootstrap',
    'Wang Bootstrap', 'Empirical Bootstrap', 'Wild Bootstrap'
)

METHOD_STYLES = {
    'RI_W': {'label': 'RI', 'linestyle': '-.', 'color': 'b'},
    'TCQ': {'label': 'CQ', 'linestyle': '-.', 'color': 'r'},
    'New Bootstrap': {'label': 'New', 'linestyle': '-', 'color': 'black'},
    'Wang Bootstrap': {'label': 'Wang', 'linestyle': '--', 'color': 'g'},
    'Empirical Bootstrap': {'label': 'EB', 'linestyle': ':', 'color': '#FFA500'},
    'Wild Bootstrap': {'label': 'WB', 'linestyle': ':', 'color': 'purple'},
}

MODEL_NAMES = {
    1: 'independent_gaussian',
    2: 'correlated_gaussian',
    3: 'heterogeneous_chisquare_factor',
}

MAIN_PARAMS = [(10, 15, 200), (10, 15, 600),
               (20, 30, 200), (20, 30, 600),
               (60, 80, 200), (60, 80, 600)]
SUPPLEMENT_PARAMS = [(90, 120, 200), (90, 120, 600),
                     (20, 200, 200), (20, 200, 600)]


def generate_model_samples(model, n1, n2, p):
    """Generate null-mean samples and return (Y1, Y2, Sigma1, Sigma2).

    Use the exact original generators, including model 3's latent factors.
    The caller seeds NumPy before invoking this function.
    """
    if model == 1:
        sigma1 = np.eye(p)
        sigma2 = np.eye(p)
        y1 = np.random.normal(0, 1, size=(n1, p))
        y2 = np.random.normal(0, 1, size=(n2, p))
        return y1, y2, sigma1, sigma2
    if model == 2:
        sigma1 = generate_Sigma0(p)
        sigma2 = sigma1.copy()
        y1, y2 = generate_samples(sigma1, sigma2, n1, n2, p)
        return y1, y2, sigma1, sigma2
    if model == 3:
        if p % 4 != 0:
            raise ValueError('Model 3 requires p to be a multiple of 4.')
        return generate_samples1(n1, n2, p)
    raise ValueError('model must be 1, 2, or 3')


def method_pvalues(x1, x2, W, bootstrap_times):
    """Return the six original p-values; preserve method call order."""
    test = Algorithm(x1, x2, W)
    return {
        'Wild Bootstrap': test.run_wild_RI_W(bootstrap_times),
        'Empirical Bootstrap': test.run_empirical_RI_W(bootstrap_times),
        'Wang Bootstrap': test.run_algorithm_half(bootstrap_times),
        'New Bootstrap': test.run_new_RI_W_half(bootstrap_times),
        'RI_W': test.run_asymptotic_RI_W(),
        'TCQ': test.run_asymptotic_TCQ(),
    }


def parse_parameter_grid(specs, section):
    """Parse optional n1,n2,p triples or choose the original paper grid."""
    if not specs:
        return MAIN_PARAMS if section == 'main' else SUPPLEMENT_PARAMS
    grid = []
    for spec in specs:
        try:
            n1, n2, p = [int(value.strip()) for value in spec.split(',')]
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Invalid --params {spec!r}; use n1,n2,p.') from exc
        if min(n1, n2) < 4 or p < 4:
            raise ValueError('n1, n2 and p must be at least 4.')
        grid.append((n1, n2, p))
    return grid


def run_tasks(worker, tasks, workers):
    """Run deterministic per-replication seeds with stable output ordering."""
    if workers <= 1:
        return [worker(*task) for task in tasks]
    import concurrent.futures
    from itertools import repeat
    # executor.map preserves the input order independently of worker scheduling.
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_invoke, repeat(worker), tasks))


def _invoke(fn, args):
    return fn(*args)
