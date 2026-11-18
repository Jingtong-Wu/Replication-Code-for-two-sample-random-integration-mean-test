"""Sparse-alternative power; original statistics retained, all models enabled."""
import argparse
import math
import os
import pickle
import csv
import numpy as np
from scipy.special import comb
from scipy.stats import norm, chi2
from tqdm import tqdm
import concurrent.futures


# ============================================================
# Data-generating mechanisms
# ============================================================

def generate_Sigma0(p):
    Sigma0 = np.full((p, p), 0.4)
    np.fill_diagonal(Sigma0, 1.0)
    return Sigma0


def generate_samples(Sigma1, Sigma2, m, n, p):
    Z1 = norm.rvs(size=(m, p))
    Z2 = norm.rvs(size=(n, p))
    chol_Sigma1 = np.linalg.cholesky(Sigma1)
    chol_Sigma2 = np.linalg.cholesky(Sigma2)
    X1 = Z1 @ chol_Sigma1.T
    X2 = Z2 @ chol_Sigma2.T
    return X1, X2


def generate_standardized_chi2_4(size):
    chi2_rvs = chi2.rvs(df=4, size=size)
    return (chi2_rvs - 4.0) / np.sqrt(8.0)


def generate_samples1(m, n, p):
    if p % 4 != 0:
        raise ValueError("Model 3 requires p to be divisible by 4.")

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

    X1 = Z1 @ chol_Sigma1.T
    X2 = Z2 @ chol_Sigma2.T
    return X1, X2


# ============================================================
# Weight matrix W
# ============================================================

def define_W(p):
    alpha_vec = np.full(p, 2 * p ** (-3 / 8))
    i_vals = np.arange(1, p + 1)
    beta_vec = np.sqrt(2) * (p + i_vals) / p
    B_matrix = np.diag(beta_vec ** 2)
    a = alpha_vec.reshape(-1, 1)
    return B_matrix + a @ a.T


# ============================================================
# Proposed method + RID 
# ============================================================

class Algorithm:
    def __init__(self, X1: np.ndarray, X2: np.ndarray, W: np.ndarray):
        self.X1 = X1
        self.X2 = X2
        self.W = W
        self.n1, self.p = X1.shape
        self.n2, _ = X2.shape

        # Original weighted RI statistic
        self.M11_W = np.zeros((self.n1, self.n1))
        for i in range(self.n1):
            for j in range(i + 1, self.n1):
                self.M11_W[i, j] = X1[i, :] @ W @ X1[j, :]

        self.M22_W = np.zeros((self.n2, self.n2))
        for i in range(self.n2):
            for j in range(i + 1, self.n2):
                self.M22_W[i, j] = X2[i, :] @ W @ X2[j, :]

        self.M12_W = np.zeros((self.n1, self.n2))
        for i in range(self.n1):
            for j in range(self.n2):
                self.M12_W[i, j] = X1[i, :] @ W @ X2[j, :]

        # Paired differences for the proposed bootstrap procedure
        self.tmp1 = self.n1 // 2
        self.tmp2 = self.n2 // 2

        self.M_red11_half_W = np.zeros((self.tmp1, self.tmp1))
        for i in range(self.tmp1):
            x1_diff_i = (self.X1[2 * i, :] - self.X1[2 * i + 1, :]) / 2
            for j in range(i + 1, self.tmp1):
                x1_diff_j = (self.X1[2 * j, :] - self.X1[2 * j + 1, :]) / 2
                self.M_red11_half_W[i, j] = x1_diff_i @ W @ x1_diff_j

        self.M_red22_half_W = np.zeros((self.tmp2, self.tmp2))
        for i in range(self.tmp2):
            x2_diff_i = (self.X2[2 * i, :] - self.X2[2 * i + 1, :]) / 2
            for j in range(i + 1, self.tmp2):
                x2_diff_j = (self.X2[2 * j, :] - self.X2[2 * j + 1, :]) / 2
                self.M_red22_half_W[i, j] = x2_diff_i @ W @ x2_diff_j

        self.M_red12_half_W = np.zeros((self.tmp1, self.tmp2))
        for i in range(self.tmp1):
            x1_diff = (self.X1[2 * i, :] - self.X1[2 * i + 1, :]) / 2
            for j in range(self.tmp2):
                x2_diff = (self.X2[2 * j, :] - self.X2[2 * j + 1, :]) / 2
                self.M_red12_half_W[i, j] = x1_diff @ W @ x2_diff

    # ---------- Original weighted RI statistic ----------
    def RI_W(self) -> float:
        RI1 = np.sum(self.M11_W) / (self.n1 * (self.n1 - 1) / 2.0)
        RI2 = np.sum(self.M22_W) / (self.n2 * (self.n2 - 1) / 2.0)
        RI3 = np.sum(self.M12_W) / (self.n1 * self.n2 / 2.0)
        return RI1 + RI2 - RI3

    # ---------- Randomized weighted RI statistic ----------
    def RI_W_red_half(self, X1_rand: np.ndarray, X2_rand: np.ndarray) -> float:
        component1 = 0.0
        for i in range(self.tmp1):
            for j in range(i + 1, self.tmp1):
                component1 += (
                    self.M_red11_half_W[i, j]
                    * X1_rand[i, 0]
                    * X1_rand[j, 0]
                )
        component1 /= (self.tmp1 * (self.tmp1 - 1) / 2.0)

        component2 = 0.0
        for i in range(self.tmp2):
            for j in range(i + 1, self.tmp2):
                component2 += (
                    self.M_red22_half_W[i, j]
                    * X2_rand[i, 0]
                    * X2_rand[j, 0]
                )
        component2 /= (self.tmp2 * (self.tmp2 - 1) / 2.0)

        component12 = 0.0
        for i in range(self.tmp1):
            for j in range(self.tmp2):
                component12 += (
                    self.M_red12_half_W[i, j]
                    * X1_rand[i, 0]
                    * X2_rand[j, 0]
                )
        component12 /= (self.tmp1 * self.tmp2 / 2.0)

        return component1 + component2 - component12

    # ---------- Proposed bootstrap procedure ----------
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

            rand_stats.append(self.RI_W_red_half(X1_rand, X2_rand))

        extreme_count = sum(1 for s in rand_stats if s >= origin_stat) + 1
        return extreme_count / (randomization_times + 1)

    # ---------- RID variance estimators ----------
    def _normal_cdf(self, x):
        cdf_val = 0.5 * math.erfc(-x * math.sqrt(0.5))
        return max(0.0, min(1.0, cdf_val))

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
        return total / (m * (m - 1))

    def _estimate_tr_WSigma2_sq(self):
        n = self.n2
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
        return total / (n * (n - 1))

    def _estimate_tr_WSigma1WSigma2(self):
        m, n = self.n1, self.n2
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
        return total / (m * n)

    # ---------- Original RID procedure (Jiang et al., 2024) ----------
    def run_asymptotic_RI_W(self) -> float:
        riw_stat = self.RI_W()

        tr_WSigma1_sq = self._estimate_tr_WSigma1_sq()
        tr_WSigma2_sq = self._estimate_tr_WSigma2_sq()
        tr_WSigma1WSigma2 = self._estimate_tr_WSigma1WSigma2()

        Cm2 = comb(self.n1, 2)
        Cn2 = comb(self.n2, 2)
        Cm1Cn1 = self.n1 * self.n2

        riw_var = (
            tr_WSigma1_sq / Cm2
            + tr_WSigma2_sq / Cn2
            + 4 * tr_WSigma1WSigma2 / Cm1Cn1
        )

        riw_se = math.sqrt(max(1e-10, riw_var))
        z_score = riw_stat / riw_se if riw_se > 1e-8 else 0.0
        return max(0.0, min(1.0, 1.0 - self._normal_cdf(z_score)))


# ============================================================
# Sparse alternatives
# ============================================================

def sparse_signal_parameters(p, n1, n2, rho, r):
    s = max(1, int(np.floor(p ** (1.0 - rho))))
    omega = np.sqrt(
        2.0 * r * (1.0 / n1 + 1.0 / n2) * np.log(p)
    )
    return s, omega


def generate_sparse_base_samples(model, n1, n2, p):
    if model == 1:
        X1 = np.random.normal(0, 1, size=(n1, p))
        X2 = np.random.normal(0, 1, size=(n2, p))
        return X1, X2

    if model == 2:
        Sigma1 = generate_Sigma0(p)
        Sigma2 = generate_Sigma0(p)
        return generate_samples(Sigma1, Sigma2, n1, n2, p)

    if model == 3:
        return generate_samples1(n1, n2, p)

    raise ValueError("model must be 1, 2, or 3.")


def run_one_sparse_power_simulation(sim, model, n1, n2, p, W, bootstrap_times, rho, r, alpha=0.05,):
   
    np.random.seed(42 + sim)

    Y1, Y2 = generate_sparse_base_samples(model, n1, n2, p)
    s, omega = sparse_signal_parameters(p, n1, n2, rho, r)

    support_rng = np.random.RandomState(100000 + sim)
    support_order = support_rng.permutation(p)
    support = support_order[:s]

    mu1 = np.zeros(p)
    mu2 = np.zeros(p)
    mu1[support] = omega

    X1 = Y1 + mu1
    X2 = Y2 + mu2

    test = Algorithm(X1, X2, W)

    p_new = test.run_new_RI_W_half(bootstrap_times)
    p_rid = test.run_asymptotic_RI_W()

    return {
        "Proposed": p_new < alpha,
        "RID": p_rid < alpha,
    }


def run_sparse_power_set(model, n1, n2, p, W, rho_values, r_values, simulation_times, bootstrap_times=1000, alpha=0.05, max_workers=28,):
    
    methods = ["Proposed", "RID"]

    power_results = {
        rho: {
            r: {method: 0.0 for method in methods}
            for r in r_values
        }
        for rho in rho_values
    }

    print(
        f"\nRunning Model {model}: n1={n1}, n2={n2}, p={p}, "
        f"simulations={simulation_times}, bootstrap={bootstrap_times}"
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for rho in rho_values:
            s, _ = sparse_signal_parameters(p, n1, n2, rho, r_values[0])
            print(
                f"\n--- Model={model}, p={p}, rho={rho}, "
                f"number of nonzero signals={s} ---"
            )

            for r in r_values:
                futures = [
                    executor.submit(run_one_sparse_power_simulation, sim,model, n1, n2, p, W, bootstrap_times, rho, r, alpha,)
                    for sim in range(simulation_times)
                ]

                reject_counts = {method: 0 for method in methods}

                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=simulation_times,
                    desc=f"M{model}, p={p}, rho={rho}, r={r}",
                ):
                    reject = future.result()
                    for method in methods:
                        reject_counts[method] += int(reject[method])

                for method in methods:
                    power_results[rho][r][method] = (
                        reject_counts[method] / simulation_times
                    )

                print(
                    f"Completed: Model={model}, p={p}, rho={rho}, r={r}; "
                    f"Proposed={power_results[rho][r]['Proposed']:.4f}, "
                    f"RID={power_results[rho][r]['RID']:.4f}"
                )

    return power_results


# ============================================================
# Table output
# ============================================================

def results_to_rows(model, n1, n2, p, power_results, rho_values, r_values):
    rows = []
    for rho in rho_values:
        for r in r_values:
            s, omega = sparse_signal_parameters(p, n1, n2, rho, r)
            rows.append({
                "model": model,
                "n1": n1,
                "n2": n2,
                "p": p,
                "rho": rho,
                "num_signals": s,
                "r": r,
                "omega": omega,
                "Proposed": power_results[rho][r]["Proposed"],
                "RID": power_results[rho][r]["RID"],
            })
    return rows


def save_model_table(rows, model, filename=None):
    if filename is None:
        filename = f"sparse_power_model{model}_proposed_rid.csv"

    fieldnames = [
        "model", "n1", "n2", "p", "rho", "num_signals",
        "r", "omega", "Proposed", "RID"
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = row.copy()
            out["omega"] = f"{out['omega']:.8f}"
            out["Proposed"] = f"{out['Proposed']:.4f}"
            out["RID"] = f"{out['RID']:.4f}"
            writer.writerow(out)

    print(f"\nModel {model} table saved to: {filename}")


def print_model_table(rows, model):
    print("\n" + "=" * 94)
    print(f"MODEL {model}: SPARSE POWER TABLE (Proposed vs RID)")
    print("=" * 94)
    header = (
        f"{'p':>5} {'rho':>6} {'s_p':>6} {'r':>6} "
        f"{'omega':>12} {'Proposed':>12} {'RID':>12}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['p']:>5d} "
            f"{row['rho']:>6.1f} "
            f"{row['num_signals']:>6d} "
            f"{row['r']:>6.1f} "
            f"{row['omega']:>12.6f} "
            f"{row['Proposed']:>12.4f} "
            f"{row['RID']:>12.4f}"
        )



def main():
    parser = argparse.ArgumentParser(description='Sparse power comparison for all three models.')
    parser.add_argument('--models', type=int, nargs='+', choices=[1, 2, 3], default=[1, 2, 3])
    parser.add_argument('--params', nargs='+', metavar='n1,n2,p', default=['60,80,200', '60,80,600'])
    parser.add_argument('--rho', type=float, nargs='+', default=[0.2, 0.5, 0.8])
    parser.add_argument('--r', type=float, nargs='+', default=[0.01, 0.1, 1, 10, 50, 100])
    parser.add_argument('--reps', type=int, default=1000)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='outputs/sparse')
    args = parser.parse_args()
    if args.reps < 1 or args.bootstrap < 1 or args.workers < 1 or not 0 < args.alpha < 1:
        parser.error('Invalid reps, bootstrap, workers, or alpha.')
    from pathlib import Path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = []
    for item in args.params:
        try:
            n1, n2, p = [int(s) for s in item.split(',')]
        except ValueError:
            parser.error('--params must be n1,n2,p triples.')
        if p % 4 != 0 and 3 in args.models:
            parser.error('Model 3 requires p to be a multiple of 4.')
        params.append((n1, n2, p))
    for model in args.models:
        rows = []
        for n1, n2, p in params:
            W = define_W(p)
            result = run_sparse_power_set(
                model=model, n1=n1, n2=n2, p=p, W=W,
                rho_values=args.rho, r_values=args.r,
                simulation_times=args.reps, bootstrap_times=args.bootstrap,
                alpha=args.alpha, max_workers=args.workers)
            rows.extend(results_to_rows(model, n1, n2, p, result, args.rho, args.r))
        print_model_table(rows, model)
        save_model_table(rows, model, filename=output_dir / f'sparse_power_model{model}_proposed_rid.csv')


if __name__ == '__main__':
    main()
