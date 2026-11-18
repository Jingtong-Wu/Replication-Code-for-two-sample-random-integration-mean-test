from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.stats import norm


METHOD_LABELS = {
    "Wild Bootstrap": "WB",
    "Empirical Bootstrap": "EB",
    "New Bootstrap": "New",
    "Wang Bootstrap": "Wang",
    "RI_W": "RI",
    "TCQ": "CQ",
}


def parse_args() -> argparse.Namespace:
    if "__file__" in globals():
        output_dir = Path(__file__).resolve().parent
    else:
        candidates = (Path.cwd(), Path.cwd() / "outputs")
        output_dir = next(
            (path for path in candidates if (path / "data" / "tian").is_dir()),
            Path.cwd(),
        )
    base = output_dir / "data" / "tian"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=base,
        help="Directory containing the Tian expression matrix and group labels.",
    )
    parser.add_argument(
        "--bootstrap-times", type=int, default=10000,
        help="Number of bootstrap replications for WB, EB, Wang, and New (default: 10000).",
    )
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument(
        "--output", type=Path,
        default=output_dir / "tian_pvalues.csv",
    )
    # Ignore Jupyter kernel arguments while retaining normal command-line use.
    args, _ = parser.parse_known_args()
    return args


class Weights:
    """Implicit representation of W used in the simulation code."""

    def __init__(self, p: int):
        alpha = 2.0 * p ** (-3.0 / 8.0)
        i_vals = np.arange(1, p + 1, dtype=np.float64)
        beta = np.sqrt(2.0) * (p + i_vals) / p
        self.a = np.full(p, alpha, dtype=np.float64)
        self.beta_sq = beta**2

    def gram(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x * self.beta_sq) @ y.T + np.outer(x @ self.a, y @ self.a)


def upper_sum(g: np.ndarray) -> float:
    return float(np.sum(np.triu(g, k=1)))


def statistic_from_grams(g11: np.ndarray, g22: np.ndarray, g12: np.ndarray) -> float:
    n1, n2 = g11.shape[0], g22.shape[0]
    part1 = upper_sum(g11) / comb(n1, 2)
    part2 = upper_sum(g22) / comb(n2, 2)
    part12 = np.sum(g12) / (n1 * n2 / 2.0)
    return float(part1 + part2 - part12)


def off_diagonal(g: np.ndarray) -> np.ndarray:
    ans = g.copy()
    np.fill_diagonal(ans, 0.0)
    return ans


def signed_statistics(
    g11: np.ndarray,
    g22: np.ndarray,
    g12: np.ndarray,
    signs1: np.ndarray,
    signs2: np.ndarray,
) -> np.ndarray:
    """Vectorized equivalent of the i<j loops in the supplied code."""
    h11 = off_diagonal(g11)
    h22 = off_diagonal(g22)
    q1 = 0.5 * np.sum((signs1 @ h11) * signs1, axis=1) / comb(g11.shape[0], 2)
    q2 = 0.5 * np.sum((signs2 @ h22) * signs2, axis=1) / comb(g22.shape[0], 2)
    q12 = np.sum((signs1 @ g12) * signs2, axis=1) / (
        g11.shape[0] * g22.shape[0] / 2.0
    )
    return q1 + q2 - q12


def rademacher(rng: np.random.Generator, b: int, n: int) -> np.ndarray:
    return rng.choice(np.array([-1.0, 1.0]), size=(b, n), replace=True)


def paired_differences(x: np.ndarray) -> np.ndarray:
    m = x.shape[0] // 2
    return (x[1 : 2 * m : 2] - x[: 2 * m : 2]) / 2.0


def bootstrap_pvalue(
    origin_stat: float,
    d1: np.ndarray,
    d2: np.ndarray,
    bootstrap_times: int,
    rng: np.random.Generator,
    weights: Weights | None,
) -> float:
    if weights is None:
        g11, g22, g12 = d1 @ d1.T, d2 @ d2.T, d1 @ d2.T
    else:
        g11 = weights.gram(d1, d1)
        g22 = weights.gram(d2, d2)
        g12 = weights.gram(d1, d2)

    s1 = rademacher(rng, bootstrap_times, d1.shape[0])
    s2 = rademacher(rng, bootstrap_times, d2.shape[0])
    boot_stats = signed_statistics(g11, g22, g12, s1, s2)
    return float((np.count_nonzero(boot_stats >= origin_stat) + 1) / (bootstrap_times + 1))


def wild_bootstrap_pvalue(
    origin_stat: float,
    x1_centered: np.ndarray,
    x2_centered: np.ndarray,
    bootstrap_times: int,
    rng: np.random.Generator,
    weights: Weights,
) -> float:
    """Wild bootstrap, with exactly the centered quadratic forms in the supplied code."""
    g11 = weights.gram(x1_centered, x1_centered)
    g22 = weights.gram(x2_centered, x2_centered)
    g12 = weights.gram(x1_centered, x2_centered)
    s1 = rademacher(rng, bootstrap_times, x1_centered.shape[0])
    s2 = rademacher(rng, bootstrap_times, x2_centered.shape[0])
    boot_stats = signed_statistics(g11, g22, g12, s1, s2)
    return float((np.count_nonzero(boot_stats >= origin_stat) + 1) / (bootstrap_times + 1))


def empirical_bootstrap_pvalue(
    origin_stat: float,
    x1_centered: np.ndarray,
    x2_centered: np.ndarray,
    bootstrap_times: int,
    rng: np.random.Generator,
    weights: Weights,
) -> float:

    g11 = weights.gram(x1_centered, x1_centered)
    g22 = weights.gram(x2_centered, x2_centered)
    g12 = weights.gram(x1_centered, x2_centered)
    n1, n2 = x1_centered.shape[0], x2_centered.shape[0]

    c1 = rng.multinomial(n1, np.full(n1, 1.0 / n1), size=bootstrap_times)
    c2 = rng.multinomial(n2, np.full(n2, 1.0 / n2), size=bootstrap_times)
    within1 = 0.5 * (
        np.sum((c1 @ g11) * c1, axis=1) - c1 @ np.diag(g11)
    ) / comb(n1, 2)
    within2 = 0.5 * (
        np.sum((c2 @ g22) * c2, axis=1) - c2 @ np.diag(g22)
    ) / comb(n2, 2)
    cross = np.sum((c1 @ g12) * c2, axis=1) / (n1 * n2 / 2.0)
    boot_stats = within1 + within2 - cross
    return float((np.count_nonzero(boot_stats >= origin_stat) + 1) / (bootstrap_times + 1))


def estimate_same_sample_trace(g: np.ndarray) -> float:
    """Vectorized leave-two-out trace estimator from the supplied code."""
    n = g.shape[0]
    if n <= 2:
        raise ValueError("At least three observations are required.")
    diag = np.diag(g)
    row_sum = np.sum(g, axis=1)
    term = ((n - 1.0) * g + diag[:, None] - row_sum[:, None]) / (n - 2.0)
    products = term * term.T
    np.fill_diagonal(products, 0.0)
    return float(np.sum(products) / (n * (n - 1.0)))


def estimate_cross_trace(g12: np.ndarray) -> float:
    """Vectorized leave-one-out cross-trace estimator from the supplied code."""
    n1, n2 = g12.shape
    row_sum = np.sum(g12, axis=1)
    col_sum = np.sum(g12, axis=0)
    term1 = (n2 * g12 - row_sum[:, None]) / (n2 - 1.0)
    term2 = (n1 * g12 - col_sum[None, :]) / (n1 - 1.0)
    return float(np.sum(term1 * term2) / (n1 * n2))


def asymptotic_pvalue(
    stat: float,
    g11: np.ndarray,
    g22: np.ndarray,
    g12: np.ndarray,
) -> float:
    n1, n2 = g11.shape[0], g22.shape[0]
    tr1 = estimate_same_sample_trace(g11)
    tr2 = estimate_same_sample_trace(g22)
    tr12 = estimate_cross_trace(g12)
    variance = tr1 / comb(n1, 2) + tr2 / comb(n2, 2) + 4.0 * tr12 / (n1 * n2)
    standard_error = math.sqrt(max(1.0e-10, variance))
    return float(norm.sf(stat / standard_error))


def load_groups(data_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    expression = np.load(data_dir / "tian_expression.npy")
    groups = pd.read_csv(data_dir / "groups.csv")["group"].astype(str).to_numpy()
    if expression.shape != (173, 12625) or groups.shape != (173,):
        raise ValueError("Unexpected Tian data dimensions.")

    lesion_label = "MRI-lytic-lesion sample"
    no_lesion_label = "MRI-no-lytic-lesion sample"
    x1 = expression[groups == lesion_label]
    x2 = expression[groups == no_lesion_label]
    if (len(x1), len(x2)) != (137, 36):
        raise ValueError("Unexpected Tian group sizes.")

    info = {
        "dataset": "Tian et al. (2003)",
        "group_1": "one or more focal bone lesions detected by MRI",
        "group_2": "no focal bone lesions detected by MRI",
        "n1": len(x1),
        "n2": len(x2),
        "p": expression.shape[1],
        "preprocessing": "dataset used as provided; no additional centering or scaling",
    }
    return x1, x2, info


def run_analysis(
    x1: np.ndarray,
    x2: np.ndarray,
    bootstrap_times: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    weights = Weights(x1.shape[1])

    g11 = x1 @ x1.T
    g22 = x2 @ x2.T
    g12 = x1 @ x2.T
    gw11 = weights.gram(x1, x1)
    gw22 = weights.gram(x2, x2)
    gw12 = weights.gram(x1, x2)

    tcq_stat = statistic_from_grams(g11, g22, g12)
    ri_stat = statistic_from_grams(gw11, gw22, gw12)

    d1 = paired_differences(x1)
    d2 = paired_differences(x2)
    seed_sequence = np.random.SeedSequence(seed)
    wild_rng, empirical_rng, new_rng, wang_rng = [
        np.random.default_rng(s) for s in seed_sequence.spawn(4)
    ]

    x1_centered = x1 - np.mean(x1, axis=0)
    x2_centered = x2 - np.mean(x2, axis=0)

    pvalues = {
        "Wild Bootstrap": wild_bootstrap_pvalue(
            ri_stat, x1_centered, x2_centered, bootstrap_times, wild_rng, weights
        ),
        "Empirical Bootstrap": empirical_bootstrap_pvalue(
            ri_stat,
            x1_centered,
            x2_centered,
            bootstrap_times,
            empirical_rng,
            weights,
        ),
        "New Bootstrap": bootstrap_pvalue(
            ri_stat, d1, d2, bootstrap_times, new_rng, weights
        ),
        "Wang Bootstrap": bootstrap_pvalue(
            tcq_stat, d1, d2, bootstrap_times, wang_rng, None
        ),
        "RI_W": asymptotic_pvalue(ri_stat, gw11, gw22, gw12),
        "TCQ": asymptotic_pvalue(tcq_stat, g11, g22, g12),
    }
    statistics = {"RI_W": ri_stat, "TCQ": tcq_stat}
    return pvalues, statistics


def main() -> None:
    args = parse_args()
    if args.bootstrap_times < 1:
        raise ValueError("--bootstrap-times must be positive.")

    x1, x2, info = load_groups(args.data_dir)
    print(
        f"Tian: MRI-lesion n1={x1.shape[0]}, no-MRI-lesion n2={x2.shape[0]}, "
        f"p={x1.shape[1]}"
    )
    pvalues, statistics = run_analysis(x1, x2, args.bootstrap_times, args.seed)

    rows = []
    for method in (
        "Wild Bootstrap",
        "Empirical Bootstrap",
        "Wang Bootstrap",
        "New Bootstrap",
        "RI_W",
        "TCQ",
    ):
        stat_key = (
            "RI_W"
            if method in ("Wild Bootstrap", "Empirical Bootstrap", "New Bootstrap", "RI_W")
            else "TCQ"
        )
        rows.append(
            {
                "method": METHOD_LABELS[method],
                "p_value": pvalues[method],
                "statistic": statistics[stat_key],
                "bootstrap_replications": (
                    args.bootstrap_times if "Bootstrap" in method else 0
                ),
            }
        )
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {**info, "seed": args.seed, "bootstrap_times": args.bootstrap_times},
            indent=2,
        )
        + "\n"
    )
    print(result.to_string(index=False, float_format=lambda value: f"{value:.8g}"))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

