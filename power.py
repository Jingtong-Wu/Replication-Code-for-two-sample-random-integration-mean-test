"""Power curves for Models 1, 2, and 3 (main and supplementary grids).

Examples:
  python power.py --section main --models 1 2 3
  python power.py --section supplementary --models 1 2 3
  python power.py --models 1 2 3 --params 6,8,8 --beta 1 --reps 2 --bootstrap 3 --workers 1
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from simulation_core import (METHODS, METHOD_STYLES, MODEL_NAMES,
                             compute_c, define_W, generate_model_samples,
                             method_pvalues, parse_parameter_grid, run_tasks)


def one_power_replication(sim, model, n1, n2, p, W, bootstrap, beta, alpha, seed):
    # Keep the source notebook's common-random-number convention across beta values.
    np.random.seed(seed + sim)
    y1, y2, sigma1, sigma2 = generate_model_samples(model, n1, n2, p)
    h = np.ones(p) @ W @ np.ones(p)
    shift = compute_c(h, n1, n2, sigma1, sigma2, W, beta)
    x1 = y1  # mu1 = 0
    x2 = y2 + shift  # mu2 = c * ones(p)
    pvals = method_pvalues(x1, x2, W, bootstrap)
    return {method: pvals[method] < alpha for method in METHODS}


def run_power(model, params, beta_values, reps, bootstrap, seed, workers, alpha):
    rows = []
    results = {}
    for n1, n2, p in params:
        W = define_W(p)
        for beta in beta_values:
            print(f'Power: model={model}, n1={n1}, n2={n2}, p={p}, beta={beta}, reps={reps}, B={bootstrap}', flush=True)
            tasks = [(sim, model, n1, n2, p, W, bootstrap, beta, alpha, seed) for sim in range(reps)]
            rejections = run_tasks(one_power_replication, tasks, workers)
            for method in METHODS:
                estimated_power = sum(bool(r[method]) for r in rejections) / reps
                rows.append({'model': model, 'n1': n1, 'n2': n2, 'p': p, 'beta': beta,
                             'method': method, 'power': estimated_power, 'reps': reps,
                             'bootstrap': bootstrap, 'alpha': alpha})
                results[(n1, n2, p, beta, method)] = estimated_power
    return rows, results


def load_power(csv_path):
    with csv_path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    return {(int(r['n1']), int(r['n2']), int(r['p']), float(r['beta']), r['method']): float(r['power'])
            for r in rows}


def plot_power(model, section, params, beta_values, results, output_dir):
    cols = 2 if len(params) > 1 else 1
    rows = int(np.ceil(len(params) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)
    axes = axes.flat
    legend_handles, legend_labels = [], []
    markers = {'RI_W': 'o', 'TCQ': 's', 'New Bootstrap': '^',
               'Wang Bootstrap': 'D', 'Empirical Bootstrap': 'v', 'Wild Bootstrap': '<'}
    for idx, (n1, n2, p) in enumerate(params):
        ax = axes[idx]
        for method in METHODS:
            style = METHOD_STYLES[method]
            ys = [results[(n1, n2, p, beta, method)] for beta in beta_values]
            line, = ax.plot(beta_values, ys, label=style['label'],
                            color=style['color'], linestyle=style['linestyle'],
                            marker=markers[method], linewidth=1.5, alpha=0.75)
            if idx == 0:
                legend_handles.append(line)
                legend_labels.append(style['label'])
        ax.set(title=f'$n_1={n1}, n_2={n2}, p={p}$', xlabel=r'$\beta$', ylabel='Power')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(beta_values)
        ax.grid(alpha=0.3)
    for ax in list(axes)[len(params):]:
        ax.set_visible(False)
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=len(METHODS),
               bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    basename = output_dir / f'power_{section}_model{model}_curves'
    fig.savefig(basename.with_suffix('.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(basename.with_suffix('.eps'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {basename}.pdf and .eps')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--section', choices=['main', 'supplementary'], default='main')
    parser.add_argument('--models', type=int, nargs='+', choices=[1, 2, 3], default=[1, 2, 3])
    parser.add_argument('--params', nargs='+', metavar='n1,n2,p')
    parser.add_argument('--beta', type=float, nargs='+', default=[1, 1.5, 2, 2.5, 3])
    parser.add_argument('--reps', type=int, default=1000)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/power'))
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if args.reps <= 0 or args.bootstrap <= 0 or args.workers <= 0 or not 0 < args.alpha < 1 or any(b < 0 for b in args.beta):
        parser.error('Invalid reps, bootstrap, workers, alpha, or beta.')
    params = parse_parameter_grid(args.params, args.section)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        if model == 3 and any(p % 4 for _, _, p in params):
            parser.error('Model 3 requires p divisible by 4.')
        result_path = args.output_dir / f'power_{args.section}_model{model}_results.csv'
        if args.plot_only:
            results = load_power(result_path)
        else:
            rows, results = run_power(model, params, args.beta, args.reps, args.bootstrap,
                                      args.seed, args.workers, args.alpha)
            with result_path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        plot_power(model, args.section, params, args.beta, results, args.output_dir)
        print(f'Finished Model {model} ({MODEL_NAMES[model]}): {result_path}', flush=True)


if __name__ == '__main__':
    main()
