"""Empirical size and ECDF study for all three models.

Examples:
  python size.py --section main --models 1 2 3
  python size.py --section supplementary --models 1 2 3
  python size.py --models 1 2 3 --params 6,8,8 --reps 2 --bootstrap 3 --workers 1
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Safe for servers and headless replication.
import matplotlib.pyplot as plt
import numpy as np

from simulation_core import (METHODS, METHOD_STYLES, MODEL_NAMES,
                             define_W, generate_model_samples, method_pvalues,
                             parse_parameter_grid, run_tasks)


def one_size_replication(sim, model, n1, n2, p, W, bootstrap, seed):
    # The original notebook uses seed 42 + sim.
    np.random.seed(seed + sim)
    x1, x2, _, _ = generate_model_samples(model, n1, n2, p)
    return method_pvalues(x1, x2, W, bootstrap)


def run_size(model, params, reps, bootstrap, seed, workers, output_dir, alpha):
    all_pvalues = []
    summary = []
    for n1, n2, p in params:
        print(f'Size: model={model}, n1={n1}, n2={n2}, p={p}, reps={reps}, B={bootstrap}', flush=True)
        W = define_W(p)
        tasks = [(sim, model, n1, n2, p, W, bootstrap, seed) for sim in range(reps)]
        outcomes = run_tasks(one_size_replication, tasks, workers)
        pvals = {method: np.array([item[method] for item in outcomes], dtype=float) for method in METHODS}
        file = output_dir / f'size_model{model}_n1{n1}_n2{n2}_p{p}_pvalues.npz'
        np.savez_compressed(file, **pvals)
        for method in METHODS:
            size = float(np.mean(pvals[method] < alpha))
            summary.append({'model': model, 'n1': n1, 'n2': n2, 'p': p,
                            'method': method, 'size': size, 'reps': reps,
                            'bootstrap': bootstrap, 'alpha': alpha})
            print(f'  {method}: {size:.4f}')
        all_pvalues.append(pvals)
    return all_pvalues, summary


def load_size(model, params, output_dir, alpha):
    all_pvalues = []
    summary = []
    for n1, n2, p in params:
        file = output_dir / f'size_model{model}_n1{n1}_n2{n2}_p{p}_pvalues.npz'
        with np.load(file, allow_pickle=False) as saved:
            pvals = {method: saved[method].copy() for method in METHODS}
        for method in METHODS:
            summary.append({'model': model, 'n1': n1, 'n2': n2, 'p': p,
                            'method': method, 'size': float(np.mean(pvals[method] < alpha)),
                            'reps': len(pvals[method]), 'bootstrap': '', 'alpha': alpha})
        all_pvalues.append(pvals)
    return all_pvalues, summary


def plot_ecdf(model, section, params, all_pvalues, output_dir):
    cols = 2 if len(params) > 1 else 1
    rows = int(np.ceil(len(params) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False)
    axes = axes.flat
    legend_handles = []
    legend_labels = []
    for idx, ((n1, n2, p), pvals) in enumerate(zip(params, all_pvalues)):
        ax = axes[idx]
        for method in METHODS:
            sorted_vals = np.sort(pvals[method])
            y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            style = METHOD_STYLES[method]
            line, = ax.step(sorted_vals, y, where='post', label=style['label'],
                            color=style['color'], linestyle=style['linestyle'], linewidth=1.4)
            if idx == 0:
                legend_handles.append(line)
                legend_labels.append(style['label'])
        ax.set(title=f'$n_1={n1}, n_2={n2}, p={p}$', xlabel='p-value', ylabel='ECDF')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.grid(alpha=0.3)
    for ax in list(axes)[len(params):]:
        ax.set_visible(False)
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=len(METHODS),
               bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path = output_dir / f'size_{section}_model{model}_ecdf.pdf'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--section', choices=['main', 'supplementary'], default='main')
    parser.add_argument('--models', type=int, nargs='+', choices=[1, 2, 3], default=[1, 2, 3])
    parser.add_argument('--params', nargs='+', metavar='n1,n2,p',
                        help='Optional override; e.g. --params 10,15,200 20,30,200')
    parser.add_argument('--reps', type=int, default=1000)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/size'))
    parser.add_argument('--plot-only', action='store_true', help='Read previously saved NPZ results')
    args = parser.parse_args()
    if args.reps <= 0 or args.bootstrap <= 0 or args.workers <= 0 or not 0 < args.alpha < 1:
        parser.error('reps, bootstrap, and workers must be positive; alpha must lie in (0,1).')
    params = parse_parameter_grid(args.params, args.section)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        if any(p % 4 for _, _, p in params) and model == 3:
            parser.error('Model 3 requires p divisible by 4.')
        if args.plot_only:
            data, summary = load_size(model, params, args.output_dir, args.alpha)
        else:
            data, summary = run_size(model, params, args.reps, args.bootstrap,
                                     args.seed, args.workers, args.output_dir, args.alpha)
        summary_path = args.output_dir / f'size_{args.section}_model{model}_summary.csv'
        with summary_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        plot_ecdf(model, args.section, params, data, args.output_dir)
        print(f'Finished Model {model} ({MODEL_NAMES[model]}): {summary_path}', flush=True)


if __name__ == '__main__':
    main()
