# Code for High-Dimensional Two-Sample Testing

This repository contains the Python code and datasets used for the numerical simulations and real-data analyses in our paper. The code implements six testing procedures and includes empirical size, power, and sparse-alternative experiments.

## File Description

| File | Description |
|---|---|
| `simulation_core.py` | Core test statistics, bootstrap procedures, and data-generating models |
| `size.py` | Empirical size and p-value distributions |
| `power.py` | Power analysis under dense alternatives |
| `sparse_power.py` | Power analysis under sparse alternatives |
| `real_data_alon.py` | Analysis of the Alon et al. (1999) dataset |
| `real_data_tian.py` | Analysis of the Tian et al. (2003) dataset |
| `data/` | Datasets for the real-data analyses |
| `requirements.txt` | Required Python packages |

## Simulation Settings

Three data-generating models are considered:

- **Model 1:** Independent Gaussian observations with identity covariance matrices.
- **Model 2:** Correlated Gaussian observations with compound-symmetric covariance matrices.
- **Model 3:** Non-Gaussian observations with factor-structured covariance matrices.

All three models are available for the size and power experiments.

## Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Running the Code

The simulation experiments can be reproduced using the following commands:

**Empirical size**

```bash
python size.py --section main --models 1 2 3 --reps 1000 --bootstrap 1000 --workers 4
```

**Power under dense alternatives**

```bash
python power.py --section main --models 1 2 3 --reps 1000 --bootstrap 1000 --workers 4
```

**Power under sparse alternatives**

```bash
python sparse_power.py --models 1 2 3 --reps 1000 --bootstrap 1000 --workers 4
```

For supplementary experiments, replace `--section main` with `--section supplementary` in the size and dense-power commands.

## Real-Data Analysis

The datasets are provided in the `data/` directory. The real-data analyses can be reproduced by running:

```bash
python real_data_alon.py
python real_data_tian.py --data-dir data/tian --output outputs/tian_pvalues.csv
```

All commands should be executed from the repository's root directory.