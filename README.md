# Insurance-Claims-Forecasting

A machine learning pipeline for forecasting the number of car insurance claims per policy period, built on the French motor third-party liability dataset (`freMTPL2freq`).

## Problem

Accurately predicting claim frequency is critical for insurance providers to optimise premium pricing, manage risk reserves, and identify high-risk customers. This project compares a classical statistical model (Poisson GLM) against a gradient boosting model (XGBoost) on this task.

## Dataset

**freMTPL2freq** — a publicly available French motor insurance dataset containing policy-level features and observed claim counts.

Key variables:
- `ClaimNb` — number of claims (target)
- `Exposure` — fraction of year the policy was active
- `BonusMalus`, `DrivAge`, `VehAge`, `VehPower`, `Density`, `VehBrand`, `Region`, `Area`, `VehGas`

## Project Structure

| Phase | Description |
|---|---|
| 1 | Data loading and inspection |
| 2 | Target variable EDA (`ClaimNb`, `Exposure`, overdispersion check) |
| 3 | Univariate EDA across all features |
| 4 | Multivariate EDA |
| 5 | Poisson GLM |
| 6 | XGBoost |
| 7 | Model comparison and error analysis |

## Methodology

### Poisson GLM
- Features one-hot encoded; numeric features standard scaled
- Regularisation strength (`alpha`) tuned via 5-fold cross-validation grid search
- Exposure used as `sample_weight` during fitting
- Evaluated on MAE, RMSE, and Mean Poisson Deviance

### XGBoost
- Custom feature engineering: log-transform of `Density`, interaction flags (`Is_New_Veh`, `Young_HighPower`)
- Native categorical encoding (no one-hot encoding needed)
- Hyperparameters tuned via Bayesian search (`BayesSearchCV`, 20 iterations, 3-fold CV) optimising Mean Poisson Deviance
- `count:poisson` objective; predictions multiplied by `Exposure` to yield expected claim counts

### Error Analysis
Both models are evaluated on their top 10 largest residuals, with breakdowns by driver age, vehicle brand, region, and bonus-malus class to understand where each model struggles.

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scikit-optimize
xgboost
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scikit-optimize xgboost
```

## Usage

1. Download `freMTPL2freq.csv` from [OpenML](https://www.openml.org/d/41214) or the `CASdatasets` R package
2. Place it in the same directory as the notebook
3. Run all cells in order

## Results

Both models are compared on the held-out test set using MAE, RMSE, Mean Poisson Deviance, and Total Deviance. Feature importance is extracted from the GLM coefficients and XGBoost's built-in importance scores to provide actionable insights into the key risk drivers.
