# Simple Versus Complex Machine Learning Models for Coronary Heart Disease Risk Prediction

Code and results for the paper by Rishi Sharma, Nisarg Roy, Kushal Shah, Pranshu Vaidya and Dhairya Jingar
(Dept. of Computer Engineering, Karnavati University).

We compare logistic regression (with and without SMOTE), k-NN, SVM, random forest, gradient boosting,
XGBoost and a stacking ensemble for predicting ten-year coronary heart disease (CHD) on the Framingham
Heart Study extract, using leakage-free nested cross-validation (10 outer × 3 inner folds).

## Data
The dataset is **not included** in this repository. Download the Framingham Heart Study extract from Kaggle
and save it as `framingham_heart_study.csv` in the same folder as the scripts.
(4,240 participants, 15 predictors, outcome column `TenYearCHD`.)

## Setup
```
pip install -r requirements.txt
```
All randomness is fixed with seed 42. The library versions in `requirements.txt` reproduce the paper's numbers exactly.

## Scripts (`scripts/`)

| Script | Produces | Paper | Run time* |
|---|---|---|---|
| `pipeline_diagram.py` | `pipeline.png` | Fig. 1 | instant |
| `stacking_ensemble.py` | `roc.png`, `auc.png`, `confusion.png`, `per_fold_auc.csv`, `oof_predictions.csv` | Figs. 2, 3, 5; Tables II, III | ~20 min |
| `ensemble_figures_from_results.py` | redraws Figs. 2, 3, 5 from saved results | Figs. 2, 3, 5 | instant |
| `calibration_plot.py` | `calibration.png` | Fig. 4 | ~4 min |
| `shap_plot.py` | `shap_tuned_xgboost.png` | Fig. 6 | <1 min |
| `roc_curves.py`, `auc_boxplot.py`, `confusion_strip.py` | the same figures without the ensemble | — | ~10 min each |

*On a 2-core machine such as the Google Colab free tier.

## Results (`results/`)
- `cv_results_with_ensemble.csv` — nested cross-validation metrics for all eight models (Table II)
- `per_fold_auc.csv` — AUC of every model in each outer fold (Fig. 3)
- `significance_with_ensemble.csv` — Wilcoxon tests with Holm correction, logistic regression vs. each model (Table III)
- `holdout_with_ensemble.csv` — 20% hold-out results (Table V)
- `threshold_analysis.csv`, `threshold_ensemble.csv` — decision-threshold analysis
- `calibration_brier.csv` — Brier scores and pooled AUCs
- `lr_odds_ratios.csv`, `shap_importance_tuned.csv` — interpretation
- `meta_weights.csv` — stacking meta-learner weights
- `best_params_per_fold.csv` — hyperparameters selected in every outer fold
- `oof_predictions.csv` — pooled out-of-fold predicted probabilities
