"""
Stacking ensemble + all tuned models in one nested cross-validation run.
Saves oof_predictions.csv and per_fold_auc.csv, then draws roc.png, auc.png and confusion.png
(the three figures that include the ensemble) using ensemble_figures_from_results.py.

Run:  python stacking_ensemble.py   (framingham_heart_study.csv and ensemble_figures_from_results.py in the same folder)
Takes ~20 minutes on a 2-core machine (Colab free tier).
Needs: pip install scikit-learn imbalanced-learn xgboost matplotlib pandas scipy
"""
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

SEED = 42
N_JOBS = 2

# ---- 1. Load data
df = pd.read_csv("framingham_heart_study.csv").drop_duplicates()
y = df.pop("TenYearCHD").astype(int)
X = df

# ---- 2. Pipeline: impute -> scale -> (SMOTE) -> model, all fitted on training folds only
def pipe(model, smote=True):
    steps = [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
    if smote:
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("m", model))
    return Pipeline(steps)

# ---- 3. Models and hyperparameter search spaces: name -> (pipeline, search space, candidates)
SPECS = {
    "Logistic Regression": (pipe(LogisticRegression(max_iter=3000, random_state=SEED)),
                            {"m__C": loguniform(1e-3, 1e1)}, 8),
    "Logistic Regression (no SMOTE)": (pipe(LogisticRegression(max_iter=3000, random_state=SEED), smote=False),
                                       {"m__C": loguniform(1e-3, 1e1)}, 8),
    "KNN": (pipe(KNeighborsClassifier()),
            {"m__n_neighbors": randint(5, 60), "m__weights": ["uniform", "distance"]}, 8),
    "SVM (RBF)": (pipe(SVC(probability=False, random_state=SEED)),
                  {"m__C": loguniform(1e-2, 1e1), "m__gamma": loguniform(1e-4, 1e-1)}, 6),
    "Random Forest": (pipe(RandomForestClassifier(random_state=SEED, n_jobs=N_JOBS)),
                      {"m__n_estimators": [100, 200], "m__max_depth": [3, 5, 8, None],
                       "m__min_samples_leaf": [1, 5, 10, 20]}, 8),
    "Gradient Boosting": (pipe(GradientBoostingClassifier(random_state=SEED)),
                          {"m__n_estimators": [100, 150], "m__learning_rate": loguniform(0.02, 0.2),
                           "m__max_depth": [2, 3], "m__subsample": [0.7, 0.85, 1.0]}, 8),
    "XGBoost": (pipe(XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=N_JOBS)),
                {"m__n_estimators": [100, 200, 300], "m__learning_rate": loguniform(0.02, 0.2),
                 "m__max_depth": [2, 3, 4], "m__subsample": [0.7, 0.85, 1.0],
                 "m__colsample_bytree": [0.7, 0.85, 1.0]}, 8),
}

def fit_tuned(name, Xa, ya, seed):
    """Inner loop: 3-fold random search scored by AUC; returns the best settings."""
    base, dist, n_iter = SPECS[name]
    rs = RandomizedSearchCV(clone(base), dist, n_iter=n_iter, scoring="roc_auc",
                            cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                            random_state=seed, n_jobs=N_JOBS, refit=False)
    rs.fit(Xa, ya)
    return rs.best_params_

def build(name, params):
    est = clone(SPECS[name][0]).set_params(**params)
    if name == "SVM (RBF)":
        est.set_params(m__probability=True)   # probabilities are needed for AUC and for stacking
    return est

# The six SMOTE-trained models are the base learners of the stacking ensemble
BASES = ["Logistic Regression", "KNN", "SVM (RBF)", "Random Forest", "Gradient Boosting", "XGBoost"]
def stacker(params, seed):
    return StackingClassifier(
        estimators=[(n.replace(" ", "_"), build(n, params[n])) for n in BASES],
        final_estimator=LogisticRegression(max_iter=1000),          # meta-learner, original class balance
        cv=StratifiedKFold(5, shuffle=True, random_state=seed),     # 5-fold out-of-fold base predictions
        stack_method="predict_proba")

# ---- 4. Outer loop: 10-fold nested CV, collect out-of-fold probabilities for every participant
folds = list(StratifiedKFold(10, shuffle=True, random_state=SEED).split(X, y))
names = list(SPECS) + ["Stacking Ensemble"]
oof = {name: np.zeros(len(y)) for name in names}
for k, (tr, te) in enumerate(folds):
    params = {}
    for name in SPECS:
        params[name] = fit_tuned(name, X.iloc[tr], y.iloc[tr], SEED + k)
        oof[name][te] = build(name, params[name]).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    oof["Stacking Ensemble"][te] = stacker(params, SEED + k).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    print(f"fold {k + 1}/10 done", flush=True)

# ---- 5. Save results and draw the figures
auc = pd.DataFrame({n: [roc_auc_score(y.iloc[te], oof[n][te]) for tr, te in folds] for n in names})
print("\nMean AUC over folds:\n", auc.mean().round(3).sort_values(ascending=False))
auc.round(4).to_csv("per_fold_auc.csv", index_label="fold")
pd.DataFrame(oof).assign(y=y.values).to_csv("oof_predictions.csv", index=False)
exec(open("ensemble_figures_from_results.py").read())
