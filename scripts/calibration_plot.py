"""
Calibration curves from pooled out-of-fold predictions (paper Fig. 4).

Run:  python calibration_plot.py   (framingham_heart_study.csv must be in the same folder)
Output: figures/calibration.png    (takes ~3-4 minutes because of nested cross-validation)
Needs: pip install scikit-learn imbalanced-learn xgboost matplotlib pandas scipy
"""
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

SEED = 42
N_JOBS = 2
os.makedirs("figures", exist_ok=True)

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

# ---- 3. The three models shown in the figure: name -> (pipeline, search space, candidates)
SPECS = {
    "Logistic Regression": (pipe(LogisticRegression(max_iter=3000, random_state=SEED)),
                            {"m__C": loguniform(1e-3, 1e1)}, 8),
    "Logistic Regression (no SMOTE)": (pipe(LogisticRegression(max_iter=3000, random_state=SEED), smote=False),
                                       {"m__C": loguniform(1e-3, 1e1)}, 8),
    "XGBoost": (pipe(XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=N_JOBS)),
                {"m__n_estimators": [100, 200, 300], "m__learning_rate": loguniform(0.02, 0.2),
                 "m__max_depth": [2, 3, 4], "m__subsample": [0.7, 0.85, 1.0],
                 "m__colsample_bytree": [0.7, 0.85, 1.0]}, 8),
}

def fit_tuned(name, Xa, ya, seed):
    """Inner loop: 3-fold random search scored by AUC, then refit best settings on the training part."""
    base, dist, n_iter = SPECS[name]
    rs = RandomizedSearchCV(clone(base), dist, n_iter=n_iter, scoring="roc_auc",
                            cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                            random_state=seed, n_jobs=N_JOBS, refit=False)
    rs.fit(Xa, ya)
    return clone(base).set_params(**rs.best_params_).fit(Xa, ya)

# ---- 4. Outer loop: 10-fold nested CV, collect out-of-fold probabilities for every participant
folds = list(StratifiedKFold(10, shuffle=True, random_state=SEED).split(X, y))
oof = {name: np.zeros(len(y)) for name in SPECS}
for name in SPECS:
    for k, (tr, te) in enumerate(folds):
        est = fit_tuned(name, X.iloc[tr], y.iloc[tr], SEED + k)
        oof[name][te] = est.predict_proba(X.iloc[te])[:, 1]
    print(f"done: {name}  Brier = {brier_score_loss(y, oof[name]):.3f}")

# ---- 5. Calibration plot: 10 bins with equal numbers of people (quantile bins)
plt.figure(figsize=(5.5, 5))
for name in SPECS:
    frac_pos, mean_pred = calibration_curve(y, oof[name], n_bins=10, strategy="quantile")
    plt.plot(mean_pred, frac_pos, marker="o", label=f"{name} (Brier={brier_score_loss(y, oof[name]):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")     # perfectly calibrated line
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed CHD fraction")
plt.title("Calibration (out-of-fold predictions)")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("figures/calibration.png", dpi=200)
plt.show()
