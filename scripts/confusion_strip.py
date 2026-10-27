"""
Compact one-row confusion matrices (paper Fig. 5), pooled out-of-fold predictions at the 0.5 threshold.
Sized for the full page width of the IEEE two-column template (7.16 in).

Run:  python confusion_strip.py      (framingham_heart_study.csv must be in the same folder)
Output: figures/confusion_strip.png  (takes ~10 minutes because of nested cross-validation)
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
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
    """Inner loop: 3-fold random search scored by AUC, then refit best settings on the training part."""
    base, dist, n_iter = SPECS[name]
    rs = RandomizedSearchCV(clone(base), dist, n_iter=n_iter, scoring="roc_auc",
                            cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                            random_state=seed, n_jobs=N_JOBS, refit=False)
    rs.fit(Xa, ya)
    est = clone(base).set_params(**rs.best_params_)
    if name == "SVM (RBF)":
        est.set_params(m__probability=True)   # SVM needs probabilities for the ROC curve
    return est.fit(Xa, ya)

# ---- 4. Outer loop: 10-fold nested CV, collect out-of-fold probabilities for every participant
folds = list(StratifiedKFold(10, shuffle=True, random_state=SEED).split(X, y))
oof = {name: np.zeros(len(y)) for name in SPECS}
for name in SPECS:
    for k, (tr, te) in enumerate(folds):
        est = fit_tuned(name, X.iloc[tr], y.iloc[tr], SEED + k)
        oof[name][te] = est.predict_proba(X.iloc[te])[:, 1]
    print(f"done: {name}  pooled AUC = {roc_auc_score(y, oof[name]):.3f}")

# ---- 5. One row of 7 small confusion matrices
# Numbers are participant counts; shading is the share of each TRUE class (row-normalised)
order = ["Logistic Regression", "Logistic Regression (no SMOTE)", "SVM (RBF)", "KNN",
         "Random Forest", "Gradient Boosting", "XGBoost"]
short = {"Logistic Regression": "LR", "Logistic Regression (no SMOTE)": "LR (no SMOTE)", "SVM (RBF)": "SVM",
         "KNN": "k-NN", "Random Forest": "RF", "Gradient Boosting": "GB", "XGBoost": "XGBoost"}

fig, axes = plt.subplots(1, 7, figsize=(7.16, 1.45))
for ax, name in zip(axes, order):
    mat = confusion_matrix(y, (oof[name] >= 0.5).astype(int)).astype(float)   # [[TN, FP], [FN, TP]]
    print(f"{name:32s} TN={int(mat[0,0]):5d} FP={int(mat[0,1]):5d} FN={int(mat[1,0]):4d} TP={int(mat[1,1]):4d}")
    share = mat / mat.sum(axis=1, keepdims=True)
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{int(v):,}", ha="center", va="center", fontsize=6.3,
                color="white" if share[i, j] > 0.6 else "#1F2933")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No", "CHD"], fontsize=6)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No", "CHD"] if name == order[0] else ["", ""], fontsize=6)
    ax.set_title(short[name], fontsize=6.8, pad=2)
    ax.tick_params(length=1.5, pad=1)
axes[0].set_ylabel("True", fontsize=6.5, labelpad=1)
fig.text(0.5, 0.0, "Predicted", ha="center", fontsize=6.5)
fig.tight_layout(pad=0.3, w_pad=0.4)
fig.subplots_adjust(bottom=0.2)
fig.savefig("figures/confusion_strip.png", dpi=250)
plt.show()
