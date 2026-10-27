"""
Per-fold AUC box plot of the tuned models (paper Fig. 3), from 10-fold nested cross-validation.
Sized for one column of the IEEE two-column template (3.45 in).

Run:  python auc_boxplot.py         (framingham_heart_study.csv must be in the same folder)
Output: figures/auc_boxplot.png     (takes ~10 minutes because of nested cross-validation)
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
from sklearn.metrics import roc_auc_score
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

# ---- 5. AUC of every model in each of the 10 outer folds
auc = pd.DataFrame({name: [roc_auc_score(y.iloc[te], oof[name][te]) for tr, te in folds] for name in SPECS})
print("\nMean AUC over folds:\n", auc.mean().round(3).sort_values(ascending=False))
auc.round(4).to_csv("per_fold_auc.csv", index_label="fold")

# ---- 6. Horizontal box plots with the 10 fold values as dots, ordered by median
short = {"Logistic Regression": "LR", "Logistic Regression (no SMOTE)": "LR (no SMOTE)", "KNN": "k-NN",
         "SVM (RBF)": "SVM", "Random Forest": "RF", "Gradient Boosting": "GB", "XGBoost": "XGBoost"}
order = list(auc.median().sort_values().index)          # lowest median at the bottom
fig, ax = plt.subplots(figsize=(3.45, 2.2))
box = ax.boxplot([auc[c] for c in order], vert=False, widths=0.55, patch_artist=True,
                 medianprops=dict(color="#1F2933", lw=1.2), flierprops=dict(markersize=3))
for patch in box["boxes"]:
    patch.set(facecolor="#DCE8F5", edgecolor="#5B6B7C")
jitter = np.random.RandomState(1)
for i, c in enumerate(order, start=1):
    ax.scatter(auc[c], i + jitter.uniform(-0.15, 0.15, len(auc)), s=7, color="#2F5D8A", zorder=3, alpha=0.8)
ax.set_yticks(range(1, len(order) + 1))
ax.set_yticklabels([short[c] for c in order], fontsize=7)
ax.set_xlabel("ROC-AUC per outer fold", fontsize=7.5)
ax.tick_params(axis="x", labelsize=7)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig("figures/auc_boxplot.png", dpi=250)
plt.show()
