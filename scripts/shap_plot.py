"""
SHAP summary plot for the tuned XGBoost model on the 20% hold-out set (paper Fig. 6).

Run:  python shap_plot.py           (framingham_heart_study.csv must be in the same folder)
Output: figures/shap_tuned_xgboost.png   (takes under a minute)
Needs: pip install scikit-learn imbalanced-learn xgboost shap matplotlib pandas scipy
"""
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

SEED = 42
N_JOBS = 2
os.makedirs("figures", exist_ok=True)

# ---- 1. Load data and make the 80/20 stratified hold-out split
df = pd.read_csv("framingham_heart_study.csv").drop_duplicates()
y = df.pop("TenYearCHD").astype(int)
X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# ---- 2. XGBoost pipeline: impute -> scale -> SMOTE -> model (fitted on the training part only)
pipeline = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("sc", StandardScaler()),
    ("smote", SMOTE(random_state=SEED)),
    ("m", XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=N_JOBS)),
])
search_space = {"m__n_estimators": [100, 200, 300], "m__learning_rate": loguniform(0.02, 0.2),
                "m__max_depth": [2, 3, 4], "m__subsample": [0.7, 0.85, 1.0],
                "m__colsample_bytree": [0.7, 0.85, 1.0]}

# ---- 3. Tune with 3-fold random search (8 candidates, scored by AUC), then refit on the training part
rs = RandomizedSearchCV(clone(pipeline), search_space, n_iter=8, scoring="roc_auc",
                        cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                        random_state=SEED, n_jobs=N_JOBS, refit=False)
rs.fit(X_train, y_train)
model = clone(pipeline).set_params(**rs.best_params_).fit(X_train, y_train)
print("Best settings:", rs.best_params_)

# ---- 4. Explain the hold-out predictions with SHAP
# SHAP needs the test data in the same (imputed + scaled) form the model saw
X_test_t = pd.DataFrame(model.named_steps["sc"].transform(model.named_steps["imp"].transform(X_test)),
                        columns=X.columns)
shap_values = shap.TreeExplainer(model.named_steps["m"]).shap_values(X_test_t)

# ---- 5. Summary (beeswarm) plot
plt.figure()
shap.summary_plot(shap_values, X_test_t, show=False)
plt.tight_layout()
plt.savefig("figures/shap_tuned_xgboost.png", dpi=200, bbox_inches="tight")
plt.show()

# Feature ranking by mean |SHAP| (the "SHAP rank" column in the odds-ratio table)
ranking = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns).sort_values(ascending=False)
print(ranking.round(4))
