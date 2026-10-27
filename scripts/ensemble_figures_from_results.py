"""
Draw the ROC curves, per-fold AUC box plot and confusion strip WITH the stacking ensemble,
from saved results (no retraining, runs instantly).

Needs in the same folder:  oof_predictions.csv  (pooled out-of-fold probabilities, one column per model + y)
                           per_fold_auc.csv     (AUC of every model in each of the 10 outer folds)
Output: roc.png, auc.png, confusion.png  (same names as in the Overleaf project)
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

oof = pd.read_csv("oof_predictions.csv")
y = oof.pop("y").values
auc = pd.read_csv("per_fold_auc.csv", index_col=0)
SHORT = {"Logistic Regression": "LR", "Logistic Regression (no SMOTE)": "LR (no SMOTE)", "KNN": "k-NN",
         "SVM (RBF)": "SVM", "Random Forest": "RF", "Gradient Boosting": "GB", "XGBoost": "XGBoost",
         "Stacking Ensemble": "Stack"}

# ---- ROC curves (pooled out-of-fold); the ensemble is drawn as a dashed black line
plt.figure(figsize=(6, 5))
for name in oof.columns:
    f, t, _ = roc_curve(y, oof[name])
    style = dict(color="black", ls="--", lw=1.6) if name == "Stacking Ensemble" else {}
    plt.plot(f, t, label=f"{name} (AUC={roc_auc_score(y, oof[name]):.3f})", **style)
plt.plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC curves (tuned, out-of-fold)")
plt.legend(fontsize=7); plt.tight_layout(); plt.savefig("roc.png", dpi=200); plt.close()

# ---- per-fold AUC box plot, ordered by median
order = list(auc.median().sort_values().index)
fig, ax = plt.subplots(figsize=(3.45, 2.4))
box = ax.boxplot([auc[c] for c in order], vert=False, widths=0.55, patch_artist=True,
                 medianprops=dict(color="#1F2933", lw=1.2), flierprops=dict(markersize=3))
for patch, c in zip(box["boxes"], order):
    patch.set(facecolor="#F3E3C3" if c == "Stacking Ensemble" else "#DCE8F5", edgecolor="#5B6B7C")
jitter = np.random.RandomState(1)
for i, c in enumerate(order, start=1):
    ax.scatter(auc[c], i + jitter.uniform(-0.15, 0.15, len(auc)), s=7, color="#2F5D8A", zorder=3, alpha=0.8)
ax.set_yticks(range(1, len(order) + 1)); ax.set_yticklabels([SHORT[c] for c in order], fontsize=7)
ax.set_xlabel("ROC-AUC per outer fold", fontsize=7.5); ax.tick_params(axis="x", labelsize=7)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
fig.tight_layout(); fig.savefig("auc.png", dpi=250); plt.close(fig)

# ---- one row of 8 confusion matrices at the 0.5 threshold
cm_order = ["Logistic Regression", "Logistic Regression (no SMOTE)", "Stacking Ensemble", "SVM (RBF)", "KNN",
            "Random Forest", "Gradient Boosting", "XGBoost"]
fig, axes = plt.subplots(1, 8, figsize=(7.16, 1.4))
for ax, name in zip(axes, cm_order):
    mat = confusion_matrix(y, (oof[name] >= 0.5).astype(int)).astype(float)
    share = mat / mat.sum(axis=1, keepdims=True)
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{int(v):,}", ha="center", va="center", fontsize=5.8,
                color="white" if share[i, j] > 0.6 else "#1F2933")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No", "CHD"], fontsize=5.8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No", "CHD"] if name == cm_order[0] else ["", ""], fontsize=5.8)
    ax.set_title(SHORT[name], fontsize=6.5, pad=2); ax.tick_params(length=1.5, pad=1)
axes[0].set_ylabel("True", fontsize=6.3, labelpad=1)
fig.text(0.5, 0.0, "Predicted", ha="center", fontsize=6.3)
fig.tight_layout(pad=0.3, w_pad=0.35); fig.subplots_adjust(bottom=0.2)
fig.savefig("confusion.png", dpi=250); plt.close(fig)
print("saved roc.png, auc.png, confusion.png")
