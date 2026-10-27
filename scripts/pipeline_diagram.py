"""
Nested cross-validation workflow diagram (paper Fig. 1).
Pure drawing - does not need the dataset. Sized for the full page width of the IEEE template (7.16 in).

Run:  python pipeline_diagram.py
Output: figures/pipeline.png   (runs instantly)
Needs: pip install matplotlib
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs("figures", exist_ok=True)

# (box width, text, fill colour) - the inner-loop box is shaded slightly darker
boxes = [
    (15, "Framingham data\n4,240 participants\n15 predictors", "#E9EEF3"),
    (14, "Outer loop:\nstratified 10-fold\nsplit", "#E9EEF3"),
    (26, "Inner loop (training part only)\nimpute, scale, SMOTE, model\n3-fold random search,\nscored by AUC", "#DCE8F5"),
    (15, "Refit best settings\non the outer\ntraining part", "#E9EEF3"),
    (18, "Score on held-out fold\n(never resampled);\nmean and SD over folds", "#E9EEF3"),
]

fig, ax = plt.subplots(figsize=(7.16, 1.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 24)
ax.axis("off")

gap = (100 - sum(w for w, _, _ in boxes)) / (len(boxes) - 1)   # equal space between boxes
x = 0
for i, (w, text, colour) in enumerate(boxes):
    # rounded box
    ax.add_patch(FancyBboxPatch((x, 3), w, 18, boxstyle="round,pad=0.0,rounding_size=1.2",
                                fc=colour, ec="#5B6B7C", lw=0.9))
    # centred text
    ax.text(x + w / 2, 12, text, ha="center", va="center", fontsize=6.3, color="#1F2933", linespacing=1.35)
    # arrow to the next box
    if i < len(boxes) - 1:
        ax.add_patch(FancyArrowPatch((x + w + 0.2, 12), (x + w + gap - 0.2, 12), arrowstyle="-|>",
                                     mutation_scale=8, lw=1.0, color="#5B6B7C"))
    x += w + gap

fig.savefig("figures/pipeline.png", dpi=250, bbox_inches="tight", pad_inches=0.03)
plt.show()
