"""Draws the block diagram of the proposed AESE-NIDS as a PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from config import FIGURES


def box(ax, x, y, w, h, text, fc="#E8F0FE", ec="#1A56DB", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, p, q, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#333", **kw))


fig, ax = plt.subplots(figsize=(14, 7.2))
ax.set_xlim(-0.2, 13.8); ax.set_ylim(0, 7.2); ax.axis("off")

# Row 1: data + preprocessing
box(ax, 0.3, 5.6, 2.1, 1.0, "NSL-KDD traffic records\n(41 features)\nKDDTrain+ / KDDTest+", fc="#FFF4E5", ec="#B45309", bold=True)
box(ax, 2.9, 5.6, 2.4, 1.0, "Pre-processing\n• log1p on heavy-tailed counters\n• Min-Max scaling\n• One-hot (protocol, service, flag)")
box(ax, 5.8, 5.6, 3.0, 1.0, "Hybrid Rank-Aggregated Feature Selection\nMI rank + ANOVA-F rank + Extra-Trees rank\n→ Borda fusion → top-k = 40")
box(ax, 9.3, 5.6, 3.4, 1.0, "Benign-only anomaly features (all 122 columns)\n• single AE 122-32-8-32-122 → 1 score (5-class task)\n• 31 subspace AEs, m=10 → error profile (binary task)", fc="#ECFDF5", ec="#047857")
arrow(ax, (2.4, 6.1), (2.9, 6.1)); arrow(ax, (5.3, 6.1), (5.8, 6.1)); arrow(ax, (5.3, 6.4), (9.3, 6.4), connectionstyle="arc3,rad=-0.25")

# Fused representation
box(ax, 4.2, 3.95, 4.6, 0.85, "Fused representation: 40 selected features + anomaly features\n(41-D for 5-class, 40 + 31 = 71-D for binary)", fc="#F3E8FF", ec="#6D28D9", bold=True)
arrow(ax, (7.3, 5.6), (6.5, 4.75)); arrow(ax, (11.0, 5.6), (8.5, 4.75))

# Level-0 learners
names = ["Random Forest\n(300 trees, balanced)", "Extra Trees\n(300 trees, balanced)",
         "XGBoost\n(400 rounds, depth 8)", "LightGBM\n(400 rounds, 63 leaves)"]
xs = [1.2, 4.1, 7.0, 9.9]
for x, n in zip(xs, names):
    box(ax, x, 2.3, 2.5, 0.9, n, fc="#E0F2FE", ec="#0369A1")
    arrow(ax, (6.5, 4.0), (x + 1.25, 3.2))
ax.text(1.2, 3.3, "Level-0 learners  (dual-regime OOF: stratified folds + attack-type-grouped folds)", fontsize=8, va="bottom", color="#444")

# Meta learner
box(ax, 3.2, 0.9, 6.0, 0.8, "Level-1 meta-learner: Logistic Regression\non stacked OOF probabilities + anomaly features", fc="#FEE2E2", ec="#B91C1C", bold=True)
for x in xs:
    arrow(ax, (x + 1.25, 2.3), (6.2, 1.7))

# Output
box(ax, 10.2, 0.85, 3.3, 0.9, "Decision\n• Binary: Normal / Attack\n• 5-class: Normal, DoS, Probe, R2L, U2R", fc="#FFF4E5", ec="#B45309", bold=True)
arrow(ax, (9.2, 1.3), (10.2, 1.3))
# anomaly score also reaches the meta-learner directly
arrow(ax, (12.9, 5.6), (12.9, 2.0), connectionstyle="arc3,rad=0")
arrow(ax, (12.9, 2.0), (9.2, 1.55))
ax.text(13.0, 3.9, "anomaly\nscore", fontsize=8, color="#047857")

ax.set_title("Figure 1. Block diagram of the proposed AESE-NIDS (Autoencoder-augmented Stacked Ensemble)", fontsize=11, pad=8)
fig.tight_layout()
fig.savefig(FIGURES / "fig_system_diagram.png", dpi=200, bbox_inches="tight")
print("saved fig_system_diagram.png")
