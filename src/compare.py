"""Compare the proposed model against results reported in the literature on KDDTest+.

Values for prior work are the overall accuracies reported in the cited papers
(see report/REPORT.md, References). Our numbers are read from results/tables.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURES, REPORT_FIGURES, TABLES
from evaluate import PALETTE

LITERATURE = [
    # reference, method, binary acc, 5-class acc
    ("Tavallaee et al. (2009) [1]", "Naive Bayes", 76.56, None),
    ("Tavallaee et al. (2009) [1]", "SVM", 69.52, None),
    ("Tavallaee et al. (2009) [1]", "Multi-layer Perceptron", 77.41, None),
    ("Tavallaee et al. (2009) [1]", "J48 decision tree", 81.05, None),
    ("Tavallaee et al. (2009) [1]", "Random Forest", 80.67, None),
    ("Tavallaee et al. (2009) [1]", "NB Tree", 82.02, None),
    ("Javaid et al. (2016) [4]", "Self-taught learning (sparse AE + softmax)", 88.39, 79.10),
    ("Yin et al. (2017) [5]", "RNN-IDS", 83.28, 81.29),
    ("Shone et al. (2018) [6]", "Stacked NDAE + RF", None, 85.42),
    ("Su et al. (2020) [11]", "BAT-MC (BLSTM + attention)", None, 84.25),
    ("Ieracitano et al. (2020) [12]", "Statistical analysis + autoencoder", 87.00, None),
]


def main():
    bin_df = pd.read_csv(TABLES / "results_binary.csv", index_col=0)
    mc_df = pd.read_csv(TABLES / "results_multiclass.csv", index_col=0)
    ours_bin = bin_df.loc["Proposed AESE-NIDS", "accuracy"] * 100
    ours_mc = mc_df.loc["Proposed AESE-NIDS", "accuracy"] * 100

    rows = [dict(reference=r, method=m, binary_accuracy=b, multiclass_accuracy=c) for r, m, b, c in LITERATURE]
    rows.append(dict(reference="This work (2026)", method="Proposed AESE-NIDS (HRA-FS + AE + stacking)",
                     binary_accuracy=round(ours_bin, 2), multiclass_accuracy=round(ours_mc, 2)))
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "literature_comparison.csv", index=False)
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, title in zip(axes, ["binary_accuracy", "multiclass_accuracy"],
                              ["Binary accuracy on KDDTest+ (%)", "5-class accuracy on KDDTest+ (%)"]):
        sub = df.dropna(subset=[col]).copy()
        sub["label"] = sub["method"] + "\n" + sub["reference"].str.replace(r" \[\d+\]", "", regex=True)
        colors = [PALETTE[3] if "This work" in r else PALETTE[0] for r in sub["reference"]]
        ax.barh(sub["label"], sub[col], color=colors)
        for i, v in enumerate(sub[col]):
            ax.text(v + 0.3, i, f"{v:.2f}", va="center", fontsize=8)
        ax.set_xlim(60, 100); ax.set_title(title); ax.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout()
    for d in (FIGURES, REPORT_FIGURES):
        fig.savefig(d / "fig_literature_comparison.png", dpi=200, bbox_inches="tight")
    print("saved fig_literature_comparison.png")


if __name__ == "__main__":
    main()
