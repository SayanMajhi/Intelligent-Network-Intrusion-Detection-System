"""Hybrid Rank-Aggregated Feature Selection (HRA-FS).

Three complementary relevance views are computed on the training split:
  1. Mutual information with the class label     (information-theoretic, non-linear)
  2. ANOVA F-statistic                            (statistical, linear separability)
  3. Extra-Trees impurity importance              (embedded, interaction-aware)

Each view produces a ranking; the rankings are fused with a Borda count so that
a feature that is consistently useful under all three views rises to the top.
The top-k features are retained.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif

from config import RANDOM_STATE


def _rank(scores):
    """Rank 1 = best. Ties broken by order."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def hybrid_rank_selection(X, y, feature_names, k=40, random_state=RANDOM_STATE):
    mi = mutual_info_classif(X, y, random_state=random_state, n_neighbors=5)
    f_stat, _ = f_classif(X, y)
    f_stat = np.nan_to_num(f_stat, nan=0.0, posinf=0.0)
    et = ExtraTreesClassifier(n_estimators=200, random_state=random_state, n_jobs=-1,
                              class_weight="balanced")
    et.fit(X, y)
    imp = et.feature_importances_

    r_mi, r_f, r_et = _rank(mi), _rank(f_stat), _rank(imp)
    n = len(feature_names)
    # Borda count: higher is better
    borda = (n - r_mi) + (n - r_f) + (n - r_et)

    table = pd.DataFrame({
        "feature": feature_names,
        "mutual_info": mi, "rank_mi": r_mi,
        "anova_f": f_stat, "rank_f": r_f,
        "et_importance": imp, "rank_et": r_et,
        "borda_score": borda,
    }).sort_values("borda_score", ascending=False).reset_index(drop=True)
    selected_idx = table.index[:k].to_numpy()
    selected_names = table["feature"].iloc[:k].to_numpy()
    selected_mask = np.isin(feature_names, selected_names)
    return selected_mask, table
