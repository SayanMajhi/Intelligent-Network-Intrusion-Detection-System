"""End-to-end experiment runner for the proposed AESE-NIDS.

Pipeline
  1. Load NSL-KDD (KDDTrain+ / KDDTest+)
  2. Pre-process (log1p, min-max, one-hot)
  3. Hybrid Rank-Aggregated Feature Selection (top-k)
  4. Normal-profile autoencoder -> reconstruction-error feature
  5. Stacked heterogeneous ensemble (RF, ET, XGB, LGBM -> LR meta-learner)
  6. Evaluate binary and 5-class tasks against baselines + ablations
All tables are written to results/tables and all figures to results/figures.
"""
import json
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from autoencoder import NormalProfileAutoencoder
from subspace_autoencoder import KitNETLite
from config import (CLASS_ORDER, FIGURES, MODELS, N_SELECTED_FEATURES, RANDOM_STATE,
                    REPORT_FIGURES, SUBSPACE_MAX_SIZE, TABLES)
from data_loader import load_nsl_kdd
from evaluate import PALETTE, compute_metrics, plot_bar, plot_confusion, plot_roc
from feature_selection import hybrid_rank_selection
from models import baseline_models, build_aese
from preprocessing import preprocess

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_STATE)
timings = {}


def tic(name):
    timings[name] = time.perf_counter()


def toc(name):
    timings[name] = time.perf_counter() - timings[name]
    return timings[name]


def save_fig(fig, name):
    for d in (FIGURES, REPORT_FIGURES):
        fig.savefig(d / name, dpi=200)
    plt.close(fig)


def both(name):
    """Return the results path; the figure is copied to the report folder too."""
    return FIGURES / name


def copy_to_report(name):
    import shutil
    shutil.copy(FIGURES / name, REPORT_FIGURES / name)


# --------------------------------------------------------------------------- #
# 1. Data
# --------------------------------------------------------------------------- #
print("[1/6] Loading NSL-KDD ...")
train_df, test_df = load_nsl_kdd()
dist = pd.DataFrame({
    "KDDTrain+": train_df["category"].value_counts().reindex(CLASS_ORDER),
    "KDDTest+": test_df["category"].value_counts().reindex(CLASS_ORDER),
})
dist.loc["Total"] = dist.sum()
dist.to_csv(TABLES / "class_distribution.csv")
print(dist)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, col in zip(axes, ["KDDTrain+", "KDDTest+"]):
    vals = dist.loc[CLASS_ORDER, col]
    ax.bar(CLASS_ORDER, vals, color=PALETTE[:5])
    ax.set_yscale("log"); ax.set_title(f"{col} class distribution (log scale)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=8)
fig.tight_layout(); save_fig(fig, "fig_class_distribution.png")

unseen_attacks = sorted(set(test_df["label"]) - set(train_df["label"]))
n_unseen_rows = int(test_df["label"].isin(unseen_attacks).sum())

# --------------------------------------------------------------------------- #
# 2. Pre-processing
# --------------------------------------------------------------------------- #
print("[2/6] Pre-processing ...")
X_train_full, X_test_full, feat_names, preprocessor = preprocess(train_df, test_df)
y_train_bin, y_test_bin = train_df["is_attack"].to_numpy(), test_df["is_attack"].to_numpy()
groups_train = train_df["label"].to_numpy()   # attack type per record ("normal" for benign)
le = LabelEncoder().fit(CLASS_ORDER)
y_train_mc = le.transform(train_df["category"])
y_test_mc = le.transform(test_df["category"])
print(f"   encoded feature space: {X_train_full.shape[1]} columns")

# --------------------------------------------------------------------------- #
# 3. Hybrid rank-aggregated feature selection
# --------------------------------------------------------------------------- #
print("[3/6] Hybrid rank-aggregated feature selection ...")
tic("feature_selection")
mask, fs_table = hybrid_rank_selection(X_train_full, y_train_mc, feat_names, k=N_SELECTED_FEATURES)
toc("feature_selection")
fs_table.to_csv(TABLES / "feature_ranking.csv", index=False)
X_train_sel, X_test_sel = X_train_full[:, mask], X_test_full[:, mask]
sel_names = feat_names[mask]
print(f"   kept {mask.sum()} / {len(mask)} features")

fig, ax = plt.subplots(figsize=(8, 7))
top = fs_table.head(25).iloc[::-1]
ax.barh(top["feature"], top["borda_score"], color=PALETTE[0])
ax.set_xlabel("Borda aggregated score (MI + ANOVA-F + Extra-Trees)")
ax.set_title("Top-25 features by hybrid rank aggregation")
fig.tight_layout(); save_fig(fig, "fig_feature_ranking.png")

# --------------------------------------------------------------------------- #
# 4. Normal-profile autoencoder
# --------------------------------------------------------------------------- #
print("[4/6] Training normal-profile autoencoder ...")
tic("autoencoder")
# The autoencoder sees the FULL 122-column representation (benign records only):
# features that the supervised ranking discards can still describe the benign profile.
ae = NormalProfileAutoencoder().fit(X_train_full[y_train_bin == 0])
toc("autoencoder")
ae_train = ae.score(X_train_full)
ae_test = ae.score(X_test_full)
# Scale the anomaly score to [0,1] using training statistics
ae_hi = np.quantile(ae_train, 0.999)
ae_train_s = np.clip(ae_train / ae_hi, 0, 1).reshape(-1, 1).astype(np.float32)
ae_test_s = np.clip(ae_test / ae_hi, 0, 1).reshape(-1, 1).astype(np.float32)
X_train = np.hstack([X_train_sel, ae_train_s])
X_test = np.hstack([X_test_sel, ae_test_s])
all_names = np.append(sel_names, "ae_recon_error")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(ae.loss_curve_, color=PALETTE[0]); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE loss")
axes[0].set_title("Autoencoder training loss (benign traffic only)")
for cls, c in zip(CLASS_ORDER, PALETTE):
    sel = test_df["category"].to_numpy() == cls
    axes[1].hist(np.log10(ae_test[sel] + 1e-7), bins=60, alpha=0.55, label=cls, color=c, density=True)
axes[1].axvline(np.log10(ae.threshold_), color="k", ls="--", lw=1, label="95% benign threshold")
axes[1].set_xlabel("log10(reconstruction error)"); axes[1].set_ylabel("Density")
axes[1].set_title("Reconstruction error on KDDTest+ by class"); axes[1].legend(fontsize=8)
fig.tight_layout(); save_fig(fig, "fig_autoencoder.png")

ae_only_metrics = compute_metrics(y_test_bin, ae.predict(X_test_full))

# --------------------------------------------------------------------------- #
# 4b. Subspace anomaly profile (KitNET-style, Mirsky et al. 2018) -- v2, binary task
# --------------------------------------------------------------------------- #
print("[4b/6] Training subspace autoencoder ensemble (m=10) ...")
tic("subspace_ae")
kit = KitNETLite(m=SUBSPACE_MAX_SIZE).fit(X_train_full[y_train_bin == 0])
toc("subspace_ae")
g_train, g_test = kit.group_scores(X_train_full), kit.group_scores(X_test_full)
n_sub = g_train.shape[1]
print(f"   {n_sub} subspaces, sizes {[len(g) for g in kit.groups_]}")
X_train_b, X_test_b = np.hstack([X_train_sel, g_train]), np.hstack([X_test_sel, g_test])
sub_names = [f"subspace_{i}_rmse" for i in range(n_sub)]
json.dump({"m": SUBSPACE_MAX_SIZE, "n_groups": n_sub, "groups": [list(map(str, feat_names[g])) for g in kit.groups_]},
          open(TABLES / "subspace_groups.json", "w"), indent=1)
kit_only_metrics = compute_metrics(y_test_bin, kit.predict(X_test_full))

# Figure: mean normalised subspace RMSE per class (which part of a connection is abnormal)
prof = pd.DataFrame(g_test, columns=[f"S{i}" for i in range(n_sub)])
prof["class"] = test_df["category"].to_numpy()
prof = prof.groupby("class").mean().loc[CLASS_ORDER]
fig, ax = plt.subplots(figsize=(12, 3.8))
sns.heatmap(prof, cmap="magma", ax=ax, cbar_kws={"label": "mean normalised RMSE"})
ax.set_title("Subspace anomaly profile on KDDTest+: which feature subspaces reconstruct badly, per class")
ax.set_xlabel("subspace (correlation-clustered feature group, m=10)"); ax.set_ylabel("")
fig.tight_layout(); save_fig(fig, "fig_subspace_profile.png")

# --------------------------------------------------------------------------- #
# 5. Models
# --------------------------------------------------------------------------- #
def run_models(X_tr, y_tr, X_te, y_te, n_classes, tag, n_pass=1):
    average = "binary" if n_classes == 2 else "macro"
    rows, roc_curves, preds = [], [], {}
    models = baseline_models(n_classes)
    models["Proposed AESE-NIDS"] = build_aese(n_classes, n_passthrough=n_pass)
    for name, model in models.items():
        print(f"   [{tag}] training {name} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        if name.startswith("Proposed"):
            model.fit(X_tr, y_tr, groups=groups_train)
        else:
            model.fit(X_tr, y_tr)
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter(); y_pred = model.predict(X_te); t_pred = time.perf_counter() - t0
        m = compute_metrics(y_te, y_pred, average=average)
        m.update({"model": name, "train_time_s": t_fit, "test_time_s": t_pred,
                  "us_per_sample": 1e6 * t_pred / len(X_te)})
        rows.append(m); preds[name] = y_pred
        if n_classes == 2:
            if hasattr(model, "predict_proba"):
                score = model.predict_proba(X_te)[:, 1]
            else:
                score = model.decision_function(X_te)
            roc_curves.append((name, y_te, score))
        print(f"acc={m['accuracy']:.4f} f1={m['f1']:.4f} ({t_fit:.1f}s)")
    df = pd.DataFrame(rows).set_index("model")
    return df, roc_curves, preds, models


print("[5/6] Binary classification (Normal vs Attack) ...")
# v2: binary task uses the subspace anomaly profile (n_sub columns) instead of the single AE score
bin_df, bin_roc, bin_preds, bin_models = run_models(X_train_b, y_train_bin, X_test_b, y_test_bin, 2, "binary", n_pass=n_sub)
bin_df.loc["Autoencoder only (unsupervised)"] = {**ae_only_metrics, "train_time_s": timings["autoencoder"],
                                                 "test_time_s": np.nan, "us_per_sample": np.nan}
bin_df.loc["Subspace AE ensemble only (unsupervised)"] = {**kit_only_metrics, "train_time_s": timings["subspace_ae"],
                                                          "test_time_s": np.nan, "us_per_sample": np.nan}
bin_df.to_csv(TABLES / "results_binary.csv")
print(bin_df[["accuracy", "precision", "recall", "f1", "detection_rate", "false_alarm_rate"]].round(4))

print("[5/6] Multi-class classification (5 categories) ...")
mc_df, _, mc_preds, mc_models = run_models(X_train, y_train_mc, X_test, y_test_mc, 5, "5-class")
mc_df.to_csv(TABLES / "results_multiclass.csv")
print(mc_df[["accuracy", "precision", "recall", "f1", "mcc"]].round(4))

# Per-class report for the proposed model (5-class)
from sklearn.metrics import classification_report
rep = classification_report(y_test_mc, mc_preds["Proposed AESE-NIDS"], target_names=list(le.classes_),
                            output_dict=True, zero_division=0)
pd.DataFrame(rep).T.to_csv(TABLES / "per_class_report_aese.csv")

# Detection of attack types never seen in training
unseen_mask = test_df["label"].isin(unseen_attacks).to_numpy()
unseen_rows = []
for name in ["Random Forest", "XGBoost", "LightGBM", "Proposed AESE-NIDS"]:
    unseen_rows.append({"model": name,
                        "unseen_attack_detection_rate": float(bin_preds[name][unseen_mask].mean())})
unseen_df = pd.DataFrame(unseen_rows).set_index("model")
unseen_df.to_csv(TABLES / "unseen_attack_detection.csv")
print("\nUnseen-attack detection rate:\n", unseen_df.round(4))

# KDDTest-21 (Tavallaee et al. 2009): records that fewer than all 21 reference learners classified correctly
test21 = (test_df["difficulty"] < 21).to_numpy()
rows21 = []
for name in bin_preds:
    m = compute_metrics(y_test_bin[test21], bin_preds[name][test21])
    rows21.append({"model": name, "task": "binary", "accuracy": m["accuracy"], "f1": m["f1"],
                   "detection_rate": m["detection_rate"], "false_alarm_rate": m["false_alarm_rate"]})
for name in mc_preds:
    m = compute_metrics(y_test_mc[test21], mc_preds[name][test21], average="macro")
    rows21.append({"model": name, "task": "5-class", "accuracy": m["accuracy"], "f1": m["f1"],
                   "detection_rate": np.nan, "false_alarm_rate": np.nan})
test21_df = pd.DataFrame(rows21)
test21_df.to_csv(TABLES / "results_test21.csv", index=False)
print(f"\nKDDTest-21 ({test21.sum()} records):\n", test21_df[test21_df.model.str.startswith("Proposed")].round(4))

# Operating points: detection rate at fixed false-alarm budgets (binary, proposed and best baselines)
op_rows = []
for name in ["XGBoost", "LightGBM", "Random Forest", "Proposed AESE-NIDS"]:
    prob = bin_models[name].predict_proba(X_test_b)[:, 1]
    fpr, tpr, thr = roc_curve(y_test_bin, prob)
    for budget in (0.01, 0.02, 0.05, 0.10):
        i = np.searchsorted(fpr, budget, side="right") - 1
        op_rows.append({"model": name, "far_budget": budget, "detection_rate": tpr[i]})
pd.DataFrame(op_rows).to_csv(TABLES / "operating_points.csv", index=False)

# --------------------------------------------------------------------------- #
# 6. Ablation study + cross-validation + figures
# --------------------------------------------------------------------------- #
print("[6/6] Ablation study ...")
ablation = []
configs = {
    "A: all 122 features, RF": (X_train_full, X_test_full, "rf"),
    "B: HRA-FS (k=40), RF": (X_train_sel, X_test_sel, "rf"),
    "C: HRA-FS, standard stacking": (X_train_sel, X_test_sel, "stack_std_noae"),
    "D: HRA-FS + AE feature, RF": (X_train, X_test, "rf"),
    "E: HRA-FS + AE, standard stacking": (X_train, X_test, "stack_std"),
    "F: HRA-FS + AE score, novelty-aware stacking (v1)": (X_train, X_test, "stack_na"),
    "G: HRA-FS + subspace profile, novelty-aware stacking (v2)": (X_train_b, X_test_b, "stack_na_sub"),
}
from sklearn.ensemble import RandomForestClassifier
for cname, (Xtr, Xte, kind) in configs.items():
    for task, (ytr, yte, nc) in {"binary": (y_train_bin, y_test_bin, 2), "5-class": (y_train_mc, y_test_mc, 5)}.items():
        if kind == "rf":
            model = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=-1,
                                           random_state=RANDOM_STATE)
            model.fit(Xtr, ytr)
        elif kind == "stack_std_noae":
            model = build_aese(nc, novelty_aware=False, ae_passthrough=False); model.fit(Xtr, ytr)
        elif kind == "stack_std":
            model = build_aese(nc, novelty_aware=False, ae_passthrough=True); model.fit(Xtr, ytr)
        elif kind == "stack_na":
            model = build_aese(nc, novelty_aware=True, ae_passthrough=True); model.fit(Xtr, ytr, groups=groups_train)
        else:
            model = build_aese(nc, novelty_aware=True, ae_passthrough=True, n_passthrough=n_sub); model.fit(Xtr, ytr, groups=groups_train)
        m = compute_metrics(yte, model.predict(Xte), average="binary" if nc == 2 else "macro")
        ablation.append({"configuration": cname, "task": task, "accuracy": m["accuracy"], "f1": m["f1"]})
        print(f"   {cname:50s} {task:8s} acc={m['accuracy']:.4f} f1={m['f1']:.4f}")
ablation_df = pd.DataFrame(ablation)
ablation_df.to_csv(TABLES / "ablation.csv", index=False)

print("   5-fold stratified CV of proposed model on KDDTrain+ (binary) ...")
tic("cv")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = []
for tr_idx, te_idx in cv.split(X_train_b, y_train_bin):
    m = build_aese(2, n_passthrough=n_sub).fit(X_train_b[tr_idx], y_train_bin[tr_idx], groups=groups_train[tr_idx])
    cv_scores.append(float((m.predict(X_train_b[te_idx]) == y_train_bin[te_idx]).mean()))
cv_scores = np.array(cv_scores)
toc("cv")
pd.DataFrame({"fold": range(1, 6), "accuracy": cv_scores}).to_csv(TABLES / "cv_binary.csv", index=False)
print(f"   CV accuracy = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Figures
plot_confusion(y_test_bin, bin_preds["Proposed AESE-NIDS"], ["Normal", "Attack"],
               "Proposed AESE-NIDS: binary confusion matrix (KDDTest+)", FIGURES / "fig_cm_binary.png")
copy_to_report("fig_cm_binary.png")
plot_confusion(y_test_mc, mc_preds["Proposed AESE-NIDS"], list(le.classes_),
               "Proposed AESE-NIDS: 5-class confusion matrix (KDDTest+)", FIGURES / "fig_cm_multiclass.png")
copy_to_report("fig_cm_multiclass.png")
plot_roc(bin_roc, FIGURES / "fig_roc_binary.png"); copy_to_report("fig_roc_binary.png")
plot_bar(bin_df.drop(index=["Autoencoder only (unsupervised)", "Subspace AE ensemble only (unsupervised)"]), ["accuracy", "precision", "recall", "f1"],
         FIGURES / "fig_binary_comparison.png", "Binary classification on KDDTest+", ylim=(50, 100))
copy_to_report("fig_binary_comparison.png")
plot_bar(mc_df, ["accuracy", "precision", "recall", "f1"], FIGURES / "fig_multiclass_comparison.png",
         "5-class classification on KDDTest+ (macro-averaged P/R/F1)", ylim=(30, 100))
copy_to_report("fig_multiclass_comparison.png")

fig, ax = plt.subplots(figsize=(11, 5))
piv = ablation_df.pivot(index="configuration", columns="task", values="accuracy").loc[list(configs)] * 100
piv.plot(kind="barh", ax=ax, color=PALETTE[:2])
ax.set_xlabel("Accuracy on KDDTest+ (%)"); ax.set_ylabel(""); ax.set_title("Ablation study")
ax.set_xlim(60, 100)
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)
fig.tight_layout(); save_fig(fig, "fig_ablation.png")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, 6), cv_scores * 100, marker="o", color=PALETTE[0])
ax.axhline(cv_scores.mean() * 100, color=PALETTE[3], ls="--", label=f"mean = {cv_scores.mean()*100:.2f}%")
ax.set_xlabel("Fold"); ax.set_ylabel("Accuracy (%)"); ax.set_title("5-fold CV on KDDTrain+ (proposed, binary)")
ax.legend(); fig.tight_layout(); save_fig(fig, "fig_cv.png")

# Meta-learner weights (which base learner the stack trusts)
for tag, mdl in (("binary", bin_models["Proposed AESE-NIDS"]), ("multiclass", mc_models["Proposed AESE-NIDS"])):
    names, coef = mdl.meta_weights()
    pd.DataFrame(coef, columns=names, index=[f"class_{c}" for c in range(coef.shape[0])]).T.to_csv(
        TABLES / f"meta_learner_weights_{tag}.csv")
pd.Series(bin_models["Proposed AESE-NIDS"].fold_assignment_, name="fold").to_csv(TABLES / "novelty_fold_assignment.csv")

# Persist artefacts
joblib.dump({"preprocessor": preprocessor, "feature_mask": mask, "autoencoder": ae, "ae_scale": ae_hi,
             "subspace_ae": kit, "n_subspaces": n_sub,
             "binary_model": bin_models["Proposed AESE-NIDS"], "multiclass_model": mc_models["Proposed AESE-NIDS"],
             "label_encoder": le, "feature_names": all_names}, MODELS / "aese_nids.joblib")

summary = {
    "n_train": int(len(train_df)), "n_test": int(len(test_df)),
    "n_encoded_features": int(X_train_full.shape[1]), "n_selected_features": int(mask.sum()),
    "test21_binary": test21_df[(test21_df.model == "Proposed AESE-NIDS") & (test21_df.task == "binary")].iloc[0].to_dict(),
    "test21_multiclass": test21_df[(test21_df.model == "Proposed AESE-NIDS") & (test21_df.task == "5-class")].iloc[0].to_dict(),
    "unseen_attack_types": unseen_attacks, "n_unseen_test_rows": n_unseen_rows,
    "ae_threshold": ae.threshold_, "n_subspaces": int(n_sub), "timings_s": timings,
    "cv_accuracy_mean": float(cv_scores.mean()), "cv_accuracy_std": float(cv_scores.std()),
    "proposed_binary": bin_df.loc["Proposed AESE-NIDS"].to_dict(),
    "proposed_multiclass": mc_df.loc["Proposed AESE-NIDS"].to_dict(),
}
with open(TABLES / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)
print("\nDone. Tables -> results/tables, figures -> results/figures")
