"""Experiments inspired by three papers. Every result row is appended to
experiments/results.csv so nothing gets lost between runs.

E0  baseline        : the pipeline from the report (single AE error score, novelty-aware stack)
E1  Tavallaee 2009  : same models scored on KDDTest-21 (records with difficulty < 21)
E2  Shone 2018      : add the autoencoder bottleneck code (8 dims) as classifier features
E3  Mirsky 2018     : KitNET-style subspace AEs; per-group RMSE vector -> meta-learner
E4  Mirsky 2018     : KitNET output-layer score as a single anomaly feature (drop-in for E0's AE)
E5  E3 + E0         : both the single AE score and the group RMSE vector
E6  Kitsune m sweep : max subspace size m in {5, 10, 20} (binary only)
E7  operating points: detection rate of E0/E3 at fixed false-alarm budgets (free, from probabilities)
"""
import json
import warnings
warnings.filterwarnings("ignore")
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve

from autoencoder import NormalProfileAutoencoder
from config import CLASS_ORDER, N_SELECTED_FEATURES, RANDOM_STATE
from data_loader import load_nsl_kdd
from evaluate import compute_metrics
from feature_selection import hybrid_rank_selection
from subspace_autoencoder import KitNETLite
from models import build_aese
from preprocessing import preprocess

EXP = ROOT / "experiments"
RESULTS = EXP / "results.csv"
ONLY = sys.argv[1:]   # optional: run only these experiment ids


def log(exp, source, config, task, test_set, m, extra=None, train_s=np.nan, notes=""):
    row = {"exp": exp, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "paper": source,
           "config": config, "task": task, "test_set": test_set,
           "accuracy": m.get("accuracy"), "precision": m.get("precision"), "recall": m.get("recall"),
           "f1": m.get("f1"), "mcc": m.get("mcc"), "far": m.get("false_alarm_rate"),
           "unseen_dr": (extra or {}).get("unseen_dr"), "train_s": train_s, "notes": notes}
    df = pd.DataFrame([row])
    df.to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)
    print(f"  -> {exp:4s} {task:8s} {test_set:10s} acc={m['accuracy']:.4f} f1={m['f1']:.4f}"
          + (f" unseenDR={extra['unseen_dr']:.4f}" if extra and 'unseen_dr' in extra else ""))


def want(e):
    return not ONLY or e in ONLY


# ----------------------------------------------------------------------------- data
t0 = time.time()
train, test = load_nsl_kdd()
Xtr_full, Xte_full, names, _ = preprocess(train, test)
ytr_b, yte_b = train["is_attack"].to_numpy(), test["is_attack"].to_numpy()
le = LabelEncoder().fit(CLASS_ORDER)
ytr_m, yte_m = le.transform(train["category"]), le.transform(test["category"])
groups = train["label"].to_numpy()
unseen = (~test["label"].isin(set(train["label"]))).to_numpy()
test21 = (test["difficulty"] < 21).to_numpy()      # Tavallaee's KDDTest-21
print(f"KDDTest-21 has {test21.sum()} records ({test21.mean()*100:.1f}% of KDDTest+)")

mask, _ = hybrid_rank_selection(Xtr_full, ytr_m, names, k=N_SELECTED_FEATURES)
Ztr, Zte = Xtr_full[:, mask], Xte_full[:, mask]
benign = Xtr_full[ytr_b == 0]

# single AE (as in the report)
ae = NormalProfileAutoencoder().fit(benign)
a_tr, a_te = ae.score(Xtr_full), ae.score(Xte_full)
hi = np.quantile(a_tr, 0.999)
s_tr, s_te = np.clip(a_tr / hi, 0, 1)[:, None].astype(np.float32), np.clip(a_te / hi, 0, 1)[:, None].astype(np.float32)
# bottleneck code (Shone-style representation): run the encoder half of the MLP manually
def encode(model, X):
    h = np.asarray(X, dtype=np.float64)
    W, b = model.coefs_, model.intercepts_
    for i in range(2):                       # 122->32->8 : first two layers
        h = np.maximum(0, h @ W[i] + b[i])
    return h.astype(np.float32)
code_tr, code_te = encode(ae.model, Xtr_full), encode(ae.model, Xte_full)
cmax = np.maximum(code_tr.max(axis=0), 1e-6)
code_tr, code_te = code_tr / cmax, np.clip(code_te / cmax, 0, 5)
print(f"setup done in {time.time()-t0:.0f}s")


def evaluate(model, Xte, task, exp, source, config, train_s, notes=""):
    y = yte_b if task == "binary" else yte_m
    avg = "binary" if task == "binary" else "macro"
    p = model.predict(Xte)
    m = compute_metrics(y, p, average=avg)
    extra = {"unseen_dr": float(p[unseen].mean())} if task == "binary" else None
    log(exp, source, config, task, "KDDTest+", m, extra, train_s, notes)
    m21 = compute_metrics(y[test21], p[test21], average=avg)
    e21 = {"unseen_dr": float(p[unseen & test21].mean())} if task == "binary" else None
    log(exp + "/E1", "Tavallaee 2009", config, task, "KDDTest-21", m21, e21, train_s, "hard subset, difficulty<21")
    return p


def run(exp, source, config, Xtr, Xte, n_pass, tasks=("binary", "5-class"), notes=""):
    out = {}
    for task in tasks:
        nc = 2 if task == "binary" else 5
        y = ytr_b if task == "binary" else ytr_m
        t = time.time()
        model = build_aese(nc, novelty_aware=True, ae_passthrough=True, n_passthrough=n_pass)
        model.fit(Xtr, y, groups=groups)
        out[task] = (model, evaluate(model, Xte, task, exp, source, config, time.time() - t, notes))
    return out


# ----------------------------------------------------------------------------- E0
X0_tr, X0_te = np.hstack([Ztr, s_tr]), np.hstack([Zte, s_te])
if want("E0"):
    print("E0 baseline (report pipeline)")
    res0 = run("E0", "this report", "HRA-FS40 + AE err + novelty stack", X0_tr, X0_te, 1)

# ----------------------------------------------------------------------------- E2 Shone
if want("E2"):
    print("E2 Shone 2018: + AE bottleneck code as features")
    X2_tr, X2_te = np.hstack([Ztr, code_tr, s_tr]), np.hstack([Zte, code_te, s_te])
    run("E2", "Shone 2018", "HRA-FS40 + AE code(8) + AE err + novelty stack", X2_tr, X2_te, 1,
        notes="code = ReLU activations of the 8-unit bottleneck, scaled by train max")

# ----------------------------------------------------------------------------- E3/E4/E5 Kitsune
if want("E3") or want("E4") or want("E5"):
    t = time.time()
    kit = KitNETLite(m=10).fit(benign)
    g_tr, g_te = kit.group_scores(Xtr_full), kit.group_scores(Xte_full)
    k_tr, k_te = kit.score(Xtr_full), kit.score(Xte_full)
    khi = np.quantile(k_tr, 0.999)
    k_tr, k_te = np.clip(k_tr / khi, 0, 1)[:, None].astype(np.float32), np.clip(k_te / khi, 0, 1)[:, None].astype(np.float32)
    print(f"KitNET-lite: {len(kit.groups_)} subspaces (sizes {[len(g) for g in kit.groups_]}) in {time.time()-t:.0f}s")
    json.dump({"m": 10, "n_groups": len(kit.groups_), "groups": [list(map(str, names[g])) for g in kit.groups_]},
              open(EXP / "kitnet_groups_m10.json", "w"), indent=1)
    mk = compute_metrics(yte_b, kit.predict(Xte_full))
    log("E4u", "Mirsky 2018", "KitNET-lite alone, unsupervised, 95% benign threshold", "binary", "KDDTest+", mk,
        {"unseen_dr": float(kit.predict(Xte_full)[unseen].mean())}, time.time() - t)
    if want("E3"):
        print("E3 Mirsky 2018: per-subspace RMSE vector -> meta-learner")
        X3_tr, X3_te = np.hstack([Ztr, g_tr]), np.hstack([Zte, g_te])
        res3 = run("E3", "Mirsky 2018", f"HRA-FS40 + {g_tr.shape[1]} subspace RMSEs (m=10) + novelty stack", X3_tr, X3_te, g_tr.shape[1])
    if want("E4"):
        print("E4 Mirsky 2018: KitNET output score replaces single AE score")
        run("E4", "Mirsky 2018", "HRA-FS40 + KitNET output score + novelty stack", np.hstack([Ztr, k_tr]), np.hstack([Zte, k_te]), 1)
    if want("E5"):
        print("E5: single AE err + subspace RMSE vector")
        X5_tr, X5_te = np.hstack([Ztr, g_tr, s_tr]), np.hstack([Zte, g_te, s_te])
        run("E5", "Mirsky 2018 + this report", f"HRA-FS40 + {g_tr.shape[1]} subspace RMSEs + AE err + novelty stack", X5_tr, X5_te, g_tr.shape[1] + 1)

# ----------------------------------------------------------------------------- E6 m sweep
if want("E6"):
    for m in (5, 20):
        t = time.time()
        kit_m = KitNETLite(m=m).fit(benign)
        g_tr, g_te = kit_m.group_scores(Xtr_full), kit_m.group_scores(Xte_full)
        print(f"E6 m={m}: {len(kit_m.groups_)} subspaces")
        run(f"E6m{m}", "Mirsky 2018", f"HRA-FS40 + {g_tr.shape[1]} subspace RMSEs (m={m}) + novelty stack",
            np.hstack([Ztr, g_tr]), np.hstack([Zte, g_te]), g_tr.shape[1], tasks=("binary",))

# ----------------------------------------------------------------------------- E7 operating points
if want("E7") and want("E3"):
    if "res0" not in globals():      # resumed run: refit the E0 binary model without logging
        m0 = build_aese(2, novelty_aware=True, ae_passthrough=True, n_passthrough=1).fit(X0_tr, ytr_b, groups=groups)
        res0 = {"binary": (m0, None)}
    rows = []
    for name, res in (("E0 single AE", res0), ("E3 subspace RMSEs", res3)):
        model = res["binary"][0]
        Xte = X0_te if name.startswith("E0") else X3_te
        prob = model.predict_proba(Xte)[:, 1]
        fpr, tpr, thr = roc_curve(yte_b, prob)
        for budget in (0.01, 0.02, 0.05, 0.10):
            i = np.searchsorted(fpr, budget, side="right") - 1
            rows.append({"model": name, "far_budget": budget, "detection_rate": tpr[i], "threshold": thr[i]})
    pd.DataFrame(rows).to_csv(EXP / "operating_points.csv", index=False)
    print(pd.DataFrame(rows))

print(f"all done in {(time.time()-t0)/60:.1f} min")
