"""Baseline classifiers and the proposed AE-augmented, novelty-aware stacked ensemble (AESE)."""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config import RANDOM_STATE, STACK_CV_FOLDS


def base_learners(n_classes):
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    return [
        ("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=RANDOM_STATE)),
        ("et", ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                    n_jobs=-1, random_state=RANDOM_STATE)),
        ("xgb", XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.08,
                              subsample=0.9, colsample_bytree=0.8, objective=objective,
                              tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE,
                              eval_metric="logloss")),
        ("lgbm", LGBMClassifier(n_estimators=400, num_leaves=63, learning_rate=0.05,
                                subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
                                class_weight="balanced", random_state=RANDOM_STATE,
                                n_jobs=-1, verbose=-1)),
    ]


class NoveltyAwareStacking(BaseEstimator, ClassifierMixin):
    """Stacked heterogeneous ensemble with attack-type-grouped out-of-fold predictions.

    Standard stacking builds the meta-training set with random stratified folds.
    On NSL-KDD the base learners are almost perfect on such folds, so the
    meta-learner never sees a situation in which they fail and learns to ignore
    any auxiliary signal (such as an anomaly score). Novelty-aware stacking
    instead assigns every *attack type* to exactly one fold: when a fold is held
    out, the base learners are trained without ever having seen those attack
    types. Their out-of-fold probabilities therefore reflect how they behave on
    novel attacks, and the meta-learner learns how much to trust each learner
    versus the autoencoder anomaly score (passed through as an extra meta-feature).

    Parameters
    ----------
    n_classes : 2 or 5
    novelty_aware : use attack-type-grouped folds (True) or stratified folds (False)
    ae_passthrough : append the last column of X (anomaly score) to the meta-features
    """

    def __init__(self, n_classes=2, n_folds=STACK_CV_FOLDS, novelty_aware=True,
                 ae_passthrough=True, n_passthrough=1, random_state=RANDOM_STATE):
        self.n_classes = n_classes
        self.n_folds = n_folds
        self.novelty_aware = novelty_aware
        self.ae_passthrough = ae_passthrough
        self.n_passthrough = n_passthrough      # how many trailing columns of X go straight to the meta-learner
        self.random_state = random_state

    # ------------------------------------------------------------------ folds
    def _fold_ids(self, y, groups):
        rng = np.random.RandomState(self.random_state)
        K = self.n_folds
        if not self.novelty_aware or groups is None:
            fold = np.empty(len(y), dtype=int)
            skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=self.random_state)
            for f, (_, te) in enumerate(skf.split(np.zeros(len(y)), y)):
                fold[te] = f
            return fold
        groups = np.asarray(groups)
        fold = np.empty(len(y), dtype=int)
        normal = groups == "normal"
        # attack types -> greedy balanced partition into K folds (largest first)
        labels, counts = np.unique(groups[~normal], return_counts=True)
        load = np.zeros(K)
        assign = {}
        for idx in np.argsort(-counts, kind="stable"):
            f = int(np.argmin(load))
            assign[labels[idx]] = f
            load[f] += counts[idx]
        fold[~normal] = [assign[g] for g in groups[~normal]]
        # benign records -> random balanced split
        nidx = np.flatnonzero(normal)
        rng.shuffle(nidx)
        fold[nidx] = np.arange(len(nidx)) % K
        self.fold_assignment_ = assign
        return fold

    # -------------------------------------------------------------------- fit
    def _oof(self, X, y, fold, C):
        oof = np.zeros((len(y), len(self.base_names_) * C), dtype=np.float32)
        for f in range(self.n_folds):
            tr, te = fold != f, fold == f
            tr_classes = np.unique(y[tr])
            remap = {c: i for i, c in enumerate(tr_classes)}   # contiguous labels for XGBoost
            y_tr = np.array([remap[v] for v in y[tr]])
            for m, (_, est) in enumerate(base_learners(len(tr_classes))):
                est = clone(est).fit(X[tr], y_tr)
                p = est.predict_proba(X[te])
                cols = m * C + np.searchsorted(self.classes_, tr_classes)
                oof[np.ix_(te, cols)] = p
        return oof

    def fit(self, X, y, groups=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        C = len(self.classes_)
        self.base_names_ = [n for n, _ in base_learners(self.n_classes)]
        # regime 1: stratified folds -> how the learners behave on familiar traffic
        self_novelty = self.novelty_aware and groups is not None
        oof_std = self._oof(X, y, self._stratified_folds(y), C)
        meta_X, meta_y = self._meta_features(oof_std, X), y
        if self_novelty:
            # regime 2: attack-type-grouped folds -> how they behave on novel attacks
            oof_grp = self._oof(X, y, self._fold_ids(y, groups), C)
            meta_X = np.vstack([meta_X, self._meta_features(oof_grp, X)])
            meta_y = np.concatenate([y, y])
        self.meta_ = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        self.meta_.fit(meta_X, meta_y)
        # refit every base learner on the full training set
        self.estimators_ = []
        for _, est in base_learners(self.n_classes):
            self.estimators_.append(clone(est).fit(X, y))
        return self

    def _stratified_folds(self, y):
        fold = np.empty(len(y), dtype=int)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        for f, (_, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            fold[te] = f
        return fold

    def _meta_features(self, oof, X):
        if self.ae_passthrough:
            return np.hstack([oof, X[:, -self.n_passthrough:]])
        return oof

    def _level0(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.hstack([est.predict_proba(X) for est in self.estimators_]).astype(np.float32)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        return self.meta_.predict_proba(self._meta_features(self._level0(X), X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def meta_weights(self):
        """Return a (n_meta_classes, n_meta_features) table of meta-learner coefficients."""
        names = [f"{b}:P({c})" for b in self.base_names_ for c in self.classes_]
        if self.ae_passthrough:
            names += [f"passthrough_{i}" for i in range(self.n_passthrough)] if self.n_passthrough > 1 else ["ae_score"]
        return names, self.meta_.coef_


def build_aese(n_classes, novelty_aware=True, ae_passthrough=True, n_passthrough=1):
    return NoveltyAwareStacking(n_classes=n_classes, novelty_aware=novelty_aware,
                                ae_passthrough=ae_passthrough, n_passthrough=n_passthrough)


def baseline_models(n_classes):
    """Classical baselines commonly reported in the NIDS literature."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Linear SVM": LinearSVC(C=0.5, class_weight="balanced", max_iter=5000),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=80, early_stopping=True,
                             random_state=RANDOM_STATE),
    }
    for name, est in base_learners(n_classes):
        pretty = {"rf": "Random Forest", "et": "Extra Trees", "xgb": "XGBoost", "lgbm": "LightGBM"}[name]
        models[pretty] = est
    return models
