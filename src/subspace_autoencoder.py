"""KitNET-style subspace autoencoder ensemble (after Mirsky et al., NDSS 2018), adapted
for offline NSL-KDD features.

Kitsune's idea: instead of one autoencoder over all n features, cluster the features by
correlation distance into k groups of at most m features (agglomerative clustering on
the correlation matrix, breaking the largest link until every cluster fits), train one
small autoencoder per group on benign traffic (hidden size ceil(beta * dim)), and feed
the 0-1 normalised RMSE of each group into an output autoencoder whose own RMSE is the
final anomaly score.

What we keep from the paper: correlation-distance clustering with a max group size m,
per-group three-layer autoencoders with beta = 0.75, RMSE as the error signal, output
autoencoder over the normalised group RMSEs.
What we change: training is offline (we have the whole benign training split), using
scikit-learn's MLPRegressor instead of their online SGD, and we expose the per-group
RMSE vector as features for a downstream classifier, which Kitsune does not do.
"""
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.neural_network import MLPRegressor


def correlation_groups(X, m=10, min_var=1e-8):
    """Cluster feature indices into groups of at most m features by correlation distance."""
    X = np.asarray(X, dtype=np.float64)
    var = X.var(axis=0)
    live = np.flatnonzero(var > min_var)
    dead = np.flatnonzero(var <= min_var)
    C = np.corrcoef(X[:, live], rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), method="average")
    # Kitsune: cut the dendrogram until no cluster is larger than m
    groups = None
    for n_clusters in range(1, len(live) + 1):
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        sizes = np.bincount(labels)[1:]
        if sizes.max() <= m:
            groups = [live[labels == c] for c in range(1, labels.max() + 1)]
            break
    if dead.size:
        groups.append(dead)   # constant columns go into one harmless group
    return [np.asarray(g) for g in groups if len(g) > 0]


class KitNETLite:
    def __init__(self, m=10, beta=0.75, epochs=40, random_state=42):
        self.m, self.beta, self.epochs, self.random_state = m, beta, epochs, random_state

    def _make_ae(self, dim):
        hidden = max(1, int(np.ceil(self.beta * dim)))
        return MLPRegressor(hidden_layer_sizes=(hidden,), activation="relu", solver="adam",
                            learning_rate_init=1e-3, batch_size=256, max_iter=self.epochs,
                            early_stopping=True, validation_fraction=0.1, n_iter_no_change=6,
                            random_state=self.random_state)

    def fit(self, X_normal):
        X_normal = np.asarray(X_normal, dtype=np.float32)
        self.groups_ = correlation_groups(X_normal, m=self.m)
        self.ensemble_ = []
        for g in self.groups_:
            ae = self._make_ae(len(g)).fit(X_normal[:, g], X_normal[:, g])
            self.ensemble_.append(ae)
        Z = self._group_rmse(X_normal)
        self.z_min_, self.z_max_ = Z.min(axis=0), Z.max(axis=0)
        Zn = self._norm(Z)
        self.output_ = self._make_ae(Zn.shape[1]).fit(Zn, Zn)
        s = self.score(X_normal)
        self.threshold_ = float(np.quantile(s, 0.95))
        return self

    def _group_rmse(self, X):
        X = np.asarray(X, dtype=np.float32)
        out = np.empty((len(X), len(self.groups_)), dtype=np.float32)
        for i, (g, ae) in enumerate(zip(self.groups_, self.ensemble_)):
            rec = np.asarray(ae.predict(X[:, g])).reshape(len(X), -1)   # 1-feature groups return 1-D
            out[:, i] = np.sqrt(np.mean((X[:, g] - rec) ** 2, axis=1))
        return out

    def _norm(self, Z):
        return np.clip((Z - self.z_min_) / np.maximum(self.z_max_ - self.z_min_, 1e-9), 0, 1)

    def group_scores(self, X):
        """0-1 normalised per-subspace RMSE vector (k columns)."""
        return self._norm(self._group_rmse(X)).astype(np.float32)

    def score(self, X):
        """Output-layer RMSE: the KitNET anomaly score."""
        Zn = self.group_scores(X)
        rec = np.asarray(self.output_.predict(Zn)).reshape(len(Zn), -1)
        return np.sqrt(np.mean((Zn - rec) ** 2, axis=1))

    def predict(self, X):
        return (self.score(X) > self.threshold_).astype(int)
