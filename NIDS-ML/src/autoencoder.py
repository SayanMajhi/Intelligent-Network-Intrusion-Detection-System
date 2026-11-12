"""Normal-profile autoencoder used as an anomaly scorer.

The autoencoder is trained ONLY on benign training traffic. At inference the
per-sample reconstruction error is large for traffic that does not resemble the
benign profile, which gives the downstream ensemble a signal that generalises to
attack types never seen during supervised training.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor

from config import AE_EPOCHS, AE_HIDDEN, RANDOM_STATE


class NormalProfileAutoencoder:
    def __init__(self, hidden=AE_HIDDEN, epochs=AE_EPOCHS, random_state=RANDOM_STATE):
        self.model = MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                                  solver="adam", learning_rate_init=1e-3,
                                  batch_size=256, max_iter=epochs,
                                  early_stopping=True, validation_fraction=0.1,
                                  n_iter_no_change=8, random_state=random_state)
        self.threshold_ = None

    def fit(self, X_normal, quantile=0.95):
        self.model.fit(X_normal, X_normal)
        err = self.score(X_normal)
        self.threshold_ = float(np.quantile(err, quantile))
        return self

    def score(self, X):
        recon = self.model.predict(X)
        return np.mean((X - recon) ** 2, axis=1)

    def predict(self, X):
        """1 = anomaly (attack), 0 = normal."""
        return (self.score(X) > self.threshold_).astype(int)

    @property
    def loss_curve_(self):
        return self.model.loss_curve_
