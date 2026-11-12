"""Score new NSL-KDD-formatted records with the trained AESE-NIDS.

Usage:
    python src/predict.py data/raw/KDDTest+.txt [--out predictions.csv]
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from config import COLUMNS, MODELS
from preprocessing import FEATURE_COLUMNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV with 41 NSL-KDD features (label columns optional)")
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    art = joblib.load(MODELS / "aese_nids.joblib")
    df = pd.read_csv(args.csv, header=None)
    df.columns = COLUMNS[: df.shape[1]]

    X_full = art["preprocessor"].transform(df[FEATURE_COLUMNS]).astype(np.float32)
    ae_score = np.clip(art["autoencoder"].score(X_full) / art["ae_scale"], 0, 1).reshape(-1, 1).astype(np.float32)
    X = np.hstack([X_full[:, art["feature_mask"]], ae_score])

    out = pd.DataFrame({
        "attack_probability": art["binary_model"].predict_proba(X)[:, 1],
        "is_attack": art["binary_model"].predict(X),
        "category": art["label_encoder"].inverse_transform(art["multiclass_model"].predict(X)),
        "ae_anomaly_score": ae_score.ravel(),
    })
    out.to_csv(args.out, index=False)
    print(out.head(10)); print(f"\n{len(out)} records scored -> {args.out}")
    print(out["category"].value_counts())


if __name__ == "__main__":
    main()
