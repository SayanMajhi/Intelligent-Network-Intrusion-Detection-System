"""Encoding, log-scaling and normalisation of NSL-KDD features.

The transformer is fit on the training split only so that no information from
the evaluation split leaks into the model.
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder

from config import CATEGORICAL, COLUMNS, LOG_FEATURES

FEATURE_COLUMNS = [c for c in COLUMNS if c not in ("label", "difficulty")]
NUMERIC = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL]
NUMERIC_LOG = [c for c in NUMERIC if c in LOG_FEATURES]
NUMERIC_PLAIN = [c for c in NUMERIC if c not in LOG_FEATURES]


def build_preprocessor():
    log_pipe = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, validate=False)),
        ("scale", MinMaxScaler()),
    ])
    return ColumnTransformer([
        ("log", log_pipe, NUMERIC_LOG),
        ("num", MinMaxScaler(), NUMERIC_PLAIN),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


def get_feature_names(preprocessor):
    names = list(NUMERIC_LOG) + list(NUMERIC_PLAIN)
    ohe = preprocessor.named_transformers_["cat"]
    names += list(ohe.get_feature_names_out(CATEGORICAL))
    return np.array(names)


def preprocess(train_df, test_df):
    pre = build_preprocessor()
    X_train = pre.fit_transform(train_df[FEATURE_COLUMNS])
    X_test = pre.transform(test_df[FEATURE_COLUMNS])
    names = get_feature_names(pre)
    return X_train.astype(np.float32), X_test.astype(np.float32), names, pre
