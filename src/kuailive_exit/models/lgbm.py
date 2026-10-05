"""LightGBM fitting and prediction helpers."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd


def fit_classifier(x, y, x_validation, y_validation, params: dict,
                   *, n_estimators: int, early_stopping_rounds: int,
                   seed: int = 20261001, threads: int = 4):
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=n_estimators, random_state=seed,
        n_jobs=threads, verbosity=-1, **params)
    model.fit(x, y, eval_set=[(x_validation, y_validation)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
    return model


def predict_probability(model, frame: pd.DataFrame, feature_names: list[str], batch_size: int = 25_000):
    predictions = []
    for start in range(0, len(frame), batch_size):
        matrix = frame.iloc[start:start + batch_size][feature_names].to_numpy(dtype=np.float32, copy=True)
        predictions.append(model.predict_proba(matrix)[:, 1])
    return np.concatenate(predictions) if predictions else np.empty(0, dtype=float)
