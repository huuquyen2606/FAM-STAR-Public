"""Preprocessing for the CICAndMal2020 tabular dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class PreparedData:
    """Train/test arrays and fitted preprocessing objects."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_train_raw_shape: tuple[int, int]
    X_test_raw_shape: tuple[int, int]
    label_encoder: LabelEncoder
    imputer: SimpleImputer
    selector: VarianceThreshold
    scaler: StandardScaler
    selected_feature_cols: list[str]

    @property
    def preprocessors(self) -> dict[str, Any]:
        return {
            "imputer": self.imputer,
            "selector": self.selector,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
        }


def preprocess_dataframe(
    df: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.2,
    label_col: str = "Label",
) -> PreparedData:
    """Split, encode, impute, select, and scale the dataset.

    The imputer, feature selector, and scaler are fit on training data only,
    then applied to the test data, matching the original notebooks.
    """
    drop_cols = ["Hash", "Category", "Family", label_col]
    feature_cols = [column for column in df.columns if column not in drop_cols]

    X_raw = df[feature_cols].copy()
    y_raw = df[label_col].copy()
    X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw.values,
        y_encoded,
        test_size=test_size,
        random_state=seed,
        stratify=y_encoded,
    )

    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_test_imp = imputer.transform(X_test_raw)

    selector = VarianceThreshold(threshold=0.0)
    X_train_sel = selector.fit_transform(X_train_imp)
    X_test_sel = selector.transform(X_test_imp)

    selected_feature_cols = np.array(feature_cols)[selector.get_support()].tolist()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_sel)
    X_test = scaler.transform(X_test_sel)

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        label_encoder=label_encoder,
        X_train_raw_shape=X_train_raw.shape,
        X_test_raw_shape=X_test_raw.shape,
        imputer=imputer,
        selector=selector,
        scaler=scaler,
        selected_feature_cols=selected_feature_cols,
    )
