#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 17:33:25 2025

@author: batuhanatas
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple


def clean_missing(data_3d: np.ndarray, dates: np.ndarray, name: str = 'dataset') -> List[np.ndarray]:
    """
    Cleans missing values (-99.99) from a 3D data array.

    Args:
        data_3d (np.ndarray): Raw data array of shape (T, N, D).
        dates (np.ndarray): Dates array (not used functionally here).
        name (str): Dataset name for logging/debugging.

    Returns:
        List[np.ndarray]: List of cleaned snapshots per timestamp, each of shape (N_t, D).
    """
    T, N, D = data_3d.shape
    clean_data = []
    for t in range(T):
        snapshot = data_3d[t]
        mask = np.all(snapshot != -99.99, axis=1)
        clean_snapshot = snapshot[mask]
        clean_data.append(clean_snapshot)
    return clean_data


def split_features_and_targets(clean_data: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Splits cleaned data into features (X) and returns (y) per timestamp.

    Args:
        clean_data (List[np.ndarray]): Cleaned snapshots per timestamp.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: X_list (features), y_list (returns)
    """
    X_list, y_list = [], []
    for snapshot in clean_data:
        y_t = snapshot[:, 0]
        X_t = snapshot[:, 1:]
        X_list.append(X_t)
        y_list.append(y_t)
    return X_list, y_list


def scale_features(
    X_train_list: List[np.ndarray],
    X_valid_list: List[np.ndarray],
    X_test_list: List[np.ndarray]
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], StandardScaler]:
    """
    Scales features using StandardScaler fit on all training data.

    Returns:
        Tuple of scaled X lists and the fitted scaler.
    """
    scaler = StandardScaler()
    all_train_X = np.vstack(X_train_list)
    scaler.fit(all_train_X)

    X_train_scaled = [scaler.transform(X) for X in X_train_list]
    X_valid_scaled = [scaler.transform(X) for X in X_valid_list]
    X_test_scaled  = [scaler.transform(X) for X in X_test_list]

    return X_train_scaled, X_valid_scaled, X_test_scaled, scaler
