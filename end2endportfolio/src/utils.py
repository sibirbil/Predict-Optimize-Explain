# -*- coding: utf-8 -*-
"""
Utilities (Python 3.9 compatible typing)
"""

from typing import Any, Optional, Union, Tuple, List, Dict, Callable

import jax
import jax.numpy as jnp
from jax import random as random
from jax.tree_util import tree_map
import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin

######################
## SCHEDULERS ET AL ##
######################

def as_scheduler(value):
    """
    Turns a scalar into a constant step-size function, or returns as-is if callable.
    """
    if callable(value):
        return value
    return lambda step: value

def power_decay(
    init_lr: float,           # starting learning rate
    alpha: float,             # decay rate exponent
    offset: float = 1.0,      # in case step count starts from 0
    rate: Union[int, float] = 100,   # how many steps per decay horizon
) -> Callable[[int], float]:
    """
    Returns a scheduler decaying by 1/(step/rate + offset)^alpha.
    """
    def schedule(step: int) -> float:
        return init_lr / ((step / rate + offset) ** alpha)
    return schedule

def sqrt_decay(init_lr: float) -> Callable[[int], float]:
    return power_decay(init_lr, 0.5)

def harmonic_decay(init_lr: float) -> Callable[[int], float]:
    return power_decay(init_lr, 1.0)

##############################
## MAKE PARAMS a TRAJECTORY ##
##############################

def make_traj(params):
    """
    Add a vacuous leading dimension so it can be fed into F function.
    """
    return tree_map(lambda x: jnp.expand_dims(x, 0), params)

###########################
## DATA PREPARATION UTILS ##
###########################

def _make_probability(weights: jnp.ndarray) -> jnp.ndarray:
    """
    Clip to [0,1] then normalize to sum=1 (with tiny epsilon).
    """
    weights = jnp.maximum(0.0, jnp.minimum(1.0, weights))
    total = jnp.sum(weights) + 1e-4
    return weights / total

def _re_one_hotify_one_row(key, weights: jnp.ndarray) -> jnp.ndarray:
    """
    Return a one-hot vector with probabilities given by 'weights'.
    """
    probs = _make_probability(weights)
    a = random.choice(key, len(weights), p=probs)
    return jax.nn.one_hot(a, num_classes=len(weights))

def re_one_hotify_probabilistic(key, X: jax.Array, idx: slice) -> jax.Array:
    """
    Replace columns X[:, idx] by one-hots sampled according to rowwise weights in that slice.
    """
    coi = X[:, idx]  # columns of interest (weights per row)
    keys = random.split(key, X.shape[0])
    one_hots = jax.vmap(_re_one_hotify_one_row)(keys, coi)  # (keys, weights)
    return X.at[:, idx].set(one_hots)

def re_one_hotify_argmax(X: jax.Array, idx: slice) -> jax.Array:
    """
    Replace columns X[:, idx] by deterministic one-hot of argmax along that slice.
    """
    coi = X[:, idx]
    one_hots = jax.vmap(lambda v: (v >= jnp.max(v)).astype(float))(coi)
    return X.at[:, idx].set(one_hots)

def get_indices_with_prefix(df: pd.DataFrame, prefix: str) -> slice:
    """
    Returns a slice of indices of columns that start with the given prefix.
    All such columns are assumed contiguous.
    """
    prefix_cols = df.columns[df.columns.str.startswith(prefix)]
    idx = [df.columns.get_loc(col) for col in prefix_cols]
    if not idx:
        raise ValueError(f"No columns start with prefix '{prefix}'.")
    return slice(idx[0], idx[-1] + 1)

def _modify_to_series(X: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """
    Ensure we operate on a single Series.
    """
    if isinstance(X, pd.DataFrame) and X.shape[1] > 1:
        raise ValueError("WeightedImputer can only fit/transform a single column.")
    if isinstance(X, pd.DataFrame):
        X = X.iloc[:, 0]
    return X

class WeightedImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.category_frequencies_: Optional[pd.Series] = None

    def fit(self, X: Union[pd.DataFrame, pd.Series], _: Optional[Any] = None):
        Xs = _modify_to_series(X)
        self.category_frequencies_ = Xs.value_counts(normalize=True)
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        if self.category_frequencies_ is None:
            raise RuntimeError("WeightedImputer must be fit() before transform().")

        Xs = _modify_to_series(X)

        # Ensure categorical/object dtype
        if not np.issubdtype(Xs.dtype, np.object_):
            raise ValueError("WeightedImputer can only be applied to categorical/object data.")

        missing_mask = Xs.isna()
        X_copy = Xs.copy()

        if missing_mask.any():
            imputed_values = np.random.choice(
                self.category_frequencies_.index,
                size=int(missing_mask.sum()),
                p=self.category_frequencies_.values,
            )
            X_copy.loc[missing_mask] = imputed_values

        return X_copy.to_frame()

    def get_feature_names_out(self, input_features: Optional[Any] = None):
        return input_features

def fn_from_dict(d: Dict[Any, Any]) -> Callable[[np.ndarray], np.ndarray]:
    """
    Vectorize a dict lookup over a numpy array.
    """
    f = lambda k: d.get(k, k)
    def F(X: np.ndarray) -> np.ndarray:
        return np.vectorize(f)(X)
    return F
