from typing import Literal

import numpy as np
import pandas as pd


def portfolio_return(X: pd.DataFrame, weights: list):
    return (weights*np.array(X.mean())).sum()


def portfolio_volatility(
        X: pd.DataFrame,
        weights: list,
        measure: Literal['std', 'var'] = 'std'):
    
    weights = np.array(weights) if not isinstance(weights, np.ndarray) else weights

    if measure=='std':
        return np.sqrt(np.dot(weights.T, np.dot(X.cov(), weights)))
    elif measure=='var':
        return np.dot(weights.T, np.dot(X.cov(), weights))
    else:
        raise ValueError(f"Argument `measure` should be equal 'std' or 'var', "
                         f"meanwhile {measure} has been given.")
