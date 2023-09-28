import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LognTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, period: int = 1) -> None:
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.log(X)
