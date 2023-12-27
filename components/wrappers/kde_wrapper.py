from copy import deepcopy
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.bandwidths import bw_silverman, bw_scott


class KernelDensityWrapper:
    
    def __init__(self, *,
                 bandwidth: float | Literal['scott', 'silverman'] = 1,
                 algorithm: Literal['kd_tree', 'ball_tree', 'auto'] = "auto", 
                 kernel: Literal['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'] = "gaussian",
                 metric: str = "euclidean",
                 atol: float = 0,
                 rtol: float = 0,
                 breadth_first: bool = True,
                 leaf_size: int = 40,
                 metric_params: dict | None = None) -> None:
        
        self.bandwidth=bandwidth
        self.algorithm=algorithm
        self.kernel=kernel
        self.metric=metric
        self.atol=atol
        self.rtol=rtol
        self.breadth_first=breadth_first
        self.leaf_size=leaf_size
        self.metric_params=metric_params

        self.__is_fitted = False
        self._density_functions = None
        self._cdf_functions = None
        self._x_ranges = None
        self._bandwidths = {}


    def fit(self, X, y=None, sample_weight=None):
        
        self._x_ranges = pd.DataFrame()
        self._density_functions = pd.DataFrame()
        self._cdf_functions = pd.DataFrame()
        for variable in X.columns:

            # Prepare data compatible with sklearn KDE
            sample_dist = X[variable].to_numpy().reshape(-1, 1)
            x_range = np.linspace(sample_dist.min(), sample_dist.max(), len(X[variable]))

            bandwidth_ = self.__calculate_bandwidth(X[variable])
            self._bandwidths[variable] = bandwidth_

            # Fitting KDE
            # TODO: track changes in sklearn in order to be able in the future
            # to accommodate ability to inherite from KernelDensity rather
            # than re-instantiate seperately the object for each of the variables
            kde = KernelDensity(
                bandwidth=bandwidth_,
                algorithm=self.algorithm,
                kernel=self.kernel,
                metric=self.metric,
                atol=self.atol,
                rtol=self.rtol,
                breadth_first=self.breadth_first,
                leaf_size=self.leaf_size,
                metric_params=self.metric_params
            ).fit(X=sample_dist, y=y, sample_weight=sample_weight)

            # Compute the log-likelihood of each sample and apply exp to get density
            density_function = np.exp(kde.score_samples(x_range.reshape(-1,1)))
            
            cdf_function = density_function.cumsum()
            # Normalize the cdf function to have values <0,1>
            cdf_function /= cdf_function[-1]
        
            # Store functions in `pd.DataFrame` attributes
            self._x_ranges[variable] = x_range
            self._density_functions[variable] = density_function
            self._cdf_functions[variable] = cdf_function

        self.__is_fitted = True

        return self

    @property
    def density_functions(self):
        return self._density_functions

    @property
    def cdf_functions(self):
        return self._cdf_functions
    
    @property
    def bandwidths(self):
        return self._bandwidths

    def cdf(self, X: pd.DataFrame) -> pd.DataFrame:
        "Returns the cdf values for given random variables"
        self.__check_is_fitted()

        # TODO: apply parallelization
        X_cdf = deepcopy(X)
        for col in X_cdf.columns:
            X_cdf[col] = X_cdf[col].apply(
                lambda x_val: self.__get_cdf(
                    x_val,
                    self._cdf_functions[col],
                    self._x_ranges[col])
            )
        
        return X_cdf

    def ppf(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the probability point value (inverse cdf)
        """
        self.__check_is_fitted()

        # TODO: apply parallelization
        X_ppf = deepcopy(X)
        for col in X_ppf.columns:
            X_ppf[col] = X_ppf[col].apply(
                lambda cdf_val: self.__get_ppf(
                    cdf_val,
                    self._cdf_functions[col],
                    self._x_ranges[col])
            )

        return X_ppf

    def __get_cdf(self,
                  x_val: float,
                  cdf_function: pd.Series,
                  x_range: pd.Series):

        idx = np.abs(x_range-x_val).argmin()
        return cdf_function[idx]

    def __get_ppf(self,
                  cdf_val: float,
                  cdf_function: pd.Series,
                  x_range: pd.Series):

        idx = np.abs(cdf_function-cdf_val).argmin()
        return x_range[idx]

    def __calculate_bandwidth(self, dist: pd.Series):

        if isinstance(self.bandwidth, str):
            if self.bandwidth == 'silverman':
                calculated_bandwidth = bw_silverman(dist)
            elif self.bandwidth == 'scott':
                calculated_bandwidth = bw_scott(dist)
        else:
            calculated_bandwidth = self.bandwidth

        return calculated_bandwidth

    def __check_is_fitted(self):
        if not self.__is_fitted:
            raise NotFittedError("Estimator not fitted!")
