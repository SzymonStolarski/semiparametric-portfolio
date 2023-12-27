from cvxpy import *
import pandas as pd


class CVXPYWrapper:

    def __init__(self, minimal_weights_asset: float = 0) -> None:

        self.__cov_matrix = None
        self.__weights = None
        self.minimal_weights_asset = minimal_weights_asset
        self.__target_value = None

    def fit(self, X: pd.DataFrame):

        self.__cov_matrix = X.cov()
        weights_variable = Variable(len(X.columns))
        target = quad_form(weights_variable, self.__cov_matrix)

        self.__target_value = self.__solve(target, weights_variable)
        self.__weights = dict(zip(X.columns, weights_variable.value))

        return self

    @property
    def cov_matrix(self):
        return self.__cov_matrix
    
    @property
    def weights(self):
        return self.__weights

    @property
    def target_value(self):
        return self.__target_value
    
    def __solve(self, target, weights):
        optimized_problem = Problem(
            Minimize(target),
            [sum(weights)==1, weights >= self.minimal_weights_asset]).solve()

        return optimized_problem
