from copulae.copula import BaseCopula
import pandas as pd
import warnings

from components.evaluate.cv.monthly_splitters import BaseMonthlySplitter
from components.functions.portfolio import portfolio_return, portfolio_volatility
from components.wrappers.cvxpy_wrapper import CVXPYWrapper
from components.wrappers.kde_wrapper import KernelDensityWrapper


warnings.filterwarnings("ignore")


def evaluate(
        X: pd.DataFrame,
        X_benchmark: pd.Series,
        kernel: KernelDensityWrapper,
        copula: BaseCopula,
        cv: BaseMonthlySplitter,
        random_seed: int,
        n_copula_simulations: int) -> pd.DataFrame:
    
    # train test split
    results_df = pd.DataFrame()
    for _, split in cv.split_generator(X):
        X_train = X.loc[split['train_start']:split['train_end']]
        X_test = X.loc[split['test_start']:split['test_end']]

        X_benchmark_test = X_benchmark.loc[split['test_start']:split['test_end']]

        kdw = kernel
        kdw.fit(X_train)
        X_train_cdf_vals = kdw.cdf(X_train)

        # copula fit
        cpl = copula
        cpl.fit(X_train_cdf_vals, to_pobs=False, verbose=0)
        # Simulate cdf values from the copula
        copula_simulations = cpl.random(n=n_copula_simulations, seed=random_seed)
        # Map the copula cdfs back to returns via the nonparametric distributions
        copula_simulation_values = kdw.ppf(copula_simulations)

        # model optimization
        model_markowitz = CVXPYWrapper()
        model_markowitz.fit(X_train)
        markowitz_weights = model_markowitz.weights

        model_semiparametric = CVXPYWrapper()
        model_semiparametric.fit(copula_simulation_values)
        semiparametric_weights = model_semiparametric.weights


        markowitz_return = portfolio_return(X_test, list(markowitz_weights.values()))
        semiparametric_return = portfolio_return(X_test, list(semiparametric_weights.values()))
        markowitz_volatility = portfolio_volatility(X_test, list(markowitz_weights.values()))
        semiparametric_volatility = portfolio_volatility(X_test, list(semiparametric_weights.values()))

        benchmark_return = X_benchmark_test.mean()
        benchmark_volatility = X_benchmark_test.std()

        markowitz_sharpe = (markowitz_return-benchmark_return)/markowitz_volatility
        semiparametric_sharpe = (semiparametric_return-benchmark_return)/semiparametric_volatility

        row = pd.DataFrame({
           'Train start': [split['train_start']],
            'Train end': [split['train_end']],
            'Test start': [split['test_start']],
            'Test end': [split['test_end']],
            'Markowitz weights': [list(markowitz_weights.values())],
            'Semiparametric weights': [list(semiparametric_weights.values())],
            'Markowitz return': [markowitz_return],
            'Semiparametric return': [semiparametric_return],
            'Benchmark return': [benchmark_return],
            'Markowitz volatility': [markowitz_volatility],
            'Semiparametric volatility': [semiparametric_volatility],
            'Benchmark volatility': [benchmark_volatility],
            'Markowitz sharpe': [markowitz_sharpe],
            'Semiparametric sharpe': [semiparametric_sharpe]
        })
        results_df = pd.concat([results_df, row], ignore_index=True)
    
    return results_df
