from copy import deepcopy

from copulae.archimedean import ClaytonCopula, FrankCopula, GumbelCopula
from copulae.elliptical import GaussianCopula, StudentCopula
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import simps
from scipy.stats import bootstrap, norm
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from statsmodels.nonparametric.bandwidths import bw_silverman, bw_scott

from components.functions.portfolio import portfolio_return, portfolio_volatility
from components.transformers.rate_transformer import RateTransformer
from components.transformers.log_transformer import LognTransformer
from components.wrappers.cvxpy_wrapper import CVXPYWrapper
from components.wrappers.kde_wrapper import KernelDensityWrapper


X_TRAIN_START = '2010-01-01'
X_TRAIN_END = '2022-12-30'
X_TEST_START = '2023-01-02'
X_TEST_END = '2023-02-28'


SELECTED_PORTFOLIOS = {
    'P1': ['Nickel', 'Copper'],
    'P2': ['Brent Oil', 'Gas US'],
    'P3': ['Gold', 'Silver'],
    'P4': ['Nickel', 'Copper', 'Gold'],
    'P5': ['Nickel', 'Copper', 'Silver'],
    'P6': ['Brent Oil', 'Gas US', 'Gold'],
    'P7': ['Brent Oil', 'Gas US', 'Silver']
}

SELECTED_BENCHMARK = 'SPGSCI'

# KDE configuration
# 'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine',
KERNEL = 'epanechnikov'
# 'silverman', 'scott', float
KERNEL_BANDWIDTH = 'silverman'

# Copula
copula_dict = {
    'gaussian': GaussianCopula,
    'student': StudentCopula,
    'frank': FrankCopula,
    'clayton': ClaytonCopula,
    'gumbel': GumbelCopula
}
# 'gaussian', 'student', 'frank', 'clayton', 'gumbel'
COPULA = 'student'
# N_COPULA_SIMULATIONS = 10000
RANDOM_SEED = 5

# Portfolio simulations
WEIGHT_SIMULATIONS = 10000

assets_in_portfolios = list(set([i for j in SELECTED_PORTFOLIOS.values() for i in j]))
assets_in_portfolios.append(SELECTED_BENCHMARK)

# Load data
# load prices from dataset
df = pd.read_excel('data/BazaMSA.xlsx', sheet_name='Ceny')
# Dates as index
df.set_index('Dates', inplace=True)

# Select only necessary columns
df = df[assets_in_portfolios]

# Comprehensive scikitlearn `Pipeline` that first creates the rates
# and then performs logn on top
log_rate_pipe = Pipeline(
    steps=[
        ('rates', RateTransformer(period=1)),
        ('logn', LognTransformer())
    ]
)
df_transformed = log_rate_pipe.fit_transform(X=df)
# Drop all the `na` values
df_transformed.dropna(inplace=True)


# train/test split - currently only one split
# Select dataset to train (fit) and test
X_train = df_transformed.loc[X_TRAIN_START:X_TRAIN_END]
X_test = df_transformed.loc[X_TEST_START:X_TEST_END]

# density estimation
kdw = KernelDensityWrapper(bandwidth='silverman')
kdw.fit(X_train)
X_train_cdf_vals = kdw.cdf(X_train)

results_df = pd.DataFrame()
for portfolio, assets in SELECTED_PORTFOLIOS.items():
    # copula fit
    copula = copula_dict[COPULA](dim=len(assets))
    copula.fit(X_train_cdf_vals[assets], to_pobs=False, verbose=False)
    # Simulate cdf values from the copula
    copula_simulations = copula.random(n=len(X_train_cdf_vals), seed=RANDOM_SEED)
    # Map the copula cdfs back to returns via the nonparametric distributions
    copula_simulation_values = kdw.ppf(copula_simulations)

    # model optimization
    model_markowitz = CVXPYWrapper()
    model_markowitz.fit(X_train[assets])
    markowitz_weights = model_markowitz.weights

    model_semiparametric = CVXPYWrapper()
    model_semiparametric.fit(copula_simulation_values)
    semiparametric_weights = model_semiparametric.weights

    # testing
    testing_portfolio = X_test[assets]
    testing_benchmark = X_test[SELECTED_BENCHMARK]

    markowitz_return = portfolio_return(testing_portfolio, list(markowitz_weights.values()))
    semiparametric_return = portfolio_return(testing_portfolio, list(semiparametric_weights.values()))
    markowitz_volatility = portfolio_volatility(testing_portfolio, list(markowitz_weights.values()))
    semiparametric_volatility = portfolio_volatility(testing_portfolio, list(semiparametric_weights.values()))

    benchmark_return = testing_benchmark.mean()
    benchmark_volatility = testing_benchmark.std()

    markowitz_sharpe = (markowitz_return-benchmark_return)/markowitz_volatility
    semiparametric_sharpe = (semiparametric_return-benchmark_return)/semiparametric_volatility

    row = pd.DataFrame({
        'Portfolio': [portfolio],
        'Assets': [assets],
        'Train period': [[X_TRAIN_START, X_TRAIN_END]],
        'Test period': [[X_TEST_START, X_TEST_END]],
        'Markowitz weights': [list(markowitz_weights.values())],
        'Semiparametric weights': [list(semiparametric_weights.values())],
        'Markowitz return': [markowitz_return],
        'Semiparametric return': [semiparametric_return],
        f'Benchmark {SELECTED_BENCHMARK} return': [benchmark_return],
        'Markowitz volatility': [markowitz_volatility],
        'Semiparametric volatility': [semiparametric_volatility],
        f'Benchmark {SELECTED_BENCHMARK} volatility': [benchmark_volatility],
        'Markowitz sharpe': [markowitz_sharpe],
        'Semiparametric sharpe': [semiparametric_sharpe]
    })
    results_df = pd.concat([results_df, row], ignore_index=True)

print(results_df.head())