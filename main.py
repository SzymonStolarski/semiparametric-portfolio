from copy import deepcopy
from datetime import datetime

from copulae.archimedean import ClaytonCopula, FrankCopula, GumbelCopula
from copulae.elliptical import GaussianCopula, StudentCopula
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from components.transformers.rate_transformer import RateTransformer
from components.transformers.log_transformer import LognTransformer
from components.wrappers.kde_wrapper import KernelDensityWrapper
from components.evaluate.cv.monthly_splitters import ExpandingMonthlySplitter, SlidingMonthlySplitter
from components.evaluate.evaluate import evaluate


RANDOM_SEED = 5

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

# 'gaussian', 'student', 'frank', 'clayton', 'gumbel'
COPULA = 'student'

copula_dict = {
    'gaussian': GaussianCopula,
    'student': StudentCopula,
    'frank': FrankCopula,
    'clayton': ClaytonCopula,
    'gumbel': GumbelCopula
}

# Cross validation configuration
# expanding window
CV = ExpandingMonthlySplitter(
    initial_window=150, step_length=1, test_horizon=1)
# sliding window
# CV = SlidingMonthlySplitter(
#     initial_window=130, step_length=1, test_horizon=1, window_length=24)


assets_in_portfolios = list(
    set([i for j in SELECTED_PORTFOLIOS.values() for i in j]))
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

# Evaluation part
results_df = pd.DataFrame()
for portfolio, assets in SELECTED_PORTFOLIOS.items():

    X = df_transformed[assets]
    X_benchmark = df_transformed[SELECTED_BENCHMARK]

    portfolio_results_df = evaluate(
        X=X,
        X_benchmark=X_benchmark,
        kernel=KernelDensityWrapper(kernel=KERNEL, bandwidth=KERNEL_BANDWIDTH),
        copula=copula_dict[COPULA](dim=len(assets)),
        cv=CV,
        random_seed=RANDOM_SEED,
        n_copula_simulations=X.shape[0]
    )
    portfolio_results_df['Portfolio'] = portfolio
    portfolio_results_df['Assets'] = [
        assets for _ in range(0, len(portfolio_results_df))]
    
    results_df = pd.concat(
        [results_df, portfolio_results_df], ignore_index=True)

col_order = ['Portfolio', 'Assets']
col_order.extend(portfolio_results_df.columns[:-2])

results_df = results_df[col_order]

# Prep data for saving and save into excel file
results_condensed = deepcopy(results_df)
results_condensed['Markowitz weights'] = [
    np.around(i, 5) for i in results_condensed['Markowitz weights'].values]
results_condensed['Semiparametric weights'] = [
    np.around(i, 5) for i in results_condensed['Semiparametric weights'].values]

results_condensed.to_excel(
    f"results_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.xlsx", index=False)
