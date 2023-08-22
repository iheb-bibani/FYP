import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.optimize as optimization
import sqlite3
from typing import Tuple, List

start_date = '2022-01-01'
end_date = '2022-12-31'

def download_data() -> pd.DataFrame:
    connection = sqlite3.connect('database.db')
    cursor = connection.cursor()
    tickers = cursor.execute("SELECT ticker FROM equities").fetchall()
    stock_data = {}

    for stock_tuple in tickers:
        stock = stock_tuple[0]
        table_name = f"stock_{stock[:3]}" # Remove the .SI suffix
        query = f'SELECT Date, Close FROM {table_name} WHERE Date >= "{start_date}" AND Date <= "{end_date}"'
        data = cursor.execute(query).fetchall()
        dates, closes = zip(*data)
        stock_data[stock] = pd.Series(closes, index=pd.to_datetime(dates))

    cursor.close()
    connection.close()

    return pd.DataFrame(stock_data)

def show_data(data: pd.DataFrame) -> None:
    data.plot(figsize=(10,5))
    plt.show()

def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    log_return = np.log(data/data.shift(1))[1:] # Daily log returns
    return log_return

def show_statistics(returns: pd.DataFrame, num_trading_days: int = 252) -> None:
    print(f"Expected daily return:\n{returns.mean()*num_trading_days}%")
    print(f"Expected covariance:\n{returns.cov()*num_trading_days}")
    print(f"Expected correlation:\n{returns.corr()}")

def generate_random_weights(num_stocks: int) -> np.ndarray:
    # Randomly select the number of stocks to include between 4 and 20
    num_selected_stocks = np.random.randint(4, 21)

    # Randomly select the indices of the stocks to include
    selected_stocks = np.random.choice(num_stocks, num_selected_stocks, replace=False)

    # Create a weight array of zeros
    weights = np.zeros(num_stocks)

    # Randomly assign weights above 5% to the selected stocks
    random_weights = np.random.uniform(0.05, 1, num_selected_stocks)
    random_weights /= random_weights.sum()

    # Update the weight array with the random weights
    for idx, weight in zip(selected_stocks, random_weights):
        weights[idx] = weight

    return weights

def generate_portfolios(returns: pd.DataFrame, num_portfolios: int = 10000, num_trading_days: int = 252, stocks: List[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []
    sharpe_ratios = []
    num_stocks = len(stocks)

    for _ in range(num_portfolios):
        w = generate_random_weights(num_stocks)
        portfolio_weights.append(w)
        mean_return = np.sum(returns.mean() * w) * num_trading_days
        risk = np.sqrt(np.dot(w.T, np.dot(returns.cov() * num_trading_days, w)))
        portfolio_means.append(mean_return)
        portfolio_risks.append(risk)
        sharpe_ratios.append(mean_return / risk)

    # Sorting the portfolios by Sharpe ratio
    sorted_indices = np.argsort(sharpe_ratios)[::-1] # Reverse to sort in descending order
    portfolio_weights = np.array(portfolio_weights)[sorted_indices]
    portfolio_means = np.array(portfolio_means)[sorted_indices]
    portfolio_risks = np.array(portfolio_risks)[sorted_indices]

    return portfolio_weights, portfolio_means, portfolio_risks


def show_portfolios(returns: pd.DataFrame, volatilities: np.ndarray) -> None:
    plt.figure(figsize=(10,6))
    plt.scatter(volatilities, returns, c=returns/volatilities, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')
    plt.show()

# Sharpe Ratio = (Expected Return - Risk Free Rate) / Expected Volatility
def statistics(weights: np.ndarray, returns: pd.DataFrame, num_trading_days: int = 252) -> np.ndarray:
    portfolio_return = np.sum(returns.mean()*weights)*num_trading_days
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*num_trading_days, weights)))
    return np.array([portfolio_return, portfolio_volatility, portfolio_return/portfolio_volatility])

# Minimize the negative Sharpe Ratio: Maximize the Sharpe Ratio
def min_func_sharpe(weights: np.ndarray, returns: pd.DataFrame, num_trading_days: int = 252) -> float:
    return -statistics(weights, returns, num_trading_days)[2]

# Finds the optimal portfolio weights that maximizes the Sharpe Ratio
def optimize_portfolio(weights: np.ndarray, returns: pd.DataFrame, stocks: List[str] = None) -> np.ndarray:
    len_stocks = len(stocks)
    selected_stocks_count = 0
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len_stocks))
    optimum_weights = weights[0]
    
    while selected_stocks_count < 4 or selected_stocks_count > 20 :
        print("Optimizing portfolio...")
        optimum = optimization.minimize(fun=min_func_sharpe, x0=optimum_weights, args=returns, method='SLSQP', bounds=bounds, constraints=constraints)
        optimum_weights = optimum['x']
        optimum_weights[optimum_weights < 0.05] = 0
        selected_stocks_count = np.sum(optimum_weights > 0)
        
        total_weights = np.sum(optimum_weights)
        optimum_weights = np.round(optimum_weights / total_weights, 3)
    
    total_weights = np.sum(optimum_weights)
    print(f"Total weights: {total_weights}")
    return optimum_weights

def print_optimal_portfolio(optimum: np.ndarray, returns: pd.DataFrame) -> None:
    print(f"Optimal weights: {optimum}")
    print(f"Expected return, volatility and Sharpe Ratio: {statistics(optimum, returns)}")

def show_optimal_portfolio(optimum: np.ndarray, returns: pd.DataFrame, portfolio_returns: np.ndarray, portfolio_volatilities: np.ndarray) -> None:
    plt.figure(figsize=(10,6))
    plt.scatter(portfolio_volatilities, portfolio_returns, c=portfolio_returns/portfolio_volatilities, marker='o')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label='Sharpe Ratio')

    # Add red star to optimal portfolio
    plt.scatter(statistics(optimum, returns)[1], statistics(optimum, returns)[0], marker='*', color='r', s=500, label='Optimal Portfolio')
    plt.show()

if __name__ == "__main__":
    data = download_data()
    show_data(data)
    log_return = calculate_returns(data)
    show_statistics(log_return)

    weights, means, risks = generate_portfolios(log_return, stocks=data.columns)
    show_portfolios(means, risks)
    optimum = optimize_portfolio(weights, log_return, data.columns)
    print_optimal_portfolio(optimum, log_return)
    show_optimal_portfolio(optimum, log_return, means, risks)



# Expected value = Summation of all possible values * probability of each value
# Variance = Summation of all possible values * probability of each value * (value - expected value)^2
# Covariance = Summation of all possible values * probability of each value * (value - expected value of x) * (value - expected value of y)
# Correlation = Covariance / (standard deviation of x * standard deviation of y)
# Stocks which are highly correlated are not good for diversification.

"""
Assumptions of Markowitz Model:
1. Returns are normally distributed with a mean and variance
2. Investors are risk averse
3. Long positions only (no short selling)
Goal:
1. Maximize return for a given level of risk
2. Minimize risk for a given level of return
Parameters:
1. Weights of each asset
2. Return of each asset based on historical data
3. Expected return of each asset
"""

# Sharpe Ratio = (Expected return of portfolio - risk free rate) / standard deviation of portfolio. Higher the better.