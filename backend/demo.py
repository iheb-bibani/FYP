import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def show_time_series_data(time_series_data: pd.DataFrame) -> None:
    time_series_data.plot(figsize=(10, 5))
    plt.show()


def show_statistics(returns: pd.DataFrame, num_trading_days: int = 252) -> None:
    print(f"Expected daily return:\n{returns.mean()*num_trading_days}%")
    print(f"Expected covariance:\n{returns.cov()*num_trading_days}")
    print(f"Expected correlation:\n{returns.corr()}")


def show_random_portfolios(returns: pd.DataFrame, volatilities: np.ndarray) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c=returns / volatilities, marker="o")
    plt.grid(True)
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Return")
    plt.colorbar(label="Sharpe Ratio")
    plt.show()


def print_optimal_portfolio(
    optimum: np.ndarray,
    expected_return: float,
    volatility: float,
    sharpe_ratio: float
) -> None:
    print(f"Optimal weights: {optimum}")
    print(f"Expected return: {expected_return}")
    print(f"Volatility: {volatility}")
    print(f"Sharpe Ratio: {sharpe_ratio}")


def show_optimal_portfolio(
    expected_return: float,
    volatility: float,
    portfolio_returns: np.ndarray,
    portfolio_volatilities: np.ndarray,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(
        portfolio_volatilities,
        portfolio_returns,
        c=portfolio_returns / portfolio_volatilities,
        marker="o",
    )
    plt.grid(True)
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Return")
    plt.colorbar(label="Sharpe Ratio")

    # Add red star to optimal portfolio
    plt.scatter(
        volatility,
        expected_return,
        marker="*",
        color="r",
        s=500,
        label="Optimal Portfolio",
    )
    plt.show()


def show_efficient_frontier(
    portfolio_returns: np.ndarray,
    portfolio_volatilities: np.ndarray,
    random_portfolios_returns: np.ndarray,
    random_portfolios_volatilities: np.ndarray,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(
        portfolio_volatilities,
        portfolio_returns,
        c=portfolio_returns / portfolio_volatilities,
        marker="x",
    )

    # Add black dashdot line for the efficient frontier
    sorted_idx = np.argsort(portfolio_volatilities)
    plt.plot(
        portfolio_volatilities[sorted_idx],
        portfolio_returns[sorted_idx],
        "k-.",
        linewidth=1,
        label="Efficient Frontier",
    )

    plt.scatter(
        random_portfolios_volatilities,
        random_portfolios_returns,
        c=random_portfolios_returns / random_portfolios_volatilities,
        marker="o",
    )

    plt.grid(True)
    plt.xlabel("Expected Volatility")
    plt.ylabel("Expected Return")
    plt.colorbar(label="Sharpe Ratio")
    plt.legend(loc="upper left")
    plt.show()
