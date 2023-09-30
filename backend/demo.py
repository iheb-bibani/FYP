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


def show_random_portfolios(returns: pd.DataFrame, volatilities: np.ndarray, num_days: int = 252) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities * np.sqrt(num_days), returns * num_days, c=returns / volatilities, marker="o")
    plt.grid(True)
    plt.xlabel("Expected Downside Risk")
    plt.ylabel("Expected Return")
    plt.colorbar(label="Sortino Ratio")
    plt.show()


def print_optimal_portfolio(
    optimum: np.ndarray, expected_return: float, downside: float, sortino_ratio: float, num_days: int = 252
) -> None:
    print(f"Optimal weights: {optimum}")
    print(f"Expected annual return: {expected_return * num_days}")
    print(f"Annualized downside risk: {downside * np.sqrt(num_days)}")
    print(f"Sortino Ratio: {sortino_ratio}")
    print(f"Avg daily return: {expected_return}")


def show_optimal_portfolio(
    expected_return: float,
    downside: float,
    portfolio_returns: np.ndarray,
    portfolio_volatilities: np.ndarray,
    num_days: int = 252,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(
        portfolio_volatilities * np.sqrt(num_days),
        portfolio_returns * num_days,
        c=portfolio_returns / portfolio_volatilities,
        marker="o",
    )
    plt.grid(True)
    plt.xlabel("Expected Downside Risk")
    plt.ylabel("Expected Return")
    plt.colorbar(label="Sortino Ratio")

    # Add red star to optimal portfolio
    plt.scatter(
        downside * np.sqrt(num_days),
        expected_return * num_days,
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
    num_days: int = 252,
) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(
        portfolio_volatilities * np.sqrt(num_days),
        portfolio_returns * num_days,
        c=portfolio_returns / portfolio_volatilities,
        marker="x",
    )

    # Add black dashdot line for the efficient frontier
    sorted_idx = np.argsort(portfolio_volatilities)
    plt.plot(
        portfolio_volatilities[sorted_idx] * np.sqrt(num_days),
        portfolio_returns[sorted_idx] * num_days,
        "k-.",
        linewidth=1,
        label="Efficient Frontier",
    )

    plt.scatter(
        random_portfolios_volatilities * np.sqrt(num_days),
        random_portfolios_returns * num_days,
        c=random_portfolios_returns / random_portfolios_volatilities,
        marker="o",
    )

    plt.grid(True)
    plt.xlabel("Expected Downside Risk")
    plt.ylabel("Expected Return")
    plt.colorbar(label="Sortino Ratio")
    plt.legend(loc="upper left")
    plt.show()
