import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List
import psycopg2
from postgres import connection
from scipy.stats import norm
import pandas_market_calendars as mcal
import streamlit as st

# For 1 to 3 year historic VAR:
# Use the mean_return, and portfolio_volatility from the weights of the optimal portfolio based on 1, 2, or 3 year historic data.
st.cache_data()


def main(
    run_id: int,
    start_date: str,
    portfolio_value: int,
    confidence_levels: List[float] = [0.95, 0.97, 0.99],
    days_to_simulate: int = 63,
) -> Tuple[
    List[float], List[float], List[float], List[float], List[float], List[float]
]:
    """
    1. Fetches optimal weights for a portfolio based on a given run ID.
    2. Downloads historical stock data for the tickers in the portfolio.
    3. Calculates the log returns, mean returns, and volatility for the portfolio.
    4. Calculates Value at Risk (VaR) using Monte Carlo simulation, historical data, and parametric method.
    """
    # 1.
    optimal_weights = get_optimal_weights(connection, run_id)
    # 2.
    tickers = get_ticker_data(connection, run_id)
    filtered_tickers, filtered_weights = filter_tickers(tickers, optimal_weights)
    data = download_data(filtered_tickers, start_date, connection)
    # 3.
    log_returns = calculate_returns(data)
    yearly_mean_returns, yearly_volatilities = calculate_returns_volatility(
        log_returns, filtered_weights, start_date
    )
    # 4.
    monte_carlo_var, simulated_portfolio_val = monte_carlo_method(
        portfolio_value,
        yearly_mean_returns,
        yearly_volatilities,
        confidence_levels,
        days=days_to_simulate,
    )
    historical_var_1yr = calculate_historical_var(
        portfolio_value,
        confidence_levels,
        log_returns,
        filtered_weights,
        start_date,
        days=365,
    )
    historical_var_2yr = calculate_historical_var(
        portfolio_value,
        confidence_levels,
        log_returns,
        filtered_weights,
        start_date,
        days=730,
    )
    historical_var_3yr = calculate_historical_var(
        portfolio_value,
        confidence_levels,
        log_returns,
        filtered_weights,
        start_date,
        days=1095,
    )
    parametric_var = calculate_parametric_var(
        portfolio_value,
        confidence_levels,
        yearly_mean_returns,
        yearly_volatilities,
        days=days_to_simulate,
    )

    return (
        monte_carlo_var,
        historical_var_1yr,
        historical_var_2yr,
        historical_var_3yr,
        parametric_var,
        simulated_portfolio_val,
    )


def get_optimal_weights(
    _connection: psycopg2.extensions.connection, run_id: int
) -> np.ndarray:
    cursor = _connection.cursor()
    query = "SELECT weight FROM optimal_weights WHERE run_id=%s"
    cursor.execute(query, [run_id])
    row = cursor.fetchone()
    cursor.close()
    if row:
        return np.array(list(map(float, row[0].split(","))))
    return np.array([])


def get_ticker_data(
    _connection: psycopg2.extensions.connection, run_id: int
) -> List[str]:
    cursor = _connection.cursor()
    query = "SELECT tickers FROM ticker_run WHERE id=%s"
    cursor.execute(query, [run_id])
    row = cursor.fetchone()
    cursor.close()
    return row[0].split(",") if row else []


def filter_tickers(
    tickers: List[str], weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
    df = df[df["Weight"] > 0]
    return df["Ticker"].to_numpy(), df["Weight"].to_numpy()


def download_data(
    tickers: np.ndarray, start_date: str, connection: psycopg2.extensions.connection
) -> pd.DataFrame:
    three_years_ago = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(
        days=3 * 365
    )
    start_date, end_date = (
        three_years_ago.strftime("%Y-%m-%d"),
        start_date,
    )  # Download from 3 years ago to the start date
    cursor = connection.cursor()
    stock_data = {}
    for stock in tickers:
        table_name = f"stock_{stock[:3]}"  # Remove the .SI suffix
        query = f"SELECT Date, Adj_Close FROM {table_name} WHERE Date >= %s AND Date <= %s"
        cursor.execute(query, (start_date, end_date))
        data = cursor.fetchall()
        if data:
            dates, closes = zip(*data)
            stock_data[stock] = pd.Series(closes, index=pd.to_datetime(dates))

    cursor.close()

    return pd.DataFrame(stock_data)


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    return np.log(data / data.shift(1)).dropna()


# Calculate portfolio statistics for multiple periods
def calculate_returns_volatility(
    log_returns: pd.DataFrame, weights: np.ndarray, start_date: str, days: int = 365
) -> Tuple[float, float]:
    new_start_date = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=days)
    trading_days = trading_days_between_dates(new_start_date, start_date)
    filtered_returns = log_returns.tail(trading_days)
    mean_return = np.dot(weights, filtered_returns.mean())
    volatility = np.sqrt(np.dot(weights.T, np.dot(filtered_returns.cov(), weights)))

    return mean_return, volatility


# Calculate Parametric VaR for multiple periods and confidence levels
def calculate_parametric_var(
    portfolio_value: int,
    confidence_levels: List[float],
    mean_returns: float,
    volatilities: float,
    days: int = 63,
) -> List[List[float]]:
    var_results = []
    for conf in confidence_levels:
        var = portfolio_value * (
            mean_returns * days - volatilities * np.sqrt(days) * norm.ppf(1 - conf)
        )
        var_results.append(var)
    return var_results


# Calculate VaR for multiple periods and confidence levels
def calculate_historical_var(
    portfolio_value: int,
    confidence_levels: List[float],
    log_returns: pd.DataFrame,
    weights: np.ndarray,
    start_date: str,
    days: int = 365,
    target_days: int = 63,
) -> List[List[float]]:
    new_start_date = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=days)
    trading_days = trading_days_between_dates(new_start_date, start_date)
    var_results = []
    portfolio_log_returns = log_returns.tail(trading_days).dot(weights)
    rolling_window_returns = portfolio_log_returns.rolling(window=target_days).apply(
        lambda x: np.prod(1 + x) - 1
    )

    for conf in confidence_levels:
        var_value = portfolio_value * np.percentile(
            rolling_window_returns.dropna(), (1 - conf) * 100
        )
        var_results.append(-var_value)

    return var_results


# Calculate VaR using Monte Carlo simulation
def monte_carlo_method(
    portfolio_value: int,
    mean_return: float,
    volatility: float,
    confidence_levels: List[float],
    days: int = 63,
    iterations: int = 10000,
) -> List[float]:
    random_numbers = np.random.normal(0, 1, [1, iterations])
    simulated_portfolio_value = portfolio_value * np.exp(
        days * (mean_return - 0.5 * volatility**2)
        + volatility * np.sqrt(days) * random_numbers
    )
    simulated_portfolio_value = np.sort(simulated_portfolio_value)

    var_results = []
    for conf in confidence_levels:
        percentile = np.percentile(simulated_portfolio_value, (1 - conf) * 100)
        var_value = portfolio_value - percentile
        var_results.append(var_value)

    return var_results, simulated_portfolio_value


def trading_days_between_dates(
    start_date: datetime, end_date: datetime, exchange: str = "XSES"
) -> int:
    cal = mcal.get_calendar(exchange)
    trading_days = cal.schedule(start_date=start_date, end_date=end_date)
    return len(trading_days)


if __name__ == "__main__":
    # Testing
    main(4, "2022-01-01", 1000000)
