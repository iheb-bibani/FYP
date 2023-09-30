import numpy as np
import pandas as pd
import scipy.optimize as optimization
import argparse
from datetime import datetime
from typing import Tuple, List
from scipy.stats import jarque_bera
from demo import (
    show_random_portfolios,
    show_optimal_portfolio,
    print_optimal_portfolio,
    show_efficient_frontier,
)
import psycopg2
from postgres import connection
import cupy as cp

DEBUG = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio analysis.")
    parser.add_argument(
        "--start_date",
        type=str,
        default="2022-01-03",
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2022-12-30",
        help="End date in YYYY-MM-DD format",
    )

    args = parser.parse_args()
    start_date = args.start_date
    end_date = args.end_date

    # Validate start_date and end_date
    if not is_valid_date(start_date) or not is_valid_date(end_date):
        print(
            "Invalid date format. Dates should be in YYYY-MM-DD format and should be valid."
        )
        return

    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

    if start_date_dt >= end_date_dt:
        print("Start date must be before end date.")
        return

    data = download_data(start_date, end_date, connection)
    log_return = calculate_returns(data)
    
    normally_distributed = 0
    for col in log_return.columns:
        result = check_normality(log_return[col])
        if result:
            normally_distributed += 1
            print(f"{col} is normally distributed.")
        
    print(f"{normally_distributed} out of {len(log_return.columns)} stocks are normally distributed.")
    
    tickers = data.columns.tolist()

    weights, means, risks = generate_portfolios(log_return, stocks=tickers)

    if DEBUG:
        show_random_portfolios(means, risks)
    else:
        run_id = store_ticker_run(start_date, end_date, tickers, connection)
        store_portfolio_weights(weights, means, risks, run_id, connection)

    modified_returns, top_n_indices, original_len = sort_by_sharpe_ratio(log_return, top_n=100)
    
    optimum = optimize_portfolio(modified_returns, top_n_indices, original_len, lambda_val=0.1)
    
    expected_return, volatility, sharpe_ratio = statistics(optimum, log_return)
    
    if DEBUG:
        print_optimal_portfolio(optimum, expected_return, volatility, sharpe_ratio)
        show_optimal_portfolio(expected_return, volatility, means, risks)
    else:
        store_optimal_weights(optimum, expected_return, volatility, run_id, connection)

    efficient_list = [[optimum, expected_return, volatility, sharpe_ratio]]
    target_returns = np.linspace(expected_return - 0.001, expected_return + 0.001, 20)

    for target in target_returns:
        print(f"Target return: {target}")
        final_optimum_weights = efficient_portfolios(modified_returns, top_n_indices, original_len, target)
        if final_optimum_weights is None:
            print("No efficient portfolio found.")
            continue
        else:
            expected_return, volatility, sharpe_ratio = statistics(final_optimum_weights, log_return)
            values = (final_optimum_weights, expected_return, volatility, sharpe_ratio)
        efficient_list.append(values)

    if DEBUG:
        show_efficient_frontier(
            np.array([x[1] for x in efficient_list]),
            np.array([x[2] for x in efficient_list]),
            means,
            risks,
        )
    else:
        efficient_list.sort(key=lambda x: x[1])  # Sort by returns
        store_efficient_frontier(
            [x[0] for x in efficient_list],
            [x[1] for x in efficient_list],
            [x[2] for x in efficient_list],
            run_id,
            connection,
        )
        pass
    connection.close()


def is_valid_date(date_str: str, format: str = "%Y-%m-%d") -> bool:
    try:
        datetime.strptime(date_str, format)
        return True
    except ValueError:
        return False


def download_data(
    start_date: str, end_date: str, connection: psycopg2.extensions.connection
) -> pd.DataFrame:
    cursor = connection.cursor()
    cursor.execute("SELECT ticker FROM equities")
    tickers = cursor.fetchall()
    stock_data = {}

    for stock_tuple in tickers:
        stock = stock_tuple[0]
        if stock == "^STI":
            continue
        table_name = f"stock_{stock[:3]}"  # Remove the .SI suffix
        query = (
            f"SELECT Date, Adj_Close FROM {table_name} WHERE Date >= %s AND Date <= %s"
        )
        cursor.execute(query, (start_date, end_date))
        data = cursor.fetchall()
        if data:
            dates, closes = zip(*data)
            if datetime.strptime(start_date, "%Y-%m-%d") != datetime.strptime(
                dates[0], "%Y-%m-%d"
            ) or datetime.strptime(end_date, "%Y-%m-%d") != datetime.strptime(
                dates[-1], "%Y-%m-%d"
            ):
                print(
                    f"Insufficient data for {stock}. Data not retrieved. {dates[0]} to {dates[-1]}"
                )
                continue
            stock_data[stock] = pd.Series(closes, index=pd.to_datetime(dates))

    cursor.close()

    return pd.DataFrame(stock_data)


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    column_names = data.columns
    shifted_data = cp.array(data.shift(1).values)
    data_values = cp.array(data.values)
    log_return = cp.where(
        (data_values > 0) & (shifted_data > 0),
        cp.log(data_values / shifted_data),
        cp.nan,
    )[
        1:
    ]  # Remove the first row of NaNs
    
    return pd.DataFrame(cp.asnumpy(log_return), columns=column_names)

def check_normality(returns) -> bool:
    statistic, p_value = jarque_bera(returns)
    if p_value < 0.05:
        # print("Warning: Data does not appear to be normally distributed.")
        return False
    return True

def generate_random_weights(num_stocks: int, existing_portfolios: set) -> cp.ndarray:
    while True:
        num_selected_stocks = cp.random.randint(5, 21)
        selected_stocks = cp.random.choice(num_stocks, int(num_selected_stocks), replace=False)
        weights = cp.zeros(num_stocks, dtype=cp.float32)

        # Randomly assign weights above 5% to the selected stocks
        random_weights = cp.random.uniform(0.05, 1, int(num_selected_stocks))
        random_weights /= random_weights.sum()

        # Update the weights array with the random weights
        for idx, weight in zip(selected_stocks, random_weights):
            weights[idx] = weight

        # Check uniqueness
        weight_tuple = tuple(weights.tolist())
        if weight_tuple not in existing_portfolios:
            existing_portfolios.add(weight_tuple)
            return weights


def generate_portfolios(
    returns: pd.DataFrame,
    num_portfolios: int = 50000,
    stocks: List[str] = None,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    num_stocks = len(stocks)
    existing_portfolios = set()
    
    portfolio_weights = cp.array(
        [generate_random_weights(num_stocks, existing_portfolios) for _ in range(num_portfolios)]
    )
    
    mean_return = cp.sum(cp.array(returns.mean().values) * portfolio_weights, axis=1)
    
    # Calculate downside risk
    portfolio_return_series = cp.dot(cp.array(returns.values), portfolio_weights.T)
    target = cp.zeros(portfolio_return_series.shape[1])
    downside_diff = portfolio_return_series - target[cp.newaxis, :]
    downside_diff = cp.where(downside_diff < 0, downside_diff, 0)
    portfolio_downside_risk = cp.sqrt(cp.mean(downside_diff ** 2, axis=0))

    # Calculate Sortino Ratios
    sortino_ratios = mean_return / portfolio_downside_risk
    sorted_indices = cp.argsort(sortino_ratios)[::-1] 
    
    portfolio_weights = portfolio_weights[sorted_indices]
    portfolio_means = mean_return[sorted_indices]
    portfolio_downside_risks = portfolio_downside_risk[sorted_indices]

    return portfolio_weights.get(), portfolio_means.get(), portfolio_downside_risks.get()


# Sortino Ratio = (Expected Return - Risk Free Rate) / Downside Risk
def statistics(weights: np.ndarray, returns: np.ndarray, lambda_val: float = 0.1) -> np.ndarray:
    portfolio_return = np.sum(np.mean(returns, axis=0) * weights)
    downside_diff = np.dot(returns, weights) - 0
    downside_diff = downside_diff[downside_diff < 0]
    downside_risk = np.sqrt(np.mean(np.square(downside_diff)))
    raw_sortino_ratio = portfolio_return / downside_risk
    l1_norm = np.sum(np.abs(weights))
    regularized_sortino_ratio = raw_sortino_ratio - lambda_val * l1_norm
    return np.array([portfolio_return, downside_risk, regularized_sortino_ratio])


# Minimize the negative Sortino Ratio: Maximize the Sortino Ratio
def min_func_sortino(weights: np.ndarray, returns: pd.DataFrame, lambda_val: float) -> float:
    sortino_ratio = statistics(weights, returns, lambda_val)[2]
    adjusted_weights = np.copy(weights)
    adjusted_weights[adjusted_weights < 0.05] = 0
    non_zero_weights = np.sum(adjusted_weights > 0)
    penalty = 1000 if non_zero_weights < 5 or non_zero_weights > 20 else 0
    return -sortino_ratio + penalty



# Finds the optimal portfolio weights that maximize the Sortino Ratio
def optimize_portfolio(modified_returns: np.ndarray, top_n_indices: int, original_len: int, lambda_val: float = 0.1) -> np.ndarray:
    len_stocks = len(top_n_indices)
    bounds = [(0, 1) for _ in range(len_stocks)]
    init_guess = np.ones(len_stocks) / len_stocks
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    optimum = optimization.minimize(
        fun=min_func_sortino,
        x0=init_guess,
        args=(modified_returns, lambda_val),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    optimum_weights = optimum["x"]
    optimum_weights[optimum_weights < 0.05] = 0
    optimum_weights = optimum_weights / np.sum(optimum_weights)
    non_zero_weights = np.sum(optimum_weights > 0)
    if 5 <= non_zero_weights <= 20:
        final_optimum_weights = np.zeros(original_len)
        for idx, original_idx in enumerate(top_n_indices):
            final_optimum_weights[original_idx] = optimum_weights[idx]
        return final_optimum_weights
    else:
        return 0

def min_func_variance(weights: np.ndarray, returns: pd.DataFrame) -> float:
    variance = statistics(weights, returns)[1]

    # Calculate penalty based on the original weights
    adjusted_weights = np.copy(weights)
    adjusted_weights[adjusted_weights < 0.05] = 0
    non_zero_weights = np.sum(adjusted_weights > 0)
    if non_zero_weights == 0:
        penalty = 1000
    else:
        adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
        penalty = 1000 if non_zero_weights < 5 or non_zero_weights > 20 else 0

    return variance + penalty


def portfolio_return(weights: np.ndarray, returns: pd.DataFrame) -> float:
    return statistics(weights, returns)[0]


def efficient_portfolios(
    returns: np.ndarray, top_n_indices: int , original_len: int, target_return: float
) -> np.ndarray:
    # Optimization
    len_stocks = len(top_n_indices)
    bounds = [(0, 1) for _ in range(len_stocks)]

    # Initialize the weights to be equal
    init_guess = np.ones(len_stocks) / len_stocks

    constraints = [
        {
            "type": "eq",
            "fun": lambda x: portfolio_return(x, returns) - target_return,
        },
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
    ]

    optimum = optimization.minimize(
        fun=min_func_variance,
        x0=init_guess,
        args=returns,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    optimum_weights = optimum["x"]
    optimum_weights[optimum_weights < 0.05] = 0
    optimum_weights = optimum_weights / np.sum(optimum_weights)

    non_zero_weights = np.sum(optimum_weights > 0)

    if 5 <= non_zero_weights <= 20:
        final_optimum_weights = np.zeros(original_len)

        for idx, original_idx in enumerate(top_n_indices):
            final_optimum_weights[original_idx] = optimum_weights[idx]

        return final_optimum_weights
    else:
        return None

def sort_by_sharpe_ratio(returns: pd.DataFrame, top_n: int = 100) -> Tuple[np.ndarray, List[int], int]:
    # Step 0: Pre-filter stocks based on Sharpe ratio
    mean_returns = cp.array(returns.mean())
    std_dev = cp.array(returns.std())
    sharpe_ratios = mean_returns / std_dev
    sorted_indices = cp.argsort(-sharpe_ratios)  # Sort in descending order
    top_n_indices = sorted_indices[:top_n].tolist()

    # Reduce the size of returns and initial guess
    reduced_returns = cp.array(returns.iloc[:, top_n_indices])

    # Convert to numpy for scipy optimization
    reduced_returns_np = cp.asnumpy(reduced_returns)
    return reduced_returns_np, top_n_indices, len(returns.columns)


def store_ticker_run(
    start_date: str,
    end_date: str,
    tickers: List[str],
    connection: psycopg2.extensions.connection,
) -> int:
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ticker_run (
            id SERIAL PRIMARY KEY,
            start_date TEXT,
            end_date TEXT,
            tickers TEXT
        )
    """
    )
    tickers_str = ",".join(tickers)
    cursor.execute(
        "INSERT INTO ticker_run (start_date, end_date, tickers) VALUES (%s, %s, %s) RETURNING id",
        (start_date, end_date, tickers_str),
    )
    run_id = cursor.fetchone()[0]
    connection.commit()
    cursor.close()
    return run_id


def store_portfolio_weights(
    portfolio_weights: np.ndarray,
    means: np.ndarray,
    risks: np.ndarray,
    run_id: int,
    connection: psycopg2.extensions.connection,
):
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio_weights (
            id SERIAL PRIMARY KEY,
            run_id INTEGER,
            weight TEXT,
            returns REAL,
            volatility REAL,
            FOREIGN KEY(run_id) REFERENCES ticker_run(id)
        )
    """
    )
    for weights, mean, risk in zip(portfolio_weights, means, risks):
        weights_str = ",".join([str(weight) for weight in weights])
        cursor.execute(
            "INSERT INTO portfolio_weights (run_id, weight, returns, volatility) VALUES (%s, %s, %s, %s)",
            (run_id, weights_str, mean, risk),
        )
    connection.commit()
    cursor.close()

def store_optimal_weights(
    optimum_weights: np.ndarray,
    means: np.float_,
    risks: np.float_,
    run_id: int,
    connection: psycopg2.extensions.connection,
) -> int:
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS optimal_weights (
            id SERIAL PRIMARY KEY,
            run_id INTEGER,
            weight TEXT,
            returns REAL,
            volatility REAL,
            FOREIGN KEY(run_id) REFERENCES ticker_run(id)
        )
    """
    )
    weights_str = ",".join([str(weight) for weight in optimum_weights])
    cursor.execute(
        "INSERT INTO optimal_weights (run_id, weight, returns, volatility) VALUES (%s, %s, %s, %s)",
        (run_id, weights_str, means, risks),
    )
    connection.commit()
    cursor.close()


def store_efficient_frontier(
    optimum_weights: np.ndarray,
    means: np.ndarray,
    risks: np.ndarray,
    run_id: int,
    connection: psycopg2.extensions.connection,
) -> int:
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS efficient_frontier (
            id SERIAL PRIMARY KEY,
            run_id INTEGER,
            weight TEXT,
            returns REAL,
            volatility REAL,
            FOREIGN KEY(run_id) REFERENCES ticker_run(id)
        )
    """
    )
    for weights, mean, risk in zip(optimum_weights, means, risks):
        weights_str = ",".join([str(weight) for weight in weights])
        cursor.execute(
            "INSERT INTO efficient_frontier (run_id, weight, returns, volatility) VALUES (%s, %s, %s, %s)",
            (run_id, weights_str, mean, risk),
        )
    connection.commit()
    cursor.close()


if __name__ == "__main__":
    main()
