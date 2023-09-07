import numpy as np
import pandas as pd
import scipy.optimize as optimization
import argparse
from datetime import datetime
from typing import Tuple, List
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
    tickers = data.columns.tolist()

    weights, means, risks = generate_portfolios(log_return, stocks=tickers)

    if DEBUG:
        show_random_portfolios(means, risks)
    else:
        run_id = store_ticker_run(start_date, end_date, tickers, connection)
        store_portfolio_weights(weights, means, risks, run_id, connection)

    optimum, expected_return, volatility, sharpe_ratio = optimize_portfolio(
        weights[0], log_return, tickers
    )

    if DEBUG:
        print_optimal_portfolio(optimum, expected_return, volatility, sharpe_ratio)
        show_optimal_portfolio(expected_return, volatility, means, risks)
    else:
        store_optimal_weights(optimum, expected_return, volatility, run_id, connection)

    efficient_list = [[optimum, expected_return, volatility, sharpe_ratio]]
    target_returns = np.linspace(expected_return - 0.001, expected_return + 0.001, 20)

    for target in target_returns:
        print(f"Target return: {target}")
        values = efficient_portfolios(log_return, tickers, target)
        if values == 0:
            print("No efficient portfolio found for target return. Skipping...")
            continue
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


def generate_random_weights(num_stocks: int) -> cp.ndarray:
    # Randomly select the number of stocks to include between 4 and 20
    num_selected_stocks = cp.random.randint(4, 21)

    # Randomly select the indices of the stocks to include
    selected_stocks = cp.random.choice(
        num_stocks, int(num_selected_stocks), replace=False
    )

    # Create a weight array of zeros
    weights = cp.zeros(num_stocks, dtype=cp.float32)

    # Randomly assign weights above 5% to the selected stocks
    random_weights = cp.random.uniform(0.05, 1, int(num_selected_stocks))
    random_weights /= random_weights.sum()

    # Update the weight array with the random weights
    for idx, weight in zip(selected_stocks, random_weights):
        weights[idx] = weight

    return weights


def generate_portfolios(
    returns: pd.DataFrame,
    num_portfolios: int = 10000,
    stocks: List[str] = None,
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    num_stocks = len(stocks)

    portfolio_weights = cp.array(
        [generate_random_weights(num_stocks) for _ in range(num_portfolios)]
    )
    mean_return = cp.sum(cp.array(returns.mean().values) * portfolio_weights, axis=1)
    cov_matrix = cp.array(returns.cov().values)

    portfolio_risk = cp.sqrt(
        cp.sum(portfolio_weights @ cov_matrix * portfolio_weights, axis=1)
    )
    sharpe_ratios = mean_return / portfolio_risk
    sorted_indices = cp.argsort(sharpe_ratios)[::-1]  # Sort in descending order

    portfolio_weights = portfolio_weights[sorted_indices]
    portfolio_means = mean_return[sorted_indices]
    portfolio_risks = portfolio_risk[sorted_indices]

    return portfolio_weights.get(), portfolio_means.get(), portfolio_risks.get()


# Sharpe Ratio = (Expected Return - Risk Free Rate) / Expected Volatility
def statistics(weights: np.ndarray, returns: pd.DataFrame) -> np.ndarray:
    portfolio_return = np.sum(returns.mean() * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
    return np.array(
        [
            portfolio_return,
            portfolio_volatility,
            portfolio_return / portfolio_volatility,
        ]
    )


# Minimize the negative Sharpe Ratio: Maximize the Sharpe Ratio
def min_func_sharpe(weights: np.ndarray, returns: pd.DataFrame) -> float:
    return -statistics(weights, returns)[2]


# Finds the optimal portfolio weights that maximizes the Sharpe Ratio
def optimize_portfolio(
    init_guess: np.ndarray, returns: pd.DataFrame, stocks: List[str] = None
) -> np.ndarray:
    # Step 1: Pre-Optimization
    len_stocks = len(stocks)
    bounds = [(0, 1) for _ in range(len_stocks)]
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    pre_optimum = optimization.minimize(
        fun=min_func_sharpe,
        x0=init_guess,
        args=returns,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    pre_optimum_weights = pre_optimum["x"]
    sorted_indices = np.argsort(-pre_optimum_weights)
    top_N_indices = sorted_indices[:30]

    # Step 2: Final Optimization with 5-20 stock constraint
    reduced_returns = returns.iloc[:, top_N_indices]
    reduced_weights = pre_optimum_weights[top_N_indices]

    final_optimum = optimization.minimize(
        fun=min_func_sharpe,
        x0=reduced_weights,
        args=reduced_returns,
        method="SLSQP",
        bounds=bounds[: len(top_N_indices)],
        constraints=constraints,
    )

    final_optimum_weights = final_optimum["x"]
    final_optimum_weights[final_optimum_weights < 0.05] = 0
    non_zero_weights = final_optimum_weights > 0

    # Ensure only 5-20 stocks are selected
    if np.sum(non_zero_weights) < 5 or np.sum(non_zero_weights) > 20:
        return None

    final_optimum_weights = final_optimum_weights[non_zero_weights]
    final_optimum_weights = final_optimum_weights / np.sum(final_optimum_weights)
    mean_return, volatility, sharpe_ratio = statistics(
        final_optimum_weights, reduced_returns.iloc[:, non_zero_weights]
    )

    return final_optimum_weights, mean_return, volatility, sharpe_ratio


def min_func_variance(weights: np.ndarray, returns: pd.DataFrame) -> float:
    return statistics(weights, returns)[1]


def portfolio_return(weights: np.ndarray, returns: pd.DataFrame) -> float:
    return statistics(weights, returns)[0]


def efficient_portfolios(
    returns: pd.DataFrame, stocks: List[str] = None, target_return: float = 0.1
) -> np.ndarray:
    # Step 1: Pre-Optimization
    len_stocks = len(stocks)
    bounds = [(0, 1) for _ in range(len_stocks)]

    init_guess = np.array([1.0 / len_stocks for _ in range(len_stocks)])

    pre_optimum = optimization.minimize(
        fun=min_func_variance,
        x0=init_guess,
        args=returns,
        method="SLSQP",
        bounds=bounds,
        constraints=(
            {
                "type": "eq",
                "fun": lambda x: portfolio_return(x, returns) - target_return,
            },
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        ),
    )

    pre_optimum_weights = pre_optimum["x"]
    sorted_indices = np.argsort(-pre_optimum_weights)
    top_N_indices = sorted_indices[:30]

    # Step 2: Final Optimization with 5-20 stock constraint
    reduced_returns = returns.iloc[:, top_N_indices]
    reduced_weights = pre_optimum_weights[top_N_indices]

    final_optimum = optimization.minimize(
        fun=min_func_variance,
        x0=reduced_weights,
        args=reduced_returns,
        method="SLSQP",
        bounds=bounds[: len(top_N_indices)],
        constraints=(
            {
                "type": "eq",
                "fun": lambda x: portfolio_return(x, reduced_returns) - target_return,
            },
            {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        ),
    )

    final_optimum_weights = final_optimum["x"]
    final_optimum_weights[final_optimum_weights < 0.05] = 0
    non_zero_weights = final_optimum_weights > 0

    # Ensure only 5-20 stocks are selected
    if np.sum(non_zero_weights) < 5 or np.sum(non_zero_weights) > 20:
        return None

    final_optimum_weights = final_optimum_weights[non_zero_weights]
    final_optimum_weights = final_optimum_weights / np.sum(final_optimum_weights)

    mean_return, volatility, sharpe_ratio = statistics(
        final_optimum_weights, reduced_returns.iloc[:, non_zero_weights]
    )

    return final_optimum_weights, mean_return, volatility, sharpe_ratio


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
