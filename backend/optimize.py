import numpy as np
import pandas as pd
import scipy.optimize as optimization
import argparse
from datetime import datetime
from typing import Tuple, List
from demo import show_random_portfolios, show_optimal_portfolio, print_optimal_portfolio
import psycopg2
from postgres import connection

def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio analysis.")
    parser.add_argument('--start_date', type=str, default="2022-01-01", help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, default="2022-12-31", help='End date in YYYY-MM-DD format')

    args = parser.parse_args()
    start_date = args.start_date
    end_date = args.end_date

    # Validate start_date and end_date
    if not is_valid_date(start_date) or not is_valid_date(end_date):
        print("Invalid date format. Dates should be in YYYY-MM-DD format and should be valid.")
        return

    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

    if start_date_dt >= end_date_dt:
        print("Start date must be before end date.")
        return
    
    data = download_data(start_date, end_date, connection)
    log_return = calculate_returns(data)
    tickers = data.columns.tolist()  # Extracting the tickers as a list
    run_id = store_ticker_run(
        start_date, end_date, tickers, connection
    )  # Storing the run information

    # Generating and storing random portfolios
    weights, means, risks = generate_portfolios(log_return, stocks=tickers)
    store_portfolio_weights(weights, means, risks, run_id, connection)
    #show_random_portfolios(means, risks)

    # Optimizing and storing the optimal portfolio
    optimum, expected_return, volatility, sharpe_ratio = optimize_portfolio(
        weights, log_return, tickers
    )
    store_optimal_weights(optimum, expected_return, volatility, run_id, connection)

    # Printing and showing the optimal portfolio
    # print_optimal_portfolio(optimum, expected_return, volatility, sharpe_ratio)
    # show_optimal_portfolio(expected_return, volatility, means, risks)

    connection.close()

def is_valid_date(date_str: str, format: str = "%Y-%m-%d") -> bool:
    try:
        datetime.strptime(date_str, format)
        return True
    except ValueError:
        return False
    
import psycopg2
import pandas as pd

def download_data(start_date: str, end_date: str, connection: psycopg2.extensions.connection) -> pd.DataFrame:
    cursor = connection.cursor()
    cursor.execute("SELECT ticker FROM equities")
    tickers = cursor.fetchall()
    stock_data = {}

    for stock_tuple in tickers:
        stock = stock_tuple[0]
        table_name = f"stock_{stock[:3]}"  # Remove the .SI suffix
        query = f'SELECT Date, Close FROM {table_name} WHERE Date >= %s AND Date <= %s'
        cursor.execute(query, (start_date, end_date))
        data = cursor.fetchall()
        if data:
            dates, closes = zip(*data)
            stock_data[stock] = pd.Series(closes, index=pd.to_datetime(dates))

    cursor.close()

    return pd.DataFrame(stock_data)


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    log_return = np.log(data / data.shift(1))[1:]  # Daily log returns
    return log_return


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


def generate_portfolios(
    returns: pd.DataFrame,
    num_portfolios: int = 10000,
    stocks: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    portfolio_means = []
    portfolio_risks = []
    portfolio_weights = []
    sharpe_ratios = []
    num_stocks = len(stocks)

    for _ in range(num_portfolios):
        w = generate_random_weights(num_stocks)
        portfolio_weights.append(w)
        mean_return = np.sum(returns.mean() * w)
        risk = np.sqrt(np.dot(w.T, np.dot(returns.cov(), w)))
        portfolio_means.append(mean_return)
        portfolio_risks.append(risk)
        sharpe_ratios.append(mean_return / risk)

    # Sorting the portfolios by Sharpe ratio
    sorted_indices = np.argsort(sharpe_ratios)[
        ::-1
    ]  # Reverse to sort in descending order
    # Round all weights to 3dp
    portfolio_weights = np.array([np.round(weight, 3) for weight in portfolio_weights])[
        sorted_indices
    ]
    portfolio_means = np.array(portfolio_means)[sorted_indices]
    portfolio_risks = np.array(portfolio_risks)[sorted_indices]

    return portfolio_weights, portfolio_means, portfolio_risks


# Sharpe Ratio = (Expected Return - Risk Free Rate) / Expected Volatility
def statistics(
    weights: np.ndarray, returns: pd.DataFrame
) -> np.ndarray:
    portfolio_return = np.sum(returns.mean() * weights)
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(returns.cov(), weights))
    )
    return np.array(
        [
            portfolio_return,
            portfolio_volatility,
            portfolio_return / portfolio_volatility,
        ]
    )


# Minimize the negative Sharpe Ratio: Maximize the Sharpe Ratio
def min_func_sharpe(
    weights: np.ndarray, returns: pd.DataFrame
) -> float:
    return -statistics(weights, returns)[2]


# Finds the optimal portfolio weights that maximizes the Sharpe Ratio
def optimize_portfolio(
    weights: np.ndarray, returns: pd.DataFrame, stocks: List[str] = None
) -> np.ndarray:
    len_stocks = len(stocks)
    selected_stocks_count = 0
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(len_stocks))
    optimum_weights = weights[0]

    while selected_stocks_count < 4 or selected_stocks_count > 20:
        print("Optimizing portfolio...")
        optimum = optimization.minimize(
            fun=min_func_sharpe,
            x0=optimum_weights,
            args=returns,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        optimum_weights = optimum["x"]
        optimum_weights[optimum_weights < 0.05] = 0
        selected_stocks_count = np.sum(optimum_weights > 0)

        total_weights = np.sum(optimum_weights)
        optimum_weights = np.round(optimum_weights / total_weights, 3)

    total_weights = np.sum(optimum_weights)
    print(f"Total weights: {total_weights}")
    mean_return, volatility, sharpe_ratio = statistics(optimum_weights, returns)
    return optimum_weights, mean_return, volatility, sharpe_ratio


def store_ticker_run(
    start_date: str, end_date: str, tickers: List[str], connection: psycopg2.extensions.connection
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

if __name__ == "__main__":
    main()
