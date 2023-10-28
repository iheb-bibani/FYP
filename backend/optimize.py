import numpy as np
import pandas as pd
import scipy.optimize as optimization
import pandas_market_calendars as mcal
import argparse
from datetime import datetime
from typing import Tuple, List
from demo import (
    show_random_portfolios,
    show_optimal_portfolio,
    print_optimal_portfolio,
    show_efficient_frontier,
)
import asyncio
from postgres import create_pool
from scipy.stats import normaltest, skew, kurtosis
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

DEBUG = True


async def main() -> None:
    parser = argparse.ArgumentParser(description="Portfolio analysis.")
    parser.add_argument(
        "--start_date", "-sd",
        type=str,
        default="2022-01-03",
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end_date", "-ed",
        type=str,
        default="2022-12-30",
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--risk_free_rate", "-rfr",
        type=float,
        default=0.001,
        help="Risk free rate",
    )
    args = parser.parse_args()
    start_date = args.start_date
    end_date = args.end_date
    risk_free_rate = args.risk_free_rate
    
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

    trading_days = trading_days_between_dates(start_date_dt, end_date_dt)
    pool = await create_pool()
    try:
        async with pool.acquire() as connection:
            data = await download_data(start_date, end_date, connection)
            print(f"Number of tickers: {len(data.columns)}")
            log_return = calculate_returns(data)
            tickers = log_return.columns.tolist()

            weights, means, risks = generate_portfolios(log_return, 50000, tickers, trading_days, risk_free_rate)

            if DEBUG:
                show_random_portfolios(means, risks)
            else:
                run_id = await store_ticker_run(
                    start_date, end_date, tickers, connection
                )
                store_portfolio_db = asyncio.create_task(
                    run_with_new_connection(
                        pool, store_portfolio_weights, weights, means, risks, run_id
                    )
                )

            modified_returns, top_n_indices, original_len = sort_by_sharpe_ratio(
                log_return, top_n=100
            )

            optimum = optimize_portfolio(
                modified_returns, top_n_indices, original_len, lambda_val=0.1, num_trading_days=trading_days, risk_free_rate=risk_free_rate, skewness_weight=0, kurtosis_weight=0
            )

            expected_return, volatility, sharpe_ratio, skewness_optimum, kurtosis_optimum = statistics(
                optimum, log_return, lambda_val=0, num_trading_days=trading_days
            )
            
            backtesting_abs_return = 0
            optimum_tickers_to_weights = {}
            for idx in top_n_indices:
                if optimum[idx] > 0:
                    print(f"{idx}: {tickers[idx]}: {optimum[idx]}")
                    ticker = tickers[idx]
                    weight = optimum[idx]
                    optimum_tickers_to_weights[ticker] = weight
                    start_price = data[tickers[idx]][0]
                    end_price = data[tickers[idx]][-1]
                    backtesting_abs_return += (end_price - start_price)/start_price * optimum[idx]
                    
                    
            backtesting_log_return = np.log(1 + backtesting_abs_return)
            expected_absolute_return = np.exp(expected_return) - 1
            print(f"Backtesting absolute return: {backtesting_abs_return}")
            print(f"Backtesting log return: {backtesting_log_return}")
            print(f"Expected absolute return: {expected_absolute_return}")
            print(f"Expected log return: {expected_return}")
            print(f"Skewness: {skewness_optimum}")
            print(f"Kurtosis: {kurtosis_optimum}")
            
            if DEBUG:
                print_optimal_portfolio(
                    optimum, expected_return, volatility, sharpe_ratio
                )
                show_optimal_portfolio(expected_return, volatility, means, risks)
                
            efficient_list = [[optimum, expected_return, volatility, skewness_optimum, kurtosis_optimum]]
            target_returns = np.linspace(expected_return*0.5, expected_return*1.2, 10)
            
            for target in target_returns:
                print(f"Target return: {target}")
                final_optimum_weights = efficient_portfolios(
                    modified_returns, top_n_indices, original_len, target,  num_trading_days=trading_days
                )
                if final_optimum_weights is None:
                    print("No efficient portfolio found.")
                    continue
                else:
                    expected_return, volatility, sharpe_ratio, skewness_efficient, kurtosis_efficient = statistics(
                        final_optimum_weights, log_return
                    )
                    values = (
                        final_optimum_weights,
                        expected_return,
                        volatility,
                        skewness_efficient,
                        kurtosis_efficient,
                    )
                efficient_list.append(values)
            max_length = max([len([x for x in lst[0] if x != 0]) for lst in efficient_list])
            print(f"Maximum length of portfolio with non-zero weights: {max_length}")

            efficient_list.sort(key=lambda x: x[1])
            if DEBUG:
                show_efficient_frontier(
                    np.array([x[1] for x in efficient_list]),
                    np.array([x[2] for x in efficient_list]),
                    means,
                    risks,
                )
            else:
                
                store_efficient_db = asyncio.create_task(
                    run_with_new_connection(
                        pool,
                        store_efficient_frontier,
                        efficient_list,
                        run_id,
                    )
                )

            if not DEBUG:
                await asyncio.gather(
                    store_portfolio_db, store_efficient_db
                )
    except asyncio.CancelledError as e:
        print(f"Connection cancelled: {e}")
    finally:
        await pool.close()


def is_valid_date(date_str: str, format: str = "%Y-%m-%d") -> bool:
    try:
        datetime.strptime(date_str, format)
        return True
    except ValueError:
        return False


def trading_days_between_dates(
    start_date: datetime, end_date: datetime, exchange: str = "XSES"
) -> int:
    cal = mcal.get_calendar(exchange)
    trading_days = cal.schedule(start_date=start_date, end_date=end_date)
    return len(trading_days)


async def run_with_new_connection(pool, func, *args):
    async with pool.acquire() as connection:
        await func(*args, connection=connection)


async def download_data(start_date: str, end_date: str, connection) -> pd.DataFrame:
    tickers = await connection.fetch("SELECT ticker FROM equities")
    stock_data = {}
    # Subtract 5 years from start_date
    # end_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    # start_date = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=5*365)).strftime("%Y-%m-%d")

    for stock_tuple in tickers:
        stock = stock_tuple["ticker"]
        if stock == "^STI":
            continue
        table_name = f"stock_{stock[:3]}"  # Remove the .SI suffix
        query = (
            f"SELECT Date, close FROM {table_name} WHERE Date >= $1 AND Date <= $2"
        )
        try:
            data = await connection.fetch(query, start_date, end_date)
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

                if closes[-1] > 0.2:
                    stock_data[stock] = pd.Series(closes, index=pd.to_datetime(dates))
                else:
                    print(f"Skipping {stock} as close price is less than $0.2")

        except Exception as e:
            continue

    return pd.DataFrame(stock_data)


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    column_names = data.columns
    shifted_data = data.shift(1).values
    data_values = data.values
    
    # Calculate log returns
    log_return = np.where(
        (data_values > 0) & (shifted_data > 0),
        np.log(data_values / shifted_data),
        np.nan,
    )[1:]
    
    # Drop columns with negative or zero daily prices
    cols_to_drop_negative = column_names[(data <= 0).any()].tolist()
    
    # Drop columns with NaN in log returns
    log_return_df = pd.DataFrame(log_return, columns=column_names)
    cols_to_drop_nan = log_return_df.columns[log_return_df.isna().any()].tolist()
    
    # Combine both lists and drop from DataFrame
    cols_to_drop = set(cols_to_drop_negative + cols_to_drop_nan)
    log_return_df.drop(columns=cols_to_drop, inplace=True)
    
    print(f"Columns dropped: {cols_to_drop}")
    
    normal_returns = count_normal_returns(log_return_df.to_numpy())
    print(f"{normal_returns} returns are normally distributed.")
    
    return log_return_df

def count_normal_returns(returns: np.ndarray) -> int:
    _, p_values = normaltest(returns)
    return sum(p_values > 0.05)


def generate_random_weights(num_stocks: int) -> np.ndarray:
    num_selected_stocks = np.random.randint(5, 21)
    selected_stocks = np.random.choice(
        num_stocks, int(num_selected_stocks), replace=False
    )
    weights = np.zeros(num_stocks, dtype=np.float32)
    random_weights = np.random.uniform(0.05, 1, int(num_selected_stocks))
    random_weights = np.round(random_weights/np.sum(random_weights), 4)
    for idx, weight in zip(selected_stocks, random_weights):
        weights[idx] = weight

    return weights


def generate_portfolios(
    returns: pd.DataFrame,
    num_portfolios: int = 50000,
    stocks: List[str] = None,
    num_trading_days: int = 252,
    risk_free_rate: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_stocks = len(stocks)
    unique_portfolios = set()
    portfolio_weights = []

    while len(portfolio_weights) < num_portfolios:
        weights = generate_random_weights(num_stocks)
        weights_tuple = tuple(weights)
        if weights_tuple not in unique_portfolios:
            unique_portfolios.add(weights_tuple)
            portfolio_weights.append(weights_tuple)
            
    portfolio_weights = np.array(portfolio_weights)

    mean_return = np.sum(np.array(returns.mean().values) * portfolio_weights, axis=1) * num_trading_days

    cov_matrix = returns.cov().values

    portfolio_risk = np.sqrt(
        np.sum(portfolio_weights @ cov_matrix * portfolio_weights, axis=1)
    ) * np.sqrt(num_trading_days)

    sharpe_ratios = (mean_return - risk_free_rate) / portfolio_risk

    sorted_indices = np.argsort(sharpe_ratios)[::-1]  # Sort in descending order

    portfolio_weights = portfolio_weights[sorted_indices]
    portfolio_means = mean_return[sorted_indices]
    portfolio_risks = portfolio_risk[sorted_indices]

    return portfolio_weights, portfolio_means, portfolio_risks

def plot_distributions(normal_returns, log_returns):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(normal_returns, bins=50, kde=True, color='blue')
    plt.title('Distribution of Normal Returns')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    sns.histplot(log_returns, bins=50, kde=True, color='green')
    plt.title('Distribution of Log Returns')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    
    plt.legend().remove()
    plt.tight_layout()
    plt.show()
    

def downside_risk(returns, target=0.0):
    downside_diff = returns - target
    squared_downside_diff = downside_diff * (downside_diff < 0)
    downside_var = np.mean(squared_downside_diff)
    return np.sqrt(downside_var)
    
# Sharpe Ratio = (Expected Return - Risk Free Rate) / Expected Volatility
# Adjusted Sharpe Ratio = Sharpe Ratio * sqrt((1 - skewness * Sharpe Ratio + (kurtosis - 1) * (Sharpe Ratio ** 2)) / 2)
def statistics(
    weights: np.ndarray, log_returns: np.ndarray, lambda_val: float = 0.1, num_trading_days: int = 252, risk_free_rate: float = 0.00
) -> np.ndarray:
    
    mean_log_returns = np.mean(log_returns, axis=0)
    mean_abs_returns = np.exp(mean_log_returns) - 1
    total_mean_abs_return = np.sum(mean_abs_returns * weights)
    total_mean_log_return = np.log(1 + total_mean_abs_return)
    portfolio_return = total_mean_log_return * num_trading_days
    
    portfolio_log_returns = np.dot(log_returns, weights)
    portfolio_normal_returns = np.exp(portfolio_log_returns) - 1
    
    if isinstance(log_returns, pd.DataFrame):
        cov_matrix = log_returns.cov().values
    else:  # Assuming it's a NumPy array
        cov_matrix = np.cov(log_returns, rowvar=False)

    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(num_trading_days)

    raw_sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    if lambda_val != 0:
        l1_norm = np.sum(np.abs(weights))
        regularized_sharpe_ratio = raw_sharpe_ratio - lambda_val * l1_norm
    else:
        if DEBUG: plot_distributions(portfolio_normal_returns, portfolio_log_returns)
        regularized_sharpe_ratio = raw_sharpe_ratio
        
    # Calculate the skewness and kurtosis
    portfolio_skewness = skew(portfolio_log_returns)
    portfolio_kurtosis = kurtosis(portfolio_log_returns)
    
    portfolio_downside_risk = downside_risk(portfolio_log_returns, risk_free_rate)
    sortino_ratio = (portfolio_return - risk_free_rate) / portfolio_downside_risk
    
    return np.array([portfolio_return, portfolio_volatility, regularized_sharpe_ratio, portfolio_skewness, portfolio_kurtosis])


# Minimize the negative Sharpe Ratio: Maximize the Sharpe Ratio
def min_func_sharpe(
    weights: np.ndarray, returns: pd.DataFrame, lambda_val: float, num_trading_days: int = 252, risk_free_rate: float = 0.001,
    skewness_weight: float = 0.1, kurtosis_weight: float = 0.1
) -> float:
    stats = statistics(weights, returns, lambda_val, num_trading_days, risk_free_rate)
    sharpe_ratio = stats[2]
    portfolio_skewness = stats[3]
    portfolio_kurtosis = stats[4]

    adjusted_weights = np.copy(weights)
    adjusted_weights[adjusted_weights < 0.05] = 0
    non_zero_weights = np.sum(adjusted_weights > 0)
    if non_zero_weights == 0:
        penalty = 1000
    else:
        adjusted_weights = adjusted_weights / np.sum(adjusted_weights)
        penalty = 1000 if non_zero_weights < 5 or non_zero_weights > 20 else 0

    objective = -sharpe_ratio + penalty - skewness_weight * portfolio_skewness + kurtosis_weight * (portfolio_kurtosis - 3) ** 2
    
    return objective

# Finds the optimal portfolio weights that maximizes the Sharpe Ratio
def optimize_portfolio(
    modified_returns: np.ndarray,
    top_n_indices: int,
    original_len: int,
    lambda_val: float = 0.1,
    num_trading_days: int = 252,
    risk_free_rate: float = 0.001,
    skewness_weight: float = 0.1,
    kurtosis_weight: float = 0.1,
) -> np.ndarray:
    # Optimization
    len_stocks = len(top_n_indices)
    bounds = [(0, 1) for _ in range(len_stocks)]

    # Initialize the weights to be equal
    init_guess = np.ones(len_stocks) / len_stocks

    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    optimum = optimization.minimize(
        fun=min_func_sharpe,
        x0=init_guess,
        args=(modified_returns, lambda_val, num_trading_days, risk_free_rate, skewness_weight, kurtosis_weight),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    optimum_weights = optimum["x"]
    optimum_weights[optimum_weights < 0.05] = 0
    optimum_weights = np.round(optimum_weights / np.sum(optimum_weights), 4)
    non_zero_weights = np.sum(optimum_weights > 0)

    if 5 <= non_zero_weights <= 20:
        final_optimum_weights = np.zeros(original_len)

        for idx, original_idx in enumerate(top_n_indices):
            final_optimum_weights[original_idx] = optimum_weights[idx]

        return final_optimum_weights
    else:
        return 0


def portfolio_return(weights: np.ndarray, returns: pd.DataFrame, num_trading_days: int = 252, risk_free_rate: float = 0.001) -> float:
    return statistics(weights, returns, num_trading_days=num_trading_days, risk_free_rate=risk_free_rate)[0]


def efficient_portfolios(
    returns: np.ndarray, top_n_indices: int, original_len: int, target_return: float, lambda_val: float = 0.1, num_trading_days: int = 252, risk_free_rate: float = 0.001
) -> np.ndarray:
    # Optimization
    len_stocks = len(top_n_indices)
    bounds = [(0, 1) for _ in range(len_stocks)]

    # Initialize the weights to be equal
    init_guess = np.ones(len_stocks) / len_stocks

    constraints = [
        {
            "type": "eq",
            "fun": lambda x: portfolio_return(x, returns, num_trading_days, risk_free_rate) - target_return,
        },
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
    ]

    # Note: Changed objective function to min_func_sharpe
    optimum = optimization.minimize(
        fun=min_func_sharpe,
        x0=init_guess,
        args=(returns, lambda_val, num_trading_days, risk_free_rate),
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

def sort_by_sharpe_ratio(
    returns: pd.DataFrame, top_n: int = 100
) -> Tuple[np.ndarray, List[int], int]:
    
    mean_returns = np.array(returns.mean())
    std_dev = np.array(returns.std())
    skewness_values = skew(returns)
    kurtosis_values = kurtosis(returns)
    
    raw_sharpe_ratios = mean_returns / std_dev

    # Calculate Adjusted Sharpe Ratios
    adj_sharpe_ratios = raw_sharpe_ratios * np.sqrt(
        (1 - skewness_values * raw_sharpe_ratios + (kurtosis_values - 1) * (raw_sharpe_ratios ** 2)) / 2
    )
    
    sorted_indices = np.argsort(-adj_sharpe_ratios) 
    top_n_indices = sorted_indices[:top_n].tolist()
    reduced_returns = returns.iloc[:, top_n_indices]

    return reduced_returns, top_n_indices, len(returns.columns)


async def store_ticker_run(
    start_date: str, end_date: str, tickers: List[str], connection
) -> int:
    await connection.execute(
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
    run_id = await connection.fetchval(
        "INSERT INTO ticker_run (start_date, end_date, tickers) VALUES ($1, $2, $3) RETURNING id",
        start_date,
        end_date,
        tickers_str,
    )
    return run_id


async def store_portfolio_weights(
    portfolio_weights: np.ndarray,
    means: np.ndarray,
    risks: np.ndarray,
    run_id: int,
    connection,
):
    await connection.execute(
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
        await connection.execute(
            "INSERT INTO portfolio_weights (run_id, weight, returns, volatility) VALUES ($1, $2, $3, $4)",
            run_id,
            weights_str,
            mean,
            risk,
        )

async def store_efficient_frontier(
    efficient_list: List[np.ndarray],
    run_id: int,
    connection,
) -> int:
    await connection.execute(
        """
        CREATE TABLE IF NOT EXISTS efficient_frontier (
            id SERIAL PRIMARY KEY,
            run_id INTEGER,
            weight TEXT,
            returns REAL,
            volatility REAL,
            skewness REAL,
            kurtosis REAL,
            FOREIGN KEY(run_id) REFERENCES ticker_run(id)
        )
    """
    )
    for weights, mean, risk, skew, kurt in efficient_list:
        weights_str = ",".join([str(weight) for weight in weights])
        await connection.execute(
            "INSERT INTO efficient_frontier (run_id, weight, returns, volatility, skewness, kurtosis) VALUES ($1, $2, $3, $4, $5, $6)",
            run_id,
            weights_str,
            mean,
            risk,
            skew,
            kurt,
        )


if __name__ == "__main__":
    asyncio.run(main())
