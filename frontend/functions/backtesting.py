import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime
from postgres import connection
from typing import List, Tuple
from functions.database import get_ticker_data, filter_tickers


def main(
    run_id: int, portfolio_value: int, optimal_weights: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    financial_crisis = get_financial_crisis(connection)
    if not run_id:
        return
    tickers = get_ticker_data(connection, run_id)
    filtered_tickers, filtered_weights = filter_tickers(tickers, optimal_weights)

    # Create an empty list to store the results
    scenario_summary_data = []
    sector_returns = []
    for crisis in financial_crisis:
        crisis_name, start_date, end_date = crisis
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")

        data = download_data(filtered_tickers, start_date, end_date, connection)
        benchmark = download_data(["^STI"], start_date, end_date, connection)

        portfolio_return, benchmark_return, profit_and_loss = calculate_profit_and_loss(
            data, benchmark, filtered_weights
        )
        stock_and_sector_returns = get_stock_and_sector_returns(
            data, filtered_tickers, crisis_name, portfolio_value
        )
        scenario_summary_data.append(
            {
                "Scenario": crisis_name,
                "P/L ($)": profit_and_loss * portfolio_value / 100,
                "P/L (%)": profit_and_loss,
                "P. Return ($)": portfolio_return * portfolio_value / 100,
                "P. Return (%)": portfolio_return,
                "B. Return ($)": benchmark_return * portfolio_value / 100,
                "B. Return (%)": benchmark_return,
            }
        )
        sector_returns.append(stock_and_sector_returns)

    scenario_summary_df = pd.DataFrame(scenario_summary_data)
    scenario_summary_df = (
        scenario_summary_df.sort_values(
            by=scenario_summary_df.columns[3], ascending=False
        )
        .reset_index(drop=True)
        .set_index("Scenario")
    )

    combined_sector_returns = pd.concat(sector_returns, axis=1)

    return scenario_summary_df, combined_sector_returns


def get_financial_crisis(
    connection: psycopg2.extensions.connection,
) -> List[Tuple[str, datetime, datetime]]:
    cursor = connection.cursor()
    cursor.execute("SELECT crisis_name, start_date, end_date FROM financial_crisis")
    rows = cursor.fetchall()
    cursor.close()
    return rows


def download_data(
    tickers: np.ndarray,
    start_date: str,
    end_date: str,
    connection: psycopg2.extensions.connection,
) -> pd.DataFrame:
    cursor = connection.cursor()
    stock_data = {}

    for stock in tickers:
        table_name = (
            f"stock_{stock[:3]}" if stock != "^STI" else "STI"
        )  # Remove the .SI suffix
        query = f"SELECT Adj_Close FROM {table_name} WHERE Date >= %s AND Date <= %s ORDER BY Date"
        cursor.execute(query, (start_date, end_date))
        data = cursor.fetchall()
        if data:
            closes = [item[0] for item in data]
            stock_data[stock] = pd.Series(closes)

    cursor.close()

    return pd.DataFrame(stock_data)


def calculate_profit_and_loss(
    data: pd.DataFrame, benchmark: pd.DataFrame, weights: np.ndarray
) -> Tuple[float, float, float]:
    benchmark_return = calculate_total_return(benchmark)
    portfolio_return = calculate_total_return(data, weights)
    profit_and_loss = portfolio_return - benchmark_return
    return portfolio_return, benchmark_return, profit_and_loss


def calculate_total_return(data: pd.DataFrame, weights: np.ndarray = None) -> float:
    start_price = data.iloc[0]
    end_price = data.iloc[-1]

    if weights is not None:
        start_price = np.dot(start_price, weights)
        end_price = np.dot(end_price, weights)
    else:
        start_price = start_price.iloc[0]
        end_price = end_price.iloc[0]

    total_return = ((end_price - start_price) / start_price) * 100
    return total_return


def calculate_stock_return(df):
    start_price = df.iloc[0, 0]
    end_price = df.iloc[-1, 0]
    return (end_price - start_price) / start_price * 100


def get_stock_and_sector_returns(
    data: pd.DataFrame, filtered_tickers: list, crisis_name: str, portfolio_value: int
) -> pd.DataFrame:
    cursor = connection.cursor()
    cursor.execute(
        "SELECT * FROM equities WHERE ticker IN %s", (tuple(filtered_tickers),)
    )
    equities_data = cursor.fetchall()
    cursor.close()

    equities_df = pd.DataFrame(
        equities_data, columns=["id", "name", "ticker", "sector"]
    )

    result_list = []

    for _, row in equities_df.iterrows():
        ticker = row["ticker"]
        sector = row["sector"]
        name = row["name"]

        if ticker not in filtered_tickers:
            continue

        stock_df = data[[ticker]].dropna()
        if stock_df.empty:
            print(f"No data for {ticker}")
            continue

        stock_return = calculate_stock_return(stock_df)
        crisis_name_intials = "".join([name[0] for name in crisis_name.split(" ")])
        result_list.append(
            {
                "Ticker": ticker,
                "Name": name,
                "Sector": sector,
                f"{crisis_name} Returns %": stock_return,
                f"{crisis_name_intials} Returns $": stock_return
                * portfolio_value
                / 100,
            }
        )

    result_df = pd.DataFrame(result_list)
    grouped_df = result_df.set_index(["Sector", "Name", "Ticker"])

    return grouped_df


if __name__ == "__main__":
    main(2, 100000)
