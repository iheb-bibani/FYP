import backtrader as bt
import yfinance as yf
import psycopg2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from postgres import connection
from typing import List, Tuple

def main(run_id: int):
    financial_crisis = get_financial_crisis(connection)
    if not run_id: return
    optimal_weights = get_optimal_weights(connection, run_id)
    tickers = get_ticker_data(connection, run_id)
    filtered_tickers, filtered_weights = filter_tickers(tickers, optimal_weights)
    
    for crisis in financial_crisis[:1]:
        crisis_name, start_date, end_date = crisis
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")
        print(f"Analyzing {crisis_name} from {start_date} to {end_date}")
        data = download_data(filtered_tickers, start_date, end_date, connection)
        benchmark = download_data(["^STI"], start_date, end_date, connection)
        print(data)
        print(benchmark)
    
def get_financial_crisis(connection: psycopg2.extensions.connection) -> List[Tuple[str, datetime, datetime]]:
    cursor = connection.cursor()
    cursor.execute("SELECT crisis_name, start_date, end_date FROM financial_crisis")
    rows = cursor.fetchall()
    cursor.close()
    return rows

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
    tickers: np.ndarray, start_date: str, end_date:str, connection: psycopg2.extensions.connection
) -> pd.DataFrame:
    cursor = connection.cursor()
    stock_data = {}
    
    for stock in tickers:
        table_name = f"stock_{stock[:3]}" if stock != "^STI" else "STI"  # Remove the .SI suffix
        query = f"SELECT Date, Adj_Close FROM {table_name} WHERE Date >= %s AND Date <= %s"
        cursor.execute(query, (start_date, end_date))
        data = cursor.fetchall()
        if data:
            dates, closes = zip(*data)
            stock_data[stock] = pd.Series(closes, index=pd.to_datetime(dates))

    cursor.close()

    return pd.DataFrame(stock_data)
    
if __name__=="__main__":
    main(1)