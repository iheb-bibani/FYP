import pandas as pd
import yfinance as yf
import datetime
import sqlite3
import time
from urllib.error import HTTPError


def main() -> None:
    connection = sqlite3.connect("../databases/relational.db")
    cursor = connection.cursor()
    ticker_source = cursor.execute("SELECT * FROM equities").fetchall()
    cursor.close()
    data = pd.DataFrame(ticker_source, columns=["id", "name", "ticker", "sector"])
    print(data)
    download_and_store_historical_data()
    stock_data_dict = get_all_stock_data()
    for ticker, data in stock_data_dict.items():
        print(f"Data for {ticker}:")
        print(data)


def download_and_store_historical_data() -> None:
    connection = sqlite3.connect("../databases/relational.db")
    cursor = connection.cursor()
    tickers = cursor.execute("SELECT ticker FROM equities").fetchall()

    for ticker_tuple in tickers:
        ticker = ticker_tuple[0]

        # Create a new table for each stock if it doesn't exist
        table_name = f"stock_{ticker[:3]}"  # Remove the .SI suffix
        cursor.execute(
            f"CREATE TABLE IF NOT EXISTS {table_name} (Date TEXT, Open REAL, High REAL, Low REAL, Close REAL, Volume INTEGER, Dividends REAL, Stock_Splits REAL)"
        )

        # Get the latest date in the database
        latest_date_in_db = cursor.execute(
            f"SELECT MAX(Date) FROM {table_name}"
        ).fetchone()[0]
        if latest_date_in_db:
            # Start from the day after the latest date in the database
            yesterday_date = (
                datetime.datetime.today() - datetime.timedelta(days=1)
            ).strftime("%Y-%m-%d")
            if latest_date_in_db >= yesterday_date:
                # If the latest date is today, skip
                print(f"Data for {ticker} is up to date.")
                continue
            else:
                print(f"Data for {ticker} is not up to date. Updating...")
                start_date = (
                    datetime.datetime.strptime(latest_date_in_db, "%Y-%m-%d")
                    + datetime.timedelta(days=1)
                ).strftime("%Y-%m-%d")
        else:
            # Starts from 2010-01-01 if the table is empty
            print(f"Data for {ticker} is empty. Downloading...")
            start_date = "2010-01-01"

        # Handling Yahoo's rate limiter
        while True:
            try:
                stock_data = yf.Ticker(ticker)
                historical_data = stock_data.history(period="1d", start=start_date)
                historical_data.fillna(0, inplace=True)
                break
            except HTTPError:
                print(f"Rate limit exceeded for {ticker}. Retrying in 60 seconds...")
                time.sleep(60)

        # Insert new rows into the table
        for index, row in historical_data.iterrows():
            cursor.execute(
                f"INSERT INTO {table_name} VALUES ('{index.strftime('%Y-%m-%d')}', {row['Open']}, {row['High']}, {row['Low']}, {row['Close']}, {row['Volume']}, {row['Dividends']}, {row['Stock Splits']})"
            )

    connection.commit()
    connection.close()


def get_all_stock_data() -> dict:
    connection = sqlite3.connect("../databases/relational.db")
    cursor = connection.cursor()
    tickers = cursor.execute("SELECT ticker FROM equities").fetchall()

    stock_data_dict = {}
    for ticker_tuple in tickers:
        ticker = ticker_tuple[0]
        table_name = f"stock_{ticker[:3]}"  # Remove the .SI suffix
        query = f"SELECT * FROM {table_name}"
        data = cursor.execute(query).fetchall()
        columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Dividends",
            "Stock_Splits",
        ]
        stock_data_dict[ticker] = pd.DataFrame(data, columns=columns)

    cursor.close()
    connection.close()
    return stock_data_dict


if __name__ == "__main__":
    main()
