import pandas as pd
import yfinance as yf
import datetime
import sqlite3
from typing import List


def main() -> None:
    ticker_source = fetch_tickers()
    print(ticker_source)
    filtered_stocks = filter_stocks(ticker_source)
    print(f"{len(filtered_stocks)} stocks remaining")
    store_to_db(filtered_stocks)


def fetch_tickers() -> pd.DataFrame:
    return pd.read_html(
        "https://topforeignstocks.com/listed-companies-lists/the-complete-list-of-listed-companies-in-singapore/"
    )[0]


def filter_stocks(ticker_source: pd.DataFrame) -> pd.DataFrame:
    counter = 0
    delisted = []
    non_equity = []
    insufficient_data = []
    insufficient_market_cap = []

    for idx, ticker in enumerate(ticker_source["Code"]):
        try:
            data = yf.Ticker(ticker)
            quote_type = data.info["quoteType"]
        except Exception as e:
            print(f"{ticker} is delisted")
            delisted.append(idx)
            continue

        if quote_type != "EQUITY":
            print(f"{ticker} is not an equity. Data not retrieved.")
            non_equity.append(idx)
            continue

        if "marketCap" not in data.info or "averageVolume" not in data.info:
            print(idx, ticker)
            print(f"{ticker} has insufficient data. Data not retrieved.")
            insufficient_data.append(idx)
            continue

        market_cap = data.info["marketCap"]
        avg_volume = data.info["averageVolume"]
        first_trade_date = datetime.datetime.fromtimestamp(
            data.info["firstTradeDateEpochUtc"]
        ).strftime("%Y-%m-%d")
        if market_cap < 10e6 or avg_volume < 500e3 or first_trade_date > "2010-01-01":
            print(
                f"{ticker} has insufficient market cap or average volume. Data not retrieved."
            )
            insufficient_market_cap.append(idx)
        else:
            counter += 1

    print(f"{counter} tickers have sufficient market cap and average volume.")
    to_drop = delisted + non_equity + insufficient_data + insufficient_market_cap
    ticker_source.drop(to_drop, inplace=True)
    return ticker_source


def store_to_db(ticker_source: pd.DataFrame) -> None:
    connection =  sqlite3.connect("../databases/relational.db")
    cursor = connection.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS equities (id INTEGER PRIMARY KEY, name TEXT, ticker TEXT, sector TEXT)"
    )
    data = cursor.execute("SELECT * FROM equities").fetchall()

    if data == []:
        for idx, stock in enumerate(ticker_source.iterrows()):
            cursor.execute(
                f'INSERT INTO equities VALUES ({idx}, "{stock[1][1]}", "{stock[1][2]}", "{stock[1][3]}")'
            )
    else:
        data = pd.DataFrame(data, columns=["id", "name", "ticker", "sector"])
        print(data)
    connection.commit()
    connection.close()


if __name__ == "__main__":
    main()
