import pandas as pd
import yfinance as yf
import datetime
import psycopg2
from postgres import connection


def main() -> None:
    ticker_source = fetch_tickers()
    print(ticker_source)
    filtered_stocks = filter_stocks(ticker_source)
    print(f"{len(filtered_stocks)} stocks remaining")
    next_index = len(filtered_stocks)
    sti_index = [next_index+1,"Straight Times Index","^STI","Benchmark"] # Adding STI as a benchmark
    filtered_stocks.loc[next_index] = sti_index
    store_to_db(filtered_stocks, connection)
    connection.close()


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


def store_to_db(
    ticker_source: pd.DataFrame, connection: psycopg2.extensions.connection
) -> None:
    cursor = connection.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS equities (id SERIAL PRIMARY KEY, name TEXT, ticker TEXT UNIQUE, sector TEXT)"
    )
    
    # Checking for duplicate tickers
    for _, stock in ticker_source.iterrows():
        cursor.execute(
            "INSERT INTO equities (name, ticker, sector) VALUES (%s, %s, %s) ON CONFLICT (ticker) DO NOTHING",
            (stock[1], stock[2], stock[3])
        )
        
    connection.commit()
    cursor.close()


if __name__ == "__main__":
    main()
