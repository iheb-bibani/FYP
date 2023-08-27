import pandas as pd
import yfinance as yf
import datetime
import time
import talib
from urllib.error import HTTPError
import numpy as np
import psycopg2
from postgres import connection


def main() -> None:
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM equities")
    ticker_source = cursor.fetchall()
    cursor.close()
    data = pd.DataFrame(ticker_source, columns=["id", "name", "ticker", "sector"])
    print(data)
    download_and_store_historical_data(connection)

    # Prototyping for adding TA to data
    stock_data_dict = add_ta_to_data(connection)
    for ticker, data in stock_data_dict.items():
        print(f"Data for {ticker}:")
        data.replace(0, np.nan, inplace=True)
        print(data.describe())
        print(data.info())
        print(data)

    connection.close()


def download_and_store_historical_data(
    connection: psycopg2.extensions.connection,
) -> None:
    cursor = connection.cursor()
    cursor.execute("SELECT ticker FROM equities")
    tickers = cursor.fetchall()

    for ticker_tuple in tickers:
        ticker = ticker_tuple[0]

        # Create a new table for each stock if it doesn't exist
        table_name = f"stock_{ticker[:3]}" if ticker != "^STI" else "STI" # Remove the .SI suffix
        cursor.execute(
            f"CREATE TABLE IF NOT EXISTS {table_name} (Date TEXT, Open REAL, High REAL, Low REAL, Close REAL, Adj_Close REAL, Volume REAL)"
        )

        # Get the latest date in the database
        cursor.execute(f"SELECT MAX(Date) FROM {table_name}")
        latest_date_in_db = cursor.fetchone()[0]
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
            start_date = "2009-12-01"

        while True:
            try:
                historical_data = yf.download(ticker, start=start_date, interval="1d")
                break
            except HTTPError:
                print(f"Rate limit exceeded for {ticker}. Retrying in 60 seconds...")
                time.sleep(60)

        # Insert new rows into the table
        for index, row in historical_data.iterrows():
            values = (
                index.strftime("%Y-%m-%d"),
                row["Open"],
                row["High"],
                row["Low"],
                row["Close"],
                row["Adj Close"],
                row["Volume"],
            )
            cursor.execute(
                f"INSERT INTO {table_name} VALUES (%s, %s, %s, %s, %s, %s, %s)",
                values,
            )
        connection.commit()

    cursor.close()


def add_ta_to_data(connection: psycopg2.extensions.connection) -> dict:
    cursor = connection.cursor()
    cursor.execute("SELECT ticker FROM equities LIMIT 1")
    tickers = cursor.fetchall()
    stock_data_dict = {}
    for ticker_tuple in tickers:
        ticker = ticker_tuple[0]
        table_name = f"stock_{ticker[:3]}"  # Remove the .SI suffix
        query = f"SELECT * FROM {table_name}"
        cursor.execute(query)
        data = cursor.fetchall()
        columns = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj_Close",
            "Volume",
        ]
        df = pd.DataFrame(data, columns=columns)
        df[["Open", "High", "Low", "Close", "Adj_Close", "Volume"]] = df[
            ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]
        ].replace(0, np.nan)

        # Momentum Indicators
        df["APO"] = talib.APO(df["Adj_Close"])
        df["KAMA"] = talib.KAMA(df["Adj_Close"])
        df["PPO"] = talib.PPO(df["Adj_Close"])
        df["ROC"] = talib.ROC(df["Adj_Close"])
        df["RSI"] = talib.RSI(df["Adj_Close"])
        df["StochRSI_K"], df["StochRSI_D"] = talib.STOCHRSI(df["Adj_Close"])
        df["Stochastic_K"], df["Stochastic_D"] = talib.STOCH(
            df["High"], df["Low"], df["Adj_Close"]
        )
        df["Ultimate_Oscillator"] = talib.ULTOSC(df["High"], df["Low"], df["Adj_Close"])
        df["Williams_R"] = talib.WILLR(df["High"], df["Low"], df["Adj_Close"])
        df["DX"] = talib.DX(df["High"], df["Low"], df["Adj_Close"])
        df["MINUS_DI"] = talib.MINUS_DI(df["High"], df["Low"], df["Adj_Close"])
        df["PLUS_DI"] = talib.PLUS_DI(df["High"], df["Low"], df["Adj_Close"])
        df["MINUS_DM"] = talib.MINUS_DM(df["High"], df["Low"])
        df["PLUS_DM"] = talib.PLUS_DM(df["High"], df["Low"])

        # Volume Indicators
        df["ADI"] = talib.AD(df["High"], df["Low"], df["Adj_Close"], df["Volume"])
        df["OBV"] = talib.OBV(df["Adj_Close"], df["Volume"])
        df["ADOSC"] = talib.ADOSC(df["High"], df["Low"], df["Adj_Close"], df["Volume"])
        df["OBV"] = talib.OBV(df["Adj_Close"], df["Volume"])

        # Volatility Indicators
        df["ATR"] = talib.ATR(df["High"], df["Low"], df["Adj_Close"])
        df["Upper_Band"], df["Middle_Band"], df["Lower_Band"] = talib.BBANDS(
            df["Adj_Close"]
        )
        df["NATR"] = talib.NATR(df["High"], df["Low"], df["Adj_Close"])
        df["TRANGE"] = talib.TRANGE(df["High"], df["Low"], df["Adj_Close"])

        # Trend Indicators
        df["ADX"] = talib.ADX(df["High"], df["Low"], df["Adj_Close"])
        df["Aroon_Up"], df["Aroon_Down"] = talib.AROON(df["High"], df["Low"])
        df["CCI"] = talib.CCI(df["High"], df["Low"], df["Adj_Close"])
        df["EMA"] = talib.EMA(df["Adj_Close"])
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = talib.MACD(df["Adj_Close"])
        df["PSAR"] = talib.SAR(df["High"], df["Low"])
        df["SMA"] = talib.SMA(df["Adj_Close"])
        df["TRIX"] = talib.TRIX(df["Adj_Close"])
        df["WMA"] = talib.WMA(df["Adj_Close"])
        df["HT_TRENDLINE"] = talib.HT_TRENDLINE(df["Adj_Close"])
        df["HT_DCPERIOD"] = talib.HT_DCPERIOD(df["Adj_Close"])
        df["HT_DCPHASE"] = talib.HT_DCPHASE(df["Adj_Close"])
        df["HT_PHASOR_INPHASE"], df["HT_PHASOR_QUADRATURE"] = talib.HT_PHASOR(
            df["Adj_Close"]
        )
        df["HT_SINE_SINE"], df["HT_SINE_LEADSINE"] = talib.HT_SINE(df["Adj_Close"])
        df["HT_TRENDMODE"] = talib.HT_TRENDMODE(df["Adj_Close"])

        # Other Indicators
        df["BETA"] = talib.BETA(df["High"], df["Low"])
        df["CORREL"] = talib.CORREL(df["High"], df["Low"])
        df["LINEARREG"] = talib.LINEARREG(df["Adj_Close"])
        df["LINEARREG_ANGLE"] = talib.LINEARREG_ANGLE(df["Adj_Close"])
        df["LINEARREG_INTERCEPT"] = talib.LINEARREG_INTERCEPT(df["Adj_Close"])
        df["LINEARREG_SLOPE"] = talib.LINEARREG_SLOPE(df["Adj_Close"])
        df["STDDEV"] = talib.STDDEV(df["Adj_Close"])
        df["TSF"] = talib.TSF(df["Adj_Close"])
        df["VAR"] = talib.VAR(df["Adj_Close"])

        stock_data_dict[ticker] = df

    cursor.close()
    return stock_data_dict


if __name__ == "__main__":
    main()
