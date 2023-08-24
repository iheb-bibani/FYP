import pandas as pd
import yfinance as yf
import datetime
import sqlite3
import time
import talib
from urllib.error import HTTPError
import numpy as np


def main() -> None:
    connection = sqlite3.connect("../databases/relational.db")
    cursor = connection.cursor()
    ticker_source = cursor.execute("SELECT * FROM equities").fetchall()
    cursor.close()
    data = pd.DataFrame(ticker_source, columns=["id", "name", "ticker", "sector"])
    print(data)
    download_and_store_historical_data()
    
    # Prototyping for adding TA to data
    stock_data_dict = add_ta_to_data()
    for ticker, data in stock_data_dict.items():
        print(f"Data for {ticker}:")
        data.replace(0, np.nan, inplace=True)
        print(data.describe())
        print(data.info())
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
            start_date = "2009-12-01"

        while True:
            try:
                stock_data = yf.Ticker(ticker)
                historical_data = stock_data.history(period="1d", start=start_date)
                break
            except HTTPError:
                print(f"Rate limit exceeded for {ticker}. Retrying in 60 seconds...")
                time.sleep(60)

        # Insert new rows into the table
        for index, row in historical_data.iterrows():
            values = (
                index.strftime('%Y-%m-%d'),
                row['Open'],
                row['High'],
                row['Low'],
                row['Close'],
                row['Volume'],
                row['Dividends'],
                row['Stock Splits']
            )
            cursor.execute(
                f"INSERT INTO {table_name} VALUES (?, ?, ?, ?, ?, ?, ?, ?)", values
            )


    connection.commit()
    connection.close()


def add_ta_to_data() -> dict:
    connection = sqlite3.connect("../databases/relational.db")
    cursor = connection.cursor()
    tickers = cursor.execute("SELECT ticker FROM equities LIMIT 1").fetchall()

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
        df = pd.DataFrame(data, columns=columns)
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].replace(0, np.nan)
        
        # Momentum Indicators
        df['APO'] = talib.APO(df['Close'])
        df['KAMA'] = talib.KAMA(df['Close'])
        df['PPO'] = talib.PPO(df['Close'])
        df['ROC'] = talib.ROC(df['Close'])
        df['RSI'] = talib.RSI(df['Close'])
        df['StochRSI_K'], df['StochRSI_D'] = talib.STOCHRSI(df['Close'])
        df['Stochastic_K'], df['Stochastic_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
        df['Ultimate_Oscillator'] = talib.ULTOSC(df['High'], df['Low'], df['Close'])
        df['Williams_R'] = talib.WILLR(df['High'], df['Low'], df['Close'])
        df['DX'] = talib.DX(df['High'], df['Low'], df['Close'])
        df['MINUS_DI'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'])
        df['PLUS_DI'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'])
        df['MINUS_DM'] = talib.MINUS_DM(df['High'], df['Low'])
        df['PLUS_DM'] = talib.PLUS_DM(df['High'], df['Low'])


        # Volume Indicators
        df['ADI'] = talib.AD(df['High'], df['Low'], df['Close'], df['Volume'])
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        df['ADOSC'] = talib.ADOSC(df['High'], df['Low'], df['Close'], df['Volume'])
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])


        # Volatility Indicators
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'])
        df['Upper_Band'], df['Middle_Band'], df['Lower_Band'] = talib.BBANDS(df['Close'])
        df['NATR'] = talib.NATR(df['High'], df['Low'], df['Close'])
        df['TRANGE'] = talib.TRANGE(df['High'], df['Low'], df['Close'])


        # Trend Indicators
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'])
        df['Aroon_Up'], df['Aroon_Down'] = talib.AROON(df['High'], df['Low'])
        df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'])
        df['EMA'] = talib.EMA(df['Close'])
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = talib.MACD(df['Close'])
        df['PSAR'] = talib.SAR(df['High'], df['Low'])
        df['SMA'] = talib.SMA(df['Close'])
        df['TRIX'] = talib.TRIX(df['Close'])
        df['WMA'] = talib.WMA(df['Close'])
        df['HT_TRENDLINE'] = talib.HT_TRENDLINE(df['Close'])
        df['HT_DCPERIOD'] = talib.HT_DCPERIOD(df['Close'])
        df['HT_DCPHASE'] = talib.HT_DCPHASE(df['Close'])
        df['HT_PHASOR_INPHASE'], df['HT_PHASOR_QUADRATURE'] = talib.HT_PHASOR(df['Close'])
        df['HT_SINE_SINE'], df['HT_SINE_LEADSINE'] = talib.HT_SINE(df['Close'])
        df['HT_TRENDMODE'] = talib.HT_TRENDMODE(df['Close'])

        
        # Other Indicators
        df['BETA'] = talib.BETA(df['High'], df['Low'])
        df['CORREL'] = talib.CORREL(df['High'], df['Low'])
        df['LINEARREG'] = talib.LINEARREG(df['Close'])
        df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(df['Close'])
        df['LINEARREG_INTERCEPT'] = talib.LINEARREG_INTERCEPT(df['Close'])
        df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(df['Close'])
        df['STDDEV'] = talib.STDDEV(df['Close'])
        df['TSF'] = talib.TSF(df['Close'])
        df['VAR'] = talib.VAR(df['Close'])
        
        stock_data_dict[ticker] = df

    cursor.close()
    connection.close()
    return stock_data_dict



if __name__ == "__main__":
    main()
