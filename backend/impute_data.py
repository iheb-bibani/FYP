import psycopg2
import pandas as pd
from postgres import connection

def main():
    data = download_data(connection)
    for key, val in data.items():
        null_count = val.isnull().sum()
        for column, count in null_count.items():
            if count > 0 and column != "Volume":
                print(f"Ticker: {key}")
                print(f"Column: {column}, Null Count: {count}")
    connection.close()

def download_data(connection: psycopg2.extensions.connection) -> pd.DataFrame:
    cursor = connection.cursor()
    cursor.execute("SELECT ticker FROM equities")
    tickers = cursor.fetchall()
    stock_data = {}
    maximum = 0

    for stock_tuple in tickers:
        stock = stock_tuple[0]
        if stock == "^STI":
            continue
        table_name = f"stock_{stock[:3]}"  # Remove the .SI suffix
        query = (
            f"SELECT * FROM {table_name}"
        )
        cursor.execute(query)
        data = cursor.fetchall()
        if data:
            dates, open_price, high, low, close, adj_close, volume = zip(*data)
            df = pd.DataFrame({
                'Date': pd.to_datetime(dates),
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Adj_Close': adj_close,
                'Volume': volume
            })
            df.replace(to_replace=0, value=pd.NA, inplace=True)
            stock_data[stock] = df

    cursor.close()

    return stock_data

if __name__ == "__main__":
    main()