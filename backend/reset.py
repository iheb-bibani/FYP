import sqlite3

def reset_tables():
    # Connect to the SQLite database
    connection =  sqlite3.connect("../databases/relational.db")
    cursor = connection.cursor()
    tickers = cursor.execute("SELECT ticker FROM equities").fetchall()
    
    # Delete all data from the portfolio_weights table
    for table_name in tickers:
        table_name = table_name[0]
        table_name = f"stock_{table_name[:3]}"
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    
    # Delete all data from the portfolio_weights table
    cursor.execute("DROP TABLE IF EXISTS portfolio_weights")
    print("Portfolio weights table has been reset.")

    # Delete all data from the optimal_weights table
    cursor.execute("DROP TABLE IF EXISTS optimal_weights")
    print("Optimal weights table has been reset.")
    
    cursor.execute("DROP TABLE IF EXISTS ticker_run")
    print("Ticker run table has been reset.")

    # Commit the changes and close the connection
    connection.commit()
    connection.close()

if __name__ == "__main__":
    reset_tables()
