from postgres import connection


def reset_tables(connection):
    cursor = connection.cursor()

    # tickers = cursor.execute("SELECT ticker FROM equities").fetchall()

    # Delete all data from the portfolio_weights table
    # for table_name in tickers:
    #     table_name = table_name[0]
    #     table_name = f"stock_{table_name[:3]}"
    #     cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    # Delete all data from the portfolio_weights table
    cursor.execute("DELETE FROM portfolio_weights WHERE run_id > 4")
    print("Portfolio weights table has been reset.")
    connection.commit()
    # Delete all data from the optimal_weights table
    cursor.execute("DELETE FROM optimal_weights WHERE run_id > 4")
    print("Optimal weights table has been reset.")
    connection.commit()
    cursor.execute("DELETE FROM ticker_run WHERE id > 4")
    print("Ticker run table has been reset.")
    connection.commit()
    # Commit the changes and close the connection
    cursor.close()
    connection.close()


if __name__ == "__main__":
    reset_tables(connection)
