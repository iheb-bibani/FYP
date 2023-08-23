import sqlite3

def reset_tables():
    # Connect to the SQLite database
    connection =  sqlite3.connect("../databases/relational.db")
    cursor = connection.cursor()

    # Delete all data from the log_returns table
    cursor.execute("DELETE FROM log_returns")
    print("Log returns table has been reset.")

    # Delete all data from the portfolio_weights table
    cursor.execute("DELETE FROM portfolio_weights")
    print("Portfolio weights table has been reset.")

    # Delete all data from the optimal_weights table
    cursor.execute("DELETE FROM optimal_weights")
    print("Optimal weights table has been reset.")

    # Commit the changes and close the connection
    connection.commit()
    connection.close()

if __name__ == "__main__":
    reset_tables()
