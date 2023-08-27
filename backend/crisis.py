import psycopg2
from postgres import connection

def main():
    store_financial_crisis("European Debt Crisis", "2011-07-01", "2012-06-30", connection)

    store_financial_crisis("Chinese Stock Market Crash", "2015-06-12", "2015-08-31", connection)

    store_financial_crisis("US-China Trade War Impact", "2018-06-15", "2019-12-13", connection)

    store_financial_crisis("2019 Singapore Economic Slowdown", "2019-01-01", "2019-12-31", connection)

    store_financial_crisis("COVID-19 Impact", "2020-02-20", "2020-03-23", connection)

    connection.close()    
        
def store_financial_crisis(
    crisis_name: str, start_date: str, end_date: str, connection: psycopg2.extensions.connection
) -> None:
    cursor = connection.cursor()
    
    # Create the table if it doesn't exist
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS financial_crisis (id SERIAL PRIMARY KEY, crisis_name TEXT, start_date DATE, end_date DATE)"
    )
    
    # Insert the crisis details
    cursor.execute(
        "INSERT INTO financial_crisis (crisis_name, start_date, end_date) VALUES (%s, %s, %s) ON CONFLICT (crisis_name) DO NOTHING",
        (crisis_name, start_date, end_date)
    )
    
    connection.commit()
    cursor.close()
    print(f"Stored details of {crisis_name}.")
    
if __name__ == "__main__":
    main()