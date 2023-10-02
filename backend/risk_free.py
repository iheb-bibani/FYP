import asyncio
from postgres import create_pool
import pandas as pd

async def upload_risk_free_rate(pool):
    async with pool.acquire() as connection:
        async with connection.transaction():
            
            await connection.execute("""
                CREATE TABLE IF NOT EXISTS risk_free_rate (
                    id SERIAL PRIMARY KEY,
                    date DATE,
                    rate FLOAT
                );
            """)
            
            csv_data = '../documents/Risk_Free.csv'
            risk_free_rate_df = pd.read_csv(csv_data)
            risk_free_rate_df['Date'] = pd.to_datetime(risk_free_rate_df['Date'], format='%d-%b-%y').dt.date
            
            for index, row in risk_free_rate_df.iterrows():
                date = row['Date']
                rate = row['Tenor (6-month)']
                await connection.execute(
                    "INSERT INTO risk_free_rate (date, rate) VALUES ($1, $2);",
                    date, rate
                )

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    pool = loop.run_until_complete(create_pool())
    loop.run_until_complete(upload_risk_free_rate(pool))
