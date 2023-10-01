from dotenv import load_dotenv
import os
import asyncio
from postgres import create_pool

load_dotenv()


async def reset_tables(pool):
    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.execute("DROP TABLE IF EXISTS portfolio_weights")
            print("Portfolio weights table has been reset.")

            await connection.execute("DROP TABLE IF EXISTS optimal_weights")
            print("Optimal weights table has been reset.")

            await connection.execute("DROP TABLE IF EXISTS efficient_frontier")
            print("Efficient frontier table has been reset.")

            await connection.execute("DROP TABLE IF EXISTS ticker_run")
            print("Ticker run table has been reset.")


async def main():
    pool = await create_pool()
    await reset_tables(pool)
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
