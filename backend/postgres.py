from dotenv import load_dotenv
import os
import asyncio
import asyncpg
import ssl

load_dotenv()


async def create_pool():
    ssl_context = ssl.create_default_context(
        cafile=os.getenv("CA_CERT", "../ca-certificate.crt")
    )
    return await asyncpg.create_pool(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT")),
        user=os.getenv("DB_USERNAME"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        ssl=ssl_context,
        min_size=1,
        max_size=10,
    )
