import streamlit as st
import numpy as np
from typing import List
import psycopg2


def get_date_ranges(_connection: psycopg2.extensions.connection) -> list:
    cursor = _connection.cursor()
    cursor.execute("SELECT DISTINCT start_date, end_date FROM ticker_run")
    date_ranges = cursor.fetchall()
    cursor.close()
    return date_ranges


@st.cache_data(ttl="30d")
def get_portfolio_weights(
    _connection: psycopg2.extensions.connection, run_id: int
) -> list:
    cursor = _connection.cursor()
    cursor.execute(
        "SELECT returns, volatility FROM portfolio_weights WHERE run_id=%s", (run_id,)
    )
    rows = cursor.fetchall()
    cursor.close()
    portfolio_returns = [float(row[0]) for row in rows]
    portfolio_volatilities = [float(row[1]) for row in rows]
    return np.array(portfolio_returns), np.array(portfolio_volatilities)


@st.cache_data(ttl="30d")
def get_optimal_weights(
    _connection: psycopg2.extensions.connection, run_id: int
) -> list:
    cursor = _connection.cursor()
    cursor.execute(
        "SELECT returns, volatility FROM optimal_weights WHERE run_id=%s",
        (run_id,),
    )
    rows = cursor.fetchall()
    cursor.close()
    #optimal_weights = [list(map(float, row[0].split(","))) for row in rows][0]
    optimal_returns = [float(row[0]) for row in rows][0]
    optimal_volatilities = [float(row[1]) for row in rows][0]
    return (
       # np.array(optimal_weights),
        np.array(optimal_returns),
        np.array(optimal_volatilities),
    )


@st.cache_data(ttl="30d")
def get_efficient_frontier(
    _connection: psycopg2.extensions.connection, run_id: int
) -> list:
    cursor = _connection.cursor()
    cursor.execute(
        "SELECT weight, returns, volatility FROM efficient_frontier WHERE run_id=%s",
        (run_id,),
    )
    rows = cursor.fetchall()
    cursor.close()
    efficient_weights = [list(map(float, row[0].split(","))) for row in rows]
    portfolio_returns = [float(row[1]) for row in rows]
    portfolio_volatilities = [float(row[2]) for row in rows]
    return (
        np.array(efficient_weights),
        np.array(portfolio_returns),
        np.array(portfolio_volatilities),
    )


@st.cache_data(ttl="30d")
def get_ticker_data(
    _connection: psycopg2.extensions.connection, run_id: int
) -> List[str]:
    cursor = _connection.cursor()
    cursor.execute("SELECT tickers FROM ticker_run WHERE id=%s", (run_id,))
    row = cursor.fetchone()
    cursor.close()
    tickers = row[0].split(",") if row else []
    return tickers


@st.cache_data(ttl="30d")
def get_run_id(
    _connection: psycopg2.extensions.connection, start_date: str, end_date: str
) -> int:
    cursor = _connection.cursor()
    cursor.execute(
        "SELECT id FROM ticker_run WHERE start_date=%s AND end_date=%s ORDER BY id DESC LIMIT 1",
        (start_date, end_date),
    )
    run_id = cursor.fetchone()[0]
    cursor.close()
    return run_id


@st.cache_data(ttl="30d")
def get_ticker_names(_connection: psycopg2.extensions.connection) -> dict:
    cursor = _connection.cursor()
    cursor.execute("SELECT ticker, name FROM equities")
    ticker_names = {row[0]: row[1] for row in cursor.fetchall()}
    cursor.close()
    return ticker_names
