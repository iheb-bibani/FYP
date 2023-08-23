import sqlite3
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

@st.cache_data
def get_date_ranges(_connection: sqlite3.Connection):
    cursor = _connection.cursor()
    cursor.execute("SELECT DISTINCT start_date, end_date FROM portfolio_weights")
    date_ranges = cursor.fetchall()
    cursor.close()
    return date_ranges

@st.cache_data
def get_portfolio_weights(_connection: sqlite3.Connection, start_date: str, end_date: str) -> list:
    cursor = _connection.cursor()
    cursor.execute(f"SELECT DISTINCT weights FROM portfolio_weights WHERE start_date='{start_date}' AND end_date='{end_date}'")
    rows = cursor.fetchall()
    cursor.close()
    portfolio_weights = [list(map(float, row[0].split(','))) for row in rows]
    return np.array(portfolio_weights)

@st.cache_data
def get_optimal_weights(_connection: sqlite3.Connection, start_date: str, end_date: str) -> list:
    cursor = _connection.cursor()
    cursor.execute(f"SELECT DISTINCT weights FROM optimal_weights WHERE start_date='{start_date}' AND end_date='{end_date}'")
    rows = cursor.fetchall()
    cursor.close()
    optimal_weights = [list(map(float, row[0].split(','))) for row in rows]
    return np.array(optimal_weights)

@st.cache_data
def get_log_returns(_connection: sqlite3.Connection, start_date: str, end_date: str) -> pd.DataFrame:
    cursor = _connection.cursor()
    query = f"SELECT * FROM log_returns WHERE date >= '{start_date}' AND date <= '{end_date}' ORDER BY date"
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()

    # Extract column names (tickers)
    cursor = _connection.cursor()
    cursor.execute("PRAGMA table_info(log_returns)")
    columns = [column[1] for column in cursor.fetchall()]
    cursor.close()

    # Create DataFrame
    log_returns_df = pd.DataFrame(rows, columns=columns)
    log_returns_df['date'] = pd.to_datetime(log_returns_df['date'])
    log_returns_df.set_index('date', inplace=True)

    return log_returns_df

def statistics(weights: np.ndarray, returns: pd.DataFrame, num_trading_days: int = 252) -> np.ndarray:
    portfolio_return = np.sum(returns.mean() * weights) * num_trading_days
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(returns.cov() * num_trading_days, weights))
    )
    return np.array(
        [
            portfolio_return,
            portfolio_volatility,
            portfolio_return / portfolio_volatility,
        ]
    )

def show_optimal_portfolio_streamlit(optimum: np.ndarray, returns: pd.DataFrame, portfolio_weights: list) -> None:
    portfolio_returns = []
    portfolio_volatilities = []
    sharpe_ratios = []
    
    for weights in portfolio_weights:
        stats = statistics(np.array(weights), returns)
        portfolio_returns.append(stats[0])
        portfolio_volatilities.append(stats[1])
        sharpe_ratios.append(stats[2])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_volatilities, y=portfolio_returns, mode='markers',
                             marker=dict(size=10, color=sharpe_ratios, colorscale='Viridis'),
                             name='Portfolios'))

    # Add red star for optimal portfolio
    optimal_stats = statistics(optimum, returns)
    fig.add_trace(go.Scatter(x=[optimal_stats[1]], y=[optimal_stats[0]], mode='markers',
                             marker=dict(size=20, color='red'),
                             name='Optimal Portfolio'))

    fig.update_layout(title='Portfolio Weights Scatter Plot',
                      xaxis_title='Expected Volatility',
                      yaxis_title='Expected Return',
                      showlegend=True)

    st.plotly_chart(fig)

def main():
    connection = sqlite3.connect("../backend/database.db")

    # Get available date ranges
    date_ranges = get_date_ranges(connection)
    date_options = [f"Start Date: {start} - End Date: {end}" for start, end in date_ranges]

    # Select date range
    length_date_options = len(date_options)
    selected_date_option = st.selectbox("Select Date Range:", date_options, index=0)
    selected_start_date, selected_end_date = selected_date_option.split(" - ")
    selected_start_date = selected_start_date.split(": ")[1]
    selected_end_date = selected_end_date.split(": ")[1]

    # Get portfolio weights for selected dates
    portfolio_weights = get_portfolio_weights(connection, selected_start_date, selected_end_date)

    # Get optimal weights for selected dates
    optimal_weights = get_optimal_weights(connection, selected_start_date, selected_end_date)
    st.subheader("Optimal Weights")

    # Scatter plot using portfolio weights
    log_returns = get_log_returns(connection, selected_start_date, selected_end_date)
    show_optimal_portfolio_streamlit(optimal_weights[0], log_returns, portfolio_weights)

    connection.close()

if __name__ == "__main__":
    main()
