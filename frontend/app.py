import sqlite3
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime


@st.cache_data
def get_date_ranges(_connection: sqlite3.Connection) -> list:
    cursor = _connection.cursor()
    cursor.execute("SELECT DISTINCT start_date, end_date FROM portfolio_weights")
    date_ranges = cursor.fetchall()
    cursor.close()
    return date_ranges

@st.cache_data
def get_ticker_names(_connection: sqlite3.Connection) -> dict:
    cursor = _connection.cursor()
    cursor.execute("SELECT DISTINCT name, ticker FROM equities")
    ticker_names = {ticker[:3]: name for name, ticker in cursor.fetchall()} # Remove the .SI suffix
    cursor.close()
    return ticker_names
    

@st.cache_data
def get_three_month_yield() -> float:
    yield_rates = pd.read_html("http://www.worldgovernmentbonds.com/country/singapore/")[1]
    three_mnth_yield = float(yield_rates.iloc[5, 2].replace("%", ""))
    return three_mnth_yield

@st.cache_data
def get_portfolio_weights(
    _connection: sqlite3.Connection, start_date: str, end_date: str
) -> list:
    cursor = _connection.cursor()
    cursor.execute(
        f"SELECT DISTINCT weights FROM portfolio_weights WHERE start_date='{start_date}' AND end_date='{end_date}'"
    )
    rows = cursor.fetchall()
    cursor.close()
    portfolio_weights = [list(map(float, row[0].split(","))) for row in rows]
    return np.array(portfolio_weights)


@st.cache_data
def get_optimal_weights(
    _connection: sqlite3.Connection, start_date: str, end_date: str
) -> list:
    cursor = _connection.cursor()
    cursor.execute(
        f"SELECT DISTINCT weights FROM optimal_weights WHERE start_date='{start_date}' AND end_date='{end_date}'"
    )
    rows = cursor.fetchall()
    cursor.close()
    optimal_weights = [list(map(float, row[0].split(","))) for row in rows]
    return np.array(optimal_weights)


@st.cache_data
def get_log_returns(
    _connection: sqlite3.Connection, start_date: str, end_date: str
) -> pd.DataFrame:
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
    log_returns_df["date"] = pd.to_datetime(log_returns_df["date"])
    log_returns_df.set_index("date", inplace=True)

    return log_returns_df


@st.cache_resource
def statistics(
    weights: np.ndarray,
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0,
    num_trading_days: int = 252,
) -> np.ndarray:
    portfolio_return = np.sum(mean_returns * weights) * num_trading_days
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix * num_trading_days, weights))
    )
    if np.isnan(portfolio_volatility):
        sharpe_ratio = 0
    else:
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return np.array([portfolio_return, portfolio_volatility, sharpe_ratio])

@st.cache_resource
def show_optimal_portfolio_streamlit(
    optimum: np.ndarray,
    returns: pd.DataFrame,
    portfolio_weights: list,
    risk_free_rate: float = 0,
) -> None:
    portfolio_returns = []
    portfolio_volatilities = []
    sharpe_ratios = []
    
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    progress_container = st.empty()
    total_weights = len(portfolio_weights)
    
    for index, weights in enumerate(portfolio_weights):
        stats = statistics(np.array(weights), mean_returns, cov_matrix, risk_free_rate)
        portfolio_returns.append(stats[0])
        portfolio_volatilities.append(stats[1])
        sharpe_ratios.append(stats[2])
        
        progress_container.progress((index + 1) / total_weights, text=":hourglass_flowing_sand: Calculating portfolio statistics...")
    progress_container.empty()
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=portfolio_volatilities,
            y=portfolio_returns,
            mode="markers",
            marker=dict(size=10, color=sharpe_ratios, colorscale="Viridis"),
            name="Portfolios",
        )
    )

    # Add red star for optimal portfolio
    optimal_stats = statistics(optimum, mean_returns, cov_matrix, risk_free_rate)
    fig.add_trace(
        go.Scatter(
            x=[optimal_stats[1]],
            y=[optimal_stats[0]],
            mode="markers",
            marker=dict(size=20, color="red"),
            name="Optimal Portfolio",
        )
    )

    fig.update_layout(
        title="Portfolio Weights Scatter Plot",
        xaxis_title="Expected Volatility",
        yaxis_title="Expected Return",
        showlegend=True,
    )

    st.plotly_chart(fig)


def main():
    connection = sqlite3.connect("../databases/relational.db")

    # Get available date ranges
    date_ranges = get_date_ranges(connection)
    date_options = [f"From {start} to {end}" for start, end in date_ranges]

    # Select date range
    with st.sidebar:
        st.header("Parameters")
        length_date_options = len(date_options)
        selected_date_option = st.selectbox(
            "Select Date Range:",
            date_options,
            index=length_date_options - 1,
            help="Select the date range to generate the optimal portfolio.",
        )
        selected_start_date, selected_end_date = selected_date_option.split(" to ")
        selected_start_date = selected_start_date.split("From ")[1].strip()
        selected_end_date = selected_end_date.strip()
        
        three_mnth_yield = get_three_month_yield()
        risk_free_rate = (
            st.number_input(
                "Risk Free Rate (%)",
                value=three_mnth_yield,
                step=0.1,
                format="%.3f",
                min_value=0.0,
                max_value=10.0,
                help="""
                    Risk free rate is used to calculate the Sharpe Ratio.\n
                    The current risk free rate is the 3-month yield of Singapore Government Bonds.\n
                    Source: http://www.worldgovernmentbonds.com/country/singapore/
                    """,
            )/ 100
        )

    # Get portfolio weights for selected dates
    portfolio_weights = get_portfolio_weights(
        connection, selected_start_date, selected_end_date
    )
    log_returns = get_log_returns(connection, selected_start_date, selected_end_date)

    st.subheader(":chart_with_upwards_trend: Optimal Portfolio Allocation")
    st.subheader(
        f"for {datetime.strptime(selected_start_date, '%Y-%m-%d').strftime('%d %B %Y')} to {datetime.strptime(selected_end_date, '%Y-%m-%d').strftime('%d %B %Y')}"
    )
    st.markdown(
        f"""
    :blue[{len(portfolio_weights)} portfolios] were generated from a universe of :blue[{len(log_returns.columns)} stocks] available on SGX.\n
    Below are the :green[optimal allocation and statistics] for the portfolio with the :green[highest Sharpe Ratio].\n
    The ***higher the Sharpe Ratio***, the ***better the risk-adjusted return*** of the portfolio.\n
    """
    )
    with st.expander("Show More Information About Sharpe Ratio"):
        st.markdown(
            """
    Sharpe Ratio is calculated using the formula: $$\\frac{{E[R_{{p}}] - R_{{f}}}}{{\\sigma_{{p}}}}$$\n
    where $$E[R_{{p}}]$$ is the expected return of the portfolio, $$R_{{f}}$$ is the risk-free rate, and $$\\sigma_{{p}}$$ is the standard deviation of the portfolio.
    """
        )
    with st.expander("Show More Information About How Portfolios Were Generated"):
        st.markdown(
            f"""
    Portfolios were generated using the following steps:\n
    1. Generate a random portfolio of weights for each stock in the universe.\n
    2. Calculate the expected return and volatility of the portfolio.\n
    Portfolios were generated and using the following parameters:\n
    - Calculated using {len(log_returns)} trading days
    - Risk-free rate: 0%\n
    """
        )
    st.markdown("Portfolios contain :green[4 to 20 stocks] with a :green[minimum weight of 5%].")
    # Get optimal weights for selected dates
    optimal_weights = get_optimal_weights(
        connection, selected_start_date, selected_end_date
    )

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("Optimal Portfolio Weights")
        tickers = [ticker[1:] for ticker in log_returns.columns]

        # Creating DataFrame to hold the ticker and weights
        ticker_weights_df = pd.DataFrame({
            "Ticker": tickers,
            "Weight (%)": np.round(optimal_weights[0] * 100, 1)
        })

        # Remove rows with weights equal to zero
        ticker_weights_df = ticker_weights_df[ticker_weights_df['Weight (%)'] > 0]

        # Fetching the names corresponding to the tickers
        names_tickers = get_ticker_names(connection)
        ticker_weights_df['Name'] = ticker_weights_df['Ticker'].map(names_tickers)

        # Appending the total weight
        total_weight_row = pd.DataFrame({
            "Name": ["Total"],
            "Ticker": [""],
            "Weight (%)": [optimal_weights[0].sum() * 100]
        }, index=['Total'])

        ticker_weights_df = pd.concat([ticker_weights_df, total_weight_row])

        # Displaying the DataFrame with Name as the index
        st.dataframe(ticker_weights_df.set_index("Name"), width=400, height=200)
    with right_col:
        st.subheader("Portfolio Statistics")
        mean_returns = log_returns.mean()
        cov_matrix = log_returns.cov()
        stats = statistics(optimal_weights[0], mean_returns, cov_matrix, risk_free_rate)
        st.write(f"Return: {np.round(stats[0]*100, 2)}%")
        st.write(f"Volatility: {np.round(stats[1]*100, 2)}%")
        st.write(f"Sharpe Ratio: {np.round(stats[2], 2)}")
    # Scatter plot using portfolio weights
    st.write(
        "Below is a scatter plot of the :blue[Risk Adjusted Return] of each portfolio generated and the :red[Optimal Portfolio]."
    )
    show_optimal_portfolio_streamlit(
        optimal_weights[0], log_returns, portfolio_weights, risk_free_rate
    )

    connection.close()


if __name__ == "__main__":
    main()
