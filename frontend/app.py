import sqlite3
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from typing import List
from database import get_date_ranges, get_portfolio_weights, get_optimal_weights, get_ticker_data, get_run_id, get_ticker_names

@st.cache_data
def get_three_month_yield() -> float:
    yield_rates = pd.read_html("http://www.worldgovernmentbonds.com/country/singapore/")[1]
    three_mnth_yield = float(yield_rates.iloc[5, 2].replace("%", ""))
    return three_mnth_yield

@st.cache_resource
def show_optimal_portfolio_streamlit(
    portfolio_returns: list = None,
    portfolio_volatilities: list = None,
    optimal_returns: list = None,
    optimal_volatilities: list = None,
) -> None:
    sharpe_ratios = []
    progress_container = st.empty()
    total_weights = len(portfolio_returns)

    for index, (portfolio_return, portfolio_volatility) in enumerate(zip(portfolio_returns, portfolio_volatilities)):
        sharpe_ratio = (np.float64(portfolio_return)) / np.float64(portfolio_volatility)
        sharpe_ratios.append(sharpe_ratio)

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

    fig.add_trace(
        go.Scatter(
            x=optimal_volatilities,
            y=optimal_returns,
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
        run_id = get_run_id(connection, selected_start_date, selected_end_date)

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
                    Risk free rate is used to calculate the Sharpe\n
                    The current risk free rate is the 3-month yield of Singapore Government Bonds.\n
                    Source: http://www.worldgovernmentbonds.com/country/singapore/
                    """,
            )/ 100
        )

    # Get portfolio and optimal weights for selected dates
    portfolio_weights, portfolio_returns, portfolio_volatilities = get_portfolio_weights(connection, run_id)
    optimal_weights, optimal_returns, optimal_volatilities = get_optimal_weights(connection, run_id)
    tickers = get_ticker_data(connection, run_id)
    
    st.subheader(":chart_with_upwards_trend: Optimal Portfolio Allocation")
    st.subheader(
        f"for {datetime.strptime(selected_start_date, '%Y-%m-%d').strftime('%d %B %Y')} to {datetime.strptime(selected_end_date, '%Y-%m-%d').strftime('%d %B %Y')}"
    )
    st.markdown(
        f"""
    :blue[{len(portfolio_weights)} portfolios] were generated from a universe of :blue[{len(tickers)} stocks] available on SGX.\n
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
    - Calculated using 252 trading days
    - Risk-free rate: 0%\n
    """
        )
    st.markdown("Portfolios contain :green[4 to 20 stocks] with a :green[minimum weight of 5%].")
    st.subheader("Optimal Portfolio Allocation and Statistics")
    left_col, right_col = st.columns(2)
    with left_col:

        # Creating DataFrame to hold the ticker and weights
        ticker_weights_df = pd.DataFrame({
            "Ticker": tickers,
            "Weight (%)": np.round(optimal_weights * 100, 1)
        })


        # Remove rows with weights equal to zero
        ticker_weights_df = ticker_weights_df[ticker_weights_df['Weight (%)'] > 0]

        # Fetching the names corresponding to the tickers
        ticker_names = get_ticker_names(connection)
        ticker_weights_df['Name'] = ticker_weights_df['Ticker'].map(ticker_names)

        # Appending the total weight
        total_weight_row = pd.DataFrame({
            "Name": ["Total"],
            "Ticker": [""],
            "Weight (%)": [optimal_weights.sum() * 100]
        }, index=['Total'])

        ticker_weights_df = pd.concat([ticker_weights_df, total_weight_row])

        # Displaying the DataFrame with Name as the index
        st.dataframe(ticker_weights_df.set_index("Name"), width=400, height=200)
    with right_col:
        st.write(f"Return: {np.round(optimal_returns*100,2)}%")
        st.write(f"Volatility: {np.round(optimal_volatilities*100, 2)}%")
        st.write(f"Sharpe Ratio: {np.round((optimal_returns-risk_free_rate)/optimal_volatilities, 2)}")
    # Scatter plot using portfolio weights
    st.write(
        "Below is the scatter plot of the :blue[Risk Adjusted Returns] of each portfolio generated and the :red[Optimal Portfolio]."
    )
    
    show_optimal_portfolio_streamlit(portfolio_returns, portfolio_volatilities, optimal_returns, optimal_volatilities)
    connection.close()


if __name__ == "__main__":
    main()
