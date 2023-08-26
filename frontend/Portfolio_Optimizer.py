import psycopg2
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from typing import List
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.chart_container import chart_container
from postgres import connection

from database import (
    get_date_ranges,
    get_portfolio_weights,
    get_optimal_weights,
    get_ticker_data,
    get_run_id,
    get_ticker_names,
)
from descriptions import (
    more_info_portfolios,
    more_info_sharpe_ratio,
    main_description,
    risk_free_info,
)


@st.cache_data(ttl="30d")
def get_three_month_yield() -> float:
    yield_rates = pd.read_html(
        "http://www.worldgovernmentbonds.com/country/singapore/"
    )[1]
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

    for index, (portfolio_return, portfolio_volatility) in enumerate(
        zip(portfolio_returns, portfolio_volatilities)
    ):
        sharpe_ratio = (np.float64(portfolio_return)) / np.float64(portfolio_volatility)
        sharpe_ratios.append(sharpe_ratio)

        progress_container.progress(
            (index + 1) / total_weights,
            text=":hourglass_flowing_sand: Calculating portfolio statistics...",
        )
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
            marker=dict(size=20, color="green"),
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
    st.set_page_config(page_icon=":chart_with_upwards_trend:", page_title="Portfolio Optimizer", layout="centered")

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
                help=risk_free_info,
            )
            / 100
        )
        
        portfolio_value = st.number_input("Enter Portfolio Value ($)", value=10000.0, step=1000.0, format="%.2f", min_value=1000.0, max_value=1000000.0)

    # Get portfolio and optimal weights for selected dates
    (
        #portfolio_weights,
        portfolio_returns,
        portfolio_volatilities,
    ) = get_portfolio_weights(connection, run_id)
    optimal_weights, optimal_returns, optimal_volatilities = get_optimal_weights(
        connection, run_id
    )
    tickers = get_ticker_data(connection, run_id)

    st.subheader(":chart_with_upwards_trend: Optimal Portfolio Allocation")
    st.subheader(
        f"for {datetime.strptime(selected_start_date, '%Y-%m-%d').strftime('%d %B %Y')} to {datetime.strptime(selected_end_date, '%Y-%m-%d').strftime('%d %B %Y')}"
    )
    st.markdown(main_description(len(portfolio_returns), len(tickers)))
    with st.expander("Show More Information About Sharpe Ratio"):
        st.markdown(more_info_sharpe_ratio)
    with st.expander("Show More Information About How Portfolios Were Generated"):
        st.markdown(more_info_portfolios)
    st.markdown(
        "Portfolios contain :green[4 to 20 stocks] with a :green[minimum weight of 5%]."
    )
    st.subheader("Optimal Portfolio Allocation and Statistics")
    left_col, right_col = st.columns((2,1))
    with left_col:
        # Creating DataFrame to hold the ticker and weights
        ticker_weights_df = pd.DataFrame(
            {"Ticker": tickers, "Weight (%)": np.round(optimal_weights * 100, 1)}
        )

        # Remove rows with weights equal to zero
        ticker_weights_df = ticker_weights_df[ticker_weights_df["Weight (%)"] > 0]

        # Fetching the names corresponding to the tickers
        ticker_names = get_ticker_names(connection)
        ticker_weights_df["Name"] = ticker_weights_df["Ticker"].map(ticker_names)
        # Calculating the expected returns based on the weights
        ticker_weights_df["Expected Returns ($)"] = ticker_weights_df["Weight (%)"]/100 * portfolio_value * optimal_returns
        ticker_weights_df["Expected Returns ($)"] = ticker_weights_df["Expected Returns ($)"].map(lambda x: np.round(x, 2))
        expected_return = ticker_weights_df["Expected Returns ($)"].sum()
        
        # Appending the total weight
        total_weight_row = pd.DataFrame(
            {
                "Name": ["Total"],
                "Ticker": [""],
                "Weight (%)": [optimal_weights.sum() * 100],
                "Expected Returns ($)": [expected_return]
            },
            index=["Total"],
        )

        ticker_weights_df = pd.concat([ticker_weights_df, total_weight_row])

        # Displaying the DataFrame with Name as the index
        st.dataframe(ticker_weights_df.set_index("Name"), width=450, height=200)
    with right_col:
        st.write(f"Return: {np.round(optimal_returns*100,2)}%")
        st.write(f"Volatility: {np.round(optimal_volatilities*100, 2)}%")
        st.write(
            f"Sharpe Ratio: {np.round((optimal_returns-risk_free_rate)/optimal_volatilities, 2)}"
        )
    # Scatter plot using portfolio weights
    st.write(
        "Below is the scatter plot of the :blue[Risk Adjusted Returns] of each portfolio generated and the :green[Optimal Portfolio]."
    )

    show_optimal_portfolio_streamlit(
        portfolio_returns, portfolio_volatilities, optimal_returns, optimal_volatilities
    )
    
    st.subheader("Value At Risk (VaR)")
    st.markdown("VaR is a measure of the :red[losses] that a portfolio may experience over a given time period and confidence level.")
    #view_simulations = st.button("View Monte Carlo Simulations")
    # if view_simulations:
    #     switch_page("monte")
    tab1, tab2, tab3, tab4 = st.tabs(["VAR Summary", "Monte Carlo VAR", "Historic VAR", "Parametric VAR"])
    st.subheader("Scenario Analysis")
    st.markdown("Scenario analysis is a technique used to :red[analyze decisions] through :red[speculation] of various possible outcomes in financial investments.")
    tab1, tab2 = st.tabs(["Scenario Summary", "Affected Sectors"])
    connection.close()


if __name__ == "__main__":
    main()
