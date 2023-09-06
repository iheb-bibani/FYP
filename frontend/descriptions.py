def main_description(length_portfolio_returns: int, length_tickers: int) -> str:
    return f"""
    :blue[{length_portfolio_returns} portfolios] were generated from a universe of :blue[{length_tickers} stocks] available on SGX.\n
    Below are the :green[optimal allocation and statistics] for the portfolios with the :green[highest Sharpe Ratios].\n
    The ***higher the Sharpe Ratio***, the ***better the risk-adjusted return*** of the portfolio.\n
    These portfolios are considered to be the ***most optimal*** portfolios in the universe of stocks, and are part of the ***efficient frontier***.\n
    """


risk_free_info = """
    Risk free rate is used to calculate the Sharpe\n
    The current risk free rate is the 3-month yield of Singapore Government Bonds.\n
    Source: http://www.worldgovernmentbonds.com/country/singapore/
    """

more_info_sharpe_ratio = """
    Sharpe Ratio is calculated using the formula: $$\\frac{{E[R_{{p}}] - R_{{f}}}}{{\\sigma_{{p}}}}$$\n
    where $$E[R_{{p}}]$$ is the expected return of the portfolio, $$R_{{f}}$$ is the risk-free rate, and $$\\sigma_{{p}}$$ is the standard deviation of the portfolio.
    """

more_info_portfolios = f"""
    Portfolios were generated using the following steps:\n
    1. Generate a random portfolio of weights for each stock in the universe.\n
    2. Calculate the expected return and volatility of the portfolio.\n
    Portfolios were generated and using the following parameters:\n
    - Calculated using 252 trading days
    - Risk-free rate: 0%\n
    """

scenario_analysis_info = """
            Scenario analysis is a technique used to analyze decisions through speculation of various possible outcomes in financial investments.\n
            In this case, we are using scenario analysis to determine the impact of a market crash on the portfolio.\n
            The market crash is simulated by looking at the returns of the :blue[STI] which is the :blue[benchmark] for the Singapore stock market.\n
            The returns of :green[each stock] in the portfolio is then historically calculated for the same period and then
            compared to the returns of the :blue[benchmark] to determine the :red[impact] of the market crash on the portfolio.
        """
