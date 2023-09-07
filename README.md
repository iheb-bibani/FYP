# Optimal Portfolio Allocation

## Overview

This repository contains the codebase for the Optimal Portfolio Allocation project, which aims to provide an efficient and automated way to allocate assets in a financial portfolio using machine learning and mathematical optimization techniques. The project is divided into four main components:

- **Backend**: Handles data collection, preprocessing, and optimization algorithms.
- **Frontend**: Provides a user interface for interacting with the system.
- **Data Science**: Contains notebooks and scripts for data analysis and machine learning models.
- **Document**: Includes all the documentation related to the project.

## Live Demo

You can access the live demo of the project [here](https://vrsbfyp.me/).

## Components

### Backend

- `crisis.py`: Simulating a financial crisis.
- `demo.py`: Testing and plotting results from `optimize.py`
- `macro_data.py`: Handling macroeconomic data from IMF.
- `obtain_data.py`: To download and store technical data from Yahoo Finance.
- `obtain_tickers.py`: To download and store ticker symbols from [TopForeignStocks.com](https://topforeignstocks.com/listed-companies-lists/the-complete-list-of-listed-companies-in-singapore/)
- `optimize.py`: Optimal portfolio allocation and Efficient Frontier.
- `postgres.py`: PostgreSQL database connection.
- `reset.py`: Resetting the database.

### Frontend

- `database.py`: Database functions for the frontend.
- `description.py`: Description for frontend.
- `Portfolio_Optimizer.py`: Streamlit app, Frontend.
- `postgre.py`: PostgreSQL database connection.
- `functions/backtest.py`: Backtesting for Scenario Analysis.
- `functions/VAR.py`: Value at Risk.
