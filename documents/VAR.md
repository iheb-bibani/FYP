# Value at Risk (VaR)

- **Introduction**: VaR is a statistical measure that quantifies the level of financial risk within a portfolio over a specific time frame. It provides an estimate of the potential loss that could occur in an investment portfolio due to adverse market movements.

## Key Concepts

- **Confidence Level**: The probability that losses will not exceed the VaR estimate.
- **Time Horizon**: The period over which risk is assessed, often daily, monthly, or yearly.
- **Loss Amount**: The amount of money (or percentage of the portfolio) that could be lost.

## Variance-Covariance Method (Parametric VaR)

This method assumes that asset returns are normally distributed. It is also known as the analytical method.

### Steps:

1. **Calculate Portfolio Mean and Volatility**: Use historical data to calculate the mean and standard deviation of the portfolio's returns.

2. **Normal Distribution**: Assume that the returns are normally distributed, which allows us to use Z-scores.

3. **Calculate VaR**: Use the formula:
    $$
    VaR = Z \times \text{Portfolio Volatility} \times \text{Portfolio Value}
    $$
    where \( Z \) is the Z-score corresponding to the desired confidence level.

4. **Probability Integral**: The integral of the normal distribution function yields the probability of the event:
    $$
    P(R \leq -VaR) = \int_{-\infty}^{-VaR} f(x) dx
    $$

5. **Solve for VaR**: Set the probability equal to the confidence level \( \alpha \) to solve for \( -VaR \):
    $$
    P(R \leq -VaR) = \alpha
    $$

## Monte Carlo Simulation Method

This approach does not assume normally distributed returns. Instead, it simulates future asset prices based on stochastic (random) processes, often using Geometric Brownian Motion.

$$
S(t) = S(0) \times \exp \left( (r - 0.5 \times \sigma^2) \times t + \sigma \times \sqrt{t} \times Z \right)
$$

Where:

- \( S(t) \) is the stock price at time \( t \)
- \( S(0) \) is the initial stock price
- \( r \) is the mean return
- \( \sigma \) is the volatility
- \( t \) is the time in days
- \( Z \) is a random number from a standard normal distribution