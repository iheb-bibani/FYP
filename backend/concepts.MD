# Statistical Concepts

- **Expected Value**:
  $$\text{Expected value} = \sum (\text{all possible values} \times \text{probability of each value})$$
- **Variance**:
  $$\text{Variance} = \sum (\text{all possible values} \times \text{probability of each value} \times (\text{value} - \text{expected value})^2)$$
- **Covariance**:
  $$\text{Covariance} = \sum (\text{all possible values} \times \text{probability of each value} \times (\text{value} - \text{expected value of x}) \times (\text{value} - \text{expected value of y}))$$
- **Correlation**:
  $$\text{Correlation} = \frac{\text{Covariance}}{(\text{standard deviation of x} \times \text{standard deviation of y})}$$
- **Stocks and Diversification**:
  Stocks which are highly correlated are not good for diversification.

## Assumptions of Markowitz Model

- **Returns** are normally distributed with a mean and variance.
- **Investors** are risk averse.
- **Long positions** only (no short selling).

### Goal

1. Maximize return for a given level of risk.
2. Minimize risk for a given level of return.

### Parameters

1. Weights of each asset.
2. Return of each asset based on historical data.
3. Expected return of each asset.

## Sharpe Ratio

- **Sharpe Ratio**:
  $$\text{Sharpe Ratio} = \frac{\text{Expected Return} - \text{Risk Free Rate}}{\text{Standard Deviation}}$$
  The higher the Sharpe Ratio, the better the investment.
