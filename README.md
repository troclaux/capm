# Tangency Portfolio Calculator

CLI tool that calculates the **tangency portfolio** (the portfolio with the maximum Sharpe ratio) for a set of stocks. Given a list of ticker symbols, it fetches historical prices from Yahoo Finance, estimates expected returns and the covariance matrix, and outputs the optimal portfolio weights.

## What is the tangency portfolio?

In Modern Portfolio Theory, every combination of risky assets defines a point on the **risk-return plane** (volatility on the x-axis, expected return on the y-axis). The set of portfolios that offer the best return for each level of risk forms a curve called the **efficient frontier**.

The **tangency portfolio** is the single portfolio on the efficient frontier that, when connected to the risk-free rate by a straight line, produces the steepest possible slope. That slope is the **Sharpe ratio** — the amount of excess return you earn per unit of risk. In other words, the tangency portfolio is the **best risk-adjusted portfolio** you can build from a given set of stocks.

### The math

For $N$ stocks with expected return vector $\mu$, covariance matrix $\Sigma$, and risk-free rate $r_f$:

1. **Risk premiums**: subtract the risk-free rate from each stock's expected return: $\mu - r_f$
2. **Invert the covariance matrix**: $\Sigma^{-1}$
3. **Raw weights**: multiply the inverse covariance matrix by the risk premium vector: $x = \Sigma^{-1}(\mu - r_f)$
4. **Normalize**: divide each weight by the sum so they add up to 1: $w_i = x_i / \sum x_i$

The resulting weights define the unique portfolio that maximizes the Sharpe ratio. To verify correctness, the ratio $(\mu_i - r_f) / \text{Cov}(r_i, r_p)$ should be **identical** for every asset — meaning no reallocation can improve the risk-return tradeoff.

## Installation

```bash
pip install -r requirements.txt
```

For development (running tests):

```bash
pip install -r requirements-dev.txt
```

## Usage

```bash
python tangency_portfolio.py TICKER [TICKER ...] [options]
```

### Passing tickers directly

```bash
python tangency_portfolio.py AAPL MSFT GOOG
```

### Brazilian stocks (B3)

Use the `.SA` suffix for B3 tickers:

```bash
python tangency_portfolio.py PETR4.SA VALE3.SA ITUB4.SA
```

### Reading tickers from a file

Create a `.txt` file with one ticker per line. Lines starting with `#` are comments and blank lines are ignored.

```
# tickers.txt
AAPL
MSFT
GOOG
AMZN
JPM
```

```bash
python tangency_portfolio.py --file tickers.txt
```

You can combine both — tickers from the file and the command line are merged:

```bash
python tangency_portfolio.py NVDA --file tickers.txt
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--file`, `-f` | | Path to a `.txt` file with one ticker per line |
| `--lookback` | `252` | Lookback period in calendar days (~1 year) |
| `--risk-free-rate` | `0.05` | Annual risk-free rate |
| `--verbose`, `-v` | off | Print intermediate values (prices, returns, covariance matrix) |

## Examples

```bash
# US tech stocks with default settings (252-day lookback, 5% risk-free rate)
python tangency_portfolio.py AAPL MSFT GOOG AMZN

# 2-year lookback with a custom risk-free rate
python tangency_portfolio.py SPY QQQ IWM --lookback 504 --risk-free-rate 0.045

# Brazilian stocks
python tangency_portfolio.py PETR4.SA VALE3.SA ITUB4.SA WEGE3.SA

# Read tickers from a file with verbose output
python tangency_portfolio.py -f my_portfolio.txt -v
```

## Output

The tool prints:

- **Asset weights**: the fraction of capital to allocate to each stock (negative means short)
- **Expected return**: the portfolio's annualized expected return
- **Volatility**: the portfolio's annualized standard deviation
- **Sharpe ratio**: (expected return - risk-free rate) / volatility
- **Verification**: confirms the tangency condition holds (ratio of risk premium to marginal covariance is equal across all assets)

Warnings are printed to stderr, including short position alerts, small sample warnings, and disclaimers about the Gaussian assumption.

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Argument error (missing tickers, file not found) |
| 2 | Data or computation error (invalid ticker, singular covariance) |

## Running tests

```bash
# Unit tests only (fast, no network)
pytest tests/test_calc.py -v

# All tests including integration (requires network)
pytest tests/ -v
```

## How it works

1. Fetch daily closing prices from Yahoo Finance for the requested lookback period
2. Compute daily simple returns from prices
3. Estimate the annualized mean return vector and covariance matrix (daily values x 252)
4. Compute the tangency portfolio weights: $w = \Sigma^{-1}(\mu - r_f) / \mathbf{1}^T \Sigma^{-1}(\mu - r_f)$
5. Verify the result: the ratio $(\mu_i - r_f) / \text{Cov}(r_i, r_p)$ must be identical for all assets
