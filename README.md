# Tangency Portfolio Calculator

CLI tool that calculates the **tangency portfolio** (the portfolio with the maximum Sharpe ratio) for a set of stocks using the **Capital Asset Pricing Model (CAPM)**. Given a list of ticker symbols, it fetches historical prices from Yahoo Finance, estimates expected returns and the covariance matrix, and outputs optimal portfolio weights, betas, and Capital Market Line allocations.

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

### Capital Market Line (CML)

The **Capital Market Line** is the straight line from the risk-free rate through the tangency portfolio:

$$E[r] = r_f + \text{Sharpe} \times \sigma$$

Every point on this line represents an optimal combination of the risk-free asset and the tangency portfolio. Your position on the line depends on your **risk aversion** ($A$):

- **Conservative** ($A \geq 5$): mostly risk-free asset, small allocation to the tangency portfolio
- **Moderate** ($A \approx 2$): balanced mix
- **Aggressive** ($A \leq 1$): leverage — borrow at the risk-free rate to invest more than 100% in the tangency portfolio

The optimal fraction in the tangency portfolio is: $w_T = (E[r_T] - r_f) / (A \cdot \sigma_T^2)$

### Betas

Each asset's **beta** measures its sensitivity to the portfolio:

$$\beta_i = \frac{\text{Cov}(r_i, r_p)}{\text{Var}(r_p)}$$

A beta greater than 1 means the asset amplifies portfolio moves; less than 1 means it dampens them. If a market proxy is provided (e.g., SPY), the tool also reports each asset's market beta.

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
| `--no-short` | off | Forbid short positions (constrain all weights >= 0) |
| `--market-proxy` | | Market benchmark ticker for beta comparison (e.g. `SPY`, `^BVSP`) |
| `--risk-aversion` | | Risk aversion parameter A for CML allocation (omit to see A=1, 2, 5) |
| `--verbose`, `-v` | off | Print intermediate values (prices, returns, covariance matrix) |

## Examples

```bash
# US tech stocks with default settings
python tangency_portfolio.py AAPL MSFT GOOG AMZN

# Forbid short selling
python tangency_portfolio.py AAPL MSFT GOOG --no-short

# Compare against S&P 500 (shows market betas)
python tangency_portfolio.py AAPL MSFT GOOG --market-proxy SPY

# Brazilian stocks with Ibovespa as benchmark
python tangency_portfolio.py PETR4.SA VALE3.SA ITUB4.SA --market-proxy ^BVSP

# Specify your risk aversion level
python tangency_portfolio.py AAPL MSFT GOOG --risk-aversion 3

# 2-year lookback with custom risk-free rate and no short selling
python tangency_portfolio.py SPY QQQ IWM --lookback 504 --risk-free-rate 0.045 --no-short

# Read tickers from a file with verbose output
python tangency_portfolio.py -f my_portfolio.txt -v
```

## Output

The tool prints four sections:

### 1. Tangency Portfolio
- **Asset weights**: the fraction of capital to allocate to each stock (negative means short, unless `--no-short` is used)
- **Expected return**: the portfolio's annualized expected return
- **Volatility**: the portfolio's annualized standard deviation
- **Sharpe ratio**: (expected return - risk-free rate) / volatility
- **Verification**: confirms the tangency condition holds (only for unconstrained optimization)

### 2. Asset Betas
- **Portfolio beta**: each asset's beta relative to the tangency portfolio
- **Market beta**: each asset's beta relative to the market proxy (if `--market-proxy` is provided)

### 3. Capital Market Line (CML)
- The CML equation: `E[r] = rf + sharpe * sigma`

### 4. CML Allocation (Risk Aversion Tuning)
- Shows the optimal split between the risk-free asset and the tangency portfolio for different risk aversion levels
- If `--risk-aversion` is specified, shows that specific level; otherwise shows A=1 (aggressive), A=2 (moderate), A=5 (conservative)

Warnings are printed to stderr, including short position alerts, small sample warnings, and CAPM disclaimers.

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
4. Compute the tangency portfolio weights: $w = \Sigma^{-1}(\mu - r_f) / \mathbf{1}^T \Sigma^{-1}(\mu - r_f)$, or use scipy constrained optimization if `--no-short`
5. Verify the result: the ratio $(\mu_i - r_f) / \text{Cov}(r_i, r_p)$ must be identical for all assets
6. Compute per-asset betas (and market betas if a proxy is provided)
7. Compute the Capital Market Line and risk-aversion-based allocations

## CAPM Limitations

The CAPM makes strong assumptions that don't perfectly hold in real markets:

- **Gaussian returns**: real returns have fat tails and skewness
- **Stationary distributions**: past correlations and means may not persist (regime shifts)
- **Size effect**: small-cap stocks tend to earn higher returns than CAPM predicts (Fama-French)
- **Value effect**: high book-to-market stocks tend to outperform (Fama-French)
- **Momentum**: recent winners tend to keep winning in the short term
- **Thin trading**: illiquid or thinly traded stocks produce unreliable covariance estimates
