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
| `--risk-free-rate` | `0.05` | Annual risk-free rate as a decimal |
| `--rf-proxy` | | Fetch risk-free rate from a yield ticker (e.g. `^IRX` for 13-week T-bill, `^TNX` for 10-year Treasury) |
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

# Use the 13-week T-bill yield as the risk-free rate (auto-fetched)
python tangency_portfolio.py AAPL MSFT GOOG --rf-proxy ^IRX

# Use the 10-year Treasury yield for longer-horizon analysis
python tangency_portfolio.py AAPL MSFT GOOG --rf-proxy ^TNX

# 2-year lookback with custom risk-free rate and no short selling
python tangency_portfolio.py SPY QQQ IWM --lookback 504 --risk-free-rate 0.045 --no-short

# Read tickers from a file with verbose output
python tangency_portfolio.py -f my_portfolio.txt -v
```

## Understanding the output

The tool prints four sections. Here's what each one means for your actual investment decisions, using a concrete example with $10,000 to invest.

Suppose you run:

```bash
python tangency_portfolio.py AAPL MSFT GOOG AMZN JPM --no-short --market-proxy SPY --risk-aversion 15
```

### Section 1: Tangency Portfolio

```
Asset          Weight
----------------------
AAPL          25.34%
MSFT           0.00%
GOOG          65.15%
AMZN           0.00%
JPM            9.52%

Expected Return:  68.74%
Volatility:       20.43%
Sharpe Ratio:     3.1193
```

This is the core answer: **how to split your money among the risky stocks**. If you invest $10,000 purely in stocks using these weights:

- **AAPL**: $2,534 (25.34%)
- **MSFT**: $0 (0.00%) — the optimizer says it doesn't improve risk-adjusted returns
- **GOOG**: $6,515 (65.15%)
- **AMZN**: $0 (0.00%)
- **JPM**: $952 (9.52%)

The statistics describe this specific mix:

- **Expected Return (68.74%)**: based on the past year's data, this mix would have returned ~68.74% annualized. This is **not a forecast** — it's what historical data suggests.
- **Volatility (20.43%)**: the portfolio's annualized standard deviation. Roughly, in any given year, returns could swing ~20% above or below the expected value.
- **Sharpe Ratio (3.12)**: for every 1% of risk you take, you earn ~3.12% of excess return above the risk-free rate. Higher is better. No other combination of these 5 stocks produces a higher Sharpe ratio.

### Section 2: Asset Betas

```
Asset       Portfolio Beta      Market Beta (SPY)
-------------------------------------------------
AAPL                0.6220                 0.9632
MSFT                0.1024                 0.8916
GOOG                1.2436                 1.1752
AMZN                0.5651                 1.4966
JPM                 0.3388                 1.0805
```

Beta measures how sensitive each stock is to portfolio or market movements:

- **Portfolio Beta**: GOOG at 1.24 means if your portfolio goes up 10%, GOOG tends to go up 12.4%. JPM at 0.34 means it barely moves with the portfolio — it acts as a diversifier.
- **Market Beta (SPY)**: AMZN at 1.50 means it's 50% more volatile than the S&P 500. JPM at 1.08 roughly tracks the market.

This helps you understand **why** the optimizer chose certain weights — it favors stocks that contribute return without adding too much correlated risk.

### Section 3: Capital Market Line (CML)

```
E[r] = 5.00% + 3.1193 * sigma
```

This is the equation for the best possible risk-return tradeoff. It means: for every 1% of volatility you accept, you should earn 3.12% above the risk-free rate. This line connects the risk-free asset (0% volatility, 5% return) to the tangency portfolio (20.43% volatility, 68.74% return).

### Section 4: CML Allocation (Risk Aversion Tuning)

```
    A    Tangency   Risk-Free      E[r]  Volatility
   15.0     101.8%      -1.8%   69.87%     20.80%
```

This is the final step: **how much of your total wealth goes into the tangency portfolio vs. a risk-free asset** (like T-bills).

- The tangency weights from Section 1 tell you how to split **within** risky stocks.
- This section tells you **how much to put in risky stocks at all**.

With risk aversion A=15, the model says ~100% tangency portfolio. For $10,000 that means:

1. CML says **101.8% tangency**, so ~$10,180 in stocks
2. Within that $10,180, split by Section 1 weights:
   - AAPL: $10,180 x 25.34% = **$2,580**
   - GOOG: $10,180 x 65.15% = **$6,632**
   - JPM: $10,180 x 9.52% = **$969**

If you use a higher risk aversion (e.g., A=50), more money goes to T-bills instead. If you use a lower A (e.g., A=1), the model suggests leveraging — borrowing to invest more than 100% in stocks. Values of w > 100% imply leverage and are unrealistic for most individual investors.

### Warnings (stderr)

Printed separately to stderr, these include:
- Short position alerts (when `--no-short` is not used)
- Small sample size warnings (fewer than 60 observations)
- CAPM disclaimers (Gaussian assumption, regime shifts, borrowing/lending rate differences, maturity mismatch, Fama-French factors, thin trading)

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

## Risk-free rate

The risk-free rate ($r_f$) is a critical input. You can provide it in three ways:

1. **Manual input**: `--risk-free-rate 0.045` (annualized, as a decimal)
2. **Auto-fetch from a yield ticker**: `--rf-proxy ^IRX` fetches the latest yield from Yahoo Finance
3. **Default**: 5% if neither flag is provided

Common yield ticker proxies:

| Ticker | Description | Typical use |
|--------|-------------|-------------|
| `^IRX` | 13-week US T-bill | Short-term / academic standard |
| `^FVX` | 5-year US Treasury | Medium-term horizon |
| `^TNX` | 10-year US Treasury | Long-term horizon |

The program validates that $r_f$ is below the minimum-variance portfolio return. If it isn't, the tangency portfolio may not exist, and a warning is displayed.

**Important**: match the rate's maturity to your investment horizon. A 13-week T-bill rate is appropriate for short-term analysis; for multi-year horizons, use a longer-duration bond yield.

## How it works

1. Resolve the risk-free rate (manual, auto-fetched from `--rf-proxy`, or default 0.05)
2. Fetch daily closing prices from Yahoo Finance for the requested lookback period
3. Compute daily simple returns from prices
4. Estimate the annualized mean return vector and covariance matrix (daily values x 252)
5. Validate that $r_f$ < minimum-variance portfolio return (warn if not)
6. Compute the tangency portfolio weights: $w = \Sigma^{-1}(\mu - r_f) / \mathbf{1}^T \Sigma^{-1}(\mu - r_f)$, or use scipy constrained optimization if `--no-short`
7. Verify the result: the ratio $(\mu_i - r_f) / \text{Cov}(r_i, r_p)$ must be identical for all assets
8. Compute per-asset betas (and market betas if a proxy is provided)
9. Compute the Capital Market Line and risk-aversion-based allocations

## CAPM Limitations

The CAPM makes strong assumptions that don't perfectly hold in real markets:

- **Gaussian returns**: real returns have fat tails and skewness
- **Stationary distributions**: past correlations and means may not persist (regime shifts)
- **Size effect**: small-cap stocks tend to earn higher returns than CAPM predicts (Fama-French)
- **Value effect**: high book-to-market stocks tend to outperform (Fama-French)
- **Momentum**: recent winners tend to keep winning in the short term
- **Thin trading**: illiquid or thinly traded stocks produce unreliable covariance estimates
- **Borrowing vs. lending**: the CML assumes you can borrow and lend at the same rate; in reality, borrowing rates are higher, which limits leveraged positions
- **Risk-free rate stability**: historical excess returns over $r_f$ are less stable than raw market returns; a static rate may not persist
