"""Yahoo Finance data fetching."""

import datetime

import pandas as pd
import yfinance as yf


def fetch_prices(
    tickers: list[str],
    lookback_years: float = 1.0,
    interval: str = "1mo",
) -> pd.DataFrame:
    """Fetch adjusted closing prices from Yahoo Finance.

    Args:
        tickers: List of ticker symbols.
        lookback_years: Number of years to look back.
        interval: Data frequency ("1mo" for monthly, "1d" for daily).

    Returns:
        DataFrame with tickers as columns and dates as index.

    Raises:
        ValueError: If any ticker returns no data or insufficient data.
    """
    end = datetime.date.today()
    start = end - datetime.timedelta(days=int(lookback_years * 365))

    data = yf.download(tickers, start=str(start), end=str(end), interval=interval, auto_adjust=True)

    if data.empty:
        raise ValueError(f"No data returned for tickers: {tickers}")

    # yf.download returns MultiIndex columns (field, ticker) for multiple tickers
    if len(tickers) == 1:
        prices = data[["Close"]].copy()
        prices.columns = tickers
    else:
        prices = data["Close"].copy()

    # Check all tickers returned data
    for ticker in tickers:
        if ticker not in prices.columns:
            raise ValueError(f"No data returned for ticker: {ticker}")
        if prices[ticker].dropna().empty:
            raise ValueError(f"No data returned for ticker: {ticker}")

    # Drop rows where all values are NaN, then forward-fill single gaps
    prices = prices.dropna(how="all")
    prices = prices.ffill(limit=1)

    # Drop any remaining rows with NaN
    prices = prices.dropna()

    if len(prices) < 2:
        raise ValueError(
            f"Insufficient data: got {len(prices)} rows, need at least 2"
        )

    return prices


def fetch_risk_free_rate(ticker: str, lookback_days: int = 30) -> float:
    """Fetch the latest annualized risk-free rate from a yield ticker.

    Yahoo Finance yield tickers (e.g. ^IRX, ^TNX) report yields as
    percentages (e.g. 4.25 meaning 4.25%). This function returns the
    rate as a decimal (e.g. 0.0425).

    Args:
        ticker: Yield ticker symbol (e.g. ^IRX for 13-week T-bill,
                ^TNX for 10-year Treasury).
        lookback_days: Days to look back to find the latest quote.

    Returns:
        The latest annualized yield as a decimal.

    Raises:
        ValueError: If no data is returned for the ticker.
    """
    end = datetime.date.today()
    start = end - datetime.timedelta(days=lookback_days)

    data = yf.download(ticker, start=str(start), end=str(end), auto_adjust=True)

    if data.empty:
        raise ValueError(f"No data returned for risk-free proxy: {ticker}")

    latest_yield = data["Close"].dropna().iloc[-1]

    # Yahoo yield tickers report in percentage points (e.g. 4.25 = 4.25%)
    # Handle both scalar and single-element Series
    if hasattr(latest_yield, 'item'):
        rate = latest_yield.item() / 100.0
    else:
        rate = float(latest_yield) / 100.0

    return rate
