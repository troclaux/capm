"""Yahoo Finance data fetching."""

import datetime

import pandas as pd
import yfinance as yf


def fetch_prices(tickers: list[str], lookback_days: int = 252) -> pd.DataFrame:
    """Fetch adjusted closing prices from Yahoo Finance.

    Args:
        tickers: List of ticker symbols.
        lookback_days: Number of calendar days to look back.

    Returns:
        DataFrame with tickers as columns and dates as index.

    Raises:
        ValueError: If any ticker returns no data or insufficient data.
    """
    end = datetime.date.today()
    start = end - datetime.timedelta(days=lookback_days)

    data = yf.download(tickers, start=str(start), end=str(end), auto_adjust=True)

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
