"""Yahoo Finance data fetching."""

import datetime

import numpy as np
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
    # Anchor the window to the last market day of the previous month.
    # yfinance's `end` is exclusive, so the first of the current month makes
    # the last included row the last trading day of the previous month.
    end = datetime.date.today().replace(day=1)
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
    # Anchor to the last market day of the previous month (end is exclusive).
    end = datetime.date.today().replace(day=1)
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


def parse_portfolio_file(path: str, expected_tickers: list[str]) -> np.ndarray:
    """Parse a portfolio file and return weights ordered to match expected_tickers.

    Format: one 'TICKER WEIGHT' pair per line, whitespace-separated. Lines
    starting with '#' are comments; blank lines are ignored. Tickers in the
    file must match expected_tickers exactly (same set).

    Raises:
        ValueError: on malformed lines, duplicate tickers, or ticker mismatch.
    """
    weights_dict: dict[str, float] = {}
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) != 2:
                raise ValueError(
                    f"{path}:{lineno}: expected 'TICKER WEIGHT', got {stripped!r}"
                )
            ticker = parts[0].upper()
            try:
                weight = float(parts[1])
            except ValueError:
                raise ValueError(
                    f"{path}:{lineno}: invalid weight {parts[1]!r} for {ticker}"
                )
            if ticker in weights_dict:
                raise ValueError(f"{path}:{lineno}: duplicate ticker {ticker}")
            weights_dict[ticker] = weight

    expected_set = set(expected_tickers)
    given_set = set(weights_dict.keys())
    if given_set != expected_set:
        missing = sorted(expected_set - given_set)
        extra = sorted(given_set - expected_set)
        msgs = []
        if missing:
            msgs.append(f"missing: {missing}")
        if extra:
            msgs.append(f"unknown: {extra}")
        raise ValueError(
            f"portfolio file tickers must match analyzed tickers — {'; '.join(msgs)}"
        )

    return np.array([weights_dict[t] for t in expected_tickers])
