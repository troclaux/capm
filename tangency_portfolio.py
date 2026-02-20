"""CLI entry point for tangency portfolio calculator."""

import argparse
import sys

import numpy as np

from calc import (
    compute_returns,
    compute_tangency_weights,
    estimate_parameters,
    portfolio_statistics,
    verify_tangency,
)
from data import fetch_prices
from display import print_results, print_verbose, print_warnings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate the tangency portfolio (maximum Sharpe ratio) for a set of stocks."
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        type=str,
        help="Ticker symbols (e.g. AAPL MSFT GOOG)",
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to a .txt file with one ticker per line",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=252,
        help="Lookback period in calendar days (default: 252, ~1 year)",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.05,
        help="Annual risk-free rate (default: 0.05)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print intermediate values (prices, returns, covariance) for cross-checking",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Collect tickers from positional args and/or file
    tickers = list(args.tickers) if args.tickers else []
    if args.file:
        try:
            with open(args.file) as fh:
                for line in fh:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        tickers.append(stripped)
        except FileNotFoundError:
            print(f"File not found: {args.file}", file=sys.stderr)
            return 1

    if not tickers:
        print("Error: no tickers provided. Pass them as arguments or via --file.", file=sys.stderr)
        return 1

    tickers = [t.upper() for t in tickers]

    # Fetch data
    try:
        prices = fetch_prices(tickers, lookback_days=args.lookback)
    except ValueError as e:
        print(f"Data error: {e}", file=sys.stderr)
        return 2

    # Compute
    returns = compute_returns(prices)
    expected_returns, cov_matrix = estimate_parameters(returns, args.risk_free_rate)

    try:
        weights = compute_tangency_weights(expected_returns, cov_matrix, args.risk_free_rate)
    except ValueError as e:
        print(f"Computation error: {e}", file=sys.stderr)
        return 2

    stats = portfolio_statistics(weights, expected_returns, cov_matrix, args.risk_free_rate)
    is_valid, ratios = verify_tangency(
        weights, expected_returns, cov_matrix, args.risk_free_rate
    )

    # Display
    if args.verbose:
        print_verbose(prices, returns, expected_returns, cov_matrix, tickers, args.risk_free_rate)

    print_results(tickers, weights, stats, is_valid, ratios)
    print_warnings(tickers, weights, num_observations=len(returns))

    return 0


if __name__ == "__main__":
    sys.exit(main())
