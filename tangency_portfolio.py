"""CLI entry point for tangency portfolio calculator."""

import argparse
import sys

import numpy as np

from calc import (
    compute_betas,
    compute_bloomberg_adjusted_betas,
    compute_cml,
    compute_cml_allocation,
    compute_efficient_frontier,
    compute_market_betas,
    compute_returns,
    compute_tangency_weights,
    compute_tangency_weights_constrained,
    estimate_parameters,
    portfolio_statistics,
    validate_risk_free_rate,
    verify_tangency,
)
from data import fetch_prices, fetch_risk_free_rate
from display import (
    print_betas,
    print_cml,
    print_cml_allocations,
    print_results,
    print_verbose,
    print_warnings,
)


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
        default=None,
        help="Annual risk-free rate as a decimal (default: 0.05). Overridden by --rf-proxy if both given.",
    )
    parser.add_argument(
        "--rf-proxy",
        type=str,
        help="Fetch risk-free rate from a yield ticker (e.g. ^IRX for 13-week T-bill, ^TNX for 10-year Treasury)",
    )
    parser.add_argument(
        "--no-short",
        action="store_true",
        help="Forbid short positions (constrain all weights >= 0)",
    )
    parser.add_argument(
        "--market-proxy",
        type=str,
        default="^BVSP",
        help="Market benchmark ticker for beta calculation (default: ^BVSP / Ibovespa)",
    )
    parser.add_argument(
        "--risk-aversion",
        type=float,
        help="Risk aversion parameter A for CML allocation (omit to see examples for A=1,2,5)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print intermediate values (prices, returns, covariance) for cross-checking",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="show",
        default=None,
        metavar="FILE",
        help="Show mean-variance diagram. Optionally save to FILE (e.g. plot.png)",
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

    # Resolve risk-free rate
    if args.rf_proxy:
        try:
            risk_free_rate = fetch_risk_free_rate(args.rf_proxy.upper())
            print(f"Fetched risk-free rate from {args.rf_proxy.upper()}: {risk_free_rate:.4%}")
        except ValueError as e:
            print(f"Data error: {e}", file=sys.stderr)
            return 2
    elif args.risk_free_rate is not None:
        risk_free_rate = args.risk_free_rate
    else:
        risk_free_rate = 0.05

    # Build fetch list (tickers + optional market proxy)
    market_proxy = args.market_proxy.upper() if args.market_proxy else None
    fetch_tickers = list(tickers)
    if market_proxy and market_proxy not in fetch_tickers:
        fetch_tickers.append(market_proxy)

    # Fetch data
    try:
        prices = fetch_prices(fetch_tickers, lookback_days=args.lookback)
    except ValueError as e:
        print(f"Data error: {e}", file=sys.stderr)
        return 2

    # Compute returns for all fetched tickers
    all_returns = compute_returns(prices)

    # Separate market proxy returns if present
    market_returns = None
    if market_proxy:
        market_returns = all_returns[market_proxy]
        asset_returns = all_returns[tickers]
    else:
        asset_returns = all_returns

    # Estimate parameters from asset returns only
    expected_returns, cov_matrix = estimate_parameters(asset_returns, risk_free_rate)

    # Validate risk-free rate vs min-variance portfolio
    rf_warning = validate_risk_free_rate(risk_free_rate, expected_returns, cov_matrix)
    if rf_warning:
        print(f"\nWARNING: {rf_warning}", file=sys.stderr)

    # Compute tangency weights
    try:
        if args.no_short:
            weights = compute_tangency_weights_constrained(
                expected_returns, cov_matrix, risk_free_rate
            )
        else:
            weights = compute_tangency_weights(
                expected_returns, cov_matrix, risk_free_rate
            )
    except ValueError as e:
        print(f"Computation error: {e}", file=sys.stderr)
        return 2

    stats = portfolio_statistics(weights, expected_returns, cov_matrix, risk_free_rate)

    # Verification (only meaningful for unconstrained solution)
    is_valid = None
    ratios = None
    if not args.no_short:
        is_valid, ratios = verify_tangency(
            weights, expected_returns, cov_matrix, risk_free_rate
        )

    # Betas
    portfolio_betas = compute_betas(cov_matrix, weights)
    market_betas = None
    adjusted_betas = None
    if market_returns is not None:
        market_betas = compute_market_betas(asset_returns, market_returns)
        adjusted_betas = compute_bloomberg_adjusted_betas(market_betas)

    # CML
    cml = compute_cml(
        risk_free_rate, stats["expected_return"], stats["volatility"]
    )

    # CML allocations
    if args.risk_aversion:
        risk_aversions = [args.risk_aversion]
    else:
        risk_aversions = [1.0, 2.0, 5.0]
    allocations = [
        compute_cml_allocation(
            a, risk_free_rate, stats["expected_return"], stats["volatility"]
        )
        for a in risk_aversions
    ]

    # Display
    if args.verbose:
        print_verbose(
            prices[tickers], asset_returns, expected_returns, cov_matrix,
            tickers, risk_free_rate,
        )

    print_results(tickers, weights, stats, is_valid, ratios)
    print_betas(tickers, portfolio_betas, market_betas, adjusted_betas, market_proxy)
    print_cml(cml)
    print_cml_allocations(allocations, risk_aversions)
    print_warnings(tickers, weights, num_observations=len(asset_returns))

    # Plot
    if args.plot is not None:
        from plot import plot_tangency_portfolio

        frontier_vols, frontier_rets = compute_efficient_frontier(
            expected_returns, cov_matrix, allow_short=not args.no_short,
        )
        asset_vols = np.sqrt(np.diag(cov_matrix))
        output_file = None if args.plot == "show" else args.plot
        plot_tangency_portfolio(
            frontier_vols=frontier_vols,
            frontier_rets=frontier_rets,
            tangency_vol=stats["volatility"],
            tangency_ret=stats["expected_return"],
            risk_free_rate=risk_free_rate,
            sharpe_ratio=cml["slope"],
            asset_vols=asset_vols,
            asset_rets=expected_returns,
            tickers=tickers,
            output_file=output_file,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
