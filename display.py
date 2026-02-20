"""Formatted output and warnings."""

import sys

import numpy as np
import pandas as pd


def print_verbose(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    tickers: list[str],
    annual_rf: float,
) -> None:
    """Print intermediate values for cross-checking."""
    daily_rf = annual_rf / 252

    print("\n" + "=" * 60)
    print("VERBOSE: Intermediate Values")
    print("=" * 60)

    print(f"\nRisk-free rate (annual): {annual_rf:.4f} ({annual_rf * 100:.2f}%)")
    print(f"Risk-free rate (daily):  {daily_rf:.6f} ({daily_rf * 100:.4f}%)")

    print(f"\nClosing prices (last 5 of {len(prices)} rows):")
    print(prices.tail().to_string())

    print(f"\nDaily returns (last 5 of {len(returns)} rows):")
    print(returns.tail().to_string())

    print("\nAnnualized expected returns:")
    for i, ticker in enumerate(tickers):
        daily_mean = returns[ticker].mean()
        print(f"  {ticker:<10} daily: {daily_mean:.6f}  annualized: {expected_returns[i]:.4f}")

    print("\nAnnualized covariance matrix:")
    cov_df = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)
    print(cov_df.to_string())

    print("=" * 60 + "\n")


def print_results(
    tickers: list[str],
    weights: np.ndarray,
    stats: dict,
    is_valid: bool,
    ratios: np.ndarray,
) -> None:
    """Print tangency portfolio results."""
    print("\n" + "=" * 60)
    print("Tangency Portfolio (Maximum Sharpe Ratio)")
    print("=" * 60)

    print(f"\n{'Asset':<10} {'Weight':>10}")
    print("-" * 22)
    for ticker, w in zip(tickers, weights):
        print(f"{ticker:<10} {w:>9.2%}")

    print("-" * 22)
    print(f"\nExpected Return: {stats['expected_return']:>7.2%}")
    print(f"Volatility:      {stats['volatility']:>7.2%}")
    print(f"Sharpe Ratio:    {stats['sharpe_ratio']:>7.4f}")

    print(f"\nVerification (risk premium / marginal covariance):")
    for ticker, r in zip(tickers, ratios):
        print(f"  {ticker:<10} {r:.6f}")
    status = "PASS" if is_valid else "FAIL"
    print(f"  Status: {status}")
    print("=" * 60 + "\n")


def print_warnings(
    tickers: list[str],
    weights: np.ndarray,
    num_observations: int,
) -> None:
    """Print risk warnings to stderr."""
    warnings = []

    for i, ticker in enumerate(tickers):
        if weights[i] < 0:
            warnings.append(
                f"  - {ticker}: weight is {weights[i]:.2%} (short position)"
            )

    if warnings:
        print("\nWARNING: Short positions detected:", file=sys.stderr)
        for w in warnings:
            print(w, file=sys.stderr)

    if num_observations < 60:
        print(
            f"\nWARNING: Small sample size ({num_observations} observations). "
            "Estimates may be unreliable.",
            file=sys.stderr,
        )

    print(
        "\nDISCLAIMERS:",
        file=sys.stderr,
    )
    print(
        "  - The tangency portfolio assumes returns are Gaussian and i.i.d. "
        "Real markets deviate from these assumptions.",
        file=sys.stderr,
    )
    print(
        "  - Past return distributions may not persist (regime shifts).",
        file=sys.stderr,
    )
    print(
        "  - No short-sale constraints are applied. Negative weights mean short positions.",
        file=sys.stderr,
    )
