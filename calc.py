"""Pure math functions for tangency portfolio calculations. No I/O."""

import numpy as np
import pandas as pd


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple arithmetic returns: (P_t - P_{t-1}) / P_{t-1}."""
    returns = prices.pct_change().iloc[1:]
    return returns


def compute_tangency_weights(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
) -> np.ndarray:
    """Compute tangency portfolio weights via w = Sigma^-1 (mu - rf) / 1^T Sigma^-1 (mu - rf)."""
    n = len(expected_returns)
    if cov_matrix.shape != (n, n):
        raise ValueError(
            f"Covariance matrix shape {cov_matrix.shape} does not match "
            f"{n} assets"
        )

    risk_premiums = expected_returns - risk_free_rate

    try:
        inv_cov = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix is singular and cannot be inverted")

    raw_weights = inv_cov @ risk_premiums
    weights = raw_weights / np.sum(raw_weights)
    return weights


def portfolio_statistics(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
) -> dict:
    """Compute portfolio expected return, volatility, and Sharpe ratio."""
    expected_return = weights @ expected_returns
    volatility = np.sqrt(weights @ cov_matrix @ weights)
    sharpe_ratio = (expected_return - risk_free_rate) / volatility
    return {
        "expected_return": expected_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
    }


def verify_tangency(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
    tolerance: float = 1e-8,
) -> tuple[bool, np.ndarray]:
    """Verify that (mu_i - rf) / Cov(r_i, r_p) is constant across all assets."""
    risk_premiums = expected_returns - risk_free_rate
    marginal_cov = cov_matrix @ weights  # Cov(r_i, r_p) for each i
    ratios = risk_premiums / marginal_cov
    is_valid = np.max(ratios) - np.min(ratios) < tolerance
    return is_valid, ratios


def estimate_parameters(
    returns: pd.DataFrame, annual_rf: float
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate annualized expected returns and covariance matrix from daily returns.

    Args:
        returns: DataFrame of daily simple returns.
        annual_rf: Annual risk-free rate (used to shift mean returns:
                   annualized_mean = daily_mean * 252).

    Returns:
        Tuple of (expected_returns, cov_matrix), both annualized.
    """
    expected_returns = returns.mean().values * 252
    cov_matrix = returns.cov().values * 252
    return expected_returns, cov_matrix
