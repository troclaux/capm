"""Pure math functions for tangency portfolio calculations. No I/O."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


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


def compute_tangency_weights_constrained(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
) -> np.ndarray:
    """Compute tangency portfolio weights with no-short-selling constraint.

    Uses scipy.optimize.minimize to maximize Sharpe ratio subject to
    w_i >= 0 and sum(w) == 1.
    """
    n = len(expected_returns)
    if cov_matrix.shape != (n, n):
        raise ValueError(
            f"Covariance matrix shape {cov_matrix.shape} does not match "
            f"{n} assets"
        )

    def neg_sharpe(w):
        port_return = w @ expected_returns
        port_vol = np.sqrt(w @ cov_matrix @ w)
        if port_vol < 1e-12:
            return 1e10
        return -(port_return - risk_free_rate) / port_vol

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n
    x0 = np.ones(n) / n

    result = minimize(
        neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints
    )
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    return result.x


def compute_betas(
    cov_matrix: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute beta of each asset relative to the tangency portfolio.

    beta_i = Cov(r_i, r_p) / Var(r_p)
    """
    cov_with_portfolio = cov_matrix @ weights
    portfolio_variance = weights @ cov_matrix @ weights
    return cov_with_portfolio / portfolio_variance


def compute_market_betas(
    returns: pd.DataFrame,
    market_returns: pd.Series,
) -> np.ndarray:
    """Compute beta of each asset relative to a market proxy.

    beta_i = Cov(r_i, r_m) / Var(r_m)
    """
    market_var = market_returns.var() * 252
    betas = np.array([
        (returns[col].cov(market_returns) * 252) / market_var
        for col in returns.columns
    ])
    return betas


def compute_cml(
    risk_free_rate: float,
    tangency_return: float,
    tangency_volatility: float,
) -> dict:
    """Compute Capital Market Line parameters.

    CML: E[r] = rf + (sharpe) * sigma
    """
    sharpe = (tangency_return - risk_free_rate) / tangency_volatility
    return {"intercept": risk_free_rate, "slope": sharpe}


def compute_cml_allocation(
    risk_aversion: float,
    risk_free_rate: float,
    tangency_return: float,
    tangency_volatility: float,
) -> dict:
    """Compute the optimal CML allocation for a given risk aversion level.

    w_tangency = (E[r_T] - rf) / (A * sigma_T^2)
    """
    tangency_variance = tangency_volatility ** 2
    w_tangency = (tangency_return - risk_free_rate) / (risk_aversion * tangency_variance)
    w_riskfree = 1.0 - w_tangency
    expected_return = w_riskfree * risk_free_rate + w_tangency * tangency_return
    volatility = abs(w_tangency) * tangency_volatility
    return {
        "w_tangency": w_tangency,
        "w_riskfree": w_riskfree,
        "expected_return": expected_return,
        "volatility": volatility,
    }
