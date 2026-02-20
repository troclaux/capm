import numpy as np


def compute_tangency_weights(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
) -> np.ndarray:
    """Compute tangency portfolio weights via w = Σ⁻¹(μ - rf) / 1ᵀΣ⁻¹(μ - rf)."""
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
    """Verify that (μᵢ - rf) / Cov(rᵢ, rₚ) is constant across all assets."""
    risk_premiums = expected_returns - risk_free_rate
    marginal_cov = cov_matrix @ weights  # Cov(rᵢ, rₚ) for each i
    ratios = risk_premiums / marginal_cov
    is_valid = np.max(ratios) - np.min(ratios) < tolerance
    return is_valid, ratios


def print_results(
    asset_names: list[str],
    weights: np.ndarray,
    stats: dict,
    is_valid: bool,
    ratios: np.ndarray,
) -> None:
    """Pretty-print tangency portfolio results."""
    print("=== Tangency Portfolio ===\n")

    print("Asset Weights:")
    for name, w in zip(asset_names, weights):
        print(f"  {name:>6s}: {w:7.2%}")

    print(f"\nPortfolio Statistics:")
    print(f"  Expected Return: {stats['expected_return']:7.2%}")
    print(f"  Volatility:      {stats['volatility']:7.2%}")
    print(f"  Sharpe Ratio:    {stats['sharpe_ratio']:7.4f}")

    print(f"\nVerification (risk premium / marginal covariance):")
    for name, r in zip(asset_names, ratios):
        print(f"  {name:>6s}: {r:.6f}")
    status = "PASS" if is_valid else "FAIL"
    print(f"  Status: {status}")


if __name__ == "__main__":
    asset_names = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]

    expected_returns = np.array([0.12, 0.11, 0.10, 0.13, 0.09])

    risk_free_rate = 0.04

    vols = np.array([0.25, 0.22, 0.27, 0.30, 0.20])
    corr = np.array([
        [1.00, 0.65, 0.55, 0.50, 0.35],
        [0.65, 1.00, 0.60, 0.55, 0.40],
        [0.55, 0.60, 1.00, 0.65, 0.30],
        [0.50, 0.55, 0.65, 1.00, 0.25],
        [0.35, 0.40, 0.30, 0.25, 1.00],
    ])
    cov_matrix = np.outer(vols, vols) * corr

    weights = compute_tangency_weights(expected_returns, cov_matrix, risk_free_rate)
    stats = portfolio_statistics(weights, expected_returns, cov_matrix, risk_free_rate)
    is_valid, ratios = verify_tangency(
        weights, expected_returns, cov_matrix, risk_free_rate
    )
    print_results(asset_names, weights, stats, is_valid, ratios)
