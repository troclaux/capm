import numpy as np
import pytest

from tangency_portfolio import (
    compute_tangency_weights,
    portfolio_statistics,
    verify_tangency,
)


def _sample_data():
    """5-stock sample data used across tests."""
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
    return expected_returns, cov_matrix, risk_free_rate


def test_weights_sum_to_one():
    expected_returns, cov_matrix, rf = _sample_data()
    weights = compute_tangency_weights(expected_returns, cov_matrix, rf)
    assert np.isclose(np.sum(weights), 1.0)


def test_two_asset_analytical():
    """Verify against hand-computed 2-asset tangency weights."""
    mu = np.array([0.10, 0.06])
    rf = 0.02
    vols = np.array([0.20, 0.15])
    rho = 0.3
    corr = np.array([[1.0, rho], [rho, 1.0]])
    cov = np.outer(vols, vols) * corr

    weights = compute_tangency_weights(mu, cov, rf)

    # Hand calculation:
    # risk_premiums = [0.08, 0.04]
    # cov = [[0.04, 0.009], [0.009, 0.0225]]
    # inv_cov @ risk_premiums gives raw weights, then normalize
    inv_cov = np.linalg.inv(cov)
    raw = inv_cov @ (mu - rf)
    expected_weights = raw / raw.sum()

    np.testing.assert_allclose(weights, expected_weights, atol=1e-10)


def test_verification_passes():
    expected_returns, cov_matrix, rf = _sample_data()
    weights = compute_tangency_weights(expected_returns, cov_matrix, rf)
    is_valid, ratios = verify_tangency(weights, expected_returns, cov_matrix, rf)
    assert is_valid


def test_sharpe_is_maximal():
    """No random portfolio should have a higher Sharpe ratio."""
    expected_returns, cov_matrix, rf = _sample_data()
    weights = compute_tangency_weights(expected_returns, cov_matrix, rf)
    tangency_stats = portfolio_statistics(weights, expected_returns, cov_matrix, rf)
    tangency_sharpe = tangency_stats["sharpe_ratio"]

    rng = np.random.default_rng(42)
    for _ in range(1000):
        raw = rng.standard_normal(len(expected_returns))
        w = raw / raw.sum()
        stats = portfolio_statistics(w, expected_returns, cov_matrix, rf)
        assert stats["sharpe_ratio"] <= tangency_sharpe + 1e-10


def test_singular_matrix_raises():
    mu = np.array([0.10, 0.08])
    cov = np.array([[1.0, 1.0], [1.0, 1.0]])  # singular
    with pytest.raises(ValueError, match="singular"):
        compute_tangency_weights(mu, cov, 0.02)
