"""Unit tests for calc.py — hand-computed values, no network."""

import numpy as np
import pandas as pd
import pytest

from calc import (
    compute_returns,
    compute_tangency_weights,
    estimate_parameters,
    portfolio_statistics,
    verify_tangency,
)


class TestComputeReturns:
    def test_simple_returns(self):
        prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]})
        returns = compute_returns(prices)
        expected = pd.DataFrame({"A": [0.1, 0.1]}, index=[1, 2])
        pd.testing.assert_frame_equal(returns, expected)

    def test_multiple_assets(self):
        prices = pd.DataFrame({"A": [100.0, 110.0], "B": [200.0, 210.0]})
        returns = compute_returns(prices)
        assert returns["A"].iloc[0] == pytest.approx(0.1)
        assert returns["B"].iloc[0] == pytest.approx(0.05)

    def test_drops_first_row(self):
        prices = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1]})
        returns = compute_returns(prices)
        assert len(returns) == 3


class TestEstimateParameters:
    def test_annualizes_correctly(self):
        """Mean and covariance should be multiplied by 252."""
        returns = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [0.02, 0.01, 0.03]})
        mu, cov = estimate_parameters(returns, annual_rf=0.05)
        expected_mu = returns.mean().values * 252
        expected_cov = returns.cov().values * 252
        np.testing.assert_allclose(mu, expected_mu)
        np.testing.assert_allclose(cov, expected_cov)


class TestComputeTangencyWeights:
    def test_weights_sum_to_one(self):
        mu = np.array([0.12, 0.11, 0.10, 0.13, 0.09])
        rf = 0.04
        vols = np.array([0.25, 0.22, 0.27, 0.30, 0.20])
        corr = np.array([
            [1.00, 0.65, 0.55, 0.50, 0.35],
            [0.65, 1.00, 0.60, 0.55, 0.40],
            [0.55, 0.60, 1.00, 0.65, 0.30],
            [0.50, 0.55, 0.65, 1.00, 0.25],
            [0.35, 0.40, 0.30, 0.25, 1.00],
        ])
        cov = np.outer(vols, vols) * corr
        weights = compute_tangency_weights(mu, cov, rf)
        assert np.isclose(np.sum(weights), 1.0)

    def test_two_asset_analytical(self):
        """Verify against hand-computed 2-asset tangency weights."""
        mu = np.array([0.10, 0.06])
        rf = 0.02
        vols = np.array([0.20, 0.15])
        rho = 0.3
        corr = np.array([[1.0, rho], [rho, 1.0]])
        cov = np.outer(vols, vols) * corr

        weights = compute_tangency_weights(mu, cov, rf)

        inv_cov = np.linalg.inv(cov)
        raw = inv_cov @ (mu - rf)
        expected_weights = raw / raw.sum()
        np.testing.assert_allclose(weights, expected_weights, atol=1e-10)

    def test_singular_matrix_raises(self):
        mu = np.array([0.10, 0.08])
        cov = np.array([[1.0, 1.0], [1.0, 1.0]])
        with pytest.raises(ValueError, match="singular"):
            compute_tangency_weights(mu, cov, 0.02)

    def test_dimension_mismatch_raises(self):
        mu = np.array([0.10, 0.08])
        cov = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.raises(ValueError, match="does not match"):
            compute_tangency_weights(mu, cov, 0.02)


class TestVerifyTangency:
    def test_verification_passes(self):
        mu = np.array([0.12, 0.11, 0.10, 0.13, 0.09])
        rf = 0.04
        vols = np.array([0.25, 0.22, 0.27, 0.30, 0.20])
        corr = np.array([
            [1.00, 0.65, 0.55, 0.50, 0.35],
            [0.65, 1.00, 0.60, 0.55, 0.40],
            [0.55, 0.60, 1.00, 0.65, 0.30],
            [0.50, 0.55, 0.65, 1.00, 0.25],
            [0.35, 0.40, 0.30, 0.25, 1.00],
        ])
        cov = np.outer(vols, vols) * corr
        weights = compute_tangency_weights(mu, cov, rf)
        is_valid, ratios = verify_tangency(weights, mu, cov, rf)
        assert is_valid


class TestPortfolioStatistics:
    def test_sharpe_is_maximal(self):
        """No random portfolio should have a higher Sharpe ratio."""
        mu = np.array([0.12, 0.11, 0.10, 0.13, 0.09])
        rf = 0.04
        vols = np.array([0.25, 0.22, 0.27, 0.30, 0.20])
        corr = np.array([
            [1.00, 0.65, 0.55, 0.50, 0.35],
            [0.65, 1.00, 0.60, 0.55, 0.40],
            [0.55, 0.60, 1.00, 0.65, 0.30],
            [0.50, 0.55, 0.65, 1.00, 0.25],
            [0.35, 0.40, 0.30, 0.25, 1.00],
        ])
        cov = np.outer(vols, vols) * corr
        weights = compute_tangency_weights(mu, cov, rf)
        tangency_stats = portfolio_statistics(weights, mu, cov, rf)
        tangency_sharpe = tangency_stats["sharpe_ratio"]

        rng = np.random.default_rng(42)
        for _ in range(1000):
            raw = rng.standard_normal(len(mu))
            w = raw / raw.sum()
            stats = portfolio_statistics(w, mu, cov, rf)
            assert stats["sharpe_ratio"] <= tangency_sharpe + 1e-10
