"""Unit tests for calc.py — hand-computed values, no network."""

import numpy as np
import pandas as pd
import pytest

from calc import (
    compute_betas,
    compute_cml,
    compute_cml_allocation,
    compute_market_betas,
    compute_returns,
    compute_tangency_weights,
    compute_tangency_weights_constrained,
    estimate_parameters,
    portfolio_statistics,
    verify_tangency,
)


def _sample_data():
    """5-stock sample data reused across tests."""
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
    return mu, cov, rf


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
        mu, cov, rf = _sample_data()
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


class TestComputeTangencyWeightsConstrained:
    def test_all_weights_non_negative(self):
        mu, cov, rf = _sample_data()
        weights = compute_tangency_weights_constrained(mu, cov, rf)
        assert np.all(weights >= -1e-10)

    def test_weights_sum_to_one(self):
        mu, cov, rf = _sample_data()
        weights = compute_tangency_weights_constrained(mu, cov, rf)
        assert np.isclose(np.sum(weights), 1.0)

    def test_matches_unconstrained_when_no_shorts(self):
        """When unconstrained solution has no negative weights, both should match."""
        # Use parameters where all risk premiums are positive and correlations
        # are moderate, so unconstrained solution naturally has all positive weights
        mu = np.array([0.10, 0.08])
        rf = 0.02
        vols = np.array([0.15, 0.20])
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        cov = np.outer(vols, vols) * corr

        w_unc = compute_tangency_weights(mu, cov, rf)
        # Only test if unconstrained is indeed all positive
        if np.all(w_unc >= 0):
            w_con = compute_tangency_weights_constrained(mu, cov, rf)
            np.testing.assert_allclose(w_con, w_unc, atol=1e-4)


class TestVerifyTangency:
    def test_verification_passes(self):
        mu, cov, rf = _sample_data()
        weights = compute_tangency_weights(mu, cov, rf)
        is_valid, ratios = verify_tangency(weights, mu, cov, rf)
        assert is_valid


class TestPortfolioStatistics:
    def test_sharpe_is_maximal(self):
        """No random portfolio should have a higher Sharpe ratio."""
        mu, cov, rf = _sample_data()
        weights = compute_tangency_weights(mu, cov, rf)
        tangency_stats = portfolio_statistics(weights, mu, cov, rf)
        tangency_sharpe = tangency_stats["sharpe_ratio"]

        rng = np.random.default_rng(42)
        for _ in range(1000):
            raw = rng.standard_normal(len(mu))
            w = raw / raw.sum()
            stats = portfolio_statistics(w, mu, cov, rf)
            assert stats["sharpe_ratio"] <= tangency_sharpe + 1e-10


class TestComputeBetas:
    def test_portfolio_beta_is_one(self):
        """Beta of the tangency portfolio relative to itself should be 1."""
        mu, cov, rf = _sample_data()
        weights = compute_tangency_weights(mu, cov, rf)
        betas = compute_betas(cov, weights)
        portfolio_beta = weights @ betas
        assert portfolio_beta == pytest.approx(1.0)

    def test_known_two_asset_betas(self):
        """Hand-computed betas for a 2-asset case."""
        mu = np.array([0.10, 0.06])
        rf = 0.02
        cov = np.array([[0.04, 0.009], [0.009, 0.0225]])
        weights = compute_tangency_weights(mu, cov, rf)

        betas = compute_betas(cov, weights)
        # Verify: Cov(r_i, r_p) / Var(r_p)
        cov_with_p = cov @ weights
        var_p = weights @ cov @ weights
        expected_betas = cov_with_p / var_p
        np.testing.assert_allclose(betas, expected_betas, atol=1e-10)


class TestComputeMarketBetas:
    def test_market_beta_of_market_is_one(self):
        """Beta of the market relative to itself should be 1."""
        market = pd.Series([0.01, -0.02, 0.015, 0.005, -0.01], name="MKT")
        returns = pd.DataFrame({"MKT": market})
        betas = compute_market_betas(returns, market)
        assert betas[0] == pytest.approx(1.0)


class TestComputeCml:
    def test_cml_slope_is_sharpe(self):
        cml = compute_cml(risk_free_rate=0.04, tangency_return=0.12, tangency_volatility=0.20)
        expected_sharpe = (0.12 - 0.04) / 0.20
        assert cml["slope"] == pytest.approx(expected_sharpe)

    def test_cml_intercept_is_rf(self):
        cml = compute_cml(risk_free_rate=0.04, tangency_return=0.12, tangency_volatility=0.20)
        assert cml["intercept"] == pytest.approx(0.04)


class TestComputeCmlAllocation:
    def test_high_risk_aversion(self):
        """A=100 should put nearly all weight in risk-free."""
        alloc = compute_cml_allocation(
            risk_aversion=100.0, risk_free_rate=0.04,
            tangency_return=0.12, tangency_volatility=0.20,
        )
        assert alloc["w_tangency"] < 0.05
        assert alloc["w_riskfree"] > 0.95

    def test_low_risk_aversion(self):
        """A=0.5 should use leverage (w_tangency > 1)."""
        alloc = compute_cml_allocation(
            risk_aversion=0.5, risk_free_rate=0.04,
            tangency_return=0.12, tangency_volatility=0.20,
        )
        assert alloc["w_tangency"] > 1.0
        assert alloc["w_riskfree"] < 0.0

    def test_allocation_return_matches_cml(self):
        """The allocation's return should lie on the CML."""
        rf = 0.04
        t_ret = 0.12
        t_vol = 0.20
        sharpe = (t_ret - rf) / t_vol

        alloc = compute_cml_allocation(
            risk_aversion=2.0, risk_free_rate=rf,
            tangency_return=t_ret, tangency_volatility=t_vol,
        )
        # CML: E[r] = rf + sharpe * sigma
        expected_return = rf + sharpe * alloc["volatility"]
        assert alloc["expected_return"] == pytest.approx(expected_return)
