"""Tangency portfolio mean-variance diagram."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def plot_tangency_portfolio(
    frontier_vols: np.ndarray,
    frontier_rets: np.ndarray,
    tangency_vol: float,
    tangency_ret: float,
    risk_free_rate: float,
    sharpe_ratio: float,
    asset_vols: np.ndarray,
    asset_rets: np.ndarray,
    tickers: list[str],
    output_file: str | None = None,
) -> None:
    """Draw the mean-standard deviation diagram with tangency portfolio.

    Shows: feasible set, efficient frontier, minimum variance portfolio,
    risk-free rate, CML, tangency point, and individual assets.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # --- Feasible set (shaded region) ---
    # The frontier traces the boundary; fill the area to the right
    # Sort by returns to get a proper boundary curve
    sort_idx = np.argsort(frontier_rets)
    sorted_vols = frontier_vols[sort_idx]
    sorted_rets = frontier_rets[sort_idx]

    # Fill feasible region: from frontier curve to the right
    max_vol = max(sorted_vols.max(), asset_vols.max()) * 1.3
    ax.fill_betweenx(
        sorted_rets, sorted_vols, max_vol,
        alpha=0.08, color="steelblue", label="Feasible set",
    )

    # --- Full minimum-variance boundary (hyperbola) ---
    ax.plot(
        sorted_vols, sorted_rets,
        color="steelblue", linewidth=1.5, alpha=0.5, linestyle="--",
        label="Min-variance boundary",
    )

    # --- Efficient frontier (upper portion above min-variance point) ---
    min_var_idx = np.argmin(sorted_vols)
    min_var_vol = sorted_vols[min_var_idx]
    min_var_ret = sorted_rets[min_var_idx]

    efficient_mask = sorted_rets >= min_var_ret
    ax.plot(
        sorted_vols[efficient_mask], sorted_rets[efficient_mask],
        color="steelblue", linewidth=2.5,
        label="Efficient frontier",
    )

    # --- Minimum variance portfolio (V) ---
    ax.plot(
        min_var_vol, min_var_ret,
        marker="D", color="darkgreen", markersize=9, zorder=5,
    )
    ax.annotate(
        "V (min variance)", (min_var_vol, min_var_ret),
        textcoords="offset points", xytext=(12, -5),
        fontsize=9, color="darkgreen", fontweight="bold",
    )

    # --- Risk-free rate on Y-axis ---
    ax.plot(
        0, risk_free_rate,
        marker="o", color="red", markersize=8, zorder=5,
    )
    ax.annotate(
        f"$r_f$ = {risk_free_rate:.2%}", (0, risk_free_rate),
        textcoords="offset points", xytext=(8, -12),
        fontsize=9, color="red", fontweight="bold",
    )

    # --- Capital Market Line (CML) ---
    cml_vol_max = max_vol
    cml_vols = np.linspace(0, cml_vol_max, 200)
    cml_rets = risk_free_rate + sharpe_ratio * cml_vols

    # Segment before tangency (lending)
    lending_mask = cml_vols <= tangency_vol
    ax.plot(
        cml_vols[lending_mask], cml_rets[lending_mask],
        color="darkorange", linewidth=2.2, linestyle="-",
        label="CML (lending)",
    )

    # Segment after tangency (borrowing / leverage)
    borrowing_mask = cml_vols >= tangency_vol
    ax.plot(
        cml_vols[borrowing_mask], cml_rets[borrowing_mask],
        color="darkorange", linewidth=2.2, linestyle="--",
        label="CML (leverage)",
    )

    # --- Tangency portfolio (T) ---
    ax.plot(
        tangency_vol, tangency_ret,
        marker="*", color="darkorange", markersize=16, zorder=6,
        markeredgecolor="black", markeredgewidth=0.5,
    )
    ax.annotate(
        f"T (tangency)\n$\\mu$={tangency_ret:.2%}, $\\sigma$={tangency_vol:.2%}",
        (tangency_vol, tangency_ret),
        textcoords="offset points", xytext=(14, 8),
        fontsize=9, color="darkorange", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    # --- Individual assets ---
    for i, ticker in enumerate(tickers):
        ax.plot(
            asset_vols[i], asset_rets[i],
            marker="o", color="gray", markersize=6, zorder=4,
        )
        ax.annotate(
            ticker, (asset_vols[i], asset_rets[i]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8, color="dimgray",
        )

    # --- Labels and formatting ---
    ax.set_xlabel("Standard Deviation ($\\sigma$)", fontsize=12)
    ax.set_ylabel("Expected Return ($E[r]$)", fontsize=12)
    ax.set_title("Mean-Variance Diagram: Tangency Portfolio & CML", fontsize=13)

    # Format axes as percentages
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    ax.set_xlim(left=-0.01)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add Sharpe ratio annotation
    ax.text(
        0.02, 0.98,
        f"Sharpe Ratio = {sharpe_ratio:.4f}",
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

    plt.close(fig)
