"""
OPTIMIZED GPU CAPM - Trusts Clean Data from Data Processing
"""

import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from config import CONFIG, DEVICE

logger = logging.getLogger(__name__)


def apply_capm_filter(tokens, market_index, price_data):
    """
    ✅ SIMPLIFIED: Trust data_processing gave us aligned data
    """
    if market_index not in price_data:
        return {}

    # ✅ Define valid tokens FIRST
    valid_tokens = [t for t in tokens if t in price_data and t != market_index]

    if not valid_tokens:
        return {}

    # ✅ Common aligned dates (from INDEX)
    common_dates = price_data[market_index].index
    logger.info(f"CAPM: {len(common_dates)} aligned days")

    # ======================================
    # CAPM INPUT DATE ALIGNMENT VERIFICATION
    # ======================================
    logger.info("🔍 CAPM INPUT VERIFICATION")

    logger.info(
        f"   INDEX | rows={len(common_dates)} | "
        f"dates={common_dates[0].date()} → {common_dates[-1].date()}"
    )

    for token in valid_tokens:
        same_dates = price_data[token].index.equals(common_dates)

        logger.info(
            f"   {token:<10} | "
            f"rows={len(price_data[token]):4d} | "
            f"dates_match_index={same_dates}"
        )

        if not same_dates:
            logger.error(f"❌ CAPM input date mismatch: {token}")
            raise RuntimeError(f"CAPM input misalignment: {token}")

    # ✅ NO DATE CHECKING NEEDED - already aligned
    valid_tokens = [t for t in tokens if t in price_data and t != market_index]

    if not valid_tokens:
        return {}

    # Extract prices directly
    index_prices = price_data[market_index]["close"].to_numpy(dtype=np.float64)
    token_prices_list = [
        price_data[token]["close"].to_numpy(dtype=np.float64) for token in valid_tokens
    ]

    # ✅ IMPORTANT: Check for any NaN in prices
    if np.any(np.isnan(index_prices)):
        logger.error("INDEX prices contain NaN!")
        return {}

    for i, token in enumerate(valid_tokens):
        if np.any(np.isnan(token_prices_list[i])):
            logger.error(f"Token {token} prices contain NaN!")
            return {}

    # Batch CAPM computation
    token_prices_stack = torch.tensor(
        np.vstack(token_prices_list), device=DEVICE, dtype=torch.float32
    )
    index_prices_gpu = torch.tensor(index_prices, device=DEVICE, dtype=torch.float32)

    betas, alphas, residuals_stack = compute_capm(token_prices_stack, index_prices_gpu)

    # Create results
    capm_results = {}
    residual_dates = common_dates[1:]

    for i, token in enumerate(valid_tokens):
        capm_results[token] = {
            "beta": float(betas[i].item()),
            "alpha": float(alphas[i].item()),
            "residuals": pd.Series(
                residuals_stack[i].cpu().numpy(),
                index=residual_dates,
                name=f"{token}_residuals",
            ),
            "common_days_used": len(residual_dates),
            "common_days": len(residual_dates),
            "start_date": residual_dates[0],
            "end_date": common_dates[-1],
        }

    logger.info(f"✅ CAPM complete: {len(capm_results)} tokens")

    # Log summary statistics
    if capm_results:
        beta_vals = [result["beta"] for result in capm_results.values()]
        alpha_vals = [result["alpha"] for result in capm_results.values()]
        logger.info(f"   Beta range: [{min(beta_vals):.3f}, {max(beta_vals):.3f}]")
        logger.info(f"   Alpha range: [{min(alpha_vals):.4f}, {max(alpha_vals):.4f}]")

    return capm_results


def compute_capm(token_prices_stack, index_prices):
    """CAPM with simple returns - corrected"""
    rf_annual = CONFIG.get("risk_free_rate", 0.0)
    rf_daily = rf_annual / 252
    # Simple returns
    token_returns = (token_prices_stack[:, 1:] / token_prices_stack[:, :-1]) - 1
    index_returns = (index_prices[1:] / index_prices[:-1]) - 1

    # Assuming risk-free rate = 0 for crypto

    token_excess = token_returns - rf_daily
    market_excess = index_returns - rf_daily

    n_tokens, n_obs = token_excess.shape

    # Design matrix
    X = torch.stack(
        [torch.ones(n_obs, device=DEVICE, dtype=torch.float32), market_excess], dim=1
    )

    X_batch = X.unsqueeze(0).expand(n_tokens, -1, -1)
    y_batch = token_excess.unsqueeze(-1)

    # OLS regression
    coeffs = torch.linalg.lstsq(X_batch, y_batch).solution.squeeze(-1)
    alphas = coeffs[:, 0]
    betas = coeffs[:, 1]

    # Predicted returns: alpha + beta * market_excess
    predicted = alphas.unsqueeze(1) + betas.unsqueeze(1) * market_excess.unsqueeze(0)

    # Residuals = actual - predicted
    residuals = token_excess - predicted

    return betas, alphas, residuals


def plot_capm_results(
    capm_results: Dict[str, Dict[str, Any]], save_path: Optional[Path] = None
) -> None:
    """OPTIMIZED CAPM visualization"""
    if not capm_results:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tokens = list(capm_results.keys())
        betas = [capm_results[token]["beta"] for token in tokens]
        alphas = [capm_results[token]["alpha"] for token in tokens]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Beta distribution
        ax1.hist(betas, bins=15, alpha=0.7, color="skyblue", edgecolor="black")
        ax1.set_xlabel("Beta Values")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of CAPM Betas")
        ax1.grid(True, alpha=0.3)

        # Alpha distribution (annualized)
        alphas_annualized = [alpha * 252 for alpha in alphas]
        ax2.hist(
            alphas_annualized, bins=15, alpha=0.7, color="lightcoral", edgecolor="black"
        )
        ax2.set_xlabel("Alpha Values (Annualized)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of CAPM Alphas")
        ax2.grid(True, alpha=0.3)

        # Save plot
        final_save_path = (
            save_path or Path(CONFIG["results_dir"]) / "capm_research_summary.png"
        )
        final_save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(final_save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"📊 CAPM visualization saved to {final_save_path}")

    except Exception as e:
        logger.debug(f"Visualization skipped: {e}")
