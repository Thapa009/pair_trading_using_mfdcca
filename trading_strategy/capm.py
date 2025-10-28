import torch
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from config import CONFIG, DEVICE
from typing import Optional, Tuple, Union, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================================
# ✅ FLEXIBLE LOOKBACK CAPM WITH AVAILABLE TRADING DAYS
# ============================================================================

def capm_filter_flexible(token_df: pd.DataFrame, index_df: pd.DataFrame, target_lookback: int = 252) -> Tuple[Optional[float], Optional[float], Optional[torch.Tensor], Optional[np.ndarray]]:
    """
    Flexible CAPM that uses whatever trading days are available
    (Recommended approach - handles holidays, missing data, variable-length weeks)
    """
    
    # ✅ Get aligned data - convert to numpy arrays explicitly
    token_prices = np.asarray(token_df["close_token"].values)
    index_prices = np.asarray(index_df["close_index"].values)
    dates = np.asarray(token_df["date"].values)
    
    actual_days = len(token_prices)
    logger.debug(f"Flexible CAPM: {actual_days} trading days available (target: {target_lookback})")
    
    # ✅ Move to GPU
    token_gpu = torch.tensor(token_prices, dtype=torch.float32, device=DEVICE)
    index_gpu = torch.tensor(index_prices, dtype=torch.float32, device=DEVICE)
    
    # ✅ Compute returns (length = T-1)
    token_ret = (token_gpu[1:] / token_gpu[:-1] - 1.0)
    index_ret = (index_gpu[1:] / index_gpu[:-1] - 1.0)
    
    # ✅ Handle NaN/Inf
    token_ret = token_ret.nan_to_num_(0.0)
    index_ret = index_ret.nan_to_num_(0.0)
    
    # ✅ CRITICAL FIX: Align dates with returns (T-1 length)
    aligned_dates = dates[1:]  # Dates match returns exactly
    
    # ✅ Excess returns
    rf_daily = CONFIG["risk_free_rate"] / 252
    token_excess = token_ret - rf_daily
    index_excess = index_ret - rf_daily
    
    # ✅ Validate alignment
    T = token_excess.shape[0]
    
    # ✅ FIX: Add proper length checks before assertion
    if len(aligned_dates) != T:
        logger.error(f"Date alignment failed: dates={len(aligned_dates)}, returns={T}")
        return None, None, None, None
    
    # ✅ Regression
    ones = torch.ones(T, dtype=torch.float32, device=DEVICE)
    X = torch.stack([ones, index_excess], dim=1)
    y = token_excess.unsqueeze(1)
    
    try:
        coeffs = torch.linalg.lstsq(X, y, driver='gels').solution
        alpha = coeffs[0, 0]
        beta = coeffs[1, 0]
        
        residuals_gpu = y.squeeze() - (X @ coeffs).squeeze()
        
        # ✅ FIX: Add proper null checks before final validation
        if residuals_gpu is None or aligned_dates is None:
            logger.error("Residuals or dates are None after regression")
            return None, None, None, None
        
        # ✅ FIX: Safe length comparison
        residuals_len = len(residuals_gpu) if residuals_gpu is not None else 0
        dates_len = len(aligned_dates) if aligned_dates is not None else 0
        
        if residuals_len != dates_len:
            logger.error(f"Final alignment failed: residuals={residuals_len}, dates={dates_len}")
            return None, None, None, None
        
        logger.debug(f"Flexible CAPM completed: {actual_days} days, {T} returns")
        
        return beta.item(), alpha.item(), residuals_gpu, aligned_dates
    
    except Exception as e:
        logger.error(f"CAPM regression failed: {e}")
        return None, None, None, None

def apply_capm_filter_flexible(tokens, market_index, price_data, start_date, end_date, 
                              target_lookback=252, save_summary=True):
    """
    FLEXIBLE CAPM: Uses whatever trading days are available in the period
    """
    capm_results = {}
    
    # Load market index data with flexible date handling
    index_prices = []
    common_dates = None
    
    # Find common dates across all tokens and index
    for token in tokens:
        if token in price_data and 'INDEX' in price_data:
            token_dates = price_data[token].index
            index_dates = price_data['INDEX'].index
            token_common = token_dates.intersection(index_dates)
            
            if common_dates is None:
                common_dates = token_common
            else:
                common_dates = common_dates.intersection(token_common)
    
    if common_dates is None or len(common_dates) < 10:
        logger.error("No common dates found across tokens and index")
        return {}
    
    # Filter to requested date range
    common_dates = common_dates[(common_dates >= start_date) & (common_dates <= end_date)]
    
    actual_days = len(common_dates)
    logger.info(f"Flexible CAPM: Using {actual_days} available trading days (target: {target_lookback})")
    
    if actual_days < 10:
        logger.error(f"Insufficient common dates in requested range: {actual_days} days")
        return {}
    
    # Prepare index data for all tokens - ensure numpy arrays
    index_df = pd.DataFrame({
        'date': np.asarray(common_dates),  # ✅ Explicit conversion to numpy
        'close_index': np.asarray(price_data['INDEX'].loc[common_dates, 'close'])
    })
    
    # Process each token
    for token in tokens:
        if token not in price_data:
            continue
            
        # Create aligned token dataframe - ensure numpy arrays
        token_prices = np.asarray(price_data[token].loc[common_dates, 'close'])
        token_df = pd.DataFrame({
            'date': np.asarray(common_dates),  # ✅ Explicit conversion to numpy
            'close_token': token_prices
        })
        
        # ✅ Use FLEXIBLE capm_filter
        beta, alpha, residuals_gpu, aligned_dates = capm_filter_flexible(
            token_df, index_df, target_lookback
        )
        
        # ✅ FIX: Add comprehensive null checking
        if (beta is None or alpha is None or 
            residuals_gpu is None or aligned_dates is None):
            logger.warning(f"CAPM failed for token {token}, skipping")
            continue
            
        # Convert GPU residuals to CPU for consistency
        residuals_cpu = residuals_gpu.cpu().numpy() if isinstance(residuals_gpu, torch.Tensor) else residuals_gpu
        
        # ✅ FIX: Add additional safety checks
        if residuals_cpu is None or aligned_dates is None:
            logger.warning(f"Invalid residuals or dates for token {token}, skipping")
            continue
        
        # ✅ Use the EXACT dates from capm_filter
        capm_results[token] = {
            'beta': beta,
            'alpha': alpha,
            'residuals': residuals_cpu,
            'date_index': aligned_dates,  # Direct from CAPM - guaranteed alignment
            'residuals_gpu': residuals_gpu,  # Keep GPU version for MFDCCA
            'actual_days_used': actual_days  # Track how many days were actually used
        }
        
        logger.debug(
            f"Flexible CAPM for {token}: {actual_days} days, "
            f"residuals={len(residuals_cpu)}, dates={len(aligned_dates)}"
        )
 
    logger.info(f"✅ Flexible CAPM completed for {len(capm_results)} tokens using {actual_days} trading days")
    return capm_results

# ============================================================================
# ✅ MAIN ENTRY POINT - UPDATED TO USE FLEXIBLE APPROACH
# ============================================================================

def apply_capm_filter(tokens, market_index, price_data, start_date, end_date, save_summary=True):
    """
    MAIN ENTRY POINT - Now uses flexible lookback by default
    """
    target_lookback = CONFIG.get('window', 252)
    return apply_capm_filter_flexible(
        tokens, market_index, price_data, start_date, end_date, 
        target_lookback, save_summary
    )

# ============================================================================
# ✅ COMPATIBILITY WRAPPER FOR EXISTING CODE
# ============================================================================

def capm_filter(token_df: pd.DataFrame, index_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float], Optional[torch.Tensor], Optional[np.ndarray]]:
    """
    Compatibility wrapper - calls the new flexible function
    """
    return capm_filter_flexible(token_df, index_df, CONFIG.get('window', 252))

# ============================================================================
# ✅ HELPER FUNCTION FOR SAFE LENGTH CHECKING
# ============================================================================

def safe_len(obj: Any) -> int:
    """
    Safely get the length of an object, handling None and other edge cases
    """
    if obj is None:
        return 0
    try:
        return len(obj)
    except (TypeError, AttributeError):
        return 0

# ============================================================================
# ✅ HELPER FUNCTION FOR SAFE ARRAY CONVERSION
# ============================================================================

def to_numpy_array(data: Any) -> np.ndarray:
    """
    Safely convert any data to numpy array, handling pandas ExtensionArray
    """
    if data is None:
        return np.array([])
    if isinstance(data, np.ndarray):
        return data
    try:
        return np.asarray(data)
    except Exception as e:
        logger.warning(f"Failed to convert data to numpy array: {e}")
        return np.array([])

# ============================================================================
# ✅ GPU-OPTIMIZED VISUALIZATION (Optional - only if needed)
# ============================================================================

def plot_capm_results(capm_results: dict, save_path=None):
    """Plot CAPM scatter plots (CPU operation - only for final visualization)"""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import math
    
    save_path = Path(save_path) if save_path else Path(CONFIG['results_dir']) / "CAPM_ScatterPlots.png"
    tokens = list(capm_results.keys())
    
    if not tokens:
        logger.warning("No CAPM results to plot.")
        return
    
    cols = 5
    rows = math.ceil(len(tokens) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()
    
    for idx, token in enumerate(tokens):
        try:
            data = capm_results[token]
            
            # Extract data (convert to numpy if needed)
            X = to_numpy_array(data.get('ret_index'))
            y = to_numpy_array(data.get('ret_token'))
            beta = data.get('beta', 0)
            alpha = data.get('alpha', 0)
            
            # ✅ FIX: Safe length checking
            if safe_len(X) == 0 or safe_len(y) == 0:
                logger.warning(f"Skipping plot for {token}: no return data")
                axes[idx].axis('off')
                continue
            
            axes[idx].scatter(X, y, label='Returns', alpha=0.4, s=10)
            axes[idx].plot(X, alpha + beta * X, color='red', 
                          label=f"α={alpha:.4f}, β={beta:.4f}")
            axes[idx].set_title(f"{token} vs Index")
            axes[idx].set_xlabel("Index Return")
            axes[idx].set_ylabel("Token Return")
            axes[idx].legend()
            axes[idx].grid(True)
        except Exception as e:
            logger.warning(f"Skipping plot for {token} due to error: {e}")
            axes[idx].axis('off')
    
    for idx in range(len(tokens), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"CAPM scatter plot saved to {save_path}")


def plot_amfdcca(tokens, capm_results: dict, data_dir, delta_H_matrix, 
                 delta_alpha_matrix, save_dir, save_path="Return_and_Idiosyncratic.png"):
    """
    Plot returns and CAPM residuals.
    ✅ CPU operation - only for final visualization
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    try:
        save_dir = Path(save_dir)
        n = len(tokens)
        fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(14, 6 * n))
        if n == 1:
            axes = np.array([axes])

        for i, token in enumerate(tokens):
            if token not in capm_results:
                axes[i, 0].axis('off')
                axes[i, 1].axis('off')
                continue

            data = capm_results[token]

            # Extract data as numpy arrays using safe conversion
            ret_token = to_numpy_array(data.get('ret_token'))
            ret_index = to_numpy_array(data.get('ret_index'))
            residuals = to_numpy_array(data.get('residuals'))

            # ✅ FIX: Safe length checking
            if safe_len(ret_token) == 0 or safe_len(ret_index) == 0 or safe_len(residuals) == 0:
                axes[i, 0].axis('off')
                axes[i, 1].axis('off')
                continue

            # Get index
            date_index = data.get('date_index')
            if date_index is not None and safe_len(date_index) > 0:
                df_index = np.arange(safe_len(ret_index))
            else:
                df_index = np.arange(safe_len(ret_index))

            # Plot Returns
            ax_ret = axes[i, 0]
            ax_ret.plot(df_index, ret_token, label=f"{token} Return", color='blue')
            ax_ret.plot(df_index, ret_index, label=f"{CONFIG['market_index']} Return", color='orange')
            ax_ret.set_title(f"{token} and {CONFIG['market_index']} Returns")
            ax_ret.set_ylabel("Return")

            min_ret = min(ret_token.min(), ret_index.min())
            max_ret = max(ret_token.max(), ret_index.max())
            margin = 0.1 * (max_ret - min_ret)
            ax_ret.set_ylim(min_ret - margin, max_ret + margin)
            ax_ret.legend()
            ax_ret.grid(True)

            # Plot Residuals
            ax_idio = axes[i, 1]
            ax_idio.plot(df_index, residuals, label="Idiosyncratic Premium", color='green')
            ax_idio.axhline(0, linestyle='--', color='red')
            ax_idio.set_title(f"{token} Idiosyncratic Premium")

            min_resid, max_resid = residuals.min(), residuals.max()
            margin_resid = 0.1 * (max_resid - min_resid)
            ax_idio.set_ylim(min_resid - margin_resid, max_resid + margin_resid)
            ax_idio.set_ylabel("Residual")
            ax_idio.legend()
            ax_idio.grid(True)

        plt.tight_layout()
        save_path = Path(save_dir) / save_path
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        logger.info(f"A-MFDCCA plot saved to {save_path}")

    except Exception as e:
        logger.error(f"Error in plot_amfdcca: {e}")
        plt.close('all')