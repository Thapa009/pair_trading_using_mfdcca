import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import torch
from config import CONFIG, DEVICE

logger = logging.getLogger(__name__)

# ============================================================================
# PRECOMPUTED VERSIONS (for Optuna trials)
# ============================================================================

def select_pairs_dcca_precomputed(
    symbols: List[str], 
    dcca_features: Dict[Tuple[str, str], Dict[str, Any]],
    pair_hxy_threshold: float
) -> List[Tuple[str, str]]:
    """DCCA pair selection using precomputed features"""
    selected_pairs = []
    
    for (t1, t2), features in dcca_features.items():
        if t1 in symbols and t2 in symbols:
            hurst_exp = features['hurst_exponent']
            if hurst_exp <= pair_hxy_threshold:
                selected_pairs.append(tuple(sorted([t1, t2])))
    
    logger.info(f"✅ Precomputed DCCA: Selected {len(selected_pairs)} pairs (threshold: {pair_hxy_threshold})")
    return selected_pairs

def select_pairs_pearson_precomputed(
    symbols: List[str],
    corr_matrix: np.ndarray,
    token_list: List[str],
    rho_threshold: float
) -> List[Tuple[str, str]]:
    """Pearson pair selection using precomputed correlation matrix"""
    if corr_matrix.size == 0 or corr_matrix.shape[0] != len(token_list):
        logger.warning("Invalid correlation matrix for precomputed Pearson")
        return []
    
    n = len(token_list)
    # FIXED: Use proper numpy triu function
    mask = np.triu_indices(n, k=1)
    corr_values = corr_matrix[mask]
    
    # Create token to index mapping
    token_to_idx = {token: i for i, token in enumerate(token_list)}
    
    selected_pairs = []
    
    for idx in range(len(mask[0])):
        i, j = mask[0][idx], mask[1][idx]
        if abs(corr_values[idx]) > rho_threshold:
            t1, t2 = token_list[i], token_list[j]
            if t1 in symbols and t2 in symbols:
                selected_pairs.append(tuple(sorted([t1, t2])))
    
    logger.info(f"✅ Precomputed Pearson: Selected {len(selected_pairs)} pairs (threshold: {rho_threshold})")
    return selected_pairs

def select_pairs_cointegration_precomputed(
    symbols: List[str],
    cointegration_features: Dict[Tuple[str, str], Dict[str, Any]],
    pval_threshold: float
) -> List[Tuple[str, str]]:
    """Cointegration pair selection using precomputed features"""
    selected_pairs = []
    
    for (t1, t2), features in cointegration_features.items():
        if t1 in symbols and t2 in symbols:
            pvalue = features['pvalue']
            if pvalue < pval_threshold:
                selected_pairs.append(tuple(sorted([t1, t2])))
    
    logger.info(f"✅ Precomputed Cointegration: Selected {len(selected_pairs)} pairs (pval_threshold: {pval_threshold})")
    return selected_pairs

# ============================================================================
# ORIGINAL IMPLEMENTATIONS (for fallback/live computation)
# ============================================================================

from typing import Tuple, List, Dict, Any
import torch
import logging
from config import DEVICE, CONFIG

logger = logging.getLogger(__name__)

def select_pairs_mfdcca(hurst_dict, hxy_matrix, delta_H_matrix, delta_alpha_matrix,
                             symbols, pair_hxy_threshold, threshold_h, threshold_alpha):
    """
    ✅ FIXED MFDCCA pair selection with proper validation
    """
    # ✅ STEP 1: Dynamic q=2 index finding
    q_list = CONFIG['q_list']
    if 2 not in q_list:
        logger.error("q=2 not in q_list - cannot compute Hₓᵧ(2)")
        return []
    
    q2_idx = q_list.index(2)  # Finds position of q=2 in the list
    logger.info(f"Using q=2 at index {q2_idx} in q_list {q_list}")
    
    N = len(symbols)
    selected_pairs = []
    valid_pairs = 0
    invalid_pairs = 0
    
    # ✅ STEP 2: Iterate through all possible pairs
    for i in range(N):
        for j in range(i + 1, N):  # Only upper triangle to avoid duplicates
            
            # ✅ STEP 3: Extract MFDCCA features for this pair
            hxy_2 = hxy_matrix[i, j, q2_idx]      # Hₓᵧ(2) - mean reversion strength
            delta_H = delta_H_matrix[i, j]         # ΔH - multifractal consistency  
            delta_alpha = delta_alpha_matrix[i, j] # Δα - complexity measure
            
            # ✅ STEP 4: CRITICAL - Validate numerical values
            if (torch.isnan(hxy_2) or torch.isnan(delta_H) or torch.isnan(delta_alpha)):
                invalid_pairs += 1
                logger.debug(f"❌ Pair {symbols[i]}-{symbols[j]}: NaN values detected")
                continue
            
            # ✅ STEP 5: Apply your research selection criteria
            if (hxy_2 < pair_hxy_threshold and     # Mean-reverting at large scales
                delta_H < threshold_h and          # Consistent bull/bear behavior
                delta_alpha < threshold_alpha):    # Simple multifractal structure
                
                pair = tuple(sorted([symbols[i], symbols[j]]))
                selected_pairs.append(pair)
                valid_pairs += 1
                
                logger.debug(f"✅ Selected {symbols[i]}-{symbols[j]}: "
                           f"H(2)={hxy_2:.3f}, ΔH={delta_H:.3f}, Δα={delta_alpha:.3f}")
            else:
                invalid_pairs += 1
                logger.debug(f"❌ Rejected {symbols[i]}-{symbols[j]}: "
                           f"H(2)={hxy_2:.3f}, ΔH={delta_H:.3f}, Δα={delta_alpha:.3f}")
    
    # ✅ STEP 6: Final summary
    logger.info(f"✅ MFDCCA pair selection: {valid_pairs} selected, {invalid_pairs} rejected "
               f"({len(selected_pairs)} total pairs)")
    
    return selected_pairs
    
def select_pairs_dcca(symbols: List[str], residuals: dict, window: int, pair_hxy_threshold: float, num_scales: int = 10) -> List[Tuple[str, str]]:
    """Fallback: Live DCCA computation"""
    logger.warning("Using live DCCA computation (precomputed features not available)")
    
    # Stack profiles [n_tokens, window]
    profiles = []
    valid_symbols = []
    for sym in symbols:
        if sym not in residuals:
            continue
        res = residuals[sym]
        if isinstance(res, torch.Tensor):
            prof = torch.cumsum(res - res.mean(), dim=0)
        else:
            res_t = torch.tensor(res.values, device=DEVICE, dtype=torch.float32)
            prof = torch.cumsum(res_t - res_t.mean(), dim=0)
        profiles.append(prof)
        valid_symbols.append(sym)

    if len(profiles) < 2:
        return []

    profiles_stack = torch.stack(profiles)
    n_tokens = profiles_stack.size(0)

    # Generate all pair combinations
    n_pairs = n_tokens * (n_tokens - 1) // 2
    pair_idx1 = []
    pair_idx2 = []
    for i in range(n_tokens):
        for j in range(i + 1, n_tokens):
            pair_idx1.append(i)
            pair_idx2.append(j)

    pair_idx1 = torch.tensor(pair_idx1, device=DEVICE, dtype=torch.long)
    pair_idx2 = torch.tensor(pair_idx2, device=DEVICE, dtype=torch.long)

    profiles1 = profiles_stack[pair_idx1]
    profiles2 = profiles_stack[pair_idx2]

    # Generate scales
    min_scale = 10
    max_scale = max(min_scale + 1, window // 4)
    scales = torch.logspace(
        torch.log10(torch.tensor(min_scale, device=DEVICE)),
        torch.log10(torch.tensor(max_scale, device=DEVICE)),
        steps=num_scales, device=DEVICE
    ).round().int().unique()

    # Pre-allocate output tensor
    h_xy_all_scales = torch.empty(
        len(scales), n_pairs,
        device=DEVICE, dtype=torch.float32
    )

    for scale_idx, scale in enumerate(scales):
        scale = int(scale.item())
        if scale >= window:
            continue

        # Use unfold for vectorized segmentation
        seg1 = profiles1.unfold(1, scale, 1)
        seg2 = profiles2.unfold(1, scale, 1)

        # Cached design matrix for detrending
        t = torch.arange(scale, dtype=torch.float32, device=DEVICE)
        X = torch.stack([t, torch.ones_like(t)], dim=1)
        XtX_inv_Xt = torch.linalg.inv(X.T @ X) @ X.T

        # Vectorized detrending
        seg1_flat = seg1.reshape(-1, scale)
        seg2_flat = seg2.reshape(-1, scale)
        coeffs1 = seg1_flat @ XtX_inv_Xt.T
        coeffs2 = seg2_flat @ XtX_inv_Xt.T
        fitted1 = coeffs1 @ X.T
        fitted2 = coeffs2 @ X.T
        detrended1 = seg1_flat - fitted1
        detrended2 = seg2_flat - fitted2

        # Cross-correlation and store
        Fxy = (detrended1 * detrended2).mean(dim=1)
        h_xy_all_scales[scale_idx] = Fxy.reshape(n_pairs, -1).mean(dim=1)

    if h_xy_all_scales.size(0) < 2:
        return []

    # Single batched regression
    log_scales = torch.log(scales.float())
    log_Fxy = torch.log(h_xy_all_scales.T + 1e-8)

    X_reg = torch.stack([log_scales, torch.ones_like(log_scales)], dim=1)
    coeffs = torch.linalg.lstsq(
        X_reg.unsqueeze(0).expand(n_pairs, -1, -1),
        log_Fxy.unsqueeze(-1)
    ).solution

    h_xy_slopes = coeffs[:, 0, 0].clamp(0, 1)

    # GPU filtering
    mask = h_xy_slopes <= pair_hxy_threshold
    selected_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)

    # Minimal CPU transfer
    selected_pairs = []
    indices_cpu = selected_idx.cpu().numpy()
    for idx in indices_cpu:
        i = int(pair_idx1[idx].item())
        j = int(pair_idx2[idx].item())
        selected_pairs.append(tuple(sorted([valid_symbols[i], valid_symbols[j]])))

    logger.info(f"✅ Live DCCA selected {len(selected_pairs)} pairs")
    return selected_pairs

def select_pairs_pearson(symbols: List[str], residuals: dict, window: int, rho_threshold: float) -> List[Tuple[str, str]]:
    """Fallback: Live Pearson computation"""
    logger.warning("Using live Pearson computation (precomputed features not available)")
    
    # Convert all residuals to GPU ONCE
    res_dict = {}
    for sym in symbols:
        res = residuals.get(sym)
        if res is None:
            continue
        if isinstance(res, torch.Tensor):
            res_dict[sym] = res[-window:].to(DEVICE).float()
        elif hasattr(res, 'values'):
            res_dict[sym] = torch.from_numpy(res[-window:].values).float().to(DEVICE)
        else:
            res_dict[sym] = torch.tensor(res[-window:], device=DEVICE, dtype=torch.float32)

    valid_symbols = list(res_dict.keys())
    if len(valid_symbols) < 2:
        return []

    # Stack all residuals [n_tokens, window]
    res_stack = torch.stack([res_dict[s] for s in valid_symbols])

    # Single batched correlation matrix [n_tokens, n_tokens]
    corr_matrix = torch.corrcoef(res_stack)

    # Extract upper triangle (pairs) on GPU
    n = len(valid_symbols)
    mask = torch.triu(torch.ones(n, n, device=DEVICE), diagonal=1).bool()
    corr_values = corr_matrix[mask]

    # GPU filtering
    selected_mask = torch.abs(corr_values) > rho_threshold
    pair_idx = torch.nonzero(mask, as_tuple=False)
    selected_pairs_idx = pair_idx[selected_mask]

    # Minimal CPU transfer at end
    pairs_cpu = selected_pairs_idx.cpu().numpy()
    selected_pairs = [
        tuple(sorted([valid_symbols[i], valid_symbols[j]]))
        for i, j in pairs_cpu
    ]

    logger.info(f"✅ Live Pearson selected {len(selected_pairs)} pairs")
    return selected_pairs

def select_pairs_cointegration(
    symbols: List[str],
    data: dict,
    window: int,
    pval_threshold: float,
    use_log: bool = True
) -> List[Tuple[str, str]]:
    """Fallback: Live Cointegration computation"""
    logger.warning("Using live Cointegration computation (precomputed features not available)")
    
    selected_pairs = []
    pair_metrics = []
    
    for i, t1 in enumerate(symbols):
        for j in range(i + 1, len(symbols)):
            t2 = symbols[j]
            
            df1 = data.get(t1)
            df2 = data.get(t2)
            
            if df1 is None or df2 is None:
                continue
            
            if not hasattr(df1, 'index') or not hasattr(df2, 'index'):
                logging.warning(f"Invalid data structure for {t1} or {t2}")
                continue
            
            common_dates = df1.index.intersection(df2.index)
            if len(common_dates) < window:
                continue
            
            try:
                if hasattr(df1, 'loc'):
                    prices1_np = df1.loc[common_dates, 'close'].to_numpy()[-window:]
                    prices2_np = df2.loc[common_dates, 'close'].to_numpy()[-window:]
                    prices1 = torch.from_numpy(prices1_np).float().to(DEVICE)
                    prices2 = torch.from_numpy(prices2_np).float().to(DEVICE)
                else:
                    prices1_np = df1.cpu().numpy()[-window:] if hasattr(df1, 'cpu') else np.array(df1)[-window:]
                    prices2_np = df2.cpu().numpy()[-window:] if hasattr(df2, 'cpu') else np.array(df2)[-window:]
                    prices1 = torch.from_numpy(prices1_np).float().to(DEVICE)
                    prices2 = torch.from_numpy(prices2_np).float().to(DEVICE)
                
                if torch.isnan(prices1).any() or torch.isnan(prices2).any():
                    continue
                
                if use_log:
                    prices1 = torch.log(prices1)
                    prices2 = torch.log(prices2)
                
                x_ols = torch.cat([prices1.unsqueeze(1), torch.ones_like(prices1).unsqueeze(1)], dim=1)
                y_ols = prices2.unsqueeze(1)
                try:
                    beta = torch.linalg.lstsq(x_ols, y_ols, driver='gels').solution
                    residuals = prices2 - (beta[0, 0] * prices1 + beta[1, 0])
                    residuals_cpu = residuals.cpu().numpy()
                except RuntimeError:
                    logging.warning(f"OLS failed for {t1}-{t2}")
                    continue
                
                if len(residuals_cpu) < 10:
                    continue
                
                from statsmodels.tsa.stattools import adfuller
                adf_result = adfuller(residuals_cpu, autolag='AIC')
                pvalue = adf_result[1]
                logging.debug(f"Engle-Granger p-value for {t1}-{t2}: {pvalue:.3f}")
                
                if pvalue < pval_threshold:
                    selected_pairs.append(tuple(sorted([t1, t2])))
                    pair_metrics.append((t1, t2, pvalue))
                    
            except Exception as e:
                logging.warning(f"Skipping pair ({t1}, {t2}): {e}")
                continue
    
    logging.info(f"✅ Live Cointegration selected {len(selected_pairs)} pairs")
    return selected_pairs# precompute.py - COMPLETE UPDATED VERSION
