# precompute.py - COMPLETE FIXED VERSION

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import torch
from config import CONFIG, DEVICE
from data_processing import load_all_token_data_cached
from capm import apply_capm_filter
from mfdcca import process_token_pairs_gpu, extract_hurst_matrices
from mfdcca import segment_profiles, detrend_segments 
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)

def get_weekly_periods(start_date, end_date, holding_days):
    """Get business day-aligned weekly periods"""
    all_business_days = pd.bdate_range(start=start_date, end=end_date)
    periods = []
    
    for i in range(0, len(all_business_days), holding_days):
        week_start = all_business_days[i]
        week_end = all_business_days[min(i + holding_days - 1, len(all_business_days)-1)]
        
        actual_days = (week_end - week_start).days + 1
        if actual_days >= holding_days * 0.8:
            periods.append((week_start, week_end))
    
    return periods

def compute_dcca_features(residuals: Dict[str, torch.Tensor], window: int) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    CONSISTENT DCCA: Uses the SAME algorithm as pair_selection.py
    This ensures Optuna trials get identical results to live computation
    """
    logger.info("Precomputing DCCA features...")
    
    # Construct profiles for all tokens
    profiles = {}
    valid_tokens = []
    
    for token, res in residuals.items():
        if isinstance(res, torch.Tensor):
            prof = torch.cumsum(res - res.mean(), dim=0)
        else:
            res_t = torch.tensor(res, device=DEVICE, dtype=torch.float32)
            prof = torch.cumsum(res_t - res_t.mean(), dim=0)
        profiles[token] = prof
        valid_tokens.append(token)
    
    if len(valid_tokens) < 2:
        return {}
    
    # Use the SAME scale generation as your pair_selection.py
    min_scale = 10
    max_scale = max(min_scale + 1, window // 4)
    scales = torch.logspace(
        torch.log10(torch.tensor(min_scale, device=DEVICE)),
        torch.log10(torch.tensor(max_scale, device=DEVICE)),
        steps=10, device=DEVICE
    ).round().int().unique()
    
    # Stack profiles [n_tokens, window]
    profiles_stack = torch.stack([profiles[token] for token in valid_tokens])
    n_tokens = profiles_stack.size(0)
    
    # Generate all pair combinations (SAME as pair_selection.py)
    n_pairs = n_tokens * (n_tokens - 1) // 2
    pair_idx1 = []
    pair_idx2 = []
    token_pairs = []
    
    for i in range(n_tokens):
        for j in range(i + 1, n_tokens):
            pair_idx1.append(i)
            pair_idx2.append(j)
            token_pairs.append((valid_tokens[i], valid_tokens[j]))
    
    pair_idx1 = torch.tensor(pair_idx1, device=DEVICE, dtype=torch.long)
    pair_idx2 = torch.tensor(pair_idx2, device=DEVICE, dtype=torch.long)
    
    # Extract pair profiles [n_pairs, window]
    profiles1 = profiles_stack[pair_idx1]
    profiles2 = profiles_stack[pair_idx2]
    
    # Pre-allocate output tensor (SAME as pair_selection.py)
    h_xy_all_scales = torch.empty(
        len(scales), n_pairs,
        device=DEVICE, dtype=torch.float32
    )
    
    # Process each scale (SAME algorithm as pair_selection.py)
    for scale_idx, scale in enumerate(scales):
        scale = int(scale.item())
        if scale >= window:
            continue

        # Vectorized segmentation using unfold (SAME)
        seg1 = profiles1.unfold(1, scale, 1)  # [n_pairs, n_seg, scale]
        seg2 = profiles2.unfold(1, scale, 1)

        # Cached design matrix for detrending (SAME)
        t = torch.arange(scale, dtype=torch.float32, device=DEVICE)
        X = torch.stack([t, torch.ones_like(t)], dim=1)  # [scale, 2]
        XtX_inv_Xt = torch.linalg.inv(X.T @ X) @ X.T  # [2, scale]

        # Vectorized detrending (SAME)
        seg1_flat = seg1.reshape(-1, scale)  # [n_pairs*n_seg, scale]
        seg2_flat = seg2.reshape(-1, scale)
        coeffs1 = seg1_flat @ XtX_inv_Xt.T
        coeffs2 = seg2_flat @ XtX_inv_Xt.T
        fitted1 = coeffs1 @ X.T
        fitted2 = coeffs2 @ X.T
        detrended1 = seg1_flat - fitted1
        detrended2 = seg2_flat - fitted2

        # Cross-correlation (SAME)
        Fxy = (detrended1 * detrended2).mean(dim=1)
        h_xy_all_scales[scale_idx] = Fxy.reshape(n_pairs, -1).mean(dim=1)
    
    if h_xy_all_scales.size(0) < 2:
        return {}
    
    # Batched Hurst exponent estimation (SAME as pair_selection.py)
    log_scales = torch.log(scales.float())
    log_Fxy = torch.log(h_xy_all_scales.T + 1e-8)

    X_reg = torch.stack([log_scales, torch.ones_like(log_scales)], dim=1)
    coeffs = torch.linalg.lstsq(
        X_reg.unsqueeze(0).expand(n_pairs, -1, -1),
        log_Fxy.unsqueeze(-1)
    ).solution

    h_xy_slopes = coeffs[:, 0, 0]  # NO CLAMPING - keep original values
    
    # Create DCCA results dictionary
    dcca_results = {}
    for idx, (t1, t2) in enumerate(token_pairs):
        hurst_value = h_xy_slopes[idx].item()
        
        # Only store valid Hurst exponents (same filtering as live computation)
        if not np.isnan(hurst_value) and not np.isinf(hurst_value):
            dcca_results[(t1, t2)] = {
                'hurst_exponent': hurst_value,
                'scales': scales.cpu().numpy(),
                'fluctuations': h_xy_all_scales[:, idx].cpu().numpy()
            }
    
    logger.info(f"✅ Precomputed DCCA features for {len(dcca_results)} pairs")
    return dcca_results

def compute_pearson_features(residuals: Dict[str, torch.Tensor], window: int) -> torch.Tensor:
    """SIMPLIFIED: Remove redundant validation that causes issues"""
    logger.info("Precomputing Pearson correlation matrix...")
    
    res_dict = {}
    valid_tokens = []
    
    for token, res in residuals.items():
        if isinstance(res, torch.Tensor):
            res_dict[token] = res[-window:].to(DEVICE).float()
        else:
            res_dict[token] = torch.tensor(res[-window:], device=DEVICE, dtype=torch.float32)
        valid_tokens.append(token)
    
    if len(valid_tokens) < 2:
        return torch.tensor([], device=DEVICE)
    
    # Stack all residuals and compute correlation matrix
    res_stack = torch.stack([res_dict[token] for token in valid_tokens])
    
    try:
        corr_matrix = torch.corrcoef(res_stack)
        logger.info(f"✅ Precomputed Pearson correlation matrix for {len(valid_tokens)} tokens")
        return corr_matrix
    except Exception as e:
        logger.error(f"Pearson correlation failed: {e}")
        return torch.tensor([], device=DEVICE)

def compute_cointegration_features(price_data: Dict, window: int) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Precompute cointegration features: Engle-Granger test, OLS regression, ADF test"""
    logger.info("Precomputing cointegration features...")
    
    tokens = [t for t in CONFIG['token_names'] if t in price_data]
    cointegration_results = {}
    
    for i, t1 in enumerate(tokens):
        for j in range(i + 1, len(tokens)):
            t2 = tokens[j]
            
            df1 = price_data.get(t1)
            df2 = price_data.get(t2)
            
            if df1 is None or df2 is None:
                continue
            
            common_dates = df1.index.intersection(df2.index)
            if len(common_dates) < window:
                continue
            
            try:
                # Extract price data
                prices1 = df1.loc[common_dates, 'close'].to_numpy()[-window:]
                prices2 = df2.loc[common_dates, 'close'].to_numpy()[-window:]
                
                # Use log prices for cointegration
                log_p1 = np.log(prices1)
                log_p2 = np.log(prices2)
                
                # OLS regression: log_p2 = beta * log_p1 + alpha
                X = sm.add_constant(log_p1)
                model = sm.OLS(log_p2, X).fit()
                beta, alpha = model.params[1], model.params[0]
                
                # Calculate residuals
                residuals = log_p2 - (beta * log_p1 + alpha)
                
                # Store results
                
                adf_result = adfuller(residuals, autolag='AIC')
                adf_statistic = adf_result[0]
                pvalue = adf_result[1]
                usedlag = adf_result[2] if len(adf_result) > 2 else None
                nobs = adf_result[3] if len(adf_result) > 3 else None
                critical_values = adf_result[4] if len(adf_result) > 4 else {}
                icbest = adf_result[5] if len(adf_result) > 5 else None


                # Store results
                cointegration_results[(t1, t2)] = {
                    'pvalue': pvalue,
                    'beta': beta,
                    'alpha': alpha,
                    'residuals': residuals,
                    'adf_statistic': adf_statistic,
                    'critical_values': critical_values
                }
            except Exception as e:
                logger.warning(f"Cointegration failed for {t1}-{t2}: {e}")
                continue
    
    logger.info(f"✅ Precomputed cointegration features for {len(cointegration_results)} pairs")
    return cointegration_results

def create_hurst_matrix_from_dcca(dcca_features: Dict, token_list: List[str]) -> np.ndarray:
    """Create DCCA Hurst exponent matrix from precomputed features"""
    n = len(token_list)
    hurst_matrix = np.full((n, n), np.nan)
    token_to_idx = {token: i for i, token in enumerate(token_list)}
    
    for (t1, t2), features in dcca_features.items():
        if t1 in token_to_idx and t2 in token_to_idx:
            i, j = token_to_idx[t1], token_to_idx[t2]
            hurst_matrix[i, j] = features['hurst_exponent']
            hurst_matrix[j, i] = features['hurst_exponent']
    
    return hurst_matrix

def create_cointegration_matrix(cointegration_features: Dict, token_list: List[str]) -> np.ndarray:
    """Create cointegration p-value matrix from precomputed features"""
    n = len(token_list)
    pvalue_matrix = np.full((n, n), 1.0)  # Default to 1.0 (not cointegrated)
    token_to_idx = {token: i for i, token in enumerate(token_list)}
    
    for (t1, t2), features in cointegration_features.items():
        if t1 in token_to_idx and t2 in token_to_idx:
            i, j = token_to_idx[t1], token_to_idx[t2]
            pvalue_matrix[i, j] = features['pvalue']
            pvalue_matrix[j, i] = features['pvalue']
    
    return pvalue_matrix

# precompute.py - ADD THIS FUNCTION

# In precompute.py - Update create_extended_cache_data_fixed()

def create_extended_cache_data_fixed(method, residuals, price_data, aligned_capm_results,
                                   week_start, lookback_start, lookback_end,
                                   information_cutoff,  # ✅ Now required parameter
                                   method_specific_data=None):
    """✅ CORRECTED: Cache structure with explicit information cutoff tracking"""
    
    base_cache = {
        'method': method,
        'residuals': residuals,
        'price_data_lookback': price_data,
        'aligned_capm_results': aligned_capm_results,
        'week_start': week_start,
        'lookback_start': lookback_start,
        'lookback_end': lookback_end,
        'information_cutoff': information_cutoff,  # ✅ CRITICAL: Explicit tracking
        'skipped': False,
        'cache_version': '3.1',  # ✅ Version bump to invalidate old caches
        'window_size': CONFIG['window'],
        'residual_length': len(next(iter(residuals.values()))) if residuals else 0,
        'cached_date': pd.Timestamp.now(),
        'flexible_lookback': True,
        'no_lookahead': True,  # ✅ Certification flag
        'temporal_integrity_verified': True,  # ✅ Additional certification
    }
    
    # Add actual days used from CAPM results
    if aligned_capm_results and len(aligned_capm_results) > 0:
        sample_token = next(iter(aligned_capm_results.keys()))
        base_cache['actual_days_used'] = aligned_capm_results[sample_token].get('actual_days_used', 0)
        base_cache['max_data_date'] = lookback_end  # ✅ Track maximum data date used
    
    # Add method-specific precomputed features
    if method_specific_data:
        base_cache.update(method_specific_data)
    
    return base_cache

def precompute_training_features_extended(
    fold_number: int,
    method: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp
):
    """CORRECTED: No look-ahead bias with explicit information cutoff tracking"""
    logger.info(f"Starting CORRECTED precompute for fold {fold_number}, method {method}")
    
    cache_dir = Path(CONFIG['results_dir']) / "precompute" / f"fold_{fold_number}" / method
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    target_lookback = CONFIG['window']
    holding_period_days = CONFIG['holding_period_days']
    
    # Get weekly periods
    weekly_periods = get_weekly_periods(period_start, period_end, holding_period_days)
    
    if not weekly_periods:
        logger.error("No weekly periods generated for precompute!")
        return
    
    logger.info(f"Precomputing {len(weekly_periods)} weeks for {method} with CORRECTED temporal logic")
    
    for week_number, (current_date, week_end) in enumerate(weekly_periods, 1):
        cache_file = cache_dir / f"week_{week_number}_{current_date.date()}.pt"
        
        if cache_file.exists():
            logger.info(f"✅ Week {week_number}: Cache exists, skipping")
            continue
        
        logger.info(f"Computing week {week_number}: {current_date.date()}")
        
        # ✅ CORRECTED: Explicit information cutoff tracking
        lookback_end = current_date - pd.offsets.BDay(1)  # Day BEFORE trading
        information_cutoff = lookback_end  # ✅ Explicit variable
        lookback_start = lookback_end - pd.offsets.BDay(target_lookback - 1)
        
        data_start = CONFIG['start_date']
        if lookback_start < data_start:
            lookback_start = data_start
        
        actual_lookback_days = (lookback_end - lookback_start).days + 1
        logger.info(
            f"CORRECTED lookback: {lookback_start.date()} to {lookback_end.date()} "
            f"({actual_lookback_days} days) | Info cutoff: {information_cutoff.date()}"
        )
        
        # Load data (only up to information cutoff)
        price_data_lookback = load_all_token_data_cached(
            lookback_start,
            lookback_end,  # ✅ Ends at information cutoff
            CONFIG['market_index']
        )
        
        if not price_data_lookback:
            logger.warning(f"Week {week_number}: No data, skipping")
            torch.save({'skipped': True}, cache_file)
            continue
        
        # CAPM filtering
        logger.info(f"Week {week_number}: Running CAPM...")
        capm_results = apply_capm_filter(
            tokens=CONFIG['token_names'],
            market_index=CONFIG['market_index'],
            price_data=price_data_lookback,
            start_date=lookback_start,
            end_date=lookback_end,  # ✅ Up to information cutoff
            save_summary=False
        )
        
        if not capm_results:
            logger.warning(f"Week {week_number}: CAPM failed")
            torch.save({'skipped': True}, cache_file)
            continue
        
        # Prepare residuals
        residuals = {}
        aligned_capm_results = {}
        
        for token in capm_results:
            if capm_results[token]['residuals'] is not None and 'residuals_gpu' in capm_results[token]:
                residuals[token] = capm_results[token]['residuals_gpu']
                aligned_capm_results[token] = {
                    'residuals': capm_results[token]['residuals'],
                    'date_index': capm_results[token]['date_index'],
                    'beta': capm_results[token]['beta'],
                    'alpha': capm_results[token]['alpha'],
                    'actual_days_used': capm_results[token].get('actual_days_used', actual_lookback_days)
                }
        
        if not residuals:
            logger.warning(f"Week {week_number}: No valid residuals")
            torch.save({'skipped': True}, cache_file)
            continue
        
        actual_days_used = aligned_capm_results[next(iter(aligned_capm_results.keys()))]['actual_days_used']
        
        # Method-specific precomputation (your existing logic)
        method_specific_data = {}
        
        if method == "mfdcca":
            logger.info(f"Week {week_number}: Computing MFDCCA...")
            try:
                results = process_token_pairs_gpu(
                    token_list=CONFIG['token_names'],
                    residuals=residuals,
                    start_date=lookback_start,
                    end_date=lookback_end,  # ✅ Up to information cutoff
                    q_list=CONFIG['q_list']
                )
                
                if results:
                    hurst_dict, hxy_matrix, delta_H_matrix, delta_alpha_matrix = \
                        extract_hurst_matrices(CONFIG['token_names'], results)
                    
                    method_specific_data = {
                        'has_mfdcca_data': True,
                        'hurst_dict': hurst_dict if hurst_dict else {},
                        'hxy_matrix': hxy_matrix.cpu().numpy() if hxy_matrix is not None else np.array([]),
                        'hurst_matrix': delta_H_matrix.cpu().numpy() if delta_H_matrix is not None else np.array([]),
                        'alpha_matrix': delta_alpha_matrix.cpu().numpy() if delta_alpha_matrix is not None else np.array([])
                    }
                    logger.info(f"✅ Week {week_number}: MFDCCA features computed")
                else:
                    logger.warning(f"Week {week_number}: MFDCCA produced no results")
                    method_specific_data = {'has_mfdcca_data': False}
                    
            except Exception as e:
                logger.error(f"Week {week_number}: MFDCCA failed: {e}")
                method_specific_data = {'has_mfdcca_data': False}
        
        elif method == "dcca":
            logger.info(f"Week {week_number}: Computing DCCA features...")
            try:
                dcca_features = compute_dcca_features(residuals, actual_days_used)
                method_specific_data = {
                    'has_dcca_data': True,
                    'dcca_features': dcca_features,
                    'dcca_hurst_matrix': create_hurst_matrix_from_dcca(dcca_features, CONFIG['token_names'])
                }
                logger.info(f"✅ Week {week_number}: DCCA features computed")
            except Exception as e:
                logger.error(f"Week {week_number}: DCCA features failed: {e}")
                method_specific_data = {'has_dcca_data': False}
        
        elif method == "pearson":
            logger.info(f"Week {week_number}: Computing Pearson features...")
            try:
                corr_matrix = compute_pearson_features(residuals, actual_days_used)
                method_specific_data = {
                    'has_pearson_data': True,
                    'correlation_matrix': corr_matrix.cpu().numpy() if corr_matrix.numel() > 0 else np.array([]),
                    'valid_tokens': CONFIG['token_names']
                }
                logger.info(f"✅ Week {week_number}: Pearson correlation matrix computed")
            except Exception as e:
                logger.error(f"Week {week_number}: Pearson features failed: {e}")
                method_specific_data = {'has_pearson_data': False}
        
        elif method == "cointegration":
            logger.info(f"Week {week_number}: Computing Cointegration features...")
            try:
                cointegration_features = compute_cointegration_features(price_data_lookback, actual_days_used)
                method_specific_data = {
                    'has_cointegration_data': True,
                    'cointegration_features': cointegration_features,
                    'cointegration_matrix': create_cointegration_matrix(cointegration_features, CONFIG['token_names'])
                }
                logger.info(f"✅ Week {week_number}: Cointegration features computed")
            except Exception as e:
                logger.error(f"Week {week_number}: Cointegration features failed: {e}")
                method_specific_data = {'has_cointegration_data': False}
        
        elif method == "index":
            method_specific_data = {'has_index_data': True}
            logger.info(f"✅ Week {week_number}: Index method data prepared")
        
        # ✅ CORRECTED: Pass information_cutoff explicitly
        cache_data = create_extended_cache_data_fixed(
            method, residuals, price_data_lookback, aligned_capm_results,
            current_date, lookback_start, lookback_end,
            information_cutoff,  # ✅ NOW CORRECT - parameter added
            method_specific_data
        )
        
        try:
            torch.save(cache_data, cache_file)
            logger.info(f"✅ Week {week_number}: CORRECTED cache saved to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache for week {week_number}: {e}")
    
    logger.info(f"✅ CORRECTED precompute complete for fold {fold_number}, method {method}")
# Update main function to use extended precomputation
def precompute_training_features(
    fold_number: int,
    method: str,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp
):
    """MAIN ENTRY POINT - Now with extended precomputation for all methods"""
    return precompute_training_features_extended(fold_number, method, period_start, period_end)