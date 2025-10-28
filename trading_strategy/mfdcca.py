# mfdcca.py - Perfect Stepwise MFDCCA Implementation
"""
Multifractal Detrended Cross-Correlation Analysis (MFDCCA)
A complete, optimized GPU implementation for pair trading research.

Algorithm Steps:
1. Profile Construction: Convert residuals to cumulative profiles
2. Segmentation: Divide profiles into forward/backward segments
3. Detrending: Remove local trends via least-squares polynomial fitting
4. Cross-Correlation: Compute detrended covariance (Fxy) per segment
5. Fluctuation Function: Aggregate Fxy across segments for each q-order
6. Scaling Analysis: Estimate generalized Hurst exponents via log-log regression
7. Multifractal Spectrum: Compute singularity spectrum (α, f(α))
"""

import pandas as pd
import torch
import numpy as np
import logging
from config import CONFIG, DEVICE

logger = logging.getLogger(__name__)


# ============================================================================
# STEP 1: PROFILE CONSTRUCTION
# ============================================================================
def construct_profile(series: torch.Tensor) -> torch.Tensor:
    """
    Convert time series to cumulative profile (integration).
    
    Mathematical Definition:
        Y(i) = Σ[x(k) - <x>] for k=1 to i
    
    Args:
        series: Input time series [T] on GPU
        
    Returns:
        Profile tensor [T] on GPU
        
    Reference:
        Podobnik & Stanley (2008), Phys. Rev. Lett.
    """
    if torch.isnan(series).any():
        logger.error("Series contains NaN values")
        return torch.tensor([], device=DEVICE)
    
    # Subtract mean and compute cumulative sum
    profile = torch.cumsum(series - series.mean(), dim=0)
    return profile


# ============================================================================
# STEP 2: SEGMENTATION
# ============================================================================
def segment_profiles(profile1: torch.Tensor, profile2: torch.Tensor, scale: int) -> tuple[torch.Tensor, torch.Tensor]:
    N = profile1.size(0)
    if scale < 4 or N < scale:
        return torch.empty(0, scale, device=DEVICE), torch.empty(0, scale, device=DEVICE)
    
    num_segments = N // scale
    total_segments = 2 * num_segments
    
    # ✅ Pre-allocate
    segments1 = torch.empty(total_segments, scale, device=DEVICE)
    segments2 = torch.empty(total_segments, scale, device=DEVICE)
    
    # Forward segmentation
    segments1[:num_segments] = profile1[:num_segments * scale].view(num_segments, scale)
    segments2[:num_segments] = profile2[:num_segments * scale].view(num_segments, scale)
    
    # Backward segmentation
    segments1[num_segments:] = profile1[-(num_segments * scale):].view(num_segments, scale)
    segments2[num_segments:] = profile2[-(num_segments * scale):].view(num_segments, scale)
    
    return segments1, segments2


# ============================================================================
# STEP 3: DETRENDING (Polynomial Fitting)
# ============================================================================
# Global cache for design matrices
_DESIGN_MATRIX_CACHE = {}

def get_design_matrix(scale: int, device: torch.device):
    """Cached design matrix computation"""
    if scale not in _DESIGN_MATRIX_CACHE:
        t = torch.arange(scale, dtype=torch.float32, device=device)
        X = torch.stack([t, torch.ones(scale, device=device)], dim=1)
        XtX_inv_Xt = torch.linalg.inv(X.T @ X) @ X.T  # Pre-multiply
        _DESIGN_MATRIX_CACHE[scale] = XtX_inv_Xt
    return _DESIGN_MATRIX_CACHE[scale]

def detrend_segments(segments1, segments2, order=1):
    """GPU-optimized detrending with matrix caching"""
    num_segments, scale = segments1.shape
    
    # ✅ Use cached pre-inverted matrix
    XtX_inv_Xt = get_design_matrix(scale, segments1.device)
    
    # ✅ Single batched operation
    coeffs1 = segments1 @ XtX_inv_Xt.T  # [num_segments, 2]
    coeffs2 = segments2 @ XtX_inv_Xt.T
    
    # ✅ Efficient broadcasting
    t = torch.arange(scale, device=segments1.device)
    fitted1 = coeffs1[:, 0:1] * t + coeffs1[:, 1:2]
    fitted2 = coeffs2[:, 0:1] * t + coeffs2[:, 1:2]
    
    return segments1 - fitted1, segments2 - fitted2, coeffs1[:, 0]# ============================================================================
# STEP 4: CROSS-CORRELATION COMPUTATION
# ============================================================================
def compute_cross_correlation(residuals1: torch.Tensor, 
                               residuals2: torch.Tensor) -> torch.Tensor:
    """
    Compute detrended cross-correlation (covariance) for each segment.
    
    Mathematical Definition:
        Fxy(v, s) = (1/s) Σ[ε₁(t) · ε₂(t)] for t in segment v
    
    Args:
        residuals1: Detrended segments from series 1 [num_segments, scale]
        residuals2: Detrended segments from series 2 [num_segments, scale]
        
    Returns:
        Cross-correlations [num_segments]
        
    Reference:
        Podobnik & Stanley (2008) - DCCA coefficient definition
    """
    # Element-wise multiplication and mean over time dimension
    Fxy = (residuals1 * residuals2).mean(dim=1)  # [num_segments]
    
    return Fxy


# ============================================================================
# STEP 5: FLUCTUATION FUNCTION (q-order Aggregation)
# ============================================================================
def aggregate_fluctuations(Fxy: torch.Tensor, q: float, epsilon: float = 1e-8) -> torch.Tensor:
    """
    ✅ FIXED: Proper numerical stability for fluctuation aggregation
    """
    if len(Fxy) == 0:
        return torch.tensor(float('nan'), device=DEVICE)
    
    # ✅ Keep original signs for proper multifractal analysis
    Fxy_signed = Fxy
    
    # ✅ Better filtering of near-zero values
    abs_Fxy = torch.abs(Fxy_signed)
    valid_mask = abs_Fxy > epsilon
    
    # ✅ Stricter threshold: need at least 60% valid segments
    if valid_mask.sum() < max(5, len(Fxy) * 0.5):
        return torch.tensor(float('nan'), device=DEVICE)
    
    # ✅ Only replace truly problematic values, keep good ones intact
    Fxy_filtered = torch.where(valid_mask, Fxy_signed, torch.nan)
    Fxy_valid = Fxy_signed[valid_mask]
    
    if abs(q) < 1e-6:  # q ≈ 0 (special case)
        # ✅ Geometric mean for q=0 (more robust)
        log_values = torch.log(torch.abs(Fxy_valid) + epsilon)
        Fq = torch.exp(torch.mean(log_values))
    else:
        try:
            # ✅ Proper handling of negative Fxy values
            # Use signed power: sign(x) * |x|^q
            signed_power = torch.sign(Fxy_valid) * torch.abs(Fxy_valid).pow(q)
            mean_power = torch.mean(signed_power)
            
            # ✅ Avoid division by zero and extreme values
            if torch.abs(mean_power) < 1e-12:
                return torch.tensor(float('nan'), device=DEVICE)
                
            Fq = mean_power.pow(1.0 / q)
            
            if torch.isnan(Fq) or torch.isinf(Fq) or Fq < 1e-12:
                return torch.tensor(float('nan'), device=DEVICE)
                
        except Exception as e:
            return torch.tensor(float('nan'), device=DEVICE)
    
    return Fq

# ============================================================================
# STEP 6: SCALING ANALYSIS (Hurst Exponent Estimation)
# ============================================================================
def estimate_hurst_exponent(Fq_values: torch.Tensor, scales: torch.Tensor) -> float:
    """
    ✅ FIXED: Stricter validation for reliable Hurst exponent estimation
    """
    # Filter out NaN values
    valid = ~torch.isnan(Fq_values)
    
    # ✅ FIX: Stricter validation criteria
    min_valid_scales = max(8, int(len(scales) * 0.7))  # Need 70% valid scales
    
    if valid.sum() < min_valid_scales:
        return float('nan')
    
    log_scales = torch.log(scales[valid])
    log_Fq = torch.log(Fq_values[valid])
    
    # ✅ Additional quality checks
    if torch.std(log_Fq) < 1e-4:
        return float('nan')
    
    scale_range = log_scales.max() - log_scales.min()
    if scale_range < 1.0:
        return float('nan')
    
    try:
        X = torch.stack([log_scales, torch.ones_like(log_scales)], dim=1)
        y = log_Fq.unsqueeze(1)
        
        coeffs = torch.linalg.lstsq(X, y).solution
        H_q = coeffs[0, 0].item()
        
        # ✅ Validate Hurst exponent range
        if not (0.1 <= H_q <= 1.5):
            return float('nan')
            
        return H_q
        
    except Exception:
        return float('nan')
# ============================================================================
# STEP 7: MULTIFRACTAL SPECTRUM
# ============================================================================
def compute_multifractal_spectrum(q_values: torch.Tensor, 
                                  Hq_values: torch.Tensor) -> tuple:
    """
    Compute singularity spectrum (Hölder exponents and fractal dimensions).
    
    Mathematical Definitions:
        τ(q) = q·H(q) - 1           [Scaling exponent]
        α(q) = dτ/dq = H(q) + q·H'(q) [Hölder exponent]
        f(α) = q·α(q) - τ(q)         [Fractal dimension]
        
    Multifractality width:
        Δα = α_max - α_min
        
    Args:
        q_values: Moment orders [num_q]
        Hq_values: Hurst exponents at each q [num_q]
        
    Returns:
        (delta_alpha, alpha, f_alpha): Width, Hölder exponents, dimensions
        
    Reference:
        Halsey et al. (1986), Phys. Rev. A - Multifractal formalism
    """
    valid = ~torch.isnan(Hq_values)
    
    if valid.sum() < 3:
        empty = torch.tensor([], device=DEVICE)
        return torch.tensor(float('nan'), device=DEVICE), empty, empty
    
    q = q_values[valid]
    Hq = Hq_values[valid]
    
    # Sort by q for gradient computation
    q_sorted, indices = torch.sort(q)
    Hq_sorted = Hq[indices]
    
    # Scaling exponent: τ(q) = q·H(q) - 1
    tau_q = q_sorted * Hq_sorted - 1.0
    
    # Hölder exponent: α = dτ/dq (via numerical gradient)
    alpha = torch.gradient(tau_q, spacing=(q_sorted,))[0]
    
    # Fractal dimension: f(α) = q·α - τ
    f_alpha = q_sorted * alpha - tau_q
    
    # Multifractality width
    delta_alpha = torch.max(alpha) - torch.min(alpha)
    
    return delta_alpha, alpha, f_alpha


# ============================================================================
# COMPLETE MFDCCA PIPELINE FOR SINGLE PAIR
# ============================================================================
def mfdcca_single_pair(series1: torch.Tensor, series2: torch.Tensor,
                       q_list: list[float], scales: list[int],
                       epsilon: float = 1e-8) -> dict | None:
    """
    Complete MFDCCA analysis for a single pair of time series.
    """
    # STEP 1: Profile Construction
    profile1 = construct_profile(series1)
    profile2 = construct_profile(series2)
    
    if profile1.numel() == 0 or profile2.numel() == 0:
        return None
    
    q_tensor = torch.tensor(q_list, dtype=torch.float32, device=DEVICE)
    num_q = len(q_list)
    num_scales = len(scales)
    
    # Storage for fluctuation functions
    Fq_all_scales = torch.full((num_scales, num_q), float('nan'), device=DEVICE)
    Fq_plus_scales = torch.full((num_scales, num_q), float('nan'), device=DEVICE)
    Fq_minus_scales = torch.full((num_scales, num_q), float('nan'), device=DEVICE)
    
    # STEP 2-5: Loop over scales
    for scale_idx, scale in enumerate(scales):
        # STEP 2: Segmentation
        segments1, segments2 = segment_profiles(profile1, profile2, scale)
        
        if segments1.size(0) == 0:
            continue
        
        # STEP 3: Detrending
        residuals1, residuals2, slopes = detrend_segments(segments1, segments2, order=1)
        
        # STEP 4: Cross-Correlation
        Fxy = compute_cross_correlation(residuals1, residuals2)
        
        # Split by market regime (slope)
        plus_mask = slopes > 0
        minus_mask = slopes < 0
        
        Fxy_plus = Fxy[plus_mask]
        Fxy_minus = Fxy[minus_mask]
        
        # STEP 5: Aggregate for each q
        for q_idx, q in enumerate(q_list):
            Fq_all_scales[scale_idx, q_idx] = aggregate_fluctuations(Fxy, q, epsilon)
            
            if len(Fxy_plus) > 0:
                Fq_plus_scales[scale_idx, q_idx] = aggregate_fluctuations(Fxy_plus, q, epsilon)
            
            if len(Fxy_minus) > 0:
                Fq_minus_scales[scale_idx, q_idx] = aggregate_fluctuations(Fxy_minus, q, epsilon)
    
    scales_tensor = torch.tensor(scales, dtype=torch.float32, device=DEVICE)

    Hq_all = torch.full((num_q,), float('nan'), device=DEVICE)
    Hq_plus = torch.full((num_q,), float('nan'), device=DEVICE)
    Hq_minus = torch.full((num_q,), float('nan'), device=DEVICE)

    # ✅ MINIMIZED: Changed to DEBUG
    logger.debug(f"Fq_all_scales - shape: {Fq_all_scales.shape}, NaN count: {torch.isnan(Fq_all_scales).sum().item()}")
    
    for q_idx in range(num_q):
        Fq_all = Fq_all_scales[:, q_idx]
        Fq_plus = Fq_plus_scales[:, q_idx] 
        Fq_minus = Fq_minus_scales[:, q_idx]
        
        # ✅ MINIMIZED: Changed to DEBUG
        valid_all = (~torch.isnan(Fq_all)).sum().item()
        valid_plus = (~torch.isnan(Fq_plus)).sum().item()
        valid_minus = (~torch.isnan(Fq_minus)).sum().item()
        
        logger.debug(f"q={q_list[q_idx]}: valid points - all:{valid_all}/{len(scales)}, plus:{valid_plus}, minus:{valid_minus}")
        
        Hq_all[q_idx] = estimate_hurst_exponent(Fq_all, scales_tensor)
        Hq_plus[q_idx] = estimate_hurst_exponent(Fq_plus, scales_tensor)
        Hq_minus[q_idx] = estimate_hurst_exponent(Fq_minus, scales_tensor)
    
    # ✅ MINIMIZED: Changed to DEBUG
    logger.debug(f"Final Hq_all: {Hq_all}")
    logger.debug(f"NaN in Hq_all: {torch.isnan(Hq_all).sum().item()}/{num_q}")
    
    # STEP 7: Multifractal Spectrum
    delta_alpha, alpha, f_alpha = compute_multifractal_spectrum(q_tensor, Hq_all)
    
    # ✅ KEPT AS INFO: Only show summary per pair
    valid_hurst = (~torch.isnan(Hq_all)).sum().item()
    if valid_hurst < num_q:
        logger.info(f"MFDCCA pair: {valid_hurst}/{num_q} valid Hurst exponents")
    
    return {
        'Hq_all': Hq_all,
        'Hq_plus': Hq_plus,
        'Hq_minus': Hq_minus,
        'delta_H': torch.mean(torch.abs(Hq_plus - Hq_minus)),
        'delta_alpha': delta_alpha,
        'alpha': alpha,
        'f_alpha': f_alpha
    }

# ============================================================================
# BATCHED MFDCCA FOR MULTIPLE PAIRS (GPU-Optimized)
# ============================================================================
# Add this function to your mfdcca.py file

    
def process_token_pairs_gpu_flexible(token_list, residuals, start_date, end_date, q_list):
    """
    FLEXIBLE MFDCCA: Uses whatever residual length is available from CAPM
    (Recommended approach - handles variable input lengths gracefully)
    """
    N = len(token_list)
    all_results = []

    # Prepare residual tensors on GPU
    token_tensors = {}
    residual_lengths = []
    
    for token in token_list:
        if token not in residuals:
            continue

        res = residuals[token]

        # ✅ FLEXIBLE: Handle whatever residual length CAPM produced
        if isinstance(res, torch.Tensor):
            tensor = res.to(DEVICE, dtype=torch.float32)
        else:
            tensor = torch.tensor(res, device=DEVICE, dtype=torch.float32)
        
        actual_length = tensor.size(0)
        if actual_length < 30:  # Minimum reasonable length
            logger.debug(f"Token {token} has insufficient residuals: {actual_length} days")
            continue
            
        token_tensors[token] = tensor
        residual_lengths.append(actual_length)
        
        logger.debug(f"Token {token}: {actual_length} residuals available")

    if len(token_tensors) < 2:
        logger.warning("Insufficient tokens with valid data")
        return []

    # ✅ FLEXIBLE: Use minimum available length across all tokens
    min_length = min(residual_lengths)
    max_length = max(residual_lengths)
    
    logger.info(f"Flexible MFDCCA: Residual lengths range {min_length}-{max_length}, using {min_length}")

    # Generate all unique pairs
    pairs = [
        (token_list[i], token_list[j]) 
        for i in range(N) for j in range(i+1, N)
        if token_list[i] in token_tensors and token_list[j] in token_tensors
    ]
    
    logger.info(f"Processing {len(pairs)} pairs with flexible lookback...")

    # Construct profiles for all tokens
    profiles = {token: construct_profile(tensor[-min_length:]) for token, tensor in token_tensors.items()}

    min_scale = 15
    max_scale = min(int(min_length // 4), 60)
    num_scales = min(12, max_scale - min_scale)
    
    scales_tensor = torch.logspace(
        torch.log10(torch.tensor(float(min_scale), device=DEVICE)),
        torch.log10(torch.tensor(float(max_scale), device=DEVICE)),
        steps=num_scales,  # Now 15 instead of 8
        device=DEVICE
    )
    scales = torch.unique(scales_tensor.round().to(torch.int64)).tolist()
    scales = [s for s in scales if s >= 4]  # Keep valid scales
    
    if not scales:
        scales = [min_scale]
    
    logger.info(f"✅ Scales: {len(scales)} scales from {min(scales)} to {max(scales)}")

    # Process each pair
    for t1, t2 in pairs:
        # Use the same length for both tokens
        series1 = token_tensors[t1][-min_length:]
        series2 = token_tensors[t2][-min_length:]
        
        result = mfdcca_single_pair(
            series1,
            series2,
            q_list,
            scales,
            epsilon=CONFIG['mfdcca_epsilon']
        )

        if result is not None:
            all_results.append({
                'token1': t1,
                'token2': t2,
                'Hq_all': result['Hq_all'].cpu().tolist(),
                'Hq_plus': result['Hq_plus'].cpu().tolist(),
                'Hq_minus': result['Hq_minus'].cpu().tolist(),
                'delta_H': result['delta_H'],
                'delta_alpha': result['delta_alpha'],
                'q_list': q_list,
                'actual_days_used': min_length  # Track actual data used
            })

    logger.info(f"✅ Flexible MFDCCA: Processed {len(all_results)} pairs using {min_length} days")
    return all_results

# Update the main function to use flexible approach
def process_token_pairs_gpu(token_list: list, residuals: dict, start_date, end_date, q_list: list[float]):
    """
    MAIN ENTRY POINT - Now uses flexible lookback by default
    """
    return process_token_pairs_gpu_flexible(token_list, residuals, start_date, end_date, q_list)
# ============================================================================
# EXTRACT MATRICES FOR PAIR SELECTION
# ============================================================================
# mfdcca.py - CORRECTED extract_hurst_matrices()
def extract_hurst_matrices(token_list: list, results: list):
    """
    ✅ FIXED: Store complete Hₓᵧ(q) matrix including Hₓᵧ(2) for each pair
    """
    N = len(token_list)
    q_list = CONFIG['q_list']
    num_q = len(q_list)
    
    # ✅ NEW: Store ALL Hₓᵧ(q) values [N, N, num_q]
    hxy_matrix = torch.full((N, N, num_q), float('nan'), device=DEVICE)
    delta_H_matrix = torch.full((N, N), float('nan'), device=DEVICE)
    delta_alpha_matrix = torch.full((N, N), float('nan'), device=DEVICE)
    
    token_index = {token: idx for idx, token in enumerate(token_list)}
    hurst_dict = {}
    
    # ✅ DEBUG: Count valid results
    valid_results = 0
    invalid_pairs = 0
    
    for result in results:
        if result is None:
            continue
            
        t1, t2 = result['token1'], result['token2']
        i, j = token_index[t1], token_index[t2]
        
        hq_all_data = result['Hq_all']
        
        if hq_all_data is None or len(hq_all_data) != num_q:
            invalid_pairs += 1
            logger.debug(f"❌ Pair {t1}-{t2}: Hq_all has {len(hq_all_data) if hq_all_data else 0} values, expected {num_q}. Skipping.")
            continue
        
        try:
            hxy_tensor = torch.tensor(hq_all_data, device=DEVICE, dtype=torch.float32)
            
            # ✅ ADD THIS NEW VALIDATION:
            valid_count = (~torch.isnan(hxy_tensor)).sum().item()
            
            if valid_count < num_q * 0.5:  # Accept pairs with 60% valid q-orders
                invalid_pairs += 1
                logger.debug(f"❌ Pair {t1}-{t2}: Only {valid_count}/{num_q} valid Hurst values")
                continue
            
            # Rest of the loop stays the same
            logger.debug(f"Pair {t1}-{t2}: hxy_tensor shape: {hxy_tensor.shape}")
            
            hxy_matrix[i, j, :] = hxy_tensor
            hxy_matrix[j, i, :] = hxy_tensor
                        
            # Keep existing structure for compatibility
            hurst_dict[(t1, t2)] = {
                'Hq_all': hxy_tensor,
                'Hq_plus': torch.tensor(result['Hq_plus'], device=DEVICE, dtype=torch.float32),
                'Hq_minus': torch.tensor(result['Hq_minus'], device=DEVICE, dtype=torch.float32)
            }
            
            delta_H_matrix[i, j] = delta_H_matrix[j, i] = result['delta_H']
            delta_alpha_matrix[i, j] = delta_alpha_matrix[j, i] = result['delta_alpha']
            
            valid_results += 1
            
        except Exception as e:
            logger.error(f"Failed to process pair {t1}-{t2}: {e}")
            continue
    
    # ✅ KEPT AS INFO: Final summary only
    logger.info(f"✅ extract_hurst_matrices: {valid_results} valid pairs, {invalid_pairs} invalid pairs, {len(results)} total")
    logger.info(f"✅ Final hxy_matrix shape: {hxy_matrix.shape}")
    
    return hurst_dict, hxy_matrix, delta_H_matrix, delta_alpha_matrix

