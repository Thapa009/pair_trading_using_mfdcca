import torch
import logging
import numpy as np
from config import CONFIG, DEVICE

logger = logging.getLogger(__name__)

_DESIGN_MATRIX_CACHE = {}


def compute_hurst_exponent_robust(log_scales, log_Fq):
    """Robust Hurst exponent via least squares regression"""
    valid_mask = torch.isfinite(log_Fq)
    if valid_mask.sum() < 2:
        return torch.tensor(float("nan"), device=DEVICE)

    try:
        X_reg = torch.stack(
            [log_scales[valid_mask], torch.ones_like(log_scales[valid_mask])], dim=1
        )
        coeffs = torch.linalg.lstsq(X_reg, log_Fq[valid_mask].unsqueeze(1)).solution
        return coeffs[0, 0]
    except:
        return torch.tensor(float("nan"), device=DEVICE)


def scale_selection(data_length, num_scales=25):
    """
    Generate log-spaced scales for MF-DCCA

    Args:
        data_length (int): Length of the time series
        num_scales (int): Number of scales to generate

    Returns:
        torch.Tensor: 1D tensor of scales (int)
    """
    s_min = 10
    s_max = data_length // 4

    if s_max <= s_min:
        return torch.tensor([], device=DEVICE, dtype=torch.int32)

    # Log-spaced values (float)
    scales_float = np.logspace(np.log10(s_min), np.log10(s_max), num=num_scales)

    # Round to nearest integer and remove duplicates
    scales = np.unique(np.round(scales_float).astype(int))

    return torch.tensor(scales, device=DEVICE, dtype=torch.int32)


def get_design_matrix(scale: int, device: torch.device, dtype=torch.float32):
    """
    Cached design matrix for linear detrending
    Returns: (X^T X)^-1 X^T for fast least squares
    """
    key = (scale, str(device), str(dtype))
    if key not in _DESIGN_MATRIX_CACHE:
        t = torch.arange(scale, dtype=dtype, device=device)
        X = torch.stack([t, torch.ones_like(t)], dim=1)
        XtX_inv_Xt = torch.linalg.inv(X.T @ X) @ X.T
        _DESIGN_MATRIX_CACHE[key] = XtX_inv_Xt
    return _DESIGN_MATRIX_CACHE[key]


def compute_fluctuation_function(profiles1, profiles2, scale, q, design_matrix):
    """
    CORRECTED MF-DCCA implementation matching the algorithm exactly
    """
    n_pairs = profiles1.shape[0]
    N = profiles1.shape[1]

    # 1. Determine number of segments (Ns)
    Ns = N // scale  # This is int(N/s) as per algorithm

    # 2. Total segments = 2 * Ns (forward + reverse)
    total_segments = 2 * Ns

    # 3. Initialize segment arrays
    seg1 = torch.zeros((n_pairs, total_segments, scale), device=DEVICE)
    seg2 = torch.zeros((n_pairs, total_segments, scale), device=DEVICE)

    # 4. Extract FORWARD segments (v = 1, 2, ..., Ns)
    for ν in range(Ns):
        start_idx = ν * scale
        seg1[:, ν, :] = profiles1[:, start_idx : start_idx + scale]
        seg2[:, ν, :] = profiles2[:, start_idx : start_idx + scale]

    # 5. Extract REVERSE segments (v = Ns+1, ..., 2Ns)
    #    Starting from the END and going backward
    for ν in range(Ns):
        # Formula from algorithm: X(N - (v - Ns)s + i)
        # For v = Ns + ν (where ν runs from 0 to Ns-1)
        # start_idx = N - (ν + 1) * scale  # This is CORRECT
        start_idx = N - (ν + 1) * scale
        seg1[:, Ns + ν, :] = profiles1[:, start_idx : start_idx + scale]
        seg2[:, Ns + ν, :] = profiles2[:, start_idx : start_idx + scale]

    # Flatten for batch processing
    seg1_flat = seg1.reshape(-1, scale)
    seg2_flat = seg2.reshape(-1, scale)

    # 6. Linear detrending (same as before)
    coeffs1 = seg1_flat @ design_matrix.T
    coeffs2 = seg2_flat @ design_matrix.T

    t = torch.arange(scale, device=DEVICE, dtype=torch.float32)
    fitted1 = coeffs1[:, 0:1] * t + coeffs1[:, 1:2]
    fitted2 = coeffs2[:, 0:1] * t + coeffs2[:, 1:2]

    residuals1 = seg1_flat - fitted1
    residuals2 = seg2_flat - fitted2

    # 7. Compute F²(s, ν) = (1/s) Σ |X-X̃|·|Y-Ỹ|
    F2_segment = (torch.abs(residuals1) * torch.abs(residuals2)).mean(dim=1)

    F2_segment = F2_segment.view(n_pairs, total_segments)

    # 8. Compute q-order fluctuation function
    q_val = float(q)

    if abs(q_val) < 1e-10:  # q = 0
        # F₀(s) = exp{ 1/(4Ns) Σ ln[F²(s,ν)] }
        log_sum = torch.log(F2_segment + 1e-10).sum(dim=1)
        Fq = torch.exp(log_sum / (4 * Ns))  # 4Ns = 2 * total_segments
    else:  # q ≠ 0
        # F_q(s) = { 1/(2Ns) Σ [F²(s,ν)^(q/2)] }^(1/q)
        power_sum = F2_segment.pow(q_val / 2.0).sum(dim=1)
        Fq = torch.pow((1.0 / (2 * Ns)) * power_sum, 1.0 / q_val)

    return Fq


import torch
from scipy.interpolate import UnivariateSpline


def compute_multifractal_spectrum(q_vals, Hq_vals):
    """
    Correct Legendre transform for MF-DCCA
    Implements Equations 4-5 from the paper:
        α = H(q) + q * dH/dq
        f(α) = q * (α - H(q)) + 1

    Args:
        q_vals: 1D tensor of q values
        Hq_vals: 1D tensor of generalized Hurst exponents H(q)

    Returns:
        dict with keys: 'alpha', 'f_alpha', 'delta_alpha', 'tau_q'
    """
    import numpy as np
    import torch

    # Convert to numpy for gradient calculation
    q = q_vals.cpu().numpy()
    Hq = Hq_vals.cpu().numpy()

    # Step 1: Renyi exponent
    tau_q = q * Hq - 1.0

    # Step 2: derivative of H(q)
    dH_dq = np.gradient(Hq, q)

    # Step 3: singularity strength α
    alpha = Hq + q * dH_dq

    # Step 4: multifractal spectrum f(α)
    f_alpha = q * (alpha - Hq) + 1.0

    # Step 5: multifractality width Δα
    delta_alpha = np.max(alpha) - np.min(alpha)

    # Convert back to tensors
    return {
        "alpha": torch.tensor(alpha, device=DEVICE, dtype=torch.float32),
        "f_alpha": torch.tensor(f_alpha, device=DEVICE, dtype=torch.float32),
        "delta_alpha": torch.tensor(delta_alpha, device=DEVICE, dtype=torch.float32),
        "tau_q": torch.tensor(tau_q, device=DEVICE, dtype=torch.float32),
    }


def compute_delta_metrics(Hq_all, q_values):
    """
    Compute ΔH and Δα using corrected Legendre transform
    """
    valid_mask = torch.isfinite(Hq_all)
    valid_Hq = Hq_all[valid_mask]
    valid_q = q_values[valid_mask]

    if valid_Hq.numel() < 2:
        nan_tensor = torch.tensor(float("nan"), device=DEVICE)
        return nan_tensor, nan_tensor

    # ΔH metric
    delta_H = valid_Hq.max() - valid_Hq.min()

    # Δα metric via corrected Legendre transform
    try:
        spectrum = compute_multifractal_spectrum(valid_q, valid_Hq)
        delta_alpha = spectrum["delta_alpha"]
        if not torch.isfinite(delta_alpha):
            delta_alpha = torch.tensor(float("nan"), device=DEVICE)
    except:
        delta_alpha = torch.tensor(float("nan"), device=DEVICE)

    return delta_H, delta_alpha


def process_token_pairs(token_list, residuals, q_list):
    """
    Optimized batch MF-DCCA for cryptocurrency pair analysis

    Process:
    1. Generate all unique pairs
    2. Compute cumulative profiles (integrated residuals)
    3. For each scale s and moment q:
       - Segment profiles
       - Detrend via linear regression
       - Compute cross-covariance fluctuations
       - Calculate Fq(s)
    4. Extract H(q) via log-log regression: Fq(s) ~ s^H(q)
    5. Compute multifractality metrics
    """
    # Stack residuals on GPU
    residuals_stack = torch.stack(
        [
            torch.tensor(residuals[token].values, device=DEVICE, dtype=torch.float32)
            for token in token_list
        ]
    )
    n_tokens = len(token_list)

    # Generate unique pairs (upper triangular indices)
    i_idx, j_idx = torch.triu_indices(n_tokens, n_tokens, offset=1, device=DEVICE)
    n_pairs = i_idx.shape[0]

    logger.info(f"Processing {n_pairs} pairs from {n_tokens} tokens")

    # Extract pair series
    series1_batch = residuals_stack[i_idx]
    series2_batch = residuals_stack[j_idx]

    # Scale selection
    min_length = residuals_stack.shape[1]
    scales_tensor = scale_selection(min_length)

    if len(scales_tensor) == 0:
        logger.error("No valid scales - data too short for MF-DCCA")
        return []

    num_scales = len(scales_tensor)
    num_q = len(q_list)

    # **CRITICAL STEP 1: Compute cumulative profiles (integration)**
    # Remove mean to avoid trend in cumsum
    series1_centered = series1_batch - series1_batch.mean(dim=1, keepdim=True)
    series2_centered = series2_batch - series2_batch.mean(dim=1, keepdim=True)

    profiles1 = torch.cumsum(series1_centered, dim=1)
    profiles2 = torch.cumsum(series2_centered, dim=1)

    logger.info(f"Profiles computed for {n_pairs} pairs")

    # Initialize Hurst exponent storage
    Hq_all_batch = torch.full((n_pairs, num_q), float("nan"), device=DEVICE)

    # Log-scales for regression
    log_scales = torch.log(scales_tensor.float())

    # **STEP 2: Compute Fq(s) for each q**
    for q_idx, q in enumerate(q_list):
        Fq_all_scales = torch.zeros((n_pairs, num_scales), device=DEVICE)

        # Compute fluctuation function for each scale
        for scale_idx in range(num_scales):
            scale = int(scales_tensor[scale_idx].item())
            design_matrix = get_design_matrix(scale, DEVICE)

            Fq = compute_fluctuation_function(
                profiles1, profiles2, scale, q, design_matrix
            )
            Fq_all_scales[:, scale_idx] = Fq

        # **STEP 3: Extract H(q) via log-log regression**
        epsilon = 1e-10
        log_Fq_vals = torch.log(Fq_all_scales + epsilon)

        # Fit H(q) for each pair
        for pair_idx in range(n_pairs):
            valid_mask = torch.isfinite(log_Fq_vals[pair_idx])

            # Need at least 4 points for reliable regression
            if valid_mask.sum() >= 4:
                try:
                    Hq_val = compute_hurst_exponent_robust(
                        log_scales[valid_mask], log_Fq_vals[pair_idx, valid_mask]
                    )
                    Hq_all_batch[pair_idx, q_idx] = Hq_val
                except:
                    pass

    # **STEP 4: Compute multifractality metrics**
    delta_H_all = torch.full((n_pairs,), float("nan"), device=DEVICE)
    delta_alpha_all = torch.full((n_pairs,), float("nan"), device=DEVICE)
    q_tensor = torch.tensor(
        [float(q) for q in q_list], device=DEVICE, dtype=torch.float32
    )

    for pair_idx in range(n_pairs):
        Hq_pair = Hq_all_batch[pair_idx]
        valid_mask = torch.isfinite(Hq_pair)

        # Need at least 3 q-values for spectrum computation
        # (5+ recommended for stable Legendre transform)
        if valid_mask.sum() >= 3:
            delta_H, delta_alpha = compute_delta_metrics(
                Hq_pair[valid_mask], q_tensor[valid_mask]
            )
            delta_H_all[pair_idx] = delta_H
            delta_alpha_all[pair_idx] = delta_alpha

    # **STEP 5: Prepare results**
    i_idx_cpu = i_idx.cpu().numpy()
    j_idx_cpu = j_idx.cpu().numpy()

    all_results = []
    for pair_idx in range(n_pairs):
        i, j = i_idx_cpu[pair_idx], j_idx_cpu[pair_idx]

        # Extract H(q=2) for cross-correlation analysis
        Hxy2 = float("nan")
        if 2 in q_list:
            q2_idx = q_list.index(2)
            if q2_idx < Hq_all_batch.shape[1]:
                Hxy2 = Hq_all_batch[pair_idx, q2_idx].item()

        all_results.append(
            {
                "token1": token_list[i],
                "token2": token_list[j],
                "Hq_all": Hq_all_batch[pair_idx].cpu().numpy(),
                "Hxy2": Hxy2,
                "delta_H": delta_H_all[pair_idx].item(),
                "delta_alpha": delta_alpha_all[pair_idx].item(),
            }
        )

    valid_count = sum(1 for r in all_results if not np.isnan(r["delta_H"]))
    logger.info(f"✅ MF-DCCA Complete: {valid_count}/{n_pairs} valid pairs")

    return all_results


def extract_hurst_matrices(token_list: list, results: list, q_list: list):
    """
    Extract symmetric matrices of MF-DCCA metrics

    Returns:
    - Hxy(2): Cross-correlation Hurst exponent at q=2
    - ΔH: Generalized Hurst exponent range
    - Δα: Singularity spectrum width
    """
    N = len(token_list)
    token_index = {token: idx for idx, token in enumerate(token_list)}

    # Initialize symmetric matrices on GPU
    hxy2_matrix = torch.full((N, N), float("nan"), device=DEVICE)
    delta_H_matrix = torch.full((N, N), float("nan"), device=DEVICE)
    delta_alpha_matrix = torch.full((N, N), float("nan"), device=DEVICE)

    # Find q=2 index
    q2_idx = q_list.index(2) if 2 in q_list else -1

    # Populate matrices
    for result in results:
        t1, t2 = result["token1"], result["token2"]

        if t1 not in token_index or t2 not in token_index:
            continue

        i, j = token_index[t1], token_index[t2]

        # Extract H(q=2)
        Hq_all_np = result["Hq_all"]
        if q2_idx >= 0 and not np.isnan(Hq_all_np[q2_idx]):
            val = float(Hq_all_np[q2_idx])
            hxy2_matrix[i, j] = hxy2_matrix[j, i] = val

        # Extract ΔH
        delta_H_val = result["delta_H"]
        if not np.isnan(delta_H_val):
            delta_H_matrix[i, j] = delta_H_matrix[j, i] = float(delta_H_val)

        # Extract Δα
        delta_alpha_val = result["delta_alpha"]
        if not np.isnan(delta_alpha_val):
            delta_alpha_matrix[i, j] = delta_alpha_matrix[j, i] = float(delta_alpha_val)

    logger.info(f"✅ Extracted matrices for {len(results)} pairs")

    return hxy2_matrix, delta_H_matrix, delta_alpha_matrix
