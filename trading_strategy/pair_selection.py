# pair_selection.py - FIXED VERSION
import logging
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional
from config import DEVICE

logger = logging.getLogger(__name__)

# ============================================================================
# GPU UTILITY FUNCTIONS
# ============================================================================

# pair_selection.py - FIXED VERSION
import logging
import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional
from config import DEVICE

logger = logging.getLogger(__name__)


def ensure_gpu_tensor(data: Any) -> Optional[torch.Tensor]:
    """Robust GPU tensor conversion with detailed error logging"""
    if data is None:
        logger.debug("ensure_gpu_tensor: Input is None")
        return None

    try:
        if isinstance(data, torch.Tensor):
            # Already a tensor, move to GPU
            return data.to(DEVICE)
        elif isinstance(data, np.ndarray):
            # Convert numpy array to GPU tensor
            return torch.tensor(data, device=DEVICE, dtype=torch.float32)
        else:
            # Try to convert other types
            return torch.tensor(data, device=DEVICE, dtype=torch.float32)
    except Exception as e:
        logger.error(f"Cannot convert to GPU tensor: {type(data)} - {e}")
        # Log the shape if it's an array-like object
        if hasattr(data, "shape"):
            logger.error(f"  Shape: {data.shape}")
        elif hasattr(data, "__len__"):
            logger.error(f"  Length: {len(data)}")
        return None


def select_pairs_mfdcca(
    features: Dict[str, Any],
    pair_hxy_threshold: float,
    threshold_h: float,
    threshold_alpha: float,
    token_list: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """
    ✅ CORRECTED: Extract matrices from features before using them
    """
    if not features.get("has_data", False):
        return []

    # ✅ FIX: Get tokens_used from features OR use provided token_list
    tokens_used = features.get("tokens_used")
    if tokens_used is None:
        if token_list is not None:
            tokens_used = token_list
        else:
            logger.warning("No tokens_used found in features, using CONFIG")
            tokens_used = CONFIG["token_names"]

    # ✅✅✅ CRITICAL FIX: Extract matrices from features
    hxy_matrix = ensure_gpu_tensor(features.get("hxy_matrix"))
    delta_H_matrix = ensure_gpu_tensor(features.get("delta_H_matrix"))
    delta_alpha_matrix = ensure_gpu_tensor(features.get("delta_alpha_matrix"))

    # ✅ Validate that matrices exist
    if hxy_matrix is None or delta_H_matrix is None or delta_alpha_matrix is None:
        logger.warning("MFDCCA: One or more matrices are None")
        return []

    n_tokens = len(tokens_used)
    if n_tokens < 2:
        return []

    # Get upper-triangular indices (i < j)
    i_idx, j_idx = torch.triu_indices(n_tokens, n_tokens, offset=1, device=DEVICE)

    # ✅ Now these variables are properly defined
    valid_mask = (
        ~torch.isnan(hxy_matrix[i_idx, j_idx])
        & ~torch.isnan(delta_H_matrix[i_idx, j_idx])
        & ~torch.isnan(delta_alpha_matrix[i_idx, j_idx])
        & (hxy_matrix[i_idx, j_idx] < pair_hxy_threshold)
        & (delta_H_matrix[i_idx, j_idx] < threshold_h)
        & (delta_alpha_matrix[i_idx, j_idx] < threshold_alpha)
    )

    valid_i_idx = i_idx[valid_mask]
    valid_j_idx = j_idx[valid_mask]

    selected_pairs = [
        (tokens_used[i], tokens_used[j])
        for i, j in zip(valid_i_idx.cpu(), valid_j_idx.cpu())
    ]

    logger.info(f"✅ MFDCCA: {len(selected_pairs)} pairs selected")
    logger.info(
        f"   Thresholds: Hxy<{pair_hxy_threshold}, ΔH<{threshold_h}, Δα<{threshold_alpha}"
    )

    # ✅ ADD DEBUG INFO: Show matrix statistics
    hxy_values = hxy_matrix[~torch.isnan(hxy_matrix)].cpu().numpy()
    if len(hxy_values) > 0:
        logger.info(
            f"   Hxy stats - Mean: {hxy_values.mean():.3f}, "
            f"Min: {hxy_values.min():.3f}, Max: {hxy_values.max():.3f}"
        )

    return selected_pairs


# ============================================================================
# DCCA PAIR SELECTION
# ============================================================================


def select_pairs_dcca(
    features: Dict[Tuple[str, str], Dict[str, Any]],
    pair_hxy_threshold: float,
    token_list: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """
    ✅ PURE DCCA SELECTION (Podobnik & Stanley 2008)
    Simple H_xy threshold only - no additional filters
    """

    if not features:
        logger.warning("⚠️ No DCCA features available")
        return []

    token_filter = set(token_list) if token_list else None
    selected_pairs = []

    stats = {"total": 0, "valid": 0, "selected": 0}

    for (t1, t2), feat in features.items():
        stats["total"] += 1

        if token_filter and (t1 not in token_filter or t2 not in token_filter):
            continue

        # ✅ Only check H_xy - nothing else
        H_xy = feat.get("H_xy")

        if H_xy is None:
            continue

        stats["valid"] += 1

        # ✅ PURE DCCA: Simple H_xy threshold
        if H_xy < pair_hxy_threshold:
            selected_pairs.append((t1, t2))
            stats["selected"] += 1

    # Paper-style reporting - UPDATED FORMATTING
    logger.info(f"📊 PURE DCCA Selection:")
    logger.info(f"   Method: Simple Hₓᵧ threshold (Podobnik & Stanley 2008)")
    logger.info(f"   Threshold: Hₓᵧ < {pair_hxy_threshold}")
    logger.info(f"   Total pairs analyzed: {stats['total']}")
    logger.info(f"   Valid Hₓᵧ estimates: {stats['valid']}")
    logger.info(f"   Selected (mean-reverting): {stats['selected']}")

    if stats["valid"] > 0:
        selection_rate = stats["selected"] / stats["valid"]
        logger.info(f"   Selection rate: {selection_rate:.1%}")

    return selected_pairs


# ============================================================================
# PEARSON PAIR SELECTION
# ============================================================================


def select_pairs_pearson(
    features: Dict[str, Any], rho_threshold: float  # Keep as float, not Optional
) -> List[Tuple[str, str]]:
    """GPU-accelerated Pearson selection"""

    # ✅ Assert it's not None (helps with type checking)
    assert rho_threshold is not None, "rho_threshold cannot be None"

    """GPU-accelerated Pearson selection with proper None handling"""

    # ✅ Use .get() for safe access
    corr_matrix = ensure_gpu_tensor(features.get("correlation_matrix"))
    token_list = features.get("token_list", [])

    # ✅ CRITICAL: Check for None values
    if corr_matrix is None:
        logger.warning("Pearson: Correlation matrix is None")
        return []

    if not token_list:
        logger.warning("Pearson: Token list is empty")
        return []

    n = len(token_list)

    # ✅ Check shape consistency
    if corr_matrix.shape != (n, n):
        logger.warning(
            f"Pearson: Shape mismatch. Expected ({n},{n}), got {corr_matrix.shape}"
        )
        return []

    # Create mask for upper triangular (excluding diagonal)
    mask = torch.triu(torch.ones(n, n, device=DEVICE), diagonal=1).bool()

    # Extract correlation values
    corr_values = corr_matrix[mask]

    # GPU filtering: |ρ| > threshold
    selected_mask = torch.abs(corr_values) > rho_threshold

    # Get indices of selected pairs
    pair_idx = torch.nonzero(mask, as_tuple=False)
    selected_pairs_idx = pair_idx[selected_mask]

    # Convert to Python list of tuples
    if len(selected_pairs_idx) > 0:
        pairs_cpu = selected_pairs_idx.cpu().numpy()
        selected_pairs = [
            tuple(sorted([token_list[i], token_list[j]])) for i, j in pairs_cpu
        ]
    else:
        selected_pairs = []

    # Add logging for Pearson
    logger.info(f"📈 Pearson: {len(selected_pairs)} pairs selected")
    logger.info(f"   Threshold: |ρ| > {rho_threshold}")
    logger.info(f"   Token count: {len(token_list)}")

    return selected_pairs


# ============================================================================
# COINTEGRATION PAIR SELECTION
# ============================================================================


def select_pairs_cointegration(
    features: Dict[Tuple[str, str], Dict[str, Any]],
    pval_threshold: float,
    token_list: List[str],
) -> List[Tuple[str, str]]:
    """
    ✅ PURE Engle–Granger pair selection
    """

    if not features:
        logger.warning("⚠️ No cointegration features available")
        return []

    allowed_tokens = set(token_list)
    selected_pairs = []
    analyzed_pairs = 0

    for (t1, t2), feat in features.items():
        if t1 not in allowed_tokens or t2 not in allowed_tokens:
            continue

        analyzed_pairs += 1

        if feat["pvalue"] < pval_threshold:
            selected_pairs.append((t1, t2))

    logger.info("🔗 Cointegration Pair Selection:")
    logger.info(f"   Threshold: p < {pval_threshold}")
    logger.info(f"   Pairs analyzed: {analyzed_pairs}")
    logger.info(f"   Pairs selected: {len(selected_pairs)}")

    return selected_pairs
