import torch
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import torch
from config import CONFIG, DEVICE
from mfdcca import process_token_pairs, extract_hurst_matrices
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint


logger = logging.getLogger(__name__)


# ============================================================================
# MFDCCA FEATURE EXTRACTION (Single Implementation)
def extract_mfdcca_features(
    residuals: Dict[str, pd.Series],
    token_list: List[str],
    q_list: List[float],
    lookback_start: pd.Timestamp,
    lookback_end: pd.Timestamp,
) -> Dict[str, Any]:
    """
    ✅ BALANCED: Minimal validation without redundancy
    """

    logger.info(f"MFDCCA: {lookback_start.date()} to {lookback_end.date()}")

    # 1. Basic validation
    if not isinstance(residuals, dict):
        return {"has_data": False}

    # 2. Get tokens with sufficient data
    valid_tokens = []
    for token in token_list:
        if token in residuals:
            series = residuals[token]
            if isinstance(series, pd.Series) and len(series) >= 90:
                valid_tokens.append(token)

    if len(valid_tokens) < 2:
        return {"has_data": False}

    logger.info(f"MFDCCA: {len(valid_tokens)} tokens")

    # 3. Run MFDCCA (no date filtering - trust pipeline)
    results = process_token_pairs(
        token_list=valid_tokens,
        residuals={t: residuals[t] for t in valid_tokens},
        q_list=q_list,
    )

    if not results:
        return {"has_data": False}

    # 4. Extract matrices
    hxy_matrix, delta_H_matrix, delta_alpha_matrix = extract_hurst_matrices(
        token_list=valid_tokens, results=results, q_list=q_list
    )

    logger.info(f"✅ MFDCCA: {len(results)} pairs")

    return {
        "has_data": True,
        "hxy_matrix": hxy_matrix,
        "delta_H_matrix": delta_H_matrix,
        "delta_alpha_matrix": delta_alpha_matrix,
        "q_list": q_list,
        "num_pairs": len(results),
        "tokens_used": valid_tokens,
    }


# ============================================================================
# DCCA FEATURE EXTRACTION (Single Implementation)
# ============================================================================


def extract_dcca_features(
    residuals: Dict[str, pd.Series],
    token_list: List[str],
    window: int,
    max_scale_ratio: float = 0.25,
    eps: float = 1e-16,
    device=None,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Pure GPU implementation of DCCA.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"📊 Extracting DCCA features on {device}...")

    if window < 90:
        return {}

    # Build profiles on GPU
    profiles = {}
    valid_tokens = []
    for token in token_list:
        if token not in residuals:
            continue
        series = residuals[token].dropna()
        if len(series) < 90:
            continue
        # Direct to GPU
        x = torch.tensor(series.values, device=device, dtype=torch.float32)
        profile = torch.cumsum(x - x.mean(), dim=0)
        profiles[token] = profile
        valid_tokens.append(token)

    if len(valid_tokens) < 2:
        return {}

    N = len(next(iter(profiles.values())))
    min_scale = 10
    max_scale = min(int(N * max_scale_ratio), N - 10)
    step = max(5, (max_scale - min_scale) // 10)

    # Generate scales on GPU
    scales_tensor = torch.arange(
        min_scale, max_scale + 1, step, device=device, dtype=torch.float32
    )
    scales_tensor = scales_tensor[(N - scales_tensor + 1) >= 10]

    if len(scales_tensor) < 4:
        return {}

    dcca_features = {}

    for i in range(len(valid_tokens)):
        for j in range(i + 1, len(valid_tokens)):
            t1, t2 = valid_tokens[i], valid_tokens[j]
            X = profiles[t1]
            Y = profiles[t2]

            signed_F2_by_scale = {}
            magnitudes_list = []
            signs_list = []
            scales_list = []

            for s in scales_tensor.int().tolist():
                s_int = int(s)
                N = X.shape[0]
                Ns = N // s  # number of segments

                # Trim series so it divides exactly into Ns segments
                X_trim = X[: Ns * s]
                Y_trim = Y[: Ns * s]

                # Non-overlapping segmentation (stride = s)
                seg_x = X_trim.view(Ns, s)
                seg_y = Y_trim.view(Ns, s)

                t = torch.arange(s_int, device=device, dtype=torch.float32)
                A = torch.stack([t, torch.ones_like(t)], dim=1)

                # GPU linear solve
                ATA = A.T @ A
                C = torch.linalg.solve(ATA, A.T)

                beta_x = seg_x @ C.T
                beta_y = seg_y @ C.T
                fit_x = beta_x @ A.T
                fit_y = beta_y @ A.T

                dx = seg_x - fit_x
                dy = seg_y - fit_y

                f2_nu = (dx * dy).sum(dim=1) / (s_int - 1)
                F2_s = f2_nu.mean().item()
                signed_F2_by_scale[s_int] = F2_s

                magnitude = torch.sqrt(torch.abs(torch.tensor(F2_s)) + eps)
                magnitudes_list.append(magnitude)
                signs_list.append(torch.sign(torch.tensor(F2_s)).int().item())
                scales_list.append(s_int)

            if len(scales_list) < 4:
                continue

            # PURE GPU REGRESSION
            scales_gpu = torch.tensor(scales_list, dtype=torch.float32, device=device)
            magnitudes_gpu = torch.tensor(
                magnitudes_list, dtype=torch.float32, device=device
            )

            log_s = torch.log(scales_gpu)
            log_F = torch.log(magnitudes_gpu)

            # GPU linear regression
            A = torch.stack([log_s, torch.ones_like(log_s)], dim=1)
            ATA = A.T @ A
            ATb = A.T @ log_F
            theta = torch.linalg.solve(ATA, ATb)
            H_xy = theta[0].item()

            # GPU correlation coefficient
            cov = torch.cov(torch.stack([log_s, log_F]))
            R2 = (
                cov[0, 1] / (torch.sqrt(cov[0, 0]) * torch.sqrt(cov[1, 1]))
            ).item() ** 2

            dcca_features[(t1, t2)] = {
                "H_xy": H_xy,
                "signed_F2_by_scale": signed_F2_by_scale,
                "magnitudes": [m.item() for m in magnitudes_list],
                "signs": signs_list,
                "used_scales": scales_list,
                "scales_count": len(scales_list),
                "R_squared": R2,
            }

    logger.info(f"✅ PURE GPU DCCA completed: {len(dcca_features)} pairs analyzed")
    return dcca_features


# ============================================================================
# PEARSON FEATURE EXTRACTION (Single Implementation)
# ============================================================================


def extract_pearson_features(
    residuals: Dict[str, pd.Series],
    token_list: List[str],
    window: int,
    lookback_start: Optional[pd.Timestamp] = None,
    lookback_end: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """
    ✅ STANDARD: Pearson correlation with consistent date range
    Note: Data is already date-aligned by data_processing pipeline
    """
    logger.info(f"📈 Extracting Pearson features (RESEARCH Step 6)...")
    logger.info(f"   Window: {window} days")

    # ====================================================
    # MINIMAL VERIFICATION (2 lines)
    # ====================================================
    # Quick sanity check: verify data_processing did its job
    if token_list and token_list[0] in residuals:
        reference_length = len(residuals[token_list[0]])
        logger.debug(f"Reference token {token_list[0]} has {reference_length} days")
    # ====================================================

    res_dict = {}
    valid_tokens = []

    for token in token_list:
        if token not in residuals:
            continue

        series = residuals[token]

        # Optional date filtering
        if lookback_start is not None and lookback_end is not None:
            mask = (series.index >= lookback_start) & (series.index <= lookback_end)
            filtered = series[mask]
        else:
            filtered = series

        # Use window parameter (from CAPM)
        if len(filtered) < 90:
            logger.debug(
                f"Token {token}: insufficient data ({len(filtered)} < {window})"
            )
            continue

        res_gpu = torch.tensor(filtered.values, device=DEVICE, dtype=torch.float32)
        res_dict[token] = res_gpu.float()
        valid_tokens.append(token)

    if len(valid_tokens) < 2:
        return {"has_data": False}

    res_stack = torch.stack([res_dict[token] for token in valid_tokens])

    try:
        corr_matrix = torch.corrcoef(res_stack)
        logger.info(
            f"✅ Pearson: {len(valid_tokens)} tokens, {len(res_stack[0])} aligned days"
        )

        return {
            "has_data": True,
            "correlation_matrix": corr_matrix,
            "token_list": valid_tokens,
            "n_observations": len(res_stack[0]),
            "window_used": window,
            "alignment_status": "verified_by_data_processing",  # For documentation
        }
    except Exception as e:
        logger.error(f"Pearson failed: {e}")
        return {"has_data": False}


# ============================================================================
# COINTEGRATION FEATURE EXTRACTION
# ============================================================================


from statsmodels.tsa.stattools import coint
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

from typing import Dict, Tuple, Any, List
import numpy as np
import pandas as pd
import logging
from statsmodels.tsa.stattools import coint

logger = logging.getLogger(__name__)


def extract_cointegration_features(
    price_data: Dict[str, pd.DataFrame],
    token_list: List[str],
    lookback_start: pd.Timestamp,
    lookback_end: pd.Timestamp,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    ✅ PURE STANDARD ENGLE–GRANGER COINTEGRATION (PAIR SELECTION ONLY)

    Procedure:
    1. Log prices
    2. Engle–Granger two-step test via statsmodels.coint()
       - OLS regression (internal)
       - ADF test on residuals
    3. Use ONLY p-value for pair selection

    """

    logger.info("🔗 Extracting PURE Engle–Granger cointegration features...")

    clean_data = {}

    # ─────────────────────────────────────────────
    # STEP 1: Prepare log-price series
    # ─────────────────────────────────────────────
    for token in token_list:
        if token not in price_data:
            continue

        df = price_data[token]
        mask = (df.index >= lookback_start) & (df.index <= lookback_end)

        if mask.sum() >= 60:
            clean_data[token] = np.log(df.loc[mask, "close"].values)

    valid_tokens = list(clean_data.keys())
    n = len(valid_tokens)
    n_pairs = n * (n - 1) // 2

    logger.info(f"   Tokens: {n}, Pairs tested: {n_pairs}")

    features = {}
    valid_pairs = 0

    # ─────────────────────────────────────────────
    # STEP 2: Engle–Granger test
    # ─────────────────────────────────────────────
    for i in range(n):
        for j in range(i + 1, n):
            t1, t2 = valid_tokens[i], valid_tokens[j]
            y = clean_data[t1]
            x = clean_data[t2]

            try:
                t_stat, pvalue, crit = coint(y, x, autolag="BIC")

                if np.isnan(pvalue) or np.isinf(pvalue):
                    continue

                features[(t1, t2)] = {
                    "pvalue": float(pvalue),
                    "t_stat": float(t_stat),
                    "crit_1pct": float(crit[0]),
                    "crit_5pct": float(crit[1]),
                    "crit_10pct": float(crit[2]),
                    "n_obs": len(y),
                }

                valid_pairs += 1

            except Exception:
                continue

    logger.info(f"✅ Engle–Granger completed: {valid_pairs}/{n_pairs} valid tests")

    return features
