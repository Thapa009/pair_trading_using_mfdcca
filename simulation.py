import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from pandas.tseries.offsets import BDay
from config import CONFIG, DEVICE
from data_processing import load_all_token_data_cached
from capm import apply_capm_filter

from feature_extraction import (
    extract_mfdcca_features,
    extract_dcca_features,
    extract_pearson_features,
    extract_cointegration_features,
)

from pair_selection import (
    select_pairs_mfdcca,
    select_pairs_dcca,
    select_pairs_pearson,
    select_pairs_cointegration,
    ensure_gpu_tensor,
)

from trading import (
    apply_price_divergence_filter,
    simulate_pair_trades,
    calculate_performance_metrics,
    create_empty_metrics,
    create_weekly_result,
)


logger = logging.getLogger(__name__)


def calculate_index_daily_returns(
    benchmark_data: pd.DataFrame, current_date: pd.Timestamp, week_end: pd.Timestamp
) -> Tuple[pd.Series, float]:
    """
    Robust index daily returns calculation
    """
    try:
        # Ensure indices are datetime
        if not isinstance(benchmark_data.index, pd.DatetimeIndex):
            benchmark_data = benchmark_data.copy()
            benchmark_data.index = pd.to_datetime(benchmark_data.index)

        # Filter to trading week
        mask = (benchmark_data.index >= current_date) & (
            benchmark_data.index <= week_end
        )
        week_data = benchmark_data.loc[mask]

        if len(week_data) < 2:
            return pd.Series(dtype=np.float64), 0.0

        # Ensure 'close' column exists
        if "close" not in week_data.columns:
            logger.error("No 'close' column in benchmark data")
            return pd.Series(dtype=np.float64), 0.0

        # Calculate returns safely
        close_prices = week_data["close"].astype(np.float64)
        daily_returns = close_prices.pct_change()

        # Remove NaN from first element
        daily_returns = (
            daily_returns.iloc[1:]
            if len(daily_returns) > 1
            else pd.Series(dtype=np.float64)
        )

        # Calculate weekly return
        if len(daily_returns) > 0:
            # Use np.nanprod to handle any potential NaN values
            weekly_return = np.nanprod(1 + daily_returns.fillna(0).to_numpy()) - 1
        else:
            weekly_return = 0.0

        return daily_returns, float(weekly_return)

    except Exception as e:
        logger.error(f"Index return calculation failed: {e}", exc_info=True)
        return pd.Series(dtype=np.float64), 0.0


def generate_trading_weeks(start_date, end_date, holding_period_days=5):
    """
    Generate weeks based on ACTUAL business days available in data
    """
    # Load sample data to see actual business days
    sample = load_all_token_data_cached(start_date, end_date, CONFIG["market_index"])
    if not sample:
        return []

    # Get actual business days from data
    actual_days = sample[CONFIG["market_index"]].index

    # Group into weeks of holding_period_days
    weeks = []
    for i in range(0, len(actual_days), holding_period_days):
        if i + holding_period_days <= len(actual_days):
            week_start = actual_days[i]
            week_end = actual_days[i + holding_period_days - 1]
            weeks.append((week_start, week_end))

    return weeks


def get_lookback_and_cutoff(current_date: pd.Timestamp, target_lookback: int):
    """
    Returns: lookback_start, lookback_end, information_cutoff
    """
    # Ensure current_date is Monday
    if current_date.weekday() != 0:
        current_date -= pd.Timedelta(days=current_date.weekday())

    # information_cutoff is the last available business day before current_date
    information_cutoff = current_date - BDay(1)

    # Lookback start/end
    lookback_end = information_cutoff
    lookback_start = lookback_end - BDay(target_lookback - 1)

    return lookback_start, lookback_end, information_cutoff


def load_all_required_data(
    current_date: pd.Timestamp,
    week_end: pd.Timestamp,
    market_index: str,
    divergence_lookback: int,
    target_lookback: int = 250,
) -> Optional[Dict[str, Any]]:
    """
    ✅ FIXED: Correct date ranges for divergence data
    """
    try:
        # ===============================
        # 1. VALIDATE & CORRECT WEEK DAYS
        # ===============================
        if current_date.weekday() != 0:  # Not Monday
            days_to_monday = current_date.weekday()
            current_date = current_date - pd.Timedelta(days=days_to_monday)
            logger.debug(f"Adjusted to Monday: {current_date.date()}")

        if week_end.weekday() != 4:  # Not Friday
            days_to_friday = (4 - week_end.weekday()) % 7
            week_end = week_end + pd.Timedelta(days=days_to_friday)
            logger.debug(f"Adjusted to Friday: {week_end.date()}")

        logger.info(f"📅 Trading week: {current_date.date()} to {week_end.date()}")

        # ===============================
        # 2. CALCULATE LOOKBACK & CUTOFF
        # ===============================
        lookback_start, lookback_end, information_cutoff = get_lookback_and_cutoff(
            current_date, target_lookback
        )

        # ===============================
        # 3. LOAD ALL DATA
        # ===============================
        # Lookback data (for feature calculation)
        price_data_lookback = load_all_token_data_cached(
            lookback_start, lookback_end, market_index
        )

        # ✅ CRITICAL FIX: Divergence data should END on information_cutoff
        # For N-day returns, need N+1 prices
        if divergence_lookback > 0:
            # ✅ CORRECT: End on information_cutoff (not cutoff - 1)
            divergence_end = information_cutoff

            # ✅ CORRECT: Start N business days before end
            # This gives us N+1 prices for N-day return calculation
            divergence_start = divergence_end - BDay(divergence_lookback)

            price_data_divergence = load_all_token_data_cached(
                divergence_start, divergence_end, market_index
            )

            logger.info(
                f"📊 Divergence data: {divergence_lookback} days "
                f"({divergence_start.date()} to {divergence_end.date()}, "
                f"{divergence_lookback + 1} prices needed)"
            )
        else:
            price_data_divergence = {}

        # Trading data for next week
        price_data_trade = load_all_token_data_cached(
            current_date, week_end, market_index
        )

        # ===============================
        # 4. VALIDATE DATA
        # ===============================
        if not price_data_lookback or not price_data_trade:
            logger.warning("Missing required price data")
            return None

        # ===============================
        # 5. RETURN ALL DATA
        # ===============================
        return {
            "price_data_lookback": price_data_lookback,
            "price_data_divergence": price_data_divergence,
            "price_data_trade": price_data_trade,
            "lookback_start": lookback_start,
            "lookback_end": lookback_end,
            "information_cutoff": information_cutoff,
            "current_date": current_date,
            "week_end": week_end,
        }

    except Exception as e:
        logger.error(f"Data loading failed: {e}", exc_info=True)
        return None


def compute_live(
    method: str,
    lookback_start: pd.Timestamp,
    lookback_end: pd.Timestamp,
    information_cutoff: pd.Timestamp,
    params: Dict[str, Any],
    price_data_lookback: Dict[str, pd.DataFrame],
) -> Tuple[List[Tuple[str, str]], Dict[str, pd.Series]]:
    """
    ✅ RESEARCH CORRECT: Compute features live with proper error handling
    Returns: (selected_pairs, residuals)
    """
    # ✅ INDEX: Return immediately - no computation needed
    if method == "index":
        logger.info("INDEX method - skipping all feature extraction")
        return [], {}

    # Initialize return values
    selected_pairs: List[Tuple[str, str]] = []
    residuals: Dict[str, pd.Series] = {}

    # ✅ CRITICAL ADDITION: Validate information cutoff matches lookback_end
    if information_cutoff != lookback_end:
        logger.warning(
            f"⚠️  information_cutoff ({information_cutoff.date()}) != lookback_end ({lookback_end.date()})"
        )
        logger.warning("   This could indicate data leakage risk!")

    # ✅ CRITICAL: Ensure price data doesn't exceed information_cutoff
    for token, df in price_data_lookback.items():
        if len(df) > 0:
            last_date = df.index[-1]
            if last_date > information_cutoff:
                logger.error(f"❌ DATA LEAKAGE DETECTED for {token}")
                logger.error(
                    f"   Last data point: {last_date.date()} > information_cutoff: {information_cutoff.date()}"
                )
                logger.error(f"   Truncating data to prevent leakage")
                # Truncate data to information_cutoff
                price_data_lookback[token] = df[df.index <= information_cutoff]

    try:
        # ============================================================================
        # METHODS USING CAPM RESIDUALS (MFDCCA, DCCA, Pearson)
        # ============================================================================
        if method in ["mfdcca", "dcca", "pearson"]:
            logger.info(f"{method}: Using CAPM residuals for analysis")
            logger.info(f"   Information cutoff: {information_cutoff.date()}")

            # ✅ Ensure CAPM doesn't use data beyond cutoff
            capm_results = apply_capm_filter(
                tokens=CONFIG["token_names"],
                market_index=CONFIG["market_index"],
                price_data=price_data_lookback,
            )

            if not capm_results:
                logger.warning(f"{method}: CAPM filtering failed")
                return [], {}

            # ✅ Validate CAPM results don't exceed cutoff
            for token in capm_results:
                if "residuals" in capm_results[token]:
                    residuals_last_date = capm_results[token]["residuals"].index[-1]
                    if residuals_last_date > information_cutoff:
                        logger.error(f"❌ CAPM residuals exceed cutoff for {token}")
                        return [], {}

            # Extract residuals
            residuals = {
                t: capm_results[t]["residuals"]
                for t in capm_results
                if "residuals" in capm_results[t]
            }

            if not residuals:
                logger.warning(f"{method}: No residuals available")
                return [], {}

            # Get actual days from CAPM
            sample_token = next(iter(capm_results.keys()))
            actual_days = capm_results[sample_token]["common_days_used"]
            logger.info(f"✅ {method}: Data length = {actual_days} days (from CAPM)")

            # Feature extraction on residuals
            if method == "mfdcca":
                features = extract_mfdcca_features(
                    residuals=residuals,
                    token_list=CONFIG["token_names"],
                    q_list=CONFIG["q_list"],
                    lookback_start=lookback_start,
                    lookback_end=lookback_end,
                )

                if features.get("has_data", False):
                    selected_pairs = select_pairs_mfdcca(
                        features=features,
                        pair_hxy_threshold=params["pair_hxy_threshold"],
                        threshold_h=params["threshold_h"],
                        threshold_alpha=params["threshold_alpha"],
                        token_list=features["tokens_used"],
                    )

            elif method == "dcca":
                features = extract_dcca_features(
                    residuals=residuals,
                    token_list=CONFIG["token_names"],
                    window=actual_days,
                )
                selected_pairs = select_pairs_dcca(
                    features=features,
                    pair_hxy_threshold=params["pair_hxy_threshold"],
                    token_list=CONFIG["token_names"],
                )

            elif method == "pearson":
                features = extract_pearson_features(
                    residuals=residuals,
                    token_list=CONFIG["token_names"],
                    window=actual_days,
                    lookback_start=lookback_start,
                    lookback_end=lookback_end,
                )

                if features.get("has_data", False):
                    selected_pairs = select_pairs_pearson(
                        features=features,
                        rho_threshold=float(params["rho_threshold"]),
                    )

            logger.info(f"✅ {method}: {len(selected_pairs)} pairs selected")
            return selected_pairs, residuals

        # ============================================================================
        # COINTEGRATION (uses raw price data, not residuals)
        # ============================================================================
        elif method == "cointegration":
            logger.info("Cointegration: Direct analysis on raw price data (NO CAPM)")

            # ✅ Cointegration on raw prices (NO residuals)
            features = extract_cointegration_features(
                price_data=price_data_lookback,
                token_list=CONFIG["token_names"],
                lookback_start=lookback_start,
                lookback_end=lookback_end,
            )

            selected_pairs = select_pairs_cointegration(
                features=features,
                pval_threshold=params["pval_threshold"],
                token_list=CONFIG["token_names"],
            )

            # ✅ CORRECT: Cointegration doesn't use residuals, return empty dict
            logger.info(f"✅ Cointegration: {len(selected_pairs)} pairs selected")
            return selected_pairs, {}

    except Exception as e:
        logger.error(f"Feature extraction failed for {method}: {e}", exc_info=True)
        return [], {}

    # ✅ ADD THIS: Final fallback return for any unexpected code path
    logger.error(f"Unexpected code path reached for method: {method}")
    return [], {}


def try_load_cache(
    cache_file: Path,
    information_cutoff: pd.Timestamp,
    method: str,
    params: Dict[str, Any],
) -> Tuple[bool, List, Dict]:
    """
    ✅ FIXED: Consistent GPU tensor handling
    Returns: (cache_hit, selected_pairs, residuals)
    """
    if not cache_file.exists():
        logger.debug(f"Cache file not found: {cache_file}")
        return False, [], {}

    try:
        cache_data = torch.load(
            cache_file,
            map_location=DEVICE,
            weights_only=False,
        )

        # Validation - use safer comparison for timestamps
        cache_cutoff = cache_data.get("information_cutoff")
        if cache_cutoff is None:
            logger.debug("No information_cutoff in cache")
            return False, [], {}

        # Convert to Timestamp for comparison
        try:
            cache_cutoff_ts = pd.Timestamp(cache_cutoff)
            info_cutoff_ts = pd.Timestamp(information_cutoff)

            if cache_cutoff_ts != info_cutoff_ts:
                logger.debug(f"Cache mismatch: {cache_cutoff_ts} != {info_cutoff_ts}")
                return False, [], {}
        except Exception as e:
            logger.debug(f"Timestamp conversion failed: {e}")
            return False, [], {}

        if cache_data.get("skipped", False):
            logger.debug("Cache marked as skipped")
            return False, [], {}

        # ✅ EXTRACT PRICE DATA FROM CACHE (for all methods)
        price_data_lookback: Dict[str, pd.DataFrame] = cache_data.get(
            "price_data_lookback", {}
        )
        if not price_data_lookback:
            logger.debug("No price data in cache")
            return False, [], {}

        selected_pairs: List[Any] = []

        if method == "mfdcca":
            if not cache_data.get("has_mfdcca_data", False):
                logger.debug("No MFDCCA data in cache")
                return False, [], {}

            # Get residuals
            residuals: Dict[str, pd.Series] = cache_data.get("residuals", {})
            if not residuals:
                logger.debug("No residuals for MFDCCA in cache")
                return False, [], {}

            # ✅✅✅ CONSISTENT GPU TENSOR HANDLING
            hxy_matrix = ensure_gpu_tensor(cache_data.get("hxy_matrix"))
            delta_H_matrix = ensure_gpu_tensor(cache_data.get("delta_H_matrix"))
            delta_alpha_matrix = ensure_gpu_tensor(cache_data.get("delta_alpha_matrix"))

            # Validate tensors
            if (
                hxy_matrix is None
                or delta_H_matrix is None
                or delta_alpha_matrix is None
            ):
                logger.debug("Missing MFDCCA matrices in cache")
                return False, [], {}

            # Get hurst_dict
            hurst_dict = cache_data.get("method_specific_data", {}).get(
                "hurst_dict", {}
            )
            if not hurst_dict:
                hurst_dict = cache_data.get("hurst_dict", {})

            features = {
                "has_data": True,
                "hurst_dict": hurst_dict,
                "hxy_matrix": hxy_matrix,
                "delta_H_matrix": delta_H_matrix,
                "delta_alpha_matrix": delta_alpha_matrix,
                "tokens_used": cache_data.get("tokens_used", CONFIG["token_names"]),
            }

            selected_pairs = select_pairs_mfdcca(
                features=features,
                pair_hxy_threshold=params["pair_hxy_threshold"],
                threshold_h=params["threshold_h"],
                threshold_alpha=params["threshold_alpha"],
                token_list=features["tokens_used"],
            )

            return True, selected_pairs, residuals

        elif method == "dcca":
            # ✅ Apply same pattern for DCCA
            if not cache_data.get("has_dcca_data", False):
                logger.debug("No DCCA data in cache")
                return False, [], {}

            residuals: Dict[str, pd.Series] = cache_data.get("residuals", {})
            if not residuals:
                logger.debug("No residuals for DCCA in cache")
                return False, [], {}

            # Use ensure_gpu_tensor for any DCCA tensors if needed
            # (Assuming dcca_features is a dict, not tensor)
            features = cache_data.get("dcca_features", {})
            if not features:
                logger.debug("Empty DCCA features in cache")
                return False, [], {}

            selected_pairs = select_pairs_dcca(
                features=features,
                pair_hxy_threshold=params["pair_hxy_threshold"],
                token_list=CONFIG["token_names"],
            )

            return True, selected_pairs, residuals

        elif method == "pearson":
            # ✅ Apply same pattern for Pearson
            if not cache_data.get("has_pearson_data", False):
                logger.debug("No Pearson data in cache")
                return False, [], {}

            residuals: Dict[str, pd.Series] = cache_data.get("residuals", {})
            if not residuals:
                logger.debug("No residuals for Pearson in cache")
                return False, [], {}

            # ✅ Consistent tensor handling
            corr_matrix = ensure_gpu_tensor(cache_data.get("correlation_matrix"))

            if corr_matrix is None:
                logger.debug("No correlation matrix in cache")
                return False, [], {}

            token_list = cache_data.get("token_list", CONFIG["token_names"])
            if not token_list:
                token_list = CONFIG["token_names"]

            features = {
                "has_data": True,
                "correlation_matrix": corr_matrix,
                "token_list": token_list,
            }

            selected_pairs = select_pairs_pearson(
                features=features,
                rho_threshold=params.get("rho_threshold", 0.7),  # ✅ Add default
            )

            return True, selected_pairs, residuals

        elif method == "cointegration":
            # ✅ Cointegration doesn't need residuals
            if not cache_data.get("has_cointegration_data", False):
                logger.debug("No cointegration data in cache")
                return False, [], {}

            features = cache_data.get("cointegration_features", {})
            if not features:
                logger.debug("Empty cointegration features in cache")
                return False, [], {}

            selected_pairs = select_pairs_cointegration(
                features=features,
                pval_threshold=params["pval_threshold"],
                token_list=CONFIG["token_names"],
            )

            # ✅ CORRECT: Return empty residuals dict for cointegration
            return True, selected_pairs, {}

        else:
            logger.debug(f"Unknown method in cache: {method}")
            return False, [], {}

    except Exception as e:
        logger.error(f"Cache load failed: {e}", exc_info=True)
        return False, [], {}


def run_simulation(
    fold_number: int,
    method: str,
    tune_mode: bool = False,
    use_precompute: bool = False,
    **params: Any,
) -> Dict[str, Dict[str, Any]]:
    """
    ✅ CORRECTED: Main simulation function with all fixes
    """

    # ========================================================================
    # 0. VALIDATION
    # ========================================================================
    if fold_number < 1 or fold_number > len(CONFIG["walk_forward_periods"]):
        logger.error(f"Invalid fold_number: {fold_number}")
        return {method: create_empty_metrics()}

    # ========================================================================
    # 1. INITIALIZATION
    # ========================================================================
    period = CONFIG["walk_forward_periods"][fold_number - 1]
    target_lookback = CONFIG["window"]
    holding_period_days = CONFIG["holding_period_days"]

    # Determine period
    if tune_mode:
        start_date, end_date = period["training_period"]
        period_name = "train"
    else:
        start_date, end_date = period["test_period"]
        period_name = "test"

    # Initialize variables
    weekly_results = []
    weekly_periods = generate_trading_weeks(start_date, end_date, holding_period_days)

    if not weekly_periods:
        logger.error("No weekly periods generated!")
        return {method: create_empty_metrics()}

    # Setup cache
    cache_dir = None
    if use_precompute and method != "index":
        cache_dir = (
            Path(CONFIG["results_dir"]) / "precompute" / f"fold_{fold_number}" / method
        )

    # Log divergence parameters
    divergence_lookback = int(params.get("divergence_lookback", 5))
    divergence_threshold = float(params.get("divergence_threshold", 0.10))

    # Logging
    logger.info(f"\n{'='*80}")
    logger.info(f"SIMULATION: {method.upper()} - {period_name.upper()}")
    logger.info(f"{'='*80}")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Weeks: {len(weekly_periods)}")
    logger.info(f"Cache: {cache_dir if cache_dir else 'None'}")
    logger.info(f"Divergence Lookback: {divergence_lookback} days")
    logger.info(f"Divergence Threshold: {divergence_threshold:.2%}")
    logger.info(f"{'='*80}\n")

    # ========================================================================
    # 2. PROCESS EACH WEEK
    # ========================================================================
    for week_number, (current_date, week_end) in enumerate(weekly_periods, 1):
        logger.info(f"\n{'─'*60}")
        logger.info(
            f"Week {week_number}/{len(weekly_periods)}: {current_date.date()} to {week_end.date()}"
        )
        logger.info(f"{'─'*60}")

        # ✅ SINGLE CALL: Get ALL data + dates
        data_result = load_all_required_data(
            current_date=current_date,
            week_end=week_end,
            market_index=CONFIG["market_index"],
            divergence_lookback=divergence_lookback,
            target_lookback=target_lookback,  # 250
        )

        if not data_result:
            logger.warning(
                f"Week {week_number}: Data loading failed - recording zero returns"
            )
            weekly_results.append(
                create_weekly_result(
                    week_number=week_number,
                    week_start=current_date,
                    week_end=week_end,
                    lookback_start=current_date,  # Fallback values
                    lookback_end=week_end,
                    information_cutoff=current_date,
                    num_selected=0,
                    num_filtered=0,
                    weekly_return_pct=0.0,
                    daily_returns=pd.Series(dtype=float),
                )
            )
            continue

        # ✅ Extract everything from the single result
        price_data_lookback = data_result["price_data_lookback"]
        price_data_divergence = data_result["price_data_divergence"]
        price_data_trade = data_result["price_data_trade"]
        lookback_start = data_result["lookback_start"]
        lookback_end = data_result["lookback_end"]
        information_cutoff = data_result["information_cutoff"]
        # current_date and week_end are already available

        logger.info(
            f"Week {week_number}/{len(weekly_periods)} | "
            f"Trading: {current_date.date()}→{week_end.date()} | "
            f"Lookback: {lookback_start.date()}→{lookback_end.date()} | "
            f"Cutoff: {information_cutoff.date()}"
        )
        # Check data availability
        if not price_data_lookback or not price_data_trade:
            logger.warning(f"Week {week_number}: Missing data - recording zero returns")
            weekly_results.append(
                create_weekly_result(
                    week_number=week_number,
                    week_start=current_date,
                    week_end=week_end,
                    lookback_start=lookback_start,
                    lookback_end=lookback_end,
                    information_cutoff=information_cutoff,
                    num_selected=0,
                    num_filtered=0,
                    weekly_return_pct=0.0,
                    daily_returns=pd.Series(dtype=float),
                )
            )
            continue

        # ====================================================================
        # 3. PAIR SELECTION (FIXED VERSION)
        # ====================================================================
        selected_pairs = []
        filtered_pairs = []
        residuals = {}

        if method == "index":
            logger.info("INDEX: No pair selection")
        else:
            # Try cache first
            cache_hit = False
            if use_precompute and cache_dir and cache_dir.exists():
                cache_file = cache_dir / f"week_{week_number}_{current_date.date()}.pt"

                # ✅ FIXED: Use underscore for unused residuals_cache
                cache_hit, selected_pairs, residuals = try_load_cache(
                    cache_file, information_cutoff, method, params
                )

                if cache_hit:
                    logger.info(f"✅ Cache: {len(selected_pairs)} pairs")

            # Compute live if no cache
            if not cache_hit:
                logger.info("Computing live...")
                # ✅ FIXED: Use residuals variable
                selected_pairs, residuals = compute_live(
                    method,
                    lookback_start,
                    lookback_end,
                    information_cutoff,
                    params,
                    price_data_lookback=price_data_lookback,
                )

                logger.info(f"Selected: {len(selected_pairs)} pairs")

        # ====================================================================
        # 4. DIVERGENCE FILTER - USE PRICE DATA FOR ALL METHODS
        # ====================================================================

        if method != "index" and selected_pairs:
            if divergence_lookback > 0 and divergence_threshold > 0:
                # ✅ CRITICAL FIX: ALL METHODS use price data for divergence filter
                if price_data_divergence and len(price_data_divergence) > 0:
                    filtered_pairs = apply_price_divergence_filter(
                        candidate_pairs=selected_pairs,
                        price_data=price_data_divergence,  # Always use price data
                        lookback_days=divergence_lookback,
                        divergence_threshold=divergence_threshold,
                    )
                    logger.info(
                        f"✅ Applied price-based divergence filter for {method}"
                    )
                else:
                    logger.warning(f"No divergence price data available for {method}")
                    filtered_pairs = selected_pairs
            else:
                filtered_pairs = selected_pairs
                logger.info("No divergence filter applied")
        else:
            filtered_pairs = selected_pairs

        # ====================================================================
        # 5. TRADING EXECUTION (FIXED - No unbound variables)
        # ====================================================================

        # ✅ ALWAYS initialize variables first
        daily_profits = pd.Series(dtype=float)
        weekly_profit_pct = 0.0

        # INDEX method
        if method == "index":
            benchmark_name = CONFIG["market_index"]
            if benchmark_name in price_data_trade:
                daily_profits, weekly_return = calculate_index_daily_returns(
                    price_data_trade[benchmark_name], current_date, week_end
                )
                weekly_profit_pct = weekly_return * 100
            else:
                logger.error(f"INDEX {benchmark_name} not found!")
                # Variables already initialized to empty/default values

        # Trading methods with pairs
        elif filtered_pairs:
            valid_pairs = []
            for pair_info in filtered_pairs:
                if isinstance(pair_info, dict) and "long_token" in pair_info:
                    valid_pairs.append(pair_info)
                elif isinstance(pair_info, tuple) and len(pair_info) == 2:
                    token1, token2 = pair_info
                    valid_pairs.append(
                        {
                            "pair": (token1, token2),
                            "long_token": token1,
                            "short_token": token2,
                            "divergence": 0.0,
                            "cum_ret1": 0.0,
                            "cum_ret2": 0.0,
                        }
                    )

            if valid_pairs:
                trade_results = simulate_pair_trades(
                    valid_pairs, price_data_trade, current_date, week_end
                )
                weekly_return = trade_results["Weekly_Return"]
                daily_profits = trade_results["Daily_Returns"]
                weekly_profit_pct = weekly_return * 100
                logger.info(
                    f"Traded {len(valid_pairs)} pairs, Return: {weekly_profit_pct:.2f}%"
                )
            else:
                logger.info("No valid pairs to trade")
                # Variables already initialized to empty/default values

        else:
            logger.info("No pairs to trade")
        # ====================================================================
        # 6. STORE RESULTS
        # ====================================================================
        weekly_results.append(
            create_weekly_result(
                week_number=week_number,
                week_start=current_date,
                week_end=week_end,
                lookback_start=lookback_start,
                lookback_end=lookback_end,
                information_cutoff=information_cutoff,
                num_selected=len(selected_pairs),
                num_filtered=len(filtered_pairs),
                weekly_return_pct=weekly_profit_pct,
                daily_returns=daily_profits,
            )
        )

    # ========================================================================
    # 7. CALCULATE METRICS
    # ========================================================================
    if not weekly_results:
        logger.error("No results!")
        return {method: create_empty_metrics()}

    results_dir = Path(CONFIG["results_dir"]) / f"fold_{fold_number}" / method
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = calculate_performance_metrics(
        weekly_results_list=weekly_results,
        result_dir=results_dir,
        period_name=period_name,
    )

    # Save weekly results
    weekly_df = pd.DataFrame(weekly_results)
    weekly_csv = results_dir / f"{period_name}_weekly_results.csv"
    weekly_df.to_csv(weekly_csv, index=False)

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"COMPLETE: {method.upper()} ({period_name})")
    logger.info(f"Weeks: {len(weekly_results)}")
    logger.info(
        f"Divergence Params: lookback={divergence_lookback}d, threshold={divergence_threshold:.2%}"
    )
    logger.info(f"Sharpe: {metrics.get('Sharpe_Ratio', 0):.4f}")
    logger.info(f"Max Drawdown: {metrics.get('Max_Drawdown_%', 0):.2f}%")
    logger.info(f"{'='*80}\n")

    return {method: metrics}
