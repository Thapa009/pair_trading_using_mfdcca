import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import pandas as pd
import torch
from pandas.tseries.offsets import BDay  # <-- ADD THIS IMPORT
from config import CONFIG, DEVICE
from data_processing import load_all_token_data_cached
from capm import apply_capm_filter
from simulation import get_lookback_and_cutoff

from feature_extraction import (
    extract_mfdcca_features,
    extract_dcca_features,
    extract_pearson_features,
    extract_cointegration_features,
)

from simulation import generate_trading_weeks

logger = logging.getLogger(__name__)


def create_cache_data(
    method,
    residuals,
    price_data_lookback,
    aligned_capm_results,
    current_date,
    lookback_start,
    lookback_end,
    information_cutoff,
    method_specific_data,
):
    """Create cache data structure with GPU tensors"""
    base_cache = {
        "method": method,
        "residuals": residuals,
        "price_data_lookback": price_data_lookback,
        "week_start": current_date,
        "lookback_start": lookback_start,
        "lookback_end": lookback_end,
        "information_cutoff": information_cutoff,
        "skipped": False,
        "cache_version": "5.0_gpu",
        "window_size": CONFIG["window"],
        "cached_date": pd.Timestamp.now(),
        "device": str(DEVICE),
    }

    if aligned_capm_results and len(aligned_capm_results) > 0:
        sample_token = next(iter(aligned_capm_results.keys()))
        base_cache["actual_days"] = aligned_capm_results[sample_token].get(
            "actual_days", 0
        )

    if method_specific_data:
        base_cache.update(method_specific_data)

    return base_cache


def precompute_training_features(
    fold_number: int, method: str, period_start: pd.Timestamp, period_end: pd.Timestamp
):
    """
    Pre-compute features for all weeks in the training period.

    This function runs ONCE per method per fold to generate cached features
    that will be reused across all Optuna trials.

    Args:
        fold_number: Fold index (1-based)
        method: Trading method (mfdcca, dcca, pearson, cointegration, index)
        period_start: Training period start date
        period_end: Training period end date
    """
    logger.info(f"🔧 PRE-COMPUTE: Starting for fold {fold_number}, method {method}")
    logger.info(f"   Period: {period_start.date()} to {period_end.date()}")

    cache_dir = (
        Path(CONFIG["results_dir"]) / "precompute" / f"fold_{fold_number}" / method
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    target_lookback = CONFIG["window"]
    holding_period_days = CONFIG["holding_period_days"]

    # Generate weekly periods
    weekly_periods = generate_trading_weeks(
        period_start, period_end, holding_period_days
    )

    if not weekly_periods:
        logger.error("No weekly periods generated!")
        return

    logger.info(f"   Weeks to precompute: {len(weekly_periods)}")

    # Process each week
    for week_number, (current_date, week_end) in enumerate(weekly_periods, 1):
        cache_file = cache_dir / f"week_{week_number}_{current_date.date()}.pt"

        if cache_file.exists():
            logger.info(f"✅ Week {week_number}: Cache exists, skipping")
            continue

        logger.info(
            f"⚙️  Week {week_number}/{len(weekly_periods)}: {current_date.date()}"
        )

        lookback_start, lookback_end, information_cutoff = get_lookback_and_cutoff(
            current_date, target_lookback
        )

        logger.debug(
            f"   Lookback: {lookback_start.date()} → {lookback_end.date()} "
            f"(cutoff: {information_cutoff.date()})"
        )

        # Load price data
        price_data_lookback = load_all_token_data_cached(
            lookback_start,
            lookback_end,
            CONFIG["market_index"],
        )

        if not price_data_lookback:
            logger.warning(f"   Week {week_number}: No data available")
            torch.save({"skipped": True}, cache_file)
            continue

        # Initialize storage
        residuals = {}
        aligned_capm_results = {}
        method_specific_data = {}

        # ========================================================================
        # FEATURE EXTRACTION BY METHOD
        # ========================================================================

        if method == "index":
            # INDEX method needs no computation
            method_specific_data = {"has_index_data": True}
            logger.info(f"   Week {week_number}: INDEX method (no features)")

        elif method == "cointegration":
            # Cointegration uses PRICE DATA directly (NO CAPM)
            logger.info(f"   Week {week_number}: Extracting cointegration features...")

            sample_token = next(iter(price_data_lookback.keys()))
            actual_days = len(price_data_lookback[sample_token])

            features = extract_cointegration_features(
                price_data=price_data_lookback,
                token_list=CONFIG["token_names"],
                lookback_start=lookback_start,
                lookback_end=lookback_end,
            )

            method_specific_data = {
                "has_cointegration_data": len(features) > 0,
                "cointegration_features": features,
                "actual_days": actual_days,
            }

            # No residuals for cointegration
            residuals = {}
            aligned_capm_results = {}

            logger.info(f"   ✅ {len(features)} cointegrated pairs cached")

        else:
            # Methods using CAPM residuals: mfdcca, dcca, pearson
            logger.info(f"   Week {week_number}: Running CAPM filtering...")

            capm_results = apply_capm_filter(
                tokens=CONFIG["token_names"],
                market_index=CONFIG["market_index"],
                price_data=price_data_lookback,
            )

            if not capm_results:
                logger.warning(f"   Week {week_number}: CAPM failed")
                torch.save({"skipped": True}, cache_file)
                continue

            # Extract residuals
            sample_token = next(iter(capm_results.keys()))
            actual_days = capm_results[sample_token]["common_days_used"]

            for token in capm_results:
                if "residuals" in capm_results[token]:
                    residuals[token] = capm_results[token]["residuals"]
                    aligned_capm_results[token] = {
                        "residuals": residuals[token],
                        "beta": capm_results[token]["beta"],
                        "alpha": capm_results[token]["alpha"],
                        "actual_days": actual_days,
                    }

            if not residuals:
                logger.warning(f"   Week {week_number}: No valid residuals")
                torch.save({"skipped": True}, cache_file)
                continue

            logger.debug(f"   CAPM: {len(residuals)} tokens, {actual_days} days")

            # Extract method-specific features
            if method == "mfdcca":
                logger.info(f"   Week {week_number}: Extracting MFDCCA features...")

                features = extract_mfdcca_features(
                    residuals=residuals,
                    token_list=CONFIG["token_names"],
                    q_list=CONFIG["q_list"],
                    lookback_start=lookback_start,
                    lookback_end=lookback_end,
                )

                if features.get("has_data", False):

                    method_specific_data = {
                        "has_mfdcca_data": True,
                        "hxy_matrix": features[
                            "hxy_matrix"
                        ],  # Already numpy from extract_mfdcca_features
                        "delta_H_matrix": features["delta_H_matrix"],
                        "delta_alpha_matrix": features["delta_alpha_matrix"],
                        "q_list": features["q_list"],
                        "tokens_used": features["tokens_used"],  # ✅ Store tokens_used
                    }
                    logger.info(
                        f"   ✅ MFDCCA cached: {features['hxy_matrix'].shape[0]} tokens"
                    )
                else:
                    method_specific_data = {"has_mfdcca_data": False}
                    logger.warning(f"   Week {week_number}: MFDCCA extraction failed")

            elif method == "dcca":
                logger.info(f"   Week {week_number}: Extracting DCCA features...")

                features = extract_dcca_features(
                    residuals=residuals,
                    token_list=CONFIG["token_names"],
                    window=actual_days,
                )

                method_specific_data = {
                    "has_dcca_data": len(features) > 0,
                    "dcca_features": features,
                }
                logger.info(f"   ✅ DCCA cached: {len(features)} pairs")

            elif method == "pearson":
                logger.info(f"   Week {week_number}: Extracting Pearson features...")

                features = extract_pearson_features(
                    residuals=residuals,
                    token_list=CONFIG["token_names"],
                    window=actual_days,
                )

                if features.get("has_data", False):
                    method_specific_data = {
                        "has_pearson_data": True,
                        "correlation_matrix": features["correlation_matrix"],
                        "token_list": features["token_list"],
                    }
                    logger.info(
                        f"   ✅ Pearson cached: {features['correlation_matrix'].shape[0]} tokens"
                    )
                else:
                    method_specific_data = {"has_pearson_data": False}
                    logger.warning(f"   Week {week_number}: Pearson extraction failed")

        # ========================================================================
        # SAVE CACHE
        # ========================================================================
        cache_data = create_cache_data(
            method,
            residuals,
            price_data_lookback,
            aligned_capm_results,
            current_date,
            lookback_start,
            lookback_end,
            information_cutoff,
            method_specific_data,
        )

        try:
            torch.save(cache_data, cache_file)
            logger.debug(f"   💾 Cache saved: {cache_file.name}")
        except Exception as e:
            logger.error(f"   ❌ Failed to save cache: {e}")

    logger.info(
        f"✅ PRE-COMPUTE COMPLETE: fold {fold_number}, method {method} "
        f"({len(weekly_periods)} weeks cached)"
    )
