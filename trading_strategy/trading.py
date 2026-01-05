import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
from pandas.tseries.offsets import BDay

logger = logging.getLogger(__name__)


def apply_price_divergence_filter(
    candidate_pairs: List[Tuple[str, str]],
    price_data: Dict[str, pd.DataFrame],
    lookback_days: int = 5,
    divergence_threshold: float = 0.10,
) -> List[Dict[str, Any]]:
    """
    ✅ CORRECTED: Check if pairs have diverged > threshold over N days

    Parameters:
    - lookback_days: N days for cumulative return calculation
    - divergence_threshold: Minimum divergence percentage

    IMPORTANT: Need N+1 prices for N-day cumulative return!
    """
    filtered_pairs = []

    logger.info(
        f"Applying divergence filter: {lookback_days} days, {divergence_threshold:.1%} threshold"
    )

    for token1, token2 in candidate_pairs:
        # Check if both tokens exist
        if token1 not in price_data or token2 not in price_data:
            continue

        # Get price series
        prices1 = price_data[token1]["close"]
        prices2 = price_data[token2]["close"]

        # ✅ CRITICAL FIX: Need N+1 prices for N-day cumulative return
        # For 5-day return, need 6 prices: day -5 to day 0
        if len(prices1) < lookback_days + 1 or len(prices2) < lookback_days + 1:
            logger.debug(
                f"Skipping {token1}-{token2}: insufficient data ({len(prices1)}, {len(prices2)} < {lookback_days + 1})"
            )
            continue

        try:
            # ✅ CORRECT: Calculate cumulative returns over N days
            # Use index -(N+1) for starting price
            start_idx = -(lookback_days + 1)

            # Convert to float for safety
            price1_start = float(prices1.iloc[start_idx])
            price1_end = float(prices1.iloc[-1])
            price2_start = float(prices2.iloc[start_idx])
            price2_end = float(prices2.iloc[-1])

            cum_ret1 = (price1_end / price1_start) - 1.0
            cum_ret2 = (price2_end / price2_start) - 1.0

            # Calculate divergence
            divergence = abs(cum_ret1 - cum_ret2)

            if divergence >= divergence_threshold:
                # Determine which token performed better
                if cum_ret1 > cum_ret2:
                    long_token, short_token = (
                        token2,
                        token1,
                    )  # Buy underperformer, short outperformer
                else:
                    long_token, short_token = token1, token2

                filtered_pairs.append(
                    {
                        "pair": (token1, token2),
                        "long_token": long_token,
                        "short_token": short_token,
                        "divergence": float(divergence),
                        "cum_ret1": float(cum_ret1),
                        "cum_ret2": float(cum_ret2),
                        "lookback_days": lookback_days,
                    }
                )

                logger.debug(
                    f"✓ {token1}-{token2}: divergence={divergence:.2%}, Long {long_token}, Short {short_token}"
                )

        except (ValueError, TypeError, KeyError, IndexError) as e:
            logger.debug(f"Error calculating divergence for {token1}-{token2}: {e}")
            continue

    logger.info(
        f"Divergence filter: {len(filtered_pairs)}/{len(candidate_pairs)} pairs passed"
    )

    # Debug info
    if filtered_pairs:
        avg_divergence = np.mean([p["divergence"] for p in filtered_pairs])
        logger.info(f"Average divergence: {avg_divergence:.2%}")
        logger.info(
            f"Range: {min([p['divergence'] for p in filtered_pairs]):.2%} to {max([p['divergence'] for p in filtered_pairs]):.2%}"
        )

    return filtered_pairs


def simulate_pair_trades(
    filtered_pairs,
    price_data,
    week_start,
    week_end,
):
    """
    ✅ RESEARCH STANDARD (NO TRANSACTION COSTS)
    Weekly rebalancing with daily close-to-close P&L
    - Monday: Enter at close
    - Tuesday–Friday: Daily P&L
    - Friday: Exit at close
    """

    # Get trading days from price data (using a sample token)
    sample_tokens = list(price_data.keys())
    if not sample_tokens:
        return {
            "Weekly_Return": 0.0,
            "Daily_Returns": pd.Series(dtype=float),
            "Active_Pairs": 0,
        }

    trading_days = price_data[sample_tokens[0]].index

    # Filter for the specific week
    week_mask = (trading_days >= week_start) & (trading_days <= week_end)
    week_days = trading_days[week_mask]

    if len(week_days) < 2:  # Need at least 2 days for returns
        return {
            "Weekly_Return": 0.0,
            "Daily_Returns": pd.Series(dtype=float, index=week_days),
            "Active_Pairs": 0,
        }

    daily_returns = []

    # ✅ MONDAY: Entry day (no cost, no return)
    daily_returns.append(0.0)

    # ✅ TUESDAY–FRIDAY: Daily P&L
    for idx in range(1, len(week_days)):
        current_day = week_days[idx]
        prev_day = week_days[idx - 1]

        day_portfolio_return = 0.0
        active_pairs = 0

        for pair_info in filtered_pairs:
            token_long = pair_info["long_token"]
            token_short = pair_info["short_token"]

            if (
                token_long not in price_data
                or token_short not in price_data
                or prev_day not in price_data[token_long].index
                or current_day not in price_data[token_long].index
                or prev_day not in price_data[token_short].index
                or current_day not in price_data[token_short].index
            ):
                continue

            try:
                # Ensure we get numeric values
                price_long_prev = float(price_data[token_long].loc[prev_day, "close"])
                price_long_curr = float(
                    price_data[token_long].loc[current_day, "close"]
                )
                price_short_prev = float(price_data[token_short].loc[prev_day, "close"])
                price_short_curr = float(
                    price_data[token_short].loc[current_day, "close"]
                )

                # Calculate returns with explicit float conversion
                r_long = (price_long_curr / price_long_prev) - 1.0
                r_short = (price_short_curr / price_short_prev) - 1.0

                # Equal-weighted pair return
                pair_daily_return = (r_long - r_short) / 2.0

                day_portfolio_return += pair_daily_return
                active_pairs += 1

            except (ValueError, TypeError, ZeroDivisionError) as e:
                logger.debug(
                    f"Error calculating returns for {token_long}-{token_short}: {e}"
                )
                continue

        if active_pairs > 0:
            daily_returns.append(day_portfolio_return / active_pairs)
        else:
            daily_returns.append(0.0)

    # Weekly compounded return
    if daily_returns:
        weekly_return = np.prod([1.0 + r for r in daily_returns]) - 1.0
    else:
        weekly_return = 0.0

    # Create Series with proper index
    daily_returns_series = pd.Series(
        daily_returns, index=week_days[: len(daily_returns)]
    )

    return {
        "Weekly_Return": weekly_return,
        "Daily_Returns": daily_returns_series,
        "Active_Pairs": len(filtered_pairs),
    }


def calculate_performance_metrics(
    weekly_results_list: List[Dict[str, Any]],
    result_dir: Path,
    holding_period_days: int = 5,
    period_name: str = "test",
    trading_days_per_year: float = 252.0,
) -> Dict[str, Any]:
    """
    RESEARCH-CORRECT PERFORMANCE METRICS (CRYPTO, MARKET-NEUTRAL)

    Metrics (STANDARD DEFINITIONS):
    - Sharpe Ratio   = mean(daily return) / std(daily return) * sqrt(252)
    - Sortino Ratio  = mean(daily return) / downside std * sqrt(252)
    - Max Drawdown   = max peak-to-trough loss
    - Profit Factor  = gross profit / gross loss
    - Calmar Ratio   = annualized return / max drawdown

    NOTES:
    - Uses RAW daily returns (NO risk-free rate)
    - Weekly rebalanced portfolio
    - Suitable for yearly benchmark comparison
    """

    # ============================================================
    # 1. VALIDATION
    # ============================================================
    if not weekly_results_list:
        return create_empty_metrics()

    # ============================================================
    # 2. COLLECT DAILY RETURNS
    # ============================================================
    all_daily_returns = {}

    for week in weekly_results_list:
        daily_returns = week.get("Daily_Returns")
        week_start = week["Week_Start"]
        week_end = week["Week_End"]

        if isinstance(daily_returns, pd.Series) and len(daily_returns) > 0:
            for d, r in daily_returns.items():
                all_daily_returns[d] = r
        else:
            # fallback (rare)
            for d in pd.bdate_range(week_start, week_end)[:holding_period_days]:
                all_daily_returns[d] = 0.0

    if not all_daily_returns:
        return create_empty_metrics()

    # ============================================================
    # 3. DAILY RETURN SERIES
    # ============================================================
    returns_series = pd.Series(all_daily_returns).sort_index()
    returns = returns_series.to_numpy(dtype=np.float64)
    returns = np.nan_to_num(returns)

    n_days = len(returns)
    if n_days < 2:
        return create_empty_metrics()

    # ============================================================
    # 4. CORE CALCULATIONS
    # ============================================================
    cumulative_returns = np.cumprod(1 + returns)
    total_return = cumulative_returns[-1] - 1

    # ============================================================
    # 4.1 SHARPE RATIO
    # ============================================================
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    sharpe_ratio = (
        (mean_return / std_return) * np.sqrt(trading_days_per_year)
        if std_return > 1e-12
        else 0.0
    )

    # ============================================================
    # 4.2 SORTINO RATIO
    # ============================================================
    downside_returns = returns[returns < 0]
    downside_std = (
        np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0.0
    )

    sortino_ratio = (
        (mean_return / downside_std) * np.sqrt(trading_days_per_year)
        if downside_std > 1e-12
        else 0.0
    )

    # ============================================================
    # 4.3 MAXIMUM DRAWDOWN
    # ============================================================
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (running_max - cumulative_returns) / running_max
    max_drawdown = np.max(drawdowns)

    # ============================================================
    # 4.4 PROFIT FACTOR
    # ============================================================
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = np.abs(np.sum(returns[returns < 0]))

    if gross_loss > 1e-12:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = np.inf
    else:
        profit_factor = 0.0

    # ============================================================
    # 4.5 CALMAR RATIO
    # ============================================================
    annualized_return = (1 + total_return) ** (trading_days_per_year / n_days) - 1

    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 1e-12 else 0.0

    # ============================================================
    # 5. SAVE ANALYSIS
    # ============================================================
    analysis_df = pd.DataFrame(
        {
            "Date": returns_series.index,
            "Daily_Return": returns,
            "Cumulative_Return": cumulative_returns,
            "Drawdown_%": drawdowns * 100,
        }
    )
    analysis_df.to_csv(result_dir / f"{period_name}_analysis.csv", index=False)

    # ============================================================
    # 6. RETURN METRICS
    # ============================================================
    return {
        "Sharpe_Ratio": round(sharpe_ratio, 4),
        "Sortino_Ratio": round(sortino_ratio, 4),
        "Max_Drawdown_%": round(max_drawdown * 100, 4),
        "Profit_Factor": round(profit_factor, 4),
        "Calmar_Ratio": round(calmar_ratio, 4),
        "Total_Return_%": round(total_return * 100, 4),
        "Annualized_Return_%": round(annualized_return * 100, 4),
        "Total_Trading_Days": n_days,
    }


def create_empty_metrics() -> Dict[str, Any]:
    """Return zero metrics for failed cases"""
    return {
        "Sharpe_Ratio": 0.0,
        "Sortino_Ratio": 0.0,
        "Max_Drawdown_%": 0.0,
        "Profit_Factor": 0.0,
        "Calmar_Ratio": 0.0,
        "Total_Return_%": 0.0,
        "Annualized_Return_%": 0.0,
        "Total_Trading_Days": 0,
    }


def create_weekly_result(
    week_number: int,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
    lookback_start: pd.Timestamp,
    lookback_end: pd.Timestamp,
    information_cutoff: pd.Timestamp,
    num_selected: int,
    num_filtered: int,
    weekly_return_pct: float,
    daily_returns: Union[pd.Series, float],
) -> Dict[str, Any]:
    """
    Helper function to create standardized weekly result dict
    """
    return {
        "Week_Number": week_number,
        "Week_Start": week_start,
        "Week_End": week_end,
        "Lookback_Start": lookback_start,
        "Lookback_End": lookback_end,
        "Information_Cutoff": information_cutoff,
        "Num_Selected_Pairs": num_selected,
        "Num_Filtered_Pairs": num_filtered,
        "Weekly_Return_%": weekly_return_pct,
        "Daily_Returns": daily_returns,
    }
