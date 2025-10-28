import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import torch
from tqdm import tqdm
from config import CONFIG
from config import DEVICE
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

def create_empty_metrics():
    """Return empty metrics for failed cases"""
    return {
        'Sharpe_Ratio': None,
        'Sortino_Ratio': None,
        'Max_Drawdown_%': None,
        'Win_Rate_%': None,
        'Profit_Factor': None,
        'Calmar_Ratio': None,
    }

def apply_divergence_filter(
    candidate_pairs: List[Tuple[str, str]], 
    residuals: Dict[str, torch.Tensor],
    lookback_days: int, 
    threshold: float, 
    current_idx: int
) -> List[Tuple]:
    """
    FIXED: GPU-accelerated divergence filter with perfect alignment
    """
    if not candidate_pairs:
        return []
    
    # ✅ FIXED: Pure integer indexing
    end_idx = current_idx  # Use current_idx directly (already the last index)
    start_idx = max(0, end_idx - lookback_days + 1)
    
    if start_idx >= end_idx:
        logger.warning(f"Insufficient lookback: start={start_idx}, end={end_idx}")
        return []
    
    logger.debug(f"Divergence filter: indices {start_idx} to {end_idx}, threshold={threshold}")
    
    # ✅ STEP 1: Stack all residuals [n_tokens, T] - STAYS ON GPU
    tokens = list(residuals.keys())
    token_to_idx = {t: i for i, t in enumerate(tokens)}
    
    # Ensure all tensors are on GPU and aligned
    res_list = []
    valid_tokens = []
    for t in tokens:
        res = residuals[t]
        if not isinstance(res, torch.Tensor):
            res = torch.tensor(res, device=DEVICE, dtype=torch.float32)
        elif res.device != DEVICE:
            res = res.to(DEVICE)
        
        # Check if we have enough data
        if len(res) > end_idx:
            res_list.append(res)
            valid_tokens.append(t)
    
    if not res_list:
        logger.warning("No valid residuals for divergence filter")
        return []
    
    res_stack = torch.stack(res_list)  # [n_tokens, T]
    token_to_idx = {t: i for i, t in enumerate(valid_tokens)}
    
    # ✅ STEP 2: Create pair indices [n_pairs, 2] - ON GPU
    pair_indices = []
    valid_pairs = []
    for t1, t2 in candidate_pairs:
        if t1 in token_to_idx and t2 in token_to_idx:
            pair_indices.append([token_to_idx[t1], token_to_idx[t2]])
            valid_pairs.append((t1, t2))
    
    if not pair_indices:
        logger.warning("No valid pairs for divergence filter")
        return []
    
    pair_indices = torch.tensor(pair_indices, device=DEVICE, dtype=torch.long)
    
    # ✅ STEP 3: Extract lookback slices [n_pairs, lookback_days] - GPU
    res1 = res_stack[pair_indices[:, 0], start_idx:end_idx+1]
    res2 = res_stack[pair_indices[:, 1], start_idx:end_idx+1]
    
    # Validate data availability
    actual_days = res1.size(1)
    if actual_days < lookback_days * 0.8:
        logger.warning(f"Only {actual_days} days available, need {lookback_days}")
        return []
    
    # ✅ STEP 4: Cumulative sums - GPU
    cum1 = res1.sum(dim=1)  # [n_pairs]
    cum2 = res2.sum(dim=1)  # [n_pairs]
    
    # ✅ STEP 5: Divergence check - GPU
    diff = torch.abs(cum1 - cum2)  # [n_pairs]
    diverged_mask = diff > threshold  # [n_pairs] boolean
    
    # ✅ STEP 6: Determine trade directions - GPU
    buy_token1 = cum1 < cum2  # [n_pairs] boolean
    
    # ✅ STEP 7: Build results (MINIMAL CPU conversion)
    filtered_pairs = []
    diverged_indices = diverged_mask.nonzero(as_tuple=True)[0]
    
    for i in diverged_indices:
        idx = int(i.item())
        t1, t2 = valid_pairs[idx]
        diff_val = float(diff[idx].item())
        
        if buy_token1[idx]:
            # Buy t1 (underperformed), short t2 (overperformed)
            filtered_pairs.append((
                t1, t2, diff_val, 
                f"{t1} buy, {t2} short", 
                t1, t2  # low_token, high_token
            ))
        else:
            # Buy t2 (underperformed), short t1 (overperformed)
            filtered_pairs.append((
                t2, t1, diff_val, 
                f"{t2} buy, {t1} short", 
                t2, t1  # low_token, high_token
            ))
    
    logger.info(f"✅ GPU divergence filter: {len(filtered_pairs)}/{len(valid_pairs)} pairs")
    return filtered_pairs

def apply_cointegration_divergence_filter(
    candidate_pairs: List[Tuple[str, str]], 
    price_data: Dict[str, torch.Tensor],
    lookback_days: int, 
    threshold: float, 
    current_idx: int
) -> List[Tuple]:
    """
    FIXED: Cointegration-specific divergence filter using SPREAD z-scores
    """
    if not candidate_pairs:
        return []
    
    # ✅ FIXED: Pure integer indexing with price data
    end_idx = current_idx  # Last index in PRICE series
    start_idx = max(0, end_idx - lookback_days + 1)
    
    if start_idx >= end_idx:
        logger.warning(f"Insufficient lookback: start={start_idx}, end={end_idx}")
        return []
    
    logger.debug(f"Cointegration filter: price indices {start_idx} to {end_idx}, threshold={threshold}")
    
    filtered_pairs = []
    
    for t1, t2 in candidate_pairs:
        if t1 not in price_data or t2 not in price_data:
            continue
            
        # Extract price series
        prices1 = price_data[t1]
        prices2 = price_data[t2]
        
        # Check if we have enough data
        if len(prices1) <= end_idx or len(prices2) <= end_idx:
            continue
            
        # Use log prices for cointegration
        log_p1 = torch.log(prices1[start_idx:end_idx+1])
        log_p2 = torch.log(prices2[start_idx:end_idx+1])
        
        actual_days = len(log_p1)
        if actual_days < lookback_days * 0.8:
            continue
            
        try:
            # Calculate cointegration relationship on lookback window
            # y = log_p2, X = [log_p1, ones]
            X = torch.stack([log_p1, torch.ones_like(log_p1)], dim=1)
            y = log_p2.unsqueeze(1)
            
            # OLS: log_p2 = beta * log_p1 + alpha
            coeffs = torch.linalg.lstsq(X, y, driver='gels').solution
            beta, alpha = coeffs[0, 0], coeffs[1, 0]
            
            # Calculate cointegration spread
            spread = log_p2 - (beta * log_p1 + alpha)
            
            # Calculate z-score of current spread
            current_spread = spread[-1]  # Most recent spread
            historical_mean = torch.mean(spread[:-1])  # Mean of historical spreads
            historical_std = torch.std(spread[:-1])    # Std of historical spreads
            
            if historical_std > 1e-8:  # Avoid division by zero
                z_score = torch.abs((current_spread - historical_mean) / historical_std)
                
                # If spread is significantly deviated, include for mean reversion
                if z_score > threshold:
                    # Determine trade direction based on spread deviation
                    if current_spread > historical_mean:
                        # Spread is wide: short the expensive token (t2), buy the cheap token (t1)
                        filtered_pairs.append((
                            t1, t2, float(z_score.item()),
                            f"{t1} buy, {t2} short (z={z_score:.2f})",
                            t1, t2
                        ))
                    else:
                        # Spread is narrow: buy the cheap token (t2), short the expensive token (t1)
                        filtered_pairs.append((
                            t2, t1, float(z_score.item()),
                            f"{t2} buy, {t1} short (z={z_score:.2f})",
                            t2, t1
                        ))
                        
        except Exception as e:
            logger.warning(f"Cointegration divergence failed for {t1}-{t2}: {e}")
            continue
    
    logger.info(f"✅ Cointegration filter: {len(filtered_pairs)}/{len(candidate_pairs)} pairs")
    return filtered_pairs

def simulate_pair_trades(
    filtered_pairs, price_data, holding_period_days, transaction_costs,
    start_date, end_date, divergence_threshold
):
    """FIXED: Vectorized pair trading simulation"""
    
    if not filtered_pairs:
        return [], 0.0, [], pd.DataFrame(), 0.0
    
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    # Extract common dates across all pairs
    all_dates = None
    for t1, t2, *_ in filtered_pairs:
        if t1 in price_data and t2 in price_data:
            pair_dates = price_data[t1].index.intersection(price_data[t2].index)
            if all_dates is None:
                all_dates = pair_dates
            else:
                all_dates = all_dates.intersection(pair_dates)
    
    if all_dates is None or len(all_dates) == 0:
        logger.warning("No common dates found for trading pairs")
        return [], 0.0, [], pd.DataFrame(), 0.0
    
    # Filter to trading period
    trading_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]
    trading_dates = trading_dates.sort_values()
    
    if len(trading_dates) < 2:
        logger.warning("Insufficient trading days")
        return [], 0.0, [], pd.DataFrame(), 0.0
    
    n_pairs = len(filtered_pairs)
    n_dates = len(trading_dates)
    
    # Equal weight portfolio
    weight = 1.0 / n_pairs if n_pairs > 0 else 0.0
    
    # Stack all prices into tensors [n_pairs, n_dates]
    prices_long = torch.zeros(n_pairs, n_dates, device=DEVICE)
    prices_short = torch.zeros(n_pairs, n_dates, device=DEVICE)
    
    valid_pairs = []
    for i, (t1, t2, *_, low_token, high_token) in enumerate(filtered_pairs):
        if low_token in price_data and high_token in price_data:
            try:
                df_low = price_data[low_token].loc[trading_dates, 'close']
                df_high = price_data[high_token].loc[trading_dates, 'close']
                
                if len(df_low) == n_dates and len(df_high) == n_dates:
                    prices_long[i] = torch.tensor(df_low.values, device=DEVICE, dtype=torch.float32)
                    prices_short[i] = torch.tensor(df_high.values, device=DEVICE, dtype=torch.float32)
                    valid_pairs.append((t1, t2, low_token, high_token))
                else:
                    logger.warning(f"Price data length mismatch for {t1}-{t2}")
            except Exception as e:
                logger.warning(f"Failed to process pair {t1}-{t2}: {e}")
                continue
    
    if not valid_pairs:
        logger.warning("No valid pairs for trading simulation")
        return [], 0.0, [], pd.DataFrame(), 0.0
    
    # Trim to valid pairs only
    n_valid_pairs = len(valid_pairs)
    prices_long = prices_long[:n_valid_pairs]
    prices_short = prices_short[:n_valid_pairs]
    weight = 1.0 / n_valid_pairs
    
    # Vectorized returns calculation
    ret_long = (prices_long[:, 1:] / prices_long[:, :-1]) - 1
    ret_short = -((prices_short[:, 1:] / prices_short[:, :-1]) - 1)  # Negative for short positions
    
    # Daily pair returns [n_pairs, n_dates-1]
    daily_pair_returns = (ret_long + ret_short) / 2
    
    # Apply transaction costs (amortized over holding period)
    daily_costs = transaction_costs / holding_period_days
    daily_pair_returns = daily_pair_returns - daily_costs
    
    # Portfolio return (equal weighted)
    weighted_returns = (daily_pair_returns * weight).sum(dim=0)
    cumulative_return = torch.prod(1 + weighted_returns) - 1
    portfolio_return = cumulative_return.item()
    
    # Convert back to CPU for output
    daily_returns_cpu = daily_pair_returns.cpu().numpy()
    weighted_returns_cpu = weighted_returns.cpu().numpy()
    
    # Build portfolio results
    portfolio = []
    pair_details = []
    total_costs = transaction_costs * n_valid_pairs
    
    for i, (t1, t2, low_token, high_token) in enumerate(valid_pairs):
        pair_cum_return = np.prod(1 + daily_returns_cpu[i]) - 1
        pair_cum_return_net = pair_cum_return - transaction_costs
        
        portfolio.append({
            'pair': (t1, t2),
            'buy_token': low_token,
            'short_token': high_token,
            'start_date': trading_dates[0],
            'end_date': trading_dates[-1],
            'return': pair_cum_return_net,
            'weight': weight
        })
        
        pair_details.append({
            'pair': f"{t1}-{t2}",
            'buy': low_token,
            'short': high_token,
            'weight': weight,
            'return': pair_cum_return_net
        })
    
    # Daily profits DataFrame
    daily_profits_dict = {'Date': trading_dates[1:], 'Portfolio_Return': weighted_returns_cpu}
    for i, (t1, t2, _, _) in enumerate(valid_pairs):
        daily_profits_dict[f"{t1}-{t2}"] = daily_returns_cpu[i]
    
    daily_profits = pd.DataFrame(daily_profits_dict)
    
    logger.info(f"✅ Trading simulation: {n_valid_pairs} pairs, return={portfolio_return:.4f}")
    return portfolio, portfolio_return, pair_details, daily_profits, total_costs

def calculate_performance_metrics(weekly_table, result_dir, holding_period_days: int):
    """
    CORRECTED: Consistent financial mathematics for trading performance
    """
    
    if isinstance(weekly_table, pd.DataFrame):
        weekly_table = weekly_table.to_dict('records')
    
    if not weekly_table:
        logger.warning("No weekly results for performance metrics")
        return create_empty_metrics()
    
    # ============================================================================
    # STEP 1: Extract Returns (CONSISTENT: Always percentage → decimal)
    # ============================================================================
    returns_list = []
    for row in weekly_table:
        if pd.notna(row['Weekly_Profit_%']):
            # ALWAYS convert percentage to decimal
            returns_list.append(float(row['Weekly_Profit_%']) / 100)
    
    if len(returns_list) < 2:
        logger.warning("Insufficient returns for performance metrics")
        return create_empty_metrics()
    
    returns_tensor = torch.tensor(returns_list, device=DEVICE, dtype=torch.float32)
    n_periods = len(returns_tensor)
    
    # ============================================================================
    # STEP 2: Annualization Parameters
    # ============================================================================
    TRADING_DAYS_PER_YEAR = 252
    periods_per_year = TRADING_DAYS_PER_YEAR / holding_period_days
    rf_annual = CONFIG.get('risk_free_rate', 0.02)
    
    # ============================================================================
    # STEP 3: Returns Calculation (SIMPLE RETURNS CONSISTENTLY)
    # ============================================================================
    # Total return (geometric)
    cumulative_return = torch.prod(1 + returns_tensor)
    total_return = cumulative_return.item() - 1
    
    # Annualized return (geometric)
    if n_periods > 0:
        years = (n_periods * holding_period_days) / TRADING_DAYS_PER_YEAR
        ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    else:
        ann_return = 0.0
    
    # Arithmetic mean return (for Sharpe ratio)
    mean_return = returns_tensor.mean()
    ann_mean_return = mean_return * periods_per_year
    
    # ============================================================================
    # STEP 4: Volatility (SIMPLE RETURNS - Consistent with mean)
    # ============================================================================
    period_volatility = returns_tensor.std()
    ann_volatility = period_volatility * torch.sqrt(torch.tensor(periods_per_year, device=DEVICE))
    
    # ============================================================================
    # STEP 5: Sharpe Ratio (Arithmetic returns for consistency)
    # ============================================================================
    excess_ann_return = ann_mean_return - rf_annual
    sharpe_ratio = (excess_ann_return / ann_volatility).item() if ann_volatility > 1e-8 else 0.0
    
    # ============================================================================
    # STEP 6: Sortino Ratio (Proper downside deviation)
    # ============================================================================
    # Use minimum acceptable return = 0 (absolute Sortino) or risk-free rate
    mar = 0.0  # Minimum Acceptable Return
    
    downside_returns = returns_tensor[returns_tensor < mar]
    if len(downside_returns) > 0:
        downside_variance = (downside_returns ** 2).mean()
        downside_volatility = torch.sqrt(downside_variance) * torch.sqrt(torch.tensor(periods_per_year, device=DEVICE))
        sortino_ratio = ((ann_mean_return - rf_annual) / downside_volatility).item() if downside_volatility > 1e-8 else 0.0
    else:
        sortino_ratio = 0.0
    
    # ============================================================================
    # STEP 7: Maximum Drawdown (CORRECT)
    # ============================================================================
    cumulative = torch.cumprod(1 + returns_tensor, dim=0)
    running_max = torch.cummax(cumulative, dim=0)[0]
    drawdowns = (running_max - cumulative) / running_max
    max_drawdown = drawdowns.max().item()
    
    # ============================================================================
    # STEP 8: Win Rate (CORRECT)
    # ============================================================================
    win_rate = (returns_tensor > 0).float().mean().item() * 100
    
    # ============================================================================
    # STEP 9: Profit Factor (CORRECT)
    # ============================================================================
    winning_returns = returns_tensor[returns_tensor > 0]
    losing_returns = returns_tensor[returns_tensor < 0]
    
    if len(losing_returns) > 0 and losing_returns.sum().abs() > 1e-8:
        profit_factor = winning_returns.sum().item() / losing_returns.sum().abs().item()
    else:
        profit_factor = float('inf') if winning_returns.sum() > 0 else 0.0
    
    # ============================================================================
    # STEP 10: Calmar Ratio
    # ============================================================================
    calmar_ratio = (ann_return / max_drawdown) if max_drawdown > 1e-8 else 0.0
    
    # ============================================================================
    # STEP 11: Assemble Metrics (ENSURE NO None VALUES)
    # ============================================================================
    metrics = {
        'Sharpe_Ratio': round(float(sharpe_ratio), 4),
        'Sortino_Ratio': round(float(sortino_ratio), 4),
        'Max_Drawdown_%': round(max_drawdown * 100, 2),
        'Win_Rate_%': round(win_rate, 2),
        'Profit_Factor': round(float(profit_factor), 4) if profit_factor != float('inf') else 'Inf',
        'Calmar_Ratio': round(float(calmar_ratio), 4),
        'Total_Return_%': round(total_return * 100, 2),
        'Annualized_Return_%': round(ann_return * 100, 2),
        'Annualized_Volatility_%': round(ann_volatility.item() * 100, 2),
        'Number_of_Periods': n_periods,
        'Risk_Free_Rate': rf_annual
    }
    
    # Save metrics (your existing code is fine)
    if result_dir:
        try:
            metrics_path = Path(result_dir) / "performance_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save performance metrics: {e}")
    
    return metrics