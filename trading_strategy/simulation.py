# simulation.py - COMPLETE FIXED VERSION

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from config import CONFIG
from data_processing import load_all_token_data_cached
from capm import apply_capm_filter
from mfdcca import process_token_pairs_gpu, extract_hurst_matrices
from pair_selection import (
    select_pairs_mfdcca, select_pairs_dcca, 
    select_pairs_pearson, select_pairs_cointegration,
    select_pairs_dcca_precomputed,
    select_pairs_pearson_precomputed,
    select_pairs_cointegration_precomputed
)
from trading import apply_divergence_filter, simulate_pair_trades, calculate_performance_metrics, apply_cointegration_divergence_filter  
import torch
from config import DEVICE

logger = logging.getLogger(__name__)

def get_weekly_periods_fixed(start_date, end_date, holding_days):
    """FIXED: Get business day-aligned periods with NO look-ahead"""
    all_business_days = pd.bdate_range(start=start_date, end=end_date)
    periods = []
    
    for i in range(0, len(all_business_days), holding_days):
        week_start = all_business_days[i]
        week_end = all_business_days[min(i + holding_days - 1, len(all_business_days)-1)]
        
        actual_days = (week_end - week_start).days + 1
        if actual_days >= max(1, holding_days * 0.5):
            periods.append((week_start, week_end))
    
    logger.info(f"Generated {len(periods)} trading periods from {start_date.date()} to {end_date.date()}")
    return periods

def get_lookback_period_no_lookahead(current_date, target_lookback):
    """
    FIXED: CRITICAL - Ensure no look-ahead bias
    Information cutoff is the day BEFORE trading starts
    """
    information_cutoff = current_date - pd.offsets.BDay(1)
    lookback_end = information_cutoff
    lookback_start = information_cutoff - pd.offsets.BDay(target_lookback - 1)
    
    return lookback_start, lookback_end, information_cutoff

def run_simulation_fixed(
    fold_number: int, 
    method: str, 
    tune_mode: bool = False, 
    use_precompute: bool = False, 
    **params: Any
) -> Dict[str, List[Dict[str, Any]]]: 
    """
    FIXED: NO LOOK-AHEAD BIAS - Uses only data available before trading period
    """
    logger.info(f"Starting FIXED simulation: fold={fold_number}, method={method}, tune_mode={tune_mode}")
    
    # VALIDATION
    if fold_number < 1 or fold_number > len(CONFIG["walk_forward_periods"]):
        raise ValueError(f"Invalid fold_number {fold_number}")

    required_params = {
        "mfdcca": ['pair_hxy_threshold', 'threshold_h', 'threshold_alpha', 
                  'divergence_threshold', 'divergence_lookback_days'],
        "dcca": ['pair_hxy_threshold', 'divergence_threshold', 'divergence_lookback_days'],
        "pearson": ['rho_threshold', 'divergence_threshold', 'divergence_lookback_days'],
        "cointegration": ['pval_threshold', 'divergence_threshold', 'divergence_lookback_days'],
        "index": []
    }
    
    if method not in required_params:
        raise ValueError(f"Unknown method: {method}")
    
    missing = [p for p in required_params[method] if p not in params]
    if missing:
        raise ValueError(f"Missing required parameters for {method}: {missing}")
    
    # Get period configuration
    period = CONFIG["walk_forward_periods"][fold_number - 1]
    
    if tune_mode:
        period_start, period_end = period["training_period"]
        period_name = "training"
    else:
        period_start, period_end = period["test_period"]
        period_name = "test"
        use_precompute = False  # Never use precompute in test mode
    
    logger.info(f"FIXED {period_name.upper()} period: {period_start.date()} to {period_end.date()}")
    
    # Setup with flexible parameters
    target_lookback = CONFIG['window']
    min_lookback = CONFIG.get('min_lookback_days', 30)
    holding_period_days = CONFIG['holding_period_days']
    transaction_costs = CONFIG['transaction_costs']
    
    # FIXED: Use corrected period generation
    weekly_periods = get_weekly_periods_fixed(period_start, period_end, holding_period_days)
    
    if not weekly_periods:
        logger.error("No valid trading periods generated!")
        return {method: []}
    
    logger.info(f"FIXED: Processing {len(weekly_periods)} weeks with target {target_lookback} days lookback")
    
    # Results storage
    weekly_results = []
    
    # Cache setup for precompute (training only)
    cache_dir = None
    if tune_mode and use_precompute and method != "index":
        cache_dir = Path(CONFIG['results_dir']) / "precompute" / f"fold_{fold_number}" / method
    
    for week_number, (current_date, week_end) in enumerate(weekly_periods, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"FIXED Week {week_number}: {current_date.date()} to {week_end.date()}")
        logger.info(f"{'='*80}")
        
        # ✅ FIXED: CRITICAL - No look-ahead bias
        lookback_start, lookback_end, information_cutoff = get_lookback_period_no_lookahead(
            current_date, target_lookback
        )
        
        # ✅ FIXED: Adjust if lookback_start is before our data start
        data_start = CONFIG['start_date']
        if lookback_start < data_start:
            lookback_start = data_start
            logger.info(f"FIXED: Adjusted lookback_start to data start: {lookback_start.date()}")
        
        actual_lookback_days = (lookback_end - lookback_start).days + 1
        logger.info(f"FIXED lookback: {lookback_start.date()} to {lookback_end.date()} ({actual_lookback_days} days)")
        logger.info(f"FIXED information cutoff: {information_cutoff.date()} (day before trading)")
        
        # ✅ FIXED: Skip if insufficient data
        if actual_lookback_days < min_lookback:
            logger.warning(f"Week {week_number}: Insufficient lookback data ({actual_lookback_days} < {min_lookback} days). Skipping.")
            continue
            
        # Initialize defaults
        selected_pairs = []
        residuals = {}
        price_data_lookback = {}
        filtered_pairs = []
        actual_days_used = actual_lookback_days
        
        # ============================================================================
        # PRECOMPUTE LOAD PATH WITH CORRECTED TEMPORAL VALIDATION
        # ============================================================================
        cache_file = None
        cache_hit = False

        if tune_mode and use_precompute and cache_dir and cache_dir.exists():
            cache_file = cache_dir / f"week_{week_number}_{current_date.date()}.pt"
            
            if cache_file.exists():
                logger.info(f"Loading precomputed features from {cache_file}")
                
                try:
                    cache_data = torch.load(cache_file, map_location='cpu')
                    
                    # ✅ CRITICAL: Enhanced temporal validation
                    if cache_data.get('skipped', False):
                        logger.warning(f"Week {week_number}: Precompute skipped. Skipping week.")
                        continue
                    
                    # ✅ STEP 1: Validate cache version (FIXED)
                    cache_version = cache_data.get('cache_version', '0.0')
                    if cache_version == '0.0':
                        logger.warning(
                            f"Completely invalid cache version {cache_version}. "
                            f"Falling back to live computation."
                        )
                        cache_hit = False
                    # ✅ STEP 2: CRITICAL - Validate information cutoff
                    elif 'information_cutoff' not in cache_data:
                        logger.warning(
                            f"Cache missing 'information_cutoff' field. "
                            f"Cannot verify temporal integrity. Falling back to live computation."
                        )
                        cache_hit = False
                    
                    else:
                        cached_info_cutoff = cache_data['information_cutoff']
                        
                        # ✅ STEP 3: Compare information cutoffs
                        if cached_info_cutoff != information_cutoff:
                            logger.warning(
                                f"❌ TEMPORAL MISMATCH: "
                                f"cached cutoff = {cached_info_cutoff.date()}, "
                                f"current cutoff = {information_cutoff.date()}. "
                                f"Falling back to live computation."
                            )
                            cache_hit = False
                        
                        # ✅ STEP 4: Validate method matches
                        elif cache_data.get('method') != method:
                            logger.warning(
                                f"Method mismatch: cached={cache_data.get('method')}, "
                                f"current={method}. Falling back."
                            )
                            cache_hit = False
                        
                        # ✅ STEP 5: Validate no-lookahead certification
                        elif not cache_data.get('no_lookahead', False):
                            logger.warning(
                                f"Cache not certified as no-lookahead. Falling back."
                            )
                            cache_hit = False
                        
                        # ✅ STEP 6: ALL CHECKS PASSED - Safe to use cache
                        else:
                            residuals = cache_data.get('residuals', {})
                            price_data_lookback = cache_data.get('price_data_lookback', {})
                            
                            if not residuals:
                                logger.warning("Cache contains no residuals. Falling back.")
                                cache_hit = False
                            else:
                                # Validate residual length
                                sample_length = len(next(iter(residuals.values())))
                                actual_days_used = cache_data.get('actual_days_used', sample_length)
                                
                                if sample_length < min_lookback:
                                    logger.warning(
                                        f"Residual length too short: {sample_length} < {min_lookback}. "
                                        f"Falling back to live computation."
                                    )
                                    cache_hit = False
                                else:
                                    # Convert to GPU tensors if needed
                                    if not isinstance(next(iter(residuals.values())), torch.Tensor):
                                        residuals = {
                                            t: torch.tensor(res, device=DEVICE, dtype=torch.float32)
                                            for t, res in residuals.items()
                                        }
                                    
                                    # ✅ METHOD-SPECIFIC CACHE LOADING
                                    if method == "mfdcca":
                                        if not cache_data.get('has_mfdcca_data', False):
                                            logger.warning("Cache missing MFDCCA data. Falling back.")
                                            cache_hit = False
                                        else:
                                            hurst_dict = cache_data.get('hurst_dict', {})
                                            hxy_matrix_np = cache_data.get('hxy_matrix', np.array([]))
                                            hurst_matrix_np = cache_data.get('hurst_matrix', np.array([]))
                                            alpha_matrix_np = cache_data.get('alpha_matrix', np.array([]))
                                            
                                            if hxy_matrix_np.size == 0:
                                                logger.warning("MFDCCA hxy_matrix is empty. Falling back.")
                                                cache_hit = False
                                            else:
                                                hxy_matrix = torch.tensor(hxy_matrix_np, device=DEVICE)
                                                hurst_matrix = torch.tensor(hurst_matrix_np, device=DEVICE)
                                                alpha_matrix = torch.tensor(alpha_matrix_np, device=DEVICE)
                                                
                                                selected_pairs = select_pairs_mfdcca(
                                                    hurst_dict, hxy_matrix, hurst_matrix, alpha_matrix,
                                                    CONFIG['token_names'],
                                                    params['pair_hxy_threshold'],
                                                    params['threshold_h'],
                                                    params['threshold_alpha']
                                                )
                                                cache_hit = True
                                                logger.info(f"✅ Validated MFDCCA cache: {len(selected_pairs)} pairs")
                                    
                                    elif method == "dcca":
                                        if not cache_data.get('has_dcca_data', False):
                                            logger.warning("Cache missing DCCA data. Falling back.")
                                            cache_hit = False
                                        else:
                                            dcca_features = cache_data.get('dcca_features', {})
                                            selected_pairs = select_pairs_dcca_precomputed(
                                                CONFIG['token_names'],
                                                dcca_features,
                                                params['pair_hxy_threshold']
                                            )
                                            cache_hit = True
                                            logger.info(f"✅ Validated DCCA cache: {len(selected_pairs)} pairs")
                                    
                                    elif method == "pearson":
                                        if not cache_data.get('has_pearson_data', False):
                                            logger.warning("Cache missing Pearson data. Falling back.")
                                            cache_hit = False
                                        else:
                                            corr_matrix = cache_data.get('correlation_matrix', np.array([]))
                                            token_list = cache_data.get('valid_tokens', CONFIG['token_names'])
                                            selected_pairs = select_pairs_pearson_precomputed(
                                                CONFIG['token_names'],
                                                corr_matrix,
                                                token_list,
                                                params['rho_threshold']
                                            )
                                            cache_hit = True
                                            logger.info(f"✅ Validated Pearson cache: {len(selected_pairs)} pairs")
                                    
                                    elif method == "cointegration":
                                        if not cache_data.get('has_cointegration_data', False):
                                            logger.warning("Cache missing Cointegration data. Falling back.")
                                            cache_hit = False
                                        else:
                                            cointegration_features = cache_data.get('cointegration_features', {})
                                            selected_pairs = select_pairs_cointegration_precomputed(
                                                CONFIG['token_names'],
                                                cointegration_features,
                                                params['pval_threshold']
                                            )
                                            cache_hit = True
                                            logger.info(f"✅ Validated Cointegration cache: {len(selected_pairs)} pairs")
                                    
                                    elif method == "index":
                                        selected_pairs = []
                                        cache_hit = True
                                        logger.info("✅ Validated Index cache")
                
                except Exception as e:
                    logger.error(f"Failed to load/validate cache: {e}")
                    cache_hit = False

        # ============================================================================
        # ✅ CRITICAL FIX: COMPLETE LIVE COMPUTE PATH
        # ============================================================================
        if not cache_hit:
            logger.info(f"FIXED computing week {week_number} live...")
            
            # ✅ Load data only up to information cutoff
            price_data_lookback = load_all_token_data_cached(
                lookback_start, lookback_end, CONFIG['market_index']
            )

            if not price_data_lookback:
                logger.warning(f"Week {week_number}: No lookback data. Skipping.")
                continue

            logger.info(f"FIXED CAPM for week {week_number}...")
            capm_results = apply_capm_filter(
                tokens=CONFIG['token_names'],
                market_index=CONFIG['market_index'],
                price_data=price_data_lookback,
                start_date=lookback_start,
                end_date=lookback_end,  # ✅ Ends at information cutoff
                save_summary=False
            )
            
            if not capm_results:
                logger.warning(f"Week {week_number}: CAPM failed. Skipping.")
                continue
            
            # Track actual days used by CAPM
            sample_token = next(iter(capm_results.keys()))
            actual_days_used = capm_results[sample_token].get('actual_days_used', actual_lookback_days)
            logger.info(f"FIXED CAPM used {actual_days_used} trading days (up to {information_cutoff.date()})")
            
            # Use GPU residuals directly from CAPM
            residuals = {
                t: capm_results[t]['residuals_gpu'] 
                for t in capm_results 
                if 'residuals_gpu' in capm_results[t]
            }

            # ✅ LIVE PAIR SELECTION
            if method == "mfdcca":
                logger.info(f"FIXED MFDCCA for week {week_number}...")
                results = process_token_pairs_gpu(
                    token_list=CONFIG['token_names'],
                    residuals=residuals,
                    start_date=lookback_start,
                    end_date=lookback_end,  # ✅ Ends at information cutoff
                    q_list=CONFIG['q_list']
                )
                
                if results:
                    hurst_dict, hxy_matrix, delta_H_matrix, delta_alpha_matrix = \
                        extract_hurst_matrices(CONFIG['token_names'], results)

                    selected_pairs = select_pairs_mfdcca(
                        hurst_dict, hxy_matrix, delta_H_matrix, delta_alpha_matrix,
                        CONFIG['token_names'], 
                        params['pair_hxy_threshold'],
                        params['threshold_h'], 
                        params['threshold_alpha']
                    )
                else:
                    selected_pairs = []
                    logger.warning(f"Week {week_number}: MFDCCA produced no results")
            
            elif method == "dcca":
                logger.info(f"FIXED DCCA for week {week_number}...")
                selected_pairs = select_pairs_dcca(
                    CONFIG['token_names'],
                    residuals,
                    actual_days_used,
                    params['pair_hxy_threshold']
                )
            
            elif method == "pearson":
                logger.info(f"FIXED Pearson for week {week_number}...")
                selected_pairs = select_pairs_pearson(
                    CONFIG['token_names'],
                    residuals,
                    actual_days_used,
                    params['rho_threshold']
                )
            
            elif method == "cointegration":
                logger.info(f"FIXED Cointegration for week {week_number}...")
                selected_pairs = select_pairs_cointegration(
                    CONFIG['token_names'],
                    price_data_lookback,
                    actual_days_used,
                    params['pval_threshold']
                )
            
            elif method == "index":
                selected_pairs = []
            
            logger.info(
                f"FIXED Week {week_number}: Selected {len(selected_pairs)} pairs "
                f"using {actual_days_used} days (up to {information_cutoff.date()})"
            )

        # ============================================================================
        # DIVERGENCE FILTER (FIXED - uses data only up to information cutoff)
        # ============================================================================
        if selected_pairs and method != "index":
            if not residuals:
                logger.warning(f"Week {week_number}: No residuals for divergence filter")
                filtered_pairs = []
            else:
                # Use residuals length for divergence calculation
                residual_length = len(next(iter(residuals.values())))
                current_idx = residual_length - 1  # Last index in residuals (at information cutoff)
                lookback_days_needed = min(params['divergence_lookback_days'], residual_length - 1)
                
                if current_idx < lookback_days_needed:
                    logger.warning(
                        f"FIXED divergence: Have {current_idx+1} residuals, "
                        f"need {lookback_days_needed}. Using available {current_idx+1} days."
                    )
                    lookback_days_needed = current_idx
                
                if lookback_days_needed < 1:
                    logger.warning(f"Week {week_number}: Insufficient data for divergence. Skipping.")
                    filtered_pairs = []
                else:
                    # Cointegration uses separate logic with PRICE data
                    if method == "cointegration":
                        price_series = {}
                        candidate_tokens = set()
                        for t1, t2 in selected_pairs:
                            candidate_tokens.add(t1)
                            candidate_tokens.add(t2)
                        
                        for token in candidate_tokens:
                            if token in price_data_lookback:
                                prices = price_data_lookback[token]['close'].values
                                price_series[token] = torch.tensor(prices, device=DEVICE, dtype=torch.float32)
                        
                        if price_series:
                            price_length = len(next(iter(price_series.values())))
                            current_price_idx = price_length - 1  # At information cutoff
                            required_lookback = min(params['divergence_lookback_days'], price_length - 1)
                            
                            logger.info(f"FIXED Cointegration: price_length={price_length}, using {required_lookback} days up to {information_cutoff.date()}")
                            
                            if current_price_idx >= required_lookback:
                                filtered_pairs = apply_cointegration_divergence_filter(
                                    candidate_pairs=selected_pairs,
                                    price_data=price_series,
                                    lookback_days=required_lookback,
                                    threshold=params['divergence_threshold'],
                                    current_idx=current_price_idx
                                )
                            else:
                                logger.warning(f"Insufficient price data. Skipping divergence.")
                                filtered_pairs = []
                        else:
                            filtered_pairs = []
                    else:
                        # Other methods use residuals
                        filtered_pairs = apply_divergence_filter(
                            candidate_pairs=selected_pairs,
                            residuals=residuals,
                            lookback_days=lookback_days_needed,
                            threshold=params['divergence_threshold'],
                            current_idx=current_idx  # At information cutoff
                        )
        
        elif method == "index":
            # Index method: buy and hold the market index
            filtered_pairs = [('INDEX', 'INDEX')]

        # ============================================================================
        # LOAD TRADING DATA (FIXED - only for trading period)
        # ============================================================================
        price_data_trade = load_all_token_data_cached(current_date, week_end, CONFIG['market_index'])
        price_data_trade = {
            token: df.loc[current_date:week_end] 
            for token, df in price_data_trade.items() 
            if not df.empty
        }
        
        if not price_data_trade:
            logger.warning(f"Week {week_number}: No trading data. Skipping.")
            continue
            
        actual_trading_days = len(next(iter(price_data_trade.values())))
        logger.info(f"✅ FIXED trading: {actual_trading_days} days from {current_date.date()} to {week_end.date()}")
        
        # ============================================================================
        # EXECUTE TRADING (FIXED - uses only trading period data)
        # ============================================================================
        if filtered_pairs and method != "index":
            try:
                portfolio, portfolio_return, pair_details, daily_profits, week_costs = simulate_pair_trades(
                    filtered_pairs,
                    price_data_trade,
                    actual_trading_days,
                    transaction_costs,
                    current_date,
                    week_end,
                    params['divergence_threshold']
                )
                
                weekly_profit_pct = portfolio_return * 100
                logger.info(f"✅ FIXED Week {week_number}: {len(filtered_pairs)} pairs, return = {weekly_profit_pct:.2f}%")
                
            except Exception as e:
                logger.error(f"FIXED trading simulation failed: {e}")
                portfolio_return = 0.0
                weekly_profit_pct = 0.0
                week_costs = 0.0
                portfolio = []
                pair_details = []
                daily_profits = pd.DataFrame()
        
        elif method == "index":
            # Index method performance
            if 'INDEX' in price_data_trade:
                index_prices = price_data_trade['INDEX']['close']
                if len(index_prices) > 1:
                    portfolio_return = (index_prices.iloc[-1] / index_prices.iloc[0] - 1)
                    weekly_profit_pct = portfolio_return * 100
                    week_costs = 0.0
                    logger.info(f"✅ FIXED INDEX Week {week_number}: return = {weekly_profit_pct:.2f}%")
                else:
                    weekly_profit_pct = 0.0
                    week_costs = 0.0
                    logger.warning(f"Week {week_number}: Insufficient INDEX data")
            else:
                weekly_profit_pct = 0.0
                week_costs = 0.0
                logger.warning(f"Week {week_number}: No INDEX data")
        else:
            portfolio_return = 0.0
            weekly_profit_pct = 0.0
            week_costs = 0.0
            logger.warning(f"Week {week_number}: No trades executed")
        
        # Store results with FIXED metadata
        weekly_results.append({
            'Week_Number': week_number,
            'Week_Start': current_date,
            'Week_End': week_end,
            'Lookback_Start': lookback_start,
            'Lookback_End': lookback_end,
            'Information_Cutoff': information_cutoff,  # ✅ Track information cutoff
            'Actual_Lookback_Days': actual_days_used,
            'Target_Lookback_Days': target_lookback,
            'Trading_Days': actual_trading_days,
            'Target_Trading_Days': holding_period_days,
            'Num_Selected_Pairs': len(selected_pairs),
            'Num_Filtered_Pairs': len(filtered_pairs) if method != "index" else 1,
            'Pairs': [f"{p[0]}-{p[1]}" for p in filtered_pairs] if filtered_pairs else [],
            'Weekly_Profit_%': weekly_profit_pct,
            'Transaction_Costs': week_costs,
            'Data_Coverage_%': (actual_days_used / target_lookback * 100) if target_lookback > 0 else 0
        })
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ FIXED SIMULATION COMPLETE: {len(weekly_results)}/{len(weekly_periods)} weeks processed")
    logger.info(f"{'='*80}")
    
    if not weekly_results:
        logger.error("No weekly results generated!")
        return {method: []}
    
    # Calculate performance metrics
    results_dir = Path(CONFIG['results_dir']) / f"fold_{fold_number}" / method
    results_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = calculate_performance_metrics(
        weekly_results,
        results_dir,
        holding_period_days
    )
    
    # Save weekly results with FIXED analysis
    weekly_df = pd.DataFrame(weekly_results)
    weekly_csv_path = results_dir / f"{period_name}_weekly_results_fixed.csv"
    weekly_df.to_csv(weekly_csv_path, index=False)
    logger.info(f"FIXED weekly results saved to {weekly_csv_path}")
    
    # Calculate and log FIXED statistics
    if len(weekly_results) > 0:
        avg_lookback = weekly_df['Actual_Lookback_Days'].mean()
        avg_coverage = weekly_df['Data_Coverage_%'].mean()
        min_lookback = weekly_df['Actual_Lookback_Days'].min()
        max_lookback = weekly_df['Actual_Lookback_Days'].max()
        
        logger.info(f"📊 FIXED STATISTICS (NO LOOK-AHEAD BIAS):")
        logger.info(f"   Average lookback: {avg_lookback:.1f} days (target: {target_lookback})")
        logger.info(f"   Data coverage: {avg_coverage:.1f}%")
        logger.info(f"   Lookback range: {min_lookback}-{max_lookback} days")
        logger.info(f"   Information cutoff: Always day before trading")
        logger.info(f"   Successful weeks: {len(weekly_results)}/{len(weekly_periods)}")
    
    # Print performance summary
    logger.info(f"\n{'='*80}")
    logger.info(f"FIXED PERFORMANCE SUMMARY - NO LOOK-AHEAD ({period_name.upper()}) - {method.upper()}")
    logger.info(f"{'='*80}")
    logger.info(f"Total Weeks: {len(weekly_results)}")
    
    if metrics['Sharpe_Ratio'] is not None:
       logger.info(f"Sharpe Ratio: {metrics['Sharpe_Ratio']:.4f}")


    
    if metrics['Sortino_Ratio'] is not None:
        logger.info(f"Sortino Ratio: {float(metrics['Sortino_Ratio']):.4f}")
    if metrics['Max_Drawdown_%'] is not None:
        logger.info(f"Max Drawdown: {float(metrics['Max_Drawdown_%']):.2f}%")
    if metrics['Win_Rate_%'] is not None:
        logger.info(f"Win Rate: {float(metrics['Win_Rate_%']):.2f}%")
    if metrics['Profit_Factor'] is not None:
        logger.info(f"Profit Factor: {float(metrics['Profit_Factor']):.4f}")
    if metrics['Calmar_Ratio'] is not None:
        logger.info(f"Calmar Ratio: {float(metrics['Calmar_Ratio']):.4f}")
    logger.info(f"{'='*80}\n")
    
    return {method: [metrics]}

# Keep the original function name for compatibility, but call the fixed version
def run_simulation(*args, **kwargs):
    """Wrapper to maintain API compatibility"""
    return run_simulation_fixed(*args, **kwargs)