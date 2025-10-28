from joblib import Parallel, delayed
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from functools import reduce
from config import CONFIG

logger = logging.getLogger(__name__)

# Global cache for full data (loaded once)
_full_data_cache: Dict = {}

def load_single_preprocess(file_path: Path, label: str) -> Optional[pd.DataFrame]:
    """Load a single CSV with preprocessing."""
    logger.info(f"Loading {file_path} for label {label}")
    if not file_path.exists():
        logger.error("[%s] File not found: %s", label, file_path)
        return None
    
    try:
        df = pd.read_csv(file_path)
        date_col = CONFIG.get('date_column', 'Date')
        price_col = CONFIG.get('price_column', 'Price')
        
        if date_col not in df.columns or price_col not in df.columns:
            logger.error("[%s] Missing required columns", label)
            return None
        
        df = df[[date_col, price_col]].rename(columns={date_col: 'date', price_col: 'close'})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'].astype(str).str.replace(',', '', regex=False), errors='coerce')
        df = df.dropna(subset=['date', 'close'])
        
        if (df['close'] <= 0).any():
            logger.error("[%s] Non-positive prices detected", label)
            return None
        
        df = df[['date', 'close']].drop_duplicates(subset='date', keep='first').sort_values('date').reset_index(drop=True)
        return df
    
    except Exception as e:
        logger.exception("[%s] Failed to load file %s: %s", label, file_path, e)
        return None



def load_and_preprocess_cca(token: str, market_index: str):
    """Load token and market index data for CAPM with STRICT alignment."""
    token_df = load_single_preprocess(CONFIG['data_dir'] / f"{token}.csv", token)
    index_df = load_single_preprocess(CONFIG['data_dir'] / f"{market_index}.csv", "index")
    
    if token_df is None or index_df is None:
        return None, None
    
    token_df = token_df.set_index('date').sort_index()
    index_df = index_df.set_index('date').sort_index()
    
    # ✅ Remove duplicates
    token_df = token_df[~token_df.index.duplicated(keep="first")]
    index_df = index_df[~index_df.index.duplicated(keep="first")]
    
    # ✅ Find common dates
    common_dates = index_df.index.intersection(token_df.index)
    
    if len(common_dates) == 0:
        logger.error(f"No common dates between {token} and {market_index}")
        return None, None
    
    # ✅ CRITICAL: Slice to EXACT same dates
    token_df = token_df.loc[common_dates]
    index_df = index_df.loc[common_dates]
    
    # ✅ Verify alignment
    if len(token_df) != len(index_df):
        logger.error(f"Alignment failed: token={len(token_df)}, index={len(index_df)}")
        return None, None
    
    # ✅ Final validation: check for NaN in prices
    if token_df['close'].isna().any() or index_df['close'].isna().any():
        logger.warning(f"NaN values detected in {token}, dropping rows")
        valid_mask = ~(token_df['close'].isna() | index_df['close'].isna())
        token_df = token_df[valid_mask]
        index_df = index_df[valid_mask]
    
    return (
        token_df.rename(columns={'close': 'close_token'}).reset_index(),
        index_df.rename(columns={'close': 'close_index'}).reset_index()
    )


def _slice_token_cached(token: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Tuple[str, Optional[pd.DataFrame]]:
    """Helper function for parallel slicing - only defined once"""
    if token in _full_data_cache:
        df_token = _full_data_cache[token].loc[start_date:end_date]
        return token, df_token
    return token, None



def load_all_token_data_cached(start_date, end_date, market_index=None):
    """
    Load all token data once into global cache and slice for the period.
    ✅ FIXED: No threading for in-memory slicing (GIL contention removed).
    """
    global _full_data_cache
    
    # ✅ STEP 1: Load full data ONCE using parallel I/O (loky)
    if not _full_data_cache:
        full_start = CONFIG['start_date']
        full_end = CONFIG['end_date']
        logger.info(f"Loading full data cache from {full_start} to {full_end}")
        
        tokens_to_load = CONFIG["token_names"] + ([market_index] if market_index else [])
        
        # ✅ Parallel I/O for initial load (loky - separate processes)
        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(load_single_preprocess)(
                CONFIG['data_dir'] / f"{token}.csv", 
                token
            )
            for token in tokens_to_load
        )
        
        _full_data_cache = {}
        for token, df in zip(tokens_to_load, results):
            if df is not None:
                df_indexed = df.set_index('date').sort_index()
                _full_data_cache[token] = df_indexed
        
        logger.info(f"Cache loaded with {len(_full_data_cache)} tokens")
    
    # ✅ STEP 2: Slice from cache WITHOUT threading (pure pandas)
    price_data = {}
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    logger.info(f"Slicing cached data for period {start_date} to {end_date}")
    
    tokens_to_slice = CONFIG["token_names"] + ([market_index] if market_index else [])
    tokens_to_slice = [t for t in tokens_to_slice if t in _full_data_cache]
    
    if not tokens_to_slice:
        logger.warning("No tokens available in cache to slice")
        return {}
    
    # ✅ Sequential slicing (FAST - no GIL overhead, no process spawning)
    for token in tokens_to_slice:
        df_aligned = _full_data_cache[token].loc[start_date:end_date]
        if df_aligned is not None and len(df_aligned) > 0:
            price_data[token] = df_aligned
    
    # Align to common dates
    if price_data:
        common_dates = reduce(
            lambda x, y: x.intersection(y),
            [df.index for df in price_data.values()]
        )
        
        if len(common_dates) > 0:
            for key in price_data:
                price_data[key] = price_data[key].loc[common_dates]
            logger.info(
                f"✅ Aligned to {len(common_dates)} common dates "
                f"from {common_dates[0].date()} to {common_dates[-1].date()}"
            )
        else:
            logger.warning("❌ No common dates found across all series")
            return {}
    
    return price_data


def validate_data_files(data_dir: Path, token_list: List[str], market_index: str) -> bool:
    """Validate CSV existence and required columns once upfront."""
    missing_files = []
    invalid_files = []
    required_columns = {'Date', 'Price'}
    
    for token in token_list + [market_index]:
        file_path = data_dir / f"{token}.csv"
        if not file_path.exists():
            missing_files.append(token)
            continue
        try:
            df = pd.read_csv(file_path, nrows=1)
            if not required_columns.issubset(df.columns):
                invalid_files.append(token)
        except Exception as e:
            invalid_files.append(token)
            logger.error(f"[{token}] CSV read failed: {e}")
    
    if missing_files or invalid_files:
        logger.error(f"Missing: {missing_files}, Invalid: {invalid_files}")
        raise ValueError("Data validation failed.")
    
    logger.info("All CSV files validated successfully.")
    return True





