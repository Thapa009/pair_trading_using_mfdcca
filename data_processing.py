from joblib import Parallel, delayed
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List
from config import CONFIG

logger = logging.getLogger(__name__)

# Global cache for full data (loaded once)
_full_data_cache: Dict = {}


def load_single_token(file_path: Path, token: str) -> Optional[pd.DataFrame]:
    """Load single token data - SIMPLIFIED"""
    try:
        df = pd.read_csv(file_path)

        # Your data has "Date" and "Price" columns
        df = df[["Date", "Price"]].rename(columns={"Date": "date", "Price": "close"})

        # Parse date (MM/DD/YYYY format)
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")

        # Parse price (remove commas)
        df["close"] = pd.to_numeric(
            df["close"].astype(str).str.replace(",", ""), errors="coerce"
        )

        # Clean data
        df = df.dropna(subset=["date", "close"])
        df = df[df["close"] > 0]

        if len(df) == 0:
            return None

        return df.set_index("date").sort_index()

    except Exception as e:
        logger.error(f"Failed to load {token}: {e}")
        return None


def load_all_token_data_cached(
    start_date, end_date, market_index: str
) -> Dict[str, pd.DataFrame]:
    """
    ✅ CLEAN & CORRECT: Align all tokens to INDEX business days
    Returns dict of DataFrames with common business days
    """
    global _full_data_cache

    # Load cache if empty
    if not _full_data_cache:
        logger.info("📥 Loading data cache...")
        tokens_to_load = list(set(CONFIG["token_names"] + [market_index]))

        # Parallel loading
        results = Parallel(n_jobs=-1)(
            delayed(load_single_token)(CONFIG["data_dir"] / f"{token}.csv", token)
            for token in tokens_to_load
        )

        # Store in cache
        for token, df in zip(tokens_to_load, results):
            if df is not None:
                _full_data_cache[token] = df

        logger.info(f"✅ Cache loaded: {len(_full_data_cache)} tokens")

    # Get INDEX data (business days only)
    if market_index not in _full_data_cache:
        logger.error(f"❌ {market_index} not in cache")
        return {}

    index_data = _full_data_cache[market_index].loc[start_date:end_date]
    if len(index_data) == 0:
        logger.error(
            f"❌ No INDEX data in period {start_date.date()} to {end_date.date()}"
        )
        return {}

    index_dates = index_data.index
    logger.info(f"📅 INDEX has {len(index_dates)} business days in period")

    # Align all tokens to INDEX dates
    aligned_data = {market_index: index_data}

    aligned_tokens = []
    for token in CONFIG["token_names"]:
        if token not in _full_data_cache:
            logger.debug(f"Skipping {token}: not in cache")
            continue

        token_data = _full_data_cache[token].loc[start_date:end_date]
        if len(token_data) == 0:
            logger.debug(f"Skipping {token}: no data in period")
            continue

        # ✅ SIMPLE: Align to INDEX business days
        token_aligned = token_data.reindex(index_dates).dropna()

        if len(token_aligned) > 0:
            aligned_data[token] = token_aligned
            aligned_tokens.append(token)
            logger.debug(f"✅ {token}: {len(token_aligned)} days aligned")

    logger.info(f"📊 {len(aligned_tokens)} tokens aligned to INDEX dates")

    # Find common dates (INDEX dates that exist in ALL tokens)
    common_dates = index_dates
    logger.info(f"✔ ALIGN start: {len(common_dates)} INDEX days")

    for token, df in aligned_data.items():
        if token == market_index:
            continue

        before = len(common_dates)
        common_dates = common_dates.intersection(df.index)
        after = len(common_dates)

        if after == 0:
            logger.error(f"❌ {token}: NO common dates left")
            raise RuntimeError(f"Alignment failed at {token}")

        elif after < before:
            logger.warning(f"⚠ {token}: {before} → {after} days")

        else:
            logger.debug(f"✔ {token}: no date loss ({after})")

    common_dates = common_dates.sort_values()
    logger.info(f"✔ ALIGN done: {len(common_dates)} common days")

    # Create final aligned data
    final_data = {}
    for token in aligned_data:
        final_data[token] = aligned_data[token].loc[common_dates]
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ DATA ALIGNMENT COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"   Tokens: {len(final_data)}")
    logger.info(f"   Common business days: {len(common_dates)}")
    logger.info(f"   Date range: {common_dates[0].date()} to {common_dates[-1].date()}")
    logger.info(f"{'='*60}")

    # ===============================
    # FINAL COMMON-DATE VERIFICATION
    # ===============================
    logger.info("🔍 DATA PROCESSING FINAL CHECK")

    reference_index = final_data[market_index].index

    for token, df in final_data.items():
        same_dates = reference_index.equals(df.index)

        logger.info(
            f"   {token:<10} | "
            f"rows={len(df):4d} | "
            f"dates_match_index={same_dates}"
        )

        if not same_dates:
            logger.error(f"❌ Date mismatch detected for {token}")
            raise RuntimeError(f"Date alignment failed for {token}")

    logger.info(
        f"✅ ALL TOKENS + INDEX SHARE IDENTICAL DATES "
        f"({reference_index[0].date()} → {reference_index[-1].date()})"
    )

    # ✅ RETURN MUST BE LAST
    return final_data


def validate_data_files(
    data_dir: Path, token_list: List[str], market_index: str
) -> bool:
    """Quick validation"""
    all_tokens = list(set(token_list + [market_index]))
    missing_files = []

    for token in all_tokens:
        file_path = data_dir / f"{token}.csv"
        if not file_path.exists():
            missing_files.append(token)
            logger.error(f"❌ {token}: File not found")

    if missing_files:
        logger.error(f"❌ Missing {len(missing_files)} files: {missing_files}")
        return False

    logger.info(f"✅ All {len(all_tokens)} data files validated")
    return True
