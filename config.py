import torch
from pathlib import Path
import pandas as pd
import os

CONFIG = {
    # ============================================================================
    # DATA PARAMETERS
    # ============================================================================
    "window": 250,
    "use_capm": False,
    "flexible_lookback": True,
    "min_lookback_days": 30,
    "min_trading_days": 1,
    "date_column": "Date",
    "price_column": "Price",
    "data_dir": Path(__file__).resolve().parent.parent / "data",
    "results_dir": Path(__file__).resolve().parent.parent / "Result",
    "start_date": pd.to_datetime("2019-12-01"),
    "end_date": pd.to_datetime("2024-12-31"),
    # ============================================================================
    # MFDCCA PARAMETERS (CONSISTENT)
    # ============================================================================
    "q_list": [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
    # ============================================================================
    # TOKENS & MARKET
    # ============================================================================
    "token_names": [
        "ADA",
        "BNB",
        "BCH",
        "BTC",
        "DASH",
        "DOGE",
        "EOS",
        "ETC",
        "ETH",
        "BSV",
        "LINK",
        "LTC",
        "MATIC",
        "TRX",
        "VET",
        "XLM",
        "XMR",
        "XRP",
        "XTZ",
        "ZCASH",
    ],
    "market_index": "INDEX",
    # ============================================================================
    # WALK-FORWARD VALIDATION
    # ============================================================================
    "walk_forward_periods": [
        {
            "training_period": (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31")),
            "test_period": (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
        },
        {
            "training_period": (
                pd.Timestamp("2022-01-01"),
                pd.Timestamp("2022-12-31"),
            ),  # Includes 2022 data
            "test_period": (
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-12-31"),
            ),  # New unseen data
        },
        {
            "training_period": (
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-12-31"),
            ),  # Includes 2023 data
            "test_period": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
        },
    ],
    # ============================================================================
    # HYPERPARAMETER OPTIMIZATION RANGES
    # ============================================================================
    "optuna_ranges": {
        "pair_hxy_threshold": {
            "min": 0.30,
            "max": 0.45,
            "step": 0.02,
        },
        "threshold_h": {
            "min": 0.05,
            "max": 0.20,
            "step": 0.02,
        },
        "threshold_alpha": {
            "min": 0.05,
            "max": 0.20,
            "step": 0.02,
        },
        "divergence_lookback": {"min": 3, "max": 9, "step": 1},
        "divergence_threshold": {
            "min": 0.03,
            "max": 0.12,
            "step": 0.01,
        },
        "pval_threshold": {"min": 0.01, "max": 0.10, "step": 0.01},
        "rho_threshold": {"min": 0.50, "max": 0.80, "step": 0.05},
    },
    # ============================================================================
    # TRADING PARAMETERS
    # ============================================================================
    "holding_period_days": 5,
    "rebalance_freq": "BMS",
    "export_results": True,
    "transaction_costs": 0.001,
    # ============================================================================
    # DEVICE CONFIGURATION
    # ============================================================================
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
        torch.cuda.set_device(device)
        return device
    return torch.device("cpu")


DEVICE = get_device()

# Set CPU thread limits
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
