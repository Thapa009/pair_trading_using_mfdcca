# config.py
import torch
from pathlib import Path
import pandas as pd
import os

CONFIG = {
    
    'window': 252,                    #  Target, not requirement
    'flexible_lookback': True,        #  Enable flexible mode
    'min_lookback_days': 30,          #  Minimum acceptable
    'min_trading_days': 1,            #  Minimum trading days
    'data_coverage_threshold': 0.3,
    
    'mfdcca_min_valid_scales_pct': 0.4,
    'mfdcca_min_valid_q_pct': 0.4,
    'mfdcca_num_scales': 15, 
    'mfdcca_max_scale_ratio': 0.5,
      
    'date_column': 'Date',
    'price_column': 'Price',
    'data_dir': Path(__file__).resolve().parent.parent / "data",
    'results_dir': Path(__file__).resolve().parent.parent / "Result",

    
    'start_date': pd.to_datetime('2020-01-01'),
    'end_date': pd.to_datetime('2024-12-31'),
    
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    'token_names': [
        'ADA', 'AAVE', 'BCH', 'BTC', 'DASH', 'DOGE', 'EOS', 'ETC', 'ETH', 'FIL',
        'LINK', 'LTC', 'MATIC', 'TRX', 'VET', 'XLM', 'XMR', 'XRP', 'XTZ', 'ZCASH'
        
        
    ],
    'market_index': "INDEX",
    'walk_forward_periods': [
        {
            "training_period": (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31")),
            "test_period": (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"))
        },
        {
            "training_period": (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
            "test_period": (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31"))
        },
        {
            "training_period": (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")),
            "test_period": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
        }
    ],
    


  
    
    'q_list': [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    'mfdcca_epsilon': 1e-7,
    
    'optuna_ranges': {
        'pair_hxy_threshold': {'min': 0.20, 'max': 0.70, 'step': 0.05},
        'threshold_h': {'min': 0.03, 'max': 0.40, 'step': 0.03},
        'threshold_alpha': {'min': 0.03, 'max': 0.40, 'step': 0.03},
        'divergence_lookback_days': {'min': 3, 'max':7, 'step': 1},
        'divergence_threshold': {'min': 0.05, 'max': 0.15, 'step': 0.01},
        'pval_threshold': {'min': 0.02, 'max': 0.04, 'step': 0.005},
        'rho_threshold': {'min': 0.60, 'max': 0.70, 'step': 0.02}
        
    },
    'holding_period_days': 5,
    'rebalance_freq': 'BMS',
    'transaction_costs': 0.001,
    'risk_free_rate': 0.02,
    'export_results': True,
    }


# Set environment variables BEFORE any torch operations
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
        torch.cuda.set_device(device)
        return device
    return torch.device('cpu')

DEVICE = get_device()

# Set CPU thread limits
torch.set_num_threads(1)
torch.set_num_interop_threads(1)