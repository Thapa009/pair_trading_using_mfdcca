import logging
import sys
import pandas as pd
import numpy as np
np.random.seed(42)
import optuna
import csv
import random
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from tabulate import tabulate
from typing import Dict, Any, List, Union
from datetime import datetime, timedelta
from simulation import run_simulation
from config import CONFIG
from data_processing import load_all_token_data_cached
from typing import Dict, Any, List, Union, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from config import DEVICE
logger = logging.getLogger(__name__)

# NEW: Import precompute
from precompute import precompute_training_features

from typing import Dict, Any, List, Union

def run_single_param_value(
    param: str,
    value: float,
    fixed_params: Dict[str, Union[int, float]],
    model_dir: Path,
    fold_idx: int,
    phase: str,  # "train" or "test"
    metrics_list: List[str]
) -> Dict[str, Union[int, float]]:
    """
    Run the simulation with one hyperparameter value and return a dict of numeric metrics
    plus the tested param value. This function guarantees the returned dict has only
    int/float values (no None), which satisfies static type checkers like Pylance.
    """
    try:
        if not isinstance(fold_idx, int):
            raise ValueError(f"fold_idx must be an integer, got {type(fold_idx)}: {fold_idx}")

        run_params = fixed_params.copy()
        # preserve ints when appropriate
        if isinstance(fixed_params.get(param), int):
            run_params[param] = int(value)
        else:
            run_params[param] = float(value)

        logging.debug(f"Running simulation for {param}={value}, fold_idx={fold_idx}, phase={phase}, run_params={run_params}")

        sim_result = run_simulation(
            **run_params,
            fold_number=fold_idx,
            method="mfdcca",
            tune_mode=(phase == "train"),
            use_precompute=(phase == "train")  # Use cache in train/sensitivity
        )

        # Initialize metrics dict with safe default floats for all expected metrics
        metrics: Dict[str, Union[int, float]] = {m: float(-1) for m in metrics_list}

        # Validate sim_result and extract numeric values if present
        if not sim_result or not isinstance(sim_result, dict) or "mfdcca" not in sim_result:
            logging.warning(f"Invalid simulation result for {param}={value} in fold {fold_idx} ({phase}): {sim_result}")
        else:
            method_results = sim_result["mfdcca"]
            if not method_results or not isinstance(method_results, list) or not isinstance(method_results[0], dict):
                logging.warning(f"No valid metrics inside simulation result for {param}={value} in fold {fold_idx} ({phase})")
            else:
                raw_metrics = method_results[0]
                # Coerce only the metrics we care about into numeric floats (fallback -1.0)
                for metric in metrics_list:
                    raw_val = raw_metrics.get(metric, None)
                    try:
                        metrics[metric] = float(raw_val) if raw_val is not None else float(-1)
                    except Exception:
                        metrics[metric] = float(-1)

        # Always include the tested parameter
        if isinstance(fixed_params.get(param), int):
            metrics[param] = int(value)
        else:
            metrics[param] = float(value)

        return metrics

    except Exception as e:
        logging.error(f"Simulation failed for {param}={value} in fold {fold_idx} ({phase}): {str(e)}")
        fallback_metrics: Dict[str, Union[int, float]] = {m: float(-1) for m in metrics_list}
        if isinstance(fixed_params.get(param), int):
            fallback_metrics[param] = int(value)
        else:
            fallback_metrics[param] = float(value)
        return fallback_metrics

def print_pretty_table(results_df: pd.DataFrame, param: str, best_value: float):
    try:
        df = results_df.copy()
        metrics_cols = ["Sharpe_Ratio", "Sortino_Ratio", "Max_Drawdown_%", "Win_Rate_%", "Profit_Factor"]
        for col in metrics_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["Sharpe_Ratio"] = df["Sharpe_Ratio"].map(lambda x: f"{x:.4f}")
        df["Sortino_Ratio"] = df["Sortino_Ratio"].map(lambda x: f"{x:.4f}")
        df["Max_Drawdown_%"] = df["Max_Drawdown_%"].map(lambda x: f"{x:.2f}%")
        df["Win_Rate_%"] = df["Win_Rate_%"].map(lambda x: f"{x:.2f}%")
        df["Profit_Factor"] = df["Profit_Factor"].map(lambda x: f"{x:.4f}")
        df[param] = pd.to_numeric(df[param], errors='coerce').fillna(0)
        best_value = float(best_value)
        df[param] = df[param].apply(
            lambda x: f"{x:.2f} (best)" if np.isclose(float(x), best_value, atol=1e-6) else f"{x:.2f}"
        )
        print("\n" + tabulate(
            df[[param] + metrics_cols].values.tolist(),
            headers=[param] + metrics_cols,
            tablefmt="pretty"
        ))
    except Exception as e:
        logging.error(f"Failed to print pretty table for {param}: {e}")

import logging
from pathlib import Path
from typing import Dict, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from config import CONFIG, DEVICE

def run_sensitivity(
    best_params: dict,
    hyperparam_ranges: dict,
    model_dir: Path,
    method: str = "mfdcca",
    fold_idx: int | None = None,
    phase: str = "train",
    max_workers: int = 4
) -> None:
    """
    ✅ CORRECT: Sensitivity analysis with your proven parameter range logic
    """
    if fold_idx is None:
        logging.error("fold_idx cannot be None")
        return

    if method != "mfdcca":
        return

    sensitivity_dir = model_dir / "sensitivity_analysis"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)

    primary_metric = "Sharpe_Ratio"
    metrics_list = [
        "Sharpe_Ratio", "Sortino_Ratio", "Max_Drawdown_%",
        "Win_Rate_%", "Profit_Factor", "Calmar_Ratio"
    ]

    # MFDCCA parameters to analyze
    params_to_analyze = [
        "threshold_h", "threshold_alpha", "pair_hxy_threshold",
        "divergence_threshold", "divergence_lookback_days"
    ]

    for param in params_to_analyze:
        if param not in hyperparam_ranges:
            logging.warning(f"Parameter {param} not in hyperparam_ranges, skipping")
            continue

        logging.info(f"Sensitivity analysis for {param}...")

        # ✅ YOUR ORIGINAL LOGIC - PROVEN CORRECT
        param_range = hyperparam_ranges[param]
        min_val = param_range.get("min", 0)
        max_val = param_range.get("max", 1)
        step = param_range.get("step", 0.1)

        try:
            if param == "divergence_lookback_days":
                values = list(range(int(min_val), int(max_val) + 1, int(step)))
            else:
                values = np.round(np.arange(min_val, max_val + step, step), 6).tolist()
        except Exception as e:
            logging.error(f"Failed to generate values for {param}: {e}")
            continue

        fixed_params = best_params.copy()
        results = []

        # Test each parameter value
        for value in values:
            try:
                result = run_single_param_value(
                    param, value, fixed_params, model_dir, fold_idx, phase, metrics_list
                )
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                results.append(result)
                
            except Exception as e:
                logging.error(f"Failed for {param}={value}: {e}")
                fallback_metrics = {m: float(-1) for m in metrics_list}
                if isinstance(fixed_params.get(param), int):
                    fallback_metrics[param] = int(value)
                else:
                    fallback_metrics[param] = float(value)
                results.append(fallback_metrics)

        # ✅ Save CSV results
        try:
            results_df = pd.DataFrame(results)
            required_columns = [param] + metrics_list
            
            # Ensure all required columns exist
            for col in required_columns:
                if col not in results_df.columns:
                    results_df[col] = float(-1)
            
            # Sort by parameter value and save CSV
            results_df = results_df[required_columns].sort_values(by=param)
            csv_path = sensitivity_dir / f"{method}_sensitivity_{param}.csv"
            results_df.to_csv(csv_path, index=False)
            
            # ✅ Find best value using Sharpe Ratio
            if primary_metric in results_df.columns and not results_df[primary_metric].isna().all():
                best_idx = results_df[primary_metric].idxmax()
                best_value = results_df.loc[best_idx, param]
                best_sharpe = results_df.loc[best_idx, primary_metric]
                optuna_best = best_params.get(param, "N/A")
                
                logging.info(f"Best {param}: {best_value:.4f} (Sharpe: {best_sharpe:.4f}), Optuna: {optuna_best}")
            else:
                best_value = results_df.loc[results_df["Profit_Factor"].idxmax(), param]
                logging.info(f"Best {param}: {best_value} (Profit_Factor fallback)")
                
        except Exception as e:
            logging.error(f"Failed to save CSV for {param}: {e}")
            
            
def objective(trial: optuna.Trial, fold_number: int, method: str, trial_results_file: Path) -> float:
    try:
        ranges = CONFIG["optuna_ranges"]
        params = {}
        if method == "mfdcca":
            params = {
                "threshold_h": trial.suggest_float(
                    "threshold_h",
                    ranges["threshold_h"]["min"],
                    ranges["threshold_h"]["max"],
                    step=ranges["threshold_h"]["step"]
                ),
                "threshold_alpha": trial.suggest_float(
                    "threshold_alpha",
                    ranges["threshold_alpha"]["min"],
                    ranges["threshold_alpha"]["max"],
                    step=ranges["threshold_alpha"]["step"]
                ),
                "pair_hxy_threshold": trial.suggest_float(
                    "pair_hxy_threshold",
                    ranges["pair_hxy_threshold"]["min"],
                    ranges["pair_hxy_threshold"]["max"],
                    step=ranges["pair_hxy_threshold"]["step"]
                ),
                "divergence_threshold": trial.suggest_float(
                    "divergence_threshold",
                    ranges["divergence_threshold"]["min"],
                    ranges["divergence_threshold"]["max"],
                    step=ranges["divergence_threshold"]["step"]
                ),
                "divergence_lookback_days": trial.suggest_int(
                    "divergence_lookback_days",
                    ranges["divergence_lookback_days"]["min"],
                    ranges["divergence_lookback_days"]["max"],
                    step=ranges["divergence_lookback_days"]["step"]
                ),
            }
        elif method == "dcca":
            params = {
                "pair_hxy_threshold": trial.suggest_float(
                    "pair_hxy_threshold",
                    ranges["pair_hxy_threshold"]["min"],
                    ranges["pair_hxy_threshold"]["max"],
                    step=ranges["pair_hxy_threshold"]["step"]
                ),
                "divergence_threshold": trial.suggest_float(
                    "divergence_threshold",
                    ranges["divergence_threshold"]["min"],
                    ranges["divergence_threshold"]["max"],
                    step=ranges["divergence_threshold"]["step"]
                ),
                "divergence_lookback_days": trial.suggest_int(
                    "divergence_lookback_days",
                    ranges["divergence_lookback_days"]["min"],
                    ranges["divergence_lookback_days"]["max"],
                    step=ranges["divergence_lookback_days"]["step"]
                ),
            }
        elif method == "pearson":
            params = {
                "divergence_threshold": trial.suggest_float(
                    "divergence_threshold",
                    ranges["divergence_threshold"]["min"],
                    ranges["divergence_threshold"]["max"],
                    step=ranges["divergence_threshold"]["step"]
                ),
                "divergence_lookback_days": trial.suggest_int(
                    "divergence_lookback_days",
                    ranges["divergence_lookback_days"]["min"],
                    ranges["divergence_lookback_days"]["max"],
                    step=ranges["divergence_lookback_days"]["step"]
                ),
                "rho_threshold": trial.suggest_float(
                    "rho_threshold",
                    ranges["rho_threshold"]["min"],
                    ranges["rho_threshold"]["max"],
                    step=ranges["rho_threshold"]["step"]
                ),
            }
        elif method == "cointegration":
            params = {
                "divergence_threshold": trial.suggest_float(
                    "divergence_threshold",
                    ranges["divergence_threshold"]["min"],
                    ranges["divergence_threshold"]["max"],
                    step=ranges["divergence_threshold"]["step"]
                ),
                "divergence_lookback_days": trial.suggest_int(
                    "divergence_lookback_days",
                    ranges["divergence_lookback_days"]["min"],
                    ranges["divergence_lookback_days"]["max"],
                    step=ranges["divergence_lookback_days"]["step"]
                ),
                "pval_threshold": trial.suggest_float(
                    "pval_threshold",
                    ranges["pval_threshold"]["min"],
                    ranges["pval_threshold"]["max"],
                    step=ranges["pval_threshold"]["step"]
                ),
            }
        elif method == "index":
            params = {}
        
        try:
            perf = run_simulation(
                fold_number=fold_number,
                method=method,
                tune_mode=True,
                use_precompute=True,  # Use cache during Optuna trials
                **params
            )
            results = perf.get(method, [])
            metrics = results[0] if results and isinstance(results, list) and len(results) > 0 else {}
            sharpe = metrics.get("Sharpe_Ratio", -1.0)
            row = {k: float(v) for k, v in params.items()}
            row['Sharpe_Ratio'] = float(sharpe) if sharpe is not None else -1.0
            logging.debug(f"Trial {trial.number}: params={params}, sharpe={sharpe}")
            file_exists = trial_results_file.exists()
            with open(trial_results_file, "a", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            return float(sharpe) if sharpe is not None else -1.0
        except Exception as e:
            logging.error(f"run_simulation failed for fold {fold_number}, method {method} with error: {e}")
            return -1.0
    except Exception as e:
        logging.error(f"Objective function failed for fold {fold_number}, method {method} with error: {e}")
        return -1.0

# main.py

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    best_params_by_fold_method = {}
    methods = ["mfdcca", "dcca", "pearson", "cointegration", "index"]
    
    for fold_number, period in enumerate(CONFIG["walk_forward_periods"], 1):
        logger.info(f"\n{'#'*100}")
        logger.info(f"# FOLD {fold_number}")
        logger.info(f"# Training: {period['training_period'][0].date()} to {period['training_period'][1].date()}")
        logger.info(f"# Test: {period['test_period'][0].date()} to {period['test_period'][1].date()}")
        logger.info(f"{'#'*100}\n")
        
        best_params_by_fold_method[fold_number] = {}
        benchmark_metrics = {}
        
        for method in methods:
            logger.info(f"\n{'='*100}")
            logger.info(f"Processing: Fold {fold_number}, Method {method.upper()}")
            logger.info(f"{'='*100}")
            
            results_dir = Path(CONFIG["results_dir"]) / f"fold_{fold_number}" / method
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # NEW: PRECOMPUTE TRAINING FEATURES (once per method/fold)
            if method != "index":
                precompute_training_features(
                    fold_number=fold_number,
                    method=method,
                    period_start=period["training_period"][0],
                    period_end=period["training_period"][1]
                )
            
            # ========================================
            # TRAINING PHASE: Hyperparameter Tuning
            # ========================================
            if method == "index":
                best_params = {}
                logger.info(f"Method 'index': No hyperparameters to tune.")
            else:
                logger.info(f"Starting hyperparameter tuning (5 trials) on TRAINING period...")
                trial_results_file = results_dir / f"{method}_trial_results.csv"
                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=42)  # Intelligent search
                )

                study.optimize(
                    lambda trial: objective(trial, fold_number, method, trial_results_file),
                    n_trials=100,  #
                    n_jobs=1,
                    show_progress_bar=True  # Add progress visualization
                )
                best_params = study.best_params
                logger.info(f"Best parameters from training: {best_params}")
                logger.info(f"Best Sharpe Ratio on training: {study.best_value:.4f}")
            
            best_params_by_fold_method[fold_number][method] = best_params
            
            # Save best parameters
            params_path = results_dir / "best_params.json"
            with open(params_path, 'w') as f:
                import json
                json.dump(best_params, f, indent=2)
            logger.info(f"Best parameters saved to {params_path}")
            
            # ========================================
            # SENSITIVITY ANALYSIS (MFDCCA only, on TRAINING period with precompute)
            # ========================================
            if method == "mfdcca":
                logger.info(f"Running sensitivity analysis for MFDCCA on TRAINING period...")
                
                run_sensitivity(
                    best_params=best_params,
                    hyperparam_ranges=CONFIG["optuna_ranges"],
                    model_dir=Path(CONFIG["results_dir"]) / f"fold_{fold_number}",
                    method="mfdcca", 
                    fold_idx=fold_number,
                    phase="train"
                )
                logger.info(f"Sensitivity analysis complete. CSV results saved.")
            # ========================================
            # TEST PHASE: Final Evaluation (No precompute - fresh compute)
            # ========================================
            logger.info(f"\n{'='*100}")
            logger.info(f"Running final evaluation on TEST period (out-of-sample)...")
            logger.info(f"{'='*100}")
            results = run_simulation(
                fold_number=fold_number,
                method=method,
                tune_mode=False,
                use_precompute=False,  # Explicitly fresh for test
                **best_params
            )
            
            if results and method in results:
                benchmark_metrics[method] = results[method]
                test_metrics = results[method][0] if results[method] else {}
                logger.info(f"Test performance - Sharpe: {test_metrics.get('Sharpe_Ratio', 'N/A'):.4f}")
        
        # ========================================
        # BENCHMARK COMPARISON (Test Period Results)
        # ========================================
        logger.info(f"\n{'='*100}")
        logger.info(f"BENCHMARK COMPARISON - FOLD {fold_number} (TEST PERIOD)")
        logger.info(f"{'='*100}")
        
        benchmark_data = {}
        for method, result_list in benchmark_metrics.items():
            if result_list:
                df = pd.DataFrame(result_list)
                mean_metrics = df.mean(numeric_only=True).round(4).to_dict()
                benchmark_data[method] = mean_metrics
        
        benchmark_df = pd.DataFrame(benchmark_data).T
        required_columns = ["Sharpe_Ratio", "Sortino_Ratio", "Max_Drawdown_%", "Win_Rate_%", "Profit_Factor", "Calmar_Ratio"]
        benchmark_df = benchmark_df.reindex(methods)[required_columns].fillna(-1).round(4)
        
        benchmark_path = Path(CONFIG["results_dir"]) / f"fold_{fold_number}" / "benchmark_comparison.csv"
        benchmark_df.to_csv(benchmark_path, index=True, index_label="method", float_format="%.4f")
        logger.info(f"Benchmark comparison saved to {benchmark_path}")
        
        # Print comparison table
        print(f"\n{'='*100}")
        print(f"FOLD {fold_number} - TEST PERIOD RESULTS (Out-of-Sample)")
        print(f"{'='*100}")
        print(benchmark_df.to_string())
        print(f"{'='*100}\n")

if __name__ == "__main__":
    main()