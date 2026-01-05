import logging
import sys
import pandas as pd
import numpy as np

np.random.seed(42)
import optuna
import csv
from pathlib import Path
from tabulate import tabulate
from typing import Dict, Any, List, Union, Optional
from simulation import run_simulation
from config import CONFIG
import torch
from config import DEVICE
from precompute import precompute_training_features

logger = logging.getLogger(__name__)

# Keep only: run_single_param_value, print_pretty_table, run_sensitivity, objective, main


def run_single_param_value(
    param: str,
    value: float,
    fixed_params: Dict[str, Union[int, float]],
    model_dir: Path,
    fold_idx: int,
    phase: str,
    metrics_list: List[str],
) -> Dict[str, Union[int, float]]:
    """
    Run the simulation with one hyperparameter value and return a dict of numeric metrics
    """
    try:
        if not isinstance(fold_idx, int):
            raise ValueError(
                f"fold_idx must be an integer, got {type(fold_idx)}: {fold_idx}"
            )

        run_params = fixed_params.copy()
        if isinstance(fixed_params.get(param), int):
            run_params[param] = int(value)
        else:
            run_params[param] = float(value)

        logging.debug(
            f"Running simulation for {param}={value}, fold_idx={fold_idx}, phase={phase}"
        )

        tune_mode = phase == "train"
        use_precompute = phase == "train"

        sim_result = run_simulation(
            **run_params,
            fold_number=fold_idx,
            method="mfdcca",
            tune_mode=tune_mode,
            use_precompute=use_precompute,
        )

        # ✅ FIXED: Use 0.0 instead of -1 for realistic metrics
        metrics: Dict[str, Union[int, float]] = {m: float(0.0) for m in metrics_list}

        if sim_result and "mfdcca" in sim_result:
            raw_metrics = sim_result["mfdcca"]

            for metric in metrics_list:
                raw_val = raw_metrics.get(metric, None)
                try:
                    metrics[metric] = (
                        float(raw_val) if raw_val is not None else float(0.0)
                    )
                except Exception:
                    metrics[metric] = float(0.0)

        if isinstance(fixed_params.get(param), int):
            metrics[param] = int(value)
        else:
            metrics[param] = float(value)

        return metrics

    except Exception as e:
        logging.error(
            f"Simulation failed for {param}={value} in fold {fold_idx} ({phase}): {str(e)}"
        )
        # ✅ FIXED: Return 0.0, not -1
        fallback_metrics: Dict[str, Union[int, float]] = {
            m: float(0.0) for m in metrics_list
        }
        if isinstance(fixed_params.get(param), int):
            fallback_metrics[param] = int(value)
        else:
            fallback_metrics[param] = float(value)
        return fallback_metrics


def print_pretty_table(results_df: pd.DataFrame, param: str, optuna_value: float):
    """Print table marking Optuna's choice, not sensitivity peak"""
    try:
        df = results_df.copy()
        metrics_cols = [
            "Sharpe_Ratio",
            "Sortino_Ratio",
            "Max_Drawdown_%",
            "Profit_Factor",
        ]

        for col in metrics_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            df[col] = df[col].apply(lambda x: f"{float(x):.4f}")

        df[param] = pd.to_numeric(df[param], errors="coerce").fillna(0.0)

        df[param] = df[param].apply(
            lambda x: (
                f"{float(x):.2f} [OPTUNA]"
                if np.isclose(float(x), optuna_value, atol=1e-6)
                else f"{float(x):.2f}"
            )
        )

        print(
            "\n"
            + tabulate(
                df[[param] + metrics_cols].values.tolist(),
                headers=[param] + metrics_cols,
                tablefmt="pretty",
            )
        )
    except Exception as e:
        logging.error(f"Failed to print pretty table for {param}: {e}")


def run_sensitivity(
    best_params: dict,
    hyperparam_ranges: dict,
    model_dir: Path,
    method: str = "mfdcca",
    fold_idx: int | None = None,
    phase: str = "train",
) -> None:
    """
    Sensitivity analysis - analyzes parameter behavior around Optuna's optimum
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
        "Sharpe_Ratio",
        "Sortino_Ratio",
        "Max_Drawdown_%",
        "Profit_Factor",
        "Calmar_Ratio",
    ]

    params_to_analyze = [
        "threshold_h",
        "threshold_alpha",
        "pair_hxy_threshold",
        "divergence_threshold",
        "divergence_lookback",
    ]

    for param in params_to_analyze:
        if param not in hyperparam_ranges:
            logging.warning(f"Parameter {param} not in hyperparam_ranges, skipping")
            continue

        logging.info(f"\n{'='*60}")
        logging.info(f"SENSITIVITY ANALYSIS: {param} on {phase.upper()} data")
        logging.info(f"{'='*60}")
        logging.info(
            f"⚡ FAST MODE: Reusing precomputed MFDCCA features from Optuna training"
        )

        param_range = hyperparam_ranges[param]
        min_val = param_range.get("min", 0)
        max_val = param_range.get("max", 1)
        step = param_range.get("step", 0.1)

        try:
            if param == "divergence_lookback":
                values = list(range(int(min_val), int(max_val) + 1, int(step)))
            else:
                values = np.round(np.arange(min_val, max_val + step, step), 6).tolist()
        except Exception as e:
            logging.error(f"Failed to generate values for {param}: {e}")
            continue

        fixed_params = best_params.copy()
        results = []

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
                # ✅ FIXED: Use 0.0, not -1
                fallback_metrics = {m: float(0.0) for m in metrics_list}
                if isinstance(fixed_params.get(param), int):
                    fallback_metrics[param] = int(value)
                else:
                    fallback_metrics[param] = float(value)
                results.append(fallback_metrics)

        try:
            results_df = pd.DataFrame(results)
            required_columns = [param] + metrics_list

            # ✅ FIXED: Fill with 0.0, not -1
            for col in required_columns:
                if col not in results_df.columns:
                    results_df[col] = float(0.0)

            results_df = results_df[required_columns].sort_values(by=param)
            csv_path = sensitivity_dir / f"{method}_sensitivity_{param}_{phase}.csv"
            results_df.to_csv(csv_path, index=False)

            if (
                primary_metric in results_df.columns
                and not results_df[primary_metric].isna().all()
                and results_df[primary_metric].max()
                > 0  # Add check for non-zero Sharpe
            ):
                # ✅ FIXED: Changed variable names for clarity
                curve_max_idx = results_df[primary_metric].idxmax()
                curve_max_value = results_df.loc[curve_max_idx, param]
                curve_max_sharpe = results_df.loc[curve_max_idx, primary_metric]
                optuna_value = best_params.get(param, "N/A")

                # ✅ FIXED: CORRECT TERMINOLOGY
                logging.info(f"Optuna global optimum: {optuna_value}")
                logging.info(
                    f"Sensitivity curve maximum: {curve_max_value:.4f} (Sharpe: {curve_max_sharpe:.4f})"
                )

                # Calculate sensitivity metrics
                sharpe_range = (
                    results_df[primary_metric].max() - results_df[primary_metric].min()
                )
                sharpe_std = results_df[primary_metric].std()
                logging.info(
                    f"Sensitivity stats: Sharpe range={sharpe_range:.4f}, std={sharpe_std:.4f}"
                )

                # Print the table
                print_pretty_table(results_df, param, optuna_value)

            else:
                # ✅ REMOVED Profit Factor fallback - just report failure
                logging.warning(f"⚠️ All Sharpe values are 0.0 for parameter {param}")
                logging.warning(f"   This indicates no trades were executed")
                logging.warning(
                    f"   Check pair selection thresholds and divergence filters"
                )

                # Still print the table but all values will be 0.0000
                print_pretty_table(results_df, param, best_params.get(param, 0.0))

        except Exception as e:
            logging.error(f"Failed to save CSV for {param}: {e}")


# [Keep objective() and main() functions as they are mostly correct]


def objective(
    trial: optuna.Trial, fold_number: int, method: str, trial_results_file: Path
) -> float:
    """
    ✅ CORRECTED: Each trial runs trading simulation on training period
    """
    try:
        ranges = CONFIG["optuna_ranges"]
        params = {}

        # Suggest hyperparameters based on method
        if method == "mfdcca":
            params = {
                "threshold_h": trial.suggest_float(
                    "threshold_h",
                    ranges["threshold_h"]["min"],
                    ranges["threshold_h"]["max"],
                    step=ranges["threshold_h"]["step"],
                ),
                "threshold_alpha": trial.suggest_float(
                    "threshold_alpha",
                    ranges["threshold_alpha"]["min"],
                    ranges["threshold_alpha"]["max"],
                    step=ranges["threshold_alpha"]["step"],
                ),
                "pair_hxy_threshold": trial.suggest_float(
                    "pair_hxy_threshold",
                    ranges["pair_hxy_threshold"]["min"],
                    ranges["pair_hxy_threshold"]["max"],
                    step=ranges["pair_hxy_threshold"]["step"],
                ),
                "divergence_threshold": trial.suggest_float(
                    "divergence_threshold",
                    ranges["divergence_threshold"]["min"],
                    ranges["divergence_threshold"]["max"],
                    step=ranges["divergence_threshold"]["step"],
                ),
                "divergence_lookback": trial.suggest_int(
                    "divergence_lookback",
                    ranges["divergence_lookback"]["min"],
                    ranges["divergence_lookback"]["max"],
                    step=ranges["divergence_lookback"]["step"],
                ),
            }
        elif method == "dcca":
            params = {
                "pair_hxy_threshold": trial.suggest_float(
                    "pair_hxy_threshold",
                    ranges["pair_hxy_threshold"]["min"],
                    ranges["pair_hxy_threshold"]["max"],
                    step=ranges["pair_hxy_threshold"]["step"],
                ),
                "divergence_threshold": trial.suggest_float(
                    "divergence_threshold",
                    ranges["divergence_threshold"]["min"],
                    ranges["divergence_threshold"]["max"],
                    step=ranges["divergence_threshold"]["step"],
                ),
                "divergence_lookback": trial.suggest_int(
                    "divergence_lookback",
                    ranges["divergence_lookback"]["min"],
                    ranges["divergence_lookback"]["max"],
                    step=ranges["divergence_lookback"]["step"],
                ),
            }
        elif method == "pearson":
            params = {
                "divergence_threshold": trial.suggest_float(
                    "divergence_threshold",
                    ranges["divergence_threshold"]["min"],
                    ranges["divergence_threshold"]["max"],
                    step=ranges["divergence_threshold"]["step"],
                ),
                "divergence_lookback": trial.suggest_int(
                    "divergence_lookback",
                    ranges["divergence_lookback"]["min"],
                    ranges["divergence_lookback"]["max"],
                    step=ranges["divergence_lookback"]["step"],
                ),
                "rho_threshold": trial.suggest_float(
                    "rho_threshold",
                    ranges["rho_threshold"]["min"],
                    ranges["rho_threshold"]["max"],
                    step=ranges["rho_threshold"]["step"],
                ),
            }
        elif method == "cointegration":
            params = {
                "divergence_threshold": trial.suggest_float(
                    "divergence_threshold",
                    ranges["divergence_threshold"]["min"],
                    ranges["divergence_threshold"]["max"],
                    step=ranges["divergence_threshold"]["step"],
                ),
                "divergence_lookback": trial.suggest_int(
                    "divergence_lookback",
                    ranges["divergence_lookback"]["min"],
                    ranges["divergence_lookback"]["max"],
                    step=ranges["divergence_lookback"]["step"],
                ),
                "pval_threshold": trial.suggest_float(
                    "pval_threshold",
                    ranges["pval_threshold"]["min"],
                    ranges["pval_threshold"]["max"],
                    step=ranges["pval_threshold"]["step"],
                ),
            }
        elif method == "index":
            params = {}  # No parameters for index method

        try:
            # ✅ CORRECT: Run simulation
            perf = run_simulation(
                fold_number=fold_number,
                method=method,
                tune_mode=True,  # Training mode for hyperparameter tuning
                use_precompute=True,  # Use precomputed features
                **params,
            )

            # Extract Sharpe Ratio
            if method in perf and perf[method]:
                metrics = perf[method]
                sharpe = metrics.get("Sharpe_Ratio", -1.0)

                # Log trial results
                row = {k: float(v) for k, v in params.items()}
                row["Sharpe_Ratio"] = float(sharpe) if sharpe is not None else -1.0

                logger.debug(f"Trial {trial.number}: params={params}, sharpe={sharpe}")

                # Save to CSV
                file_exists = trial_results_file.exists()
                with open(trial_results_file, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row)

                return float(sharpe) if sharpe is not None else 0.0
            else:
                logger.warning(
                    f"No results for method {method} in trial {trial.number}"
                )
                return -1.0

        except Exception as e:
            logger.error(f"run_simulation failed: {e}")
            return -1.0

    except Exception as e:
        logger.error(f"Objective function failed: {e}")
        return -1.0


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    best_params_by_fold_method = {}
    methods = ["mfdcca", "dcca", "pearson", "cointegration", "index"]

    for fold_number, period in enumerate(CONFIG["walk_forward_periods"], 1):
        logger.info(f"\n{'#'*100}")
        logger.info(f"# FOLD {fold_number}")
        logger.info(
            f"# Training: {period['training_period'][0].date()} to {period['training_period'][1].date()}"
        )
        logger.info(
            f"# Test: {period['test_period'][0].date()} to {period['test_period'][1].date()}"
        )
        logger.info(f"{'#'*100}\n")

        best_params_by_fold_method[fold_number] = {}
        test_metrics = {}

        for method in methods:
            # ✅ FIXED: Define results_dir at the start
            results_dir = Path(CONFIG["results_dir"]) / f"fold_{fold_number}" / method
            results_dir.mkdir(parents=True, exist_ok=True)

            # ================================================================
            # PHASE 0: PRE-COMPUTE TRAINING FEATURES (if needed)
            # ================================================================
            if method != "index":  # ✅ FIXED: Always precompute for training methods
                logger.info(
                    f"🔧 PRE-COMPUTING features for {method} training period..."
                )
                precompute_training_features(
                    fold_number=fold_number,
                    method=method,
                    period_start=period["training_period"][0],
                    period_end=period["training_period"][1],
                )

            # ================================================================
            # PHASE 1: TRAINING - HYPERPARAMETER TUNING ONLY
            # ================================================================
            if method == "index":
                best_params = {}
                logger.info(f"Method 'index': No hyperparameters to tune.")
            else:
                logger.info(
                    f"🎯 TRAINING: Hyperparameter tuning with Optuna (300 trials)"
                )
                logger.info(
                    f"   Period: {period['training_period'][0].date()} to {period['training_period'][1].date()}"
                )

                # ✅ FIXED: Ensure precompute directory exists
                cache_dir = (
                    Path(CONFIG["results_dir"])
                    / "precompute"
                    / f"fold_{fold_number}"
                    / method
                )
                if cache_dir.exists():
                    logger.info(
                        f"✅ Precomputed features available: {len(list(cache_dir.glob('*.pt')))} weeks"
                    )
                else:
                    logger.warning(f"❌ No precomputed features found for {method}")

                trial_results_file = results_dir / f"{method}_trial_results.csv"
                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=42),
                )

                # ✅ CORRECT: Optuna trials use precomputed features
                study.optimize(
                    lambda trial: objective(
                        trial, fold_number, method, trial_results_file
                    ),
                    n_trials=300,
                    n_jobs=1,
                    show_progress_bar=True,
                )
                best_params = study.best_params
                logger.info(f"✅ Best parameters found: {best_params}")
                logger.info(f"✅ Best training Sharpe Ratio: {study.best_value:.4f}")

            best_params_by_fold_method[fold_number][method] = best_params

            # Save best parameters
            params_path = results_dir / "best_params.json"
            with open(params_path, "w") as f:
                import json

                json.dump(best_params, f, indent=2)

            # ================================================================
            # PHASE 2: TEST - FINAL EVALUATION (OUT-OF-SAMPLE)
            # ================================================================
            logger.info(f"\n{'='*100}")
            logger.info(f"📊 TEST: Final evaluation on unseen data")
            logger.info(
                f"   Period: {period['test_period'][0].date()} to {period['test_period'][1].date()}"
            )
            logger.info(f"{'='*100}")

            # ✅ SIMPLE: INDEX method needs NO parameters
            if method == "index":
                # Empty dict - INDEX is pure buy-and-hold
                test_results = run_simulation(
                    fold_number=fold_number,
                    method=method,
                    tune_mode=False,
                    use_precompute=False,
                    # NO parameters for INDEX
                )
            else:
                # Trading methods use best_params from Optuna
                test_results = run_simulation(
                    fold_number=fold_number,
                    method=method,
                    tune_mode=False,
                    use_precompute=False,
                    **best_params,
                )

            # Store test metrics for benchmark comparison
            if test_results and method in test_results:
                test_metrics[method] = test_results[method]
                metrics = test_results[method]

                if metrics:
                    sharpe = metrics.get("Sharpe_Ratio", 0)
                    logger.info(f"✅ TEST Sharpe: {sharpe:.4f}")
                else:
                    logger.warning(f"No valid results for method {method}")
            else:
                logger.warning(f"No results generated for method {method}")

            # ================================================================
            # PHASE 3: SENSITIVITY ANALYSIS (REUSES TRAINING PRECOMPUTE) - MFDCCA ONLY
            # ================================================================
            if method == "mfdcca":
                logger.info(f"\n🔍 SENSITIVITY ANALYSIS on TRAINING data (MFDCCA only)")
                logger.info(
                    f"   ⚡ EFFICIENCY: Reusing precomputed features from Phase 0 & Optuna"
                )
                logger.info(
                    f"   📦 Same MFDCCA features, testing different threshold combinations"
                )
                logger.info(
                    f"   Period: {period['training_period'][0].date()} to {period['training_period'][1].date()}"
                )

                # ✅ Verify precompute cache exists
                cache_dir = (
                    Path(CONFIG["results_dir"])
                    / "precompute"
                    / f"fold_{fold_number}"
                    / method
                )
                if cache_dir.exists():
                    num_cached_weeks = len(list(cache_dir.glob("*.pt")))
                    logger.info(
                        f"   ✅ Using {num_cached_weeks} precomputed weeks from training"
                    )
                else:
                    logger.warning(
                        f"   ⚠️  No precompute cache found - will compute live"
                    )

                run_sensitivity(
                    best_params=best_params,
                    hyperparam_ranges=CONFIG["optuna_ranges"],
                    model_dir=Path(CONFIG["results_dir"]) / f"fold_{fold_number}",
                    method="mfdcca",
                    fold_idx=fold_number,
                    phase="train",
                )

        # ================================================================
        # BENCHMARK COMPARISON (TEST RESULTS ONLY)
        # ================================================================
        logger.info(f"BENCHMARK COMPARISON - FOLD {fold_number} (TEST PERIOD)")
        logger.info(f"{'='*100}")

        if test_metrics:
            benchmark_df = pd.DataFrame(test_metrics).T

            required_columns = [
                "Sharpe_Ratio",
                "Sortino_Ratio",
                "Max_Drawdown_%",
                "Profit_Factor",
                "Calmar_Ratio",
            ]

            for col in required_columns:
                if col not in benchmark_df.columns:
                    benchmark_df[col] = 0.0

            benchmark_df = benchmark_df.reindex(methods)[required_columns].round(4)

            benchmark_path = (
                Path(CONFIG["results_dir"])
                / f"fold_{fold_number}"
                / "benchmark_comparison.csv"
            )
            benchmark_df.to_csv(
                benchmark_path, index=True, index_label="method", float_format="%.4f"
            )

            print(f"\n{'='*100}")
            print(f"FOLD {fold_number} - TEST RESULTS (Out-of-Sample)")
            print(f"{'='*100}")
            print(benchmark_df.to_string())
            print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
