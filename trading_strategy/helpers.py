# helpers.py
from config import CONFIG
import logging


def update_config_for_run(config, **params):
    """
    Update the CONFIG dictionary with provided hyperparameters.

    Args:
        config (dict): The CONFIG dictionary to update.
        **params: Keyword arguments for hyperparameters.
    """
    valid_params = {
        "threshold_h": "threshold_h",
        "threshold_alpha": "threshold_alpha",
        "pair_hxy_threshold": "pair_hxy_threshold",
        "divergence_threshold": "divergence_threshold",
        "divergence_lookback_days": "divergence_lookback_days",
        "rho_threshold": "rho_threshold",
        "pval_threshold": "pval_threshold"
    }

    for key, value in params.items():
        if key in valid_params:
            config[valid_params[key]] = value
        else:
            logging.warning(f"Ignoring unexpected parameter: {key}")