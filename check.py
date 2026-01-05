"""
Data Requirements Verification for Walk-Forward Analysis
=========================================================

This script checks if you have sufficient historical data for your research.
"""

import pandas as pd
from datetime import datetime


def verify_data_requirements():
    """
    Verify that data requirements are met for all training periods
    """

    # Your walk-forward periods
    walk_forward_periods = [
        {
            "name": "Period 1",
            "training": ("2021-01-01", "2021-12-31"),
            "testing": ("2022-01-01", "2022-12-31"),
        },
        {
            "name": "Period 2",
            "training": ("2022-01-01", "2022-12-31"),
            "testing": ("2023-01-01", "2023-12-31"),
        },
        {
            "name": "Period 3",
            "training": ("2023-01-01", "2023-12-31"),
            "testing": ("2024-01-01", "2024-12-31"),
        },
    ]

    print("=" * 80)
    print("DATA REQUIREMENTS VERIFICATION")
    print("=" * 80)
    print()

    lookback_days = 250  # business days

    for period in walk_forward_periods:
        print(f"\n{'=' * 80}")
        print(
            f"{period['name']}: Training {period['training'][0]} to {period['training'][1]}"
        )
        print(f"{'=' * 80}")

        # First training week
        training_start = pd.Timestamp(period["training"][0])

        # Find first Monday
        if training_start.weekday() != 0:  # Not Monday
            days_to_monday = (7 - training_start.weekday()) % 7
            if days_to_monday == 0:
                days_to_monday = 7
            first_monday = training_start + pd.Timedelta(days=days_to_monday)
        else:
            first_monday = training_start

        print(f"\n📅 First Trading Week:")
        print(f"   First Monday: {first_monday.date()}")

        # Information cutoff = Friday before Monday
        information_cutoff = first_monday - pd.tseries.offsets.BDay(1)
        print(f"   Information Cutoff: {information_cutoff.date()}")

        # Calculate required lookback start (250 business days back)
        business_days = pd.date_range(
            end=information_cutoff, periods=lookback_days, freq="B"
        )
        required_data_start = business_days[0]

        print(f"\n📊 Data Requirements:")
        print(f"   Lookback Period: {lookback_days} business days")
        print(f"   Required Data Start: {required_data_start.date()}")
        print(f"   Required Data End: {information_cutoff.date()}")

        # Calculate how far back we need data
        days_before_training = (training_start - required_data_start).days
        print(f"\n⚠️  CRITICAL:")
        print(f"   Training starts: {training_start.date()}")
        print(f"   Data must start: {required_data_start.date()}")
        print(f"   Gap: {days_before_training} calendar days before training period")

        # Recommendation
        print(f"\n✅ RECOMMENDATION:")
        recommended_start = required_data_start - pd.Timedelta(days=30)  # 30-day buffer
        print(f"   Ensure data available from: {recommended_start.date()}")
        print(f"   (Includes 30-day safety buffer)")

        # Check if data requirement is met
        if required_data_start < training_start:
            print(f"\n⚠️  WARNING: You need data from BEFORE {period['training'][0]}!")
            print(f"   Minimum required: {required_data_start.date()}")
        else:
            print(f"\n✅ OK: Data requirement within training period")

    print("\n" + "=" * 80)
    print("SUMMARY: MINIMUM DATA START DATES")
    print("=" * 80)
    print()
    print("For your 3-period walk-forward analysis, ensure you have data from:")
    print()

    for period in walk_forward_periods:
        training_start = pd.Timestamp(period["training"][0])
        first_monday = training_start
        if training_start.weekday() != 0:
            days_to_monday = (7 - training_start.weekday()) % 7
            if days_to_monday == 0:
                days_to_monday = 7
            first_monday = training_start + pd.Timedelta(days=days_to_monday)

        information_cutoff = first_monday - pd.tseries.offsets.BDay(1)
        business_days = pd.date_range(
            end=information_cutoff, periods=lookback_days, freq="B"
        )
        required_data_start = business_days[0]
        recommended_start = required_data_start - pd.Timedelta(days=30)

        print(
            f"{period['name']}: {recommended_start.date()} (for training starting {period['training'][0]})"
        )

    print()
    print("=" * 80)
    print("OVERALL RECOMMENDATION")
    print("=" * 80)
    print()
    print("To safely run all 3 walk-forward periods, your dataset should include:")
    print()
    print("   📅 Data Range: 2019-12-01 to 2024-12-31")
    print()
    print("   This ensures:")
    print("   ✅ 250-day lookback for first week of 2021 training")
    print("   ✅ 250-day lookback for first week of 2022 training")
    print("   ✅ 250-day lookback for first week of 2023 training")
    print("   ✅ 30-day safety buffer for each period")
    print()
    print("=" * 80)


if __name__ == "__main__":
    verify_data_requirements()
