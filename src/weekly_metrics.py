"""Weekly Metrics generation pipeline.

Aggregates cleaned data into a 40-row Store x Week analytical dataset.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np


def load_cleaned_data() -> Dict[str, pd.DataFrame]:
    """Loads the cleaned CSV datasets."""
    data_dir = Path(__file__).resolve().parent.parent / "outputs" / "cleaned"
    
    files = {
        "stores": data_dir / "stores_clean.csv",
        "transactions": data_dir / "transactions_clean.csv",
        "staffing_shifts": data_dir / "staffing_shifts_clean.csv",
        "returns": data_dir / "returns_clean.csv",
    }
    
    dataframes = {}
    for name, filepath in files.items():
        if not filepath.exists():
            raise FileNotFoundError(f"Cleaned data file missing: {filepath}")
        dataframes[name] = pd.read_csv(filepath)
        
    return dataframes


def add_week_start(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Adds a Monday-based week_start column derived from date_col."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # W-SUN period means week ending on Sunday -> week starting on Monday
    df["week_start"] = df[date_col].dt.to_period("W-SUN").dt.start_time
    return df


def build_transaction_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates transactions at the store/week level."""
    df = add_week_start(df, "timestamp")
    
    metrics = df.groupby(["store_id", "week_start"]).agg(
        weekly_revenue=("amount", "sum"),
        transaction_count=("transaction_id", "nunique")
    ).reset_index()
    return metrics


def build_return_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates returns at the store/week level."""
    df = add_week_start(df, "date")
    
    metrics = df.groupby(["store_id", "week_start"]).agg(
        return_count=("return_id", "count"),
        return_amount=("amount", "sum")
    ).reset_index()
    return metrics


def build_staffing_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates staffing hours at the store/week level."""
    df = add_week_start(df, "date")
    
    metrics = df.groupby(["store_id", "week_start"]).agg(
        staffing_hours=("hours_worked", "sum")
    ).reset_index()
    return metrics


def build_store_week_grid(stores_df: pd.DataFrame) -> pd.DataFrame:
    """Creates the complete Cartesian product of valid stores and expected weeks."""
    valid_stores = sorted(stores_df["store_id"].dropna().unique().tolist())
    
    expected_weeks = [
        "2025-04-07",
        "2025-04-14",
        "2025-04-21",
        "2025-04-28",
        "2025-05-05",
        "2025-05-12",
        "2025-05-19",
        "2025-05-26"
    ]
    
    grid = []
    for store in valid_stores:
        for week in expected_weeks:
            grid.append({"store_id": store, "week_start": pd.to_datetime(week)})
            
    return pd.DataFrame(grid)


def build_weekly_metrics(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Builds the final joined dataset with derived metrics."""
    # 1. Independent aggregation
    tx_metrics = build_transaction_metrics(dfs["transactions"])
    return_metrics = build_return_metrics(dfs["returns"])
    staffing_metrics = build_staffing_metrics(dfs["staffing_shifts"])
    
    # 2. Build grid
    grid = build_store_week_grid(dfs["stores"])
    
    # 3. Left join independently aggregated datasets
    final_df = grid.merge(tx_metrics, on=["store_id", "week_start"], how="left")
    final_df = final_df.merge(return_metrics, on=["store_id", "week_start"], how="left")
    final_df = final_df.merge(staffing_metrics, on=["store_id", "week_start"], how="left")
    
    # 4. Handle Missing values as per instructions
    final_df["return_count"] = final_df["return_count"].fillna(0)
    final_df["return_amount"] = final_df["return_amount"].fillna(0.0)
    
    # 5. Derived metrics
    # return_rate = return_amount / weekly_revenue
    final_df["return_rate"] = np.where(
        final_df["weekly_revenue"].notnull() & (final_df["weekly_revenue"] > 0),
        final_df["return_amount"] / final_df["weekly_revenue"],
        np.nan
    )
    
    # sales_per_staffed_hour = weekly_revenue / staffing_hours
    final_df["sales_per_staffed_hour"] = np.where(
        final_df["staffing_hours"].notnull() & (final_df["staffing_hours"] > 0),
        final_df["weekly_revenue"] / final_df["staffing_hours"],
        np.nan
    )
    
    # Sort and order columns
    cols = [
        "store_id",
        "week_start",
        "weekly_revenue",
        "transaction_count",
        "return_amount",
        "return_count",
        "return_rate",
        "staffing_hours",
        "sales_per_staffed_hour"
    ]
    final_df = final_df[cols].sort_values(["store_id", "week_start"]).reset_index(drop=True)
    
    # Convert week_start to clear date format YYYY-MM-DD
    final_df["week_start"] = final_df["week_start"].dt.strftime("%Y-%m-%d")
    
    return final_df


def validate_weekly_metrics(final_df: pd.DataFrame, raw_dfs: Dict[str, pd.DataFrame]):
    """Validates the structure, conservation, and constraints of the final dataset."""
    print("Conservation and Validation Checks:")
    
    # A. Transaction revenue conservation
    clean_tx_rev = raw_dfs["transactions"]["amount"].sum()
    grid_tx_rev = final_df["weekly_revenue"].sum()
    rev_diff = abs(clean_tx_rev - grid_tx_rev)
    rev_pass = rev_diff < 0.01
    print(f"Revenue conservation: {'PASS' if rev_pass else 'FAIL'} (Diff: {rev_diff:.4f})")
    assert rev_pass, "Revenue conservation failed."
    
    # B. Transaction count conservation
    clean_tx_count = len(raw_dfs["transactions"])
    grid_tx_count = final_df["transaction_count"].sum()
    tx_pass = clean_tx_count == grid_tx_count
    print(f"Transaction conservation: {'PASS' if tx_pass else 'FAIL'}")
    assert tx_pass, "Transaction count conservation failed."
    
    # C. Staffing hour conservation
    clean_hours = raw_dfs["staffing_shifts"]["hours_worked"].sum()
    grid_hours = final_df["staffing_hours"].sum()
    hours_diff = abs(clean_hours - grid_hours)
    hours_pass = hours_diff < 0.01
    print(f"Staffing conservation: {'PASS' if hours_pass else 'FAIL'} (Diff: {hours_diff:.4f})")
    assert hours_pass, "Staffing hours conservation failed."
    
    # D. Return count conservation
    clean_returns = len(raw_dfs["returns"])
    grid_returns = final_df["return_count"].sum()
    ret_pass = clean_returns == grid_returns
    print(f"Return count conservation: {'PASS' if ret_pass else 'FAIL'}")
    assert ret_pass, "Return count conservation failed."
    
    # E. Return amount conservation
    clean_returns_amt = raw_dfs["returns"]["amount"].sum()
    grid_returns_amt = final_df["return_amount"].sum()
    ret_amt_diff = abs(clean_returns_amt - grid_returns_amt)
    ret_amt_pass = ret_amt_diff < 0.01
    print(f"Return amount conservation: {'PASS' if ret_amt_pass else 'FAIL'} (Diff: {ret_amt_diff:.4f})")
    assert ret_amt_pass, "Return amount conservation failed."
    
    # 1-6. Grid structure checks
    assert len(final_df) == 40, f"Expected 40 rows, got {len(final_df)}"
    assert final_df["store_id"].nunique() == 5, f"Expected 5 stores, got {final_df['store_id'].nunique()}"
    
    store_counts = final_df["store_id"].value_counts()
    assert all(count == 8 for count in store_counts), "Not all stores have 8 rows."
    assert final_df["week_start"].nunique() == 8, "Expected 8 unique weeks."
    assert not final_df.duplicated(subset=["store_id", "week_start"]).any(), "Duplicate store/week pairs found."
    
    # 7-12. Metric constraints
    assert (final_df["weekly_revenue"].dropna() >= 0).all(), "Negative revenue found."
    assert (final_df["transaction_count"].dropna() >= 0).all(), "Negative transaction count found."
    assert (final_df["return_amount"].dropna() >= 0).all(), "Negative return amount found."
    assert (final_df["return_count"].dropna() >= 0).all(), "Negative return count found."
    assert (final_df["staffing_hours"].dropna() >= 0).all(), "Negative staffing hours found."
    
    valid_spsh = final_df["sales_per_staffed_hour"].dropna()
    assert np.isfinite(valid_spsh).all(), "Non-finite sales_per_staffed_hour found."
    
    valid_return_rate = final_df["return_rate"].dropna()
    assert np.isfinite(valid_return_rate).all(), "Non-finite return_rate found."
    
    # 13. Sales per staffed hour verification
    mask = final_df["staffing_hours"] > 0
    check_spsh = final_df.loc[mask, "weekly_revenue"] / final_df.loc[mask, "staffing_hours"]
    assert np.allclose(final_df.loc[mask, "sales_per_staffed_hour"], check_spsh), "sales_per_staffed_hour calculation mismatch."
    
    # 13b. Return rate verification
    mask_rev = final_df["weekly_revenue"] > 0
    check_rr = final_df.loc[mask_rev, "return_amount"] / final_df.loc[mask_rev, "weekly_revenue"]
    assert np.allclose(final_df.loc[mask_rev, "return_rate"], check_rr), "return_rate calculation mismatch."
    
    mask_no_rev = final_df["weekly_revenue"].isnull() | (final_df["weekly_revenue"] <= 0)
    assert final_df.loc[mask_no_rev, "return_rate"].isnull().all(), "return_rate should be NaN when revenue <= 0."
    
    # 14. Special case validation
    s04_week = final_df[(final_df["store_id"] == "S04") & (final_df["week_start"] == "2025-05-12")]
    assert len(s04_week) == 1, "S04 2025-05-12 row missing or duplicated."
    assert pd.isna(s04_week.iloc[0]["staffing_hours"]), "S04 2025-05-12 staffing_hours is not NaN."
    assert pd.isna(s04_week.iloc[0]["sales_per_staffed_hour"]), "S04 2025-05-12 sales_per_staffed_hour is not NaN."
    
    print("\nValidation: PASS")


def save_weekly_metrics(df: pd.DataFrame):
    """Saves the final analytical dataset."""
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "weekly_metrics.csv"
    df.to_csv(output_file, index=False)
    
    # For output reporting, format nicely relative to project root
    try:
        rel_path = output_file.relative_to(Path(__file__).resolve().parent.parent)
    except ValueError:
        rel_path = output_file
    print(f"Output:\n{rel_path}")


def main():
    print("=" * 60)
    print("WEEKLY METRICS")
    print("=" * 60)
    
    raw_dfs = load_cleaned_data()
    final_df = build_weekly_metrics(raw_dfs)
    
    print(f"\nStore-week rows: {len(final_df)}")
    print(f"Stores: {final_df['store_id'].nunique()}")
    print(f"Weeks: {final_df['week_start'].nunique()}\n")
    
    validate_weekly_metrics(final_df, raw_dfs)
    
    missing_staffing = final_df[final_df["staffing_hours"].isnull()]
    if not missing_staffing.empty:
        print("\nMissing staffing combinations:")
        for _, row in missing_staffing.iterrows():
            print(f"{row['store_id']} | {row['week_start']}")
    
    print()
    save_weekly_metrics(final_df)


if __name__ == "__main__":
    main()
