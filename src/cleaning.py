"""Phase 2: Data Cleaning pipeline for retail operations analytics.

Loads raw CSV data, applies approved cleaning decisions, and outputs
cleaned analytical datasets.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd


def load_raw_data() -> Dict[str, pd.DataFrame]:
    """Loads the raw CSV files from the data directory."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    
    files = {
        "stores": data_dir / "stores.csv",
        "transactions": data_dir / "transactions.csv",
        "staffing_shifts": data_dir / "staffing_shifts.csv",
        "returns": data_dir / "returns.csv",
    }
    
    dataframes = {}
    for name, filepath in files.items():
        if not filepath.exists():
            raise FileNotFoundError(f"Required data file missing: {filepath}")
        dataframes[name] = pd.read_csv(filepath)
        
    return dataframes


def clean_stores(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Cleans the stores dataset.
    
    Currently no removals are needed based on audit.
    """
    input_rows = len(df)
    clean_df = df.copy()
    
    # Cast types if necessary (though pandas usually infers correctly for stores)
    if "size_sqft" in clean_df.columns:
        clean_df["size_sqft"] = pd.to_numeric(clean_df["size_sqft"], errors="coerce")
    if "opened_year" in clean_df.columns:
        clean_df["opened_year"] = pd.to_numeric(clean_df["opened_year"], errors="coerce")
        
    report = {
        "input_rows": input_rows,
        "output_rows": len(clean_df),
    }
    
    return clean_df, report


def clean_transactions(df: pd.DataFrame, valid_store_ids: set) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Cleans the transactions dataset."""
    input_rows = len(df)
    clean_df = df.copy()
    
    # 1. Parse dates and types
    if "timestamp" in clean_df.columns:
        clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], errors="coerce")
    if "amount" in clean_df.columns:
        clean_df["amount"] = pd.to_numeric(clean_df["amount"], errors="coerce")
    if "item_count" in clean_df.columns:
        clean_df["item_count"] = pd.to_numeric(clean_df["item_count"], errors="coerce")
        
    # 2. Duplicate Transactions
    # Check for duplicates by transaction_id
    dupe_mask = clean_df.duplicated(subset=["transaction_id"], keep=False)
    duplicates = clean_df[dupe_mask]
    
    for tx_id, group in duplicates.groupby("transaction_id"):
        # We fillna before dropping duplicates to safely compare all values including NaNs
        is_exact = len(group.fillna("MISSING").drop_duplicates()) == 1
        if not is_exact:
            raise ValueError(f"Conflicting duplicate group found for transaction_id: {tx_id}")
            
    # Safe to drop exact duplicates (keep='first')
    clean_df = clean_df.drop_duplicates(subset=["transaction_id"], keep="first")
    duplicate_rows_removed = input_rows - len(clean_df)
    
    # 3. Invalid Store References
    if "store_id" in clean_df.columns:
        invalid_store_mask = ~clean_df["store_id"].isin(valid_store_ids)
        invalid_store_rows = clean_df[invalid_store_mask].copy()
        
        # Exclude invalid
        clean_df = clean_df[~invalid_store_mask]
        invalid_store_rows_removed = len(invalid_store_rows)
        
        # convert excluded rows to dict for report (replace NaT/NaN to None for JSON serializability)
        excluded_records = invalid_store_rows.astype(object).where(pd.notnull(invalid_store_rows), None).to_dict(orient="records")
    else:
        invalid_store_rows_removed = 0
        excluded_records = []
        
    report = {
        "input_rows": input_rows,
        "duplicate_rows_removed": duplicate_rows_removed,
        "invalid_store_rows_removed": invalid_store_rows_removed,
        "output_rows": len(clean_df),
        "excluded_invalid_store_records": excluded_records
    }
    
    return clean_df, report


def clean_staffing(df: pd.DataFrame, valid_store_ids: set) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Cleans the staffing dataset."""
    input_rows = len(df)
    clean_df = df.copy()
    
    # 1. Parse dates and types
    if "date" in clean_df.columns:
        clean_df["date"] = pd.to_datetime(clean_df["date"], errors="coerce")
    if "hours_worked" in clean_df.columns:
        clean_df["hours_worked"] = pd.to_numeric(clean_df["hours_worked"], errors="coerce")
        
    # 2. Invalid Store References
    if "store_id" in clean_df.columns:
        invalid_store_mask = ~clean_df["store_id"].isin(valid_store_ids)
        invalid_store_rows = clean_df[invalid_store_mask].copy()
        
        # Exclude invalid
        clean_df = clean_df[~invalid_store_mask]
        invalid_store_rows_removed = len(invalid_store_rows)
        
        excluded_records = invalid_store_rows.astype(object).where(pd.notnull(invalid_store_rows), None).to_dict(orient="records")
    else:
        invalid_store_rows_removed = 0
        excluded_records = []
        
    # Note: Temp role is kept. Missing week is not imputed (kept missing).
    
    report = {
        "input_rows": input_rows,
        "invalid_store_rows_removed": invalid_store_rows_removed,
        "output_rows": len(clean_df),
        "excluded_invalid_store_records": excluded_records
    }
    
    return clean_df, report


def clean_returns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Cleans the returns dataset."""
    input_rows = len(df)
    clean_df = df.copy()
    
    # 1. Parse dates and types
    if "date" in clean_df.columns:
        clean_df["date"] = pd.to_datetime(clean_df["date"], errors="coerce")
    if "amount" in clean_df.columns:
        clean_df["amount"] = pd.to_numeric(clean_df["amount"], errors="coerce")
        
    report = {
        "input_rows": input_rows,
        "output_rows": len(clean_df),
    }
    
    return clean_df, report


def build_clean_datasets() -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Runs the full cleaning pipeline and returns cleaned dataframes and report."""
    raw_dfs = load_raw_data()
    
    cleaned_dfs = {}
    report = {}
    
    # 1. Stores (Authoritative dimension)
    stores_df = raw_dfs["stores"]
    cleaned_dfs["stores"], report["stores"] = clean_stores(stores_df)
    
    valid_store_ids = set(cleaned_dfs["stores"]["store_id"].dropna().unique())
    
    # 2. Transactions
    if "transactions" in raw_dfs:
        cleaned_dfs["transactions"], report["transactions"] = clean_transactions(
            raw_dfs["transactions"], valid_store_ids
        )
        
    # 3. Staffing Shifts
    if "staffing_shifts" in raw_dfs:
        cleaned_dfs["staffing_shifts"], report["staffing_shifts"] = clean_staffing(
            raw_dfs["staffing_shifts"], valid_store_ids
        )
        
    # 4. Returns
    if "returns" in raw_dfs:
        cleaned_dfs["returns"], report["returns"] = clean_returns(
            raw_dfs["returns"]
        )
        
    return cleaned_dfs, report


def save_cleaned_data(cleaned_dfs: Dict[str, pd.DataFrame]):
    """Saves cleaned dataframes to outputs/cleaned/ without altering raw files."""
    cleaned_dir = Path(__file__).resolve().parent.parent / "outputs" / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    
    for name, df in cleaned_dfs.items():
        # Keep original timestamp parsing for CSV by default or format nicely
        file_path = cleaned_dir / f"{name}_clean.csv"
        # don't save index
        df.to_csv(file_path, index=False)


def save_cleaning_report(report: Dict[str, Any]):
    """Saves the cleaning metadata and excluded records to a JSON report."""
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / "cleaning_report.json"
    
    # Create a serializable version of the report (handle datetimes if any)
    # The datetime conversion is already handled in the dictionary creation by using .astype(object)
    
    with open(report_file, "w") as f:
        json.dump(report, f, indent=4, default=str)


def validate_cleaning(raw_dfs: Dict[str, pd.DataFrame], cleaned_dfs: Dict[str, pd.DataFrame], valid_store_ids: set):
    """Validates the cleaned data constraints to ensure adherence to rules."""
    
    print("\n" + "=" * 60)
    print("VALIDATION AFTER CLEANING")
    print("=" * 60)
    
    tx = cleaned_dfs.get("transactions")
    if tx is not None:
        dupes = tx.duplicated(subset=["transaction_id"]).sum()
        invalid_stores = (~tx["store_id"].isin(valid_store_ids)).sum()
        print(f"Transactions duplicate IDs remaining: {dupes}")
        print(f"Transactions invalid store IDs remaining: {invalid_stores}")
        assert dupes == 0, "Duplicate transaction IDs remain!"
        assert invalid_stores == 0, "Invalid store IDs remain in transactions!"
        
    staffing = cleaned_dfs.get("staffing_shifts")
    if staffing is not None:
        invalid_stores = (~staffing["store_id"].isin(valid_store_ids)).sum()
        print(f"Staffing invalid store IDs remaining: {invalid_stores}")
        assert invalid_stores == 0, "Invalid store IDs remain in staffing!"
        
    # Verify raw CSVs were not modified
    data_dir = Path(__file__).resolve().parent.parent / "data"
    raw_csvs = {
        "stores": len(pd.read_csv(data_dir / "stores.csv")),
        "transactions": len(pd.read_csv(data_dir / "transactions.csv")),
        "staffing_shifts": len(pd.read_csv(data_dir / "staffing_shifts.csv")),
        "returns": len(pd.read_csv(data_dir / "returns.csv"))
    }
    
    print("\nRaw File Integrity Check:")
    for ds_name, expected_len in raw_csvs.items():
        actual_raw_len = len(raw_dfs[ds_name])
        print(f"  {ds_name}.csv: Unchanged (Length: {actual_raw_len})")
        assert expected_len == actual_raw_len, f"Raw data for {ds_name} was somehow modified on disk!"
        
    print("\nAll validation checks passed.")


if __name__ == "__main__":
    raw_data = load_raw_data()
    cleaned_data, clean_report = build_clean_datasets()
    
    print("=" * 60)
    print("DATA CLEANING SUMMARY")
    print("=" * 60)
    
    print("RAW -> CLEANED")
    for name in ["stores", "transactions", "staffing_shifts", "returns"]:
        if name in clean_report:
            rep = clean_report[name]
            print(f"\n{name}:")
            print(f"{rep['input_rows']} -> {rep['output_rows']}")
            
            if "duplicate_rows_removed" in rep and rep["duplicate_rows_removed"] > 0:
                print(f"  - Duplicate rows removed: {rep['duplicate_rows_removed']}")
            if "invalid_store_rows_removed" in rep and rep["invalid_store_rows_removed"] > 0:
                print(f"  - Invalid store references removed: {rep['invalid_store_rows_removed']}")
                
    save_cleaned_data(cleaned_data)
    save_cleaning_report(clean_report)
    
    valid_stores = set(cleaned_data["stores"]["store_id"].unique())
    validate_cleaning(raw_data, cleaned_data, valid_stores)

