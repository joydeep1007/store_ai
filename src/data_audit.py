"""Data-quality auditing logic for the retail datasets.

This module contains functions to audit, validate, and check
the integrity of raw retail operations data files.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


def load_data() -> Dict[str, pd.DataFrame]:
    """Loads the four required CSV files from the data directory."""
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


def audit_schema(df: pd.DataFrame, expected_columns: List[str]) -> Dict[str, Any]:
    """Checks for expected columns, unexpected columns, and returns row/col counts."""
    actual_columns = set(df.columns)
    expected_set = set(expected_columns)
    
    missing_columns = list(expected_set - actual_columns)
    unexpected_columns = list(actual_columns - expected_set)
    
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_expected_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "data_types": dtypes
    }


def audit_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates missing counts and percentages per column."""
    missing_counts = df.isnull().sum()
    missing_pcts = (missing_counts / len(df)) * 100 if len(df) > 0 else missing_counts
    
    results = {}
    for col in df.columns:
        if missing_counts[col] > 0:
            results[col] = {
                "missing_count": int(missing_counts[col]),
                "missing_pct": float(missing_pcts[col])
            }
            
    return results


def audit_duplicates(df: pd.DataFrame, id_column: str) -> Dict[str, Any]:
    """Identifies completely duplicated rows and duplicate primary IDs."""
    # Completely duplicated rows
    full_duplicates_mask = df.duplicated(keep=False)
    num_full_duplicates = int(full_duplicates_mask.sum())
    
    # Duplicate IDs
    if id_column in df.columns:
        id_duplicates_mask = df.duplicated(subset=[id_column], keep=False)
        # We only care about non-null IDs being duplicated
        valid_id_mask = id_duplicates_mask & df[id_column].notnull()
        num_id_duplicates = int(valid_id_mask.sum())
        
        # Get duplicate IDs
        duplicate_ids = df.loc[valid_id_mask, id_column].unique().tolist()
    else:
        num_id_duplicates = 0
        duplicate_ids = []
        
    return {
        "completely_duplicated_rows": num_full_duplicates,
        "rows_with_duplicate_ids": num_id_duplicates,
        "duplicate_id_values": duplicate_ids[:10], # sample up to 10
        "total_duplicate_id_values": len(duplicate_ids)
    }


def audit_store_references(df: pd.DataFrame, valid_store_ids: set) -> Dict[str, Any]:
    """Checks if store_ids in the dataframe exist in the valid stores set."""
    if "store_id" not in df.columns:
        return {}
        
    # Ignore nulls for referential integrity (they are captured in missing values)
    valid_mask = df["store_id"].notnull()
    non_null_stores = df.loc[valid_mask, "store_id"]
    
    invalid_mask = ~non_null_stores.isin(valid_store_ids)
    invalid_references = non_null_stores[invalid_mask]
    
    invalid_count = int(invalid_references.count())
    invalid_ids = invalid_references.unique().tolist()
    
    pct_affected = (invalid_count / len(df)) * 100 if len(df) > 0 else 0.0
    
    return {
        "invalid_reference_row_count": invalid_count,
        "invalid_store_ids": invalid_ids,
        "percentage_affected": pct_affected
    }


def audit_dates(df: pd.DataFrame, date_column: str) -> Dict[str, Any]:
    """Audits a date column for missing, invalid format, and range."""
    if date_column not in df.columns:
        return {}
        
    parsed_dates = pd.to_datetime(df[date_column], errors="coerce")
    
    # Identify rows where the original was not null, but parsed is null
    original_not_null = df[date_column].notnull()
    parsed_is_null = parsed_dates.isnull()
    
    invalid_format_mask = original_not_null & parsed_is_null
    invalid_format_count = int(invalid_format_mask.sum())
    
    invalid_examples = df.loc[invalid_format_mask, date_column].head(5).tolist()
    
    # Get range from valid dates
    valid_dates = parsed_dates.dropna()
    min_date = valid_dates.min().isoformat() if not valid_dates.empty else None
    max_date = valid_dates.max().isoformat() if not valid_dates.empty else None
    
    return {
        "invalid_format_count": invalid_format_count,
        "invalid_examples": invalid_examples,
        "min_valid_date": min_date,
        "max_valid_date": max_date
    }


def audit_domain_values(dataset_name: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Audits dataset-specific domain constraints."""
    issues = {}
    
    if dataset_name == "transactions":
        if "amount" in df.columns:
            # use pd.to_numeric to handle strings gracefully, coercing errors
            amt = pd.to_numeric(df["amount"], errors="coerce")
            neg_amt = int((amt < 0).sum())
            zero_amt = int((amt == 0).sum())
            if neg_amt > 0: issues["negative_amount_detected"] = neg_amt
            if zero_amt > 0: issues["zero_amount_detected"] = zero_amt
            
        if "item_count" in df.columns:
            items = pd.to_numeric(df["item_count"], errors="coerce")
            non_positive_items = int((items <= 0).sum())
            if non_positive_items > 0: issues["non_positive_item_count"] = non_positive_items
            
        if "channel" in df.columns:
            channels = df["channel"].dropna().unique().tolist()
            issues["distinct_channels"] = channels
            
    elif dataset_name == "staffing_shifts":
        if "hours_worked" in df.columns:
            hrs = pd.to_numeric(df["hours_worked"], errors="coerce")
            neg_hrs = int((hrs < 0).sum())
            zero_hrs = int((hrs == 0).sum())
            over_24_hrs = int((hrs > 24).sum())
            
            if neg_hrs > 0: issues["negative_hours_detected"] = neg_hrs
            if zero_hrs > 0: issues["zero_hours_detected"] = zero_hrs
            if over_24_hrs > 0: issues["hours_above_24_detected"] = over_24_hrs
            
        if "role" in df.columns:
            roles = df["role"].dropna().unique().tolist()
            issues["distinct_roles"] = roles
            
    elif dataset_name == "returns":
        if "amount" in df.columns:
            amt = pd.to_numeric(df["amount"], errors="coerce")
            neg_amt = int((amt < 0).sum())
            zero_amt = int((amt == 0).sum())
            if neg_amt > 0: issues["negative_amount_detected"] = neg_amt
            if zero_amt > 0: issues["zero_amount_detected"] = zero_amt
            
    elif dataset_name == "stores":
        if "size_sqft" in df.columns:
            size = pd.to_numeric(df["size_sqft"], errors="coerce")
            neg_size = int((size <= 0).sum())
            if neg_size > 0: issues["suspicious_non_positive_size_sqft"] = neg_size
            
        if "opened_year" in df.columns:
            year = pd.to_numeric(df["opened_year"], errors="coerce")
            future_years = int((year > 2025).sum())
            old_years = int((year < 1900).sum())
            if future_years > 0 or old_years > 0:
                issues["suspicious_opened_year"] = future_years + old_years
                
    return issues


def audit_store_week_coverage(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Generates expected store x week combinations and compares against actuals."""
    results = {}
    
    date_series = []
    
    if "transactions" in dfs and "timestamp" in dfs["transactions"].columns:
        date_series.append(pd.to_datetime(dfs["transactions"]["timestamp"], errors="coerce").dropna())
    if "staffing_shifts" in dfs and "date" in dfs["staffing_shifts"].columns:
        date_series.append(pd.to_datetime(dfs["staffing_shifts"]["date"], errors="coerce").dropna())
    if "returns" in dfs and "date" in dfs["returns"].columns:
        date_series.append(pd.to_datetime(dfs["returns"]["date"], errors="coerce").dropna())
        
    if not date_series:
        return {"error": "No valid dates found across datasets to determine coverage."}
        
    all_dates = pd.concat(date_series)
    min_date = all_dates.min()
    max_date = all_dates.max()
    
    # Generate weekly period range (Monday to Sunday weeks typically, pd uses W-MON to signify week ending on Mon)
    # Using 'W-MON' for weekly frequency. W-MON sets the anchor. 
    # Let's use W-SUN to align with typical ISO weeks (Mon-Sun), week ending Sunday.
    weeks = pd.date_range(start=min_date - pd.Timedelta(days=min_date.weekday()), 
                          end=max_date, 
                          freq="W-SUN")
    
    valid_stores = []
    if "stores" in dfs and "store_id" in dfs["stores"].columns:
        valid_stores = dfs["stores"]["store_id"].dropna().unique().tolist()
        
    results["observed_date_range"] = {"min": min_date.isoformat(), "max": max_date.isoformat()}
    results["total_weeks"] = len(weeks)
    results["total_stores"] = len(valid_stores)
    results["expected_combinations"] = len(weeks) * len(valid_stores)
    
    def get_coverage(df, date_col):
        if date_col not in df.columns or "store_id" not in df.columns:
            return 0
        
        valid_df = df.copy()
        valid_df["parsed_date"] = pd.to_datetime(valid_df[date_col], errors="coerce")
        valid_df = valid_df.dropna(subset=["parsed_date", "store_id"])
        valid_df = valid_df[valid_df["store_id"].isin(valid_stores)]
        
        valid_df["week"] = valid_df["parsed_date"].dt.to_period('W-SUN').dt.start_time
        unique_combos = valid_df.groupby(["store_id", "week"]).size().reset_index()
        return len(unique_combos)
        
    missing_coverage = {}
    
    if "transactions" in dfs:
        obs = get_coverage(dfs["transactions"], "timestamp")
        missing_coverage["transactions"] = results["expected_combinations"] - obs
        
    if "staffing_shifts" in dfs:
        obs = get_coverage(dfs["staffing_shifts"], "date")
        missing_coverage["staffing_shifts"] = results["expected_combinations"] - obs
        
    if "returns" in dfs:
        obs = get_coverage(dfs["returns"], "date")
        missing_coverage["returns"] = results["expected_combinations"] - obs
        
    results["missing_store_week_combinations"] = missing_coverage
    
    return results


def run_audit() -> Dict[str, Any]:
    """Runs all audit checks and returns a structured report."""
    dfs = load_data()
    
    schemas = {
        "stores": ["store_id", "region", "size_sqft", "opened_year"],
        "transactions": ["transaction_id", "store_id", "timestamp", "amount", "item_count", "channel", "promo_code"],
        "staffing_shifts": ["shift_id", "store_id", "employee_id", "date", "hours_worked", "role"],
        "returns": ["return_id", "store_id", "date", "amount", "reason_code"]
    }
    
    id_columns = {
        "stores": "store_id",
        "transactions": "transaction_id",
        "staffing_shifts": "shift_id",
        "returns": "return_id"
    }
    
    date_columns = {
        "transactions": "timestamp",
        "staffing_shifts": "date",
        "returns": "date"
    }
    
    report = {}
    
    valid_stores = set()
    if "stores" in dfs and "store_id" in dfs["stores"].columns:
        valid_stores = set(dfs["stores"]["store_id"].dropna().unique())
        
    for name, df in dfs.items():
        report[name] = {
            "schema": audit_schema(df, schemas.get(name, [])),
            "missing_values": audit_missing_values(df),
            "duplicates": audit_duplicates(df, id_columns.get(name, "")),
            "domain_checks": audit_domain_values(name, df)
        }
        
        if name in date_columns:
            report[name]["dates"] = audit_dates(df, date_columns[name])
            
        if name != "stores":
            report[name]["store_references"] = audit_store_references(df, valid_stores)
            
    report["cross_dataset"] = {
        "store_week_coverage": audit_store_week_coverage(dfs)
    }
            
    return report


def print_audit_report(report: Dict[str, Any]):
    """Prints the structured report in a human-readable format."""
    print("=" * 60)
    print("RETAIL OPERATIONS DATA AUDIT")
    print("=" * 60)
    
    for dataset in ["stores", "transactions", "staffing_shifts", "returns"]:
        if dataset not in report:
            continue
            
        data = report[dataset]
        print(f"\nDATASET: {dataset.upper()}")
        print("-" * 60)
        
        schema = data.get("schema", {})
        print(f"Rows: {schema.get('row_count', 0)}")
        print(f"Columns: {schema.get('column_count', 0)}")
        
        if schema.get("missing_expected_columns"):
            print(f"Missing expected columns: {schema['missing_expected_columns']}")
        if schema.get("unexpected_columns"):
            print(f"Unexpected columns: {schema['unexpected_columns']}")
            
        dupes = data.get("duplicates", {})
        print(f"Duplicate rows: {dupes.get('completely_duplicated_rows', 0)}")
        print(f"Duplicate IDs: {dupes.get('rows_with_duplicate_ids', 0)} (Total unique duplicate IDs: {dupes.get('total_duplicate_id_values', 0)})")
        if dupes.get("duplicate_id_values"):
            print(f"  Example duplicate IDs: {dupes['duplicate_id_values']}")
            
        print("\nMissing values:")
        missing = data.get("missing_values", {})
        if not missing:
            print("  None detected.")
        else:
            for col, info in missing.items():
                print(f"  - {col}: {info['missing_count']} ({info['missing_pct']:.2f}%)")
                
        if "store_references" in data:
            print("\nInvalid store references:")
            refs = data["store_references"]
            if refs.get("invalid_reference_row_count", 0) > 0:
                print(f"  Invalid references: {refs['invalid_reference_row_count']} rows affected ({refs['percentage_affected']:.2f}%)")
                print(f"  Invalid store IDs (examples): {refs.get('invalid_store_ids', [])[:10]}")
            else:
                print("  None detected.")
                
        if "dates" in data:
            print("\nDate issues:")
            dates = data["dates"]
            if dates.get("invalid_format_count", 0) > 0:
                print(f"  Invalid format count: {dates['invalid_format_count']}")
                print(f"  Examples: {dates['invalid_examples']}")
            else:
                print("  No invalid formats detected.")
            print(f"  Valid date range: {dates.get('min_valid_date')} to {dates.get('max_valid_date')}")
            
        print("\nDomain issues:")
        domain = data.get("domain_checks", {})
        if not domain:
            print("  None detected.")
        else:
            for k, v in domain.items():
                print(f"  - {k}: {v}")

    print("\n\nCROSS-DATASET INTEGRITY")
    print("-" * 60)
    
    coverage = report.get("cross_dataset", {}).get("store_week_coverage", {})
    if "error" in coverage:
        print(f"Coverage error: {coverage['error']}")
    else:
        print("Store/Week Coverage:")
        print(f"  Observed Date Range: {coverage.get('observed_date_range')}")
        print(f"  Total Weeks: {coverage.get('total_weeks')}")
        print(f"  Total Valid Stores: {coverage.get('total_stores')}")
        print(f"  Expected Combinations (Stores x Weeks): {coverage.get('expected_combinations')}")
        
        missing = coverage.get("missing_store_week_combinations", {})
        print("\n  Missing combinations by dataset:")
        for ds, missing_count in missing.items():
            print(f"    - {ds}: {missing_count} missing store/week combinations")

    print("\n" + "=" * 60)


def save_audit_report(report: Dict[str, Any]):
    """Saves the audit report to outputs/audit_report.json."""
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "audit_report.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Audit report saved to {output_file}")


if __name__ == "__main__":
    report_data = run_audit()
    print_audit_report(report_data)
    save_audit_report(report_data)
