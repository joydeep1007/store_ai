"""Investigation logic for the anomalies discovered during data audit.

This module drills down into specific data issues to provide evidence for
later cleaning decisions.
"""

from pathlib import Path
from typing import Dict, Any

import pandas as pd


def load_data() -> Dict[str, pd.DataFrame]:
    """Loads the required CSV files from the data directory."""
    data_dir = Path(__file__).resolve().parent.parent / "data"
    
    files = {
        "stores": data_dir / "stores.csv",
        "transactions": data_dir / "transactions.csv",
        "staffing_shifts": data_dir / "staffing_shifts.csv",
    }
    
    dataframes = {}
    for name, filepath in files.items():
        if not filepath.exists():
            raise FileNotFoundError(f"Required data file missing: {filepath}")
        dataframes[name] = pd.read_csv(filepath)
        
    return dataframes


def investigate_duplicate_transactions(df: pd.DataFrame) -> Dict[str, Any]:
    """Investigates transaction duplicates to see if they are exact or conflicting."""
    if "transaction_id" not in df.columns:
        return {}
        
    dupe_mask = df.duplicated(subset=["transaction_id"], keep=False)
    dupe_df = df[dupe_mask].copy()
    
    total_rows = len(dupe_df)
    unique_ids = dupe_df["transaction_id"].unique()
    total_ids = len(unique_ids)
    
    exact_groups = []
    conflicting_groups = []
    
    for tx_id, group in dupe_df.groupby("transaction_id"):
        # Check if all rows in the group are identical across all columns
        # fillna to safely compare NaN values
        is_exact = len(group.fillna("MISSING").drop_duplicates()) == 1
        
        group_records = group.to_dict(orient="records")
        
        if is_exact:
            exact_groups.append({
                "transaction_id": tx_id,
                "occurrences": len(group),
                "rows": group_records
            })
        else:
            conflicting_groups.append({
                "transaction_id": tx_id,
                "occurrences": len(group),
                "rows": group_records
            })
            
    return {
        "total_duplicate_ids": total_ids,
        "total_rows_involved": total_rows,
        "exact_duplicate_groups_count": len(exact_groups),
        "conflicting_duplicate_groups_count": len(conflicting_groups),
        "exact_examples": exact_groups[:2],
        "conflicting_examples": conflicting_groups[:2]
    }


def investigate_s09_transactions(df: pd.DataFrame) -> Dict[str, Any]:
    """Investigates transactions referencing the invalid store S09."""
    if "store_id" not in df.columns:
        return {}
        
    s09_df = df[df["store_id"] == "S09"].copy()
    count = len(s09_df)
    
    # We want transaction IDs, dates, amounts, and complete rows
    tx_ids = s09_df["transaction_id"].tolist() if "transaction_id" in s09_df.columns else []
    dates = s09_df["timestamp"].tolist() if "timestamp" in s09_df.columns else []
    amounts = s09_df["amount"].tolist() if "amount" in s09_df.columns else []
    
    return {
        "count": count,
        "transaction_ids": tx_ids,
        "dates": dates,
        "amounts": amounts,
        "complete_rows": s09_df.to_dict(orient="records")
    }


def investigate_s09_staffing(df: pd.DataFrame) -> Dict[str, Any]:
    """Investigates staffing shifts referencing the invalid store S09."""
    if "store_id" not in df.columns:
        return {}
        
    s09_df = df[df["store_id"] == "S09"].copy()
    count = len(s09_df)
    
    shift_ids = s09_df["shift_id"].tolist() if "shift_id" in s09_df.columns else []
    emp_ids = s09_df["employee_id"].tolist() if "employee_id" in s09_df.columns else []
    dates = s09_df["date"].tolist() if "date" in s09_df.columns else []
    hours = s09_df["hours_worked"].tolist() if "hours_worked" in s09_df.columns else []
    roles = s09_df["role"].tolist() if "role" in s09_df.columns else []
    
    return {
        "count": count,
        "shift_ids": shift_ids,
        "employee_ids": emp_ids,
        "dates": dates,
        "hours": hours,
        "roles": roles,
        "complete_rows": s09_df.to_dict(orient="records")
    }


def investigate_missing_staffing_coverage(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """Identifies the exact missing store/week combination for staffing."""
    transactions_df = dfs.get("transactions")
    staffing_df = dfs.get("staffing_shifts")
    stores_df = dfs.get("stores")
    
    if transactions_df is None or staffing_df is None or stores_df is None:
        return {}
        
    valid_stores = stores_df["store_id"].dropna().unique().tolist()
    
    # Get overall min/max date from transactions and staffing
    dates1 = pd.to_datetime(transactions_df["timestamp"], errors="coerce").dropna()
    dates2 = pd.to_datetime(staffing_df["date"], errors="coerce").dropna()
    all_dates = pd.concat([dates1, dates2])
    
    min_date = all_dates.min()
    max_date = all_dates.max()
    
    # Generate weekly period range
    # Using 'W-SUN' period's start_time aligns to Mondays
    weeks = pd.date_range(start=min_date - pd.Timedelta(days=min_date.weekday()), 
                          end=max_date, 
                          freq="W-MON")
                          
    # Construct expected combinations
    expected_combos = set()
    for store in valid_stores:
        for week in weeks:
            expected_combos.add((store, week))
            
    # Construct observed staffing combinations
    valid_staffing = staffing_df.copy()
    valid_staffing["parsed_date"] = pd.to_datetime(valid_staffing["date"], errors="coerce")
    valid_staffing = valid_staffing.dropna(subset=["parsed_date", "store_id"])
    valid_staffing = valid_staffing[valid_staffing["store_id"].isin(valid_stores)]
    valid_staffing["week"] = valid_staffing["parsed_date"].dt.to_period('W-SUN').dt.start_time
    
    observed_combos = set()
    for _, row in valid_staffing.iterrows():
        observed_combos.add((row["store_id"], row["week"]))
        
    missing = list(expected_combos - observed_combos)
    
    result = {
        "missing_combinations": []
    }
    
    for store, week in missing:
        # Get surrounding weeks for this store
        prev_week = week - pd.Timedelta(days=7)
        next_week = week + pd.Timedelta(days=7)
        
        surrounding = valid_staffing[
            (valid_staffing["store_id"] == store) & 
            (valid_staffing["week"].isin([prev_week, week, next_week]))
        ]
        
        # Summarize surrounding coverage (e.g. total hours per week)
        surrounding_summary = surrounding.groupby("week")["hours_worked"].sum().to_dict()
        surrounding_str_keys = {k.isoformat(): v for k, v in surrounding_summary.items()}
        
        result["missing_combinations"].append({
            "store_id": store,
            "missing_week": week.isoformat(),
            "surrounding_weeks_coverage_hours": surrounding_str_keys
        })
        
    return result


def investigate_temp_roles(df: pd.DataFrame) -> Dict[str, Any]:
    """Investigates staffing rows with the 'temp' role."""
    if "role" not in df.columns:
        return {}
        
    temp_df = df[df["role"] == "temp"].copy()
    
    return {
        "count": len(temp_df),
        "employee_ids": temp_df["employee_id"].tolist() if "employee_id" in temp_df.columns else [],
        "store_ids": temp_df["store_id"].tolist() if "store_id" in temp_df.columns else [],
        "dates": temp_df["date"].tolist() if "date" in temp_df.columns else [],
        "hours": temp_df["hours_worked"].tolist() if "hours_worked" in temp_df.columns else [],
        "roles": temp_df["role"].tolist() if "role" in temp_df.columns else [],
        "complete_rows": temp_df.to_dict(orient="records")
    }


def run_investigation() -> Dict[str, Any]:
    """Runs all investigations and returns structured results."""
    dfs = load_data()
    
    results = {}
    if "transactions" in dfs:
        results["duplicate_transactions"] = investigate_duplicate_transactions(dfs["transactions"])
        results["s09_transactions"] = investigate_s09_transactions(dfs["transactions"])
        
    if "staffing_shifts" in dfs:
        results["s09_staffing"] = investigate_s09_staffing(dfs["staffing_shifts"])
        results["temp_roles"] = investigate_temp_roles(dfs["staffing_shifts"])
        
    results["missing_staffing_coverage"] = investigate_missing_staffing_coverage(dfs)
    
    return results


def print_investigation_report(report: Dict[str, Any]):
    """Prints a clear human-readable investigation report."""
    print("=" * 60)
    print("AUDIT INVESTIGATION REPORT")
    print("=" * 60)
    
    dup_tx = report.get("duplicate_transactions", {})
    if dup_tx:
        print("\n1. DUPLICATE TRANSACTIONS")
        print("-" * 60)
        print(f"Total duplicate transaction IDs: {dup_tx.get('total_duplicate_ids', 0)}")
        print(f"Total rows involved: {dup_tx.get('total_rows_involved', 0)}")
        print(f"Exact duplicate groups: {dup_tx.get('exact_duplicate_groups_count', 0)}")
        print(f"Conflicting duplicate groups: {dup_tx.get('conflicting_duplicate_groups_count', 0)}")
        
        if dup_tx.get("exact_examples"):
            print("\n  [Example: Exact Duplicate Group]")
            for ex in dup_tx["exact_examples"]:
                print(f"  Transaction ID: {ex['transaction_id']} (Occurrences: {ex['occurrences']})")
                
        if dup_tx.get("conflicting_examples"):
            print("\n  [Example: Conflicting Duplicate Group]")
            for ex in dup_tx["conflicting_examples"]:
                print(f"  Transaction ID: {ex['transaction_id']} (Occurrences: {ex['occurrences']})")
                for i, row in enumerate(ex['rows'], 1):
                    print(f"    Row {i}: {row}")
                    
    s09_tx = report.get("s09_transactions", {})
    if s09_tx:
        print("\n2. INVALID S09 TRANSACTION REFERENCES")
        print("-" * 60)
        print(f"Total affected rows: {s09_tx.get('count', 0)}")
        if s09_tx.get("count", 0) > 0:
            print("\n  Sample of 3 complete rows:")
            for row in s09_tx.get("complete_rows", [])[:3]:
                print(f"  {row}")
                
    s09_staff = report.get("s09_staffing", {})
    if s09_staff:
        print("\n3. INVALID S09 STAFFING REFERENCES")
        print("-" * 60)
        print(f"Total affected rows: {s09_staff.get('count', 0)}")
        if s09_staff.get("count", 0) > 0:
            print("\n  Sample of 3 complete rows:")
            for row in s09_staff.get("complete_rows", [])[:3]:
                print(f"  {row}")
                
    missing_cov = report.get("missing_staffing_coverage", {})
    if missing_cov and missing_cov.get("missing_combinations"):
        print("\n4. MISSING STAFFING STORE/WEEK COVERAGE")
        print("-" * 60)
        for combo in missing_cov["missing_combinations"]:
            print(f"Store: {combo['store_id']}, Missing Week: {combo['missing_week']}")
            print(f"Surrounding Coverage (hours): {combo['surrounding_weeks_coverage_hours']}")
            
    temp_roles = report.get("temp_roles", {})
    if temp_roles:
        print("\n5. TEMP ROLE STAFFING")
        print("-" * 60)
        print(f"Total 'temp' role rows: {temp_roles.get('count', 0)}")
        if temp_roles.get("count", 0) > 0:
            print("\n  Sample of 3 complete rows:")
            for row in temp_roles.get("complete_rows", [])[:3]:
                print(f"  {row}")
                
    print("\n" + "=" * 60)


if __name__ == "__main__":
    investigation_results = run_investigation()
    print_investigation_report(investigation_results)
