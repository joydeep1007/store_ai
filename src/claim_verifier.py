import pandas as pd
import math
from datetime import timedelta

ALLOWED_METRICS = {
    'weekly_revenue', 
    'transaction_count', 
    'return_amount',
    'return_count',
    'return_rate', 
    'staffing_hours', 
    'sales_per_staffed_hour',
    'revenue_growth_pct'
}

def verify_claim(claim: dict, df: pd.DataFrame) -> dict:
    """
    Verifies a structured JSON claim against a pandas DataFrame.
    """
    store_id = claim.get("store_id")
    week_start = claim.get("week_start")
    metric = claim.get("metric")
    claim_type = claim.get("claim_type")
    claimed_value = claim.get("value")
    
    # Check basics
    if not store_id or not week_start or not metric or not claim_type or claimed_value is None:
        return {
            "status": "FAIL",
            "reason": "Missing required fields in claim.",
            "expected_value": None,
            "claimed_value": claimed_value
        }
        
    if metric not in ALLOWED_METRICS:
        return {
            "status": "FAIL",
            "reason": f"Metric '{metric}' is not allowed.",
            "expected_value": None,
            "claimed_value": claimed_value
        }
        
    # Ensure dataframe has parsed datetime for week_start
    if not pd.api.types.is_datetime64_any_dtype(df['week_start']):
        df = df.copy()
        df['week_start'] = pd.to_datetime(df['week_start'])
        
    try:
        target_week = pd.to_datetime(week_start)
    except Exception:
        return {
            "status": "FAIL",
            "reason": "Invalid week_start format.",
            "expected_value": None,
            "claimed_value": claimed_value
        }

    # Locate exact row
    row = df[(df['store_id'] == store_id) & (df['week_start'] == target_week)]
    
    if row.empty:
        # Is the store missing or week missing?
        if store_id not in df['store_id'].values:
            return {"status": "FAIL", "reason": f"Store {store_id} not found.", "expected_value": None, "claimed_value": claimed_value}
        if target_week not in df['week_start'].values:
            return {"status": "FAIL", "reason": f"Week {week_start} not found.", "expected_value": None, "claimed_value": claimed_value}
        
        return {"status": "FAIL", "reason": "Store + Week combination not found.", "expected_value": None, "claimed_value": claimed_value}
        
    actual_value = row.iloc[0][metric]
    
    # Metric representation documentation:
    # - weekly_revenue: raw numerical value
    # - transaction_count: raw numerical value
    # - return_amount: raw numerical value
    # - return_count: raw numerical value
    # - return_rate: decimal fraction (e.g., 0.0127 for 1.27%)
    # - staffing_hours: raw numerical value
    # - sales_per_staffed_hour: raw numerical value
    
    # Convert decimal fraction to percentage points for return_rate
    if metric == 'return_rate' and pd.notna(actual_value):
        actual_value = actual_value * 100
    # Missing data logic (e.g. staffing hours)
    if pd.isna(actual_value):
        return {
            "status": "UNSUPPORTED",
            "reason": f"Data for {metric} is missing for this store-week."
        }
        
    if claim_type == "value":
        if isinstance(actual_value, (int, float)) and isinstance(claimed_value, (int, float)):
            if metric == 'return_rate':
                # Use absolute tolerance equivalent to 2 decimal places rounding
                if math.isclose(actual_value, claimed_value, abs_tol=0.0051):
                    return {"status": "PASS", "reason": "Claim matches trusted metric (2 decimal place precision).", "expected_value": actual_value, "claimed_value": claimed_value}
                else:
                    return {"status": "FAIL", "reason": "Unsupported numerical claim.", "expected_value": actual_value, "claimed_value": claimed_value}
            else:
                if math.isclose(actual_value, claimed_value, rel_tol=1e-3, abs_tol=1e-3):
                    return {"status": "PASS", "reason": "Claim matches trusted metric.", "expected_value": actual_value, "claimed_value": claimed_value}
                else:
                    return {"status": "FAIL", "reason": "Unsupported numerical claim.", "expected_value": actual_value, "claimed_value": claimed_value}
        else:
            if str(actual_value) == str(claimed_value):
                return {"status": "PASS", "reason": "Claim matches trusted metric.", "expected_value": actual_value, "claimed_value": claimed_value}
            else:
                return {"status": "FAIL", "reason": "Unsupported numerical claim.", "expected_value": actual_value, "claimed_value": claimed_value}
                
    elif claim_type == "percentage":
        if metric == "revenue_growth_pct":
            expected_pct = actual_value
            if isinstance(claimed_value, (int, float)) and math.isclose(expected_pct, claimed_value, rel_tol=1e-2, abs_tol=1e-2):
                return {"status": "PASS", "reason": "Claim matches trusted revenue growth metric.", "expected_value": expected_pct, "claimed_value": claimed_value}
            else:
                return {"status": "FAIL", "reason": "Unsupported percentage claim.", "expected_value": expected_pct, "claimed_value": claimed_value}
                
        # Calculate percentage WoW change for the metric
        prev_week = target_week - timedelta(days=7)
        prev_row = df[(df['store_id'] == store_id) & (df['week_start'] == prev_week)]
        
        if prev_row.empty:
            return {"status": "FAIL", "reason": "Previous week data not found for percentage calculation.", "expected_value": None, "claimed_value": claimed_value}
            
        prev_val = prev_row.iloc[0][metric]
        if metric == 'return_rate' and pd.notna(prev_val):
            prev_val = prev_val * 100
            
        if pd.isna(prev_val) or prev_val == 0:
            return {"status": "FAIL", "reason": "Previous week value is missing or zero.", "expected_value": None, "claimed_value": claimed_value}
            
        expected_pct = ((actual_value - prev_val) / prev_val) * 100
        
        if isinstance(claimed_value, (int, float)) and math.isclose(expected_pct, claimed_value, rel_tol=1e-2, abs_tol=1e-2):
            return {"status": "PASS", "reason": "Percentage claim matches trusted metric.", "expected_value": expected_pct, "claimed_value": claimed_value}
        else:
            return {"status": "FAIL", "reason": "Unsupported percentage claim.", "expected_value": expected_pct, "claimed_value": claimed_value}
            
    elif claim_type == "ranking":
        ranking = claim.get("ranking")
        if ranking not in ["highest", "lowest"]:
            return {"status": "FAIL", "reason": f"Invalid ranking value: {ranking}", "expected_value": None, "claimed_value": ranking}
            
        week_df = df[df['week_start'] == target_week].copy()
        week_df = week_df.dropna(subset=[metric])
        if week_df.empty:
            return {"status": "FAIL", "reason": "No valid data to calculate ranking.", "expected_value": None, "claimed_value": None}
            
        if ranking == "highest":
            expected_store = week_df.loc[week_df[metric].idxmax(), 'store_id']
        else:
            expected_store = week_df.loc[week_df[metric].idxmin(), 'store_id']
            
        if expected_store != store_id:
            return {
                "status": "FAIL", 
                "reason": f"Ranking check failed. {store_id} is not the {ranking}.", 
                "expected_value": {"store": expected_store, "value": None}, 
                "claimed_value": {"store": store_id, "value": claimed_value}
            }
            
        # Ranking passed, now check numerical value
        if not isinstance(claimed_value, (int, float)):
            return {
                "status": "FAIL",
                "reason": "Numerical value check failed. Claimed value is not numeric.",
                "expected_value": {"store": expected_store, "value": actual_value},
                "claimed_value": {"store": store_id, "value": claimed_value}
            }
            
        if metric == 'return_rate':
            is_match = math.isclose(actual_value, claimed_value, abs_tol=0.0051)
        else:
            is_match = math.isclose(actual_value, claimed_value, rel_tol=1e-3, abs_tol=1e-3)
            
        if is_match:
            return {
                "status": "PASS", 
                "reason": f"Ranking claim ({ranking}) and numerical value verified.", 
                "expected_value": {"store": expected_store, "value": actual_value}, 
                "claimed_value": {"store": store_id, "value": claimed_value}
            }
        else:
            return {
                "status": "FAIL", 
                "reason": "Ranking passed but numerical value check failed.", 
                "expected_value": {"store": expected_store, "value": actual_value}, 
                "claimed_value": {"store": store_id, "value": claimed_value}
            }

    return {
        "status": "FAIL",
        "reason": f"Unknown claim_type: {claim_type}",
        "expected_value": None,
        "claimed_value": claimed_value
    }
