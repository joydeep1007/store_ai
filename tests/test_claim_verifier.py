import pytest
import pandas as pd
import numpy as np
from src.claim_verifier import verify_claim

@pytest.fixture
def sample_df():
    data = {
        'store_id': ['S01', 'S01', 'S02', 'S02', 'S03', 'S03'],
        'week_start': [
            '2025-05-05', '2025-05-12', 
            '2025-05-05', '2025-05-12', 
            '2025-05-05', '2025-05-12'
        ],
        'weekly_revenue': [10000.0, 11000.0, 15000.0, 12000.0, 0.0, 5000.0],
        'transaction_count': [100, 110, 150, 120, 0, 50],
        'staffing_hours': [300.0, 310.0, 350.0, np.nan, 200.0, 210.0],
        'sales_per_staffed_hour': [33.33, 35.48, 42.86, np.nan, 0.0, 23.81]
    }
    df = pd.DataFrame(data)
    df['week_start'] = pd.to_datetime(df['week_start'])
    return df

def test_correct_number(sample_df):
    claim = {
        "store_id": "S01",
        "week_start": "2025-05-05",
        "metric": "weekly_revenue",
        "claim_type": "value",
        "value": 10000.0
    }
    res = verify_claim(claim, sample_df)
    assert res['status'] == "PASS"

def test_incorrect_number(sample_df):
    claim = {
        "store_id": "S01",
        "week_start": "2025-05-05",
        "metric": "weekly_revenue",
        "claim_type": "value",
        "value": 15000.0
    }
    res = verify_claim(claim, sample_df)
    assert res['status'] == "FAIL"
    assert res['reason'] == "Unsupported numerical claim."

def test_correct_percentage(sample_df):
    # S01 revenue went from 10000 to 11000. That is +10%
    claim = {
        "store_id": "S01",
        "week_start": "2025-05-12",
        "metric": "weekly_revenue",
        "claim_type": "percentage",
        "value": 10.0
    }
    res = verify_claim(claim, sample_df)
    assert res['status'] == "PASS"
    assert res['expected_value'] == 10.0

def test_incorrect_percentage(sample_df):
    # S01 revenue went from 10000 to 11000. That is +10%
    claim = {
        "store_id": "S01",
        "week_start": "2025-05-12",
        "metric": "weekly_revenue",
        "claim_type": "percentage",
        "value": 20.0
    }
    res = verify_claim(claim, sample_df)
    assert res['status'] == "FAIL"

def test_wrong_store_or_week(sample_df):
    # S02 had 15000 in week 2025-05-05, not S01
    claim = {
        "store_id": "S01",
        "week_start": "2025-05-05",
        "metric": "weekly_revenue",
        "claim_type": "value",
        "value": 15000.0
    }
    res = verify_claim(claim, sample_df)
    assert res['status'] == "FAIL"

def test_missing_metric(sample_df):
    claim = {
        "store_id": "S01",
        "week_start": "2025-05-05",
        "metric": "invented_metric",
        "claim_type": "value",
        "value": 100
    }
    res = verify_claim(claim, sample_df)
    assert res['status'] == "FAIL"
    assert "not allowed" in res['reason']

def test_missing_store(sample_df):
    claim = {
        "store_id": "S99",
        "week_start": "2025-05-05",
        "metric": "weekly_revenue",
        "claim_type": "value",
        "value": 100
    }
    res = verify_claim(claim, sample_df)
    assert res['status'] == "FAIL"
    assert "Store S99 not found" in res['reason']

def test_missing_week(sample_df):
    claim = {
        "store_id": "S01",
        "week_start": "2025-01-01",
        "metric": "weekly_revenue",
        "claim_type": "value",
        "value": 100
    }
    res = verify_claim(claim, sample_df)
    assert res['status'] == "FAIL"
    assert "Week 2025-01-01 not found" in res['reason']

def test_missing_staffing_hours(sample_df):
    # S02 has NaN staffing_hours on 2025-05-12
    claim = {
        "store_id": "S02",
        "week_start": "2025-05-12",
        "metric": "staffing_hours",
        "claim_type": "value",
        "value": 0
    }
    res = verify_claim(claim, sample_df)
    assert res['status'] == "UNSUPPORTED"
    assert "missing" in res['reason']

def test_rounding_tolerance(sample_df):
    # 33.33 vs 33.333
    claim = {
        "store_id": "S01",
        "week_start": "2025-05-05",
        "metric": "sales_per_staffed_hour",
        "claim_type": "value",
        "value": 33.331
    }
    res = verify_claim(claim, sample_df)
    assert res['status'] == "PASS"

def test_negative_revenue_growth(sample_df):
    # S02 went from 15000 to 12000 => -20%
    claim = {
        "store_id": "S02",
        "week_start": "2025-05-12",
        "metric": "weekly_revenue",
        "claim_type": "percentage",
        "value": -20.0
    }
    res = verify_claim(claim, sample_df)
    assert res['status'] == "PASS"

def test_previous_revenue_zero_or_missing(sample_df):
    # S03 went from 0 (2025-05-05) to 5000 (2025-05-12)
    claim = {
        "store_id": "S03",
        "week_start": "2025-05-12",
        "metric": "weekly_revenue",
        "claim_type": "percentage",
        "value": 1000.0
    }
    res = verify_claim(claim, sample_df)
    assert res['status'] == "FAIL"
    assert "missing or zero" in res['reason']
