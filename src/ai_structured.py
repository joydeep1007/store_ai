import argparse
import os
import sys
import json
import pandas as pd
from datetime import timedelta
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Literal, Union

from google import genai
from google.genai import types

try:
    # Used when running as a module or during pytest
    from src.claim_verifier import verify_claim
except ModuleNotFoundError:
    # Used when running script directly via `python src/ai_structured.py`
    from claim_verifier import verify_claim

load_dotenv()

# Define allowed metrics using an Enum (represented here by Literal for Pydantic)
MetricEnum = Literal[
    'weekly_revenue', 
    'transaction_count', 
    'return_amount', 
    'return_count', 
    'return_rate', 
    'staffing_hours', 
    'sales_per_staffed_hour',
    'revenue_growth_pct'
]

ClaimTypeEnum = Literal['value', 'percentage', 'ranking']

from pydantic import BaseModel, Field, model_validator, ValidationError

class Claim(BaseModel):
    store_id: str
    week_start: str
    metric: MetricEnum
    claim_type: ClaimTypeEnum
    ranking: str | None = Field(default=None, description="Only for ranking claims. Must be exactly 'highest' or 'lowest'.")
    value: float = Field(description="The numerical value or percentage associated with the claim.")
    unit: str | None = Field(default=None, description="The unit of the value (e.g. 'percent', 'hours', '$').")
    importance: Literal['high', 'medium', 'low']
    text: str

    @model_validator(mode='after')
    def validate_ranking(self):
        if self.claim_type == 'ranking':
            if self.ranking not in ['highest', 'lowest']:
                raise ValueError("Ranking claims must have ranking 'highest' or 'lowest'")
        else:
            if self.ranking is not None:
                raise ValueError("Non-ranking claims must have ranking set to null")
        return self

class WeeklyClaims(BaseModel):
    week_start: str
    claims: list[Claim]


def load_and_prepare_data(csv_path: str = "outputs/weekly_metrics.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df['week_start'] = pd.to_datetime(df['week_start'])
    
    # Restrict development/testing data to weeks before 2025-05-26
    df = df[df['week_start'] < pd.to_datetime("2025-05-26")]
    
    return df

def generate_structured_claims(week_start: str, df: pd.DataFrame, client: genai.Client):
    target_week = pd.to_datetime(week_start)
    current_week_df = df[df['week_start'] == target_week]
    if current_week_df.empty:
        raise ValueError(f"No data found for week starting {week_start}")
        
    prev_week = target_week - timedelta(days=7)
    prev_week_df = df[df['week_start'] == prev_week]
    
    # Prepare context strictly as string
    context_lines = [f"Target Week: {week_start}", "Store Metrics:"]
    for _, row in current_week_df.iterrows():
        store_id = row['store_id']
        rev = row['weekly_revenue']
        growth = row['revenue_growth_pct']
        trans = row['transaction_count']
        ret = row['return_rate']
        staff = row['staffing_hours']
        sph = row['sales_per_staffed_hour']
        
        growth_str = f"{growth:.2f}%" if pd.notna(growth) else "MISSING"
        staff_str = f"{staff:.1f}" if pd.notna(staff) else "MISSING"
        sph_str = f"${sph:.2f}" if pd.notna(sph) else "MISSING"
        
        context_lines.append(f"- Store: {store_id}")
        context_lines.append(f"  - weekly_revenue: ${rev:,.2f}")
        context_lines.append(f"  - revenue_growth_pct: {growth_str}")
        context_lines.append(f"  - transaction_count: {trans}")
        context_lines.append(f"  - return_rate: {ret:.2%}")
        context_lines.append(f"  - staffing_hours: {staff_str}")
        context_lines.append(f"  - sales_per_staffed_hour: {sph_str}")
        
    context_str = "\n".join(context_lines)

    system_prompt = """You are a meticulous retail data analyst.
Based on the provided metrics context, generate a structured JSON list of important factual claims about the stores' performance.

Grounding rules:
1. Use ONLY the supplied trusted metrics.
2. Never invent numerical values.
3. Never invent promotions, causes, staffing events, customer behavior, inventory events, or operational facts.
4. Do not calculate metrics independently (unless extracting exact values from the text).
5. Use Python-provided values exactly.
6. Do not treat missing values as zero.
7. Clearly distinguish observations from hypotheses.
8. If a cause is not supported by the metrics, state that further investigation is required.
9. Claims must identify the correct store, week, and metric.
10. Return structured JSON only.
CLAIM TYPE DEFINITIONS:

A. VALUE CLAIM

Use claim_type='value' when the primary purpose is to report the
absolute value of a metric for a specific store and week.

Example:

Store S01 generated $14,315.48 in weekly revenue.

The structured claim should be:

{
  "metric": "weekly_revenue",
  "claim_type": "value",
  "ranking": null,
  "value": 14315.48
}

A value claim should NOT be converted into a ranking claim simply
because the value happens to be the highest or lowest.

B. RANKING CLAIM

Use claim_type='ranking' only when the business finding is that a
specific store is the highest or lowest among the stores for the
specified metric during the target week.

For a ranking claim:
- ranking MUST be exactly 'highest' or 'lowest'.
- value MUST contain the actual numerical metric value for that store.
- ranking and value represent two separate facts.
- The verifier will independently verify both facts.

Example:

{
  "metric": "weekly_revenue",
  "claim_type": "ranking",
  "ranking": "highest",
  "value": 23516.04
}

C. PERCENTAGE CLAIM

Use claim_type='percentage' when the primary finding is a percentage
or percentage change.

Examples include:
- revenue_growth_pct
- return_rate

For a percentage claim:
- ranking must be null.
- value must contain the actual numerical percentage value.
- Do not use ranking='highest' or ranking='lowest' unless the
  relative ranking itself is the actual finding.

Example:

Store S03 revenue increased 171.20% week over week.

The structured claim should be:

{
  "metric": "revenue_growth_pct",
  "claim_type": "percentage",
  "ranking": null,
  "value": 171.20
}

CLAIM SELECTION:

Generate a diverse set of the most decision-useful factual claims.

Use VALUE claims when an absolute metric value is important.

Use RANKING claims only when the relative position of a store
(highest or lowest) is itself an important business finding.

Use PERCENTAGE claims for meaningful percentage movements such as
revenue_growth_pct and return_rate.

Do not convert every observation into a ranking claim merely because
multiple stores are present.

Choose the claim_type that best represents the actual business finding.

The claim type must describe the nature of the finding, not merely
the existence of a numerical value.

Avoid generating multiple claims that express the same underlying
fact using different claim types.

Do not artificially force a fixed number of value, ranking, or
percentage claims. Select the most meaningful claims from the
provided trusted metrics.

RANKING FORMAT:
If claim_type='ranking':
- ranking MUST be exactly 'highest' or 'lowest'.
- value MUST be a valid float containing the actual metric value.
- unit may be '$', 'percent', 'hours', or another appropriate unit.

MISSING DATA:
- Never substitute zero for missing data.
- Do not create ranking claims involving a metric when the relevant
  value is missing or unsupported.
- `revenue_growth_pct` is a trusted Python-provided metric.
- Do not calculate `revenue_growth_pct` independently.
- Missing growth values (e.g., MISSING) must not be treated as zero.

The verifier will independently check every claim against the trusted
dataset.
"""

    response = client.models.generate_content(
        model='gemini-3.1-flash-lite',
        contents=context_str,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=WeeklyClaims,
        )
    )
    
    try:
        raw_data = json.loads(response.text)
        # Instantiate Pydantic model to enforce strict validation (including the custom model_validator)
        validated_data = WeeklyClaims(**raw_data)
        return validated_data.model_dump()
    except Exception as e:
        print(f"Error decoding or validating JSON from LLM: {e}")
        print("Raw response:", response.text)
        sys.exit(1)


def generate_final_digest(verified_claims: list, data_limitations: list, week_start: str, client: genai.Client):
    
    context_data = {
        "verified_claims": verified_claims,
        "data_limitations": data_limitations
    }
    
    system_prompt = """You are a Regional Multi-Store Lead writing a weekly operations digest.
Generate a concise management digest from the provided verified claims and data limitations.

STRICT GROUNDING RULES:
- You must ONLY use the provided `verified_claims` and `data_limitations`.
- Do NOT introduce new numbers.
- Do NOT introduce new stores.
- Do NOT introduce new dates.
- Do NOT introduce new metrics.
- Do NOT invent causes, promotions, or operational facts.
- Only synthesize the verified facts into the required format.

Format requirements:
WEEKLY OPERATIONS DIGEST
Week: {week_start}

1. Key Observations
2. Important Changes
3. Areas Requiring Attention
4. Recommended Focus
"""
    system_prompt = system_prompt.replace("{week_start}", week_start)
    
    response = client.models.generate_content(
        model='gemini-3.1-flash-lite',
        contents=json.dumps(context_data, indent=2),
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
        )
    )
    
    return response.text

def main():
    parser = argparse.ArgumentParser(description="Generate AI Weekly Digest - Approach 2")
    parser.add_argument("--week", default="2025-05-05", help="Week start date (YYYY-MM-DD)")
    args = parser.parse_args()
    week_start = args.week
    
    print(f"1. Selected week: {week_start}")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is missing.", file=sys.stderr)
        sys.exit(1)
        
    client = genai.Client(api_key=api_key)
    
    df = load_and_prepare_data()
    
    # Generate structured JSON claims
    print("2. Generating structured metrics claims...")
    raw_claims_data = generate_structured_claims(week_start, df, client)
    
    print("3. Raw structured JSON returned by the LLM:")
    print(json.dumps(raw_claims_data, indent=2))
    
    # Verify claims
    print("\n4. Verification result for each claim:")
    verification_results = []
    verified_pass_claims = []
    data_limitations = []
    
    for claim in raw_claims_data.get('claims', []):
        result = verify_claim(claim, df)
        verification_results.append({
            "claim": claim,
            "verification": result
        })
        
        print(f"Claim: {claim.get('text')}")
        print(f"  Status: {result['status']}")
        print(f"  Reason: {result['reason']}\n")
        
        if result['status'] == 'PASS':
            verified_pass_claims.append(claim)
        elif result['status'] == 'UNSUPPORTED':
            data_limitations.append(f"{claim.get('metric')} data limitation for {claim.get('store_id')}: {result['reason']}")
            
    # Also actively look for missing staffing hours to add to data_limitations just in case
    target_week = pd.to_datetime(week_start)
    current_week_df = df[df['week_start'] == target_week]
    for _, row in current_week_df.iterrows():
        if pd.isna(row['staffing_hours']):
            limitation = f"Staffing data is missing for {row['store_id']} during {week_start}."
            if limitation not in data_limitations:
                data_limitations.append(limitation)
                
    print("5. Verified claims (PASS only):")
    print(json.dumps(verified_pass_claims, indent=2))
    
    if data_limitations:
        print("   Data Limitations found:")
        for dl in data_limitations:
            print(f"   - {dl}")
    
    # Generate final digest
    print("\n6. Final digest generating...")
    digest = generate_final_digest(verified_pass_claims, data_limitations, week_start, client)
    
    print("\n--- FINAL DIGEST ---\n")
    print(digest)
    
    # Save outputs
    out_dir = "outputs/ai_approach2"
    os.makedirs(out_dir, exist_ok=True)
    
    with open(f"{out_dir}/claims_{week_start}.json", "w") as f:
        json.dump(raw_claims_data, f, indent=2)
        
    with open(f"{out_dir}/verification_{week_start}.json", "w") as f:
        json.dump(verification_results, f, indent=2)
        
    with open(f"{out_dir}/digest_{week_start}.txt", "w") as f:
        f.write(digest)
        
    print(f"\nArtifacts saved to {out_dir}/")

if __name__ == "__main__":
    main()
