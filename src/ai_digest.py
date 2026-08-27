import argparse
import os
import sys
import pandas as pd
from datetime import timedelta
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

def load_and_prepare_data(csv_path: str = "outputs/weekly_metrics.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")
    
    df = pd.read_csv(csv_path)
    df['week_start'] = pd.to_datetime(df['week_start'])
    
    # Restrict development/testing data to weeks before 2025-05-26
    df = df[df['week_start'] < pd.to_datetime("2025-05-26")]
    
    return df

def generate_weekly_digest(week_start: str):
    df = load_and_prepare_data()
    
    try:
        target_week = pd.to_datetime(week_start)
    except Exception as e:
        raise ValueError(f"Invalid week format: {week_start}. Use YYYY-MM-DD.") from e
        
    if target_week >= pd.to_datetime("2025-05-26"):
        raise ValueError(f"Week {week_start} is excluded for development.")
        
    current_week_df = df[df['week_start'] == target_week]
    if current_week_df.empty:
        raise ValueError(f"No data found for week starting {week_start}")
        
    prev_week = target_week - timedelta(days=7)
    prev_week_df = df[df['week_start'] == prev_week]
    
    # Prepare structured context
    context_lines = []
    context_lines.append(f"Target Week: {week_start}")
    context_lines.append("Store Metrics:")
    
    for _, row in current_week_df.iterrows():
        store_id = row['store_id']
        revenue = row['weekly_revenue']
        transactions = row['transaction_count']
        return_rate = row['return_rate']
        staffing = row['staffing_hours']
        sales_per_hour = row['sales_per_staffed_hour']
        
        # Calculate WoW revenue change
        wow_revenue = "N/A"
        if not prev_week_df.empty:
            prev_row = prev_week_df[prev_week_df['store_id'] == store_id]
            if not prev_row.empty:
                prev_rev = prev_row.iloc[0]['weekly_revenue']
                if prev_rev > 0:
                    wow_revenue = f"{((revenue - prev_rev) / prev_rev) * 100:.2f}%"
        
        staffing_str = f"{staffing:.1f}" if pd.notna(staffing) else "MISSING"
        sales_per_hour_str = f"${sales_per_hour:.2f}" if pd.notna(sales_per_hour) else "MISSING"
        
        context_lines.append(f"- Store: {store_id}")
        context_lines.append(f"  - Weekly Revenue: ${revenue:,.2f}")
        context_lines.append(f"  - WoW Revenue Change: {wow_revenue}")
        context_lines.append(f"  - Transaction Count: {transactions}")
        context_lines.append(f"  - Return Rate: {return_rate:.2%}")
        context_lines.append(f"  - Staffing Hours: {staffing_str}")
        context_lines.append(f"  - Sales per Staffed Hour: {sales_per_hour_str}")

    context_str = "\n".join(context_lines)
    
    # Print the selected week and the structured context
    print(f"--- SELECTED WEEK ---\n{week_start}\n")
    print(f"--- STRUCTURED METRICS / CONTEXT SENT TO LLM ---\n{context_str}\n")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is missing.", file=sys.stderr)
        sys.exit(1)
        
    client = genai.Client(api_key=api_key)
    
    system_prompt = """You are a Regional Multi-Store Lead.
Generate a concise WEEKLY OPERATIONS DIGEST based ONLY on the provided metrics.

Grounding rules:
- Use ONLY the supplied metrics.
- Never invent numerical values.
- Never invent promotions, operational events, staffing changes, customer behavior, causes, or explanations.
- Do not treat missing values as zero.
- Do not independently calculate percentages (use what is provided).
- Clearly distinguish observations from hypotheses.
- If a cause cannot be established from the metrics, explicitly say that further investigation is required.
- Prioritize meaningful store movements instead of repeating every row.

Format requirements:
WEEKLY OPERATIONS DIGEST
Week: {week_start}

1. Executive Summary
2. Key Store Movements
3. Areas Requiring Attention
4. Recommended Investigation
"""
    system_prompt = system_prompt.replace("{week_start}", week_start)
    
    try:
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=context_str,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
            )
        )
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AI Weekly Digest")
    parser.add_argument("--week", default="2025-05-05", help="Week start date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    try:
        digest = generate_weekly_digest(args.week)
        print("--- GENERATED DIGEST ---\n")
        print(digest)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
