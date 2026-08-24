"""
Complete pure Python data inspection, cleaning, and statistical calculations
Uses standard library (csv, math, collections, json) to guarantee immediate execution.
"""
import csv
import math
from collections import Counter, defaultdict

raw_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
proc_path = "data/processed/cleaned_churn.csv"

with open(raw_path, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

total_rows = len(rows)
headers = list(rows[0].keys())

print(f"=== DATASET INSPECTION ===")
print(f"Total Rows: {total_rows}")
print(f"Total Columns: {len(headers)}")
print(f"Columns: {', '.join(headers)}")

# Check duplicates and missing values
customer_ids = [r['customerID'] for r in rows]
print(f"Unique customerIDs: {len(set(customer_ids))}")
print(f"Duplicate customerIDs: {total_rows - len(set(customer_ids))}")

# Check blank TotalCharges
blank_tc_count = sum(1 for r in rows if r['TotalCharges'].strip() == '')
print(f"Blank TotalCharges count: {blank_tc_count}")

# Check Target Distribution
churn_counts = Counter(r['Churn'] for r in rows)
churn_rate = (churn_counts['Yes'] / total_rows) * 100
print(f"\n=== TARGET VARIABLE (Churn) ===")
print(f"Retained (No): {churn_counts['No']} ({churn_counts['No']/total_rows*100:.2f}%)")
print(f"Churned (Yes): {churn_counts['Yes']} ({churn_counts['Yes']/total_rows*100:.2f}%)")
print(f"Overall Churn Rate: {churn_rate:.2f}%")

# Clean rows and add derived columns
cleaned_rows = []
for r in rows:
    c = dict(r)
    # TotalCharges cleaning
    tc_str = c['TotalCharges'].strip()
    if tc_str == '':
        tc_val = 0.0
    else:
        tc_val = float(tc_str)
    c['TotalCharges'] = f"{tc_val:.2f}"
    
    # Numeric types
    tenure_val = int(c['tenure'])
    mc_val = float(c['MonthlyCharges'])
    
    # Derived features
    if tenure_val <= 12:
        c['Tenure_Group'] = '0-12 Months'
    elif tenure_val <= 24:
        c['Tenure_Group'] = '13-24 Months'
    elif tenure_val <= 48:
        c['Tenure_Group'] = '25-48 Months'
    elif tenure_val <= 60:
        c['Tenure_Group'] = '49-60 Months'
    else:
        c['Tenure_Group'] = '60+ Months'
        
    c['SeniorCitizen_Label'] = 'Yes' if c['SeniorCitizen'] == '1' else 'No'
    c['Churn_Numeric'] = '1' if c['Churn'] == 'Yes' else '0'
    c['Avg_Monthly_Paid'] = f"{(tc_val / tenure_val):.2f}" if tenure_val > 0 else f"{mc_val:.2f}"
    
    cleaned_rows.append(c)

# Save cleaned CSV
out_headers = list(cleaned_rows[0].keys())
with open(proc_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=out_headers)
    writer.writeheader()
    writer.writerows(cleaned_rows)

print(f"\nCleaned dataset written to: {proc_path}")
print(f"Processed columns: {len(out_headers)}")

# Calculate key EDA metrics
print("\n=== EDA SUMMARY METRICS ===")
for col in ['Contract', 'InternetService', 'PaymentMethod', 'SeniorCitizen_Label', 'Tenure_Group']:
    col_churn = defaultdict(lambda: {'total': 0, 'churn': 0})
    for r in cleaned_rows:
        val = r[col]
        col_churn[val]['total'] += 1
        if r['Churn'] == 'Yes':
            col_churn[val]['churn'] += 1
    print(f"\nChurn Breakdown by {col}:")
    for val, stats_dict in sorted(col_churn.items(), key=lambda x: x[1]['churn']/x[1]['total'], reverse=True):
        rate = (stats_dict['churn'] / stats_dict['total']) * 100
        print(f"  {val:25s}: Total={stats_dict['total']:4d}, Churned={stats_dict['churn']:4d}, Churn Rate={rate:5.1f}%")
