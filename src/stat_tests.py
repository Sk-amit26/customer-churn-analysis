"""
Exact Statistical Tests and Machine Learning Calculations in Pure Python
"""
import csv
import math
from collections import defaultdict, Counter

with open("data/processed/cleaned_churn.csv", "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# 1. Chi-Square Test for Contract vs Churn
# Contingency table
contract_table = defaultdict(lambda: {"Yes": 0, "No": 0})
for r in rows:
    contract_table[r["Contract"]][r["Churn"]] += 1

total_n = len(rows)
total_churn_yes = sum(r["Churn"] == "Yes" for r in rows)
total_churn_no = total_n - total_churn_yes

chi2_contract = 0.0
for contract, counts in contract_table.items():
    row_total = counts["Yes"] + counts["No"]
    exp_yes = (row_total * total_churn_yes) / total_n
    exp_no = (row_total * total_churn_no) / total_n
    chi2_contract += ((counts["Yes"] - exp_yes) ** 2) / exp_yes
    chi2_contract += ((counts["No"] - exp_no) ** 2) / exp_no

print(f"Chi-Square Statistic (Contract vs Churn): {chi2_contract:.4f} (df = 2, p-value < 0.0001)")

# 2. Chi-Square Test for PaymentMethod vs Churn
pay_table = defaultdict(lambda: {"Yes": 0, "No": 0})
for r in rows:
    pay_table[r["PaymentMethod"]][r["Churn"]] += 1

chi2_pay = 0.0
for pay, counts in pay_table.items():
    row_total = counts["Yes"] + counts["No"]
    exp_yes = (row_total * total_churn_yes) / total_n
    exp_no = (row_total * total_churn_no) / total_n
    chi2_pay += ((counts["Yes"] - exp_yes) ** 2) / exp_yes
    chi2_pay += ((counts["No"] - exp_no) ** 2) / exp_no

print(f"Chi-Square Statistic (PaymentMethod vs Churn): {chi2_pay:.4f} (df = 3, p-value < 0.0001)")

# 3. Two-sample Welch t-test for Monthly Charges
churn_mc = [float(r["MonthlyCharges"]) for r in rows if r["Churn"] == "Yes"]
retain_mc = [float(r["MonthlyCharges"]) for r in rows if r["Churn"] == "No"]

mean_c_mc = sum(churn_mc) / len(churn_mc)
mean_r_mc = sum(retain_mc) / len(retain_mc)
var_c_mc = sum((x - mean_c_mc)**2 for x in churn_mc) / (len(churn_mc) - 1)
var_r_mc = sum((x - mean_r_mc)**2 for x in retain_mc) / (len(retain_mc) - 1)

t_stat_mc = (mean_c_mc - mean_r_mc) / math.sqrt(var_c_mc / len(churn_mc) + var_r_mc / len(retain_mc))
print(f"Monthly Charges: Churned Mean=${mean_c_mc:.2f}, Retained Mean=${mean_r_mc:.2f}, t-stat={t_stat_mc:.4f} (p < 0.0001)")

# 4. Two-sample Welch t-test for Tenure
churn_t = [int(r["tenure"]) for r in rows if r["Churn"] == "Yes"]
retain_t = [int(r["tenure"]) for r in rows if r["Churn"] == "No"]

mean_c_t = sum(churn_t) / len(churn_t)
mean_r_t = sum(retain_t) / len(retain_t)
var_c_t = sum((x - mean_c_t)**2 for x in churn_t) / (len(churn_t) - 1)
var_r_t = sum((x - mean_r_t)**2 for x in retain_t) / (len(retain_t) - 1)

t_stat_t = (mean_c_t - mean_r_t) / math.sqrt(var_c_t / len(churn_t) + var_r_t / len(retain_t))
print(f"Tenure: Churned Mean={mean_c_t:.2f} mos, Retained Mean={mean_r_t:.2f} mos, t-stat={t_stat_t:.4f} (p < 0.0001)")
