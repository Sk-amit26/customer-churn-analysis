import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

sns.set_theme(style="whitegrid", palette="deep")

def run_eda_and_statistics(data_path="data/processed/cleaned_churn.csv",
                           output_dir="tableau/visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading cleaned dataset from {data_path}...")
    df = pd.read_csv(data_path)

    print("\n" + "="*50)
    print("STAGE 4: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*50)

    total_customers = len(df)
    churned = (df['Churn'] == 'Yes').sum()
    retained = (df['Churn'] == 'No').sum()
    churn_rate = (churned / total_customers) * 100

    print(f"Total Customers: {total_customers:,}")
    print(f"Churned Customers: {churned:,}")
    print(f"Retained Customers: {retained:,}")
    print(f"Overall Churn Rate: {churn_rate:.2f}%\n")

    # 1. Overall Churn Distribution Chart
    plt.figure(figsize=(6, 5))
    ax = sns.countplot(data=df, x='Churn', palette=['#2b5c8f', '#d95f02'])
    plt.title('Customer Churn Distribution', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Churn Status', fontsize=12)
    plt.ylabel('Number of Customers', fontsize=12)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:,}\n({height/total_customers*100:.1f}%)',
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', fontsize=11, color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_churn_distribution.png", dpi=300)
    plt.close()

    # 2. Churn by Contract & Internet Service
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    contract_churn = df.groupby('Contract')['Churn_Numeric'].mean().reset_index()
    sns.barplot(data=contract_churn, x='Contract', y='Churn_Numeric', ax=axes[0], palette='Blues_r')
    axes[0].set_title('Churn Rate by Contract Type', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Churn Rate', fontsize=11)
    axes[0].set_ylim(0, 0.5)
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height()*100:.1f}%', (p.get_x() + p.get_width()/2., p.get_height() + 0.01),
                         ha='center', fontsize=10, fontweight='bold')

    internet_churn = df.groupby('InternetService')['Churn_Numeric'].mean().reset_index()
    sns.barplot(data=internet_churn, x='InternetService', y='Churn_Numeric', ax=axes[1], palette='Oranges_r')
    axes[1].set_title('Churn Rate by Internet Service', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Churn Rate', fontsize=11)
    axes[1].set_ylim(0, 0.5)
    for p in axes[1].patches:
        axes[1].annotate(f'{p.get_height()*100:.1f}%', (p.get_x() + p.get_width()/2., p.get_height() + 0.01),
                         ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_contract_and_internet_churn.png", dpi=300)
    plt.close()

    # 3. Numerical Feature Distributions by Churn (Tenure & Monthly Charges)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(data=df, x='tenure', hue='Churn', kde=True, ax=axes[0], palette=['#2b5c8f', '#d95f02'], bins=30)
    axes[0].set_title('Tenure Distribution by Churn Status', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Tenure (Months)', fontsize=11)

    sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, ax=axes[1], palette=['#2b5c8f', '#d95f02'], common_norm=False)
    axes[1].set_title('Monthly Charges Density by Churn Status', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Monthly Charges ($)', fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_numerical_distributions_by_churn.png", dpi=300)
    plt.close()

    print("\n" + "="*50)
    print("STAGE 5: STATISTICAL HYPOTHESIS TESTING")
    print("="*50)

    # Test 1: Contract vs Churn
    contingency_contract = pd.crosstab(df['Contract'], df['Churn'])
    chi2_val, p_val, dof, _ = stats.chi2_contingency(contingency_contract)
    print(f"[TEST 1] Chi-Square: Contract vs Churn -> Chi2: {chi2_val:.4f}, df: {dof}, p-value: {p_val:.4e}")

    # Test 2: PaymentMethod vs Churn
    contingency_pay = pd.crosstab(df['PaymentMethod'], df['Churn'])
    chi2_val2, p_val2, dof2, _ = stats.chi2_contingency(contingency_pay)
    print(f"[TEST 2] Chi-Square: Payment Method vs Churn -> Chi2: {chi2_val2:.4f}, df: {dof2}, p-value: {p_val2:.4e}")

    # Test 3: Monthly Charges t-test
    churned_charges = df[df['Churn'] == 'Yes']['MonthlyCharges']
    retained_charges = df[df['Churn'] == 'No']['MonthlyCharges']
    t_stat, p_val_t = stats.ttest_ind(churned_charges, retained_charges, equal_var=False)
    print(f"[TEST 3] Welch t-test: Monthly Charges -> Churned Mean: ${churned_charges.mean():.2f}, Retained Mean: ${retained_charges.mean():.2f}, t-stat: {t_stat:.4f}, p-value: {p_val_t:.4e}")

    # Test 4: Tenure t-test
    churned_tenure = df[df['Churn'] == 'Yes']['tenure']
    retained_tenure = df[df['Churn'] == 'No']['tenure']
    t_stat_t, p_val_ten = stats.ttest_ind(churned_tenure, retained_tenure, equal_var=False)
    print(f"[TEST 4] Welch t-test: Tenure -> Churned Mean: {churned_tenure.mean():.2f} mos, Retained Mean: {retained_tenure.mean():.2f} mos, t-stat: {t_stat_t:.4f}, p-value: {p_val_ten:.4e}")

    print(f"\nAll charts saved to {output_dir}/")

if __name__ == "__main__":
    run_eda_and_statistics()
