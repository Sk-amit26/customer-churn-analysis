import pandas as pd
import numpy as np
import os

def clean_data(raw_path="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv", 
               output_path="data/processed/cleaned_churn.csv"):
    print(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)
    initial_shape = df.shape
    print(f"Initial Shape: {initial_shape[0]} rows, {initial_shape[1]} columns")

    # 1. TotalCharges conversion: blank spaces to NaN then float
    blank_mask = df['TotalCharges'].astype(str).str.strip() == ''
    num_blanks = blank_mask.sum()
    print(f"Found {num_blanks} blank values in TotalCharges. Converting to float and imputing...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

    # 2. SeniorCitizen type conversion for readability
    df['SeniorCitizen_Label'] = df['SeniorCitizen'].map({1: 'Yes', 0: 'No'})

    # 3. Check and drop duplicate records if any
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()

    # 4. Standardize text categories
    obj_cols = df.select_dtypes(include='object').columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    # 5. Create derived analytical features
    def get_tenure_cohort(tenure):
        if tenure <= 12:
            return '0-12 Months'
        elif tenure <= 24:
            return '13-24 Months'
        elif tenure <= 48:
            return '25-48 Months'
        elif tenure <= 60:
            return '49-60 Months'
        else:
            return '60+ Months'

    df['Tenure_Group'] = df['tenure'].apply(get_tenure_cohort)
    df['Avg_Monthly_Paid'] = np.where(df['tenure'] > 0, df['TotalCharges'] / df['tenure'], df['MonthlyCharges'])
    df['Churn_Numeric'] = df['Churn'].map({'Yes': 1, 'No': 0})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data successfully saved to: {output_path}")
    print(f"Final Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

if __name__ == "__main__":
    clean_data()
