# Telecom Customer Churn Analysis & Machine Learning Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.3.3-darkblue.svg)](https://pandas.pydata.org/)
[![MySQL](https://img.shields.io/badge/mysql-8.0-blue.svg)](https://www.mysql.com/)
[![Tableau](https://img.shields.io/badge/tableau-Dashboard-green.svg)](https://www.tableau.com/)

An end-to-end data analytics and machine learning portfolio project identifying churn drivers, quantifying revenue risk, performing rigorous statistical hypothesis testing, executing MySQL business queries, and deploying classification models (Logistic Regression & Random Forest) to predict customer churn in real time.

---

## 📌 Project Overview
Customer attrition is one of the most critical challenges facing the telecommunications sector. Acquiring a new customer costs 5–7x more than retaining an existing one. This project provides a full-stack analytical solution:
1. **Data Analytics & ETL:** Automated cleaning, imputation of blank charges, and categorical feature restructuring across 7,043 customer accounts.
2. **Exploratory & Statistical Analysis:** Quantification of churn patterns and hypothesis validation using Chi-Square tests of independence and Welch's two-sample $t$-tests.
3. **Enterprise SQL Warehouse:** Production MySQL schema with 10 analytical business queries targeting revenue loss and customer cohorts.
4. **Machine Learning Pipeline:** Modular Scikit-Learn pipelines evaluating balanced Logistic Regression and Random Forest classifiers ($0.843$ ROC-AUC, $78\%$ recall).
5. **Interactive Dashboard & Inference:** Tableau workbook specs and automated inference engine delivering instant customer churn risk scoring.

---

## 🛠️ Tech Stack
- **Core Analytics & Data Engineering:** Python, Pandas, NumPy
- **Statistical Testing:** SciPy (`scipy.stats.chi2_contingency`, `scipy.stats.ttest_ind`)
- **Machine Learning & Preprocessing:** Scikit-Learn (`ColumnTransformer`, `OneHotEncoder`, `StandardScaler`, `LogisticRegression`, `RandomForestClassifier`)
- **Database & Querying:** MySQL 8.0+, SQL CTEs, Window Aggregations
- **Data Visualization & BI:** Tableau, Matplotlib, Seaborn
- **Model Serialization:** Joblib

---

## 📁 Repository Structure
```text
customer-churn-analysis/
│
├── data/
│   ├── raw/
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # 7,043 raw records (IBM Telco Churn)
│   └── processed/
│       └── cleaned_churn.csv                      # Imputed, cleaned & feature-engineered dataset
│
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb                 # Interactive cleaning, distributions & stats
│   └── 02_churn_prediction.ipynb                  # ML pipelines, cross-validation & evaluation
│
├── sql/
│   └── churn_analysis.sql                         # MySQL table DDL + 10 analytical business queries
│
├── tableau/
│   ├── visualizations/                            # Exported high-res charts (ROC, distributions, etc.)
│   └── tableau_guide.md                           # Tableau calculated fields and dashboard layout
│
├── src/
│   ├── data_cleaning.py                           # Automated cleaning & ETL module
│   ├── eda_statistics.py                          # Statistical testing & chart generator
│   └── train_predict.py                           # Model training, evaluation & inference engine
│
├── models/
│   └── churn_pipeline.joblib                      # Serialized Random Forest Scikit-Learn pipeline
│
├── docs/
│   ├── 01_project_setup.md                        # Environment setup documentation
│   └── FINAL_PROJECT_DOCUMENTATION.md             # 23-section master project report
│
├── requirements.txt                               # Project dependency versions
├── .gitignore
└── README.md
```

---

## 📊 Key Findings & Business Insights
1. **Contract Type is the Dominant Factor ($\chi^2 = 1184.60, p < 10^{-250}$):** 
   - **Month-to-month** customers have a **42.7% churn rate** compared to **11.3%** for 1-year and **2.8%** for 2-year contracts.
2. **First-Year Vulnerability Window:**
   - **47.4% of all churn occurs in months 0–12** (1,037 customers lost). Churn drops to **6.6%** after 60 months.
3. **Fiber Optic & Electronic Check Attrition:**
   - Fiber Optic users churn at **41.9%** (vs DSL at 19.0%), and Electronic Check payers churn at **45.3%** (vs automated methods at ~15.5%).
4. **Higher Monthly Charges Associated with Churn ($t = 18.41, p < 10^{-70}$):**
   - Churned customers pay a mean of **$74.44/month** vs **$61.27/month** for retained customers (+\$13.17 difference).
5. **Add-on Services Drive Retention:**
   - Customers with *Tech Support* and *Online Security* exhibit churn rates under **16%**, while customers without them exceed **41%**.

---

## 🤖 Machine Learning Model Evaluation
A stratified 80/20 train/test split was used (Training: 5,634 samples, Testing: 1,409 samples) with balanced class weights to prioritize customer recall (identifying churners before they leave).

| Model | Accuracy | Precision | Recall (Churn) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | 73.81% | 50.43% | **78.34%** | 0.6136 | 0.8417 |
| **Random Forest Classifier (Balanced)** | **75.44%** | **52.54%** | 77.54% | **0.6263** | **0.8432** |

*Selected Production Model:* **Random Forest Classifier** achieving **0.8432 ROC-AUC** and **77.54% Recall**.

---

## 🚀 How to Run the Project

### 1. Environment Setup
```bash
git clone https://github.com/your-username/customer-churn-analysis.git
cd customer-churn-analysis
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Data Cleaning & ETL
```bash
python src/data_cleaning.py
```

### 3. Run Statistical Hypothesis Tests & Visualizations
```bash
python src/eda_statistics.py
```

### 4. Train Models & Run Sample Inference
```bash
python src/train_predict.py
```

### 5. Execute MySQL Warehouse Queries
Import `data/processed/cleaned_churn.csv` into MySQL and run:
```bash
mysql -u root -p < sql/churn_analysis.sql
```

---

## 💡 Business Recommendations
1. **Incentivize Annual Contract Migration:** Offer a 10% discount on 1-year contracts for month-to-month users at month 3 to bypass the 47.4% first-year churn cliff.
2. **Onboarding & Tech Support Bundling:** Bundle free 90-day *Tech Support* and *Online Security* for new Fiber Optic customers to reduce high early attrition.
3. **Automated Payment Incentive:** Provide a one-time $10 credit to migrate customers paying via Electronic Check (45.3% churn) to automated bank transfer or credit card (15.5% churn).
