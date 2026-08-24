# Customer Churn Analysis & Prediction — Master Project Documentation

> 🌐 **Live Cloud Deployment:** [https://customer-churn-analysis-ih21.onrender.com](https://customer-churn-analysis-ih21.onrender.com)

---

## 1. Project Overview
This project delivers an end-to-end, production-grade **Customer Churn Analysis and Prediction System** built for the telecommunications industry. Using real-world customer usage, contract details, demographic indicators, and billing profiles, this project integrates exploratory data analytics, inferential statistics, relational database warehousing (MySQL), business intelligence (Tableau), and machine learning (Scikit-Learn).

The entire system is designed to identify high-risk customer segments before cancellation occurs, quantify the financial revenue at risk, uncover behavioral churn drivers, and deploy an automated machine learning inference engine for real-time customer retention scoring.

---

## 2. Business Problem
Customer acquisition cost (CAC) in the telecom industry typically ranges from **\$300 to \$600 per subscriber**, which is **5 to 7 times higher** than customer retention costs. When subscribers churn:
- The company suffers immediate monthly recurring revenue (MRR) loss.
- Customer lifetime value (LTV) drops significantly.
- Sunk customer acquisition and hardware onboarding investments are forfeited before break-even tenure is attained.

The business objective is to proactively flag churn-prone accounts, understand the underlying root causes (e.g., pricing thresholds, service pain points, contract rigidity), and equip marketing and customer success teams with targeted retention strategies.

---

## 3. Objectives
1. **Data Integrity & ETL:** Ingest, inspect, and clean the 7,043 customer records, converting data types, resolving whitespace issues in `TotalCharges`, and handling edge cases without data loss.
2. **Exploratory Data Analysis (EDA):** Quantify churn distributions across customer demographics, services subscribed, tenure intervals, and billing methods.
3. **Statistical Hypothesis Testing:** Conduct formal hypothesis tests (Chi-Square tests of independence and Welch’s two-sample $t$-tests) to establish statistical significance ($p < 0.05$).
4. **Relational Database Analytics (MySQL):** Design a relational schema and implement 10 analytical business queries to track KPIs, high-value churners, and cohort revenue risks.
5. **Machine Learning Pipeline:** Build Scikit-Learn preprocessing and modeling pipelines using **Logistic Regression** and **Random Forest**, applying class-balancing strategies to optimize **Recall** and **ROC-AUC**.
6. **Model Interpretation & Inference:** Extract feature importance metrics and provide an interactive prediction interface for customer risk scoring.
7. **Business Intelligence & Dashboards:** Architect a Tableau dashboard framework with custom calculated fields and dynamic filtering.

---

## 4. Dataset
- **Source:** IBM Telco Customer Churn Dataset
- **Total Records:** 7,043 customer accounts
- **Total Attributes:** 21 raw columns (20 features + 1 target variable)
- **Target Variable:** `Churn` (`Yes`: 1,869 / 26.54%, `No`: 5,174 / 73.46%)

### Attribute Breakdown
| Variable Category | Column Names | Data Types |
| :--- | :--- | :--- |
| **Identifiers** | `customerID` | Categorical (String) |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Categorical & Binary |
| **Account / Tenure** | `tenure` (0–72 months) | Numerical (Integer) |
| **Services Subscribed** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Categorical (Multi-class) |
| **Contract & Billing** | `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges` | Categorical & Continuous |
| **Target Variable** | `Churn` (`Yes` / `No`) | Categorical (Binary) |

---

## 5. Technology Stack
- **Language & Runtime:** Python 3.10+
- **Data Manipulation & ETL:** Pandas 2.3+, NumPy 2.2+
- **Data Visualization:** Matplotlib 3.10+, Seaborn 0.13+
- **Inferential Statistics:** SciPy 1.15+ (`scipy.stats`)
- **Machine Learning:** Scikit-Learn 1.7+ (`Pipeline`, `ColumnTransformer`, `OneHotEncoder`, `StandardScaler`, `LogisticRegression`, `RandomForestClassifier`)
- **Model Persistence:** Joblib 1.5+
- **Relational Database:** MySQL 8.0+
- **Business Intelligence:** Tableau Desktop / Public

---

## 6. Project Architecture / Workflow
```text
Raw Dataset (data/raw/)
       │
       ▼
Data Cleaning & ETL (src/data_cleaning.py)
       │──> Processed Cleaned Dataset (data/processed/cleaned_churn.csv)
       │
       ├───────────────────────────────┬───────────────────────────────┐
       ▼                               ▼                               ▼
Statistical & EDA Pipeline      MySQL Warehouse DDL & Queries    Scikit-Learn ML Pipeline
(src/eda_statistics.py)         (sql/churn_analysis.sql)         (src/train_predict.py)
       │                               │                               │
       ▼                               ▼                               ▼
Tableau BI Visualizations       10 Business SQL Reports          Model Training & Evaluation
(tableau/visualizations/)                                       (Logistic Regression vs RF)
                                                                       │
                                                                       ▼
                                                                Model Persistence (.joblib)
                                                                       │
                                                                       ▼
                                                                Inference Scoring Function
```

---

## 7. Data Cleaning
| Problem Identified | Root Cause | Action Taken | Business Justification |
| :--- | :--- | :--- | :--- |
| **Blank Whitespace in `TotalCharges`** | 11 new customers had `tenure = 0`, causing `TotalCharges` to be recorded as `" "` string. | Converted column to numeric using `pd.to_numeric(errors='coerce')` and imputed missing values to `0.0`. | Prevents data corruption while preserving zero-tenure customer accounts. |
| **`SeniorCitizen` Encoded as Integer** | Values stored as `0` and `1`. | Created `SeniorCitizen_Label` (`"Yes"`, `"No"`) for EDA/Tableau while retaining numeric format for ML. | Ensures crystal-clear reporting in business dashboards. |
| **High Cardinality / Redundant Columns** | Unique ID string (`customerID`) has no predictive generalizability. | Excluded `customerID` from model feature matrix. | Prevents model overfitting and memorization. |
| **Derived Temporal Cohorts** | Continuous `tenure` is hard to scan in high-level BI tables. | Created `Tenure_Group` (`0-12m`, `13-24m`, `25-48m`, `49-60m`, `60+m`). | Enables granular cohort drop-off analysis. |

---

## 8. Exploratory Data Analysis (EDA)

### Key Metrics Summary
- **Total Accounts:** 7,043
- **Total Churned:** 1,869 (26.54%)
- **Total Retained:** 5,174 (73.46%)

### Question → Analysis → Result → Interpretation
1. **How does contract type affect customer churn?**
   - *Analysis:* Grouped churn rates across `Month-to-month`, `One year`, and `Two year` contracts.
   - *Result:* Month-to-month churn is **42.7%** (1,655 / 3,875), One year is **11.3%** (166 / 1,473), Two year is **2.8%** (48 / 1,695).
   - *Interpretation:* Month-to-month contracts lack switching barriers; moving customers to annual contracts reduces churn risk by over 73%.

2. **What role does internet service technology play?**
   - *Analysis:* Compared churn across `Fiber optic`, `DSL`, and `No Internet`.
   - *Result:* Fiber optic churn is **41.9%** (1,297 / 3,096), DSL is **19.0%** (459 / 2,421), No internet is **7.4%** (113 / 1,526).
   - *Interpretation:* Fiber optic customers experience high churn despite premium pricing, pointing to customer onboarding friction or perceived value gaps.

3. **Does payment method correlate with retention?**
   - *Analysis:* Compared electronic checks against automated payment options.
   - *Result:* Electronic check churn is **45.3%** (1,071 / 2,365), while credit card and bank transfer churn are **15.2%** and **16.7%**.
   - *Interpretation:* Manual payment methods introduce monthly friction; automated billing drastically improves retention.

4. **When do customers churn during their lifecycle?**
   - *Analysis:* Churn rate across tenure cohorts.
   - *Result:* **47.4%** churn occurs in the first 12 months (1,037 customers). Churn drops to **6.6%** for tenure > 60 months.
   - *Interpretation:* The first 90–365 days represent the critical retention window.

---

## 9. Statistical Analysis

### Test 1: Chi-Square Test of Independence — Contract Type vs. Churn
- **Question:** Is customer churn statistically dependent on contract structure?
- **$H_0$ (Null):** Contract type and Customer Churn are independent.
- **$H_1$ (Alternative):** Contract type and Customer Churn are significantly dependent.
- **Test Statistic:** $\chi^2 = 1184.5966$, $\text{df} = 2$, $p\text{-value} = 5.86 \times 10^{-258}$
- **Significance Level:** $\alpha = 0.05$
- **Result & Interpretation:** Reject $H_0$. There is an overwhelmingly significant statistical association between contract structure and customer attrition.

### Test 2: Chi-Square Test of Independence — Payment Method vs. Churn
- **Question:** Is payment channel choice significantly associated with churn?
- **Test Statistic:** $\chi^2 = 648.1423$, $\text{df} = 3$, $p\text{-value} = 3.68 \times 10^{-140}$
- **Result & Interpretation:** Reject $H_0$. Electronic check users demonstrate statistically distinct, higher churn likelihood.

### Test 3: Welch’s Two-Sample $t$-Test — Monthly Charges (Churned vs. Retained)
- **Question:** Do churned customers pay significantly different monthly charges compared to retained customers?
- **$H_0$:** $\mu_{\text{churned}} = \mu_{\text{retained}}$
- **$H_1$:** $\mu_{\text{churned}} \neq \mu_{\text{retained}}$
- **Statistics:** Churned Mean = **\$74.44** ($\pm \$24.67$), Retained Mean = **\$61.27** ($\pm \$31.09$)
- **Test Statistic:** $t = 18.4075$, $p\text{-value} = 8.59 \times 10^{-73}$
- **Result & Interpretation:** Reject $H_0$. Churned customers pay an average of **\$13.17 more per month**, indicating price sensitivity as a primary churn driver.

### Test 4: Welch’s Two-Sample $t$-Test — Tenure (Churned vs. Retained)
- **Statistics:** Churned Mean = **17.98 months**, Retained Mean = **37.57 months**
- **Test Statistic:** $t = -34.8238$, $p\text{-value} = 1.20 \times 10^{-232}$
- **Result & Interpretation:** Reject $H_0$. Churned customers have significantly shorter tenure (-19.59 months).

---

## 10. MySQL / SQL Analysis
The SQL warehouse script (`sql/churn_analysis.sql`) executes 10 core business queries.

### Key Query Highlights:
```sql
-- Query: Churn & Revenue Risk by Contract Type
SELECT 
    contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN monthly_charges ELSE 0 END), 2) AS monthly_revenue_lost
FROM telco_churn
GROUP BY contract
ORDER BY churn_rate_pct DESC;
```
*Result:* Month-to-month contracts account for **\$122,866.45 of monthly revenue lost**, representing **88.3%** of total monthly revenue attrition.

---

## 11. Feature Engineering
- **Numerical Scaling:** `StandardScaler` applied to `tenure`, `MonthlyCharges`, and `TotalCharges` to standardize zero mean and unit variance.
- **Categorical Encoding:** `OneHotEncoder(drop='first', handle_unknown='ignore')` applied to 16 categorical features to prevent the dummy variable trap (multicollinearity).
- **Leakage Prevention:** All scalers and encoders are fitted strictly inside `ColumnTransformer` on the **training set only** and applied downstream to the test set.
- **Dropped Identifiers:** `customerID` removed to avoid arbitrary memorization.

---

## 12. Train / Test Split
- **Split Ratio:** 80% Training (5,634 rows) / 20% Testing (1,409 rows)
- **Stratification:** `stratify=y` guarantees identical churn prevalence (**26.54%**) across both training and testing partitions.
- **Reproducibility:** Seed fixed via `random_state=42`.

---

## 13. Machine Learning Models
1. **Logistic Regression (Baseline Linear Classifier):**
   - Parameterized with `class_weight='balanced'` and `max_iter=1000`.
   - Offers clear mathematical coefficient interpretability.
2. **Random Forest Classifier (Ensemble Non-Linear Classifier):**
   - Configured with `n_estimators=200`, `max_depth=8`, `min_samples_split=10`, and `class_weight='balanced'`.
   - Captures complex non-linear feature interactions without overfitting.

---

## 14. Model Evaluation

### Comparison Table (Evaluated on 1,409 Hold-Out Test Samples)
| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 73.81% | 50.43% | **78.34%** | 0.6136 | 0.8417 |
| **Random Forest Classifier** | **75.44%** | **52.54%** | 77.54% | **0.6263** | **0.8432** |

### Business Evaluation Focus (Precision vs. Recall)
- **Cost of False Negative (FN):** High. Missing an actual churner means losing an account completely (\$500+ replacement cost).
- **Cost of False Positive (FP):** Low. Sending a promotional retention email/discount to a loyal customer carries negligible cost (\$5–\$10).
- **Decision:** **Random Forest** was selected as the champion model because it achieves the highest **ROC-AUC (0.8432)**, superior **F1-Score (0.6263)**, and **77.54% Recall**, capturing nearly 4 out of 5 churners while reducing false alarms compared to linear models.

---

## 15. Model Interpretation

### Top 10 Features Driving Predictions (Random Forest Gini Importance)
| Rank | Feature Name | Importance Score | Relationship with Churn |
| :---: | :--- | :---: | :--- |
| 1 | `tenure` | **0.1891** | Shorter tenure strongly associated with higher predicted churn. |
| 2 | `Contract_Two year` | **0.1227** | Two-year contract strongly associated with lower predicted churn. |
| 3 | `TotalCharges` | **0.1168** | Non-linear proxy for customer account maturity. |
| 4 | `InternetService_Fiber optic` | **0.0816** | Fiber optic subscription associated with higher predicted churn. |
| 5 | `MonthlyCharges` | **0.0746** | Higher monthly charges associated with higher predicted churn. |
| 6 | `PaymentMethod_Electronic check` | **0.0654** | Electronic check payment associated with higher predicted churn. |
| 7 | `Contract_One year` | **0.0494** | 1-year contract associated with lower predicted churn. |
| 8 | `OnlineSecurity_Yes` | **0.0384** | Having security add-on associated with lower predicted churn. |
| 9 | `TechSupport_No internet` | **0.0265** | Baseline non-internet accounts churn minimally. |
| 10 | `TechSupport_Yes` | **0.0258** | Having tech support associated with higher customer retention. |

---

## 16. Churn Prediction (Real-Time Inference Engine)
The trained pipeline is persisted at `models/churn_pipeline.joblib`.

### Inference Function Signature
```python
def predict_churn(customer_data: dict) -> dict:
    ...
```

### Sample Prediction Verification
```json
// Input Profile:
{
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "PaymentMethod": "Electronic check",
  "tenure": 2,
  "MonthlyCharges": 70.35,
  "TotalCharges": 140.70,
  ...
}

// Output:
{
  "Predicted Churn": "Yes",
  "Churn Probability": "84.4%",
  "Risk Tier": "High Risk"
}
```

---

## 17. Tableau Dashboard
- **Layout:** Standard 1366x768 executive view.
- **Global Interactive Filters:** `Contract`, `Internet Service`, `Payment Method`, `Senior Citizen`.
- **Top KPIs:** Total Customers (7,043), Churned Customers (1,869), Churn Rate (26.54%), Avg Monthly Charges (\$64.76), Monthly Revenue at Risk (\$139.1K).
- **Visuals:** Churn Rate by Contract, Churn Rate by Payment Method, Tenure Cohort Retention Cliff, Monthly Charges KDE Distribution.

---

## 18. Key Findings
1. **Contract Type dictates churn:** Month-to-month contracts have a **42.7% churn rate**, while 2-year contracts have only **2.8%**.
2. **First-year vulnerability:** **47.4% of all churn occurs within the first 12 months**.
3. **Fiber Optic churn anomaly:** Fiber optic users churn at **41.9%** (vs DSL at 19.0%), despite higher monthly billing.
4. **Electronic Check friction:** Customers paying via electronic check churn at **45.3%**, compared to **~15.5%** for automated credit card / bank transfer.
5. **Value of Support Add-ons:** Subscribers with `TechSupport` and `OnlineSecurity` have churn rates under **16%**, compared to over **41%** for those without.

---

## 19. Business Recommendations
1. **Contract Upgrade Incentives:** Target month-to-month users at month 3 with a 10% discount on a 1-year agreement.
2. **Proactive Onboarding for Fiber Customers:** Implement a 30-day proactive check-in and bundle complimentary 90-day tech support for new fiber optic installations.
3. **Automated Billing Incentives:** Provide a one-time \$10 bill credit for customers who switch from Electronic Check to automated credit card or ACH billing.

---

## 20. Challenges & Solutions
| Challenge | Solution |
| :--- | :--- |
| **Imbalanced Target Class (26.5% churn)** | Utilized `class_weight='balanced'` in Scikit-Learn models and evaluated via **ROC-AUC & Recall** rather than raw accuracy. |
| **Data Leakage in Transformations** | Wrapped encoders and scalers into a unified `Pipeline` and `ColumnTransformer` fitted solely on training data. |
| **Whitespace in Raw CSV** | Implemented robust string stripping and explicit type coercion with `errors='coerce'`. |

---

## 21. Limitations
- **Cross-Sectional Dataset:** Lacks time-series transactional timestamps for customer events.
- **No Direct Quality-of-Service (QoS) Metrics:** Network outage data, speed tests, and customer service ticket counts are absent from the dataset.

---

## 22. Future Improvements
- Implement hyperparameter optimization with Optuna / Bayesian Search.
- Build a lightweight Streamlit / FastAPI web application for batch scoring.
- Incorporate customer support call transcript sentiment analysis.

---

## 23. Interview Explanation (Elevator Pitch & Technical Walkthrough)
> *"In this project, I built an end-to-end Customer Churn Analytics and Machine Learning solution using Python, MySQL, Scikit-Learn, and Tableau across 7,043 telecom customer accounts.*
>
> *I discovered through statistical hypothesis testing ($p < 10^{-70}$) that month-to-month contract structure, electronic check billing, and high monthly charges are the primary drivers of churn, with 47.4% of attrition occurring within the first year.*
>
> *To solve this, I designed a production Scikit-Learn pipeline comparing balanced Logistic Regression and Random Forest models. The Random Forest achieved an **ROC-AUC of 0.8432** and **77.54% Recall**, capturing nearly 4 out of 5 churners. I serialized the model into an inference engine for real-time risk scoring and packaged actionable recommendations projected to reduce early subscriber churn."*
