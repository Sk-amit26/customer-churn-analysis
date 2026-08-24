# Tableau Dashboard Specifications & Calculated Fields
### Customer Churn Analysis & Prediction Dashboard

## 1. Data Source
- File: `data/processed/cleaned_churn.csv` (7,043 rows)

## 2. Calculated Fields for Tableau
Paste these formulas into Tableau Calculated Fields:

### KPI: Total Customers
```tableau
COUNT([CustomerID])
```

### KPI: Churned Customers
```tableau
SUM(IIF([Churn] = "Yes", 1, 0))
```

### KPI: Retained Customers
```tableau
SUM(IIF([Churn] = "No", 1, 0))
```

### KPI: Churn Rate (%)
```tableau
SUM(IIF([Churn] = "Yes", 1, 0)) / COUNT([CustomerID])
```
*Format as Percentage with 1 or 2 decimal places.*

### KPI: Average Monthly Charges
```tableau
AVG([MonthlyCharges])
```
*Format as Currency ($).*

### KPI: Total Monthly Revenue at Risk
```tableau
SUM(IIF([Churn] = "Yes", [MonthlyCharges], 0))
```

### Categorization: Senior Citizen Status
```tableau
IF [SeniorCitizen] = 1 THEN "Senior Citizen" ELSE "Non-Senior Citizen" END
```

### Categorization: Tenure Cohort
```tableau
IF [Tenure] <= 12 THEN "0-12 Months"
ELSEIF [Tenure] <= 24 THEN "13-24 Months"
ELSEIF [Tenure] <= 48 THEN "25-48 Months"
ELSEIF [Tenure] <= 60 THEN "49-60 Months"
ELSE "60+ Months"
END
```

### High Risk Flag (ML Rule-based Segment)
```tableau
IF [Contract] = "Month-to-month" AND [InternetService] = "Fiber optic" AND [PaymentMethod] = "Electronic check"
THEN "High Risk Segment"
ELSE "Standard Segment"
END
```

---

## 3. Dashboard Structure & Layout (1366 x 768 px)
- **Top Header:** Title, Subtitle, Global Interactive Filters (Contract, Internet Service, Payment Method, Senior Citizen).
- **Top KPI Cards:**
  1. Total Customers (7,043)
  2. Churned Customers (1,869)
  3. Overall Churn Rate (26.5%)
  4. Avg Monthly Charges ($64.76)
  5. Monthly Revenue Lost ($139.1K)
- **Middle Visuals (2 Columns):**
  - Left: Bar Chart - *Churn Rate by Contract Type (Month-to-Month 42.7% vs 1-Year 11.3% vs 2-Year 2.8%)*
  - Right: Bar Chart - *Churn Rate by Internet Service (Fiber Optic 41.9% vs DSL 19.0% vs No 7.4%)*
- **Bottom Visuals (2 Columns):**
  - Left: Histogram/Area - *Customer Tenure Cohort vs Churn Rate (0-12 Months: 47.7% churn drop-off)*
  - Right: Boxplot/KDE - *Monthly Charges Distribution (Median $79.65 for churned vs $64.40 for retained)*
