-- ==========================================================
-- Customer Churn Analysis & Business Insights
-- Database: customer_churn
-- Tech: MySQL 8.0+
-- ==========================================================

-- 1. Database Creation
CREATE DATABASE IF NOT EXISTS customer_churn;
USE customer_churn;

-- 2. Clean Table DDL
DROP TABLE IF EXISTS telco_churn;

CREATE TABLE telco_churn (
    customer_id VARCHAR(50) PRIMARY KEY,
    gender VARCHAR(10),
    senior_citizen INT,
    partner VARCHAR(10),
    dependents VARCHAR(10),
    tenure INT,
    phone_service VARCHAR(10),
    multiple_lines VARCHAR(25),
    internet_service VARCHAR(25),
    online_security VARCHAR(25),
    online_backup VARCHAR(25),
    device_protection VARCHAR(25),
    tech_support VARCHAR(25),
    streaming_tv VARCHAR(25),
    streaming_movies VARCHAR(25),
    contract VARCHAR(25),
    paperless_billing VARCHAR(10),
    payment_method VARCHAR(50),
    monthly_charges DECIMAL(10, 2),
    total_charges DECIMAL(10, 2),
    churn VARCHAR(10),
    tenure_group VARCHAR(20),
    churn_numeric INT
);

-- ==========================================================
-- ANALYTICAL QUERIES
-- ==========================================================

-- Query 1: Total Customers, Churned Customers & Overall Churn Rate
SELECT 
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    SUM(CASE WHEN churn = 'No' THEN 1 ELSE 0 END) AS retained_customers,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM telco_churn;

-- Query 2: Churn Rate by Contract Type
SELECT 
    contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM telco_churn
GROUP BY contract
ORDER BY churn_rate_pct DESC;

-- Query 3: Churn Rate by Payment Method
SELECT 
    payment_method,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM telco_churn
GROUP BY payment_method
ORDER BY churn_rate_pct DESC;

-- Query 4: Churn Rate by Internet Service Type
SELECT 
    internet_service,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
FROM telco_churn
GROUP BY internet_service
ORDER BY churn_rate_pct DESC;

-- Query 5: Average Monthly Charges & Total Charges by Churn Status
SELECT 
    churn,
    COUNT(*) AS customer_count,
    ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges,
    ROUND(AVG(total_charges), 2) AS avg_total_charges,
    ROUND(AVG(tenure), 1) AS avg_tenure_months
FROM telco_churn
GROUP BY churn;

-- Query 6: Churn Rate by Tenure Cohort
SELECT 
    tenure_group,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM telco_churn
GROUP BY tenure_group
ORDER BY 
    CASE 
        WHEN tenure_group = '0-12 Months' THEN 1
        WHEN tenure_group = '13-24 Months' THEN 2
        WHEN tenure_group = '25-48 Months' THEN 3
        WHEN tenure_group = '49-60 Months' THEN 4
        ELSE 5 
    END;

-- Query 7: Churn Rate by Tech Support & Online Security Add-on Services
SELECT 
    tech_support,
    online_security,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM telco_churn
GROUP BY tech_support, online_security
ORDER BY churn_rate_pct DESC;

-- Query 8: High-Value Churned Customers (Monthly charges > overall avg)
WITH CustomerAverages AS (
    SELECT AVG(monthly_charges) AS overall_avg_monthly FROM telco_churn
)
SELECT 
    c.customer_id,
    c.tenure,
    c.contract,
    c.payment_method,
    c.internet_service,
    c.monthly_charges,
    c.total_charges
FROM telco_churn c
CROSS JOIN CustomerAverages a
WHERE c.churn = 'Yes' 
  AND c.monthly_charges > a.overall_avg_monthly
ORDER BY c.monthly_charges DESC
LIMIT 20;

-- Query 9: Customer Segmentation: Highest Risk Churn Cohorts
-- (Month-to-month, Fiber optic, Electronic check)
SELECT 
    contract,
    internet_service,
    payment_method,
    COUNT(*) AS segment_size,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_count,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS segment_churn_rate_pct,
    ROUND(SUM(CASE WHEN churn = 'Yes' THEN monthly_charges ELSE 0 END), 2) AS monthly_revenue_lost
FROM telco_churn
GROUP BY contract, internet_service, payment_method
HAVING COUNT(*) >= 50
ORDER BY segment_churn_rate_pct DESC
LIMIT 10;

-- Query 10: Cumulative Monthly Revenue Loss from Churned Customers
SELECT 
    contract,
    COUNT(*) AS total_churned,
    ROUND(SUM(monthly_charges), 2) AS monthly_revenue_at_risk,
    ROUND(SUM(total_charges), 2) AS historical_lifetime_revenue_lost,
    ROUND(AVG(tenure), 1) AS avg_tenure_at_churn
FROM telco_churn
WHERE churn = 'Yes'
GROUP BY contract
ORDER BY monthly_revenue_at_risk DESC;
