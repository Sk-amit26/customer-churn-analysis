# Documentation — Project Setup

## 1. Objective
Establish a clean, modular, and reproducible project environment for the end-to-end **Customer Churn Analysis & Prediction** project.

## 2. Folder Structure
`	ext
customer-churn-analysis/
│
├── data/
│   ├── raw/                # Original untouched raw dataset
│   └── processed/          # Cleaned, transformed, and feature-engineered datasets
│
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb   # Interactive data cleaning, EDA, and statistical tests
│   └── 02_churn_prediction.ipynb     # Model training, evaluation, and interpretation
│
├── sql/
│   └── churn_analysis.sql           # MySQL database schema and analytical business queries
│
├── tableau/                         # Tableau workbook files (.twbx / .twb) and exported charts
│
├── src/                             # Modular Python utility scripts and reusable helper functions
│
├── models/                          # Serialized ML model artifacts (.joblib / .pkl)
│
├── docs/                            # Step-by-step project documentation reports
│
├── README.md                        # Master project documentation for GitHub portfolio
├── requirements.txt                 # Exact pinned/minimum library versions
└── .gitignore                       # Ignored files (venv, caches, temporary files)
`

## 3. Purpose of Each Folder
- **data/raw/**: Holds the original raw dataset as a single source of truth without manual modification.
- **data/processed/**: Stores cleaned datasets for SQL ingestion, Tableau reporting, and ML pipelines.
- **notebooks/**: Contains structured Jupyter notebooks for exploratory workflows and analysis documentation.
- **sql/**: Contains production-ready MySQL table definitions, data loading scripts, and analytical business queries.
- **tableau/**: Holds Tableau artifacts, calculated field definitions, and dashboard assets.
- **src/**: Houses clean, testable Python code for end-to-end inference and data processing.
- **models/**: Houses serialized Scikit-learn pipelines and trained estimator weights.
- **docs/**: Detailed documentation log created after each stage for interview and resume study.

## 4. Libraries & Tech Stack
| Library / Tool | Purpose & Justification |
| :--- | :--- |
| **Python 3.10+** | Core runtime environment for scripting, data pipelines, and modeling. |
| **pandas** | High-performance tabular data manipulation, transformation, and aggregation. |
| **numpy** | Numerical operations and vectorized array transformations. |
| **matplotlib & seaborn** | Publication-quality statistical visualizations and business charts. |
| **scipy** | Rigorous statistical hypothesis testing (Chi-square test of independence, two-sample t-tests). |
| **scikit-learn** | Machine learning preprocessing pipelines, Logistic Regression, Random Forest, metrics. |
| **joblib** | Efficient serialization and persistence of trained ML models and transformers. |
| **mysql-connector-python / sqlalchemy** | Database connectivity and SQL query integration with pandas. |
| **Jupyter / ipykernel** | Interactive execution environment for exploratory analysis. |

## 5. Environment Setup Commands
`ash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
`
