"""
Jupyter Notebook Generator
Creates structured, runnable .ipynb files for EDA, SQL, and ML modeling
"""
import json
import os

def create_notebook(cells, output_path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Generated notebook: {output_path}")

# Notebook 1: Data Cleaning, EDA & Statistical Tests
cells_nb1 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Telecom Customer Churn Analysis & EDA\n",
            "### Comprehensive Exploratory Data Analysis & Statistical Testing\n",
            "**Tech Stack:** Python | Pandas | NumPy | Matplotlib | Seaborn | SciPy"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from scipy import stats\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# Visual configuration\n",
            "sns.set_theme(style='whitegrid', palette='deep')\n",
            "plt.rcParams['figure.figsize'] = (10, 5)\n",
            "%matplotlib inline"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load Raw Dataset & Initial Inspection"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "raw_df = pd.read_csv('../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')\n",
            "print(f'Dataset Dimensions: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns')\n",
            "raw_df.head(5)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Data Cleaning & Type Conversion\n",
            "- Convert `TotalCharges` from object to float (handling blank whitespace characters).\n",
            "- Impute zero-tenure customer charges.\n",
            "- Check and drop duplicates.\n",
            "- Create analytical helper columns (`Tenure_Group`, `Churn_Numeric`)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df = raw_df.copy()\n",
            "df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce').fillna(0.0)\n",
            "df['SeniorCitizen_Label'] = df['SeniorCitizen'].map({1: 'Yes', 0: 'No'})\n",
            "df['Churn_Numeric'] = df['Churn'].map({'Yes': 1, 'No': 0})\n",
            "\n",
            "def get_tenure_group(t):\n",
            "    if t <= 12: return '0-12 Months'\n",
            "    elif t <= 24: return '13-24 Months'\n",
            "    elif t <= 48: return '25-48 Months'\n",
            "    elif t <= 60: return '49-60 Months'\n",
            "    else: return '60+ Months'\n",
            "\n",
            "df['Tenure_Group'] = df['tenure'].apply(get_tenure_group)\n",
            "df.to_csv('../data/processed/cleaned_churn.csv', index=False)\n",
            "print('Cleaned dataset saved to data/processed/cleaned_churn.csv')\n",
            "df.info()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Exploratory Data Analysis (EDA)\n",
            "### 3.1 Overall Churn Rate"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "churn_counts = df['Churn'].value_counts()\n",
            "plt.figure(figsize=(6, 4))\n",
            "sns.barplot(x=churn_counts.index, y=churn_counts.values, palette=['#2b5c8f', '#d95f02'])\n",
            "plt.title('Overall Customer Churn Distribution', fontsize=12, fontweight='bold')\n",
            "plt.ylabel('Count')\n",
            "for i, v in enumerate(churn_counts.values):\n",
            "    plt.text(i, v/2, f'{v:,}\\n({v/len(df)*100:.1f}%)', ha='center', color='white', fontweight='bold')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3.2 Churn by Contract Type & Payment Method"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "sns.barplot(data=df, x='Contract', y='Churn_Numeric', ax=axes[0], ci=None, palette='Blues_r')\n",
            "axes[0].set_title('Churn Rate by Contract Type', fontweight='bold')\n",
            "axes[0].set_ylabel('Churn Rate')\n",
            "\n",
            "sns.barplot(data=df, x='PaymentMethod', y='Churn_Numeric', ax=axes[1], ci=None, palette='Oranges_r')\n",
            "axes[1].set_title('Churn Rate by Payment Method', fontweight='bold')\n",
            "axes[1].tick_params(axis='x', rotation=30)\n",
            "axes[1].set_ylabel('Churn Rate')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Statistical Hypothesis Testing\n",
            "### 4.1 Chi-Square Test: Contract Type vs Churn"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "chi2, p_val, dof, _ = stats.chi2_contingency(pd.crosstab(df['Contract'], df['Churn']))\n",
            "print(f'Chi2: {chi2:.4f}, p-value: {p_val:.4e} -> {\"Significant\" if p_val < 0.05 else \"Not Significant\"}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4.2 Welch's Two-Sample t-Test: Monthly Charges (Churned vs Retained)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "t_stat, p_val = stats.ttest_ind(\n",
            "    df[df['Churn']=='Yes']['MonthlyCharges'],\n",
            "    df[df['Churn']=='No']['MonthlyCharges'],\n",
            "    equal_var=False\n",
            ")\n",
            "print(f't-statistic: {t_stat:.4f}, p-value: {p_val:.4e}')"
        ]
    }
]

# Notebook 2: Machine Learning Prediction
cells_nb2 = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Telecom Customer Churn Prediction & Model Evaluation\n",
            "### Scikit-Learn Pipelines, Logistic Regression, Random Forest & Feature Importances"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import joblib\n",
            "\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
            "from sklearn.compose import ColumnTransformer\n",
            "from sklearn.pipeline import Pipeline\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from sklearn.metrics import (\n",
            "    accuracy_score, precision_score, recall_score, f1_score,\n",
            "    roc_auc_score, confusion_matrix, classification_report, roc_curve\n",
            ")\n",
            "%matplotlib inline"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Load Processed Data & Define Feature Sets"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df = pd.read_csv('../data/processed/cleaned_churn.csv')\n",
            "y = df['Churn'].map({'Yes': 1, 'No': 0})\n",
            "X = df.drop(columns=['customerID', 'Churn', 'Churn_Numeric', 'SeniorCitizen_Label', 'Tenure_Group', 'Avg_Monthly_Paid'])\n",
            "\n",
            "num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']\n",
            "cat_cols = [c for c in X.columns if c not in num_cols]\n",
            "\n",
            "preprocessor = ColumnTransformer([\n",
            "    ('num', StandardScaler(), num_cols),\n",
            "    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)\n",
            "])\n",
            "\n",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
            "print(f'Training shape: {X_train.shape}, Testing shape: {X_test.shape}')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Train Models (Logistic Regression & Random Forest)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "lr_pipe = Pipeline([\n",
            "    ('pre', preprocessor),\n",
            "    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))\n",
            "])\n",
            "lr_pipe.fit(X_train, y_train)\n",
            "\n",
            "rf_pipe = Pipeline([\n",
            "    ('pre', preprocessor),\n",
            "    ('clf', RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1))\n",
            "])\n",
            "rf_pipe.fit(X_train, y_train)\n",
            "print('Both models trained successfully.')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Model Evaluation & Comparison"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "results = []\n",
            "for name, pipe in [('Logistic Regression', lr_pipe), ('Random Forest', rf_pipe)]:\n",
            "    y_pred = pipe.predict(X_test)\n",
            "    y_prob = pipe.predict_proba(X_test)[:, 1]\n",
            "    results.append({\n",
            "        'Model': name,\n",
            "        'Accuracy': f'{accuracy_score(y_test, y_pred)*100:.2f}%',\n",
            "        'Precision': f'{precision_score(y_test, y_pred)*100:.2f}%',\n",
            "        'Recall': f'{recall_score(y_test, y_pred)*100:.2f}%',\n",
            "        'F1-Score': f'{f1_score(y_test, y_pred):.4f}',\n",
            "        'ROC-AUC': f'{roc_auc_score(y_test, y_prob):.4f}'\n",
            "    })\n",
            "pd.DataFrame(results)"
        ]
    }
]

if __name__ == '__main__':
    create_notebook(cells_nb1, 'notebooks/01_data_cleaning_eda.ipynb')
    create_notebook(cells_nb2, 'notebooks/02_churn_prediction.ipynb')
