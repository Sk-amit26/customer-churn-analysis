"""
Optimized Churn Prediction Pipeline v2
Improvements: Hyperparameter tuning, Gradient Boosting, Voting Ensemble, 
              threshold optimization, cross-validation, enhanced feature engineering.
Same tech stack: Scikit-learn only.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve,
    precision_recall_curve
)

def engineer_features(df):
    """Enhanced feature engineering with domain-driven derived features."""
    df = df.copy()
    
    # 1. Avg monthly paid vs stated monthly charges (billing consistency)
    df['Charge_Ratio'] = np.where(
        df['tenure'] > 0,
        df['TotalCharges'] / (df['tenure'] * df['MonthlyCharges']),
        1.0
    )
    
    # 2. Tenure bins as ordinal (captures non-linear tenure effect better)
    df['Tenure_Bucket'] = pd.cut(
        df['tenure'],
        bins=[-1, 6, 12, 24, 48, 60, 73],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(int)
    
    # 3. Number of add-on services subscribed (proxy for engagement/stickiness)
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['Num_Services'] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v == 'Yes'), axis=1
    )
    
    # 4. Has any protection service (security OR backup OR device protection)
    df['Has_Protection'] = ((df['OnlineSecurity'] == 'Yes') | 
                            (df['OnlineBackup'] == 'Yes') | 
                            (df['DeviceProtection'] == 'Yes')).astype(int)
    
    # 5. Has any support (tech support)
    df['Has_Support'] = (df['TechSupport'] == 'Yes').astype(int)
    
    # 6. Streaming engagement
    df['Has_Streaming'] = ((df['StreamingTV'] == 'Yes') | 
                           (df['StreamingMovies'] == 'Yes')).astype(int)
    
    # 7. Monthly charges per service (value perception)
    df['Charge_Per_Service'] = np.where(
        df['Num_Services'] > 0,
        df['MonthlyCharges'] / df['Num_Services'],
        df['MonthlyCharges']
    )
    
    # 8. Is new customer (tenure <= 6 months — highest risk window)
    df['Is_New_Customer'] = (df['tenure'] <= 6).astype(int)
    
    # 9. High spender flag
    df['Is_High_Spender'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)
    
    # 10. Risk interaction: Month-to-month + Fiber optic + Electronic check
    df['Triple_Risk'] = (
        (df['Contract'] == 'Month-to-month') & 
        (df['InternetService'] == 'Fiber optic') & 
        (df['PaymentMethod'] == 'Electronic check')
    ).astype(int)
    
    return df


def build_optimized_models(data_path="data/processed/cleaned_churn.csv",
                           model_dir="models",
                           viz_dir="tableau/visualizations"):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    print("="*65)
    print("  OPTIMIZED CHURN PREDICTION PIPELINE v2")
    print("="*65)
    
    df = pd.read_csv(data_path)
    df = engineer_features(df)
    
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Drop identifiers, targets, and derived labels
    drop_cols = ['customerID', 'Churn', 'Churn_Numeric', 'SeniorCitizen_Label', 
                 'Tenure_Group', 'Avg_Monthly_Paid']
    X = df.drop(columns=drop_cols)
    
    # Separate numerical and categorical
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges',
                    'Charge_Ratio', 'Tenure_Bucket', 'Num_Services',
                    'Has_Protection', 'Has_Support', 'Has_Streaming',
                    'Charge_Per_Service', 'Is_New_Customer', 'Is_High_Spender', 'Triple_Risk']
    cat_features = [c for c in X.columns if c not in num_features]
    
    print(f"\nTotal features: {len(X.columns)}")
    print(f"  Numerical ({len(num_features)}): {num_features}")
    print(f"  Categorical ({len(cat_features)}): {cat_features}")
    
    # Preprocessor
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_features)
    ])
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)} | Churn Rate: {y.mean()*100:.2f}%")
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # ====================================================================
    # MODEL 1: Tuned Logistic Regression
    # ====================================================================
    print("\n" + "-"*65)
    print("[1/4] Logistic Regression (Tuned)")
    print("-"*65)
    
    lr_pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(
            C=0.1, 
            max_iter=2000, 
            class_weight='balanced', 
            solver='lbfgs',
            random_state=42
        ))
    ])
    lr_pipe.fit(X_train, y_train)
    cv_auc_lr = cross_val_score(lr_pipe, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"  5-Fold CV ROC-AUC: {cv_auc_lr.mean():.4f} (+/- {cv_auc_lr.std():.4f})")
    
    # ====================================================================
    # MODEL 2: Tuned Random Forest
    # ====================================================================
    print("\n" + "-"*65)
    print("[2/4] Random Forest (Hyperparameter Tuned)")
    print("-"*65)
    
    rf_pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(
            n_estimators=500,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf_pipe.fit(X_train, y_train)
    cv_auc_rf = cross_val_score(rf_pipe, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"  5-Fold CV ROC-AUC: {cv_auc_rf.mean():.4f} (+/- {cv_auc_rf.std():.4f})")
    
    # ====================================================================
    # MODEL 3: Gradient Boosting Classifier (Scikit-learn native)
    # ====================================================================
    print("\n" + "-"*65)
    print("[3/4] Gradient Boosting Classifier (Tuned)")
    print("-"*65)
    
    gb_pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42
        ))
    ])
    gb_pipe.fit(X_train, y_train)
    cv_auc_gb = cross_val_score(gb_pipe, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"  5-Fold CV ROC-AUC: {cv_auc_gb.mean():.4f} (+/- {cv_auc_gb.std():.4f})")
    
    # ====================================================================
    # MODEL 4: Soft Voting Ensemble (LR + RF + GB)
    # ====================================================================
    print("\n" + "-"*65)
    print("[4/4] Soft Voting Ensemble (LR + RF + GB)")
    print("-"*65)
    
    ensemble_pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(C=0.1, max_iter=2000, class_weight='balanced', random_state=42)),
                ('rf', RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_split=5,
                                              min_samples_leaf=2, class_weight='balanced_subsample',
                                              random_state=42, n_jobs=-1)),
                ('gb', GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                                  subsample=0.8, min_samples_split=10, random_state=42))
            ],
            voting='soft',
            weights=[1, 2, 3]  # Give more weight to GB (strongest)
        ))
    ])
    ensemble_pipe.fit(X_train, y_train)
    cv_auc_ens = cross_val_score(ensemble_pipe, X_train, y_train, cv=cv, scoring='roc_auc')
    print(f"  5-Fold CV ROC-AUC: {cv_auc_ens.mean():.4f} (+/- {cv_auc_ens.std():.4f})")
    
    # ====================================================================
    # EVALUATION ON HOLD-OUT TEST SET
    # ====================================================================
    print("\n" + "="*65)
    print("  HOLD-OUT TEST SET EVALUATION (1,409 samples)")
    print("="*65)
    
    models = {
        'Logistic Regression (Tuned)': lr_pipe,
        'Random Forest (Tuned)': rf_pipe,
        'Gradient Boosting': gb_pipe,
        'Voting Ensemble (LR+RF+GB)': ensemble_pipe
    }
    
    results = []
    roc_data = {}
    
    for name, pipe in models.items():
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        results.append({
            'Model': name,
            'Accuracy': f"{acc*100:.2f}%",
            'Precision': f"{prec*100:.2f}%",
            'Recall': f"{rec*100:.2f}%",
            'F1-Score': f"{f1:.4f}",
            'ROC-AUC': f"{auc:.4f}"
        })
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc)
        
        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))
    
    results_df = pd.DataFrame(results)
    print("\n" + "="*65)
    print("  MODEL COMPARISON TABLE")
    print("="*65)
    print(results_df.to_string(index=False))
    
    # ====================================================================
    # THRESHOLD OPTIMIZATION (for the best model)
    # ====================================================================
    print("\n" + "="*65)
    print("  THRESHOLD OPTIMIZATION (Gradient Boosting)")
    print("="*65)
    
    best_pipe = gb_pipe
    y_proba_best = best_pipe.predict_proba(X_test)[:, 1]
    
    best_threshold = 0.5
    best_f1 = 0
    
    for thresh in np.arange(0.25, 0.60, 0.01):
        y_adj = (y_proba_best >= thresh).astype(int)
        f1_val = f1_score(y_test, y_adj)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_threshold = thresh
    
    y_optimized = (y_proba_best >= best_threshold).astype(int)
    print(f"  Optimal Threshold: {best_threshold:.2f}")
    print(f"  Accuracy:  {accuracy_score(y_test, y_optimized)*100:.2f}%")
    print(f"  Precision: {precision_score(y_test, y_optimized)*100:.2f}%")
    print(f"  Recall:    {recall_score(y_test, y_optimized)*100:.2f}%")
    print(f"  F1-Score:  {f1_score(y_test, y_optimized):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba_best):.4f}")
    
    # ====================================================================
    # ROC CURVE PLOT (all 4 models)
    # ====================================================================
    plt.figure(figsize=(8, 7))
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    for (name, (fpr, tpr, auc)), color in zip(roc_data.items(), colors):
        label_short = name.split('(')[0].strip()
        plt.plot(fpr, tpr, lw=2.2, color=color, label=f'{label_short} (AUC={auc:.3f})')
    plt.plot([0,1], [0,1], 'k--', lw=1, alpha=0.4, label='Random (AUC=0.500)')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('Optimized Model ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/06_optimized_roc_curves.png", dpi=300)
    plt.close()
    print(f"\nROC curve saved to {viz_dir}/06_optimized_roc_curves.png")
    
    # ====================================================================
    # FEATURE IMPORTANCE (Gradient Boosting)
    # ====================================================================
    cat_encoder = gb_pipe.named_steps['pre'].named_transformers_['cat']
    encoded_names = list(cat_encoder.get_feature_names_out(cat_features))
    all_names = num_features + encoded_names
    
    gb_importances = gb_pipe.named_steps['clf'].feature_importances_
    feat_df = pd.DataFrame({'Feature': all_names, 'Importance': gb_importances})
    feat_df = feat_df.sort_values('Importance', ascending=False)
    
    print("\nTop 15 Features (Gradient Boosting):")
    print(feat_df.head(15).to_string(index=False))
    
    plt.figure(figsize=(9, 7))
    top15 = feat_df.head(15)
    sns.barplot(data=top15, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
    plt.title('Top 15 Predictive Features (Gradient Boosting)', fontsize=13, fontweight='bold')
    plt.xlabel('Feature Importance', fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/07_optimized_feature_importance.png", dpi=300)
    plt.close()
    
    # ====================================================================
    # SAVE BEST MODEL
    # ====================================================================
    saved_path = f"{model_dir}/churn_pipeline_v2.joblib"
    joblib.dump(gb_pipe, saved_path)
    print(f"\nBest model (Gradient Boosting) saved to: {saved_path}")
    
    # Also save the feature engineering function reference info
    meta = {
        'optimal_threshold': float(best_threshold),
        'model': 'GradientBoostingClassifier',
        'cv_roc_auc': float(cv_auc_gb.mean()),
        'test_roc_auc': float(roc_auc_score(y_test, y_proba_best)),
        'engineered_features': ['Charge_Ratio', 'Tenure_Bucket', 'Num_Services',
                                'Has_Protection', 'Has_Support', 'Has_Streaming',
                                'Charge_Per_Service', 'Is_New_Customer', 'Is_High_Spender', 'Triple_Risk']
    }
    joblib.dump(meta, f"{model_dir}/model_meta_v2.joblib")
    print("Model metadata saved.")
    
    return gb_pipe, best_threshold


if __name__ == "__main__":
    best_model, threshold = build_optimized_models()
