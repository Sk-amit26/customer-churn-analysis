"""
Machine Learning Training, Evaluation, Interpretation, and Inference Pipeline
Stages 7 - 12: Feature Engineering, Modeling (Logistic Regression & Random Forest), Evaluation & Persistence
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

def build_and_train_models(data_path="data/processed/cleaned_churn.csv",
                           model_dir="models",
                           viz_dir="tableau/visualizations"):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    print("="*60)
    print("STAGE 7: FEATURE ENGINEERING & PREPROCESSING")
    print("="*60)
    df = pd.read_csv(data_path)

    # Define target and feature sets
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Drop identifiers and derived target proxies to avoid data leakage
    drop_cols = ['customerID', 'Churn', 'Churn_Numeric', 'SeniorCitizen_Label', 'Tenure_Group', 'Avg_Monthly_Paid']
    X = df.drop(columns=drop_cols)

    # Identify numerical and categorical features
    num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_features = [col for col in X.columns if col not in num_features]

    print(f"Features selected: {len(X.columns)} features")
    print(f"Numerical features ({len(num_features)}): {num_features}")
    print(f"Categorical features ({len(cat_features)}): {cat_features}")

    # Build Preprocessor Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )

    print("\n" + "="*60)
    print("STAGE 8: STRATIFIED TRAIN / TEST SPLIT")
    print("="*60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Total Dataset: {len(X):,} samples")
    print(f"Training Set:  {len(X_train):,} samples ({(len(X_train)/len(X))*100:.1f}%) | Churn Rate: {y_train.mean()*100:.2f}%")
    print(f"Testing Set:   {len(X_test):,} samples ({(len(X_test)/len(X))*100:.1f}%) | Churn Rate: {y_test.mean()*100:.2f}%")

    print("\n" + "="*60)
    print("STAGE 9: MODEL TRAINING (Logistic Regression vs Random Forest)")
    print("="*60)

    # 1. Logistic Regression Pipeline
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    print("Training Logistic Regression (class_weight='balanced')...")
    lr_pipeline.fit(X_train, y_train)

    # 2. Random Forest Pipeline
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=10, 
                                               class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    print("Training Random Forest Classifier (max_depth=8, class_weight='balanced')...")
    rf_pipeline.fit(X_train, y_train)

    print("\n" + "="*60)
    print("STAGE 10: MODEL EVALUATION")
    print("="*60)

    models = {
        'Logistic Regression': lr_pipeline,
        'Random Forest': rf_pipeline
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

        print(f"\n--- {name} Classification Report ---")
        print(classification_report(y_test, y_pred, target_names=['Retained (0)', 'Churned (1)']))

    # Print Comparison Table
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("MODEL COMPARISON TABLE")
    print("="*60)
    print(results_df.to_string(index=False))

    # Plot ROC Curves
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Recall)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=13, fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/04_model_roc_curves.png", dpi=300)
    plt.close()

    print("\n" + "="*60)
    print("STAGE 11: MODEL INTERPRETATION & FEATURE IMPORTANCE")
    print("="*60)

    # Extract transformed feature names
    cat_encoder = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cat_names = list(cat_encoder.get_feature_names_out(cat_features))
    all_feature_names = num_features + encoded_cat_names

    # Random Forest Feature Importance
    rf_importances = rf_pipeline.named_steps['classifier'].feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Importance': rf_importances
    }).sort_values(by='Importance', ascending=False)

    print("\nTop 10 Important Features (Random Forest):")
    print(feat_imp_df.head(10).to_string(index=False))

    plt.figure(figsize=(9, 6))
    sns.barplot(data=feat_imp_df.head(10), x='Importance', y='Feature', palette='crest')
    plt.title('Top 10 Predictive Features of Customer Churn', fontsize=13, fontweight='bold')
    plt.xlabel('Gini Feature Importance', fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/05_feature_importance.png", dpi=300)
    plt.close()

    print("\n" + "="*60)
    print("STAGE 12: MODEL SERIALIZATION & INFERENCE ENGINE")
    print("="*60)

    # Save the highest ROC-AUC model pipeline
    best_model_name = "Logistic Regression" if roc_data['Logistic Regression'][2] >= roc_data['Random Forest'][2] else "Random Forest"
    best_pipeline = models[best_model_name]
    saved_path = f"{model_dir}/churn_pipeline.joblib"
    joblib.dump(best_pipeline, saved_path)
    print(f"Selected best model ({best_model_name}) saved to: {saved_path}")

    return best_pipeline

def predict_churn(customer_data: dict, model_path="models/churn_pipeline.joblib"):
    """
    Real-time Churn Prediction Function
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")
    
    pipeline = joblib.load(model_path)
    input_df = pd.DataFrame([customer_data])
    
    churn_prob = pipeline.predict_proba(input_df)[0][1]
    churn_pred = "Yes" if churn_prob >= 0.50 else "No"
    
    return {
        "Predicted Churn": churn_pred,
        "Churn Probability": f"{churn_prob * 100:.1f}%",
        "Risk Tier": "High Risk" if churn_prob >= 0.65 else ("Moderate Risk" if churn_prob >= 0.40 else "Low Risk")
    }

if __name__ == "__main__":
    best_model = build_and_train_models()
    
    print("\nTesting Sample Prediction:")
    sample_customer = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 70.35,
        'TotalCharges': 140.70
    }
    result = predict_churn(sample_customer)
    print("Input Customer Profile (Month-to-month, Fiber optic, Electronic check, Tenure=2 months):")
    for k, v in result.items():
        print(f"  {k}: {v}")
