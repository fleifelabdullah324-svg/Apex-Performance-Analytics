import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def main():
    print("==================================================")
    print("  APEX PERFORMANCE ANALYTICS - ML PIPELINE")
    print("==================================================")
    
    # 1. Load the Dataset
    print("\n[1] Loading dataset...")
    try:
        df = pd.read_csv("The_Flawless_Tuning_ERP_V20_Fixed.csv")
        print(f"Dataset loaded successfully! Shape: {df.shape}")
    except FileNotFoundError:
        print("Error: Dataset file 'The_Flawless_Tuning_ERP_V20_Fixed.csv' not found.")
        return

    # 2. Data Cleaning & Preprocessing (ETL)
    print("\n[2] Performing Data Cleaning and Imputation...")
    # Fill numerical nulls with mean
    df.fillna(df.mean(numeric_only=True), inplace=True)
    # Fill categorical nulls with 'Standard'
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna('Standard', inplace=True)
        
    print("Handling Categorical Encoding...")
    # Separate features and target
    # Assuming 'Safety_Status' is the target variable (Safe, Risky, Critical Failure)
    if 'Safety_Status' not in df.columns:
        print("Warning: 'Safety_Status' column not found. Please ensure the target column is correct.")
        # Fallback for demonstration if actual target name differs
        target_col = df.columns[-1] 
    else:
        target_col = 'Safety_Status'
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # One-Hot Encoding
    X = pd.get_dummies(X, drop_first=True)
    print(f"Features after encoding: {X.shape[1]}")

    # 3. Data Scaling
    print("\n[3] Applying Min-Max Scaling to numerical features...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Train-Test Split (80/20)
    print("\n[4] Splitting data into Training (80%) and Testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    # 5. Model Training (Random Forest)
    print("\n[5] Initializing and Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    print("Training completed!")

    # 6. Model Evaluation
    print("\n[6] Evaluating Model Performance...")
    y_pred = rf_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {acc * 100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ROC AUC (if probability distribution is available and multi-class)
    try:
        y_prob = rf_model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
        print(f"\nROC AUC Score (OvR): {roc_auc:.4f}")
    except Exception as e:
        print("\nROC AUC Score could not be computed (requires multi-class OvR format).")

    print("\n==================================================")
    print("  PIPELINE EXECUTION FINISHED")
    print("==================================================")

if __name__ == "__main__":
    main()
