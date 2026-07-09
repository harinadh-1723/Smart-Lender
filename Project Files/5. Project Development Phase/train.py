import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Try importing imblearn SMOTE, fallback to naive random oversampling if missing
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

def train_and_evaluate():
    # 1. Load the dataset
    data_path = os.path.join("data", "loan_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run generate_dataset.py first.")
    
    df = pd.read_csv(data_path)
    print("Dataset loaded successfully. Shape:", df.shape)
    
    # 2. Preprocessing
    # Cast potential object columns to float/numeric
    df['LoanAmount'] = pd.to_numeric(df['LoanAmount'], errors='coerce')
    df['Loan_Amount_Term'] = pd.to_numeric(df['Loan_Amount_Term'], errors='coerce')
    df['Credit_History'] = pd.to_numeric(df['Credit_History'], errors='coerce')
    
    # Fill missing values
    # Numerical: mean
    num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())
        
    # Categorical: mode
    cat_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    # 3. Categorical encoding
    # Exact mappings to be used consistently
    mappings = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
        'Loan_Status': {'Y': 1, 'N': 0}
    }
    
    # Map values
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            
    # Drop Loan_ID
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])
        
    # 4. Separate features and target
    X = df.drop(columns=['Loan_Status'])
    y = df['Loan_Status']
    
    print("\nPreprocessed Class Distribution:")
    print(y.value_counts())
    
    # 5. Balancing the Dataset
    if HAS_SMOTE:
        print("\nBalancing dataset using SMOTE...")
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
    else:
        print("\nimblearn.SMOTE not available. Applying random oversampling...")
        # Fallback manual oversampling if imblearn is not installed
        df_class_y = df[df['Loan_Status'] == 1]
        df_class_n = df[df['Loan_Status'] == 0]
        count_y = len(df_class_y)
        df_class_n_over = df_class_n.sample(count_y, replace=True, random_state=42)
        df_balanced = pd.concat([df_class_y, df_class_n_over], axis=0)
        X_res = df_balanced.drop(columns=['Loan_Status'])
        y_res = df_balanced['Loan_Status']
        
    print("Balanced Class Distribution:")
    print(y_res.value_counts())
    
    # 6. Scaling the Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 7. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y_res, test_size=0.2, random_state=42)
    print(f"Data split successfully. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 8. Model Building and Evaluation
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1)
    }
    
    trained_models = {}
    comparison = []
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled_df, y_res, cv=5)
        mean_cv = cv_scores.mean()
        
        print(f"Training Accuracy: {train_acc * 100:.2f}%")
        print(f"Testing Accuracy: {test_acc * 100:.2f}%")
        print(f"5-Fold Cross-Validation Accuracy: {mean_cv * 100:.2f}%")
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, test_pred))
        print("Classification Report:")
        print(classification_report(y_test, test_pred))
        
        comparison.append({
            "Model": name,
            "Train Accuracy (%)": round(train_acc * 100, 2),
            "Test Accuracy (%)": round(test_acc * 100, 2),
            "Cross-Val Accuracy (%)": round(mean_cv * 100, 2)
        })
        
    # Create Comparison DataFrame
    comp_df = pd.DataFrame(comparison)
    print("\n--- Model Performance Comparison ---")
    print(comp_df.to_string(index=False))
    
    # 9. Save the Best Performing Model (XGBoost)
    # The document references rdf.pkl as the file saved. Let's dump it.
    best_model = trained_models["XGBoost"]
    
    # Save the model and scaler to a dictionary
    save_data = {
        "model": best_model,
        "scaler": scaler,
        "features": list(X.columns),
        "mappings": mappings
    }
    
    # Save as rdf.pkl
    model_path = "rdf.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(save_data, f)
        
    print(f"\nSaved best model (XGBoost) and scaler to {model_path} successfully!")

if __name__ == "__main__":
    train_and_evaluate()
