# Credit Card Fraud Detection
# Install required libraries first:
# pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# Step 1: Load the dataset
# Download from: https://www.kaggle.com/mlg-ulb/creditcardfraud
print("Loading dataset...")
df = pd.read_csv('creditcard.csv')

# Step 2: Exploratory Data Analysis
print("\n=== Dataset Information ===")
print(f"Shape: {df.shape}")
print(f"\nClass Distribution:\n{df['Class'].value_counts()}")
print(f"\nFraud Percentage: {df['Class'].sum() / len(df) * 100:.4f}%")

# Visualize class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=df)
plt.title('Class Distribution (0: Legitimate, 1: Fraud)')
plt.savefig('class_distribution.png')
plt.close()

# Step 3: Data Preprocessing
print("\n=== Preprocessing Data ===")

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Scale the 'Amount' and 'Time' features
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))
X['Time'] = scaler.fit_transform(X['Time'].values.reshape(-1, 1))

# Step 4: Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Testing set size: {X_test.shape}")

# Step 5: Handle Imbalanced Data using SMOTE
print("\n=== Applying SMOTE ===")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"After SMOTE - Training set size: {X_train_balanced.shape}")
print(f"Class distribution after SMOTE:\n{pd.Series(y_train_balanced).value_counts()}")

# Step 6: Train Models

# Model 1: Logistic Regression
print("\n=== Training Logistic Regression ===")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_balanced, y_train_balanced)

# Model 2: Random Forest
print("=== Training Random Forest ===")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_balanced, y_train_balanced)

# Step 7: Evaluate Models
def evaluate_model(model, X_test, y_test, model_name):
    print(f"\n{'='*50}")
    print(f"Evaluation: {model_name}")
    print('='*50)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # ROC-AUC Score
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_")}.png')
    plt.close()
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.savefig(f'roc_curve_{model_name.replace(" ", "_")}.png')
    plt.close()
    
    return roc_auc

# Evaluate both models
lr_auc = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
rf_auc = evaluate_model(rf_model, X_test, y_test, "Random Forest")

# Step 8: Compare Models
print("\n=== Model Comparison ===")
print(f"Logistic Regression ROC-AUC: {lr_auc:.4f}")
print(f"Random Forest ROC-AUC: {rf_auc:.4f}")

# Step 9: Feature Importance (Random Forest)
print("\n=== Top 10 Important Features ===")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Top 10 Feature Importances - Random Forest')
plt.savefig('feature_importance.png')
plt.close()

print("\n=== Training Complete! ===")
print("Generated files:")
print("- class_distribution.png")
print("- confusion_matrix_*.png")
print("- roc_curve_*.png")
print("- feature_importance.png")