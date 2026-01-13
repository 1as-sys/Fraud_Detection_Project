import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import pickle

print("Starting Fraud Detection Model Training...")

print("Loading dataset...")
data = pd.read_csv(r"C:\Users\HP\Downloads astha\Fraud_Detection_Project\data\creditcard.csv")

print("Scaling Amount and Time...")
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data['Time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))

print("Preparing features and target...")
X = data.drop('Class', axis=1)
y = data['Class']

print("Splitting train and test data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Applying SMOTE to handle imbalance...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Training Random Forest model...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)

print("Saving model...")
with open(r"C:\Users\HP\Downloads astha\Fraud_Detection_Project\models\rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("Evaluating model...")
y_pred = rf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

print("\n✅ Training completed successfully!")
