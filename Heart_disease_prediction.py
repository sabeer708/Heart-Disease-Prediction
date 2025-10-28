import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv(r"D:\Heart_project\Heart_Disease_Prediction.csv")

print("Dataset Loaded Successfully!\n")
print(df.head())
print("\nChecking for Missing Values:")
print(df.isnull().sum())
print("\nDataset Shape:", df.shape)
print("\nStatistical Summary:")
print(df.describe())

target_col = 'Heart Disease' if 'Heart Disease' in df.columns else 'target'

sns.countplot(data=df, x=target_col, hue=target_col, palette='coolwarm', legend=False)

plt.title("Heart Disease Distribution (0 = No, 1 = Yes)")
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))

print("\nSupport Vector Machine Results")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

acc_log = accuracy_score(y_test, y_pred_log)
acc_svm = accuracy_score(y_test, y_pred_svm)

print("\nModel Comparison:")
print(f"Logistic Regression Accuracy: {acc_log*100:.2f}%")
print(f"SVM Accuracy: {acc_svm*100:.2f}%")

plt.bar(['Logistic Regression', 'SVM'], [acc_log, acc_svm], color=['blue', 'orange'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()
