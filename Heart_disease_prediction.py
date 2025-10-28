import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# Load Dataset
df = pd.read_csv(r"D:\Heart_project\Heart_Disease_Prediction.csv")

print("Dataset Loaded Successfully!\n")
print(df.head())
print("\nChecking for Missing Values:")
print(df.isnull().sum())
print("\nDataset Shape:", df.shape)
print("\nStatistical Summary:")
print(df.describe())

target_col = 'Heart Disease' if 'Heart Disease' in df.columns else 'target'

# 1️⃣ Heart Disease Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x=target_col, hue=target_col, palette='coolwarm', legend=False)
plt.title("Heart Disease Distribution (0 = No, 1 = Yes)")
plt.tight_layout()
plt.savefig("1_Heart_Disease_Distribution.png")
plt.close()

# 2️⃣ Correlation Heatmap (Handle Non-Numeric Columns)
df_encoded = df.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = df_encoded[col].astype('category').cat.codes

plt.figure(figsize=(10, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("2_Correlation_Heatmap.png")
plt.close()

# 3️⃣ Gender vs Heart Disease
if 'Sex' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Sex', hue=target_col, data=df)
    plt.title("Gender vs Heart Disease")
    plt.tight_layout()
    plt.savefig("3_Gender_vs_Heart_Disease.png")
    plt.close()

# 4️⃣ Age Distribution
if 'Age' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Age'], kde=True, bins=20, color='purple')
    plt.title("Age Distribution of Patients")
    plt.tight_layout()
    plt.savefig("4_Age_Distribution.png")
    plt.close()

# 5️⃣ Cholesterol vs Heart Disease
if 'Cholesterol' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=target_col, y='Cholesterol', data=df)
    plt.title("Cholesterol Level vs Heart Disease")
    plt.tight_layout()
    plt.savefig("5_Cholesterol_vs_Heart_Disease.png")
    plt.close()

# 6️⃣ Data Splitting and Scaling
X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7️⃣ Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

# 8️⃣ SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

# Results
print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))

print("\nSupport Vector Machine Results")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

# 9️⃣ Confusion Matrix - Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_log)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr)
disp_lr.plot(cmap='Greens')
plt.title("Confusion Matrix - Logistic Regression")
plt.savefig("6_Confusion_Matrix_LogisticRegression.png")
plt.close()

# 🔟 Confusion Matrix - SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
disp_svm.plot(cmap='Blues')
plt.title("Confusion Matrix - SVM")
plt.savefig("7_Confusion_Matrix_SVM.png")
plt.close()

# 🔢 Model Comparison
acc_log = accuracy_score(y_test, y_pred_log)
acc_svm = accuracy_score(y_test, y_pred_svm)

print("\nModel Comparison:")
print(f"Logistic Regression Accuracy: {acc_log*100:.2f}%")
print(f"SVM Accuracy: {acc_svm*100:.2f}%")

plt.bar(['Logistic Regression', 'SVM'], [acc_log, acc_svm], color=['blue', 'orange'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("8_Model_Accuracy_Comparison.png")
plt.close()

print("\n✅ All graphs have been saved successfully in your project folder!")
