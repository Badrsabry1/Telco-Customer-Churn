# ============================================
# TELCO CUSTOMER CHURN - SIMPLE ML PROJECT
# ============================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve
)

import joblib


# ============================================
# 2. Load Dataset
# ============================================

file_path = r"C:\Users\Badr\PycharmProjects\PythonProject13\WA_Fn-UseC_-Telco-Customer-Churn.xls"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ============================================
# 3. Basic Information
# ============================================

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# ============================================
# 4. Clean the Data
# ============================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


df["TotalCharges"] = df["TotalCharges"].fillna(0)


# ============================================
# 5. Convert Target Column
# ============================================

# Churn:
# No  -> 0
# Yes -> 1

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

print("\nChurn distribution:")
print(df["Churn"].value_counts())

print("\nChurn percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)


# ============================================
# 6. Remove Customer ID
# ============================================

# customerID is only an identifier.
# It doesn't help the model predict churn.

df = df.drop("customerID", axis=1)


# ============================================
# 7. Separate X and y
# ============================================

X = df.drop("Churn", axis=1)
y = df["Churn"]


# ============================================
# 8. Find Numerical and Categorical Columns
# ============================================

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)


# ============================================
# 9. Split Data
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================
# 10. Preprocessing
# ============================================

# Numerical columns:
# StandardScaler

# Categorical columns:
# OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_columns
        )
    ]
)


# ============================================
# 11. Create Model
# ============================================

model = LogisticRegression(
    max_iter=1000
)


# ============================================
# 12. Create Pipeline
# ============================================

pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", model)
    ]
)


# ============================================
# 13. Train Model
# ============================================

print("\nTraining model...")

pipeline.fit(
    X_train,
    y_train
)

print("Model trained successfully!")


# ============================================
# 14. Make Predictions
# ============================================

y_pred = pipeline.predict(X_test)

# Probability of Churn = Yes
y_probability = pipeline.predict_proba(X_test)[:, 1]


# ============================================
# 15. Model Evaluation
# ============================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n================================")
print("MODEL RESULTS")
print("================================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))


# ============================================
# 16. Classification Report
# ============================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================
# 17. Confusion Matrix
# ============================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot()

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


# ============================================
# 18. ROC Curve
# ============================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.tight_layout()
plt.show()


# ============================================
# 19. Churn Probability Distribution
# ============================================

plt.figure(figsize=(7, 5))

plt.hist(
    y_probability[y_test.values == 0],
    bins=20,
    alpha=0.6,
    label="No Churn"
)

plt.hist(
    y_probability[y_test.values == 1],
    bins=20,
    alpha=0.6,
    label="Churn"
)

plt.xlabel("Predicted Churn Probability")
plt.ylabel("Number of Customers")

plt.title("Predicted Churn Probability")

plt.legend()

plt.tight_layout()
plt.show()


# ============================================
# 20. Feature Importance
# ============================================

# Get feature names after One-Hot Encoding

feature_names = pipeline.named_steps[
    "preprocessing"
].get_feature_names_out()

# Get Logistic Regression coefficients

coefficients = pipeline.named_steps[
    "model"
].coef_[0]

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

# Add absolute coefficient
feature_importance["Absolute"] = (
    feature_importance["Coefficient"].abs()
)

# Sort by importance
feature_importance = feature_importance.sort_values(
    "Absolute",
    ascending=False
)

print("\nTop 20 important features:")
print(
    feature_importance[
        ["Feature", "Coefficient"]
    ].head(20)
)


# ============================================
# 21. Plot Feature Importance
# ============================================

top_features = feature_importance.head(15)

plt.figure(figsize=(9, 6))

plt.barh(
    top_features["Feature"],
    top_features["Coefficient"]
)

plt.xlabel("Coefficient")

plt.ylabel("Feature")

plt.title("Top Features Affecting Churn")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


# ============================================
# 22. Create Prediction Table
# ============================================

results = X_test.copy()

results["Actual_Churn"] = y_test.values

results["Predicted_Churn"] = y_pred

results["Churn_Probability"] = y_probability

results["Churn_Probability"] = (
    results["Churn_Probability"].round(3)
)

print("\nPrediction examples:")
print(results.head(10))


# ============================================
# 23. Save Predictions
# ============================================

results.to_csv(
    "churn_predictions.csv",
    index=False
)

print(
    "\nPredictions saved to: churn_predictions.csv"
)


# ============================================
# 24. Save Model
# ============================================

joblib.dump(
    pipeline,
    "churn_model.pkl"
)

print(
    "Model saved to: churn_model.pkl"
)


# ============================================
# 25. Final Message
# ============================================

print("\n================================")
print("PROJECT FINISHED SUCCESSFULLY")
print("================================")

print("Model: Logistic Regression")
print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))