import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# 🔴 Replace dataset
df = pd.read_csv("YOUR_DATASET.csv")

# 🔴 Replace target column
X = df.drop("TARGET_COLUMN", axis=1)
y = df["TARGET_COLUMN"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ---------------------------
# CLASSIFICATION MODEL
# ---------------------------
clf = LogisticRegression()
clf.fit(x_train, y_train)

y_pred_class = clf.predict(x_test)

print("----- Classification Metrics -----")
print("Accuracy:", accuracy_score(y_test, y_pred_class))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_class))
print("Report:\n", classification_report(y_test, y_pred_class))

# ---------------------------
# REGRESSION MODEL
# ---------------------------
reg = LinearRegression()
reg.fit(x_train, y_train)

y_pred_reg = reg.predict(x_test)

print("\n----- Regression Metrics -----")
print("MAE:", mean_absolute_error(y_test, y_pred_reg))
print("MSE:", mean_squared_error(y_test, y_pred_reg))
print("R2 Score:", r2_score(y_test, y_pred_reg))
