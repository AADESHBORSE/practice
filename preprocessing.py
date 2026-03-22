import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 🔴 Replace dataset path
df = pd.read_csv("YOUR_DATASET.csv")

# ---------------------------
# 1. HANDLE MISSING VALUES
# ---------------------------
# Fill numeric columns with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Fill categorical columns with mode
for col in df.select_dtypes(include='object').columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("Missing values handled")

# ---------------------------
# 2. ENCODING (Categorical → Numeric)
# ---------------------------
le = LabelEncoder()

# 🔴 Replace column names if needed
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

print("Encoding done")

# ---------------------------
# 3. FEATURE SCALING
# ---------------------------
scaler = StandardScaler()

# 🔴 Replace target column
X = df.drop("TARGET_COLUMN", axis=1)

X_scaled = scaler.fit_transform(X)

print("Feature scaling done")

print(X_scaled[:5])
