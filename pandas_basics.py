import pandas as pd
import numpy as np

# 🔴 REPLACE WITH YOUR FILE PATH
data = pd.read_csv("YOUR_DATASET.csv")

print(data.head())
print(data.tail())
print(data.info())

print(data.shape)
print(data.describe())

# Null handling
print(data.isnull().sum())

# Sorting
print(data.sort_values(by="COLUMN_NAME"))

# Filtering
filtered = data.query("COLUMN_NAME > VALUE")
print(filtered)

# Selecting
print(data.loc[:5, ["COLUMN1", "COLUMN2"]])
print(data.iloc[:5, :3])

# GroupBy
print(data.groupby("COLUMN_NAME").mean())

# Add column
data["new_col"] = np.random.randn(len(data))
print(data.head())
