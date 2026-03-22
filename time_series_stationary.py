import pandas as pd
from statsmodels.tsa.stattools import adfuller

# 🔴 Replace dataset
df = pd.read_csv("YOUR_DATASET.csv")

series = df["COLUMN_NAME"]

result = adfuller(series)

print("ADF Statistic:", result[0])
print("p-value:", result[1])
