import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 🔴 Replace dataset
df = pd.read_csv("YOUR_DATASET.csv")

# 🔴 Replace column
series = df["COLUMN_NAME"]

plt.plot(series)
plt.show()

result = seasonal_decompose(series, model='multiplicative', period=12)
result.plot()
plt.show()
