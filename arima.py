import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# 🔴 Replace dataset
df = pd.read_csv("YOUR_DATASET.csv")

# 🔴 Replace column name (time series column)
series = df["COLUMN_NAME"]

# ---------------------------
# 1. Plot original series
# ---------------------------
plt.plot(series)
plt.title("Original Time Series")
plt.show()

# ---------------------------
# 2. Check Stationarity (ADF Test)
# ---------------------------
result = adfuller(series)

print("ADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] > 0.05:
    print("Data is NOT stationary → need differencing")
else:
    print("Data is stationary")

# ---------------------------
# 3. Differencing (if needed)
# ---------------------------
series_diff = series.diff().dropna()

plt.plot(series_diff)
plt.title("Differenced Series")
plt.show()

# ---------------------------
# 4. Apply ARIMA
# (p, d, q) → 🔴 Adjust if needed
# ---------------------------
model = ARIMA(series, order=(1,1,1))
model_fit = model.fit()

print(model_fit.summary())

# ---------------------------
# 5. Forecast
# ---------------------------
forecast = model_fit.forecast(steps=5)

print("Forecast:\n", forecast)

# ---------------------------
# 6. Plot forecast
# ---------------------------
plt.plot(series, label="Original")
plt.plot(range(len(series), len(series)+5), forecast, label="Forecast", color='red')
plt.legend()
plt.show()
