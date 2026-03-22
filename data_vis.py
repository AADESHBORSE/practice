import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔴 Replace dataset
data = pd.read_csv("YOUR_DATASET.csv")

# Histogram
data.hist()
plt.show()

# Heatmap
sns.heatmap(data.corr(), annot=True)
plt.show()

# Boxplot
sns.boxplot(data=data)
plt.show()

# Pairplot
sns.pairplot(data)
plt.show()
