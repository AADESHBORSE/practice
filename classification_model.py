import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 🔴 Replace dataset
data = pd.read_csv("YOUR_DATASET.csv")

# 🔴 Replace target
X = data.drop("TARGET_COLUMN", axis=1)
y = data["TARGET_COLUMN"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

print(predictions)
