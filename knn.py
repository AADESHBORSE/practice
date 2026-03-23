import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 🔴 Replace dataset
df = pd.read_csv("YOUR_DATASET.csv")

# 🔴 Replace target
X = df.drop("TARGET_COLUMN", axis=1)
y = df["TARGET_COLUMN"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
