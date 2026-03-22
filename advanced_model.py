import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 🔴 Replace dataset
df = pd.read_csv("YOUR_DATASET.csv")

# 🔴 Replace target column
X = df.drop("TARGET_COLUMN", axis=1)
y = df["TARGET_COLUMN"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ---------------------------
# DECISION TREE
# ---------------------------
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

dt_pred = dt.predict(x_test)
dt_acc = accuracy_score(y_test, dt_pred)

# ---------------------------
# KNN
# ---------------------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

knn_pred = knn.predict(x_test)
knn_acc = accuracy_score(y_test, knn_pred)

# ---------------------------
# COMPARISON
# ---------------------------
print("Decision Tree Accuracy:", dt_acc)
print("KNN Accuracy:", knn_acc)

if dt_acc > knn_acc:
    print("Decision Tree is better")
else:
    print("KNN is better")
