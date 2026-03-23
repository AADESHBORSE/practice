import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 🔴 Replace dataset
df = pd.read_csv("YOUR_DATASET.csv")

# 🔴 Replace feature columns (important)
X = df[["COLUMN1", "COLUMN2"]]  

# ---------------------------
# 1. Elbow Method (find best K)
# ---------------------------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# ---------------------------
# 2. Apply K-Means
# 🔴 Change K based on elbow graph
# ---------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# ---------------------------
# 3. Add cluster labels to dataset
# ---------------------------
df["Cluster"] = y_kmeans

print(df.head())

# ---------------------------
# 4. Visualization
# ---------------------------
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_kmeans)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker='X'
)
plt.title("K-Means Clustering")
plt.show()
