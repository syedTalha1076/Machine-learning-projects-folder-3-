import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc

# ---------------- Load Data ----------------
customer_data = pd.read_csv('shoping.csv')
data = customer_data.iloc[:, 3:5].values  # assuming columns 3 and 4 are features
print("Customer data (used for clustering):")
print(data)

# ---------------- Draw Dendrogram ----------------
plt.figure(figsize=(10,7))
plt.title("Customer Dendrogram")
dend = shc.dendrogram(shc.linkage(data, method='ward'))
plt.show()

# ---------------- Agglomerative Clustering ----------------
cluster = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels = cluster.fit_predict(data)
print("\nCluster labels for existing customers:")
print(labels)

# ---------------- Scatter Plot ----------------
plt.figure(figsize=(10,7))
plt.scatter(data[:,0], data[:,1], c=labels, cmap="rainbow")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Customer Clusters")
plt.show()

# ---------------- User Input for Prediction ----------------
print("\nPredict cluster for a new customer:")
feature1 = float(input("Enter Feature 1 (e.g., Age or Annual Income): "))
feature2 = float(input("Enter Feature 2 (e.g., Spending Score): "))

# Compute distances to existing cluster centroids
# Since AgglomerativeClustering doesn't have centroids, we approximate
# by computing the mean of each cluster
cluster_centers = []
for i in range(4):
    cluster_points = data[labels == i]
    cluster_centers.append(cluster_points.mean(axis=0))
cluster_centers = np.array(cluster_centers)

# Find nearest cluster
distances = np.linalg.norm(cluster_centers - np.array([feature1, feature2]), axis=1)
predicted_cluster = np.argmin(distances)

print(f"\nThe new customer belongs to Cluster {predicted_cluster}.")

# Simple logic for shopping likelihood (example)
if predicted_cluster in [0, 2]:  # suppose clusters 0 and 2 are high-value
    print("Prediction: This customer is likely to shop.")
else:
    print("Prediction: This customer is less likely to shop.")
