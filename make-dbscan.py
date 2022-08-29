from cProfile import label
from sre_constants import NOT_LITERAL_UNI_IGNORE
from sklearn.datasets import make_blobs, make_moons, make_circles
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


X, y = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=0)

# Plotting data
plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], c=y, marker=".")
plt.title("Ground - Datasets Truth (make blobs)")

# Erstelle DBSCAN
dbscan_clustering = DBSCAN(eps=.3, min_samples=5).fit(X)
cluster_labels = dbscan_clustering.fit_predict(X)
print(f"Labels: {np.unique(cluster_labels)}")

silhouette_avg = silhouette_score(X, cluster_labels)

# print(f"K-Means labels Attribute: \n{kmeans.labels_}")
# print(f"K-Means Cluster Center Attribute: \n{kmeans.cluster_centers_}")

plt.figure(2)
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, marker=".")
plt.title("DBSCAN Clustering")

# Add Axes Labels
plt.xlabel("x1 axis")
plt.ylabel("x2 axis")


# ax3 = plt.figure(3)
# plt.plot(silhouette_avg, X[:,0])

# ax3.set_yticks([])  # Clear the yaxis labels / ticks
# ax3.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.show()
