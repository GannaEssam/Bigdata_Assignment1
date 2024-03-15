import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from dpre import preprocessed_df
from eda import save_insights
from vis import plot_correlation

numerical_columns = ['Age', 'Fare']
X = preprocessed_df[numerical_columns]

# Initialize KMeans model with k=3
kmeans = KMeans(n_clusters=3)

# Fit KMeans model to the data
kmeans.fit(X)

# Get cluster labels
cluster_labels = kmeans.labels_

# Count number of records in each cluster
cluster_counts = np.bincount(cluster_labels)

# Save number of records in each cluster to a text file
with open('k.txt', 'w') as file:
    for cluster, count in enumerate(cluster_counts):
        file.write(f'Cluster {cluster}: {count}\n')