import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import euclidean
from itertools import combinations
import matplotlib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

data = pd.read_excel("Walmart Sales.xlsx")

# Some basic info
print(data.head())
print("Data points:", data.shape[0])
print("Features:", data.shape[1])
print("Duplicates:", data.duplicated().sum())
data = data.drop_duplicates()
print("Missing values:", data.isna().sum().sum())
print("Single valued columns:", data.columns[data.nunique() == 1])

# Summary statistics of the dataset
data.describe()
data.info()

print("Quantity for each: ", data['Quantity'].value_counts())
print("Number of days: ", data['Purchase_Date'].nunique())
print("Number of products: ", data['Product_Name'].nunique())
print("Number of customers:", data['CustomerID'].nunique())

# RFM clustering analysis
# Recency
recency_df = data.groupby(by='CustomerID', as_index=False)['Purchase_Date'].max()
recency_df.columns = ['customer_ID', 'Last_purshace_date']
max_date = recency_df.Last_purshace_date.max()
recency_df['Recency'] = recency_df['Last_purshace_date'].apply(lambda x: (max_date - x).days)
recency_df.head(7)

# Frequency
copy_data = data
copy_data.drop_duplicates(subset=['Product_Name', 'CustomerID'], keep="first", inplace=True)

frequency_df = copy_data.groupby(by=['CustomerID'], as_index=False)['Product_Name'].count()
frequency_df.columns = ['customer_ID', 'Frequency']
frequency_df.head(7)

# Monetary
data['Cost'] = data['Quantity'] * data['UnitPrice']
monetary_df = data.groupby(by='CustomerID', as_index=False).agg({'Cost': 'sum'})
monetary_df.columns = ['customer_ID', 'Monetary']
monetary_df.head(7)

comb1 = recency_df.merge(frequency_df, on='customer_ID')
final_RFM = comb1.merge(monetary_df, on='customer_ID')
final_RFM.head(7)

# Ranking RFM  scores
final_RFM['Recency_rank'] = final_RFM['Recency'].rank(ascending=False)
final_RFM['Frequency_rank'] = final_RFM['Frequency'].rank(ascending=True)
final_RFM['Monetary_rank'] = final_RFM['Monetary'].rank(ascending=True)
final_RFM.head(7)

# Normalization of RFM ranked scores
final_RFM['Recency_rank_norm'] = (final_RFM['Recency_rank'] / final_RFM['Recency_rank'].max()) * 100
final_RFM['Frequency_rank_norm'] = (final_RFM['Frequency_rank'] / final_RFM['Frequency_rank'].max()) * 100
final_RFM['Monetary_rank_norm'] = (final_RFM['Monetary_rank'] / final_RFM['Monetary_rank'].max()) * 100
final_RFM.head(7)

# Calculate RFM score
final_RFM['RFM_Score'] = 0.2 * final_RFM['Recency_rank_norm'] + \
                         0.2 * final_RFM['Frequency_rank_norm'] + \
                         0.6 * final_RFM['Monetary_rank_norm']
final_RFM = final_RFM.round(0)
final_RFM.head()

# RFM score < 25 then ‘Leaving customers’
# RFM score >= 25 and< 50 then ‘Risky customers’
# RFM score >= 50 and< 75 then ‘Potential Loyalists’
# RFM score >= 75 then ‘Champions’

final_RFM["Customer_segment"] = np.where((final_RFM['RFM_Score'] >= 25) & (final_RFM['RFM_Score'] < 50),
                                         "Risky Customers",
                                         np.where(final_RFM['RFM_Score'] >= 75, "Champions",
                                                  np.where(final_RFM['RFM_Score'] < 25, "Leaving",
                                                           "Potential Loyalists")))
final_RFM.head(10)


# Visualization
data.rename(columns={'CustomerID': 'customer_ID'}, inplace=True)
data_rfm_merged = data.merge(final_RFM, on='customer_ID')
data_rfm_merged.groupby('Customer_segment')[['customer_ID']].count()

plt.figure(figsize=(6, 4))
pd.crosstab(data_rfm_merged['Customer_segment'], data_rfm_merged['Gender']).plot(kind='bar', stacked=True);
plt.xticks(rotation=360, ha='right', fontsize=5);
plt.yticks(fontsize=5);
plt.xlabel('Customer Segment', fontsize=7)
plt.ylabel('Count', fontsize=7)
plt.title('Customer Segment by Gender', fontsize=12)
plt.show()

pd.crosstab(data_rfm_merged['Customer_segment'], data_rfm_merged['Gender'])



# K-Means clustering analysis
data_k_means = data_rfm_merged[
    ['Recency_rank_norm', 'Frequency_rank_norm', 'Monetary_rank_norm', 'Age', 'MonthlyIncome']]

model_kmeans = KMeans(n_clusters=4, init='k-means++', random_state=50)
y_means = model_kmeans.fit_predict(data_k_means)
data_rfm_merged["cluster"] = y_means
data_rfm_merged["cluster"].value_counts()

# Using Elbow and Silhouette_score to find the optimal number of clusters
wcss = []
for i in range(1, 15):
    k_means = KMeans(n_clusters=i, init="k-means++")
    k_means.fit(data_k_means)
    wcss.append(k_means.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(range(1, 15), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# **From the above we can conclude that the optimal number of clusters is k=2**


# Silhouette Score
model_kmeans_2 = KMeans(n_clusters=2, init='k-means++', random_state=50)
y_means = model_kmeans_2.fit_predict(data_k_means)
data_rfm_merged["cluster"] = y_means
data_rfm_merged['cluster'].value_counts()

silhouette_score(data_k_means, y_means)
# 0.6 => quite close to 1 and a good clustering


# Group the data by 'Cluster' and calculate the average monthly salary for each group
average_monthlysalary_by_cluster = data_rfm_merged.groupby('cluster')['MonthlyIncome'].mean().reset_index()
print(average_monthlysalary_by_cluster)

# Size of each cluster
print(data_rfm_merged.groupby('cluster')['customer_ID'].count())


# a relatively balanced distribution of data points across the clusters,
# with Cluster 0 having slightly more data points than Cluster 1,
# is a positive indication, suggesting that the K-Means algorithm has generally
# achieved a reasonable balance in assigning data points to clusters.

# Dunn index calculation for characterizing clusters
def dunn_index(X, labels):
    num_clusters = len(np.unique(labels))
    cluster_centers = []
    cluster_diameters = np.zeros(num_clusters)

    for cluster_id in range(num_clusters):
        cluster_points = X[labels == cluster_id]
        cluster_centers.append(np.mean(cluster_points, axis=0))
        max_diameter = 0

        # Calculate the maximum pairwise distance within each cluster
        for point1, point2 in combinations(cluster_points, 2):
            distance = euclidean(point1, point2)
            if distance > max_diameter:
                max_diameter = distance

        cluster_diameters[cluster_id] = max_diameter

    min_intercluster_distances = np.full((num_clusters, num_clusters), np.inf)

    # Calculate the minimum pairwise distance between cluster centers
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            distance = euclidean(cluster_centers[i], cluster_centers[j])
            min_intercluster_distances[i, j] = distance
            min_intercluster_distances[j, i] = distance

    min_intercluster_distance = np.min(min_intercluster_distances)

    return min_intercluster_distance / np.max(cluster_diameters)


X = data_rfm_merged[['Recency_rank_norm', 'Frequency_rank_norm', 'Monetary_rank_norm', 'Age', 'MonthlyIncome']].values

# Perform K-Means clustering
kmeans = KMeans(n_clusters=2)
labels = kmeans.fit_predict(X)

# Calculate the Dunn Index
dunn_value = dunn_index(X, labels)
print(f"Dunn Index: {dunn_value}")

# Dunn index evaluates the clustering quality.
# From its value, we can conclude that our clusters are well-separated,
# i.e., the distance btw the centers of different clusters is relatively
# large compared to the diameter of the individual clusters.
# The inter-cluster distance is relatively small within each cluster,
# indicating that the data points within each cluster are close to each other.
