# --------------
# import packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

offers = pd.read_excel(path, sheet_name=0)

# Load Offers
transactions = pd.read_excel(path, sheet_name=1)

transactions["n"] = [1 for i in range(len(transactions))]

# Load Transactions


# Merge dataframes
df = offers.merge(transactions)
print(df.head())
# Look at the first 5 rows



# --------------
# Code starts here

# create pivot table
matrix = pd.pivot_table(df,index="Customer Last Name", columns="Offer #",
                values="n"    )

# replace missing values with 0
matrix.fillna(0, inplace=True)

# reindex pivot table
matrix.reset_index(inplace=True)

# display first 5 rows
matrix.head()

# Code ends here


# --------------
# import packages
from sklearn.cluster import KMeans

# Code starts here

# initialize KMeans object
KMeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)

# create 'cluster' column
matrix["cluster"] = KMeans.fit_predict(matrix[matrix.columns[1:]]) 
print(matrix.head())
# Code ends here


# --------------
# import packages
from sklearn.decomposition import PCA

# Code starts here

# initialize pca object with 2 components
pca = PCA(n_components=2, random_state=0)

# create 'x' and 'y' columns donoting observation locations in decomposed form
matrix["x"] = pca.fit_transform(matrix[matrix.columns[1:]])[:,0]
matrix["y"] = pca.fit_transform(matrix[matrix.columns[1:]])[:,1]
# dataframe to visualize clusters by customer names
clusters = matrix.iloc[:,0]
clusters = pd.concat([clusters, matrix.iloc[:,33:36]])
# visualize clusters
matrix.plot.scatter(x="x", y="y", c="cluster", colormap="viridis")
plt.show()
# Code ends here


# --------------
# Code starts here

# merge 'clusters' and 'transactions'
data = pd.merge(clusters, transactions)
print(data.head())
print('='*25)


# merge `data` and `offers`
data = pd.merge(offers, data)
print(data.head())
print('='*25)

# initialzie empty dictionary
champagne = {}

# iterate over every cluster
for val in data.cluster.unique():
    # observation falls in that cluster
    new_df = data[data.cluster == val]
    # sort cluster according to type of 'Varietal'
    counts = new_df['Varietal'].value_counts(ascending=False)
    # check if 'Champagne' is ordered mostly
    if counts.index[0] == 'Champagne':
        # add it to 'champagne'
        champagne[val] = (counts[0])

# get cluster with maximum orders of 'Champagne' 
cluster_champagne = max(champagne, key=champagne.get)

# print out cluster number
print(cluster_champagne)
#     print('='*50)


# --------------
# Code starts here

# empty dictionary
discount = {}
# iterate over cluster numbers
for i in range(5):
    # dataframe for every cluster
    new_df = data[data.cluster == i]
    # average discount for cluster
    counts = np.sum(new_df["Discount (%)"])/len(new_df)
    # adding cluster number as key and average discount as value 
    discount[i] = counts

# cluster with maximum average discount
cluster_discount = [i for i in range(5) if discount[i] == max(discount.values())][0]
print(cluster_discount)
# Code ends here


