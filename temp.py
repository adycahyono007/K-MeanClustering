from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

Data = {'x': [1,3,4,5,1,4,1,2],
        'y': [3,3,3,3,2,2,1,1],
        'z': [3,3,3,3,2,2,1,1]
       }


df = DataFrame(Data,columns=['x','y','z'])
print(df)

kmeans = KMeans(n_clusters=3).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(df['x'], df['y'], df['z'], c= kmeans.labels_.astype(float), s=30, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)