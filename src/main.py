from kmeans import *


x = np.array([[0, 2], [3, 4], [5, 2], [7, 1], [0, 3], [3, 2], [1, 6]])

kmeans = Kmeans(3, 100, x) 

print(kmeans.centroids)
print(kmeans.x)

kmeans.fit()

print(kmeans.error)
