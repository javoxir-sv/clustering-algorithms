import random
import numpy as np
from queue import Queue
from sklearn.neighbors import KDTree


class myKMeans:
  def __init__(self, n_clusters=5, max_iter=200):
    self.n_clusters = n_clusters
    self.max_iter = max_iter
    self.cluster_centers_ = {}

  def fit(self, data):
    data = np.array(data)
    np.random.shuffle(data)
    self.cluster_centers_ = [random.choice(data)]

    for _ in range(self.n_clusters-1):
      distances = np.sum([np.sqrt(np.sum((centroid - data)**2, axis=1)) for centroid in self.cluster_centers_], axis=0)
      distances /= np.sum(distances)
      new_centroid_idx = np.random.choice(range(len(data)), size=1, p=distances)[0]
      self.cluster_centers_.append(data[new_centroid_idx])

    for i in range(self.max_iter):
      self.classifications = {}
      for i in range(self.n_clusters):
        self.classifications[i] = []

      for feature in data:
        distances = [np.linalg.norm(feature-centroid) for centroid in self.cluster_centers_]
        classification = distances.index(min(distances))
        self.classifications[classification].append(feature)

      prev_centroids = list(self.cluster_centers_)
      for classification in self.classifications:
        if len(self.cluster_centers_[classification]) > 0 and len(self.classifications[classification]) > 0:
          self.cluster_centers_[classification] = np.mean(self.classifications[classification], axis=0)
        else:
          self.cluster_centers_[classification] = prev_centroids[classification]

      optimized = True
      for c in range(len(self.cluster_centers_)):
        if not np.array_equal(self.cluster_centers_[c], prev_centroids[c]):
          optimized = False
      if optimized:
        self.cluster_centers_ = np.array(self.cluster_centers_)
        print(f"Converged at iteration {i}")
        break

  def predict(self, data):
    data = np.array(data)
    self.labels_ = []
    for x in data:
      distances = [np.linalg.norm(x-centroid) for centroid in self.cluster_centers_]
      classification = distances.index(min(distances))
      self.labels_.append(classification)
    return np.asarray(self.labels_)


class MyDBSCAN:
  def __init__(self, eps=1, min_samples=5):
    self.eps = eps
    self.min_samples = min_samples
    self.cluster_count = 0
    self.noise = 0

  def getNeighbors(self, x):
    x = x.reshape(1, -1)
    return self.tree.query_radius(x, r=self.eps)[0]

  def fit(self, X):
    self.tree = KDTree(X)
    X = np.array(X)
    self.data = np.append(X, np.full((len(X), ), -1).reshape(-1, 1), axis=1)
    neighbor_check = np.full((len(self.data)), False)

    for idx in range(len(self.data)):
      if self.data[idx, 2] != -1:
        continue
      point = self.data[idx, :2]
      neighbors = self.getNeighbors(point)
      if len(neighbors) < self.min_samples:
        self.data[idx, 2] = self.noise
        continue

      self.cluster_count += 1
      self.data[idx, 2] = self.cluster_count
      found_neighbors = list(neighbors)
      que = Queue()
      for n in found_neighbors:
        neighbor_check[n] = True
        que.put(n)

      while que.empty() == False:
        current = que.get()
        if self.data[current, 2] == 0:
          self.data[current, 2] = self.cluster_count
        if self.data[current, 2] != -1:
          continue

        self.data[current, 2] = self.cluster_count
        neighbors2 = self.getNeighbors(self.data[current, :2])
        if len(neighbors2) >= self.min_samples:
          for x in neighbors2:
            if neighbor_check[x] == False:
              que.put(x)
              found_neighbors.append(x)
    self.labels_ = np.array(self.data[:, 2])

  def fit_predict(self, X):
    self.fit(X)
    return self.labels_

