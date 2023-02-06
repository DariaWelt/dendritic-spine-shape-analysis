import random
from abc import abstractmethod, ABC
from typing import List, Union, Callable

import numpy as np
from sklearn.cluster import SpectralClustering

from spine_analysis.clusterization import SpineClusterizer


class KernelSpineClusterizer(SpineClusterizer, ABC):
    def _fit(self, data: np.array, names: List[str]) -> object:
        #kernel_matrix = [[self.metric(obj1, obj2) for obj2 in data] for obj1 in data]
        self._data = data
        if isinstance(self.metric, str):
            self.metric = self.kernilaze(self.metric)
        self._labels = self._kernel_fit(data, self.metric, 'random')

        for cluster_index in set(self._labels):
            if cluster_index == -1:
                continue
            names_array = np.array(names)
            cluster_names = names_array[self._labels == cluster_index]
            self.grouping.groups[str(cluster_index + 1)] = set(cluster_names)
        return

    @staticmethod
    def kernilaze(metric: str):
        if metric == 'euclidean':
            return lambda x, y: np.linalg.norm(np.array(x)-np.array(y))
        else:
            raise ValueError(f'metric {metric} is not supported')

    @abstractmethod
    def _kernel_fit(self, data: np.ndarray, kernel: Callable, initialization: Union[str, List]) -> List[int]:
        pass


class KmeansKernelSpineClusterizer(KernelSpineClusterizer):
    _num_of_clusters: int
    _D: np.ndarray

    def __init__(self, num_of_clusters: int, pca_dim: int = -1, metric: Union[str, Callable] = "euclidean"):
        super().__init__(pca_dim=pca_dim, metric=metric)
        self._num_of_clusters = num_of_clusters

    def _kernel_fit(self, data: np.ndarray, kernel: Callable, initialization: Union[str, List] = 'random') -> object:
        self.kernel = kernel
        self._init_clusters(initialization, data)
        self._update_labels_loop()
        self._labels = self._labels.tolist()
        return self._labels

    def _init_clusters(self, initialization: Union[str, List], data: np.ndarray):
        if type(initialization) == list:
            if len(initialization) != self._num_of_clusters:
                raise ValueError('Size of centroids indices list is not equal to clusters number')
            if len(set(initialization)) != len(initialization):
                raise ValueError('Centroids indices in list is not unique')
            # if elem < 0 or > len(data) raise
            centroids = initialization
        elif initialization == 'random':
            centroids = self.get_random_centroids(data, self._num_of_clusters)
        else:
            raise ValueError('Unknown initialization key')
        self._K = np.array([[self.kernel(x_i, x_j) for x_i in data] for x_j in data])
        self._D = self._K[:, centroids]
        self._labels = np.argmin(self._D, axis=1)

    def _update_labels_loop(self, max_iter_num: int = 100, tolerance: float = 0.01):
        iter_num = 1
        while iter_num < max_iter_num:
            new_distances = np.array([[self.calculate_dist_clusters(x_ind, c_i)
                                       for c_i in range(self._num_of_clusters)]
                                      for x_ind in range(len(self._K))])
            new_labels = np.argmin(new_distances, axis=1)
            interia_sub = abs(sum([self._D[i, l] for i, l in enumerate(self._labels)]) -
                              sum([new_distances[i, l] for i, l in enumerate(new_labels)]))
            if interia_sub <= tolerance:
                return
            self._D = new_distances
            self._labels = new_labels
            iter_num += 1

    def calculate_dist_clusters(self, x_ind: int, c_ind: int):
        cluster: np.ndarray = np.where(self._labels == c_ind)
        return self._K[x_ind, x_ind] - \
               2 / len(cluster) * np.sum(self._K[x_ind, cluster]) + \
               2 / (len(cluster)*len(cluster)) * np.sum(self._K[cluster, cluster])

    @staticmethod
    def get_random_centroids(data: np.ndarray, num_clusters: int) -> List[int]:
        return random.sample(list(range(len(data))), num_clusters)


class SpectralSpineClusterizer(KernelSpineClusterizer):
    def __init__(self, num_of_clusters: int, pca_dim: int = -1, metric: Union[str, Callable] = "euclidean"):
        super().__init__(pca_dim=pca_dim, metric=metric)
        self._num_of_clusters = num_of_clusters

    def _kernel_fit(self, data: np.ndarray, kernel: Callable, initialization: Union[str, List] = None) -> List[int]:
        clusterizer = SpectralClustering(self._num_of_clusters, affinity='precomputed')
        kernel_matrix = np.array([[kernel(x_i, x_j) for x_i in data] for x_j in data])
        clustering = clusterizer.fit(kernel_matrix)
        return clustering.labels_
