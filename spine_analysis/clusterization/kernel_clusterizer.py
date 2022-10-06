from abc import abstractmethod, ABC
from typing import List, Union, Callable

import numpy as np
from tslearn.clustering import KernelKMeans

from spine_analysis.clusterization import SpineClusterizer


class KernelSpineClusterizer(SpineClusterizer, ABC):
    def _fit(self, data: np.array, names: List[str]) -> object:
        #kernel_matrix = [[self.metric(obj1, obj2) for obj2 in data] for obj1 in data]
        self._data = data
        self._labels = self._kernel_fit(data, self.metric)

        for cluster_index in set(self._labels):
            if cluster_index == -1:
                continue
            names_array = np.array(names)
            cluster_names = names_array[self._labels == cluster_index]
            self.grouping.groups[str(cluster_index + 1)] = set(cluster_names)
        return

    @abstractmethod
    def _kernel_fit(self, data: np.ndarray, kernel: Callable) -> List[int]:
        pass


class KmeansKernelSpineClusterizer(KernelSpineClusterizer):
    def __init__(self, num_of_clusters: int, pca_dim: int = -1, metric: Union[str, Callable] = "euclidean"):
        super().__init__(pca_dim=pca_dim, metric=metric)
        self._num_of_clusters = num_of_clusters

    def _kernel_fit(self, data: np.ndarray, kernel: Callable) -> object:
        clusterizer = KernelKMeans(n_clusters=self._num_of_clusters, kernel=kernel, random_state=0)
        return clusterizer.fit_predict(data)
