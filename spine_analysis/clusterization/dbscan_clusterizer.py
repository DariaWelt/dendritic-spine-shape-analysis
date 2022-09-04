from typing import Set, Union, Callable

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from spine_analysis.clusterization.clusterizer_core import SKLearnSpineClusterizer


class DBSCANSpineClusterizer(SKLearnSpineClusterizer):
    eps: float
    min_samples: int
    #num_of_noise: int

    def __init__(self, eps: float = 0.5, min_samples: int = 2,
                 metric: Union[str, Callable] = "euclidean", pca_dim: int = -1):
        super().__init__(metric=metric, pca_dim=pca_dim)
        self.metric = metric
        self.min_samples = min_samples
        self.eps = eps

    # def score(self) -> float:
    #     # TODO: change nan to something sensical
    #     # if self.num_of_noise / self.sample_size > 0.15 or self.num_of_clusters < 2 or self.sample_size - self.num_of_noise - 1 < self.num_of_clusters:
    #     if self.num_of_clusters < 2 or self.sample_size - self.num_of_noise - 1 < self.num_of_clusters:
    #         return float("nan")
    #     labels = self.get_labels()
    #     indices_to_delete = np.argwhere(np.asarray(labels) == -1)
    #     filtered_data = np.delete(self._data, indices_to_delete, 0)
    #     filtered_labels = np.delete(labels, indices_to_delete, 0)
    #     return silhouette_score(filtered_data, filtered_labels, metric=self.metric)

    def _sklearn_fit(self, data: np.array) -> object:
        self._clusterizer = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric)
        clusterized = self._clusterizer.fit(data)
        return clusterized
