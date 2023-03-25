from typing import Union, Callable

import numpy as np
from sklearn.cluster import KMeans

from spine_analysis.clusterization.clusterizer_core import SKLearnSpineClusterizer


class KMeansSpineClusterizer(SKLearnSpineClusterizer):
    _num_of_clusters: int

    def __init__(self, num_of_clusters: int, dim: int = -1, metric: Union[str, Callable] = "euclidean", reduction: str = ""):
        super().__init__(dim=dim, metric=metric, reduction=reduction)
        self._num_of_clusters = num_of_clusters

    def _sklearn_fit(self, data: np.array) -> object:
        self._clusterizer = KMeans(n_clusters=self._num_of_clusters, random_state=0)
        return self._clusterizer.fit(data)
