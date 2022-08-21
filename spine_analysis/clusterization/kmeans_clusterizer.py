from typing import Union, Callable

import numpy as np
from sklearn.cluster import KMeans

from spine_analysis.clusterization.clusterizer_core import SKLearnSpineClusterizer


class KMeansSpineClusterizer(SKLearnSpineClusterizer):
    def __init__(self, num_of_clusters: int, pca_dim: int = -1, metric: Union[str, Callable] = "euclidean"):
        super().__init__(pca_dim=pca_dim, metric=metric)
        self.num_of_clusters = num_of_clusters

    def _sklearn_fit(self, data: np.array) -> object:
        return KMeans(n_clusters=self.num_of_clusters, random_state=0).fit(data)
