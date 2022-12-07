from typing import Callable, Union, List

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from spine_analysis.clusterization.kernel_clusterizer import KernelSpineClusterizer


class HierarchicalSpineClusterizer(KernelSpineClusterizer):
    def __init__(self, num_of_clusters: int, pca_dim: int = -1, metric: Union[str, Callable] = "euclidean"):
        super().__init__(pca_dim=pca_dim, metric=metric)
        self._num_of_clusters = num_of_clusters

    def _kernel_fit(self, data: np.ndarray, kernel: Callable, initialization: Union[str, List]) -> List[int]:
        clusterizer = AgglomerativeClustering(self._num_of_clusters, affinity='precomputed', linkage='single')
        kernel_matrix = np.array([[kernel(x_i, x_j) for x_i in data] for x_j in data])
        clustering = clusterizer.fit(kernel_matrix)
        return clustering.labels_
