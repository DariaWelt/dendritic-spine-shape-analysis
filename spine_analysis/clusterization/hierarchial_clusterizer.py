from typing import Callable, Union, List

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from spine_analysis.clusterization.kernel_clusterizer import KernelSpineClusterizer


class HierarchicalSpineClusterizer(KernelSpineClusterizer):
    def __init__(self, num_of_clusters: int, dim: int = -1, metric: Union[str, Callable] = "euclidean", reduction: str = ""):
        super().__init__(dim=dim, metric=metric, reduction=reduction)
        self._num_of_clusters = num_of_clusters

    def _kernel_fit(self, data: np.ndarray, kernel: Callable, initialization: Union[str, List] = 'random') -> List[int]:
        clusterizer = AgglomerativeClustering(self._num_of_clusters, metric='precomputed', linkage='complete')
        kernel_matrix = np.array([[kernel(x_i, x_j) for x_i in data] for x_j in data])
        clustering = clusterizer.fit(kernel_matrix)
        return clustering.labels_
