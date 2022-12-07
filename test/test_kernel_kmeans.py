from typing import Callable, Optional, Any, List, Union

import numpy as np
import pytest

from spine_analysis.clusterization import KmeansKernelSpineClusterizer
from test.test_base import TestCaseBase


class ClusterizationCase(TestCaseBase):
    X: np.ndarray
    clusters_num: int
    metric: Callable

    def __init__(self, name: str, data: np.ndarray, gt: np.ndarray, clusters_num: int,
                 metric: Optional[Callable] = None, centroids: Union[str, List] = 'random'):
        super(ClusterizationCase, self).__init__(name, gt)
        self.X = data
        self.clusters_num = clusters_num
        self.centroids = centroids
        self.metric = metric if metric is not None else lambda x, y: np.linalg.norm(np.array(x)-np.array(y))

    def assert_equal(self, test_result: Any):
        gt_clusters = set(frozenset(np.where(self._gt == label)[0]) for label in np.unique(self._gt))
        clusters = set(frozenset(np.where(test_result == label)[0]) for label in np.unique(test_result))
        assert gt_clusters == clusters


@pytest.mark.parametrize('case', [
                             ClusterizationCase(
                                 name='1d_euclidean',
                                 data=np.array([1, 2, 1.5, 26, 27, 26.4]),
                                 clusters_num=2,
                                 centroids=[3, 5],
                                 gt=np.array([0, 0, 0, 1, 1, 1])),
                             ClusterizationCase(
                                 name='2d_gaussian',
                                 data=np.array([[0, 0], [1, 1], [-1, 1], [-0.5, -0.5], [0.5, -1],
                                                [13, 12], [-13, 13], [-13, -14], [12, -13], [-14, -14]]),
                                 clusters_num=2,
                                 centroids=[7, 9],
                                 metric=lambda x, y: np.linalg.norm(x**2 + y**2),
                                 gt=np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))
                         ], ids=str)
def test_kernel_kmeans(case: ClusterizationCase) -> None:
    clusterizer = KmeansKernelSpineClusterizer(case.clusters_num)
    labels = clusterizer._kernel_fit(case.X, case.metric, initialization=case.centroids)
    case.assert_equal(labels)
