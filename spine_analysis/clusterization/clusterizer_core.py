import json
from abc import ABC, abstractmethod
from typing import List, Callable, Union, Set

import numpy as np
from ipywidgets import widgets
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from spine_analysis.shape_metric.io_metric import SpineMetricDataset
from spine_analysis.spine.fitter import SpineFitter


class SpineClusterizer(SpineFitter, ABC):
    _data: np.ndarray
    metric: Union[Callable[[np.ndarray, np.ndarray], float], str]
    _labels: List[int]

    def __init__(self, metric: Union[Callable, str] = "euclidean", pca_dim: int = -1):
        super().__init__(pca_dim)
        self._labels = []
        self.metric = metric

    @property
    def clusters(self) -> List[Set[str]]: #get_clusters()
        return list(self.grouping.groups.values())

    @property
    def outlier_cluster(self) -> Set[str]:
        return self.grouping.outlier_group

    @property
    def num_of_clusters(self) -> int:
        return self.grouping.num_of_groups

    def get_cluster(self, cluster_name: str) -> Set[str]:
        return self.grouping.groups.get(cluster_name, {})


class SKLearnSpineClusterizer(SpineClusterizer, ABC):
    _fit_data: object
    _clusterizer: object

    def _fit(self, data: np.array, names: List[str]) -> None:
        self._fit_data = self._sklearn_fit(data)
        self._labels = self._fit_data.labels_

        for cluster_index in set(self._labels):
            if cluster_index == -1:
                continue
            names_array = np.array(names)
            cluster_names = names_array[self._labels == cluster_index]
            self.grouping.groups[str(cluster_index + 1)] = set(cluster_names)

    @abstractmethod
    def _sklearn_fit(self, data: np.ndarray) -> object:
        pass
