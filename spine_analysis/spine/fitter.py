from abc import ABC, abstractmethod
from typing import List

import numpy as np
from ipywidgets import widgets
from sklearn.decomposition import PCA

from spine_analysis.shape_metric.io_metric import SpineMetricDataset
from spine_analysis.spine.grouping import SpineGrouping


class SpineFitter(ABC):
    grouping: SpineGrouping
    pca_dim: int
    fit_metrics: SpineMetricDataset

    def __init__(self, pca_dim: int = -1):
        self.pca_dim = pca_dim
        self.grouping = SpineGrouping()

    # def get_representative_samples(self, group_index: int,
    #                                num_of_samples: int = 4,
    #                                distance: Callable = euclidean) -> List[str]:
    #     if distance is None:
    #         distance = euclidean
    #
    #     # get spines in cluster
    #     spine_names = self.groups[group_index]
    #     num_of_samples = min(num_of_samples, len(spine_names))
    #     spines = [self.get_spine_reduced_coord(name) for name in spine_names]
    #     # calculate center (mean reduced data)
    #     center = np.mean(spines, 0)
    #     # calculate distance to center for each spine in cluster
    #     distances = {}
    #     for (spine, name) in zip(spines, spine_names):
    #         distances[name] = distance(center, spine)
    #     # sort spines by distance
    #     output = list(spine_names)
    #     output.sort(key=lambda name: distances[name])
    #     # return first N spine names
    #     return output[:num_of_samples]

    def fit(self, spine_metrics: SpineMetricDataset) -> None:
        self.fit_metrics = spine_metrics
        data = spine_metrics.as_array()
        if self.pca_dim != -1:
            pca = PCA(self.pca_dim)
            data = pca.fit_transform(data)

        self.grouping.samples = spine_metrics.spine_names

        self._fit(data, spine_metrics.ordered_spine_names)

    @abstractmethod
    def _fit(self, data: np.array, names: List[str]) -> object:
        pass

    def show(self) -> widgets.Widget:
        return self.grouping.show(self.fit_metrics)
