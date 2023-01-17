from abc import ABC, abstractmethod
from typing import Optional, List, Dict

import numpy as np

from spine_analysis.shape_metric.io_metric import SpineMetricDataset


class MetricsComposition(ABC):
    def __init__(self, metric_dataset: SpineMetricDataset):
        self.metrics: Dict[str, int] = metric_dataset._metric_2_column
        row = next(metric_dataset.rows())
        self.shape = [np.size(m.value_as_lists()) for m in row]
        self.distances: Dict[str, callable] = metric_dataset.metric_distances

    def distance(self, row_data1: np.ndarray, row_data2: np.ndarray) -> float:
        slice_range = lambda metric_index: slice(self.shape[metric_index - 1], self.shape[metric_index - 1] + self.shape[metric_index])
        objects_distances = []
        for metric, col_i in self.metrics.items():
            data_slice = slice_range(col_i)
            objects_distances.append(self.distances[metric](row_data1[data_slice], row_data2[data_slice]))
        return self._distance(np.array(objects_distances))

    @abstractmethod
    def _distance(self, objects_distances: np.ndarray) -> float:
        pass


class AdditiveComposition(MetricsComposition):
    def _distance(self, objects_distances: np.ndarray, weights: Optional[List[int]] = None) -> float:
        if weights is None:
            weights = np.ones(len(self.metrics))
        else:
            weights = np.array(weights)
        return np.matmul(weights, objects_distances)


class SquaredRootComposition(MetricsComposition):
    def _distance(self, objects_distances: np.ndarray) -> float:
        squares = objects_distances ** 2
        return np.sqrt(np.sum(squares))
