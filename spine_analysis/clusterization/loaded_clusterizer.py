import copy
import json
from typing import List, Union, Callable, Dict, Set

import numpy as np

from spine_analysis.clusterization.clusterizer_core import SpineClusterizer
from spine_analysis.shape_metric.io_metric import SpineMetricDataset


class ManualSpineClusterizer(SpineClusterizer):
    def __init__(self, clusters: Dict[str, Set[str]], metric: Union[str, Callable] = "euclidean"):
        super().__init__(metric=metric)
        self.grouping.groups = clusters
        if self.num_of_clusters > 0:
            self.num_of_samples = sum(len(group) for _, group in clusters.items())
        else:
            self.num_of_samples = 0

    def _fit(self, data: np.array, names: List[str]) -> object:
        pass

    @staticmethod
    def load_clusterization(filename: str, dataset: SpineMetricDataset) -> SpineClusterizer:
        masks = []
        with open(filename) as file:
            data = json.load(file)
            clusterization = data["groups"]
        clusterizer = ManualSpineClusterizer(clusterization)
        if len(masks) > 0:
            clusterizer.sample_size = dataset.num_of_spines
            clusterizer.dataset = dataset
        return clusterizer

    def clusterizer_entropy(self, clusterizer: SpineClusterizer) -> float:
        if len(clusterizer.cluster_masks) == 0:
            return -1
        if clusterizer.sample_size != self.sample_size:
            return -1
        m = clusterizer.get_clusters()
        p = self.get_clusters()
        entropy = []
        for m_i in range(clusterizer.num_of_clusters):
            per_p_sum = 0
            for p_i in range(self.num_of_clusters):
                mp = set(m[m_i]).intersection(p[p_i])
                ratio = len(mp)/(len(m[m_i]) + 1e-20)
                per_p_sum += ratio * np.log(ratio+1e-20)
            per_p_sum *= -1 / (np.log(self.num_of_clusters))
            entropy.append(per_p_sum)
        return sum(entropy) / self.sample_size

    def clusterizer_purity(self, clusterizer: SpineClusterizer) -> float:
        if len(clusterizer.cluster_masks) == 0:
            return -1
        if clusterizer.sample_size != self.sample_size:
            return -1
        m = clusterizer.get_clusters()
        p = self.get_clusters()
        average_putiry = 0
        for m_i in range(clusterizer.num_of_clusters):
            max_num = 0
            for p_i in range(self.num_of_clusters):
                mp = set(m[m_i]).intersection(p[p_i])
                if len(mp) > max_num:
                    max_num = len(mp)
            average_putiry += max_num
        return average_putiry / self.sample_size
