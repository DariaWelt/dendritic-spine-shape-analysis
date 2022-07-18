import json
from typing import List, Union, Callable

import numpy as np

from spine_analysis.clusterization.clusterizer_core import SpineClusterizer


class ManualSpineClusterizer(SpineClusterizer):
    def __init__(self, cluster_masks: List[List[bool]],
                 metric: Union[str, Callable] = "euclidean"):
        super().__init__(metric=metric)
        self.cluster_masks = cluster_masks
        self.num_of_clusters = len(cluster_masks)
        if self.num_of_clusters > 0:
            self.num_of_samples = len(cluster_masks[0])
        else:
            self.num_of_samples = 0

    def _fit(self, data: np.array) -> object:
        pass

    @staticmethod
    def load_clusterization(filename: str) -> SpineClusterizer:
        masks = []
        with open(filename) as file:
            masks = json.load(file)["cluster_masks"]
        return ManualSpineClusterizer(masks)