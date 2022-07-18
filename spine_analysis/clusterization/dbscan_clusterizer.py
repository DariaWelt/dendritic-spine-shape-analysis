from typing import Set, Union, Callable

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from spine_analysis.clusterization.clusterizer_core import SKLearnSpineClusterizer


class DBSCANSpineClusterizer(SKLearnSpineClusterizer):
    eps: float
    min_samples: int
    num_of_noise: int

    def __init__(self, eps: float = 0.5, min_samples: int = 2,
                 metric: Union[str, Callable] = "euclidean", pca_dim: int = -1):
        super().__init__(metric=metric, pca_dim=pca_dim)
        self.metric = metric
        self.min_samples = min_samples
        self.eps = eps

    def score(self) -> float:
        # TODO: change nan to something sensical
        # if self.num_of_noise / self.sample_size > 0.15 or self.num_of_clusters < 2 or self.sample_size - self.num_of_noise - 1 < self.num_of_clusters:
        if self.num_of_clusters < 2 or self.sample_size - self.num_of_noise - 1 < self.num_of_clusters:
            return float("nan")
        labels = self.get_labels()
        indices_to_delete = np.argwhere(np.asarray(labels) == -1)
        filtered_data = np.delete(self._data, indices_to_delete, 0)
        filtered_labels = np.delete(labels, indices_to_delete, 0)
        return silhouette_score(filtered_data, filtered_labels, metric=self.metric)

    def _sklearn_fit(self, data: np.array) -> object:
        clusterized = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric).fit(data)

        # Number of clusters in labels, ignoring noise if present.
        labels = clusterized.labels_
        self.num_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.num_of_noise = list(labels).count(-1)

        return clusterized

    def _show(self, clusters: Set[int] = None) -> None:
        core_samples_mask = np.zeros_like(self._fit_data.labels_, dtype=bool)
        core_samples_mask[self._fit_data.core_sample_indices_] = True

        # Black removed and is used for noise instead.
        colors = [plt.cm.Spectral(each) for each in
                  np.linspace(0, 1, len(self.cluster_masks))]

        labels = self._fit_data.labels_
        for k, col in zip(set(labels), colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = labels == k

            xy = self.reduced_data[class_member_mask & core_samples_mask]
            if xy.size > 0:
                plt.plot(
                    xy[:, 0],
                    xy[:, 1],
                    "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=14,
                    label=f"{k}"
                )

            xy = self.reduced_data[class_member_mask & ~core_samples_mask]
            if xy.size > 0:
                plt.plot(
                    xy[:, 0],
                    xy[:, 1],
                    "o",
                    markerfacecolor=tuple(col),
                    markeredgecolor="k",
                    markersize=6,
                    label=f"{k}"
                )
        plt.title(f"Number of clusters: {self.num_of_clusters}, score: {self.score():.3f}")
        plt.legend(loc="upper right")
