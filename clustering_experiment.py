from dataclasses import dataclass
from typing import List, Any

import matplotlib.pyplot as plt

from spine_analysis.clusterization import SpineClusterizer
from spine_analysis.clusterization.dbscan_clusterizer import DBSCANSpineClusterizer
from spine_analysis.clusterization.kmeans_clusterizer import KMeansSpineClusterizer
from spine_analysis.clusterization.loaded_clusterizer import ManualSpineClusterizer
from spine_analysis.mesh.utils import load_spine_meshes
from spine_analysis.shape_metric.io_metric import SpineMetricDataset
import numpy as np
from scipy.special import kl_div
from notebook_widgets import create_dir


@dataclass()
class ClusterizerInfo:
    name: str
    param_str: str
    param_value: Any
    clusterizer_instance: SpineClusterizer


@dataclass()
class ClusteringExperiment:
    metrics: List[str]
    clusterizers: List[ClusterizerInfo]


dataset_folder = '0.025-0.025-0.1-dataset'
output_dir = "output/clustering_normalized"
calculate_metrics = True
save_metrics = True
standardize_metrics = True

param1 = lambda x: 0.1 * (x + 1)
param2 = lambda x: 0.01 * (x + 1)

experiments = [
    ClusteringExperiment(
        ["OldChordDistribution"],
        [
            ClusterizerInfo("DBSCAN_kldiv", f"e={(param1(i)):.2f})", param1(i),
                            DBSCANSpineClusterizer(metric=lambda x, y: np.sum(kl_div(x, y)), eps=param1(i)))
            for i in range(100)
        ]),
    ClusteringExperiment(
        ["SphericalGarmonics"],
        [
            ClusterizerInfo("DBSCAN_l1", f"e={(param2(i)):.2f})", param2(i),
                            DBSCANSpineClusterizer(metric="l1", eps=param2(i)))
            for i in range(100)] +
        [
            ClusterizerInfo("KMeans_l1", f"n={i}", i,
                            KMeansSpineClusterizer(num_of_clusters=i, metric="l1"))
            for i in range(2, 30)
        ]),
    ClusteringExperiment(
        ["OpenAngle", "CVD", "AverageDistance", "Length", "Area", "Volume", "ConvexHullVolume", "ConvexHullRatio"],
        [
            ClusterizerInfo("DBSCAN_l2", f"e={(param2(i)):.2f})", param2(i),
                            DBSCANSpineClusterizer(metric="l2", eps=param2(i)))
            for i in range(100)] +
        [
            ClusterizerInfo("KMeans_l2", f"n={i}", i,
                            KMeansSpineClusterizer(num_of_clusters=i, metric="l2"))
            for i in range(2, 30)
        ]),
]

create_dir(output_dir)

# load meshes
spine_meshes = load_spine_meshes(folder_path=dataset_folder)

if calculate_metrics:
    every_spine_metrics = SpineMetricDataset()
    every_spine_metrics.calculate_metrics(spine_meshes, list({m for e in experiments for m in e.metrics}))

    if save_metrics:
        every_spine_metrics.save(f"{output_dir}/metrics.csv")

every_spine_metrics = SpineMetricDataset.load(f"{output_dir}/metrics.csv")

if standardize_metrics:
    every_spine_metrics.standardize()

for experiment in experiments:
    reduced_metrics = every_spine_metrics.get_metrics_subset(experiment.metrics)

    base_save_dir = f"{output_dir}/{str(experiment.metrics)}"
    create_dir(base_save_dir)

    scores = {c.name: ([], []) for c in experiment.clusterizers}

    for c in experiment.clusterizers:
        save_dir = f"{base_save_dir}/{c.name}"
        create_dir(save_dir)

        c.clusterizer_instance.fit(reduced_metrics)
        #score = c.clusterizer_instance.score()
        #if np.isnan(score):  # or score <= 0:
        #    continue

        # save plot result
        filename = f"{save_dir}/{c.name}_{c.param_str}"
        c.clusterizer_instance.save_plot(filename + ".png")
        c.clusterizer_instance.save(filename + ".json")

        scores[c.name][0].append(c.param_value)
        #scores[c.name][1].append(score)

    for clusterizer_name in scores:
        plt.axhline(y=0, color='r', linestyle='-')
        plt.plot(scores[clusterizer_name][0], scores[clusterizer_name][1])
        plt.title(clusterizer_name)
        save_dir = f"{base_save_dir}/{clusterizer_name}"
        create_dir(save_dir)
        plt.savefig(f"{save_dir}/score.png")
        plt.clf()
