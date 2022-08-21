import matplotlib.pyplot as plt

from spine_analysis.clusterization.dbscan_clusterizer import DBSCANSpineClusterizer
from spine_analysis.clusterization.kmeans_clusterizer import KMeansSpineClusterizer
from spine_analysis.clusterization.loaded_clusterizer import ManualSpineClusterizer
from spine_analysis.mesh.utils import load_spine_meshes
from spine_analysis.shape_metric.io_metric import SpineMetricDataset
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
import numpy as np
from scipy.special import kl_div
from notebook_widgets import create_dir


metric_names = ["SphericalGarmonics", "OpenAngle", "CVD", "AverageDistance",
                "Length", "Area", "Volume", "ConvexHullVolume", "ConvexHullRatio"]

folder = '0.025-0.025-0.1-dataset'
# load meshes
full_mesh = Polyhedron_3(f"{folder}/3_full_res (1)/surface_mesh.off")
spine_meshes = load_spine_meshes(folder_path=folder)

every_spine_metrics = SpineMetricDataset()
every_spine_metrics.calculate_metrics(spine_meshes, metric_names)
output_dir = "output/clustering_normalized"
create_dir(output_dir)
every_spine_metrics.export_to_csv(f"{output_dir}/metrics.csv")

every_spine_metrics = SpineMetricDataset.load_metrics(f"{output_dir}/metrics.csv")
every_spine_metrics.standardize()

every_spine_metrics.standardize()

#index_subsets = [list(range(8))]
#index_subsets = [[0]]
index_subsets = [[1, 2, 3, 4, 5, 6, 8]]

# clusterizers
all_clusterizers = []

# dbscan
# for (name, metric) in zip(["euclidean", "kl_div"],
#                           ["euclidean", lambda x, y: np.sum(kl_div(x, y))]):
for i in range(100):
    eps = 0.01 * (i + 1) #pow(10,(i+1))
    all_clusterizers.append(("DBSCAN_l1", f"e={0.01*(i+1)})", eps,
                             DBSCANSpineClusterizer(metric="l1", eps=eps)))
for i in range(100):
    eps = 0.1 * (i + 1)
    all_clusterizers.append(("DBSCAN_kldiv", f"e={eps:.2f}", eps,
                             DBSCANSpineClusterizer(metric=lambda x, y: np.sum(kl_div(x, y)), eps=eps)))

#kmeans
for num_of_clusters in range(2, 30):
    all_clusterizers.append(("KMeans_l1", f"n={num_of_clusters}", num_of_clusters,
                             KMeansSpineClusterizer(num_of_clusters=num_of_clusters, metric="l1")))

# all possible metric combinations
# for L in range(1, len(metric_names) + 1):
#     for index_subset in itertools.combinations(list(range(len(metric_names))), L):
for index_subset in index_subsets:
    reduced_metric_names = [metric_names[i] for i in index_subset]
    reduced_metrics = every_spine_metrics.get_metrics_subset(reduced_metric_names)

    base_save_dir = f"{output_dir}/{str(reduced_metric_names)}"
    create_dir(base_save_dir)

    scores = {"KMeans": ([], []), "KMeans_l1": ([], []),
              "DBSCAN_eucl": ([], []), "DBSCAN_l1": ([], [])}

    for (clusterizer_name, param_string, param_value, clusterizer) in all_clusterizers:
        save_dir = f"{base_save_dir}/{clusterizer_name}"
        create_dir(save_dir)

        clusterizer.fit(reduced_metrics)
        score = clusterizer.score()
        if np.isnan(score):  # or score <= 0:
            continue

        # save plot result
        filename = f"{save_dir}/{clusterizer_name}_{param_string}_score={score:.2f}"
        clusterizer.save_plot(filename + ".png")
        clusterizer.save(filename + ".json")

        scores[clusterizer_name][0].append(param_value)
        scores[clusterizer_name][1].append(score)

    for clusterizer_name in scores:
        plt.axhline(y=0, color='r', linestyle='-')
        plt.plot(scores[clusterizer_name][0], scores[clusterizer_name][1])
        plt.title(clusterizer_name)
        save_dir = f"{base_save_dir}/{clusterizer_name}"
        create_dir(save_dir)
        plt.savefig(f"{save_dir}/score.png")
        plt.clf()
