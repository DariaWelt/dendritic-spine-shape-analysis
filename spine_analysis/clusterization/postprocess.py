import numpy as np
from sklearn.metrics import silhouette_score

from spine_analysis.clusterization import SpineClusterizer


def score(clusterizer: SpineClusterizer) -> float:
    output = 0
    for group in clusterizer.grouping.groups.values():
        center = sum(np.array(clusterizer.fit_metrics.row_as_array(spine_name)) for spine_name in group)
        output += sum(np.inner(center - clusterizer.fit_metrics.row_as_array(spine_name),
                               center - clusterizer.fit_metrics.row_as_array(spine_name)) for spine_name in group)
    return abs(output)


def silhouette(clusterizer: SpineClusterizer) -> float:
    datas = []
    labels = []
    for i, group in enumerate(clusterizer.grouping.groups.values()):
        datas.extend(clusterizer.fit_metrics.row_as_array(spine) for spine in group)
        labels.extend([i for _ in group])
    x = np.array([[clusterizer.metric(x1, x2) for x1 in datas] for x2 in datas])
    labels = np.array(labels)
    return silhouette_score(x, labels, metric='precomputed')
