import numpy as np


def score(clusterizer) -> float:
    output = 0
    for group in clusterizer.grouping.groups.values():
        center = sum(np.array(clusterizer.fit_metrics.row_as_array(spine_name)) for spine_name in group)
        output += sum(np.inner(center - clusterizer.fit_metrics.row_as_array(spine_name),
                               center - clusterizer.fit_metrics.row_as_array(spine_name)) for spine_name in group)
    return abs(output)
