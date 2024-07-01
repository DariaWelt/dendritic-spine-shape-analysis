import csv
import functools
import os
from copy import deepcopy
from multiprocessing import Pool
from typing import Dict, List, Union, Any, Set, Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tqdm
import umap

from spine_analysis.mesh.utils import MeshDataset, _mesh_to_v_f
from spine_analysis.shape_metric.approximation_metric import ApproximationSpineMetric
from spine_analysis.shape_metric.float_metric import FloatSpineMetric
from spine_analysis.shape_metric.metric_core import SpineMetric, ManualSpineMetric
from spine_analysis.shape_metric.utils import calculate_metrics, create_metric_by_name, calculate_metrics_parallel, ExceptionWrapper


class SpineMetricDataset:
    SPINE_FILE_FIELD = "Spine File"

    num_of_spines: int
    num_of_metrics: int
    spine_meshes: MeshDataset
    _spine_2_row: Dict[str, int]
    _metric_2_column: Dict[str, int]
    _table = np.ndarray

    def __init__(self, metrics: Dict[str, List[SpineMetric]] = None, distances: Dict[str, callable] = None) -> None:
        if metrics is None:
            metrics = {}
        self.num_of_spines = len(metrics)
        first_row = []
        if self.num_of_spines > 0:
            first_row = list(metrics.values())[0]
        self.num_of_metrics = len(first_row)

        self._spine_2_row = {spine_name: i for i, spine_name in enumerate(metrics.keys())}
        self._metric_2_column = {metric.name: i for i, metric in enumerate(first_row)}

        self._table = np.ndarray((self.num_of_spines, self.num_of_metrics), dtype="O")

        for i, (spine_name, row) in enumerate(metrics.items()):
            for j, metric in enumerate(row):
                self._table[i, j] = metric
        
        self._metric_2_distance = {}
        for metric_name in self._metric_2_column.keys():
            self._metric_2_distance[metric_name] = lambda x, y: np.linalg.norm(np.array(x)-np.array(y)) if distances is None or distances.get(metric_name) is None else distances[metric_name]

    def row(self, spine_name: str) -> List[SpineMetric]:
        return list(self._table[self._spine_2_row[spine_name], :])

    def rows(self) -> List[SpineMetric]:
        for row in self._table:
            yield list(row)

    def column(self, metric_name: str) -> Dict[str, SpineMetric]:
        column_idx = self._metric_2_column[metric_name]
        return {spine_name: self.row(spine_name)[column_idx] for spine_name in self.spine_names}

    def element(self, spine_name: str, metric_name: str) -> SpineMetric:
        return self._table[self._spine_2_row[spine_name], self._metric_2_column[metric_name]]
    
    def set_metric_distance(self, metric_name: str, distance: callable):
        self._metric_2_distance[metric_name] = distance
    
    @property
    def metric_distances(self) -> Dict[str, callable]:
        return dict(self._metric_2_distance)

    @property
    def ordered_spine_names(self) -> List[str]:
        names = list(self.spine_names)
        names.sort()
        return names

    @property
    def spine_names(self) -> Set[str]:
        return set(self._spine_2_row.keys())

    @property
    def metric_names(self) -> List[str]:
        return list(self._metric_2_column.keys())

    def calculate_metrics(self, spine_meshes: MeshDataset,
                          metric_names: List[str],
                          params: List[Dict] = None,
                          recalculate: bool = True,
                          processes: int = -1) -> None:
        # TODO: handle metric recalculation
        processes = min(processes, os.cpu_count()) if processes > 0 else os.cpu_count()
        chunk_size = int(np.ceil(len(spine_meshes) / processes))

        calculate_parallel = functools.partial(calculate_metrics_parallel, metric_names=metric_names, params=params)

        if processes > 1:
            spines = [spine_name for spine_name in spine_meshes.keys()]
            spines_vf = [_mesh_to_v_f(spine_meshes[spine_name]) for spine_name in spines]
            metrics = {}
            print(f"run pool with {processes} processes num")
            with Pool(processes) as pool:
                results = list(tqdm.tqdm(pool.imap(calculate_parallel, spines_vf, chunksize=chunk_size), total=len(spines_vf)))
                #results = pool.map(calculate_parallel, spines_vf, chunksize=chunk_size)
                
                for spine_name, result in zip(spines, results):
                    if isinstance(result, ExceptionWrapper):
                        print(result)
                        print(spine_name)
                        print()
                        continue
                    metrics[spine_name] = []
                    for metric_name in metric_names:
                        metrics[spine_name].append(create_metric_by_name(metric_name))
                        metrics[spine_name][-1].value = result[metric_name]
        else:
            print(f"run in one process")
            self.spine_meshes = spine_meshes
            metrics = {}
            for (spine_name, spine_mesh) in tqdm.tqdm(spine_meshes.items(), total=len(spine_meshes)):
                metrics[spine_name] = calculate_metrics(spine_mesh, metric_names, params)
        self.__init__(metrics)

    def as_dict(self) -> Dict[str, List[SpineMetric]]:
        return {spine_name: self.row(spine_name) for spine_name in self.spine_names}

    def add_metric(self, metric_values: Dict[str, SpineMetric]):
        metrics = self.as_dict()
        for spine_name in self.spine_names:
            metrics[spine_name].append(metric_values[spine_name])
        self.__init__(metrics)

    def get_spines_subset(self, reduced_spine_names: Iterable[str]) -> "SpineMetricDataset":
        reduced_spine_names = set(reduced_spine_names).intersection(self.spine_names)
        reduced_spines = {spine_name: self.row(spine_name) for spine_name in reduced_spine_names}
        return SpineMetricDataset(reduced_spines)

    def get_metrics_subset(self, reduced_metric_names: Iterable[str]) -> "SpineMetricDataset":
        index_subset = [self._metric_2_column[metric_name]
                        for metric_name in reduced_metric_names]
        reduced_metrics = {}
        for spine_name in self.spine_names:
            spine_metrics = self.row(spine_name)
            reduced_metrics[spine_name] = [spine_metrics[i] for i in index_subset]

        return SpineMetricDataset(reduced_metrics)

    def standardize(self) -> None:
        float_metric_indices = [i for i in range(self.num_of_metrics)
                                if isinstance(self._table[0, i], FloatSpineMetric)
                                or isinstance(self._table[0, i], ApproximationSpineMetric)]

        # calculate mean and std by column
        mean = {}
        std = {}
        for i in float_metric_indices:
            column_values = np.array([metric.value for metric in self._table[:, i]])
            mean[i] = np.mean(column_values, axis=0)
            std[i] = np.std(column_values, axis=0)

        for i in range(self.num_of_spines):
            for j in float_metric_indices:
                metric = self._table[i, j]
                metric.value = (metric.value - mean[j]) / std[j]

    def standardized(self) -> "SpineMetricDataset":
        output = deepcopy(self)
        output.standardize()
        return output

    def row_as_array(self, spine_name: str) -> np.array:
        data = np.array([])
        for spine_metric in self.row(spine_name):
            data = np.append(data, np.vstack(spine_metric.value_as_lists()))
        return np.asarray(data)

    def as_array(self) -> np.ndarray:
        data = [self.row_as_array(spine_name) for spine_name in self.ordered_spine_names]
        return np.asarray(data)

    def as_reduced_array(self, n_components: int = 2, method: str = "pca") -> np.ndarray:
        if method == "pca":
            return PCA(n_components).fit_transform(self.as_array())
        elif method == "tsne":
            return TSNE(n_components, init="pca", random_state=0).fit_transform(self.as_array())
        elif method == "umap":
            return umap.UMAP(n_components=n_components).fit_transform(self.as_array())
        else:
            raise NotImplemented(f"method {method} is not supported")

    def reduce(self, n_components: int = 2, method: str = "pca") -> "SpineMetricDataset":
        reduced_metrics = {spine_name: [] for spine_name in self.spine_names}
        reduced_data = self.as_reduced_array(n_components, method)
        ordered_names = self.ordered_spine_names
        for i, spine_name in enumerate(ordered_names):
            for j in range(n_components):
                conv = ManualSpineMetric(reduced_data[i, j], f"PC{j + 1}")
                reduced_metrics[spine_name].append(conv)
        return SpineMetricDataset(reduced_metrics)

    def save_as_array(self, filename) -> None:
        with open(filename, mode="w") as file:
            if self.num_of_spines == 0:
                return
            # save metrics for each spine
            writer = csv.writer(file)
            for spine_name in self.spine_names:
                writer.writerow([spine_name] + list(self.row_as_array(spine_name)))

    def save(self, filename: str) -> None:
        with open(filename, mode="w") as file:
            if self.num_of_spines == 0:
                return
            # extract from metric names from first spine
            metric_names = self.metric_names

            # save metrics for each spine
            writer = csv.DictWriter(file, fieldnames=[self.SPINE_FILE_FIELD] + metric_names)
            writer.writeheader()
            for spine_name in self.spine_names:
                writer.writerow({self.SPINE_FILE_FIELD: spine_name,
                                 **{metric.name: metric.value
                                    for metric in self.row(spine_name)}})

    @staticmethod
    def load(filename: str) -> "SpineMetricDataset":
        output = {}
        with open(filename, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                # extract spine file name
                spine_name = row.pop(SpineMetricDataset.SPINE_FILE_FIELD).replace('\\', '/')
                # extract each metric
                metrics = []
                for metric_name in row.keys():
                    if metric_name == "OldChordDistribution":
                        value_str = [row[metric_name], *row[None]]
                    value_str = row[metric_name]
                    try:
                        metric = create_metric_by_name(metric_name)
                        metric.parse_value(value_str)
                        metrics.append(metric)
                    except Exception:
                        pass
                output[spine_name] = metrics
        return SpineMetricDataset(output)

