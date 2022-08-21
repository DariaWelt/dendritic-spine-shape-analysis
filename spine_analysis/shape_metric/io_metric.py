import csv
import os
from copy import deepcopy
from typing import Dict, List, Union, Any

import numpy as np

from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from spine_analysis.shape_metric.approximation_metric import ApproximationSpineMetric
from spine_analysis.shape_metric.float_metric import FloatSpineMetric
from spine_analysis.shape_metric.metric_core import SpineMetric
from spine_analysis.shape_metric.utils import calculate_metrics, create_metric_by_name


class SpineMetricDataset:
    num_of_spines: int
    num_of_metrics: int
    _spine_2_row: Dict[str, int]
    _metric_2_column: Dict[str, int]
    _table = np.ndarray
    SPINE_FILE_FIELD = "Spine File"

    def __init__(self, metrics: Dict[str, List[SpineMetric]] = None) -> None:
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

    def row(self, spine_name: str) -> List[SpineMetric]:
        return list(self._table[self._spine_2_row[spine_name], :])

    def rows(self) -> List[SpineMetric]:
        for row in self._table:
            yield list(row)

    def column(self, metric_name: str) -> List[SpineMetric]:
        return self._table[:, self._metric_2_column[metric_name]]

    def element(self, spine_name: str, metric_name: str) -> SpineMetric:
        return self._table[self._spine_2_row[spine_name], self._metric_2_column[metric_name]]

    @property
    def spine_names(self) -> List[str]:
        return list(self._spine_2_row.keys())

    @property
    def metric_names(self) -> List[str]:
        return list(self._metric_2_column.keys())

    def calculate_metrics(self, spine_meshes: Dict[str, Polyhedron_3],
                          metric_names: List[str],
                          params: List[Dict] = None) -> None:
        metrics = {}
        for (spine_name, spine_mesh) in spine_meshes.items():
            metrics[spine_name] = calculate_metrics(spine_mesh, metric_names, params)
        self.__init__(metrics)

    def get_spines_subset(self, reduced_spine_names: List[str]) -> "SpineMetricDataset":
        reduced_spines = {spine_name: self.row(spine_name) for spine_name in reduced_spine_names}
        return SpineMetricDataset(reduced_spines)

    def get_metrics_subset(self, reduced_metric_names: List[str]) -> "SpineMetricDataset":
        index_subset = [self._metric_2_column[metric_name]
                        for metric_name in reduced_metric_names]
        reduced_metrics = {}
        for (spine_name, spine_metrics) in zip(self.spine_names, self._table):
            reduced_metrics[spine_name] = [spine_metrics[i] for i in index_subset]

        return SpineMetricDataset(reduced_metrics)

    def standardize(self) -> None:
        float_metric_indices = [i for i in range(self.num_of_metrics)
                                if isinstance(self._table[0, i], FloatSpineMetric)
                                or isinstance(self._table[0,i], ApproximationSpineMetric)]

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
        data = []
        for spine_metric in self.row(spine_name):
            data += spine_metric.value_as_list()
        return np.asarray(data)

    def as_array(self) -> np.ndarray:
        data = [self.row_as_array(spine_name) for spine_name in self.spine_names]
        return np.asarray(data)

    def export_to_csv(self, filename: Union[str, os.PathLike]):
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
    def load_metrics(filename: str) -> "SpineMetricDataset":
        output = {}
        with open(filename, mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                # extract spine file name
                spine_name = row.pop(SpineMetricDataset.SPINE_FILE_FIELD)
                # extract each metric
                metrics = []
                for metric_name in row.keys():
                    value_str = row[metric_name]
                    value: Any
                    if value_str[0] == "[":
                        value = np.fromstring(value_str[1:-1], dtype="float", sep=" ")
                    else:
                        value = float(value_str)
                    metric = create_metric_by_name(metric_name)
                    metric._value = value
                    metrics.append(metric)
                output[spine_name] = metrics
        return SpineMetricDataset(output)

