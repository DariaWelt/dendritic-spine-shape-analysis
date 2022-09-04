from abc import ABC
from typing import List, Any

import numpy as np
from ipywidgets import widgets
from matplotlib import pyplot as plt

from CGAL.CGAL_Convex_hull_3 import convex_hull_3
from CGAL.CGAL_Polygon_mesh_processing import volume
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from spine_analysis.shape_metric.metric_core import SpineMetric


class FloatSpineMetric(SpineMetric, ABC):
    def show(self) -> widgets.Widget:
        return widgets.Label(f"{self._value:.2f}")

    @classmethod
    def get_distribution(cls, metrics: List["SpineMetric"]) -> np.ndarray:
        return np.asarray([metric.value for metric in metrics])

    @classmethod
    def _show_distribution(cls, metrics: List["SpineMetric"]) -> None:
        plt.boxplot(cls.get_distribution(metrics))


class VolumeSpineMetric(FloatSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        return abs(volume(spine_mesh))


class ConvexHullVolumeSpineMetric(FloatSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        hull_mesh = Polyhedron_3()
        convex_hull_3(spine_mesh.points(), hull_mesh)
        return volume(hull_mesh)


class ConvexHullRatioSpineMetric(FloatSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        hull_mesh = Polyhedron_3()
        convex_hull_3(spine_mesh.points(), hull_mesh)
        v = abs(volume(spine_mesh))
        return (volume(hull_mesh) - v) / v
