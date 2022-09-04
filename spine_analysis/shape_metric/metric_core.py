from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
from ipywidgets import widgets
from matplotlib import pyplot as plt

from CGAL.CGAL_Polyhedron_3 import Polyhedron_3


class SpineMetric(ABC):
    name: str
    _value: Any

    def __init__(self, spine_mesh: Polyhedron_3 = None, name: str = None) -> None:
        self.name = type(self).__name__.replace("SpineMetric", "") if name is None else name
        if spine_mesh is not None:
            self.value = self._calculate(spine_mesh)

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, new_value: Any) -> None:
        self._value = new_value

    def calculate(self, spine_mesh: Polyhedron_3) -> Any:
        self.value = self._calculate(spine_mesh)
        return self.value

    @abstractmethod
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        pass

    def show(self) -> widgets.Widget:
        return widgets.Label(str(self.value))

    @classmethod
    @abstractmethod
    def get_distribution(cls, metrics: List["SpineMetric"]) -> np.ndarray:
        pass

    @classmethod
    def show_distribution(cls, metrics: List["SpineMetric"]) -> widgets.Widget:
        graph = widgets.Output()
        with graph:
            cls._show_distribution(metrics)
            plt.title(cls.__name__)
            plt.show()
        return graph

    @classmethod
    @abstractmethod
    def _show_distribution(cls, metrics: List["SpineMetric"]) -> None:
        pass

    def value_as_list(self) -> List[Any]:
        try:
            return [*self.value]
        except TypeError:
            return [self.value]

    @staticmethod
    def get_metric_class(metric_name):
        return globals()[metric_name + "SpineMetric"]


class ManualSpineMetric(SpineMetric):
    def __init__(self, value: float, name: str) -> None:
        super().__init__(name=name)
        self.value = value

    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        pass
