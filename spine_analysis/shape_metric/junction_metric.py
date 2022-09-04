import math
from abc import ABC, abstractmethod
from typing import List, Any, Set

import numpy as np

from CGAL.CGAL_Kernel import Vector_3, cross_product
from CGAL.CGAL_Polygon_mesh_processing import area, face_area
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3, Polyhedron_3_Facet_handle
from spine_analysis.shape_metric.float_metric import FloatSpineMetric
from spine_analysis.shape_metric.utils import _calculate_facet_center, _point_2_vec


class JunctionSpineMetric(FloatSpineMetric, ABC):
    _junction_center: Vector_3
    _surface_vectors: List[Vector_3]
    _junction_triangles: Set[Polyhedron_3_Facet_handle]

    @abstractmethod
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        # identify junction triangles
        self._junction_triangles = set()
        for v in spine_mesh.vertices():
            if v.vertex_degree() > 10:
                # mark adjacent triangles
                for h in spine_mesh.halfedges():
                    if h.vertex() == v:
                        self._junction_triangles.add(h.facet())

        # calculate junction center
        if len(self._junction_triangles) > 0:
            self._junction_center = Vector_3(0, 0, 0)
            for facet in self._junction_triangles:
                self._junction_center += _calculate_facet_center(facet)
            self._junction_center /= len(self._junction_triangles)
        else:
            self._junction_center = _point_2_vec(spine_mesh.points().next())

        # calculate vectors to surface
        self._surface_vectors = []
        for point in spine_mesh.points():
            self._surface_vectors.append(_point_2_vec(point) - self._junction_center)


class JunctionAreaSpineMetric(JunctionSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        super()._calculate(spine_mesh)

        return sum(face_area(triangle, spine_mesh)
                   for triangle in self._junction_triangles)


class AreaSpineMetric(JunctionAreaSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        return area(spine_mesh) - super()._calculate(spine_mesh)



class JunctionDistanceSpineMetric(JunctionSpineMetric, ABC):
    _distances: List[float]

    @abstractmethod
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        super()._calculate(spine_mesh)

        self._distances = []
        for v in self._surface_vectors:
            self._distances.append(np.sqrt(v.squared_length()))


class AverageDistanceSpineMetric(JunctionDistanceSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        super()._calculate(spine_mesh)
        return np.mean(self._distances)


class LengthSpineMetric(JunctionDistanceSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        super()._calculate(spine_mesh)
        q = np.quantile(self._distances, 0.95)
        return np.mean([d for d in self._distances if d >= q])


class LengthVolumeRatioSpineMetric(LengthSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        return super()._calculate(spine_mesh) / abs(volume(spine_mesh))


class LengthAreaRatioSpineMetric(LengthSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        return super()._calculate(spine_mesh) / area(spine_mesh)


class CVDSpineMetric(JunctionDistanceSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        super()._calculate(spine_mesh)
        return np.std(self._distances, ddof=1) / np.mean(self._distances)


class OpenAngleSpineMetric(JunctionSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        super()._calculate(spine_mesh)

        axis = np.mean(self._surface_vectors)
        angle_sum = 0
        for v in self._surface_vectors:
            angle_sum += math.atan2(np.sqrt(cross_product(axis, v).squared_length()), axis * v)

        return angle_sum / len(self._surface_vectors)