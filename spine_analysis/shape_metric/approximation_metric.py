from abc import ABC, abstractmethod
from typing import List, Iterable

import numpy as np
import meshplot as mp
from ipywidgets import widgets
from numpy import real
from scipy.special import sph_harm
from sklearn.linear_model import LinearRegression

from CGAL.CGAL_Kernel import Vector_3
from CGAL.CGAL_Point_set_3 import Point_set_3
from CGAL.CGAL_Poisson_surface_reconstruction import poisson_surface_reconstruction
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from spine_analysis.mesh.utils import _mesh_to_v_f
from spine_analysis.shape_metric.metric_core import SpineMetric
from spine_analysis.shape_metric.utils import polar2cart, cart2polar
from spine_segmentation import point_2_list, list_2_point


class ApproximationSpineMetric(SpineMetric, ABC):
    _basis: List[callable]
    _bbox: np.ndarray

    def __init__(self, spine_mesh: Polyhedron_3 = None):
        if spine_mesh is not None:
            points = [point_2_list(point) for point in spine_mesh.points()]
            self._bbox = np.array([np.min(points, axis=0),np.max(points, axis=0)])
        super().__init__(spine_mesh)

    def show(self, steps_num: List[int] = None) -> widgets.Widget:
        if steps_num is None:
            steps_num = [50, 50]
        mesh: Polyhedron_3 = self._get_mesh(steps_num)
        vertices, facets = _mesh_to_v_f(mesh)
        out = widgets.Output()
        with out:
            mp.plot(vertices, facets)
        return out

    @abstractmethod
    def _get_mesh(self, steps_num: List[int]) -> Polyhedron_3:
        pass

    def surface_value(self, point: List[float]) -> float:
        res: float = 0
        for a, f in zip(self._value, self._basis):
            res += a * f(*point)
        return real(res)

    def surface_equation(self, point: List[float]) -> float:
        r = self.surface_value(point[:-1])
        return r - point[-1]

    @abstractmethod
    def surface_norm(self, point: List[float], dx: float = 0.01) -> np.ndarray:
        pass


class SphericalGarmonicsSpineMetric(ApproximationSpineMetric):
    def __init__(self, spine_mesh: Polyhedron_3 = None, l_range: Iterable = None):
        if l_range is None:
            l_range = range(10)
        self._basis = [self._get_basis(m, l) for l in l_range for m in range(-l, l+1)]
        super().__init__(spine_mesh)

    def surface_norm(self, point: List[float], dx: float = 0.01) -> np.ndarray:
        f_p = self.surface_value(point)
        d_theta, d_phi = point[0] + dx, point[1] + dx
        point_sin, point_cos = np.sin(point), np.cos(point)

        ab = np.array([self.surface_value([d_theta, point[1]]), d_theta, point[1]]) - np.array([f_p, *point])
        ac = np.array([self.surface_value([point[0], d_phi]), point[0], d_phi]) - np.array([f_p, *point])
        norm = np.cross(ab, ac)

        norm = np.matmul(np.array([
            [point_sin[0]*point_cos[1], point_cos[0]*point_cos[1], -point_sin[1]],
            [point_sin[0]*point_sin[1], point_cos[0]*point_sin[1], point_cos[1]],
            [point_cos[0], -point_sin[0], 0]
        ]), norm)
        return norm / np.linalg.norm(norm)

    def _get_mesh(self, steps_num: List[int]) -> Polyhedron_3:
        theta = np.linspace(0, np.pi, steps_num[0])
        phi = np.linspace(0, 2 * np.pi, steps_num[1])
        tp = np.concatenate(np.stack(np.meshgrid(theta, phi), axis=-1))
        radiuses = np.array([real(self.surface_value([t, p])) for t, p in tp])
        xyz = polar2cart(tp[...,0], tp[...,1], radiuses)
        normals = [self.surface_norm([t, p]) for t, p in tp]
        out: Point_set_3 = Point_set_3()
        out.add_normal_map()
        for p, n in zip(xyz, normals):
            out.insert(list_2_point(p), Vector_3(n[0], n[1], n[2]))
        surface_poly = Polyhedron_3()
        return poisson_surface_reconstruction(out, surface_poly)

    def _get_mesh_2(self):
        pass

    def _get_basis(self, order: int, degree: int) -> callable:
        return lambda theta, phi: sph_harm(order, degree, theta, phi)

    def _calculate(self, spine_mesh: Polyhedron_3) -> np.ndarray:
        v = spine_mesh.vertices()
        points = [point_2_list(vertex.point()) for vertex in v]

        center = (np.min(points, axis=0) + np.max(points, axis=0)) / 2
        radius_theta_psi = [cart2polar(*(p - center)) for p in points]

        a_matrix = [[real(b(*point[1:])) for b in self._basis] for point in radius_theta_psi]
        b_matrix = [point[0] for point in radius_theta_psi]
        res = np.linalg.lstsq(a_matrix, b_matrix)[0]
        self._value = res.tolist()
        return res

    @classmethod
    def get_distribution(cls, metrics: List["SpineMetric"]) -> np.ndarray:
        pass

    @classmethod
    def _show_distribution(cls, metrics: List["SpineMetric"]) -> None:
        pass
