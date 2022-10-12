import ast
from abc import ABC, abstractmethod
from math import factorial
from typing import List, Iterable, Any, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import meshplot as mp
from ipywidgets import widgets
from lfd import LightFieldDistance
from numpy import real
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.special import sph_harm
from sklearn.linear_model import LinearRegression

from CGAL.CGAL_Kernel import Vector_3, Plane_3, Point_3
from CGAL.CGAL_Point_set_3 import Point_set_3
from CGAL.CGAL_Poisson_surface_reconstruction import poisson_surface_reconstruction
from CGAL.CGAL_Polygon_mesh_processing import Polygon_mesh_slicer, Polylines
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from spine_analysis.mesh.utils import _mesh_to_v_f
from spine_analysis.shape_metric.metric_core import SpineMetric
from spine_analysis.shape_metric.utils import polar2cart, cart2polar, _calculate_junction_center, _vec_2_list, \
    _point_2_list, calculate_surface_center
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
        x, y, z = self._get_mesh(steps_num)
        print(f'x: {x.min()}-{x.max()}, y: {y.min()}-{y.max()}, z: {z.min()}-{z.max()}')
        out = widgets.Output()
        with out:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot_surface(x, y, z)
            plt.show()
        return out

    @abstractmethod
    def _get_mesh(self, steps_num: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def _get_mesh(self, steps_num: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta = np.linspace(0, np.pi, steps_num[0])
        phi = np.linspace(0, 2 * np.pi, steps_num[1])
        T, P = np.meshgrid(theta, phi)
        radiuses = np.array([[real(self.surface_value([t, p])) for t, p in zip(t_values, p_values)] for t_values, p_values in zip(T, P)])
        xyz = polar2cart(T, P, radiuses)
        return xyz[..., 0], xyz[..., 1], xyz[..., 2]
        # normals = [self.surface_norm([t, p]) for t, p in tp]
        # out: Point_set_3 = Point_set_3()
        # out.add_normal_map()
        # for p, n in zip(xyz, normals):
        #     out.insert(list_2_point(p), Vector_3(n[0], n[1], n[2]))
        # surface_poly = Polyhedron_3()
        # poisson_surface_reconstruction(out, surface_poly)
        # return surface_poly

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


# TODO: move from Approximation class
class LightFieldZernikeMomentsSpineMetric(SpineMetric):
    def __init__(self, spine_mesh: Polyhedron_3 = None, radius: int = 1, view_points: int = 5, order: int = 15):
        sphere_iteratable = list(range(5, 175, 170 // (view_points-1)))
        self._view_points = np.array([[phi, theta, radius*2]
                                      for phi in sphere_iteratable for theta in sphere_iteratable])
        self._zernike_radius = radius
        self._zernike_order = order
        super().__init__(spine_mesh)

    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        self._value = []
        for projection in self.get_projections(spine_mesh):
            self._value.append(self._calculate_moment(projection, degree=self._zernike_order).real.tolist())
        return self._value

    def _get_image(self, normal: list, polyline: Tuple[Point_3]):
        # prepare rotation angle for coordinate transformation
        z_0 = [0, 0, 1]
        x, y = normal[0], normal[1]
        normal[1] = 0
        norm = np.linalg.norm(normal)
        sin_ay, cos_ay = np.linalg.norm(np.cross(normal, z_0)) / norm, np.dot(normal, z_0) / norm
        if normal[0] < 0:
            sin_ay = -sin_ay

        normal[1] = y
        normal[0] = 0
        norm = np.linalg.norm(normal)
        sin_ax, cos_ax = np.linalg.norm(np.cross(normal, z_0)) / norm, np.dot(normal, z_0) / norm
        if normal[1] > 0:
            sin_ax = -sin_ax

        # create transformation matrix
        rotation_matrix = np.array([[1, 0, 0], [0, cos_ax, -sin_ax], [0, sin_ax, cos_ax]])
        rotation_matrix = np.matmul(np.array([[cos_ay, 0, sin_ay], [0, 1, 0], [-sin_ay, 0, cos_ay]]), rotation_matrix)

        contour = np.array([np.matmul(rotation_matrix, _point_2_list(p))[:-1] for p in polyline])
        min_points = contour[..., 0].min(), contour[..., 1].min()
        for i in [0, 1]:
            contour[..., i] = (contour[..., i] - contour[..., i].min()) / (contour[..., i].max() - min_points[i]) * 199
        contour = contour.astype(int)
        mask = np.zeros((200, 200))
        cv2.fillPoly(mask, pts=[contour], color=(255,255,255))
        return mask
    
    def get_projections(self, spine_mesh: Polyhedron_3) -> Iterable[np.ndarray]:
        result = []
        slicer = Polygon_mesh_slicer(spine_mesh)
        #mp.plot(*_mesh_to_v_f(spine_mesh))
        #fig, ax = plt.subplots(ncols=2, nrows=(len(self._view_points) + 1) // 2, figsize=(12, 6 * (len(self._view_points) + 1) // 2))
        for i in range(len(self._view_points)):
            sliced = Polylines()
            normal = polar2cart(self._view_points[i, 0, ...], self._view_points[i, 1, ...], self._view_points[i, 2, ...])
            center = calculate_surface_center(spine_mesh)
            plane = Plane_3(normal[0], normal[1], normal[2], -np.dot(center, normal))
            slicer.slice(plane, sliced)
            #p.add_mesh(np.array([point_2_list(plane.projection(Point_3(0, 1, 1))), point_2_list(plane.projection(Point_3(0, 1, -1))),
            #          point_2_list(plane.projection(Point_3(-1, 0, 1)))]), np.array([[0,1,2]]))
            result.append(self._get_image(normal, sliced[sliced.size() // 2]))
            #ax[i // 2, i % 2].imshow(result[-1])
            #ax[i // 2, i % 2].set_title(f'view_point#{i}')
        plt.savefig("projections.pdf", dpi=600)
        #plt.show()
        return result

    @staticmethod
    def distance(mesh_descr1: "LightFieldZernikeMomentsSpineMetric", mesh_descr2: "LightFieldZernikeMomentsSpineMetric") -> float:
        return LightFieldZernikeMomentsSpineMetric.repr_distance(mesh_descr1._value, mesh_descr2._value)

    @staticmethod
    def repr_distance(data1: np.ndarray, data2: np.ndarray):
        if data1.ndim != 2:
            view_points = int(np.sqrt(data1.shape[0] / 25))
            data1 = data1.reshape(view_points*view_points, 25)
            data2 = data2.reshape(view_points * view_points, 25)
        cost_matrix = [[distance.cityblock(m1, m2) for m2 in data1] for m1 in data2]
        m2_ind, m1_ind = linear_sum_assignment(cost_matrix)
        return sum(distance.cityblock(data2[m2_i], data1[m1_i]) for m2_i, m1_i in zip(m2_ind, m1_ind))

    @staticmethod
    def lf_module_distance(mesh1: Polyhedron_3, mesh2: Polyhedron_3) -> float:
        return LightFieldDistance(verbose=True).get_distance(mesh1.vertices(), mesh1.facets(),
                                                             mesh2.vertices(), mesh2.facets())

    @classmethod
    def get_distribution(cls, metrics: List["SpineMetric"]) -> np.ndarray:
        pass

    @classmethod
    def _show_distribution(cls, metrics: List["SpineMetric"]) -> None:
        pass

    def parse_value(self, value_str):
        value = ast.literal_eval(value_str)
        self.value = value

    def show(self, image_size: int = 256) -> widgets.Widget:
        out = widgets.Output()
        with out:
            fig, ax = plt.subplots(ncols=2, nrows=(len(self.value) + 1) // 2, figsize=(12, 6 * (len(self.value) + 1) // 2))
            for i, projection in enumerate(self.value):
                ax[i // 2, i % 2].imshow(self._recover_projection(projection, image_size))
            plt.savefig("zernike_moments_images.pdf", dpi=600)
            #plt.show()
        return out

    def _recover_projection(self, zernike_moments: List[float], image_size: int):
        radius = image_size // 2
        Y, X = np.meshgrid(range(image_size), range(image_size))
        Y, X = ((Y - radius)/radius).ravel(), ((X - radius)/radius).ravel()

        circle_mask = (np.sqrt(X ** 2 + Y ** 2) <= 1)
        result = np.zeros(len(circle_mask), dtype=complex)
        Y, X = Y[circle_mask], X[circle_mask]
        computed = np.zeros(len(Y), dtype=complex)
        i = 0
        n = 0
        while i < len(zernike_moments):
            for l in range(n + 1):
                if (n - l) % 2 == 0:
                    vxy = self.zernike_poly(Y, X, n, l)
                    computed += zernike_moments[i] * vxy
                    i += 1
            n += 1
        #computed = computed - min(computed) #/ (max(computed) - min(computed))
        result[circle_mask] = computed
        return result.reshape((image_size,)*2).real

    def zernike_poly(self, Y, X, n, l):
        l = abs(l)
        if (n - l) % 2 == 1:
            return np.zeros(len(Y), dtype=complex)
        Rho, _, Phi = cart2polar(X, Y, np.zeros(len(X)))
        multiplier = (1.*np.cos(Phi) + 1.j*np.sin(Phi)) ** l
        radial = np.sum([(-1.) ** m * factorial(n - m) /
                         (factorial(m) * factorial((n + l - 2 * m) // 2) * factorial((n - l - 2 * m) // 2)) *
                         np.power(Rho, n - 2 * m)
                         for m in range(int((n - l) // 2 + 1))],
                        axis=0)
        return radial * multiplier

    def _calculate_moment(self, image: np.ndarray, degree: int = 8):
        radius = image.shape[0] // 2
        moments = []
        Y, X = np.meshgrid(range(image.shape[0]), range(image.shape[1]))
        Y, X = ((Y - radius) / radius).ravel(), ((X - radius) / radius).ravel()

        circle_mask = (np.sqrt(X ** 2 + Y ** 2) <= 1)
        Y, X = Y[circle_mask], X[circle_mask]

        frac_center = np.array(image.ravel()[circle_mask], np.double)
        frac_center /= frac_center.sum()

        for n in range(degree + 1):
            for l in range(n + 1):
                if (n - l) % 2 == 0:
                    vxy = self.zernike_poly(Y, X, n, l)
                    moments.append(sum(frac_center * np.conjugate(vxy)) * (n + 1)/np.pi)

        return np.array(moments)

