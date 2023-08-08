import ast
import dataclasses
import os
import tempfile
from abc import ABC, abstractmethod
from math import factorial
from typing import List, Iterable, Any, Tuple, Dict

import icosphere
import cv2
import matplotlib.pyplot as plt
import numpy as np
import meshplot as mp
from ipywidgets import widgets
from numpy import real
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.special import sph_harm

from CGAL.CGAL_AABB_tree import AABB_tree_Polyhedron_3_Facet_handle
from CGAL.CGAL_Kernel import Ray_3, Point_3
from CGAL.CGAL_Polygon_mesh_processing import Polylines
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from CGAL.CGAL_Surface_mesh_skeletonization import surface_mesh_skeletonization
from spine_analysis.mesh.utils import write_off
from spine_analysis.shape_metric.metric_core import SpineMetric
from spine_analysis.shape_metric.utils import polar2cart, cart2polar, _point_2_list, get_enclosing_circle, _vec_2_point, \
    _calculate_facet_center, subdivide_mesh
from spine_segmentation import point_2_list


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


@dataclasses.dataclass
class SGSample:
    az: float
    elev: float
    sg_coefficients: List[float]


class SphericalGarmonicsSpineMetric(ApproximationSpineMetric):
    DEFAULT_L_SIZE: int = 10
    _m_l_map: Dict[int, Tuple[int, int]]
    _mc_samples: List[SGSample] = None

    def __init__(self, spine_mesh: Polyhedron_3 = None, l_range: Iterable = None, sqrt_sample_size: int = 100):
        if l_range is None:
            l_range = range(self.DEFAULT_L_SIZE)
        self._m_l_map = {}
        self._basis = []
        i = 0
        for _l in l_range:
            for m in range(-_l, _l + 1):
                self._m_l_map[i] = (m, _l)
                self._basis.append(self._get_basis(m, _l))
                i += 1

        if SphericalGarmonicsSpineMetric._mc_samples is None:
            SphericalGarmonicsSpineMetric._mc_samples = self._generate_sample(sqrt_n=sqrt_sample_size)  # monte carlo sample on sphere
        super().__init__(spine_mesh)

    @property
    def m_l_map(self):
        return dict(self._m_l_map)

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

    def _get_mesh_2(self):
        pass

    def _get_basis(self, order: int, degree: int) -> callable:
        if order > 0:
            return lambda az, elev: sph_harm(order, degree, az, elev).real * np.sqrt(2)
        elif order < 0:
            return lambda az, elev: sph_harm(-order, degree, az, elev).imag * np.sqrt(2)
        else:
            return lambda az, elev: sph_harm(order, degree, az, elev).real

    def _generate_sample(self, sqrt_n: int) -> List[SGSample]:
        oneoverN = 1 / sqrt_n
        result = []
        r = np.random.random(sqrt_n*sqrt_n*2)
        for a in range(sqrt_n):
            for b in range(sqrt_n):
                x, y = a + r[b * 2 + a * 2 * sqrt_n], b + r[b * 2 + a * 2 * sqrt_n + 1]
                x, y = x * oneoverN, y * oneoverN
                elev = 2 * np.arccos(np.sqrt(1-x))
                az = 2 * np.pi * y
                result.append(SGSample(az, elev, [basis_f(az, elev) for basis_f in self._basis]))
        return result

    def _calculate(self, spine_mesh: Polyhedron_3) -> np.ndarray:
        subdivided_mesh = subdivide_mesh(spine_mesh, relative_max_facet_area=0.0003)
        v = subdivided_mesh.vertices()
        points = [point_2_list(vertex.point()) for vertex in v]

        def perpendicular(line, point):
            P1 = np.array(point_2_list(line[0]))
            P2 = np.array(point_2_list(line[1]))
            t = np.dot(point - P1, P2 - P1) / (np.linalg.norm(P2 - P1) ** 2)
            return P1 + t * (P2 - P1)

        skeleton_polylines = Polylines()
        correspondence_polylines = Polylines()
        surface_mesh_skeletonization(subdivided_mesh, skeleton_polylines, correspondence_polylines)
        center = np.array((np.min(points, axis=0) + np.max(points, axis=0)) / 2)
        inner_center = center
        min_dist = 1 << 20
        for line in skeleton_polylines:
            for i in range(len(line) - 1):
                p = perpendicular((line[i], line[i + 1]), center)
                d = np.linalg.norm(p - center)
                if d < min_dist:
                    inner_center = p
                    min_dist = d

        center = inner_center
        radius_elev_az = np.array([cart2polar(*(p - center)) for p in points])
        max_radius = max(radius_elev_az[:, 0]) * 2
        mean_radius = np.mean(radius_elev_az[:, 0])

        def intersecton_point(facet, ray):
            a_matrix = []
            circulator = facet.facet_begin()
            begin = facet.facet_begin()
            while circulator.hasNext():
                halfedge = circulator.next()
                pnt = halfedge.vertex().point()
                a_matrix.append([pnt.x(), pnt.y(), pnt.z()])
                # check for end of loop
                if circulator == begin:
                    break
            b_matrix = [-1] * len(a_matrix)
            plane_coefs = np.linalg.lstsq(a_matrix, b_matrix)[0]
            start, end = ray.point(0), ray.point(1)
            line_equations = [[end.y() - start.y(), start.x() - end.x(), 0],
                              [0, end.z() - start.z(), start.y() - end.y()],
                              [*plane_coefs]]
            b2_matrix = [start.x() * (end.y() - start.y()) + start.y() * (start.x() - end.x()),
                         start.y() * (end.z() - start.z()) + start.z() * (start.y() - end.y()),
                         -1]
            return np.linalg.lstsq(line_equations, b2_matrix)[0]

        def estimate_mesh(az, elev):
            intersections = []
            origin = np.array([0, 0, 0])
            tree = AABB_tree_Polyhedron_3_Facet_handle(subdivided_mesh.facets())
            ray = Ray_3(Point_3(*(origin + center)), Point_3(*(polar2cart(az, elev, max_radius) + center)))
            tree.all_intersections(ray, intersections)
            if len(intersections) != 1:
                print(f"WARNING intersections len = {len(intersections)}")
            # else:
            #     print("OK")
            if len(intersections) < 1:
                return mean_radius
            point = intersecton_point(intersections[0].second, ray)
            r, elev_est, az_est = cart2polar(*(point - center))
            return r

        res = np.zeros(len(self._basis))
        mesh_sample = [estimate_mesh(sample.az, sample.elev) for sample in SphericalGarmonicsSpineMetric._mc_samples]
        factor = 4 * np.pi / len(mesh_sample)
        for n in range(len(self._basis)):
            res[n] = sum(estim * sample.sg_coefficients[n] for estim, sample in
                         zip(mesh_sample, SphericalGarmonicsSpineMetric._mc_samples)) * factor

        return res

    def show(self, **kwargs) -> widgets.Widget:
        mesh, v, f = self.get_basis_composition()
        viewer = mp.Viewer({"width": 200, "height": 200})
        viewer.add_mesh(v, f)
        viewer._renderer.layout = widgets.Layout(border="solid 1px")
        return viewer._renderer

    @classmethod
    def get_distribution(cls, metrics: List["SpineMetric"]) -> np.ndarray:
        pass

    @classmethod
    def _show_distribution(cls, metrics: List["SpineMetric"]) -> None:
        pass

    def get_basis_composition(self) -> Polyhedron_3:

        def composition_callback(az, elev):
            return sum(a_i * self._basis[i](az, elev) for i, a_i in enumerate(self.value))

        v, c = icosphere.icosphere(10)
        rad, elev, az = cart2polar(v[:, 0], v[:, 1], v[:, 2])

        rad = np.array([composition_callback(a_i, e_i) for a_i, e_i in zip(az, elev)])
        v = polar2cart(az, elev, rad)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, 'tmp.off')
            with open(file_path, 'w') as fd:
                write_off(fd, v, c)
            mesh = Polyhedron_3(file_path)
        return mesh, v, c

    def parse_value(self, value_str: str):
        value_str = value_str.replace('\n', '')
        value_str = value_str.replace('  ', ' ')
        try:
            value = ast.literal_eval(value_str)
        except SyntaxError:
            value = np.fromstring(value_str[1:-1], dtype="float", sep=" ")
        self.value = value

    def value_as_lists(self) -> List[Any]:
        return [*self.value]


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
            self._value.append(self._calculate_moment(projection, degree=self._zernike_order).tolist())
        return self._value
    
    def _get_rotation(self, normal):
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
        normal[0] = x

        # create transformation matrix
        rotation_matrix = np.array([[1, 0, 0], [0, cos_ax, -sin_ax], [0, sin_ax, cos_ax]])
        rotation_matrix = np.matmul(np.array([[cos_ay, 0, sin_ay], [0, 1, 0], [-sin_ay, 0, cos_ay]]), rotation_matrix)
        return rotation_matrix

    def _get_image(self, poly_contour: Polygon):
        contour = np.array(poly_contour.exterior.coords)

        cx, cy, radius = get_enclosing_circle(contour)
        scale = 100 / radius

        contour = np.matmul([[scale, 0], [0, scale]], contour.T).T
        contour[:, 0] = contour[:, 0] - cx * scale + 100
        contour[:, 1] = contour[:, 1] - cy * scale + 100

        contour = contour.astype(int)
        mask = np.zeros((200, 200))
        cv2.fillPoly(mask, pts=[contour], color=(255,255,255))
        return mask
    
    def get_projections(self, spine_mesh: Polyhedron_3) -> Iterable[np.ndarray]:
        result = []
        #mp.plot(*_mesh_to_v_f(spine_mesh))
        #fig, ax = plt.subplots(ncols=2, nrows=(len(self._view_points) + 1) // 2, figsize=(12, 6 * (len(self._view_points) + 1) // 2))
        for i in range(len(self._view_points)):
            normal = polar2cart(self._view_points[i, 0, ...], self._view_points[i, 1, ...], self._view_points[i, 2, ...])
            contour = self._get_contour(normal, spine_mesh)
            result.append(self._get_image(contour))
            #ax[i // 2, i % 2].imshow(result[-1])
            #ax[i // 2, i % 2].set_title(f'view_point#{i}')
        #plt.savefig("projectionsss.pdf", dpi=600)
        #plt.show()
        return result
    
    def _get_contour(self, normal, mesh: Polyhedron_3):
        rotation_matrix = self._get_rotation(normal)
        res_poly = MultiPolygon()
        facet_points = map(lambda facet: np.array([_point_2_list(h.vertex().point()) for h in [facet.halfedge(), facet.halfedge().next(), facet.halfedge().next().next()]]), mesh.facets())
        facet_points = map(lambda points: np.matmul(points, rotation_matrix)[...,[0,1]], facet_points)
        
        for facet_2d in facet_points:
            res_poly = res_poly.union(Polygon(facet_2d).buffer(0))
            if type(res_poly) is Polygon:
                res_poly = MultiPolygon([res_poly])
            res_poly = make_valid(res_poly)
        
        res_poly = max(res_poly.geoms, key=lambda a: a.area)
        return res_poly


    @staticmethod
    def distance(mesh_descr1: "LightFieldZernikeMomentsSpineMetric", mesh_descr2: "LightFieldZernikeMomentsSpineMetric") -> float:
        return LightFieldZernikeMomentsSpineMetric.repr_distance(np.array(mesh_descr1._value), np.array(mesh_descr2._value))

    @staticmethod
    def repr_distance(data1: np.ndarray, data2: np.ndarray):
        if data1.ndim != 2:
            view_points_squared = 25
            data1 = data1.reshape(view_points_squared, int(data1.shape[0]/view_points_squared))
            data2 = data2.reshape(view_points_squared, int(data2.shape[0]/view_points_squared))
        cost_matrix = [[distance.cityblock(m1, m2) if not(np.isnan(m2).any() or np.isnan(m1).any()) else 0 for m2 in data1] for m1 in data2]
        m2_ind, m1_ind = linear_sum_assignment(cost_matrix)
        return sum(distance.cityblock(data2[m2_i], data1[m1_i]) for m2_i, m1_i in zip(m2_ind, m1_ind))

    @classmethod
    def get_distribution(cls, metrics: List["SpineMetric"]) -> np.ndarray:
        spikes_deskr = np.asarray([metric.value for metric in metrics])
        return np.mean(spikes_deskr, 0)

    @classmethod
    def _show_distribution(cls, metrics: List["SpineMetric"]) -> None:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot(cls.get_distribution(metrics))

    def parse_value(self, value_str):
        value = ast.literal_eval(value_str)
        self.value = value

    def value_as_lists(self) -> List[Any]:
        return [*self.value]

    def clasterization_preprocess(self, zernike_postprocess='real', **kwargs) -> Any:
        self.value = [[m.real if zernike_postprocess == 'real' else abs(m) for m in moments] for moments in self._value]

    def show(self, image_size: int = 30) -> widgets.Widget:
        out = widgets.Output()
        # with out:
        #     fig, ax = plt.subplots(ncols=2, nrows=(len(self.value) + 1) // 2, figsize=(12, 6 * (len(self.value) + 1) // 2))
        #     for i, projection in enumerate(self.value):
        #         ax[i // 2, i % 2].imshow(self._recover_projection(projection, image_size))
        #     plt.savefig("zernike_moments_images.pdf", dpi=600)
        #     #plt.show()
        return out

    @staticmethod
    def _recover_projection(zernike_moments: List[float], image_size: int):
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
                    vxy = LightFieldZernikeMomentsSpineMetric.zernike_poly(Y, X, n, l)
                    computed += zernike_moments[i] * vxy
                    i += 1
            n += 1
        #computed = computed - min(computed) #/ (max(computed) - min(computed))
        result[circle_mask] = computed
        return result.reshape((image_size,)*2).real

    @staticmethod
    def zernike_poly(Y, X, n, l):
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

