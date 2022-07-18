import math
from abc import abstractmethod
from typing import List, Tuple

import numpy as np
from ipywidgets import widgets
from matplotlib import pyplot as plt

from CGAL.CGAL_AABB_tree import AABB_tree_Polyhedron_3_Facet_handle
from CGAL.CGAL_Kernel import Point_3, Ray_3
from CGAL.CGAL_Polygon_mesh_processing import area, face_area
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3, Polyhedron_3_Halfedge_handle, Polyhedron_3_Facet_handle
from spine_analysis.shape_metric.metric_core import SpineMetric
from spine_analysis.shape_metric.utils import _point_2_vec, _vec_2_point, _calculate_facet_center


class HistogramSpineMetric(SpineMetric):
    num_of_bins: int
    distribution: np.array

    def __init__(self, spine_mesh: Polyhedron_3 = None, num_of_bins: int = 10) -> None:
        self.num_of_bins = num_of_bins
        super().__init__(spine_mesh)

    @SpineMetric.value.setter
    def value(self, new_value: List[float]) -> None:
        self.num_of_bins = len(new_value)
        self._value = new_value

    def show(self) -> widgets.Widget:
        out = widgets.Output()

        with out:
            left_edges = [i / len(self._value) for i in range(len(self._value))]
            width = 0.85 * (left_edges[1] - left_edges[0])
            plt.bar(left_edges, self._value, align='edge', width=width)
            plt.show()

        return out

    @classmethod
    def get_distribution(cls, metrics: List["SpineMetric"]) -> np.ndarray:
        histograms = np.asarray([metric.value for metric in metrics])
        return np.mean(histograms, 0)

    @classmethod
    def _show_distribution(cls, metrics: List["SpineMetric"]) -> None:
        value = cls.get_distribution(metrics)
        left_edges = [i / len(value) for i in range(len(value))]
        width = 0.85 * (left_edges[1] - left_edges[0])
        plt.bar(left_edges, value, align='edge', width=width)

    @abstractmethod
    def _calculate_distribution(self, spine_mesh: Polyhedron_3) -> np.array:
        pass

    def _calculate(self, spine_mesh: Polyhedron_3) -> np.array:
        self.distribution = self._calculate_distribution(spine_mesh)
        return np.histogram(self.distribution, self.num_of_bins, density=True)[0]


class ChordDistributionSpineMetric(HistogramSpineMetric):
    num_of_chords: int
    chords: List[Tuple[Point_3, Point_3]]
    chord_lengths: List[float]
    relative_max_facet_area: float

    def __init__(self, spine_mesh: Polyhedron_3 = None, num_of_chords: int = 3000,
                 num_of_bins: int = 100, relative_max_facet_area: float = 0.001) -> None:
        self.num_of_chords = num_of_chords
        self.relative_max_facet_area = relative_max_facet_area
        super().__init__(spine_mesh, num_of_bins)

    @staticmethod
    def _get_incident_halfedges(facet_halfedge: Polyhedron_3_Halfedge_handle) -> List[Polyhedron_3_Halfedge_handle]:
        return [facet_halfedge, facet_halfedge.next(), facet_halfedge.next().next()]

    @staticmethod
    def _get_side_centers(facet_halfedge: Polyhedron_3_Halfedge_handle) -> List[Polyhedron_3_Halfedge_handle]:
        return [facet_halfedge, facet_halfedge.next(), facet_halfedge.next().next()]

    @staticmethod
    def _is_triangle(facet: Polyhedron_3_Facet_handle) -> bool:
        circulator = facet.facet_begin()
        begin = facet.facet_begin()
        i = 0
        while circulator.hasNext():
            i += 1
            circulator.next()
            if circulator == begin:
                break
        return i == 3

    def _subdivide_mesh(self, mesh: Polyhedron_3,
                        relative_max_facet_area: float = 0.001) -> Polyhedron_3:
        out: Polyhedron_3 = mesh.deepcopy()
        subdivision_number = 3

        for i in range(subdivision_number):
            # split every edge
            center_vertices = set()
            edges = [edge for edge in out.edges()]
            for edge in edges:
                first_half: Polyhedron_3_Halfedge_handle = out.split_edge(edge)
                center_vertices.add(first_half.vertex())
                a = _point_2_vec(first_half.vertex().point())
                b = _point_2_vec(edge.vertex().point())
                center = (a + b) / 2
                first_half.vertex().set_point(_vec_2_point(center))
            # create center triangles
            facets = [facet for facet in out.facets()]
            for j, facet in enumerate(facets):
                halfedge = facet.halfedge()
                if halfedge.vertex() not in center_vertices:
                    halfedge = halfedge.next()
                centers = [halfedge, halfedge.next().next(),
                           halfedge.next().next().next().next()]
                new_side = out.split_facet(centers[0], centers[1])
                new_side = out.split_facet(new_side, centers[2])
                new_side = out.split_facet(new_side, centers[0])
                new_side.facet().set_id(halfedge.facet().id())
        return out

    def _calculate_raycast(self, ray_query: Ray_3,
                           tree: AABB_tree_Polyhedron_3_Facet_handle) -> None:
        intersections = []
        tree.all_intersections(ray_query, intersections)

        origin = _point_2_vec(ray_query.source())

        # sort intersections along the ray
        intersection_points = [_calculate_facet_center(intersection.second)
                               for intersection in intersections]
        intersection_points.sort(key=lambda point: (point - origin).squared_length())

        # remove doubles
        i = 0
        while i < len(intersection_points) - 1:
            if (intersection_points[i] - intersection_points[i + 1]).squared_length() < 0.0000001:
                del intersection_points[i]
            else:
                i += 1

        # if len(intersection_points) % 2 != 0:
        #     i = 0
        #     while i < len(intersection_points) - 1:
        #         if intersection_points[i] == intersection_points[i + 1]:
        #             del intersection_points[i]
        #         else:
        #             i += 1
        #     x = 30

        for j in range(1, len(intersection_points), 2):
            center_1 = intersection_points[j - 1]
            center_2 = intersection_points[j]

            # # check intersections are in correct order along the ray
            # dist = np.sqrt((center_1 - origin).squared_length())
            # if dist > prev_dist:
            #     prev_dist = dist
            #     continue
            # prev_dist = dist

            length = np.sqrt((center_2 - center_1).squared_length())
            if length == 0:
                j -= 1
                continue
            self.chord_lengths.append(length)

            self.chords.append(
                (_vec_2_point(center_1), _vec_2_point(center_2)))

    def _calculate_distribution(self, spine_mesh: Polyhedron_3) -> np.array:
        self.chord_lengths = []
        self.chords = []

        subdivided_spine_mesh = self._subdivide_mesh(spine_mesh, self.relative_max_facet_area)

        tree = AABB_tree_Polyhedron_3_Facet_handle(subdivided_spine_mesh.facets())

        facets = [[]] * spine_mesh.size_of_facets()
        for facet in subdivided_spine_mesh.facets():
            facets[facet.id()].append(facet)

        # surface_points = [self._calculate_facet_center(facet) for facet in facets]

        rand = Random()
        for i in range(self.num_of_chords):
            ind1 = rand.randrange(0, len(facets))
            ind2 = rand.randrange(0, len(facets))
            while ind1 == ind2:
                ind2 = rand.randrange(0, len(facets))
            f1 = facets[ind1][rand.randrange(0, len(facets[ind1]))]
            f2 = facets[ind2][rand.randrange(0, len(facets[ind2]))]
            p1 = _calculate_facet_center(f1)
            p2 = _calculate_facet_center(f2)
            direction = p2 - p1
            direction.normalize()

            ray_query = Ray_3(_vec_2_point(p1 - direction * 5000),
                              _vec_2_point(p2 + direction * 5000))
            self._calculate_raycast(ray_query, tree)

        # # find bounding box
        # points = [point_2_list(point) for point in spine_mesh.points()]
        # min_coord = np.min(points, axis=0)
        # max_coord = np.max(points, axis=0)
        #
        # min_coord -= [0.1, 0.1, 0.1]
        # max_coord += [0.1, 0.1, 0.1]
        #
        # step = (max_coord - min_coord) / (self.num_of_chords + 1)
        #
        # for x in range(1, self.num_of_chords + 1):
        #     for y in range(1, self.num_of_chords + 1):
        #         # ray generation
        #         # left to right
        #         ray_query = Ray_3(Point_3(min_coord[0], min_coord[1] + x * step[1], min_coord[2] + y * step[2]),
        #                           Point_3(max_coord[0], min_coord[1] + x * step[1], min_coord[2] + y * step[2]))
        #         self._calculate_raycast(ray_query, tree)
        #         # down to up
        #         ray_query = Ray_3(Point_3(min_coord[0] + x * step[0], min_coord[1], min_coord[2] + y * step[2]),
        #                           Point_3(min_coord[0] + x * step[0], max_coord[1], min_coord[2] + y * step[2]))
        #         self._calculate_raycast(ray_query, tree)
        #         # backward to forward
        #         ray_query = Ray_3(Point_3(min_coord[0] + x * step[0], min_coord[1] + y * step[1], min_coord[2]),
        #                           Point_3(min_coord[0] + x * step[0], min_coord[1] + y * step[1], max_coord[2]))
        #         self._calculate_raycast(ray_query, tree)

        max_len = np.max(self.chord_lengths)
        self.chord_lengths = np.asarray(self.chord_lengths) / max_len

        return self.chord_lengths


class OldChordDistributionSpineMetric(ChordDistributionSpineMetric):
    def _subdivide_mesh(self, mesh: Polyhedron_3,
                        relative_max_facet_area: float = 0.001) -> Polyhedron_3:
        out: Polyhedron_3 = mesh.deepcopy()
        for i, facet in enumerate(out.facets()):
            facet.set_id(i)

        facets = [facet for facet in out.facets()]
        total_area = area(out)

        for facet in facets:
            facet_area = face_area(facet, out)
            relative_area = facet_area / total_area

            # facet already small enough
            if relative_area <= relative_max_facet_area:
                continue

            subdivision_number = int(np.ceil(math.log(relative_area / relative_max_facet_area, 3)))
            triangles: List[Polyhedron_3_Halfedge_handle] = [facet.halfedge()]
            for i in range(subdivision_number):
                new_triangles = []
                for halfedge in triangles:
                    new_triangles.extend(self._get_incident_halfedges(halfedge))
                    center = _vec_2_point(_calculate_facet_center(halfedge.facet()))
                    new_v = out.create_center_vertex(halfedge).vertex()
                    new_v.set_point(center)
                triangles = new_triangles

        return out
