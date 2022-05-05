import math
from abc import ABC, abstractmethod
from random import Random
import numpy as np
from typing import Any, List, Tuple, Dict
import ipywidgets as widgets
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3, Polyhedron_3_Facet_handle, \
    Polyhedron_3_Halfedge_handle, Polyhedron_3_Vertex_handle, Polyhedron_3_Edge_iterator
from CGAL.CGAL_Polygon_mesh_processing import area, face_area, volume
from CGAL.CGAL_Kernel import Ray_3, Point_3, Vector_3, cross_product
from CGAL.CGAL_AABB_tree import AABB_tree_Polyhedron_3_Facet_handle
from matplotlib import pyplot as plt
import csv
from CGAL.CGAL_Convex_hull_3 import convex_hull_3
from copy import deepcopy


def _calculate_facet_center(facet: Polyhedron_3_Facet_handle) -> Vector_3:
    circulator = facet.facet_begin()
    begin = facet.facet_begin()
    center = Vector_3(0, 0, 0)
    while circulator.hasNext():
        halfedge = circulator.next()
        pnt = halfedge.vertex().point()
        center += Vector_3(pnt.x(), pnt.y(), pnt.z())
        # check for end of loop
        if circulator == begin:
            break
    center /= 3
    return center


def _vec_2_point(vector: Vector_3) -> Point_3:
    return Point_3(vector.x(), vector.y(), vector.z())


def _point_2_vec(point: Point_3) -> Vector_3:
    return Vector_3(point.x(), point.y(), point.z())


class SpineMetric(ABC):
    name: str
    _value: Any

    def __init__(self, spine_mesh: Polyhedron_3 = None) -> None:
        self.name = type(self).__name__.replace("SpineMetric", "")
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

    def value_as_list(self) -> List[Any]:
        try:
            return [*self.value]
        except TypeError:
            return [self.value]


class SpineMetricDataset:
    num_of_spines: int
    num_of_metrics: int
    _spine_2_row: Dict[str, int]
    _metric_2_column: Dict[str, int]
    _table = np.ndarray

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

    def get_subset(self, reduced_metric_names: List[str]) -> "SpineMetricDataset":
        index_subset = [self._metric_2_column[metric_name]
                        for metric_name in reduced_metric_names]
        reduced_metrics = {}
        for (spine_name, spine_metrics) in zip(self.spine_names, self._table):
            reduced_metrics[spine_name] = [spine_metrics[i] for i in index_subset]

        return SpineMetricDataset(reduced_metrics)

    def standardize(self) -> None:
        float_metric_indices = [i for i in range(self.num_of_metrics)
                                if isinstance(self._table[0, i], FloatSpineMetric)]

        # calculate mean and std by column
        mean = {}
        std = {}
        for i in float_metric_indices:
            column_values = [metric.value for metric in self._table[:, i]]
            mean[i] = np.mean(column_values)
            std[i] = np.std(column_values)

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


def calculate_metrics(spine_mesh: Polyhedron_3,
                      metric_names: List[str], params: List[Dict] = None) -> List[SpineMetric]:
    if params is None:
        params = [{}] * len(metric_names)

    out = []
    for i, name in enumerate(metric_names):
        klass = globals()[name + "SpineMetric"]
        out.append(klass(spine_mesh, **params[i]))
    return out


SPINE_FILE_FIELD = "Spine File"


def save_metrics(metrics: SpineMetricDataset, filename: str) -> None:
    with open(filename, mode="w") as file:
        if metrics.num_of_spines == 0:
            return
        # extract from metric names from first spine
        metric_names = metrics.metric_names

        # save metrics for each spine
        writer = csv.DictWriter(file, fieldnames=[SPINE_FILE_FIELD] + metric_names)
        writer.writeheader()
        for spine_name in metrics.spine_names:
            writer.writerow({SPINE_FILE_FIELD: spine_name,
                             **{metric.name: metric.value
                                for metric in metrics.row(spine_name)}})


def load_metrics(filename: str) -> SpineMetricDataset:
    output = {}
    with open(filename, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # extract spine file name
            spine_name = row.pop(SPINE_FILE_FIELD)
            # extract each metric
            metrics = []
            for metric_name in row.keys():
                value_str = row[metric_name]
                value: Any
                if value_str[0] == "[":
                    value = np.fromstring(value_str[1:-1], dtype="float", sep=" ")
                else:
                    value = float(value_str)
                klass = globals()[metric_name + "SpineMetric"]
                metric = klass()
                metric._value = value
                metrics.append(metric)
            output[spine_name] = metrics
    return SpineMetricDataset(output)


class FloatSpineMetric(SpineMetric, ABC):
    def show(self) -> widgets.Widget:
        return widgets.Label(f"{self._value:.2f}")


class AreaSpineMetric(FloatSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        return area(spine_mesh)


class VolumeSpineMetric(FloatSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        return volume(spine_mesh)


class ConvexHullVolumeSpineMetric(FloatSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        hull_mesh = Polyhedron_3()
        convex_hull_3(spine_mesh.points(), hull_mesh)
        return volume(hull_mesh)


class ConvexHullRatioSpineMetric(FloatSpineMetric):
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        hull_mesh = Polyhedron_3()
        convex_hull_3(spine_mesh.points(), hull_mesh)
        v = volume(spine_mesh)
        return (volume(hull_mesh) - v) / v


class JunctionSpineMetric(FloatSpineMetric, ABC):
    _junction_center: Vector_3
    _surface_vectors: List[Vector_3]

    @abstractmethod
    def _calculate(self, spine_mesh: Polyhedron_3) -> Any:
        # identify junction triangles
        junction_triangles = set()
        for v in spine_mesh.vertices():
            if v.vertex_degree() > 15:
                # mark adjacent triangles
                for h in spine_mesh.halfedges():
                    if h.vertex() == v:
                        junction_triangles.add(h.facet())

        # calculate junction center
        if len(junction_triangles) > 0:
            self._junction_center = Vector_3(0, 0, 0)
            for facet in junction_triangles:
                self._junction_center += _calculate_facet_center(facet)
            self._junction_center /= len(junction_triangles)
        else:
            self._junction_center = _point_2_vec(spine_mesh.points().next())

        # calculate vectors to surface
        self._surface_vectors = []
        for point in spine_mesh.points():
            self._surface_vectors.append(_point_2_vec(point) - self._junction_center)


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
