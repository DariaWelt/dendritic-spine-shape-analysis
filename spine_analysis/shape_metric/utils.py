from typing import List, Dict, Any, Tuple

import numpy as np

from CGAL.CGAL_Kernel import Vector_3, Point_3
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3_Facet_handle, Polyhedron_3
from spine_analysis.shape_metric.metric_core import SpineMetric
from spine_segmentation import point_2_list


def _vec_2_point(vector: Vector_3) -> Point_3:
    return Point_3(vector.x(), vector.y(), vector.z())


def _vec_2_list(vector: Vector_3) -> list:
    return [vector.x(), vector.y(), vector.z()]


def _point_2_vec(point: Point_3) -> Vector_3:
    return Vector_3(point.x(), point.y(), point.z())


def _point_2_list(point: Point_3) -> list:
    return [point.x(), point.y(), point.z()]


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


def _calculate_junction_center(spine_mesh: Polyhedron_3) -> Vector_3:
    junction_triangles = _get_junction_triangles(spine_mesh)
    if len(junction_triangles) > 0:
        junction_center = Vector_3(0, 0, 0)
        for facet in junction_triangles:
            junction_center += _calculate_facet_center(facet)
        junction_center /= len(junction_triangles)
    else:
        junction_center = _point_2_vec(spine_mesh.points().next())
    return junction_center


def _get_junction_triangles(spine_mesh: Polyhedron_3) -> set:
    junction_triangles = set()
    for v in spine_mesh.vertices():
        if v.vertex_degree() > 10:
            # mark adjacent triangles
            for h in spine_mesh.halfedges():
                if h.vertex() == v:
                    junction_triangles.add(h.facet())
    return junction_triangles


def calculate_surface_center(mesh: Polyhedron_3) -> np.ndarray:
    vertices = np.ndarray((mesh.size_of_vertices(), 3))
    for i, vertex in enumerate(mesh.vertices()):
        vertex.set_id(i)
        vertices[i, :] = point_2_list(vertex.point())
    return np.mean(vertices, axis=0)


def calculate_metrics(spine_mesh: Polyhedron_3,
                      metric_names: List[str], params: List[Dict[str, Any]] = None) -> List[SpineMetric]:
    if params is None:
        params = [{}] * len(metric_names)
    return [create_metric_by_name(name, spine_mesh, **params[i]) for i, name in enumerate(metric_names)]


def create_metric_by_name(metric_name: str, *args, **kwargs):
    path = str(__package__).split('.')
    metric = __import__(__package__)
    for i in range(1, len(path)):
        metric = getattr(metric, path[i])
    klass = getattr(metric, metric_name + "SpineMetric")
    return klass(*args, **kwargs)


def polar2cart(theta: np.ndarray, phi: np.ndarray, radius: np.ndarray) -> np.ndarray:
    return np.stack([radius * np.sin(theta) * np.cos(phi),
                     radius * np.sin(theta) * np.sin(phi),
                     radius * np.cos(theta)], axis=-1)


def cart2polar(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    x, y, z = np.array(x), np.array(y), np.array(z)
    XsqPlusYsq = np.power(x, 2) + np.power(y, 2)
    r = np.sqrt(XsqPlusYsq + np.power(z, 2))       # r
    elev = np.arctan2(z, np.sqrt(XsqPlusYsq))   # theta
    az = np.arctan2(y, x)                       # phi
    return [r, elev, az]
