from typing import List, Dict

import numpy as np

from CGAL.CGAL_Kernel import Vector_3, Point_3
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3_Facet_handle, Polyhedron_3
from spine_analysis.shape_metric.metric_core import SpineMetric


def _vec_2_point(vector: Vector_3) -> Point_3:
    return Point_3(vector.x(), vector.y(), vector.z())


def _point_2_vec(point: Point_3) -> Vector_3:
    return Vector_3(point.x(), point.y(), point.z())


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


def calculate_metrics(spine_mesh: Polyhedron_3,
                      metric_names: List[str], params: List[Dict] = None) -> List[SpineMetric]:
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
    return np.stack([radius * np.dot(np.sin(theta), np.sin(phi)),
                    radius * np.dot(np.sin(theta), np.cos(phi)),
                    radius * np.cos(theta)], axis=1)


def cart2polar(x: float, y: float, z: float) -> np.ndarray:
    XsqPlusYsq = np.dot(x,x) + np.dot(y, y)
    r = np.sqrt(XsqPlusYsq + np.dot(z,z))       # r
    elev = np.arctan2(z, np.sqrt(XsqPlusYsq))    # theta
    az = np.arctan2(y, x)                       # phi
    return np.array([r, elev, az])
