import math
from typing import List, Dict, Any

import numpy as np

from CGAL.CGAL_Kernel import Vector_3, Point_3
from CGAL.CGAL_Polygon_mesh_processing import area, face_area
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3_Facet_handle, Polyhedron_3, Polyhedron_3_Halfedge_handle
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


def polar2cart(az: np.ndarray, elev: np.ndarray, radius: np.ndarray) -> np.ndarray:
    return np.stack([radius * np.sin(elev) * np.cos(az),
                     radius * np.sin(elev) * np.sin(az),
                     radius * np.cos(elev)], axis=-1)


def cart2polar(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    x, y, z = np.array(x), np.array(y), np.array(z)
    XsqPlusYsq = np.power(x, 2) + np.power(y, 2)
    r = np.sqrt(XsqPlusYsq + np.power(z, 2))    # r
    elev = np.arctan2(np.sqrt(XsqPlusYsq), z)   # theta
    az = np.arctan2(y, x)                       # phi
    return [r, elev, az]


def point_in_circle(circle, point):
    return circle is not None and math.hypot(*np.subtract(point, circle[:-1])) - circle[-1] <= 1e-14

def get_enclosing_circle(points):
    if len(points) < 1:
        return None
    c = (*points[0], 0.0)
    for i, point_1 in enumerate(points[:]):
        if not point_in_circle(c, point_1):
            c = (*point_1, 0.0)
            for j, point_2 in enumerate(points[:i+1]):
                if not point_in_circle(c, point_2):
                    c = _make_circle(points[:j+1], point_1, point_2)
    return c


def _make_circle(points, p, q):

    def get_2points_circle(a, b):
        points = np.array([a, b])
        center = np.sum(points, axis=0) / 2
        return center[0], center[1], np.sqrt(max(np.sum((center - points) ** 2, axis=-1)))
    
    def _cross(x0, y0, x1, y1, x2, y2):
        return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    
    def get_circumscribed_circle(points: List):
        points = np.array(points)
        r = (np.min(points, axis=0) + np.max(points, axis=0)) / 2
        sides = points - r
        subtractions = np.array([np.roll(sides[:,1],-1) - np.roll(sides[:,1],-2), np.roll(sides[:,0],-2) - np.roll(sides[:,0],-1)])
        d = np.dot(sides[:, 0], subtractions[0])
        if d == 0:
            return None
        xy = np.add(np.matmul(np.sum(sides ** 2, axis=-1) / (d * 2), subtractions.T), r)
        return xy[0], xy[1], np.sqrt(max(np.sum((xy - points) ** 2, axis=-1)))

    circle = get_2points_circle(p, q)
    left, right = None, None

    for h in points:
        if point_in_circle(circle, h):
            continue

        side = _cross(*p, *q, *h)
        circle = get_circumscribed_circle([p, q, h])
        if circle is None or side == 0:
            continue

        center_orient = _cross(*p, *q, *circle[:-1])
        
        if side > 0 :
            if left is None or center_orient > _cross(*p, *q, *left[:-1]):
                left = circle
        else:
            if right is None or center_orient < _cross(*p, *q, *right[:-1]):
                right = circle

    res = None
    if left is not None and right is not None:
        res = left if left[-1] <= right[-1] else right
    elif left is None:
        res = right
    return circle if res is None else res

def get_incident_halfedges(facet_halfedge: Polyhedron_3_Halfedge_handle) -> List[Polyhedron_3_Halfedge_handle]:
    return [facet_halfedge, facet_halfedge.next(), facet_halfedge.next().next()]

def subdivide_mesh(mesh: Polyhedron_3,
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
                new_triangles.extend(get_incident_halfedges(halfedge))
                center = _vec_2_point(_calculate_facet_center(halfedge.facet()))
                new_v = out.create_center_vertex(halfedge).vertex()
                new_v.set_point(center)
            triangles = new_triangles

    return out
