from typing import Tuple, List

import meshplot as mp
import numpy as np

from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Polygon_mesh_processing import Polylines
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from spine_analysis.mesh.utils import _mesh_to_v_f, polylines_to_line_set, LineSet, V_F
from spine_segmentation import apply_scale, point_2_list


def show_3d_mesh(mesh: Polyhedron_3, scale: Tuple[float, float, float] = (1, 1, 1)) -> None:
    shown_mesh = apply_scale(mesh, scale)
    vertices, facets = _mesh_to_v_f(shown_mesh)
    mp.plot(vertices, facets)


def show_line_set(lines: List[Tuple[Point_3, Point_3]], mesh) -> None:
    # make vertices and facets
    vertices = np.ndarray((len(lines) * 2, 3))
    facets = np.ndarray((len(lines), 3)).astype("uint")
    for i, line in enumerate(lines):
        vertices[2 * i, :] = point_2_list(line[0])
        vertices[2 * i + 1, :] = point_2_list(line[1])

        facets[i, 0] = 2 * i
        facets[i, 1] = 2 * i + 1
        facets[i, 2] = 2 * i

    # render
    plot = mp.plot(vertices, facets, shading={"wireframe": True})
    v, f = _mesh_to_v_f(mesh)
    # plot.add_lines(v[f[:, 0]], v[f[:, 1]], shading={"line_color": "gray"})
    plot.add_mesh(*_mesh_to_v_f(mesh), shading={"wireframe": True})


def _add_mesh_to_viewer_as_wireframe(viewer: mp.Viewer, mesh_v_f: V_F) -> None:
    (v, f) = mesh_v_f
    starts = []
    ends = []
    for facet in f:
        starts.append(v[facet[0]])
        starts.append(v[facet[1]])
        starts.append(v[facet[2]])
        ends.append(v[facet[1]])
        ends.append(v[facet[2]])
        ends.append(v[facet[0]])
    viewer.add_lines(np.array(starts), np.array(ends),
                     shading={"line_color": "gray"})


def show_polylines(polylines: Polylines, mesh: Polyhedron_3 = None) -> None:
    show_line_set(polylines_to_line_set(polylines), mesh)


def _add_line_set_to_viewer(viewer: mp.Viewer, lines: LineSet) -> None:
    viewer.add_lines(np.array([point_2_list(line[0]) for line in lines]),
                     np.array([point_2_list(line[1]) for line in lines]),
                     shading={"line_color": "red"})


def show_line_set(lines: LineSet, mesh: Polyhedron_3 = None) -> None:
    view = mp.Viewer({})
    _add_line_set_to_viewer(view, lines)
    if mesh:
        _add_mesh_to_viewer_as_wireframe(view, mesh)
    display(view._renderer)

