from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import meshplot as mp

from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from spine_segmentation import point_2_list


V_F = Tuple[np.ndarray, np.ndarray]


def _mesh_to_v_f(mesh: Polyhedron_3) -> V_F:
    vertices = np.ndarray((mesh.size_of_vertices(), 3))
    for i, vertex in enumerate(mesh.vertices()):
        vertex.set_id(i)
        vertices[i, :] = point_2_list(vertex.point())

    facets = np.ndarray((mesh.size_of_facets(), 3)).astype("uint")
    for i, facet in enumerate(mesh.facets()):
        circulator = facet.facet_begin()
        j = 0
        begin = facet.facet_begin()
        while circulator.hasNext():
            halfedge = circulator.next()
            v = halfedge.vertex()
            facets[i, j] = (v.id())
            j += 1
            # check for end of loop
            if circulator == begin or j == 3:
                break
    return vertices, facets


def load_spine_meshes(folder_path: str = "output",
                      spine_file_pattern: str = "**/spine_*.off") -> Dict[str, Polyhedron_3]:
    output = {}
    path = Path(folder_path)
    spine_names = list(path.glob(spine_file_pattern))
    for spine_name in spine_names:
        output[str(spine_name)] = Polyhedron_3(str(spine_name))
    return output


def preprocess_meshes(spine_meshes: Dict[str, Polyhedron_3]) -> Dict[str, V_F]:
    output = {}
    for (spine_name, spine_mesh) in spine_meshes.items():
        output[spine_name] = _mesh_to_v_f(spine_mesh)
    return output


def show_3d_mesh(mesh: Polyhedron_3) -> None:
    vertices, facets = _mesh_to_v_f(mesh)
    mp.plot(vertices, facets)
    # mp.plot(vertices, facets, shading={"wireframe": True})


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

