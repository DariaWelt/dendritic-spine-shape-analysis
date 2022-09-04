from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Polygon_mesh_processing import Polylines
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from spine_segmentation import point_2_list

MeshDataset = Dict[str, Polyhedron_3]
V_F = Tuple[np.ndarray, np.ndarray]
LineSet = List[Tuple[Point_3, Point_3]]


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


def preprocess_meshes(spine_meshes: MeshDataset) -> Dict[str, V_F]:
    output = {}
    for (spine_name, spine_mesh) in spine_meshes.items():
        output[spine_name] = _mesh_to_v_f(spine_mesh)
    return output


def polylines_to_line_set(polylines: Polylines) -> LineSet:
    output = []
    for line in polylines:
        for i in range(len(line) - 1):
            output.append((line[i], line[i + 1]))
    return output
