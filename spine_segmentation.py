import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Set
import networkx as nx
import ast
from PIL import Image
from scipy.ndimage import binary_erosion
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Vector_3
from CGAL.CGAL_Point_set_3 import Point_set_3
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3, Polyhedron_3_Halfedge_handle, \
    Polyhedron_3_Vertex_handle, Polyhedron_3_Facet_handle, Polyhedron_3_Halfedge_around_facet_circulator
from CGAL.CGAL_Polygon_mesh_processing import Polylines, remove_connected_components, keep_connected_components
import json
from scipy.ndimage.filters import median_filter


Correspondence = Dict[str, Point_3]
ReverseCorrespondence = Dict[str, List[Point_3]]
Segmentation = Set[str]


def load_tif(filename: str) -> np.ndarray:
    # open image
    image = Image.open(filename)

    # extract every frame
    result = []
    try:
        i = 0
        while True:
            image.seek(i)
            i += 1
            frame = np.array(image)
            result.append(frame)
    except EOFError:
        pass

    # stack frames together into ndarray
    result = np.stack(result, -1)

    return result


def local_threshold_3d(image: np.ndarray, base_threshold: int = 127,
                       weight: float = 0.05, block_size: int = 3) -> np.ndarray:
    local_median = median_filter(image, size=block_size)
    threshold = base_threshold + weight * (base_threshold - local_median)
    output = np.zeros_like(image)
    output[image > threshold] = 255
    # for z in range(image.shape[2]):
    #     local_mean = threshold_local(image[:, :, z], block_size=block_size)
    #     threshold = base_threshold + weight * (base_threshold - local_mean)
    #     output[:, :, z] = image[:, :, z] > threshold

    return output


def is_valid_coord(image_shape, coord: Point_3) -> bool:
    return 0 <= coord.x() < image_shape[0] and \
        0 <= coord.y() < image_shape[1] and \
        0 <= coord.z() < image_shape[2]


def get_surface_points(image: np.ndarray) -> Point_set_3:
    # find surface points
    erosion = binary_erosion(image).astype(image.dtype) * 255
    edges = image - erosion
    surface_points: np.ndarray = np.argwhere(edges > 0)

    # calculate normals
    out: Point_set_3 = Point_set_3()
    out.add_normal_map()
    for i in range(surface_points.shape[0]):
        point = list_2_point(surface_points[i])
        normal = Vector_3(0, 0, 0)
        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    shift = Vector_3(x, y, z)
                    neighbour = point + shift
                    if is_valid_coord(image.shape, neighbour) and \
                            image[int(neighbour.x()), int(neighbour.y()), int(neighbour.z())] > 0:
                        normal -= shift
        magnitude = np.sqrt(normal.squared_length())
        if magnitude > 0:
            normal /= magnitude
        else:
            normal = Vector_3(0.0, 1.0, 0.0)
        out.insert(point, normal)

    return out


def point_2_list(point: Point_3) -> List[float]:
    return [point.x(), point.y(), point.z()]


def list_2_point(coords: List[float]) -> Point_3:
    return Point_3(float(coords[0]), float(coords[1]), float(coords[2]))


def hash_point(point: Point_3, decimals: int = 5) -> str:
    return str(np.around(np.asarray(point_2_list(point)),
                         decimals=decimals).tolist())


def unhash_point(hashed_point: str) -> Point_3:
    return list_2_point(ast.literal_eval(hashed_point))


def build_correspondence(corr_polylines: Polylines) -> Correspondence:
    corr: Correspondence = {}
    for line in corr_polylines:
        corr[hash_point(line[1])] = line[0]
    return corr


def build_reverse_correpondnce(corr_polylines: Polylines) -> ReverseCorrespondence:
    corr: ReverseCorrespondence = {}
    for line in corr_polylines:
        key = hash_point(line[0])
        if not key in corr:
            corr[key] = []
        corr[key].append(line[1])
    return corr


def calculate_path_angle(path: List):
    if len(path) == 0:
        return 0

    def get_edge(edge_index: int) -> Vector_3:
        return unhash_point(path[edge_index + 1]) - unhash_point(path[edge_index])

    angle = 0
    prev_edge: Vector_3 = get_edge(0)
    for i in range(1, len(path) - 1):
        cur_edge: Vector_3 = get_edge(i)
        angle += prev_edge * cur_edge / np.sqrt(prev_edge.squared_length()) / np.sqrt(cur_edge.squared_length())
        prev_edge = cur_edge

    return angle


def find_longest_path(graph: nx.Graph) -> List:
    # build simplified graph
    simplified = graph.copy()
    for node in graph.nodes():
        if simplified.degree[node] == 2:
            neighbours = np.asarray(simplified.neighbors(node))
            [u, v] = simplified.neighbors(node)
            simplified.add_edge(u, v)
            simplified.remove_node(node)
    # show_graph(graph)
    # show_graph(simplified)

    leaves = [x for x in simplified.nodes() if simplified.degree[x] == 1]
    longest_path = []
    for i in range(len(leaves) - 1):
        for j in range(i, len(leaves)):
            for path in nx.algorithms.all_simple_paths(simplified, leaves[i],
                                                       leaves[j]):
                if len(longest_path) <= len(path) and calculate_path_angle(
                        longest_path) < calculate_path_angle(path):
                    longest_path = path

    # un-simplify path
    output = [longest_path[0]]
    for i in range(1, len(longest_path)):
        for path in nx.algorithms.all_simple_paths(graph, longest_path[i - 1],
                                                   longest_path[i]):
            output += path[1:]
            break

    return output


def show_graph(graph):
    hash_to_pos = {}
    for hashed_point in graph.nodes():
        point = unhash_point(hashed_point)
        hash_to_pos[hashed_point] = (point.x(), point.y())

    nx.draw(graph, pos=hash_to_pos)
    # nx.draw_networkx_labels(graph, pos=hash_to_pos)
    plt.show()


def build_graph(polylines: Polylines) -> nx.Graph:
    output = nx.Graph()

    for line in polylines:
        for i in range(len(line) - 1):
            u: Point_3 = line[i]
            v: Point_3 = line[i + 1]

            hashed_u = hash_point(u)
            hashed_v = hash_point(v)

            if i == 0:
                output.add_node(hashed_u)
            output.add_node(hashed_v)

            output.add_edge(hashed_u, hashed_v, weight=get_distance(u, v))

    return output


def get_distance(x: Point_3, y: Point_3) -> float:
    return np.sqrt((x - y).squared_length())


def get_distance_statistic(longest_path: List[str], reverse_correspondence: ReverseCorrespondence,
                           center_index: int, window_halfsize: int = 5) -> List[float]:
    output = []
    start_index = max(center_index - window_halfsize, 0)
    end_index = min(center_index + 1 + window_halfsize, len(longest_path))
    for i in range(start_index, end_index):
        hashed_skeleton_point: str = longest_path[i]
        # some skeleton points have no surface points corresponding to them smh
        if not hashed_skeleton_point in reverse_correspondence:
            continue
        for surface_point in reverse_correspondence[hashed_skeleton_point]:
            output.append(get_distance(surface_point, unhash_point(hashed_skeleton_point)))
    return output


def get_path_statistics(path: List[str], reverse_correspondence: ReverseCorrespondence,
                        window_halfsize: int = 0) -> Dict:
    path_statistics = {}
    for i, hashed_skeleton_point in enumerate(path):
        path_statistics[hashed_skeleton_point] = \
            get_distance_statistic(path, reverse_correspondence, i,
                                   window_halfsize)
    return path_statistics


def segmentation_by_distance(polyhedron: Polyhedron_3,
                             correspondence: Correspondence,
                             reverse_correspondence: ReverseCorrespondence,
                             skeleton_graph: nx.Graph,
                             distance_sensitivity: float = 0.75) -> Segmentation:
    # find dendrite subgraph
    dendrite_skeleton_points: List = find_longest_path(skeleton_graph)

    # calculate distance from skeleton threshold
    path_statistics = get_path_statistics(dendrite_skeleton_points,
                                          reverse_correspondence, 0)
    all_distance = np.concatenate([x for x in path_statistics.values()])
    distance_threshold = np.quantile(all_distance, distance_sensitivity)

    output_segmentation: Segmentation = set()
    # make set for performance
    dendrite_set: Set = set(dendrite_skeleton_points)
    for surface_point in polyhedron.points():
        hashed_surface_point = hash_point(surface_point)
        skeleton_point = correspondence[hash_point(surface_point)]
        hashed_skeleton_point = hash_point(skeleton_point)
        if hashed_skeleton_point in dendrite_set:
            # skeleton point belongs to dendrite subgraph
            distance = get_distance(surface_point, skeleton_point)
            if distance > distance_threshold:
                output_segmentation.add(hashed_surface_point)
        else:
            # skeleton point belongs to spine subgraph
            output_segmentation.add(hashed_surface_point)

    return output_segmentation


def _erase_dendrite_facets(in_mesh: Polyhedron_3,
                           segmentation: Segmentation) -> Polyhedron_3:
    # copy mesh
    mesh: Polyhedron_3 = in_mesh.deepcopy()

    # erase dendrite facets
    for facet in mesh.facets():
        circulator: Polyhedron_3_Halfedge_around_facet_circulator = facet.facet_begin()
        begin = facet.facet_begin()
        while circulator.hasNext():
            halfedge: Polyhedron_3_Halfedge_handle = circulator.next()
            v: Polyhedron_3_Vertex_handle = halfedge.vertex()

            if not hash_point(v.point()) in segmentation:
                mesh.erase_facet(facet.halfedge())
                break
            # check for end of loop
            if circulator == begin:
                break
    return mesh
    

def get_spine_meshes(in_mesh: Polyhedron_3, segmentation: Segmentation) -> List[Polyhedron_3]:
    mesh: Polyhedron_3 = _erase_dendrite_facets(in_mesh, segmentation)

    # set halfedge ids
    for i, halfedge in enumerate(mesh.halfedges()):
        halfedge.set_id(i)

    # determine halfedges corresponding to connected components
    component_halfedge_ids: List[int] = []
    reduced_mesh: Polyhedron_3 = mesh.deepcopy()
    while reduced_mesh.size_of_facets() > 0:
        facet: Polyhedron_3_Facet_handle = reduced_mesh.facets().next()
        component_halfedge_ids.append(facet.halfedge().id())
        remove_connected_components(reduced_mesh, [facet])

    # extract connected components one by one
    output = []
    for halfedge_id in component_halfedge_ids:
        spine_mesh: Polyhedron_3 = mesh.deepcopy()
        # find facet with correct halfedge id
        component_facet: Polyhedron_3_Facet_handle = spine_mesh.facets().next()
        for halfedge in spine_mesh.halfedges():
            if halfedge.id() == halfedge_id:
                component_facet = halfedge.facet()
                break
        keep_connected_components(spine_mesh, [component_facet])
        # filter small meshes
        if spine_mesh.size_of_facets() <= 10:
            continue
        # fill holes
        for h in spine_mesh.halfedges():
            a = Polyhedron_3_Halfedge_handle()
            if h.is_border():
                spine_mesh.fill_hole(h)
        # triangulate non-triangle facets via center vertex
        for facet in spine_mesh.facets():
            if not facet.is_triangle():
                spine_mesh.create_center_vertex(facet.halfedge())
        output.append(spine_mesh)

    print(f"Detected {len(output)} spines")

    return output


# def save_segmentation(segmentation: Segmentation, filename: str) -> None:
#     # from https://stackoverflow.com/a/67572570
#     class CustomJSONizer(json.JSONEncoder):
#         def default(self, obj):
#             return super().encode(bool(obj)) \
#                 if isinstance(obj, np.bool_) \
#                 else super().default(obj)
#
#     with open(filename, "w") as file:
#         json.dump(segmentation, file, cls=CustomJSONizer)
#
#
# def load_segmentation(filename: str) -> Segmentation:
#     with open(filename, 'r') as file:
#         return json.load(file)


def get_final_segmentation(spines: List[Polyhedron_3]) -> Segmentation:
    result: Segmentation = set()
    for spine in spines:
        for point in spine.points():
            result.add(hash_point(point))
    return result


