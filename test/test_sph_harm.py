import os
import tempfile
from typing import Union, List

import numpy as np
import meshplot as mp
import pytest
from IPython.core.display import display

from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from spine_analysis.shape_metric import SphericalGarmonicsSpineMetric as SPHMetric
from spine_analysis.shape_metric.utils import polar2cart
from test import DATA_PATH
from test.test_base import TestCaseBase


class SPHApproximationCase(TestCaseBase):
    mesh: Polyhedron_3

    def __init__(self, name: str, figure: Union[Polyhedron_3, os.PathLike], ground_truth=None):
        super(SPHApproximationCase, self).__init__(name, ground_truth)
        if not isinstance(figure, Polyhedron_3):
            self.mesh = Polyhedron_3(figure)
        else:
            self.mesh = figure


def get_basis_len(l_size: int):
    return sum([2 * _l + 1 for _l in range(l_size)])


coefficients_size = get_basis_len(SPHMetric.DEFAULT_L_SIZE)


def get_basis_composition(basis_coefs: List[float], l_size: int = SPHMetric.DEFAULT_L_SIZE) -> Polyhedron_3:
    assert len(basis_coefs) == get_basis_len(l_size)
    metric = SPHMetric(l_range=range(l_size))

    def composition_callback(phi, hi):
        return sum(a_i * metric._get_basis(*metric.m_l_map[i])(phi, hi) for i, a_i in enumerate(basis_coefs)).real

    phr = np.array([[[p, h, composition_callback(p, h)] for h in range(-90, 92, 2)] for p in range(0, 362, 2)])

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, 'tmp.off')
        with open(file_path, 'w') as fd:
            v, f = write_off(fd, phr)
        p = Polyhedron_3(file_path)
    return p, v, f


def write_off(fd, phr):
    n, m = len(phr), len(phr[0])
    top = n * (m - 1)

    vertices = polar2cart(phr[..., 1], phr[..., 0], phr[..., 2])
    facets = [[i, i + 1, i + n, i + 1, i + n + 1, i + n]
              for i in range(n*m - 1 - n)]

    # vertical connection
    facets.extend([[(i + 1) * n - 1, i * n, (i + 2) * n - 1, i * n, (i + 1) * n,  (i + 2) * n - 1]
                   for i in range(m - 1)])
    # horizontal connection
    facets.extend([[top + i, top + i + 1, i, top + i + 1, i + 1, i]
                   for i in range(n - 1)])
    facets.append([n * m - 1, top, n - 1, top, 0, n - 1])

    fd.write('OFF\n')
    fd.write(f'{len(phr)*len(phr[0])} {len(facets)} 0\n')
    for v_row in vertices:
        for v in v_row:
            fd.write(f'{v[0]} {v[1]} {v[2]}\n')
    for f in facets:
        fd.write(f'3 {f[0]} {f[1]} {f[2]}\n')
        fd.write(f'3 {f[3]} {f[4]} {f[5]}\n')
    return vertices, facets


@pytest.mark.parametrize('case', [
    SPHApproximationCase(name='Y_0^0_sphere',
                         figure=os.path.join(DATA_PATH, 'identity_sphere.off'),
                         ground_truth=[1] + [0] * (coefficients_size - 1)),
    SPHApproximationCase(name='Y_1^1',
                         figure=os.path.join(DATA_PATH, 'Y_11.off'),
                         ground_truth=[0, 0, 0, 1] + [0] * (coefficients_size - 4)),
    SPHApproximationCase(name='Y_1^0',
                         figure=os.path.join(DATA_PATH, 'Y_10.off'),
                         ground_truth=[0, 0, 1] + [0] * (coefficients_size - 3)),
    SPHApproximationCase(name='basis_composition',
                         figure=os.path.join(DATA_PATH, 'Y_composition.off'),
                         ground_truth=[1, 0, 1, 0, 0, 1] + [0] * (coefficients_size - 6)),
], ids=str)
def test_basis_figures(case: SPHApproximationCase):
    metric = SPHMetric()
    basis_coefs = metric.calculate(case.mesh)
    case.assert_equal(basis_coefs)


def test_basis_composition():
    a = [1] + [0] * (coefficients_size - 1)
    mesh, v, f = get_basis_composition(a)
    metric = SPHMetric()
    basis_coefs = metric.calculate(mesh)
    print(basis_coefs)
    mp.offline()
    display(mp.plot(v.reshape(len(v)*len(v[0]), 3), np.array(f).reshape(len(f)*2, 3)))


def test_figures():
    pass
