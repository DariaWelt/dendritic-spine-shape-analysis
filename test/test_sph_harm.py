import os
import tempfile
from typing import Union, List, Any

import numpy as np
import meshplot as mp
import pytest

from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from spine_analysis.mesh.utils import get_basis_composition
from spine_analysis.shape_metric import SphericalGarmonicsSpineMetric as SPHMetric
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

    def assert_equal(self, test_result: Any, tol: float = 0.0):
        assert all(np.abs(np.array(test_result) - np.array(self._gt)) <= tol)


def get_basis_len(l_size: int):
    return sum([2 * _l + 1 for _l in range(l_size)])


coefficients_size = get_basis_len(SPHMetric.DEFAULT_L_SIZE)


@pytest.mark.parametrize('case', [
    SPHApproximationCase(name='Y_0^0_sphere',
                         figure=get_basis_composition([1] + [0] * (coefficients_size - 1))[0],
                         ground_truth=[1] + [0] * (coefficients_size - 1)),
    SPHApproximationCase(name='Y_1^1',
                         figure=get_basis_composition([0, 0, 0, 1] + [0] * (coefficients_size - 4))[0],
                         ground_truth=[0, 0, 0, 1] + [0] * (coefficients_size - 4)),
    SPHApproximationCase(name='Y_1^0',
                         figure=get_basis_composition([0, 0, 1] + [0] * (coefficients_size - 3))[0],
                         ground_truth=[0, 0, 1] + [0] * (coefficients_size - 3)),
    SPHApproximationCase(name='basis_composition',
                         figure=get_basis_composition([1, 0, 1, 0, 0, 1] + [0] * (coefficients_size - 6))[0],
                         ground_truth=[1, 0, 1, 0, 0, 1] + [0] * (coefficients_size - 6)),
], ids=str)
def test_basis_figures(case: SPHApproximationCase):
    metric = SPHMetric()
    basis_coefs = metric.calculate(case.mesh)
    case.assert_equal(basis_coefs, 0.01)


def test_basis_composition():
    a = [0, 1, 0, 0, 1]
    a.extend([0] * (coefficients_size - len(a)))
    mesh, v, f = get_basis_composition(a)
    #metric = SPHMetric()
    #basis_coefs = metric.calculate(mesh)
    #print(basis_coefs)
    #_, approx_v, approx_f = get_basis_composition(basis_coefs)
    #p = mp.plot(approx_v, approx_f)
    p = mp.plot(v, f)
    return p


def test_figures():
    pass
