from typing import List

import mahotas
import numpy as np
import pytest

from spine_analysis.shape_metric import LightFieldZernikeMomentsSpineMetric
from test.test_base import TestCaseBase


class ZernikeVisualizationCase(TestCaseBase):
    zernike_moments: List[float]

    def __init__(self, name: str, zernike_moments: List[float], ground_truth = None):
        super(ZernikeVisualizationCase, self).__init__(name, ground_truth)
        self.zernike_moments = zernike_moments


@pytest.mark.parametrize('case', [
                             ZernikeVisualizationCase(name='circle', zernike_moments=[1] + [0] * 24),
                             ZernikeVisualizationCase(name='Z_1^1', zernike_moments=[0] + [1] + [0] * 23)
                         ], ids=str)
def test_reconstruction(case: ZernikeVisualizationCase):
    light_field_metric = LightFieldZernikeMomentsSpineMetric()
    res = light_field_metric._recover_projection(case.zernike_moments, 200)
    res_moments = mahotas.features.zernike_moments(res, 2, degree=8).tolist()
    assert res_moments == case.zernike_moments
