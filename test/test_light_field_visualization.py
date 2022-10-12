from typing import List

import cv2
import numpy as np
import pytest

from spine_analysis.shape_metric import LightFieldZernikeMomentsSpineMetric
from test.test_base import TestCaseBase


class ZernikeVisualizationCase(TestCaseBase):
    zernike_moments: List[float]

    def __init__(self, name: str, zernike_moments: List[float], ground_truth = None):
        super(ZernikeVisualizationCase, self).__init__(name, ground_truth)
        self.zernike_moments = zernike_moments


class ZernikeReconstructionCase(TestCaseBase):
    threshold_value: float
    fp_scale: float

    def __init__(self, name: str, image: np.ndarray, mask_threshold: float = 0.35, fp_scale: float = 0.05):
        super(ZernikeReconstructionCase, self).__init__(name, image)
        self.threshold_value = mask_threshold
        self.fp_scale = fp_scale

    @staticmethod
    def _draw_polygon(contour) -> np.ndarray:
        output = np.zeros((200, 200))
        cv2.fillPoly(output, pts=[contour], color=(1, 1, 1))
        return output

    @staticmethod
    def get_stick_image() -> np.ndarray:
        contour = np.array([[50, 50], [150, 150], [145, 140], [50, 40]])
        return ZernikeReconstructionCase._draw_polygon(contour)

    @staticmethod
    def get_cell_polygon() -> np.ndarray:
        contour = np.array([[40, 40], [35, 57], [30, 60], [40, 100], [140, 160], [146, 147], [160, 50], [170, 40],
                            [50, 30]])
        return ZernikeReconstructionCase._draw_polygon(contour)

@pytest.mark.parametrize('case', [
                             ZernikeVisualizationCase(name='circle', zernike_moments=[1] + [0] * 24),
                             ZernikeVisualizationCase(name='Z_1^1', zernike_moments=[0] + [1] + [0] * 23)
                         ], ids=str)
def test_exact_reconstruction(case: ZernikeVisualizationCase):
    light_field_metric = LightFieldZernikeMomentsSpineMetric()
    res = light_field_metric._recover_projection(case.zernike_moments, 200)
    res_moments = light_field_metric._calculate_moment(res, degree=8).real
    assert res_moments.tolist() == case.zernike_moments


@pytest.mark.parametrize('case', [
                             ZernikeReconstructionCase(name='stick', image=ZernikeReconstructionCase.get_stick_image()),
                             ZernikeReconstructionCase(name='polygon', image=ZernikeReconstructionCase.get_cell_polygon())
                         ], ids=str)
def test_approx_reconstruction(case: ZernikeReconstructionCase):
    light_field_metric = LightFieldZernikeMomentsSpineMetric()
    zernike_moments = light_field_metric._calculate_moment(case._gt, degree=20).tolist()
    res = light_field_metric._recover_projection(zernike_moments, 200)

    res = res / abs(res).max()
    res[res < case.threshold_value] = 0
    res[res > 0] = 1

    overlap = res * case._gt  # Logical AND
    union = res + case._gt  # Logical OR
    union[union > 0] = 1
    sym_dif = union - overlap

    fp = sym_dif.sum() / (res.shape[0] * res.shape[1])
    assert fp <= case.fp_scale
