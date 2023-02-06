import os
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import pytest

from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from spine_analysis.shape_metric import LightFieldZernikeMomentsSpineMetric
from test import DATA_PATH
from test.test_base import TestCaseBase


class ProjectionCase(TestCaseBase):
    mesh: Polyhedron_3
    view_point: List[float]

    def __init__(self, name: str, figure: Union[Polyhedron_3, os.PathLike], view_point: List[float], ground_truth):
        super(ProjectionCase, self).__init__(name, ground_truth)
        if not isinstance(figure, Polyhedron_3):
            self.mesh = Polyhedron_3(figure)
        else:
            self.mesh = figure
        self.view_point = view_point


@pytest.mark.parametrize('case', [
                             ProjectionCase(
                                 name='cube',
                                 figure=os.path.join(DATA_PATH, 'cube.off'),
                                 view_point=[0, 0, 2],
                                 ground_truth=np.ones((200, 200)))
                         ], ids=str)
def test_projections(case: ProjectionCase):
    light_field_metric = LightFieldZernikeMomentsSpineMetric()
    light_field_metric._view_points = np.array([case.view_point])
    res = light_field_metric.get_projections(case.mesh)
    plt.imshow(res[0])
    plt.show()
    case.assert_equal(res)
