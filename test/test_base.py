from typing import Any


class TestCaseBase:
    _name: str
    _gt: Any

    def __init__(self, name: str, ground_truth: Any):
        self._name = name
        self._gt = ground_truth

    def __str__(self):
        return self._name

    def assert_equal(self, test_result: Any):
        assert test_result == self._gt
