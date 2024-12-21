import pytest
import ohbemn


def test_sum_as_string():
    assert ohbemn.sum_as_string(1, 1) == "2"
