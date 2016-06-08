# -*- coding: utf-8 -*-
"""
    test_pops
    ~~~~~~~~~

    tests for population code
"""
from abm import pops
import pytest
from scipy.stats.distributions import uniform
import matplotlib.pyplot as plt


@pytest.mark.unit
def test_make_points():
    cluster, size = 'A', 20
    pts = pops.make_points('A', 20, uniform(1, 10), uniform(1, 10))
    assert len(pts) == size
    assert pts[0][-1] == cluster
    assert all([x[0] > 1 for x in pts])


@pytest.mark.mpl_image_compare(tolerance=20)
def test_display(nnpop):
    nnpop.display()
