# -*- coding: utf-8 -*-
"""
    test_viz
    ~~~~~~~~
 
    test display
"""
import pytest
import matplotlib.pyplot as plt

from abm import viz, pops

@pytest.mark.mpl_image_compare(tolerance=20)
def test_display(nnpop):
    return viz.construct_network(nnpop.points, nnpop.connectivity_matrix)
