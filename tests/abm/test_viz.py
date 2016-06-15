# -*- coding: utf-8 -*-
"""
    test_viz
    ~~~~~~~~

    test display
"""
import pytest

from abm import viz

@pytest.skipif
@pytest.mark.mpl_image_compare(tolerance=5)
def test_display(nnpop):
    return viz.construct_network(nnpop.points, nnpop.connectivity_matrix)


@pytest.skipif
@pytest.mark.mpl_image_compare(tolerance=5)
def test_display_pair(nnpop):
    return viz.construct_network(nnpop.points, nnpop.connectivity_matrix, 1, 4)
