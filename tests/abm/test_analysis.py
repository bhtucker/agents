# -*- coding: utf-8 -*-
"""
    test_analysis
    ~~~~~~~~~~~~~

    Tests for environment learning assessment
"""

import pandas as pd
import numpy as np
import networkx as nx

from abm import analysis, nxpops

import pytest


@pytest.fixture
def nxpop(simple_pop_kwargs):
    """
    Provides a 6-node non-learned SoftmaxNxEnvironment
    with two paths from 0 to 5:
    0 - 1 - 2 - 3 - 4 - 5
     \ - - - - - - /
    """
    pop = nxpops.SoftmaxNxEnvironment(**simple_pop_kwargs)

    for edge in pop.graph.edges():
        pop.graph.remove_edge(*edge)

    pop.graph.add_edge(0, 1)
    pop.graph.add_edge(1, 2)
    pop.graph.add_edge(2, 3)
    pop.graph.add_edge(3, 4)
    pop.graph.add_edge(0, 4)
    pop.graph.add_edge(4, 5)
    return pop


@pytest.fixture
def learnt_nxpop(nxpop):
    """
    Provides nxpop fixtures with sequential traversal learned
    """
    task = nxpop._generate_task(5)
    for i in range(5):
        nxpop.population[i]._get_next_recipient(task)
        nxpop.population[i].last_recipient = i + 1
        nxpop.population[i].award(1000)
    nxpop.population[0].w_container[4] -= np.ones(task.features.shape)
    nxpop.population[0].w_container[1] += np.ones(task.features.shape)
    nxpop.flush_updates()
    return nxpop


@pytest.mark.unit
def test_group_sample_by_time(analysis_df_data_dict):

    df = pd.DataFrame(analysis_df_data_dict)
    groups = analysis._group_sample_by_time(df)
    assert isinstance(groups, np.ndarray)
    assert np.isclose(np.mean(groups[1]), 0.005, rtol=.02)
    assert len(groups) == 2

    groups = analysis._group_sample_by_time(df, key='learnt_over_best')
    assert np.isclose(np.mean(groups[1]), 16.16, rtol=.02)
    assert len(groups) == 2


@pytest.mark.unit
def test_fixture_learning(learnt_nxpop):
    # verify that the analysis method doesn't affect brains
    for i in range(10):
        ratio = analysis.learnt_over_shortest_path_len(learnt_nxpop, 0, 5)
        assert ratio == 2.5
    for i in range(10):
        ratio = analysis.learnt_over_shortest_path_len(learnt_nxpop, 1, 5)
        assert np.isclose(ratio, 1.333, rtol=.02)

