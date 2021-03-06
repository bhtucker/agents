# -*- coding: utf-8 -*-
"""
    test_nxpops
    ~~~~~~~~~

    tests for networkx population code
"""
import pytest

from abm import nxpops


@pytest.mark.unit
def test_feature_vec_options(simple_pop_kwargs):
    smpop = nxpops.SoftmaxNxEnvironment(**simple_pop_kwargs)
    task1 = smpop._generate_task(2)
    assert task1.features.sum() == 1
    smpop = nxpops.SoftmaxNxEnvironment(bias=True, **simple_pop_kwargs)
    task2 = smpop._generate_task(2)
    assert task2.features.sum() == 2
    smpop = nxpops.SoftmaxNxEnvironment(bias=True, node_index_indicator=True, **simple_pop_kwargs)
    task3 = smpop._generate_task(2)
    assert task3.features.sum() == 3
    assert len(task2.features) == (len(task3.features) - simple_pop_kwargs['size'])


@pytest.mark.integration
def test_performance_gains(fine_grained_pop_kwargs):
    smpop = nxpops.SoftmaxNxEnvironment(**fine_grained_pop_kwargs)
    smpop.debug = False
    smpop.show = False

    def get_sample_lens():
        sample = []
        for i in range(100):
            smpop.initiate_task()
            sample.append(len(smpop.path))
        return sample

    early_lens = get_sample_lens()
    for i in range(5000):
        smpop.initiate_task()
    late_lens = get_sample_lens()

    try:
        assert sum(early_lens) > (sum(late_lens) * 1.2)
    except AssertionError:
        late_lens = get_sample_lens()
        assert sum(early_lens) > (sum(late_lens) * 1.2)
