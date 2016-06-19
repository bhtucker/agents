# -*- coding: utf-8 -*-
"""
    test_pops
    ~~~~~~~~~

    tests for population code
"""
from abm import pops
from abm.entities import Task
import pytest
from scipy.stats.distributions import uniform


@pytest.fixture
def basicenv():
    return pops.Environment()


@pytest.mark.unit
def test_distribute_awards(basicenv):
    class MockEntity(object):
        total_award = 0

        def award(self, amount):
            self.total_award += amount

    basicenv.population = []
    for i in range(6):
        basicenv.population.append(MockEntity())
    basicenv.path = [1, 2, 3, 2, 4]
    task = Task(1, 4)
    basicenv._distribute_awards(task)

    observed = [
        x.total_award for x in basicenv.population
    ]
    expected = ([0] + [20.] * 4 + [0])

    assert observed == expected
    assert sum(observed) == 80

    basicenv.path = [0]
    basicenv._distribute_awards(task)
    assert basicenv.population[0].total_award == 100

    basicenv.path = [5] * (basicenv.path_cutoff + 1)
    basicenv._distribute_awards(task)
    assert basicenv.population[5].total_award == 0


# @pytest.mark.unit
# def test_distribute_awards(basicenv):

# @pytest.mark.unit
# def test_make_points():
#     cluster, size = 'A', 20
#     pts = pops.make_points('A', 20, uniform(1, 10), uniform(1, 10))
#     assert len(pts) == size
#     assert pts[0][-1] == cluster
#     assert all([x[0] > 1 for x in pts])


# @pytest.mark.mpl_image_compare(tolerance=20)
# def test_display(nnpop):
#     nnpop.display()
