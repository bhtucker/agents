# -*- coding: utf-8 -*-
"""
    compare
    ~~~~

    Comparing performance of different cluster sizes in abm_sketch
"""

import abm_sketch as abm
import numpy as np
from scipy.stats.distributions import norm
from scipy.stats.distributions import uniform



if __name__ == "__main__":

    # initial parameters

    y_pos_dist = norm(300, 10)

    cluster_x_dists = {
        'A': uniform(0, 50),
        'B': uniform(30, 50),
        'C': uniform(60, 50)
    }
    
    cluster_sizes = {
        'A': 8,
        'B': 10,
        'C': 8
    }


    # compare how cluster sizes affects learning

    def run_cluster_size(sizes, iters=50):
        pop = abm.make_population(y_pos_dist, cluster_x_dists, sizes)
        pop.show = False
        pair = [0, 10]

        for i in range(iters):
            pop.initiate_task(fixed_pair=pair)

        return pop.success_lens


    iters = 50
    sizes = {'A': 10, 'B': 10,'C': 10}
    lens = run_cluster_size(sizes, iters)

    print("The clusters %r passed the same message %d times." % (sizes, iters))
    print("%d messages were successul." % len(lens))
    print("It took %.2f +- %.2f steps (avg +- std)" % (np.average(lens), np.std(lens)))
    print(lens)

