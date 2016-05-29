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
import time



def log_simulation(desc, sizes, iters, pair, lens, start, end):
    print(desc)
    print("-" * 70)
    print("The clusters %r passed the same message %d times." % (sizes, iters))
    print("The message went from %d to %d" % tuple(pair))
    print("%d messages were successul." % len(lens))
    print("It took %.2f +- %.2f steps (avg +- std)" % (np.average(lens), np.std(lens)))
    print(lens)
    print("This simulation took %.4f seconds." % (end-start))
    print("-" * 70)
    print("")



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


    # Compare how cluster sizes affects learning.

    def run_cluster_size(sizes, pair, iters):
        pop = abm.make_population(y_pos_dist, cluster_x_dists, sizes)
        pop.show = False
        pop.debug = False

        for i in range(iters):
            pop.initiate_task(fixed_pair=pair)

        return pop.success_lens


    desc = ("Basic simulation with small clusters. The source and target\n"
            "of the message lie in the same cluster.")
    sizes = {'A': 10, 'B': 10,'C': 10}
    iters = 50
    pair = [0, 10]
    start = time.time()
    lens = run_cluster_size(sizes, pair, iters)
    end = time.time()

    log_simulation(desc, sizes, iters, pair, lens, start, end)


    desc = ("One giant cluster. The source and target of the message both lie\n"
            "in the giant cluster.")
    sizes = {'A': 100, 'B': 10,'C': 10}
    iters = 50
    pair = [0, 10]
    start = time.time()
    lens = run_cluster_size(sizes, pair, iters)
    end = time.time()

    log_simulation(desc, sizes, iters, pair, lens, start, end)


    desc = ("One giant cluster. The source lies within the big cluster, \n"
            "while the target lives in one of the small ones.")
    sizes = {'A': 100, 'B': 10,'C': 10}
    iters = 50
    pair = [0, 110]
    start = time.time()
    lens = run_cluster_size(sizes, pair, iters)
    end = time.time()

    log_simulation(desc, sizes, iters, pair, lens, start, end)
