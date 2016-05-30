# -*- coding: utf-8 -*-
"""
    abm.viz
    ~~~~~~~

    Displaying agent population
"""

from matplotlib.pylab import plt
from abm import CLUSTER_COLORS


def display_network(pts, connectivity_matrix, current=None, target=None):

    # Scatter plot of points, color coded by class
    size = 35
    for cluster, color in CLUSTER_COLORS.iteritems():
        class_points = [x for x in pts if x.cluster == cluster]
        plt.scatter([p.x for p in class_points], [p.y for p in class_points], c=color, s=size)

    # Draw the connections
    if connectivity_matrix is not None:
        for start_ix, connections in enumerate(connectivity_matrix):
            for connect_ix, connected in enumerate(connections):
                if connected and connect_ix != start_ix:
                    plt.plot(*zip(
                        (pts[start_ix].x, pts[start_ix].y),
                        (pts[connect_ix].x, pts[connect_ix].y)),
                        c='k', linewidth=0.5)

    # Show where the message is going and where it currently is
    if current and target:
        plt.scatter(pts[current].x, pts[current].y, c='m', s=150)
        plt.scatter(pts[target].x,  pts[target].y,  c='y', s=190)

    plt.show()
