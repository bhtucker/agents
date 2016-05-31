# -*- coding: utf-8 -*-
"""
    abm.pops
    ~~~~~~~~

    Populations of interconnected agents
"""
from scipy.stats.distributions import norm
from scipy.stats.distributions import uniform
from sklearn.metrics.pairwise import euclidean_distances

from abm.viz import display_network
from abm.entities import Entity, Task
import numpy as np
from random import choice


Y_DIST = norm(300, 10)

CLUSTER_X_DIST_MAP = {
    'A': uniform(0, 50),
    'B': uniform(30, 50),
    'C': uniform(60, 50)
}

CLUSTER_SIZES = {
    'A': 8,
    'B': 10,
    'C': 8
}


def make_points(cluster, size, y_dist, x_dist):
    """Creates a set of points using y_dist and x_dist to draw the location."""

    ys = y_dist.rvs(size)
    xs = x_dist.rvs(size)
    return list(zip(xs, ys, [cluster] * size))


class Population(object):
    """
    A set of connected Entities. Handles message passing and displaying.
    Entities are connected randomly.
    """

    def __init__(self, y_pos_dist=Y_DIST, cluster_x_dists=CLUSTER_X_DIST_MAP,
                 cluster_sizes=CLUSTER_SIZES, track_components=False, single_component=True, debug=True):
        self.points = []
        self.path = []
        self.show = True
        self.debug = debug
        self.success_lens = []
        self.connectivity_matrix = None
        self.connected_components = []
        self.node_component_map = {}

        self._set_entities(y_pos_dist, cluster_x_dists, cluster_sizes)
        self._set_connectivity_matrix()
        self._set_connections(track_components=track_components or single_component)
        if single_component:
            self._ensure_single_component()

    def log(self, msg):
        """Prints a message to stdout, only if self.debug=True."""
        if self.debug:
            print(msg)

    def pass_message(self, recipient, task, sender):
        """
        Displays current state and routes the message one connection
        down. Handles the case where the recipient is the target. Calls
        Entity.receive_task(). <recipient> is the index of the recipient
        Entity.
        """

        self.display(recipient, task.target)
        self.path.append(recipient)
        self.points[recipient].receive_task(task, sender)

        if recipient == task.target:
            for point in self.path:
                amount = self.calculate_award(task, self.path, point)
                self.points[point].award(amount)

            self.success_lens.append(len(self.path))
            self.path = []

    def calculate_award(self, task, path, entity):
        """
        Returns the amount awarded to <entity> after <task> has been
        routed through <path>.
        """
        k = k = float(len(path))
        return (task.value / k) + ((k - path.index(entity)) * 5)

    def _set_entities(self, y_pos_dist, cluster_x_dists, cluster_sizes):
        point_args = []
        for cluster, size in cluster_sizes.iteritems():
            point_args += make_points(cluster, size,
                                      y_pos_dist, cluster_x_dists[cluster])

        for ix, args in enumerate(point_args):
            pt = Entity(self, ix, *args)
            self.points.append(pt)

    def _set_connections(self, track_components=True):
        """Initializes each Entity's adjacency list.
        :param track_components: Flag for tracking connected components during graph construction
        """

        self._set_connectivity_matrix()
        for index, point in enumerate(self.points):

            # make set of connections to indices; np.where returns a tuple
            adjacencies = set(np.where(self.connectivity_matrix[index] > 0)[0])
            adjacencies.discard(index)

            # pass adjacency information down to agent
            point.set_adjacencies(adjacencies)

            if track_components:
                # track connected components as we construct edges
                if index in self.node_component_map:
                    component = self.node_component_map[index]
                else:
                    component = set([index])
                    self.node_component_map[index] = component
                    self.connected_components.append(component)
                # update the component in place with potential new members
                component.update(adjacencies)

                # update the node - component map so we can fetch this object
                # for adjacencies
                self.node_component_map.update(
                    {a: component for a in adjacencies})

                # resolve potential component connections
                self._resolve_components(component)

        n = float(len(self.points))
        k = float(np.sum(self.connectivity_matrix)) / 2
        self.edge_density = k / (n * (n - 1) / 2)

    def _ensure_single_component(self):
        """
        Iterate through disjoint component list, adding connections between sequential components
        Update other datastructures to reflect the new connections
        """
        for ix, component in enumerate(self.connected_components[:-1]):
            start, end = (choice(list(component)), choice(
                list(self.connected_components[ix + 1])))
            self.points[start].adjacencies.append(end)
            self.points[end].adjacencies.append(start)
            self.connectivity_matrix[start][end] = True
            self.connectivity_matrix[end][start] = True
            self.connected_components[ix].add(end)
            self.connected_components[ix + 1].add(start)

        self._resolve_components(self.connected_components[0])

    def _resolve_components(self, component):
        """
        Find components thought to be separate that now have intersections
        Condense these and set self.connected_components to be a list of disjoint sets 
        """
        resolved_components = [component]
        for other_component in self.connected_components:
            if other_component.intersection(component) or other_component is component:
                component.update(other_component)
                self.node_component_map.update(
                    {a: component for a in other_component})
            else:
                resolved_components.append(other_component)
        self.connected_components = resolved_components

    def _set_connectivity_matrix(self):
        """
        Computes the connectivity matrix of this Population. Each point is
        connected to each other within a radius.
        """

        if self.connectivity_matrix is not None:
            return

        # generate a random symmetric matrix
        point_count = len(self.points)
        matrix = np.random.randint(
            0, 2, point_count ** 2).reshape(point_count, point_count)
        matrix = (matrix + matrix.T) / 2
        for i in range(point_count):
            matrix[i][i] = 0

        self.connectivity_matrix = matrix

    def display(self, current=None, target=None):
        """
        Plots the state of the task. If <show> = False, doesn't plot
        anything and the simulation can run faster.
        """
        if not self.show:
            return

        display_network(self.points, self.connectivity_matrix,
                        current=current, target=target)

    def initiate_task(self, fixed_pair=None):
        """Initializes a task, and calls pass_message()."""

        start, end = [-1, -1] if not fixed_pair else fixed_pair

        if fixed_pair and start == end:
            self.log("changing fixed pair: they're the same!")

        while start == end:
            [start, end] = np.random.randint(len(self.points), size=2).tolist()

        self.log('new task created')
        self.log('starting at %s and aiming for %s' % (start, end))

        task = Task(end)
        self.pass_message(start, task, None)

    def clear(self):
        """
        Clear every Entity's history. Called when the package is stuck
        in a cycle.
        """

        for point in self.points:
            adjs = point.adjacencies, set(point.adjacencies)
            if len(adjs[0]) != len(adjs[1]):
                self.log("clearing %s" % point.index + '---' * 20)
                self.log(str(point.adjacencies) + ", " +
                         str(set(point.adjacencies)))
            point.sent = []


class CappedPreferentialPopulation(Population):
    """
    A set of connected Entities. Handles message passing and displaying. Connections are laid
    out such that entities of the same cluster are more likely to be tied together,
    proportionally to a parameter alpha. The overall density of the network is controlled
    by a parameter beta.
    """

    def __init__(self, alpha=0.8, beta=0.4, *args, **kwargs):
        self.alpha = alpha
        self.beta = beta
        super(CappedPreferentialPopulation, self).__init__(*args, **kwargs)

    def _set_connectivity_matrix(self):
        """
        Computes the connectivity matrix of this Population. Each point is
        connected to each other within a radius.
        """

        if self.connectivity_matrix is not None:
            return

        def decide_connection(point1, point2):
            # A point is connected to another point of its same cluster
            # with high probability proportional to alpha, and to
            # another point of a different clluester with probability
            # proportional to 1 - alpha.
            # Moreover, the edge density of a network is capped at a value
            # beta. That's why we choose a 0 with probability 1-beta,
            # and partition beta into alpha and 1-alpha.

            alpha = self.alpha
            beta = self.beta

            if point1.cluster == point2.cluster:
                tie = np.random.choice(
                    [0, 0, 1], p=[1 - beta, beta * (1 - alpha), beta * alpha])
            else:
                tie = np.random.choice(
                    [0, 0, 1], p=[1 - beta, beta * alpha, beta * (1 - alpha)])
            return tie

        matrix = np.array([[0] * len(self.points)
                           for _ in range(len(self.points))])

        # since the graph is undirected, the matrix is symmetric,
        # which in turn means we need only compute the lower triangular
        # elements and then copy them into the upper triangular elements
        for i, point1 in enumerate(self.points):
            for j, point2 in enumerate(self.points[:i]):
                matrix[i][j] = decide_connection(point1, point2)
                matrix[j][i] = matrix[i][j]

        self.connectivity_matrix = matrix


class NearestNeighborsPopulation(Population):
    """
    A set of connected Entities. Handles message passing and displaying. Connections laid
    out geographically: each point is connected to some of its nearest neighbors.
    """

    def _set_connectivity_matrix(self):
        """
        Computes the connectivity matrix of this Population. Each point is
        connected to each other within a radius.
        """

        if self.connectivity_matrix is not None:
            return

        points_arr = np.array([[p.x, p.y] for p in self.points])
        distance_mat = euclidean_distances(points_arr, points_arr)

        # Every point p will be connected to each other point whose distance
        # to p is less than a cut-off value. This value is computed as the
        # mean of {min_nonzero(dist_mat(p)) | p is a point}, times a factor
        min_nonzero = lambda r: min(r[r > 0])  # noqa

        # apply_along_axis(f, axis=1, arr) applies f to each row
        min_neighbor_distances = np.apply_along_axis(
            min_nonzero, axis=1, arr=distance_mat)

        factor = 2.2
        neighbor_cutoff = np.mean(min_neighbor_distances) * factor
        connectivity_matrix = distance_mat < neighbor_cutoff

        self.connectivity_matrix = connectivity_matrix
