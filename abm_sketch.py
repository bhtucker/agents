# -*- coding: utf-8 -*-
"""
    abm_sketch
    ~~~~

    Initial implementation of message passing in an agent graph
"""

from scipy.stats.distributions import norm
from scipy.stats.distributions import uniform
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib.pylab import plt
import numpy as np
from random import choice
from uuid import uuid4
from collections import defaultdict



cluster_colors = {
    'A': 'r',
    'B': 'b',
    'C': 'g'
}

MESSAGES = [
    "message uno",
    "message dos",
]



class Population(object):
    """A set of connected Entities. Handles message passing and displaying."""

    def __init__(self):
        self.points = []
        self.path = []
        self.show = True
        self.success_lens = []
        self.connectivity_matrix = None


    def pass_message(self, recipient, task, sender):
        """
        Displays current state and routes the message one connection
        down. Handles the case where the recipient is the target. Calls
        Entity.receive_task(). <recipient> is the index of the recipient
        Entity.
        """

        self.display(recipient, task.target)

        self.path.append(recipient)
        if recipient == task.target:
            # import ipdb ; ipdb.set_trace#z()  # breakpoint 223cdcc4 //

            for point in self.path:
                amount = self.calculate_award(task, self.path, point)
                self.points[point].award(amount)

            self.success_lens.append(len(self.path))
            self.path = []

        self.points[recipient].receive_task(task, sender)


    def calculate_award(self, task, path, entity):
        """
        Returns the amount awarded to <entity> after <task> has been
        routed through <path>.
        """
        k = k = float(len(path))
        return (task.value / k) + ((k - path.index(entity)) * 5)


    def set_connections(self):
        """Initializes each Entity's adjacency list."""

        self.connectivity_matrix = get_connectivity_matrix(self.points)
        for index, point in enumerate(self.points):
            point.set_adjacencies(self.connectivity_matrix[index])


    def display(self, current=None, target=None):
        """
        Plots the state of the task. If <show> = False, doesn't plot
        anything and the simulation can run faster.
        """

        if not self.show:
            return

        # Scatter plot of points, color coded by class
        pts = self.points
        size = 35
        for cluster, color in cluster_colors.iteritems():
            class_points = [x for x in pts if x.cluster == cluster]
            plt.scatter([p.x for p in class_points], [p.y for p in class_points], c=color, s=size)

        # Draw the connections
        if self.connectivity_matrix is not None:
            for start_ix, connections in enumerate(self.connectivity_matrix):
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


    def initiate_task(self, fixed_pair=None):
        """Initializes a task, and calls pass_message()."""

        start, end = [-1, -1] if not fixed_pair else fixed_pair

        if fixed_pair and start == end:
            print("changing fixed pair: they're the same!")

        while start == end:
            [start, end] = np.random.randint(len(self.points), size=2).tolist()

        message = choice(MESSAGES)
        print('new task: %s' % message)
        print('starting at %s and aiming for %s' % (start, end))

        task = Task(end, message)
        self.pass_message(start, task, len(self.points))


    def clear(self):
        """
        Clear every Entity's history. Called when the package is stuck
        in a cycle.
        """

        for point in self.points:
            adjs = point.adjacencies, set(point.adjacencies)
            if len(adjs[0]) != len(adjs[1]):
                print("clearing %s" % point.index + '---' * 20)
                print(point.adjacencies, set(point.adjacencies))
            point.sent = []



class Task(object):
    """Represents a message to be routed."""

    def __init__(self, target, message):
        self.target = target
        self.message = message
        self.id = uuid4()
        self.value = 100



class Entity(object):
    """An entity in our world"""

    def __init__(self, population, index, x, y, cluster):
        self.x = x
        self.y = y
        self.index = index
        self.cluster = cluster
        self.adjacencies = []
        self.population = population
        self.task_attempt_map = defaultdict(lambda: [index])
        self.sent = []
        self.value = 0


    def receive_task(self, task, sender):
        if sender not in self.task_attempt_map[task.id]:
            self.task_attempt_map[task.id].append(sender)
        print(sender, self.sent)
        if sender in self.sent:
            print('popping %s' % self.sent.pop(self.sent.index(sender)))
        if set(self.adjacencies).issubset(set(self.task_attempt_map[task.id])):
            print 'caught in a cycle! bailing!'
            self.population.clear()
            return
        if self.index == task.target:
            print 'message delivered! %s' % task.message
        else:
            self.pass_message(task, sender)


    def pass_message(self, task, sender):
        """
        If this Entity knows the target, send the message directly.
        Otherwise, choose randomly among the (extended) adjacency list.
        Calls Population.pass_message() with the chosen target.
        """

        next_recipient = task.target if task.target in self.adjacencies else self.index

        while next_recipient in self.task_attempt_map[task.id]:
            next_recipient = choice(self.adjacencies)

        self.task_attempt_map[task.id].append(next_recipient)
        print "passing message from %s to %s" % (self.index, next_recipient)

        self.sent.append(next_recipient)
        self.population.pass_message(next_recipient, task, self.index)


    def set_adjacencies(self, connectivity_vector):
        """
        Builds the adjacency list for this Entity. <connectivity_vector>
        is a row from an adjacency matrix.
        """

        for connect_ix, connected in enumerate(connectivity_vector):
            if connected and connect_ix != self.index:
                self.adjacencies.append(connect_ix)


    def award(self, value):
        """
        Accepts <value> as award after having participated in the
        successful routing of a message.
        """

        self.value += value
        print(self.value, self.index, self.adjacencies)

        for adj in self.sent:
            assert adj in self.adjacencies

            u_val = uniform.rvs(0, 100, 1)
            print(u_val, u_val < value)
            if u_val < value:
                self.adjacencies.append(adj)
                # import ipdb ; ipdb.set_trace() #z()  # breakpoint f35d9e23 //

        # prepare for the next message
        self.value = 0
        self.sent = []



def make_points(cluster, size, y_dist, x_dist):
    """Creates a set of points using y_dist and x_dist to draw the location."""

    ys = y_dist.rvs(size)
    xs = x_dist.rvs(size)
    return list(zip(xs, ys, [cluster] * size))


def make_population(y_pos_dist, cluster_x_dists, cluster_sizes):
    """Creates a Population and sets its connections. Uses make_points."""

    points = []
    for cluster, size in cluster_sizes.iteritems():
        points += make_points(cluster, size, y_pos_dist, cluster_x_dists[cluster])

    population = Population()
    for ix, point in enumerate(points):
        pt = Entity(population, ix, *point)
        population.points.append(pt)

    population.set_connections()
    return population


def get_connectivity_matrix(points):
    """
    Computes the connectivity matrix of <points>. Each point is
    connected to each other within a radius.
    """

    points_arr = np.array([[p.x, p.y] for p in points])
    distance_mat = euclidean_distances(points_arr, points_arr)

    min_nonzero = lambda r: min(r[r > 0])
    # axis=1 is the horizontal axis (row-wise)
    min_neighbor_distances = np.apply_along_axis(min_nonzero, axis=1, arr=distance_mat)

    # WHY 2.2 ?
    neighbor_cutoff = np.mean(min_neighbor_distances) * 2.2
    connectivity_matrix = distance_mat < neighbor_cutoff

    return connectivity_matrix


def run(y_pos_dist, cluster_x_dists, cluster_sizes):
    """Creates and sets up a Population and runs initiate_task()."""

    pop = make_population(y_pos_dist, cluster_x_dists, cluster_sizes)
    pop.initiate_task()



if __name__ == '__main__':

    y_pos_dist = norm(500, 10)

    cluster_x_dists = {
        'A': uniform(0, 50),
        'B': uniform(30, 50),
        'C': uniform(60, 50)
    }

    cluster_sizes = {
        'A': 10,
        'B': 15,
        'C': 12
    }

    run(y_pos_dist, cluster_x_dists, cluster_sizes)
