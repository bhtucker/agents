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



y_pos_dist = norm(300, 10)

cluster_x_dists = {
    'A': uniform(0, 40),
    'B': uniform(20, 60),
    'C': uniform(40, 80)
}

cluster_sizes = {
    'A': 8,
    'B': 10,
    'C': 8
}

cluster_colors = {
    'A': 'r',
    'B': 'b',
    'C': 'g'
}



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
        self.points[recipient].receive_task(task, sender)

        if recipient == task.target:
            # import ipdb ; ipdb.set_trace#z()  # breakpoint 223cdcc4 //

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

        print('new task created')
        print('starting at %s and aiming for %s' % (start, end))

        task = Task(end)
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

    def __init__(self, target):
        self.target = target
        self.id = uuid4()
        self.value = 100



class Entity(object):
    """An entity in our world"""

    def __init__(self, population, index, x, y, cluster):
        self.x = x
        self.y = y
        self.index = index
        self.cluster = cluster
        self.population = population

        self.sent = []    # list of points to whom I've passed the current message
        self.value = 0    # value awarded for a successful message

        # extended adjacency list (see pass_message() and learn())
        self.adjacencies = []

        # task_attempt_map[task_id] hold a list of the neighbors to whom
        # this entity gave the task
        self.task_attempt_map = defaultdict(lambda: [index])


    def receive_task(self, task, sender):
        """Handles a message received. Calls pass_message()."""

        print("%d: past receipients are " % (self.index,) + str(self.sent))

        if sender not in self.task_attempt_map[task.id]:
            self.task_attempt_map[task.id].append(sender)

        if sender in self.sent:
            print('popping %s' % self.sent.pop(self.sent.index(sender)))

        # If I have passed the message to all of my neighbors and I still
        # received it back, bail out.
        if set(self.adjacencies).issubset(set(self.task_attempt_map[task.id])):
            # FIXME: this catches cycles eventually, but also catches randomly
            # followed closed paths
            print 'caught in a cycle! bailing!'
            self.population.clear()
            return

        if self.index == task.target:
            print 'message delivered!'
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
        print "%s -> %s" % (self.index, next_recipient)

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
        successful routing of a message. Calls self.learn() and then
        cleans up to prepare for the next message.
        """

        self.value += value
        print("")
        print("awarding %.2f to %d" % (self.value, self.index))
        print("Map: " + str(dict(self.task_attempt_map)))
        print("Adj before: " + str(self.adjacencies))

        self.learn()

        self.value = 0
        self.sent = []


    def learn(self):
        """Learns about the world after receiving an award."""

        # After an award, we choose some adjacencies and increase the
        # likelihood of their being chosen for a message next time
        # around.

        # The probability of choosing an adjacency to increase its
        # likelihood is proportional to the value of the award received.

        # Since at every step we choose at random from our list of
        # adjacencies, to increase the likelihood of one of them being
        # chosen, we need only add them again to this list, i.e., if my
        # adjacencies are [1,2,3] and sending a message to 2 was
        # successful, our new extended adjacency list is [1,2,3,2], and
        # when we choose uniformly from this list, we will be more
        # likely to choose 2 again.

        # print("learning on %d" % self.index)

        for adj in self.sent:
            assert adj in self.adjacencies

            u_val = uniform.rvs(0, 100, 1)
            # print("u_val is %d, learn? %r" % (u_val, u_val < self.value))
            if u_val < self.value:
                self.adjacencies.append(adj)
                # import ipdb ; ipdb.set_trace() #z()  # breakpoint f35d9e23 //

        print("Adj after:  " + str(self.adjacencies))



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

    # Every point p will be connected to each other point whose distance
    # to p is less than a cut-off value. This value is computed as the
    # mean of {min_nonzero(dist_mat(p)) | p is a point}, times a factor
    min_nonzero = lambda r: min(r[r > 0])

    # apply_along_axis(f, axis=1, arr) applies f to each row
    min_neighbor_distances = np.apply_along_axis(min_nonzero, axis=1, arr=distance_mat)

    factor = 2.2
    neighbor_cutoff = np.mean(min_neighbor_distances) * factor
    connectivity_matrix = distance_mat < neighbor_cutoff

    return connectivity_matrix


def run(y_pos_dist, cluster_x_dists, cluster_sizes):
    """Creates and sets up a Population and runs initiate_task()."""

    pop = make_population(y_pos_dist, cluster_x_dists, cluster_sizes)
    pop.initiate_task()



if __name__ == '__main__':

    run(y_pos_dist, cluster_x_dists, cluster_sizes)
