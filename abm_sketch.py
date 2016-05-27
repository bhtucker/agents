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

cluster_colors = {
	'A': 'r',
	'B': 'b',
	'C': 'g'
}

MESSAGES = [
	'go fish',
	"don't go fish"
]

class Population(object):
	def __init__(self):
		self.points = []
		self.path = []
		self.show = True
		self.success_lens = []

	def pass_message(self, recipient, task, sender):
		display(self.points, self.connectivity_matrix, recipient, task.target, show=self.show)
		self.path.append(recipient)
		if recipient == task.target:
			# import ipdb ; ipdb.set_trace#z()  # breakpoint 223cdcc4 //

			k = float(len(self.path))
			for ix, point in enumerate(self.path):
				i_value = (task.value / k) + ((k - ix) * 5)
				self.points[point].award(i_value)
			self.success_lens.append(len(self.path))
			self.path = []

		self.points[recipient].receive_task(task, sender)

	def set_connections(self):
		self.connectivity_matrix = get_connectivity_matrix(self.points)
		for index, point in enumerate(self.points):
			point.set_adjacencies(self.connectivity_matrix[index])

	def display(self):
		display(self.points, self.connectivity_matrix, show=self.show)

	def initiate_task(self, fixed_pair=None):
		start, end = ([-1, -1] if not fixed_pair else fixed_pair)
		while start == end:
			[start, end] = np.random.randint(len(self.points), size=2).tolist()
		message = choice(MESSAGES)
		print('new task: %s' % message)
		print('starting at %s and aiming for %s' % (start, end))
		task = Task(end, message)
		self.pass_message(start, task, len(self.points))

	def clear(self):
		for point in self.points:
			# point.adjacencies = list(set(point.adjacencies))
			adjs = point.adjacencies, set(point.adjacencies)
			if len(adjs[0]) != len(adjs[1]):
				print("clearing %s" % point.index + '---' * 20)
				print(point.adjacencies, set(point.adjacencies))
			point.sent = []

class Task(object):
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

	def compare(self, other):
		pass

	def pass_message(self, task, sender):
		next_recipient = task.target if task.target in self.adjacencies else self.index
		while next_recipient in self.task_attempt_map[task.id]:
			next_recipient = choice(self.adjacencies)
		self.task_attempt_map[task.id].append(next_recipient)
		print "passing message from %s to %s" % (self.index, next_recipient)
		self.sent.append(next_recipient)
		self.population.pass_message(next_recipient, task, self.index)

	def set_adjacencies(self, connectivity_vector):
		for connect_ix, connected in enumerate(connectivity_vector):
			if connected and connect_ix != self.index:
				self.adjacencies.append(connect_ix)


	def award(self, value):
		self.value += value
		print(self.value, self.index, self.adjacencies)

		for adj in self.sent:
			assert adj in self.adjacencies
			u_val = uniform.rvs(0, 100, 1)
			print(u_val, u_val < value)
			if u_val < value:
				self.adjacencies.append(adj)
				# import ipdb ; ipdb.set_trace() #z()  # breakpoint f35d9e23 //
		self.value = 0
		self.sent = []

def make_points(cluster, size):
	ys = y_pos_dist.rvs(size)
	xs = cluster_x_dists[cluster].rvs(size)
	return list(zip(xs, ys, [cluster] * size))


def make_population():
	points = []
	for cluster, count in cluster_sizes.iteritems():
		points += make_points(cluster, count)

	population = Population()
	for ix, point in enumerate(points):
		pt = Entity(population, ix, *point)
		population.points.append(pt)
	population.set_connections()
	return population


def display(points, connectivity_matrix=None, current_ix=None, target_ix=None, show=True):
	if not show:
		return
	for cluster, color in cluster_colors.iteritems():
		class_points = [x for x in points if x.cluster == cluster]
		plt.scatter([p.x for p in class_points], [p.y for p in class_points], c=color, s=35)
	
	if connectivity_matrix is not None:
		for start_ix, connections in enumerate(connectivity_matrix):
			for connect_ix, connected in enumerate(connections):
				if connected and connect_ix != start_ix:
					plt.plot(*zip(
						(points[start_ix].x, points[start_ix].y),
						(points[connect_ix].x, points[connect_ix].y)),
						c='k', linewidth=0.5)
	if current_ix and target_ix:
		plt.scatter(points[current_ix].x, points[current_ix].y, c='m', s=150)
		plt.scatter(points[target_ix].x, points[target_ix].y, c='y', s=190)

	plt.show()


def get_connectivity_matrix(points):
	points_arr = np.array([[p.x, p.y] for p in points])
	min_nonzero = lambda r: min(r[r > 0])
	distance_mat = euclidean_distances(points_arr, points_arr) 
	min_neighbor_distances = np.apply_along_axis(min_nonzero, axis=1, arr=distance_mat)
	neighbor_cutoff = np.mean(min_neighbor_distances) * 2.2
	connectivity_matrix = distance_mat < neighbor_cutoff
	return connectivity_matrix

	
