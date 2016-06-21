# -*- coding: utf-8 -*-
"""
    abm.pops
    ~~~~~~~~

    Environments of interconnected agents
"""
from abm.entities import Task
from cached_property import cached_property
import numpy as np


class Environment(object):
    def __init__(self, debug=True, path_cutoff=20):
        self.path = []
        self.show = True
        self.debug = debug
        self.path_cutoff = path_cutoff

    def log(self, msg):
        """Prints a message to stdout, only if self.debug=True."""
        if self.debug:
            print(msg)

    def _distribute_awards(self, task):
        awarded = set()
        # the last entry in the path is the stopping point, which made no decisions
        for point in self.path[:-1]:
            if point in awarded:
                continue
            awarded.add(point)
            amount = self._calculate_award(task, self.path, point)
            self.population[point].award(amount)
        if hasattr(self, 'flush_updates'):
            self.flush_updates()

    def _calculate_award(self, task, path, entity):
        """
        Returns the amount awarded to <entity> after <task> has been
        routed through <path>.
        """
        if len(self.path) >= self.path_cutoff:
            # detect failure
            return -1. / self.path_cutoff
        # in a one traversal case, the one traversal gets all the credit
        k = float(len(path) - 1)
        return (task.value / k)

    def initiate_task(self, fixed_pair=None):
        """Initializes a task, and calls pass_message()."""

        start, end = [-1, -1] if not fixed_pair else fixed_pair

        if fixed_pair and start == end:
            self.log("changing fixed pair: they're the same!")

        while start == end:
            [start, end] = self._pick_start_end()

        self.log('new task created')
        self.log('starting at %s and aiming for %s' % (start, end))

        task = self._generate_task(end)
        self.path = []
        self._run_task(start, task)

    def _generate_task(self, target):
        return Task(target)

    def _run_task(self, start, task):

        self.path.append(start)
        # take first step
        recipient = self.population[start].next(task, sender=None)
        sender = start
        self.path.append(recipient)

        while not (recipient == task.target or len(self.path) >= self.path_cutoff):
            next_recipient = self.population[recipient].next(task, sender=sender)
            sender = recipient
            recipient = next_recipient
            self.path.append(recipient)

        self._distribute_awards(task)

    def _pick_start_end(self):
        return np.random.randint(self.size, size=2).tolist()

    def clear(self):
        """
        Clear every Entity's history. Called when the package is stuck
        in a cycle.
        """
        point_iterator = (self.population
                          if isinstance(self.population, list)
                          else self.population.values())

        for point in point_iterator:
            point.sent = []


class TaskFeatureMixin(object):
    """Overrides _generate_task with a feature-defining version"""
    bias = False
    node_index_indicator = False

    @cached_property
    def _attribute_categories(self):
        return [(k, v.keys()) for k, v in self.attributes.items()]

    def _generate_task(self, target):
        """
        Convert categorical data about the target into an indicator vector
        Return a task with this vector accessible as task.features
        """
        target_node = self.population[target]
        feature_vec_components = [np.ones(1)] if self.bias else []
        for attribute, categories in self._attribute_categories:
            component = np.zeros(len(categories))
            target_val = getattr(target_node, attribute)
            component[categories.index(target_val)] = 1
            feature_vec_components.append(component)

        if self.node_index_indicator:
            component = np.zeros(len(self.population))
            component[target] = 1
            feature_vec_components.append(component)

        feature_vec = np.hstack(feature_vec_components)
        task = Task(target, features=feature_vec)
        return task
