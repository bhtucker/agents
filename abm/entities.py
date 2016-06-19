# -*- coding: utf-8 -*-
"""
    abm.entities
    ~~~~~~~~~~~~

    Individual objects from agent world
"""
from uuid import uuid4
from random import choice

from abm.learners import SoftmaxLearnerMixin, DunceMixin


class Task(object):
    """Represents a message to be routed."""

    def __init__(self, target, features=None):
        self.target = target
        self.id = uuid4()
        self.value = 100
        if features is not None:
            self.features = features


class Entity(object):
    """An entity in our world"""

    def __init__(self, environment, index):
        self.index = index
        self.environment = environment

        self.value = 0    # value awarded for a successful message

    def log(self, msg):
        self.environment.log(msg)

    def _get_next_recipient(self, task):
        """
        Returns the node index that this entity would like to hand message to
        May enter a 'pending' state recording this decision for future feedback
        """
        raise NotImplementedError

    def next(self, task, sender):
        """
        Called by environment to get the next step in the path
        """
        next_recipient = self._get_next_recipient(task)

        self.log("%s -> %s" % (self.index, next_recipient))
        return next_recipient

    def award(self, value):
        """
        Accepts <value> as award to register the outcome of recent decision(s).
        Calls self._learn() and then cleans up to prepare for the next message.
        """
        self.value = value
        self.log("""
            Node {ix} receiving award {award}
            """.format(ix=self.index, award=value))

        self._learn()

        self.value = 0

    def _learn(self):
        """
        Update node's brain, however that is implemented in subclass.
        May resolve a 'pending' state by taking the current self.value as its outcome
        """
        raise NotImplementedError


class XyEntity(Entity):
    """An entity in x-y world"""

    def __init__(self, environment, index, x, y, cluster):
        super(XyEntity, self).__init__(environment, index)
        self.x = x
        self.y = y
        self.cluster = cluster
        self.adjacencies = []

    def set_adjacencies(self, adjacencies):
        """
        Builds the adjacency list for this Entity.
        :param adjacencies: set of integers referencing other entities in self.population
        """
        self.adjacencies = list(adjacencies)


class NxEntity(Entity):
    """An entity that can be a Nx graph node
    It can have more freeform attributes and doesn't need x, y positions for viz
    """
    def __init__(self, environment, index, **kwargs):
        super(NxEntity, self).__init__(environment, index)
        self.update(kwargs)

    def update(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, attr):
        return getattr(self, attr)

    @property
    def adjacencies(self):
        """
        List connected nodes via the parent graph
        """
        if self.environment and self.environment.graph:
            return self.environment.graph.neighbors(self.index)
        else:
            return []


class DunceNode(DunceMixin, NxEntity):
    """
    Nx compatible node that doesn't learn and just randomly draws its recipients
    """


class XyDunceNode(DunceMixin, XyEntity):
    """
    Xy compatible node that doesn't learn and just randomly draws its recipients
    """


class SoftmaxNode(SoftmaxLearnerMixin, NxEntity):
    """
    Nx nodes with softmax learning
    """
