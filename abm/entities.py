# -*- coding: utf-8 -*-
"""
    abm.entities
    ~~~~~~~~~~~~

    Individual objects from agent world
"""
from uuid import uuid4
from collections import defaultdict
from random import choice
from scipy.stats.distributions import uniform


class Entity(object):
    """An entity in our world"""

    def __init__(self, population, index):
        self.index = index
        self.population = population

        self.sent = []    # list of points to whom I've passed the current message
        self.value = 0    # value awarded for a successful message

        # extended adjacency list (see pass_message() and learn())
        self.adjacencies = []

        # task_attempt_map[task_id] hold a list of the neighbors to whom
        # this entity gave the task
        self.task_attempt_map = defaultdict(lambda: [index])

    def log(self, msg):
        self.population.log(msg)

    def receive_task(self, task, sender):
        """
        Handles a message received. Calls pass_message(). Use
        sender=None if this is the Entity that receives the original
        message.
        """

        self.log("%d: past receipients are " % (self.index,) + str(self.sent))

        if sender:
            if sender not in self.task_attempt_map[task.id]:
                self.task_attempt_map[task.id].append(sender)

            if sender in self.sent:
                self.log('popping %s' % self.sent.pop(self.sent.index(sender)))

        # If I have passed the message to all of my neighbors and I still
        # received it back, bail out.
        if set(self.adjacencies).issubset(set(self.task_attempt_map[task.id])):
            # FIXME: this catches cycles eventually, but also catches randomly
            # followed closed paths
            self.log('I have passed the message to all of my neighbors, so I quit')
            self.population.clear()
            return

        if self.index == task.target:
            self.log('message delivered!')
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
        self.log("%s -> %s" % (self.index, next_recipient))

        self.sent.append(next_recipient)
        self.population.pass_message(next_recipient, task, self.index)

    def set_adjacencies(self, adjacencies):
        """
        Builds the adjacency list for this Entity. 
        :param adjacencies: set of integers referencing other entities in self.population
        """
        self.adjacencies = list(adjacencies)

    def award(self, value):
        """
        Accepts <value> as award after having participated in the
        successful routing of a message. Calls self.learn() and then
        cleans up to prepare for the next message.
        """

        self.value += value
        self.log("")
        self.log("awarding %.2f to %d" % (self.value, self.index))
        self.log("Map: " + str(dict(self.task_attempt_map)))
        self.log("Adj before: " + str(self.adjacencies))

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

        # self.log("learning on %d" % self.index)

        for adj in self.sent:
            assert adj in self.adjacencies

            u_val = uniform.rvs(0, 100, 1)
            # self.log("u_val is %d, learn? %r" % (u_val, u_val < self.value))
            if u_val < self.value:
                self.adjacencies.append(adj)

        self.log("Adj after:  " + str(self.adjacencies))


class XyEntity(Entity):
    """An entity in x-y world"""

    def __init__(self, population, index, x, y, cluster):
        super(XyEntity, self).__init__(population, index)
        self.x = x
        self.y = y
        self.cluster = cluster


class NxEntity(Entity):
    """An entity that can be a Nx graph node
    It can have more freeform attributes and doesn't need x, y positions for viz
    """
    def __init__(self, population, index, **kwargs):
        super(NxEntity, self).__init__(population, index)
        self.update(kwargs)

    def update(self, kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, attr):
        return getattr(self, attr)

    # def __getitem__(self, attr):
    #     return getattr(self, attr)


class Task(object):
    """Represents a message to be routed."""

    def __init__(self, target):
        self.target = target
        self.id = uuid4()
        self.value = 100