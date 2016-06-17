# -*- coding: utf-8 -*-
"""
    abm.learners
    ~~~~~~~~~~~~

    Learning mixins to add to entity subclasses
    Should implement _get_next_recipient and _learn
"""
import numpy as np
import operator
from random import choice


class DunceMixin(object):
    """
    A learner that doesn't learn
    """
    def _get_next_recipient(self, task):
        return choice(self.adjacencies)

    def _learn(self):
        pass


class SoftmaxLearnerMixin(object):
    last_recipient = None
    w_container = None
    update_buffer = None
    policy_duration = 1

    def _get_next_recipient(self, task):
        """
        Uses task.features and self.w_container to find the best neighbor for this task
        Sets "pending" state once a decision is made
        """

        # lazily initialize random weights and weight update buffer (for policy rollouts)
        if self.w_container is None:
            self.w_container = {a: np.random.random(task.features.shape) for a in self.adjacencies}
        if self.update_buffer is None:
            self.flush_count = 0
            self.update_buffer = []

        # if you have a decision pending feedback and are asked to make another,
        # mark that last decision as 'wrong' before proceeding
        if self.last_recipient is not None:
            self.award(0)

        self.latest_x = x = task.features

        # find the best neighbor for this task
        self.softmaxes = _exp_over_sumexp(x, self.w_container)
        if task.target in self.adjacencies:
            # don't actually use your weights to decide if the neighbor is visible
            self.last_recipient = decision = task.target
        else:
            decision = max(self.softmaxes.iteritems(), key=operator.itemgetter(1))[0]
            self.last_recipient = decision
        return decision

    def _learn(self):
        assert self.last_recipient is not None

        grad = _gradient_precomputed(self.last_recipient, self.softmaxes,
                                     self.latest_x, self.value > 0)
        self.log(grad)
        w_adjustment = grad * (self.value / 100. if self.value > 0 else 1)

        self.update_buffer.append((self.last_recipient, w_adjustment))

        self.last_recipient, self.softmaxes, self.latest_x = None, None, None
        # if len(self.update_buffer) >= self.policy_duration:
        #     self.flush_updates()

    def flush_updates(self):
        self.flush_count += 1
        for adjacency_index, w_adjustment in self.update_buffer:
            self.w_container[adjacency_index] += w_adjustment
        self.update_buffer = []


def _gradient_precomputed(k, softmaxes, train_x, success):
    # set magnitude of gradient proportionally to confidence
    grad = train_x * ((1 if success else 0) - softmaxes[k])
    return grad


def _exp_over_sumexp(train_x, w_container):
    """
    Calculate an exponential over a sum of exponentials in a numerically stable way
    Returns a dict keying class values k :
        np(exp(x_t*wk)) / sum for j in K {np.exp(x_t*wj)}
    """
    x_t = np.transpose(train_x)
    w_container_keys = w_container.keys()
    dots = map(lambda j: np.dot(x_t, w_container[j]), w_container_keys)

    # the final output is unchanged by removing the maximum `weight * x` value
    # from each dot product before exponentiation
    beta = max(dots)
    sumexp = sum(map(lambda e: np.exp(e - beta), dots))

    return {
        k: (np.exp(dots[ix] - beta) / sumexp)
        for ix, k in enumerate(w_container_keys)
    }
