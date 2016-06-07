# -*- coding: utf-8 -*-
"""
    abm.generators
    ~~~~~~~~~~~~~~

    Objects for distributing node attributes and edges
"""

from random import randint
import numpy as np
from operator import mul
from random import random

NO_VAL = 'no value'
DIFF = 'diff'
ATTR_SCALE = 100.


class AttributeGenerator(object):
    def __init__(self, attributes, scale):
        """
        Accepts a dictionary of {attribute: {value: k}}
        where attribute like 'color', value like 'blue', k like '35'
        sum of all values (k) for each attribute should be <= scale.
        Provides a <get_value> method to draw an attribute value according to its probability dist
        """
        self.attributes = attributes
        self._attr_data = {}
        self._setup_attr_data()

    def _setup_attr_data(self):
        for attribute, value_dist in self.attributes.items():
            value_names = value_dist.keys()
            value_cumsum = np.cumsum([value_dist[k] for k in value_names])
            self._attr_data[attribute] = dict(names=value_names, cumsum=value_cumsum)

    def get_value(self, attribute):
        flip = randint(1, int(ATTR_SCALE))
        attr_data = self._attr_data[attribute]
        matched_value_index = np.searchsorted(attr_data['cumsum'], flip)
        if matched_value_index == len(attr_data['names']):
            return NO_VAL
        return attr_data['names'][matched_value_index]


class EdgeGenerator(object):
    def __init__(self, attributes, edge_probs, density):
        """
        Accepts a dictionary of {attribute: {value: k}},
        where attribute like 'color', value like 'blue', k like '35'
        sum of all values (k) for each attribute should be <= scale;
        initial probability vector of liklihoods of setting and edge between two nodes
        that are similar or differ on each attribute; and desired network density.
        Provides a <set_edge> method to draw an attribute value according to
        probs and self.adjustment
        """
        self.attributes = attributes
        self.edge_probs = edge_probs
        self.density = density
        self.density_adjuster = 1.0
        self._compute_adjuster()

    def _compute_adjuster(self):
        p_edge_by_attr = []
        value_weights = {
            attr: {key: value/ATTR_SCALE for key, value in values.items()}
            for attr, values in self.attributes.items()
        }
        for attr in value_weights:
            # probability vector over values for this attribute
            attr_dist = value_weights[attr]

            # first find the probability of different types of dyads (match, unmatch, noval)
            p_noval = attr_dist.get(NO_VAL, 0.)
            p_dyad_w_no_val_node = 2*p_noval - p_noval**2

            p_dyad_w_matched_val = sum(attr_dist[k]**2 for k in attr_dist if k != NO_VAL)
            p_dyad_wo_matched_val = 1.0 - p_dyad_w_matched_val - p_dyad_w_no_val_node

            # given dyad type probabilities and edge_probs, find expected edge connectivity
            p_edge_matched = sum([
                attr_dist[k]**2 * self.edge_probs[attr][k]
                for k in attr_dist if k != NO_VAL
            ])
            p_edge_wo_match = p_dyad_wo_matched_val * self.edge_probs[attr][DIFF]
            p_edge_w_no_val_node = p_dyad_w_no_val_node * self.edge_probs[attr].get(NO_VAL, 0.)
            p_edge = p_edge_matched + p_edge_wo_match + p_edge_w_no_val_node
            p_edge_by_attr.append(p_edge)

        self.density_adjuster = self.density / reduce(mul, p_edge_by_attr, 1)

    def set_edge(self, node1, node2):
        p_edge = []
        for attr in self.attributes:
            if NO_VAL in (node1[attr], node2[attr]):
                # if either node is missing this value, use the NOVAL edge prob
                p_edge.append(self.edge_probs[attr][NO_VAL])
            elif node1[attr] == node2[attr]:
                # if nodes match on this attr, use the match edge prob for that value
                p_edge.append(self.edge_probs[attr][node1[attr]])
            else:
                # if nodes do not match on this attr, use the differ edge prob
                p_edge.append(self.edge_probs[attr][DIFF])
        prob = self.density_adjuster * reduce(mul, p_edge, 1)
        return random() <= prob
