# -*- coding: utf-8 -*-
"""
    abm.nxpops
    ~~~~~~~~~~

    Network-x backed populations
"""

import networkx as nx
from operator import mul
from random import random

from random import randint
from itertools import combinations
import numpy as np
from matplotlib.pylab import plt
from collections import defaultdict

from abm import pops, entities


class AttributeGenerator(object):
    def __init__(self, attributes, scale):
        """
        Accepts a dictionary of {attribute: {value: k}}
        where attribute like 'color', value like 'blue', k like '35'
        sum of all values (k) for each attribute should be <= scale.
        Provides a <get_value> method to draw an attribute value according to its probability dist
        """
        self.attributes = attributes
        self.scale = scale
        self._attr_data = {}
        self._setup_attr_data()

    def _setup_attr_data(self):
        for attribute, value_dist in self.attributes.items():
            value_names = value_dist.keys()
            value_cumsum = np.cumsum([value_dist[k] for k in value_names])
            self._attr_data[attribute] = dict(names=value_names, cumsum=value_cumsum)

    def get_value(self, attribute):
        flip = randint(1, self.scale)
        attr_data = self._attr_data[attribute]
        matched_value_index = np.searchsorted(attr_data['cumsum'], flip)
        if matched_value_index == len(attr_data['names']):
            return 'no value'
        return attr_data['names'][matched_value_index]


class EdgeGenerator(object):
    def __init__(self, attributes, initial_probs, scale, density):
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
        self.initial_probs = initial_probs
        self.scale = float(scale)
        self.density = density
        self.density_adjuster = 1.0
        self._compute_adjuster()

    def _compute_adjuster(self):
        p_edge_by_attr = []
        value_weights = {
            attr: {key: value/self.scale for key, value in values.items()}
            for attr, values in self.attributes.items()
        }
        for attr in value_weights:
            # p_no_val = 1.0 - sum(value_weights[attr].itervalues())
            # p_dyad_w_no_val_node = 2*p_no_val - p_no_val**2
            if 'no value' in value_weights[attr]:
                p_dyad_w_no_val_node = 2*value_weights[attr]['no value'] - value_weights[attr]['no value']**2
            else:
                p_dyad_w_no_val_node = 0
            p_dyad_w_matched_val = sum(value_weights[attr][k]**2 for k in value_weights[attr].keys() if k <> 'no value')
            p_dyad_wo_matched_val = 1.0 - p_dyad_w_matched_val - p_dyad_w_no_val_node
            p_edge_matched = sum(value_weights[attr][k]**2 * self.initial_probs[attr][k] for k in value_weights[attr].keys() if k <> 'no value')
            p_edge_wo_match = p_dyad_wo_matched_val * self.initial_probs[attr]['diff']
            p_edge_w_no_val_node = p_dyad_w_no_val_node * 0.5
            p_edge = p_edge_matched + p_edge_wo_match + p_edge_w_no_val_node
            p_edge_by_attr.append(p_edge)
        self.density_adjuster = self.density / reduce(mul, p_edge_by_attr, 1)

    def set_edge(self, node1, node2):
        p_edge = []
        for attr in self.attributes:
            if node1[attr] == 'no value' or node2[attr] == 'no value':
                p_edge.append(0.5)
            elif node1[attr] == node2[attr]:
                p_edge.append(self.initial_probs[attr][node1[attr]])
            else:
                p_edge.append(self.initial_probs[attr]['diff'])
        prob = self.density_adjuster * reduce(mul, p_edge, 1)
        return random() <= prob


class NxPopulation(pops.Population):
    """docstring for NxPopulation"""
    def __init__(self, attributes, edge_probs, seed=0, size=100, density=.1, debug=True):
        super(NxPopulation, self).__init__(debug=debug)
        self.attributes = attributes
        self.edge_probs = edge_probs
        self.seed = seed
        self.size = size
        self.density = density

        self._setup_nx_graph(attributes, edge_probs, seed, size, density)
        self.points = self.graph.node

    def _setup_nx_graph(self, attributes, edge_probs, seed, size, density):
        # instantiate attribute generator to distribute attributes over nodes
        # takes attributes dictionary and the network size as parms
        attr_gen = AttributeGenerator(attributes, size)

        # instantiate edge generator to determine dyadic ties
        # takes attributes dictionary,the unscaled probabilities of ties
        # between nodes of similar or disimilar type, network size and density as parms
        edge_gen = EdgeGenerator(attributes, edge_probs, size, density)

        # create an empty graph
        G = nx.Graph()

        # create and distribute node attributes and record which nodes are in which attribute class
        attributeSets = {attr: defaultdict(set) for attr in attributes}
        attributeCounts = {attr: defaultdict(lambda: 0) for attr in attributes}

        for i in range(size):
            node_attrs = {}
            for attribute in attributes:
                value = attr_gen.get_value(attribute)
                attributeSets[attribute][value].add(i)
                attributeCounts[attribute][value] += 1
                node_attrs[attribute] = value
            entity = entities.NxEntity(index=i, population=self, **node_attrs)
            G.add_node(i, entity)

        # iterate over dyads of nodes and set an edge between them if set_edge returns true
        for dyad in combinations(nx.nodes(G), 2):
            if edge_gen.set_edge(G.node[dyad[0]], G.node[dyad[1]]):
                    G.add_edge(*dyad)
        self.graph = G

    def display(self, current=None, target=None):
        if not self.show:
            return
        nx.draw_spring(self.graph)
        plt.show()

