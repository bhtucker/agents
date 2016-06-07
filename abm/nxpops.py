# -*- coding: utf-8 -*-
"""
    abm.nxpops
    ~~~~~~~~~~

    Network-x backed populations
"""

import networkx as nx

from itertools import combinations
from matplotlib.pylab import plt
from datetime import datetime
from collections import defaultdict

from abm import pops, entities
from abm.generators import EdgeGenerator, AttributeGenerator


class NxEnvironment(pops.Environment):
    """A network graph (with NetworkX)"""
    def __init__(self, attributes, edge_probs, seed=0, size=100, density=.1,
                 entity_class=entities.NxEntity, edge_gen_class=EdgeGenerator,
                 attr_gen_class=AttributeGenerator, debug=True):
        super(NxEnvironment, self).__init__(debug=debug)
        self.created_time_utc = datetime.utcnow().isoformat()
        self.attributes = attributes
        self.edge_probs = edge_probs
        self.seed = seed
        self.size = size
        self.density = density
        self.entity_class = entity_class
        self.edge_gen_class = edge_gen_class
        self.attr_gen_class = attr_gen_class

        self._setup_nx_graph(attributes, edge_probs, seed, size, density)
        self.population = self.graph.node

    def _setup_nx_graph(self, attributes, edge_probs, seed, size, density):
        # instantiate attribute generator to distribute attributes over nodes
        # takes attributes dictionary and the network size as parms
        attr_gen = self.attr_gen_class(attributes, size)

        # instantiate edge generator to determine dyadic ties
        # takes attributes dictionary,the unscaled probabilities of ties
        # between nodes of similar or disimilar type, network size and density as parms
        edge_gen = self.edge_gen_class(attributes, edge_probs, density)

        # create an empty graph
        G = nx.Graph()

        # create and distribute node attributes and record which nodes are in which attribute class
        attribute_counts = {attr: defaultdict(lambda: 0) for attr in attributes}

        for i in range(size):
            node_attrs = {}
            for attribute in attributes:
                value = attr_gen.get_value(attribute)
                attribute_counts[attribute][value] += 1
                node_attrs[attribute] = value
            entity = self.entity_class(index=i, environment=self, **node_attrs)
            G.add_node(i, entity)

        # iterate over dyads of nodes and set an edge between them if set_edge returns true
        for dyad in combinations(nx.nodes(G), 2):
            if edge_gen.set_edge(G.node[dyad[0]], G.node[dyad[1]]):
                    G.add_edge(*dyad)
        self.graph = G
        self.attribute_counts = attribute_counts

    def display(self, current=None, target=None):
        if not self.show:
            return
        nx.draw_spring(self.graph)
        plt.show()

    def describe(self):
        """
        Provide summary statistics about the generated graph
        Returns a human readable string
        """
        return """
        Network creation time (UTC): {ts}
        Network size: {size}
        Target density: {config_density}
        Actual density: {real_density}
        Target attributes: {attributes}
        Actual attribute counts: {attribute_counts}
        """.format(
            ts=self.created_time_utc,
            size=self.size,
            config_density=self.density,
            real_density=nx.density(self.graph),
            attributes=self.attributes,
            attribute_counts={k: dict(v) for k, v in self.attribute_counts.items()})


class SoftmaxNxEnvironment(pops.TaskFeatureMixin, NxEnvironment):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('entity_class', entities.SoftmaxNode)
        super(SoftmaxNxEnvironment, self).__init__(*args, **kwargs)
