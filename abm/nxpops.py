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
from scipy.stats import norm

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

        while not nx.is_connected(self.graph):
            print("Not connected!")
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
        Logs (prints; todo: use logger) a human readable string
        """
        self.log("""
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
            attribute_counts={k: dict(v) for k, v in self.attribute_counts.items()}))


class SoftmaxNxEnvironment(pops.TaskFeatureMixin, NxEnvironment):
    """
    A NetworkX environment where tasks have categorical feature vectors
    and nodes use that vector to select the appropriate neighbor
    """
    def __init__(self, *args, **kwargs):
        _kwargs = dict(kwargs)
        _kwargs.setdefault('entity_class', entities.SoftmaxNode)
        self.node_index_indicator = _kwargs.pop('node_index_indicator', False)

        super(SoftmaxNxEnvironment, self).__init__(*args, **_kwargs)


class PathTreeMixin(object):
    """
    Track the message traversals in a nx graph datastructure
    Give awards based on direct distance from element to target,
    with highest awards given in the middle of the path.
    """
    advantage_distribution = norm(0, .33)

    def _calculate_direct_lengths(self):
        """
        Convert our path with potential loops into a directed tree
        Store the normalized direct distance for each node in the path
        """
        g = nx.DiGraph()
        path = self.path
        target = path[-1]
        for ix in range(len(path) - 2, -1, -1):
            pair = path[ix], path[ix + 1]
            if not g.neighbors(pair[0]):
                g.add_edge(*pair)

        direct_path_lens = {
            n: nx.shortest_path_length(g, n, target)
            for n in g.node
        }

        max_len = float(max(direct_path_lens.values()))

        self.normalized_node_len_map = {
            n: (p / max_len) - .5
            for n, p in direct_path_lens.iteritems()
        }

    def _distribute_awards(self, task):
        if self.path[-1] == task.target:
            self._calculate_direct_lengths()
        else:
            self.normalized_node_len_map = None

        for node in self.path_tree.node:
            amount = self._calculate_award(task, _, node)
            self.population[node].award(amount)

    def _calculate_award(self, task, path, entity):
        if self.normalized_node_len_map is None:
            return 0
        normalized_len = self.normalized_node_len_map[entity]
        advantage_coef = self.advantage_distribution.pdf(normalized_len)
        return advantage_coef * task.value


class SoftmaxNxPathTreeEnvironment(PathTreeMixin, SoftmaxNxEnvironment):
    __doc__ = '\n'.join([SoftmaxNxEnvironment.__doc__, PathTreeMixin.__doc__])
