from operator import mul
from random import random

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
        value_weights = {attr: {key: value/self.scale for key, value in values.items()}
            for attr,values in self.attributes.items()}
        for attr in value_weights:
            # p_no_val = 1.0 - sum(value_weights[attr].itervalues())
            # p_dyad_w_no_val_node = 2*p_no_val - p_no_val**2
            if 'no value' in value_weights[attr]:
                p_dyad_w_no_val_node = 2*value_weights[attr]['no value'] - value_weights[attr]['no value']**2
            else:
                p_dyad_w_no_val_node = 0
            p_dyad_w_matched_val = sum(value_weights[attr][k]**2 for k in value_weights[attr].keys() if k <> 'no value')
            p_dyad_wo_matched_val = 1.0 - p_dyad_w_matched_val - p_dyad_w_no_val_node
            p_edge_matched = sum(value_weights[attr][k] * self.initial_probs[attr][k] for k in value_weights[attr].keys() if k <> 'no value')
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
