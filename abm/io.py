# -*- coding: utf-8 -*-
"""
    abm.io
    ~~~~~~

    Reading and writing network configs / performance results
"""

from abm.generators import NO_VAL, ATTR_SCALE, DIFF

import json
import numpy as np


class ConfigReader(object):
    """
    Interface to reading node/edge settings from file
    Validates and augments settings for use in network generation
    """
    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.raw_config = json.load(f)
        self.attributes, self.edge_probs = (
            self.raw_config['attributes'], self.raw_config['edge_probs']
        )
        self._validate_configs(self.attributes, self.edge_probs)

        for attr in self.edge_probs:
            if self._fill_noval_attribute_prob(attr):
                self._set_noval_edge_prob(attr)
        self.config = dict(self.raw_config, attributes=self.attributes, edge_probs=self.edge_probs)

    def get_config(self):
        return self.config

    def _validate_configs(self, attributes, edge_probs):
        """
        Assert the attribute/edge settings are internally consistent
        """
        msg = 'configuration looks wrong'
        for attr, inner_dict in attributes.items():
            assert sum(inner_dict.values()) <= ATTR_SCALE, msg
            assert all([isinstance(v, int) for v in inner_dict.values()]), msg
            assert all([k in edge_probs[attr] for k in inner_dict]), msg

        for attr, inner_dict in edge_probs.items():
            assert DIFF in inner_dict, msg
            assert all([0 <= v <= 1 for v in inner_dict.values()]), msg

    def _set_noval_edge_prob(self, attr):
        """
        Set probability of edge creation when at least one dyad member has missing data
        defined to be mean( mean(prob for matched values), prob for unmatched values)
        """
        attr_edge_p = self.edge_probs[attr]
        mean_match_prob = np.mean([
            attr_edge_p[k] for k in attr_edge_p
            if k not in {NO_VAL, DIFF}
        ])
        attr_edge_p[NO_VAL] = np.mean([mean_match_prob, attr_edge_p[DIFF]])

    def _fill_noval_attribute_prob(self, attr):
        """
        Set probability of getting a NO_VAL value for the given attribute
        """
        attr_dist = self.attributes[attr]
        observed_scale = sum(attr_dist.values())

        if observed_scale < ATTR_SCALE:
            attr_dist[NO_VAL] = int(ATTR_SCALE - observed_scale)
            return True
        else:
            return False
