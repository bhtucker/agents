# -*- coding: utf-8 -*-
"""
abm.io
======

Reading and writing network configuration files, through
:class:`ConfigReader`.

"""

from abm.generators import NO_VAL, ATTR_SCALE, DIFF

import json
import numpy as np


class ConfigReader(object):
    """Interface to reading network settings from a file.

    Read a json configuration file, `filename`, and validate and augment
    settings for use in network generation. The final settings are stored
    in `self.config`.

    :mod:`abm` configuration files are used for determining the properties
    of a network, based on which the network is randomly created.

    :Example:

    A typical configuration file looks like this:

    .. code-block:: json

        {
          "attributes": {
            "color": {
              "blue": 40,
              "green": 25,
              "red": 35
            },
            "region": {
              "west": 45,
              "east": 50
            }
          },
          "size": 100,
          "entity_kwargs":{
            "policy_duration": 1
          },
          "edge_probs": {
            "color": {
              "blue": 0.2,
              "diff": 0.1,
              "green": 0.25,
              "red": 0.15
            },
            "region": {
              "west": 0.2,
              "east": 0.2,
              "diff": 0.08
            }
          },
          "density": 0.1
        }

    This defines a network with 100 nodes (`size`) and desired edge density
    of 0.1 (`density`).

    The keys of the `attributes` dict are used as features to define each
    node's identity, and the values are the desired proportion of nodes
    that will have this feature, normalaized to 100. This configuration
    file defines a network in which 40% of the nodes are of `color` `blue`.

    The keys of the `edge_probs` dict are the same features as before, but
    now the values are the probabilities with which an edge joining two
    nodes with the same value in this feature will be connected. There is
    also a probability for forming an edge between nodes with different
    values for this attribute. In the above example, the network will be
    built so that two nodes with the value 'blue' for the attribute `color`
    will be joined with 0.2 probability, while two nodes with different
    `color` values will be joined with 0.1 probability, coming from the
    value for `diff`.

    .. note:: Neither the total of the values in `attributes` has to add up
              to 100, nor the total of the probabilities in `edge_probs`
              has to add up to 1.0. If they don't, :class:`ConfigReader`
              fills in the missing data.

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

        self.config = dict(self.raw_config,
                           attributes=self.attributes,
                           edge_probs=self.edge_probs)

    def get_config(self):
        return self.config

    def _validate_configs(self, attributes, edge_probs):
        """Assert the attribute/edge settings are internally consistent."""
        msg = 'Configuration looks wrong.'

        for attr, inner_dict in attributes.items():
            assert sum(inner_dict.values()) <= ATTR_SCALE, msg
            assert all([isinstance(v, int) for v in inner_dict.values()]), msg
            assert all([k in edge_probs[attr] for k in inner_dict]), msg

        for attr, inner_dict in edge_probs.items():
            assert DIFF in inner_dict, msg
            assert all([0 <= v <= 1 for v in inner_dict.values()]), msg

    def _set_noval_edge_prob(self, attr):
        """Set probability of edge creation when at least one node has missing data.

        This probability is defined to be mean( mean(prob for matched values),
        prob for unmatched values).

        """
        attr_edge_p = self.edge_probs[attr]
        mean_match_prob = np.mean([
            attr_edge_p[k] for k in attr_edge_p
            if k not in {NO_VAL, DIFF}
        ])
        attr_edge_p[NO_VAL] = np.mean([mean_match_prob, attr_edge_p[DIFF]])

    def _fill_noval_attribute_prob(self, attr):
        """Set probability of getting a NO_VAL for the given attribute."""
        attr_dist = self.attributes[attr]
        observed_scale = sum(attr_dist.values())

        if observed_scale < ATTR_SCALE:
            attr_dist[NO_VAL] = int(ATTR_SCALE - observed_scale)
            return True
        else:
            return False
