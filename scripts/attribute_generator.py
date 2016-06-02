import random
import numpy as np


class AttributeGenerator(object):
    def __init__(self, attributes, scale):
        """
        Accepts a dictionary of {attribute: {value: k}}
        where attribute like 'color', value like 'blue', k like '35'
        all values of k should be <= scale
        Provides a <get_value> method to draw an attribute value according to its probability dist
        """
        self.attributes = attributes
        self.scale = scale
        self._setup_attr_data()

    def _setup_attr_data(self):
        for attribute, value_dist in self.attributes.items():
            value_names = value_dist.keys()
            value_cumsum = np.cumsum([value_dist[k] for k in value_names])
            self._attr_data[attribute] = dict(names=value_names, cumsum=value_cumsum)

    def get_value(self, attribute):
        flip = random.randint(1, self.scale)
        attr_data = self._attr_data[attribute]
        matched_value_index = np.searchsorted(attr_data['cumsum'], flip)
        if matched_value_index == len(attr_data['names']):
            return 'no value'
        return attr_data['names'][matched_value_index]

