# -*- coding: utf-8 -*-
"""
Enables, saving and loading of custom configuration files that are easily extendable
"""

import json
import os


class Dict(dict):
    """
    Dictionary that allows to access per attributes and to except names from being loaded
    """
    def __init__(self, dictionary: dict = None):
        super(Dict, self).__init__()

        if dictionary is not None:
            self.load(dictionary)

    def __getattr__(self, item):
        return self[item] if item in self else getattr(super(Dict, self), item)

    def load(self, dictionary: dict, name_list: list = None) -> None:
        """
        Loads a dictionary
        Args:
            dictionary (dict): Dictionary to be loaded
            name_list (list): List of names to be updated
        """
        for name in dictionary:
            data = dictionary[name]
            if name_list is None or name in name_list:
                if isinstance(data, dict):
                    if name in self:
                        self[name].load(data)
                    else:
                        self[name] = Dict(data)
                else:
                    self[name] = data

    def save(self, path: str) -> None:
        """
        Saves the dictionary into a json file
        Args:
            path (str): Path of the json file
        """
        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, 'config.json')
        with open(path, 'w') as file:
            json.dump(self, file, indent=True)


class Configuration(Dict):
    """
    Configuration loaded from a json file
    """
    def __init__(self, path: str) -> None:
        super(Configuration, self).__init__()
        self.load(path)

    def load_model(self, path: str) -> None:
        self.load(path, name_list=["model"])

    def load(self, path: str, name_list: list = None) -> None:
        """
        Loads attributes from a json file
        Args:
            path (str): Path of the json file
            name_list (list): List of names to be updated
        """
        with open(path) as file:
            data = json.load(file)
            super(Configuration, self).load(data, name_list)
