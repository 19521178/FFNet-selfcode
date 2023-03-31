from types import SimpleNamespace
# import json


class DictNamespace(dict):

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = self.construct_namespace(value)

    def __delattr__(self, key):
        del self[key]

    @staticmethod
    def construct_namespace(maybe_dict):
        if isinstance(maybe_dict, dict):
            for key, value in maybe_dict.items():
                if isinstance(value, dict):
                    maybe_dict[key] = DictNamespace(**value)
                elif isinstance(value, SimpleNamespace):
                    maybe_dict[key] = DictNamespace(**value.__dict__)
                else:
                    maybe_dict[key] = value
        return maybe_dict

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        DictNamespace.construct_namespace(self)