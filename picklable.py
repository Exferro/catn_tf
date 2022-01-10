import os
import pickle

from abc import ABC


class Picklable(ABC):
    def __init__(self):
        super(Picklable, self).__init__()

    def pickle(self, *, dest_dir: str = None, name: str = None):
        dest_dir = dest_dir if dest_dir is not None else './'
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        path_to_pickle = os.path.join(dest_dir, f'{self.__class__.__name__}_{name}.pickle')
        with open(path_to_pickle, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return path_to_pickle

    @classmethod
    def load(cls, path_to_pickle: str = None):
        if path_to_pickle is None:
            raise ValueError(f'None path_to_pickle was passed to the load function '
                             f'of class {cls.__name__}')
        elif not os.path.exists(path_to_pickle):
            raise ValueError(f'path_to_pickle = {path_to_pickle} passed to the load function '
                             f'of class {cls.__name__} doesn\'t exist')
        with open(path_to_pickle, 'rb') as handle:
            obj = pickle.load(handle)

        return obj
