import pickle
from pathlib import Path

from data.base_file import BaseFile


class TexterPkl(BaseFile):

    def __init__(self, path: Path):
        super().__init__(path)

    def load(self):
        with open(self.path, 'rb') as f:
            return pickle.load(f)
