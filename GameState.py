import numpy as np


class GameState:
    def __init__(self, score=0):
        self._score = score

    def copy(self):
        return GameState(self._score)

    def get_score(self) -> int:
        return self._score

    def update(self, d):
        self._score = self._score + d

    def as_obs(self):
        return np.array([self._score, self._score], dtype=np.int32)
