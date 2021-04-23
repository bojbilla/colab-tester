import numpy as np
from tf_agents.specs import array_spec


class GameState:
    def __init__(self, d1=0, d2=0):
        self._d1 = d1
        self._d2 = d2
        self._score = d1+d2

    def copy(self):
        return GameState(self._d1, self._d2)

    def get_d1(self) -> int:
        return self._d1

    def get_d2(self) -> int:
        return self._d2

    def get_score(self) -> int:
        return self._score

    def set_d1(self, d):
        self._score = self._score - self._d1 + d
        self._d1 = d

    def set_d2(self, d):
        self._score = self._score - self._d2 + d
        self._d2 = d

    def as_obs(self) -> array_spec.BoundedArraySpec:
        return np.array([self._d1, self._d2], dtype=np.int32)
