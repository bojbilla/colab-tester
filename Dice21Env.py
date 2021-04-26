from enum import Enum
from typing import Any, Optional

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec, BoundedArraySpec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories.time_step import TimeStep

from GameState import GameState


class Dice21Env(py_environment.PyEnvironment):

    def __init__(self):
        self._episode_ended = False
        self._state = GameState()
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.int32, minimum=0, maximum=28, name='observation')

    def get_state(self) -> GameState:
        return self._state

    def set_state(self, state: GameState) -> None:
        self._state = state.copy()

    def action_spec(self) -> BoundedArraySpec:
        return self._action_spec

    def observation_spec(self) -> BoundedArraySpec:
        return self._observation_spec

    def _reset(self) -> TimeStep:
        self._state = GameState()
        self._episode_ended = False
        return ts.restart(self._state.as_obs())

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()

        # Make sure episodes don't go on forever.
        if action == Action.STOP.value:
            self._episode_ended = True
        elif action == Action.THROW_D1.value:
            new_dice = np.random.randint(1, 7)
            self._state.update(new_dice)
        else:
            raise ValueError('`action` invalid.')

        if self._episode_ended or self._state.get_score() >= 21:
            reward = self._state.get_score()/21 if self._state.get_score() <= 21 else -1
            return ts.termination(self._state.as_obs(), reward)
        else:
            return ts.transition(self._state.as_obs(), reward=0.001, discount=1)


class Action(Enum):
    STOP = 0
    THROW_D1 = 1

