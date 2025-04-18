import dataclasses
import itertools
import math
from typing import List
import numpy as np

from .dart_logic import Action

@dataclasses.dataclass(frozen=True)
class RingActionSpace():
    r: List[float | int] | float | int
    theta_start: float = 0.0
    theta_end: float = 360.0
    theta_step: float = 6.0

    def __iter__(self):
        rs = [self.r] if not isinstance(self.r, list) else self.r
        return itertools.chain.from_iterable(
            (
                Action.polar_to_action(r, theta=theta / 360 * (math.pi * 2))
                for theta in np.arange(
                    self.theta_start, self.theta_end, self.theta_step
                ).tolist()
            )
            for r in rs
        )

@dataclasses.dataclass(frozen=True)
class GridActionSpace():
    x_start: float = -16.0
    x_end: float = 16.0
    x_ngrid: int = 9
    y_start: float = -16.0
    y_end: float = 16.0
    y_ngrid: int = 9

    def __iter__(self):
        return (
            Action(
                self.x_start + (self.x_end - self.x_start) / (self.x_ngrid - 1) * x,
                self.y_start + (self.y_end - self.y_start) / (self.y_ngrid - 1) * y
            )
            for x, y in itertools.product(
                range(self.x_ngrid), range(self.y_ngrid)
            )
        )

@dataclasses.dataclass(frozen=True)
class DotActionSpace():
    x: float = 0.0
    y: float = 0.0

    def __iter__(self):
        return (Action(self.x, self.y) for _ in range(1))

def complex_action_space():
    # 9x9 in x, y [-16, 16] Square
    yield from GridActionSpace(-16, 16, 9, -16, 16, 9)
    # for r in [100, 103, 106, 163, 166, 169], 120 equally distributed dots
    yield from RingActionSpace([100, 103, 106, 163, 166, 169], theta_step=3)
    # for r in [58, 135], 60 equally distributed dots
    yield from RingActionSpace([58, 135], theta_step=6)

    # Totally 921 actions


def middle_action_space():
    # 7x7 in x, y [-16, 16] Square
    yield from GridActionSpace(-16, 16, 7, -16, 16, 7)
    # for r in [100, 103, 106, 163, 166, 169], 60 equally distributed dots
    yield from RingActionSpace([100, 103, 106, 163, 166, 169], theta_step=6)
    # for r in [58, 135], 60 equally distributed dots
    yield from RingActionSpace([58, 135], theta_step=6)

    # Totally 529 states


def simple_action_space():
    yield from DotActionSpace(0, 0)
    yield from RingActionSpace([103, 134.5, 166], theta_step=18)
    # Totally 62 states
