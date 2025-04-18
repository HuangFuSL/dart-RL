import abc
import dataclasses
import enum
import functools
import logging
import math
from typing import Callable, Generator, Generic, Iterable, List, Tuple

import numpy as np
import torch
import tqdm

from .board import ALL_SECTORS, SECTOR_RADIUS, SECTOR_SCORES
from .board import get_prob, get_region, sample_hit
from .display import DartDisplay
from .env import BaseEnvironment
from .types import StateType


@dataclasses.dataclass(frozen=True)
class TurnState():
    '''
    State representation for the dart game with rounds.
    '''
    s_score: int  # Starting score of the current round
    c_score: int  # Current score of the player
    throws: int  # Number of throws used


@dataclasses.dataclass(frozen=True)
class NoTurnState():
    '''
    State representation for the dart game without rounds.
    '''
    s_score: int  # Starting score of the current round


class Termination(enum.Enum):
    '''
    Enum for termination states.
    '''
    WIN = 0  # Winning state


@dataclasses.dataclass(frozen=True)
class Action():
    '''
    Action representation for the dart game.
    '''
    x: float  # X coordinate of the action
    y: float  # Y coordinate of the action

    @property
    def polar(self):
        '''
        Return the action as polar coordinates (r, theta).
        '''
        r = np.sqrt(self.x ** 2 + self.y ** 2).reshape(1, 1)
        theta = (np.arctan2(self.y, self.x) % (2 * np.pi)).reshape(1, 1)
        return r, theta

    @property
    def np(self):
        '''
        Return the action as target coordinates.
        '''
        return np.array([self.x, self.y])

    @property
    def region(self):
        '''
        Return the action as target region.
        '''
        r, theta = self.polar
        ratio, score = get_region(r, theta)
        return ratio.item(), score.item()

    @classmethod
    def polar_to_action(cls, r: float, theta: float):
        return Action(r * math.cos(theta), r * math.sin(theta))

    @classmethod
    def region_to_action(cls, ratio: int, score: int) -> 'Action':
        if score == 25:
            # BULL
            return Action(0, 0)
        if not score:
            target_idx = 0
        else:
            target_idx = SECTOR_SCORES.index(score)
        target_theta = np.deg2rad(target_idx * 18)
        if ratio == 1:
            target_r = sum(SECTOR_RADIUS[3:5]) / 2
        elif ratio == 2:
            target_r = sum(SECTOR_RADIUS[4:6]) / 2
        elif ratio == 3:
            target_r = sum(SECTOR_RADIUS[2:4]) / 2
        else:
            # Out of range
            target_r = sum(SECTOR_RADIUS[5:7]) / 2
        return Action(
            target_r * np.cos(target_theta).item(),
            target_r * np.sin(target_theta).item()
        )

@dataclasses.dataclass(frozen=True)
class Outcome():
    '''
    Outcome representation for the dart game.
    '''
    score: int # Total score obtained from the action
    is_double: bool  # Whether the score is a double


class BaseDartEnvironment(BaseEnvironment[StateType | Termination, Action, Outcome], abc.ABC, Generic[StateType]):
    '''
    Abstract base class for the dart game environment. With or without turns.

    Parameters
    - start_score: int = 501
        The starting score of the game
    - action_space: Iterable[Action] | None = None
        An iterable that yields all possible actions.
    - Sigma: np.ndarray | None = None
        The covariance matrix of the dart throw.
    - device: str | torch.device = 'cpu'
        The device to run the environment on.
    '''
    def __init__(
        self, start_score: int = 501,
        action_space: Iterable[Action] | None = None,
        Sigma: np.ndarray | Callable[[Action], np.ndarray | None] | None = None,
        device: str | torch.device = 'cpu'
    ):
        self.start_score = start_score
        if not callable(Sigma):
            self.Sigma = lambda action: Sigma
        else:
            self.Sigma = Sigma
        if action_space is None:
            self.action_space = (
                Action.region_to_action(x, y) for x, y in ALL_SECTORS
            )
        else:
            self.action_space = action_space

        # Since parent class calls all_states() in __init__
        # We need to define all the variables before calling super().__init__()
        super().__init__(device)

    @functools.lru_cache(maxsize=1024)
    def action_probability(self, action: Action):
        if self.Sigma is None:
            score, ratio = action.region
            score = score * ratio
            is_double = ratio == 2
            return [
                (Outcome(score, is_double), 1.0)
            ]

        result = sorted(
            get_prob(action.np, self.Sigma(action)),
            key=lambda x: x.prob, reverse=True
        )
        return [
            (Outcome(x.score * x.ratio, x.ratio == 2), x.prob)
            for x in result
        ]

    def all_outcomes(self) -> Iterable[Outcome]:
        yield from [
            Outcome(x * y, x == 2) for x, y in ALL_SECTORS
        ]

    def action_to_outcome(
        self, state: StateType | Termination, action: Action
    ) -> List[Tuple[Outcome, float]]:
        '''
        Calculate the distribution of outcomes of taking an action.

        Parameters
        - action: RegionAction
            The action to be converted.

        Returns:
        - (score, is_double): tuple
            The score obtained from the action and whether it is a double.
        '''
        if isinstance(state, Termination):
            return [(Outcome(0, False), 1.0)]
        return self.action_probability(action)

    def all_actions(self) -> Iterable[Action]:
        yield from self.action_space

    @property
    @functools.lru_cache()
    def action_to_outcome_np(self) -> torch.Tensor:
        '''
        Build a distribution matrix for outcomes under all actions and states.

        Returns:
        - np.ndarray
            A 3D array with shape (num_states, num_actions, num_outcomes).
        '''
        logging.info('Building action to outcome matrix')

        # Issues with MPS computation backend

        # This fails with start_score = 501 and action_space = complex
        result = np.zeros(
            (self.num_states, self.num_actions, self.num_outcomes),
            dtype=np.float16
        )

        # Due to the known bug in torch MPS backed we need to directly allocate
        # the tensor on the device
        # Data corrupted when transferring over 4GiB to MPS
        # https://github.com/pytorch/pytorch/issues/124335
        # However, if start_score = 401 or 301, reports
        # /AppleInternal/Library/BuildRoots/01adf19d-fba1-11ef-a947-f2a857e00a32/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShaders/MPSCore/Types/MPSNDArray.mm:829: failed assertion `[MPSNDArray initWithDevice:descriptor:isTextureBacked:] Error: NDArray dimension length > INT_MAX'
        # Also, when start_score = 501, the value iteration result is not correct
        #
        # result = torch.zeros(
        #     (self.num_states, self.num_actions, self.num_outcomes),
        #     dtype=torch.float16, device=self.device
        # )

        for j in tqdm.trange(self.num_actions):
            # Since for all non-termination states
            # the outcome only depends on action, we can accelerate it
            action = self.get_action(j)

            termination = self.action_to_outcome(Termination.WIN, action)
            for i in range(self.num_states):
                state = self.get_state(i)
                if isinstance(state, Termination):
                    continue
                non_termination = self.action_to_outcome(state, action)
                break

            for outcome, prob in non_termination:
                outcome_idx = self.get_outcome(outcome)
                result[np.arange(self.num_states), j, outcome_idx] += prob
                # result[torch.arange(self.num_states), j, outcome_idx] += prob
            for state in Termination.__members__.values():
                state_id = self.get_state(state)
                for outcome, prob in termination:
                    outcome_idx = self.get_outcome(outcome)
                    result[state_id, j, outcome_idx] += prob

        logging.info(f'Action to outcome matrix: {result.nbytes / (2 ** 20):.2f} MB')
        return torch.from_numpy(result).to(self.device)
        # return result

    def replay_show(
        self, display: DartDisplay, score: int, action: Action | int,
        r: np.ndarray, theta: np.ndarray
    ):
        '''
        Replay one step of the decision process
        '''
        # Show score
        display.refresh_window()
        display.plot_score(score)
        yield

        # Show action
        if isinstance(action, int):
            action = self.get_action(action)
        mu = action.np
        display.aim(mu, self.Sigma(action))
        yield

        # Show outcome
        display.hit(r, theta)
        yield

    def replay(self, policy: torch.Tensor, display: DartDisplay):
        coro = self.replay_step(policy, display)
        next(coro)
        def hook(event):
            try:
                next(coro)
            except StopIteration:
                display.fig.canvas.stop_event_loop()
        self.hook = hook
        display.add_click_hook(hook)

    # Functions not yet implemented

    @property
    @abc.abstractmethod
    def starting_state(self):
        ...

    @abc.abstractmethod
    def all_states(self):
        ...

    @abc.abstractmethod
    def action_cost(self, state, action) -> float:
        ...

    @abc.abstractmethod
    def outcome_to_state(self, state, outcome) -> StateType:
        ...

    @abc.abstractmethod
    def replay_step(self, policy: torch.Tensor, display: DartDisplay) -> Generator[None, None, None]:
        ...


class NoTurnDartEnvironment(BaseDartEnvironment[NoTurnState | Termination]):
    '''
    Class for the dart game environment without rounds.

    Parameters
    - start_score: int = 501
        The starting score of the game
    - action_space: Iterable[Action] | None = None
        An iterable that yields all possible actions.
    - Sigma: np.ndarray | None = None
        The covariance matrix of the dart throw.
    - device: str | torch.device = 'cpu'
        The device to run the environment on.
    '''

    def __init__(
        self, *, start_score: int = 501,
        action_space: Iterable[Action] | None = None,
        Sigma: np.ndarray | Callable[[Action], np.ndarray | None] | None = None,
        device: str | torch.device = 'cpu'
    ):
        super().__init__(start_score, action_space, Sigma, device)

    @property
    def starting_state(self):
        '''
        The starting state of the environment.
        '''
        return NoTurnState(self.start_score)

    def all_states(self):
        '''
        Generate all possible states for the dart game.

        Parameters
        - START_SCORE: int
            The starting score of the game.

        Yields:
        - (s_score): int
            The starting score.
        '''
        yield from Termination.__members__.values()
        # When not yet won, start_score > 0
        for start_score in range(2, self.start_score + 1):
            yield NoTurnState(start_score)


    def action_cost(self, state, action) -> float:
        '''
        Calculate the cost of taking an action in the current state.

        Parameters
        - state: NoTurnState | Termination
            The current state of the game.
        - action: RegionAction
            The action taken.
        - next_state: State
            The next state after taking the action.

        Returns:
        - float
            The cost of the action.
        '''
        if isinstance(state, Termination):
            return 0  # No cost in the termination state

        # Move to the winning state also take cost
        return 1

    def outcome_to_state(self, state, outcome):
        '''
        Transform the current state based on the score obtained from a throw.

        Termination.LOSE is a state that can never win.
        Termination.WIN is a state that can never lose.
        '''
        if isinstance(state, Termination):
            return state  # No state transition in the termination state

        score, is_double = outcome.score, outcome.is_double
        if state.s_score == score and is_double:  # Win
            return Termination.WIN

        if (
            (state.s_score < score) or
            (state.s_score == score and not is_double)
        ):
            return NoTurnState(state.s_score)

        if state.s_score > score:  # New round
            if state.s_score - score == 1:
                return NoTurnState(state.s_score)

            return NoTurnState(state.s_score - score)

        # Not reachable here
        assert False, f"Invalid state: {state}, outcome: {outcome}"

    def replay_step(self, policy: torch.Tensor, display: DartDisplay) -> Generator[None, None, None]:
        '''
        Replay the decision process.
        '''
        state = self.starting_state
        rounds = 0
        while not isinstance(state, Termination):
            rounds += 1

            action_id = int(policy[self.get_state(state)].item())
            action = self.get_action(action_id)
            mu = action.np
            r, theta = map(
                lambda x: np.array([[x]]),
                sample_hit(mu, self.Sigma(action))
            )
            yield from self.replay_show(
                display, state.s_score, action, r, theta
            )

            ratio, score = get_region(r, theta)
            outcome = Outcome((score * ratio).item(), ratio.item() == 2)

            next_state = self.outcome_to_state(state, outcome)
            state = next_state


class TurnDartEnvironment(BaseDartEnvironment[TurnState | Termination]):
    '''
    Class for the dart game environment with rounds.

    Parameters
    - start_score: int = 501
        The starting score of the game
    - num_throws: int = 3
        The number of throws left.
    - action_space: Iterable[Action] | None = None
        An iterable that yields all possible actions.
    - Sigma: np.ndarray | None = None
        The covariance matrix of the dart throw.
    - device: str | torch.device = 'cpu'
        The device to run the environment on.
    '''

    def __init__(
        self, start_score: int = 501, num_throws: int = 3,
        action_space: Iterable[Action] | None = None,
        Sigma: np.ndarray | Callable[[Action], np.ndarray | None] | None = None,
        device: str | torch.device = 'cpu'
    ):
        self.num_throws = num_throws
        # Since parent class calls all_states() in __init__
        # We need to define all the variables before calling super().__init__()
        super().__init__(start_score, action_space, Sigma, device)

    @property
    def starting_state(self) -> TurnState | Termination:
        '''
        The starting state of the environment.
        '''
        return TurnState(self.start_score, self.start_score, 0)

    def all_states(self) -> Iterable[TurnState | Termination]:
        '''
        Generate all possible states for the dart game.

        Parameters
        - START_SCORE: int
            The starting score of the game.
        - num_throws: int
            The number of throws left.

        Yields:
        - (s_score, c_score, throws): tuple
            The starting score, current score, and number of throws left.
        '''
        MAX_SCORE = 60  # A dart can score at most 60 points (Triple 20)
        yield from Termination.__members__.values()  # WIN, LOSE

        # When not yet won, start_score > 0
        for start_score in range(2, self.start_score + 1):
            for num in range(self.num_throws):
                yield from [
                    TurnState(start_score, max(start_score - score, 0), num)
                    for score in range(0, MAX_SCORE * num + 1)
                ]

    def action_to_outcome(
        self, state: TurnState | Termination, action: Action
    ) -> List[Tuple[Outcome, float]]:
        '''
        Calculate the distribution of outcomes of taking an action.

        Parameters
        - action: RegionAction
            The action to be converted.

        Returns:
        - (score, is_double): tuple
            The score obtained from the action and whether it is a double.
        '''
        if isinstance(state, Termination):
            return [(Outcome(0, False), 1.0)]
        return self.action_probability(action)

    def action_cost(self, state, action) -> float:
        '''
        Calculate the cost of taking an action in the current state.

        Parameters
        - state: State
            The current state of the game.
        - action: RegionAction
            The action taken.
        - next_state: State
            The next state after taking the action.

        Returns:
        - float
            The cost of the action.
        '''
        if isinstance(state, Termination):
            return 0  # No cost in the termination state
        # Move to the winning state also take cost
        return 1 if state.throws == 0 else 0

    def outcome_to_state(self, state, outcome):
        '''
        Transform the current state based on the score obtained from a throw.

        Termination.LOSE is a state that can never win.
        Termination.WIN is a state that can never lose.
        '''
        if isinstance(state, Termination):
            return state  # No state transition in the termination state

        score, is_double = outcome.score, outcome.is_double
        if state.c_score == score and is_double:  # Win
            return Termination.WIN

        if (
            (state.c_score < score) or
            (state.c_score == score and not is_double)
        ):
            return TurnState(state.s_score, state.s_score, 0)

        if state.c_score > score and state.throws < self.num_throws - 1:  # Within the round
            return TurnState(
                state.s_score, state.c_score - score, state.throws + 1
            )

        if state.c_score > score and state.throws == self.num_throws - 1:  # New round
            if state.c_score - score == 1:
                # Bust
                return TurnState(state.s_score, state.s_score, 0)

            return TurnState(
                state.c_score - score, state.c_score - score, 0
            )
        assert False, f"Invalid state: {state}, outcome: {outcome}"

    def replay_step(self, policy: torch.Tensor, display: DartDisplay) -> Generator[None, None, None]:
        '''
        Replay the decision process.
        '''
        state = self.starting_state
        rounds = 0
        while not isinstance(state, Termination):
            if state.throws == 0:
                rounds += 1
                logging.info(f'Round {rounds} starts.')

            action_id = int(policy[self.get_state(state)].item())
            action = self.get_action(action_id)
            mu = action.np
            r, theta = map(
                lambda x: np.array([[x]]),
                sample_hit(mu, self.Sigma(action))
            )
            yield from self.replay_show(
                display, state.c_score, action, r, theta
            )

            ratio, score = get_region(r, theta)
            outcome = Outcome((score * ratio).item(), ratio.item() == 2)

            next_state = self.outcome_to_state(state, outcome)
            state = next_state
