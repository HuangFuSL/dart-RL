import abc
import functools
import logging
from typing import Callable, Generic, Iterable, List, Tuple, overload

import numpy as np
import torch
import tqdm

from .types import ActionType, OutcomeType, StateType


class BaseEnvironment(abc.ABC, Generic[StateType, ActionType, OutcomeType]):
    '''
    Environment class for the dart game.
    '''

    def __init__(self, device: str | torch.device = 'cpu'):
        states = [*enumerate(set(self.all_states()))]
        self.states = {
            idx: state for idx, state in states
        } | {
            state: idx for idx, state in states
        }
        logging.info(f'{len(states)} states.')

        actions = [*enumerate(set(self.all_actions()))]
        self.actions = {
            idx: action for idx, action in actions
        } | {
            action: idx for idx, action in actions
        }
        logging.info(f'{len(actions)} actions.')

        outcomes = [*enumerate(set(self.all_outcomes()))]
        self.outcomes = {
            idx: outcome for idx, outcome in outcomes
        } | {
            outcome: idx for idx, outcome in outcomes
        }
        logging.info(f'{len(outcomes)} outcomes.')

        self.device = device

    @property
    @abc.abstractmethod
    def starting_state(self) -> StateType:
        '''
        The starting state of the environment.
        '''
        pass

    @property
    def num_states(self) -> int:
        return len(self.states) // 2

    @property
    def num_actions(self) -> int:
        return len(self.actions) // 2

    @property
    def num_outcomes(self) -> int:
        return len(self.outcomes) // 2

    @overload
    def get_state(self, state: int) -> StateType:
        pass

    @overload
    def get_state(self, state: StateType) -> int:
        pass

    def get_state(self, state: StateType | int) -> StateType | int:
        return self.states[state]

    @overload
    def get_action(self, action: int) -> ActionType:
        pass

    @overload
    def get_action(self, action: ActionType) -> int:
        pass

    def get_action(self, action: ActionType | int) -> ActionType | int:
        return self.actions[action]

    @overload
    def get_outcome(self, outcome: int) -> OutcomeType:
        pass

    @overload
    def get_outcome(self, outcome: OutcomeType) -> int:
        pass

    def get_outcome(self, outcome: OutcomeType | int) -> OutcomeType | int:
        return self.outcomes[outcome]

    @functools.lru_cache()
    def get_state_mask(
        self, func: Callable[[StateType], bool],
    ):
        '''
        Return the mask of the states with starting score s_score
        '''
        states = [
            state for state in self.states
            if not isinstance(state, int) and func(state)
        ]
        state_idx = np.array([
            self.get_state(state) for state in states
        ])
        mask = np.isin(np.arange(self.num_states), state_idx, assume_unique=True)
        return torch.from_numpy(mask).to(self.device)

    @abc.abstractmethod
    def all_states(self) -> Iterable[StateType]:
        '''
        Generate all possible states for the environment.
        '''
        pass

    @abc.abstractmethod
    def all_actions(self) -> Iterable[ActionType]:
        '''
        Generate all possible actions for the environment.
        '''
        pass

    @abc.abstractmethod
    def all_outcomes(self) -> Iterable[OutcomeType]:
        '''
        Generate all possible outcomes for the environment.
        '''
        pass

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
        result = np.zeros(
            (self.num_states, self.num_actions, self.num_outcomes),
            dtype=np.float16
        )

        for j in tqdm.trange(self.num_actions):
            action = self.get_action(j)
            for i in range(self.num_states):
                state = self.get_state(i)
                for outcome, prob in self.action_to_outcome(state, action):
                    outcome_idx = self.get_outcome(outcome)
                    result[i, j, outcome_idx] += prob
        logging.info(f'Action to outcome matrix: {result.nbytes / (2 ** 20):.2f} MB')
        return torch.from_numpy(result).to(self.device)

    @abc.abstractmethod
    def action_to_outcome(
        self, state: StateType, action: ActionType
    ) -> List[Tuple[OutcomeType, float]]:
        '''
        Return the distribution of outcomes of taking an action.
        '''
        pass

    @property
    @functools.lru_cache()
    def action_cost_np(self) -> torch.Tensor:
        '''
        Build a cost matrix for all actions under all states.

        Returns:
        - np.ndarray
            A 2D array with shape (num_states, num_actions).
        '''
        logging.info('Building action cost matrix')
        result = np.full(
            (self.num_states, self.num_actions), -1,
            dtype=np.int8
        )
        for i in range(self.num_states):
            state = self.get_state(i)
            for j in range(self.num_actions):
                action = self.get_action(j)
                cost = self.action_cost(state, action)
                result[i, j] = cost
        logging.info(f'Action cost matrix: {result.nbytes / (2 ** 20):.2f} MB')
        return torch.from_numpy(result).to(self.device)

    @abc.abstractmethod
    def action_cost(
        self, state: StateType, action: ActionType
    ) -> float:
        '''
        Calculate the cost of taking an action in the current state.
        '''
        pass

    @property
    @functools.lru_cache()
    def outcome_to_state_np(self) -> torch.Tensor:
        '''
        Get the index of next state based on the outcome.

        Returns:
        - np.ndarray
            A 2D array with shape (num_states, num_outcomes).
        '''
        logging.info('Building outcome to state matrix')
        result = np.zeros(
            (self.num_states, self.num_outcomes),
            dtype=np.int32
        )
        for i in range(self.num_states):
            state = self.get_state(i)
            for j in range(self.num_outcomes):
                outcome = self.get_outcome(j)
                next_state = self.outcome_to_state(state, outcome)
                next_state_idx = self.get_state(next_state)
                result[i, j] = next_state_idx
        logging.info(f'Outcome to state matrix: {result.nbytes / (2 ** 20):.2f} MB')
        return torch.from_numpy(result).to(self.device)

    @abc.abstractmethod
    def outcome_to_state(
        self, state: StateType, outcome: OutcomeType
    ) -> StateType:
        '''
        Transform the current state based on the score obtained from a throw.
        '''
        pass
