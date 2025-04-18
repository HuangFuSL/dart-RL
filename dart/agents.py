import abc
from typing import Generic

import torch

from .dart_logic import BaseDartEnvironment, Termination
from .display import DartDisplay
from .types import StateType


class BaseDartAgent(abc.ABC, Generic[StateType]):
    def __init__(
        self, env: BaseDartEnvironment[StateType | Termination],
        eps: float = 1e-6, discount: float = 0.99
    ):
        '''
        Initialize the Base Agent.

        Parameters
        - env: DartRegionEnvironment
            The environment for the agent.
        '''
        self.env = env
        self.eps = eps

        # Every agent has a policy
        self.policy = torch.zeros((env.num_states,), dtype=torch.long, device=env.device)
        # ... and a value function
        self.state_values = torch.full((env.num_states,), 100, dtype=torch.float16, device=env.device)
        self.state_values[env.get_state(Termination.WIN)] = 0.0 # Winning state
        self.discount = discount

    @abc.abstractmethod
    def step(self) -> float:
        '''
        Perform a single step of the agent and update the policy.

        Returns:
        - delta: float
            The maximum change in state values.
        '''
        pass

    def replay(self, display: DartDisplay):
        '''
        Replay the agent's policy in the environment.

        Parameters:
        - display: DartDisplay
            The display object to visualize the agent's actions.
        '''
        self.env.replay(self.policy, display)

    def __iter__(self):
        return self

    def __next__(self):
        delta = self.step()
        if delta < self.eps:
            raise StopIteration
        return delta


class ValueIterationAgent(BaseDartAgent[StateType]):
    def __init__(
        self, env: BaseDartEnvironment[StateType | Termination],
        eps: float = 1e-6, discount: float = 0.99
    ):
        '''
        Initialize the Value Iteration Agent.

        Parameters
        - env: DartRegionEnvironment
            The environment for the agent.
        '''
        super().__init__(env, eps, discount)


    def step(self):
        '''
        Perform a single step of value iteration.

        Returns:
        - delta: float
            The maximum change in state values.
        '''
        delta = 0
        step_cost = self.env.action_cost_np # (num_states, num_actions)
        state_values = self.state_values # (num_states)
        action_to_outcome = self.env.action_to_outcome_np # (num_actions, num_outcomes)
        outcome_to_state = self.env.outcome_to_state_np # (num_states, num_outcomes)

        # Bellman equation is given by
        # V(s) = min_a [C(s, a) + gamma * sum_{o} P(o|s,a) * V(s')]

        new_state_values = state_values.clone()  # (num_states, num_actions)
        new_state_values[self.env.get_state(Termination.WIN)] = 0.0

        next_state_values = self.discount * torch.einsum(
            'jk,ik->ij',
            action_to_outcome, # (num_actions, num_outcomes)
            new_state_values[outcome_to_state] # (num_states, num_outcomes)
        )
        next_state_values += step_cost  # (num_states, num_actions)

        self.policy = torch.argmin(next_state_values, dim=1)  # (num_states)
        new_state_values = next_state_values.min(dim=1).values

        delta = (torch.abs(self.state_values - new_state_values)).sum()
        self.state_values = new_state_values
        return delta

class PolicyIterationAgent(BaseDartAgent[StateType]):
    def __init__(
        self, env: BaseDartEnvironment[StateType | Termination],
        eps: float = 1e-6, discount: float = 0.99
    ):
        '''
        Initialize the Policy Iteration Agent.

        Parameters
        - env: DartRegionEnvironment
            The environment for the agent.
        '''
        super().__init__(env, eps, discount)

    def step(self):
        '''
        Perform a single step of policy iteration.

        Returns:
        - delta: float
            The maximum change in state values.
        '''
        delta = 0
        step_cost = self.env.action_cost_np # (num_states, num_actions)
        state_values = self.state_values # (num_states)
        action_to_outcome = self.env.action_to_outcome_np # (num_actions, num_outcomes)
        outcome_to_state = self.env.outcome_to_state_np # (num_states, num_outcomes)

        # Policy evaluation
        new_state_values = state_values.clone()
        new_state_values[self.env.get_state(Termination.WIN)] = 0.0

        state_outcomes = action_to_outcome[self.policy] # (num_states, num_outcomes)
        next_state_values = self.discount * torch.einsum(
            'ij,ij->i',
            state_outcomes, # (num_states, num_outcomes)
            new_state_values[outcome_to_state] # (num_states, num_outcomes)
        )
        self.state_values = step_cost[torch.arange(self.env.num_states), self.policy] + next_state_values

        # Policy improvement
        next_state_values = self.discount * torch.einsum(
            'jk,ik->ij',
            action_to_outcome, # (num_actions, num_outcomes)
            self.state_values[outcome_to_state]  # (num_states, num_outcomes)
        )
        next_state_values += step_cost  # (num_states, num_actions)

        new_policy = torch.argmin(next_state_values, dim=1)  # (num_states)

        # Check for convergence
        delta = (new_policy != self.policy).sum()
        self.policy = new_policy

        return delta
