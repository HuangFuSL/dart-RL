import argparse
import dataclasses
import logging
import time
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from .agents import PolicyIterationAgent, ValueIterationAgent
from .dart_logic import NoTurnDartEnvironment, TurnDartEnvironment
from .action_space import complex_action_space, middle_action_space, simple_action_space
from .action_space import RingActionSpace, GridActionSpace, DotActionSpace
from .display import DartDisplay


@dataclasses.dataclass
class Config():
    # Environment
    start_score: int = 501
    action_space: str | List[Dict[str, Any]] = 'simple'
    turns: int = 1 # 1 means no turn

    # Sigma - 1
    Sigma11: float | None = None
    Sigma22: float | None = None
    Sigma12: float | None = None

    # Sigma - 2
    SigmaR1: float | None = None
    SigmaR2: float | None = None
    SigmaTheta: float | None = None

    device: str | None = None # use mps or cuda to accelerate

    # Agent
    method: str = 'value' # value or policy iteration
    eps: float = 1e-6
    discount: float = 0.99

    # Display
    replay: bool = True

    # Logging
    log_level: str = 'INFO'
    log_file: str = ''
    log_format: str = '%(asctime)s - %(levelname)s - %(message)s'

    # Config file
    config: str = 'config.yaml'


    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser(description='Dart Game')
        parser.add_argument('--start_score', type=int, default=501, help='Starting score')
        parser.add_argument('--action_space', type=str, default='simple', help='Action space type')
        parser.add_argument('--turns', type=int, default=1, help='Number of turns')
        parser.add_argument('--Sigma11', type=float, default=None, help='Sigma 11')
        parser.add_argument('--Sigma22', type=float, default=None, help='Sigma 22')
        parser.add_argument('--Sigma12', type=float, default=None, help='Sigma 12')
        parser.add_argument('--SigmaR1', type=float, default=None, help='Sigma R1')
        parser.add_argument('--SigmaR2', type=float, default=None, help='Sigma R2')
        parser.add_argument('--SigmaTheta', type=float, default=None, help='Sigma Theta')
        parser.add_argument('--device', type=str, default=None, help='Device to use (cpu, cuda, mps)')
        parser.add_argument('--method', type=str, default='value', help='Method to use (value, policy)')
        parser.add_argument('--eps', type=float, default=1e-6, help='Epsilon for convergence')
        parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
        parser.add_argument('--replay', action='store_true', help='Replay the agent\'s policy')
        parser.add_argument('--log_level', type=str, default='INFO', help='Log level (CRITICAL, ERROR, WARNING, INFO, DEBUG)')
        parser.add_argument('--log_file', type=str, default='', help='Log file')
        parser.add_argument('--log_format', type=str, default='%(asctime)s - %(levelname)s - %(message)s', help='Log format')
        parser.add_argument('--config', type=str, default='', help='Config file')
        args = parser.parse_args()

        if not args.config:
            result = cls(**vars(args))
        else:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            result = cls(**config)
        if result.device is None:
            if torch.cuda.is_available():
                result.device = 'cuda'
            else:
                try:
                    if torch.backends.mps.is_available():
                        result.device = 'mps'
                    else:
                        result.device = 'cpu'
                except AttributeError:
                    result.device = 'cpu'
        result.check_args()
        return result

    def all_actions(self):
        if config.action_space == 'simple':
            yield from simple_action_space()
        elif config.action_space == 'middle':
            yield from middle_action_space()
        elif config.action_space == 'complex':
            yield from complex_action_space()
        else:
            assert isinstance(self.action_space, list)
            for action in self.action_space:
                type_ = action['type']
                del action['type']
                if type_ == 'ring':
                    yield from RingActionSpace(**action)
                elif type_ == 'grid':
                    yield from GridActionSpace(**action)
                elif type_ == 'dot':
                    yield from DotActionSpace(**action)
                else:
                    raise ValueError(f'Unknown action space type: {type_}')

    @property
    def Sigma(self):
        if all(x is not None for x in [self.Sigma11, self.Sigma22, self.Sigma12]):
            return np.array([
                [self.Sigma11, self.Sigma12],
                [self.Sigma12, self.Sigma22]
            ])
        elif all(x is not None for x in [self.SigmaR1, self.SigmaR2, self.SigmaTheta]):
            assert self.SigmaTheta is not None
            theta = self.SigmaTheta / 360 * 2 * np.pi
            P = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            Lambda = np.array([
                [self.SigmaR1, 0],
                [0, self.SigmaR2]
            ])
            return P @ Lambda @ P.T
        raise ValueError('Invalid Sigma parameters.')

    def check_args(self):
        # 1. starting score should > 1
        if self.start_score <= 1:
            raise ValueError('Starting score should be greater than 1')
        # 2. turns should be > 0
        if self.turns <= 0:
            raise ValueError('#Turns in one round should be greater than 0')
        # 3. Sigma should be positive definite
        if not np.all(np.linalg.eigvals(self.Sigma) > 0):
            raise ValueError('Sigma should be positive definite')
        # 4. device should be cpu, cuda or mps
        if self.device not in ['cpu', 'cuda', 'mps']:
            raise ValueError('Device should be cpu, cuda or mps')
        if not torch.cuda.is_available() and self.device == 'cuda':
            raise ValueError('CUDA is not available')
        try:
            if not torch.backends.mps.is_available() and self.device == 'mps':
                raise ValueError('MPS is not available')
        except AttributeError:
            if self.device == 'mps':
                raise ValueError('MPS is not available')
        # 5. method should be value or policy
        if self.method not in ['value', 'policy']:
            raise ValueError('Method should be value or policy')
        # 6. eps should be > 0
        if self.eps <= 0:
            raise ValueError('Epsilon should be greater than 0')
        # 7. discount should be in [0, 1]
        if self.discount < 0 or self.discount > 1:
            raise ValueError('Discount should be in [0, 1]')
        # 8. log_level should be one of the logging levels
        log_levels = [
            'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'
        ]
        if self.log_level not in log_levels:
            raise ValueError(f'Log level should be one of {log_levels}')
        # 9. Action space should be 'simple', 'middle' or 'complex'
        if isinstance(self.action_space, str) and \
            self.action_space not in ['simple', 'middle', 'complex']:
            raise ValueError('Action space should be simple, middle or complex')


def main(config: Config):
    env_kwargs = {
        'start_score': config.start_score,
        'Sigma': config.Sigma,
        'device': config.device
    }
    env_kwargs['action_space'] = config.all_actions()

    if config.turns > 1:
        env = TurnDartEnvironment(num_throws=config.turns, **env_kwargs)
    else:
        env = NoTurnDartEnvironment(**env_kwargs)
    agent_kwargs = {
        'env': env,
        'eps': config.eps,
        'discount': config.discount
    }
    if config.method == 'value':
        agent = ValueIterationAgent(**agent_kwargs)
    else:
        agent = PolicyIterationAgent(**agent_kwargs)

    logging.info('Starting iteration...')
    start_time = time.time()
    for i, delta in enumerate(agent, 1):
        logging.info(f"Iteration {i}: delta = {delta}")
    end_time = time.time()
    logging.info(f'{i} iterations in {end_time - start_time:.2f} seconds.')
    logging.info(f'Average {(end_time - start_time) / i:.2f} seconds per iteration.')
    logging.info('Iteration converged.')

    print(agent.state_values[env.get_state(env.starting_state)]) # type: ignore

    if config.replay:
        display = DartDisplay()
        agent.replay(display)
        display.ready()

if __name__ == '__main__':
    config = Config.parse_args()
    if not config.log_file:
        logging.basicConfig(
            level=config.log_level,
            format=config.log_format,
        )
    else:
        logging.basicConfig(
            level=config.log_level,
            filename=config.log_file,
            format=config.log_format,
        )
    main(config)
