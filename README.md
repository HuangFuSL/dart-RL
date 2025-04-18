# dart-RL

Code for solving the dart problem using dynamic programming methods.

## Rules

![Dart Board](imgs/dart-board.png)

- The dart board is a circle divided into 20 sections. Each section corresponds to a score in 1-20.
    - The sections are numbered in a specific order 6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10, starting from x = 0 and going clockwise.
    - Each section contains a small triple section and a double section. Hitting these sections will lead to a bonus score.
    - The center of the board contains two sections, the bullseye and the outer bull. Hitting the outer bull will lead to a score of 25, while the bullseye is the double section of the outer bull.
- The player starts with a score of 501 and has to reach a score of 0.
- In each round:
    - The player can throw at most 3 darts in a turn.
    - After 3 darts, the scores are summed and reduced from the initial score.
- Busting and winning
    - If a dart results in a score of 1 or negative score, the turn is invalidated.
    - If a dart results in a score of 0:
        - If the dart is a double, the player wins.
        - Otherwise, the turn is invalidated.
- Assume the landing position of darts follows normal distribution with specified Sigma.

## Launch

Use the following command to run the code:

```bash
python -m dart [args]
```

The following arguments are available:

- `start_score`: The initial score of the player. Default is 501.
- `action_space`: The pattern of simplified action spaces (see below).
- `turns`: The number of darts in one turn. Default is 3.
- Sigma: The covariance matrix of the landing position. Can be specified in the following formats:
    - `Sigma11, Sigma12, Sigma22`: Matrix values
    - `SigmaR1, SigmaR2, SigmaTheta`: Eigenvalues and eigenvectors
    - Sigma should be positive definite.
- `device`: The device to run the code on, can be `cpu`, `cuda` and `mps`. By default, it will use the device that is available.
- `method`: The iteration method, can be `value` for value iteration and `policy` for policy iteration.
- `eps`: The convergence threshold for the value iteration method.
- `discount`: The discount factor.
- `replay`: Whether to simulate a round of game.
- `log_level`, `log_file` and `log_format`: The logging level, file and format.

All the arguments can be specified in the command line, or in the config file specified by the `--config` toggle.

### Action spaces

There are 3 levels of action spaces:

|Level|Positions|#Actions|
|:-:|:-:|:-:|
|`simple`|![Simple](imgs/dart-simple.png)|62|
|`middle`|![Middle](imgs/dart-middle.png)|529|
|`complex`|![Complex](imgs/dart-complex.png)|921|

One can customize action spaces in the config file:

```yaml
action_space:
  - type: 'grid'
    # x_ngrid x y_ngrid targets in a grid
    x_start: -16
    x_end: 16
    x_ngrid: 7
    y_start: -16
    y_end: 16
    y_ngrid: 7
  - type: 'ring'
    # (theta_end - theta_start) / theta_step targets on a ring
    r: [100, 103, 106, 163, 166, 169, 179]
    theta_step: 6
  - type: 'ring'
    r: [58, 135]
    theta_step: 6
  - type: 'dot'
    # A specific point
    x: 170
    y: 170
```

## Known Issues

- The iteration loop uses full matrix to utilize matrix operations.
    - The state - action - outcome matrix will consume a lot of memory (~8GB when the action space is `complex`, starting score is 501 and turns is 3).
    - When using MPS backend on macOS and the action is large, issues may arise. Check out the [GitHub Issue](https://github.com/pytorch/pytorch/issues/124335) for more details.
