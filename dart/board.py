import dataclasses
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

SECTOR_SCORES = [
    6, 13, 4, 18, 1, 20, 5, 12, 9, 14,
    11, 8, 16, 7, 19, 3, 17, 2, 15, 10
]
SECTOR_SCORES_NP = np.array(SECTOR_SCORES)
SECTOR_RATIO = np.array([2, 1, 1, 3, 1, 2, 0])
SECTOR_RADIUS = [
    6.35, 15.9, 99, 107, 162, 170, 680
]
ALL_SECTORS = [
    (0, 0), # Outside
    *[(_ + 1, 25) for _ in range(2)], # Bull
    *[(1, _ + 1) for _ in range(20)], # Single
    *[(2, _ + 1) for _ in range(20)], # Double
    *[(3, _ + 1) for _ in range(20)], # Triple
]

@dataclasses.dataclass
class DartResult():
    '''
    A dataclass to store the result of a dart throw.
    '''
    score: int
    ratio: int
    prob: float

    def __abs__(self):
        return self.score * self.ratio

    def __repr__(self):
        return f'Score: {self.score}, ratio: {self.ratio}, prob: {self.prob * 100:.2f}%'

def sample_hit(
    mu: np.ndarray, Sigma: np.ndarray | None
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Generate a sample hit from a 2D normal distribution.

    Parameters
    - mu: np.ndarray
        The mean of the normal distribution.
    - Sigma: np.ndarray
        The covariance matrix of the normal distribution.

    Returns
    - r: float
        The radial coordinate of the hit point.
    - theta: float
        The angular coordinate of the hit point.
    '''
    if Sigma is None:
        x, y = mu
    else:
        mvn = multivariate_normal(mean=mu, cov=Sigma) # type: ignore
        x, y = mvn.rvs()
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x) % (2 * np.pi)
    return r, theta

def get_region(
    r: np.ndarray, theta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Compute the dartboard region and score for given polar coordinates.

    Parameters
    - r: np.ndarray
        The radial coordinates of the dartboard.
    - theta: np.ndarray
        The angular coordinates of the dartboard.

    Returns
    - ring_result: np.ndarray
        The ring names corresponding to the radial coordinates.
    - score_result: np.ndarray
        The scores corresponding to the radial coordinates.
    '''
    # r: (n_r, 1), theta: (1, n_theta)

    # Broadcast both arrays to the same shape
    target_shape = (r.shape[0], theta.shape[1])
    r = np.broadcast_to(r, target_shape)
    theta = np.broadcast_to(theta, target_shape)

    theta = (theta + np.deg2rad(9)) % (2 * np.pi)
    sector_index = ((theta / (2 * np.pi)) * 20).astype(int)
    scores = SECTOR_SCORES_NP[sector_index]

    r_conditions = [r <= x for x in SECTOR_RADIUS]
    score_values = [*([25] * 2), *([scores] * 4), 0]

    ring_result = np.select(r_conditions, SECTOR_RATIO.tolist(), default=0)
    score_result = np.select(r_conditions, score_values, default=0)

    return ring_result, score_result

def get_prob(
    mu: np.ndarray, Sigma: np.ndarray | None,
    n_r=300, n_theta=360, num_std=4.0, eps=1e-6
):
    """
    Compute the probabilities of hitting each region of the dartboard
    given a 2D normal distribution N(mu, sigma).

    Parameters
    - mu: np.ndarray
        The mean of the normal distribution.
    - sigma: np.ndarray
        The covariance matrix of the normal distribution.
    - n_r: int
        The number of radial divisions.
    - n_theta: int
        The number of angular divisions.
    - num_std: float
        The number of standard deviations to extend the integration range.

    Returns
    - region_probs: list of dict
        A list of dictionaries containing the scores and their corresponding
        probabilities.
    """
    if Sigma is None:
        r = np.linalg.norm(mu)
        theta = np.arctan2(mu[1], mu[0]) % (2 * np.pi)
        ring_result, score_result = get_region(
            r.reshape(1, -1), theta.reshape(1, -1)  # type: ignore
        )
        return [DartResult(
            score=score_result.item(),
            ratio=ring_result.item(),
            prob=1.0
        )]

    eigvals = np.linalg.eigvals(Sigma)
    largest_std = np.sqrt(np.max(eigvals))
    r_max = 170 + num_std * largest_std


    rs = np.linspace(0, r_max, n_r + 1)
    thetas = np.linspace(0, 2 * np.pi, n_theta + 1)


    mvn = multivariate_normal(mean=mu, cov=Sigma) # type: ignore

    r_centers = 0.5 * (rs[:-1] + rs[1:])
    theta_centers = 0.5 * (thetas[:-1] + thetas[1:])
    drs = np.diff(rs)
    dthetas = np.diff(thetas)
    # PDF at the center of each cell
    x_centers = r_centers[:, np.newaxis] * np.cos(theta_centers)
    y_centers = r_centers[:, np.newaxis] * np.sin(theta_centers)
    pdf_vals = mvn.pdf(np.dstack([x_centers, y_centers]))
    # Cell probabilities
    cell_probs = pdf_vals.reshape(n_r, n_theta) * r_centers[:, np.newaxis] * drs[:, np.newaxis] * dthetas[np.newaxis, :]
    total_prob = np.sum(cell_probs)

    if total_prob:
        cell_probs /= np.sum(cell_probs)  # Normalize to sum to 1
        # Calculate ring and sector for each cell
        ratios, scores = get_region(
            r_centers[:, np.newaxis], theta_centers[np.newaxis, :])
        # Group by scores
        prob_df = pd.DataFrame({
            'score': scores.ravel(),
            'ratio': ratios.ravel(),
            'prob': cell_probs.ravel()
        })

        prob_df = prob_df.groupby(['score', 'ratio'], as_index=False).agg({
            'prob': 'sum'
        })
        prob_df = prob_df[prob_df['prob'] > eps]
        prob_df['prob'] = prob_df['prob'] / np.sum(prob_df['prob'])
        return [
            DartResult(**x) # type: ignore
            for x in prob_df.to_dict(orient='records')
        ]
    else:
        # Directly use mu to calculate when probability is 0
        # Happens when Sigma is too small

        r = np.linalg.norm(mu)
        theta = np.arctan2(mu[1], mu[0]) % (2 * np.pi)
        ring_result, score_result = get_region(
            r.reshape(1, -1), theta.reshape(1, -1) # type: ignore
        )
        return [DartResult(
            score=score_result.item(),
            ratio=ring_result.item(),
            prob=1.0
        )]
