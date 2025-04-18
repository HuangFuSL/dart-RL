from typing import Callable, List

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse
from matplotlib.backend_bases import Event
from scipy.stats import chi2, multivariate_normal

from .board import SECTOR_RADIUS, SECTOR_SCORES, DartResult, get_prob

# matplotlib.use('WebAgg')

def draw_dartboard_with_scores(ax: Axes):
    '''
    Draw the main appearance of the dartboard on the given Axes,
    and label the scores of 20 sectors on the outer circle.
    '''
    # Remove box
    ax.set_frame_on(False)
    radii = SECTOR_RADIUS
    r_inner = SECTOR_RADIUS[1]
    r_outer = SECTOR_RADIUS[-2]

    for r in radii:
        circle = Circle(
            (0, 0), r, fill=False, color='k', alpha=0.7
        )
        ax.add_patch(circle)

    for i in range(20):
        boundary_angle = np.deg2rad(i * 18 - 9)
        x1 = r_inner * np.cos(boundary_angle)
        y1 = r_inner * np.sin(boundary_angle)
        x2 = r_outer * np.cos(boundary_angle)
        y2 = r_outer * np.sin(boundary_angle)
        ax.plot(
            [x1, x2], [y1, y2], color='k', alpha=0.4, linewidth=0.8
        )

    for i in range(20):
        center_angle = np.deg2rad(i * 18)
        r_text = 190
        x_t = r_text * np.cos(center_angle)
        y_t = r_text * np.sin(center_angle)
        score_text = str(SECTOR_SCORES[i])
        ax.text(
            x_t, y_t, score_text,
            ha='center', va='center', fontsize=15
        )

    ax.set_title('Dart Throw Distribution')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-210, 210)
    ax.set_ylim(-210, 210)


def plot_confidence_ellipse(
    ax: Axes,
    mu: np.ndarray, Sigma: np.ndarray, conf_level: float = 0.95,
    **kwargs
):
    '''
    Plot the conf_level% confidence ellipse of the given 2D normal distribution (mu, Sigma) on ax.

    Parameters
    - ax: Axes
        The Axes to plot on.
    - mu: np.ndarray
        The mean of the normal distribution.
    - Sigma: np.ndarray
        The covariance matrix of the normal distribution.
    - conf_level: float
        The confidence level for the ellipse.
    - kwargs: dict
        Additional keyword arguments for the Ellipse patch.
    '''
    chi2_val = chi2(2).ppf(conf_level)

    vals, vecs = np.linalg.eigh(Sigma)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    major_axis_length, minor_axis_length = np.sqrt(vals).tolist()
    major_line_start = mu
    major_line_end = mu + major_axis_length * vecs[:, 0]
    minor_line_start = mu
    minor_line_end = mu + minor_axis_length * vecs[:, 1]
    ax.plot(
        [major_line_start[0], major_line_end[0]],
        [major_line_start[1], major_line_end[1]],
        color='red', linewidth=1.5
    )
    ax.plot(
        [minor_line_start[0], minor_line_end[0]],
        [minor_line_start[1], minor_line_end[1]],
        color='blue', linewidth=1.5
    )

    width, height = 2 * np.sqrt(chi2_val * vals)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    ellipse = Ellipse(xy=mu, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)


def plot_hit_distribution(
    fig: Figure, ax_main: Axes, ax_cmap: Axes,
    mu: np.ndarray, Sigma: np.ndarray,
    x_range=(-230, 230), y_range=(-230, 230),
    step=2.0
):
    '''
    Show the probability density heatmap of the 2D normal distribution (mu, Sigma) on the XY plane.

    Parameters
    - fig: Figure
        The figure to plot on.
    - ax_main: Axes
        The main Axes to plot the heatmap on.
    - ax_cmap: Axes
        The Axes for the colorbar.
    - mu: np.ndarray
        The mean of the normal distribution.
    - Sigma: np.ndarray
        The covariance matrix of the normal distribution.
    '''
    x_vals = np.arange(x_range[0], x_range[1]+step, step)
    y_vals = np.arange(y_range[0], y_range[1]+step, step)
    X, Y = np.meshgrid(x_vals, y_vals)
    pos = np.dstack((X, Y))

    mvn = multivariate_normal(mean=mu, cov=Sigma)
    Z = mvn.pdf(pos)  # shape 同 X, Y

    # 用等高线填充图(contourf)展示热力
    cs = ax_main.contourf(X, Y, Z, levels=50, cmap='BuGn')
    # Colorbar
    fig.colorbar(cs, cax=ax_cmap, label='PDF')

    # 瞄准点
    draw_dartboard_with_scores(ax_main)
    ax_main.scatter(
        *mu, color='yellow', edgecolors='k', s=50, marker='o', label='Aim', zorder=99
    )
    # 置信区间
    plot_confidence_ellipse(
        ax_main, mu, Sigma,
        edgecolor='red', facecolor='none',
        label='95% Confidence Ellipse'
    )


def set_prob_ax(ax: Axes):
    ax.set_xlabel('Probability')
    ax.set_ylabel('Score')
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])

def plot_score_distribution(ax, score_probs: List[DartResult]):
    """
    绘制飞镖得分分布的堆积柱状图。
    """
    cmap = plt.get_cmap('BuGn')

    left = 0.0
    for x in sorted(score_probs, key=lambda x: x.prob, reverse=True):
        score, prob = abs(x), x.prob
        if prob < 0.001:
            continue
        ax.barh([0], [prob], height=1, left=left,
                color=cmap(score / 20), edgecolor='k')
        kwargs = {
            's': f'{score}\n{prob * 100:.1f}%' if prob > 0.04 else f'{score}',
            'ha': 'center', 'va': 'center',
            'fontsize': 8 if prob > 0.02 else 5,
            'color': 'white' if score > 15 else 'black'
        }
        ax.text(left + prob / 2, 0, **kwargs)
        left += prob
    set_prob_ax(ax)


def register_cursor(
    fig: Figure, ax_main: Axes, ax_cmap: Axes, ax_dist: Axes,
    Sigma: np.ndarray
):

    def on_click(event):
        '''
        To be called when the mouse is clicked on the figure.
        '''
        if event.inaxes is not ax_main:
            return
        if event.xdata is None or event.ydata is None:
            return
        mu = np.array([event.xdata, event.ydata])

        # Clear the axes
        ax_main.cla()
        ax_dist.cla()
        # Redraw the dartboard and the score distribution
        plot_hit_distribution(fig, ax_main, ax_cmap, mu, Sigma)
        score_probs = get_prob(mu, Sigma)
        plot_score_distribution(ax_dist, score_probs)

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id) # type: ignore
    fig.canvas.manager.key_press_handler_id = None # type: ignore

class DartDisplay():
    def __init__(self, figsize=(8, 8), height_ratios=(10, 1), width_ratios=(10, 1)):
        fig, ((ax_main, ax_cmap), (ax_prob, ax_score)) = plt.subplots(
            2, 2, figsize=figsize,
            height_ratios=height_ratios, width_ratios=width_ratios
        )
        self.fig = fig
        self.ax_main = ax_main
        self.ax_cmap = ax_cmap
        self.ax_prob = ax_prob
        self.ax_score = ax_score
        self.hooked = False

        self.refresh_window()

    def refresh_window(self):
        self.ax_main.cla()
        self.ax_cmap.cla()
        self.ax_prob.cla()
        draw_dartboard_with_scores(self.ax_main)
        set_prob_ax(self.ax_prob)

    def plot_score(self, score: int):
        """
        Show the score on the score Axes.
        """
        self.ax_score.cla()
        self.ax_score.set_xlim(0, 1)
        self.ax_score.set_ylim(0, 1)

        self.ax_score.set_xticks([])
        self.ax_score.set_yticks([])
        self.ax_score.set_frame_on(False)
        self.ax_score.text(
            0.5, 0.5, f'Score: {score}',
            ha='center', va='center', fontsize=15
        )
        self.fig.canvas.draw()

    def aim(
        self, mu: np.ndarray, Sigma: np.ndarray | None,
        x_range=(-230, 230), y_range=(-230, 230),
        step=2.0
    ):
        if Sigma is None:
            Sigma = np.eye(2) * 0.1
        plot_hit_distribution(
            self.fig, self.ax_main, self.ax_cmap,
            mu, Sigma, x_range, y_range, step
        )
        plot_score_distribution(self.ax_prob, get_prob(mu, Sigma))
        self.fig.canvas.draw()

    def hit(self, r, theta):
        mu = np.array([
            (r * np.cos(theta)).item(), (r * np.sin(theta)).item()
        ])
        self.ax_main.scatter(
            *mu, color='red', s=50, marker='x', zorder=99
        )
        self.fig.canvas.draw()

    def add_click_hook(self, hook: Callable[[Event], None]):
        self.hooked = True
        self.fig.canvas.mpl_connect('button_press_event', hook)

    def ready(self):
        self.fig.show()
        if self.hooked:
            plt.show(block=False)
            self.fig.canvas.start_event_loop(0) # Stop is controlled by hook
            plt.close(self.fig)
        else:
            plt.show(block=True)
