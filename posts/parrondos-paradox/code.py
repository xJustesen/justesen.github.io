"""
code.py — Parrondo's Paradox: Two Losing Games Make a Winner

Simulates Game A (biased coin) and Game B (capital-dependent coin),
shows that both lose individually but win when combined.

Writes the figures for the blog post to the site's assets/ directory.

Usage:
    python code.py

Requirements:
    numpy, matplotlib
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ── Config ──────────────────────────────────────────────────
RNG = np.random.default_rng(42)
N_ROUNDS = 10_000
N_PLAYERS = 1_000
FIG_DIR = Path(__file__).resolve().parents[2] / 'assets'

# Default probabilities
P1 = 0.495       # Game A: win probability (< 0.5, losing)
P2 = 0.095       # Game B: win prob when capital % 3 == 0 (bad coin)
P3 = 0.745       # Game B: win prob when capital % 3 != 0 (good coin)

# Plot style: the site's dark terminal palette (styles.css :root) and typeface
plt.rcParams.update(
    {
        'font.family': ['IBM Plex Mono', 'DejaVu Sans Mono'],
        'font.size': 8,
        'text.color': '#c9d1d9',
        'figure.facecolor': '#0e1116',
        'savefig.facecolor': '#0e1116',
        'axes.facecolor': '#0e1116',
        'axes.edgecolor': '#2d333b',
        'axes.linewidth': 0.8,
        'axes.labelcolor': '#7d8590',
        'axes.labelsize': 7,
        'axes.titlesize': 8,
        'axes.titlecolor': '#c9d1d9',
        'axes.titlelocation': 'left',
        'axes.titlepad': 10,
        'xtick.color': '#2d333b',
        'ytick.color': '#2d333b',
        'xtick.labelcolor': '#7d8590',
        'ytick.labelcolor': '#7d8590',
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'legend.frameon': False,
        'legend.fontsize': 7,
        'legend.labelcolor': '#8b949e',
        'lines.linewidth': 1.2,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    }
)
GAIN_CMAP = LinearSegmentedColormap.from_list('gain', ['#f2cc60', '#2d333b', '#7ee787'])


# ── Core functions ──────────────────────────────────────────
def simulate(strategy, n_rounds, n_players, rng, p1=P1, p2=P2, p3=P3):
    """Return capital paths of shape (n_players, n_rounds + 1). :param strategy: str 'A', 'B' or 'AB' (50/50 switch each round) :param n_rounds: int :param n_players: int :param rng: np.random.Generator :param p1: float :param p2: float :param p3: float :rtype: np.ndarray"""
    capitals = np.zeros((n_players, n_rounds + 1), dtype=int)
    cap = np.zeros(n_players, dtype=int)

    for t in range(n_rounds):
        if strategy == 'AB':
            play_a = rng.random(n_players) < 0.5
        else:
            play_a = np.full(n_players, strategy == 'A')
        p_b = np.where(cap % 3 == 0, p2, p3)
        p_win = np.where(play_a, p1, p_b)
        cap += np.where(rng.random(n_players) < p_win, 1, -1)
        capitals[:, t + 1] = cap

    return capitals


# ── Figure 1: Capital trajectories ─────────────────────────
def figure_1_trajectories(results):
    """Plot mean capital over time per strategy. :param results: dict[str, np.ndarray] :rtype: None"""
    fig, ax = plt.subplots(figsize=(8, 4))

    styles = {
        'A':  {'color': '#79c0ff', 'ls': '-',  'label': 'game A (p=0.495)'},
        'B':  {'color': '#f2cc60', 'ls': '--', 'label': 'game B (capital-dependent)'},
        'AB': {'color': '#7ee787', 'ls': '-',  'label': 'combined (random switch)'},
    }

    for key in ['A', 'B', 'AB']:
        mean_capital = results[key].mean(axis=0)
        s = styles[key]
        ax.plot(mean_capital, color=s['color'], ls=s['ls'], label=s['label'])

    ax.axhline(0, color='#7d8590', lw=0.6, ls=':')
    ax.set_xlabel('round')
    ax.set_ylabel('average capital')
    ax.set_title('capital over time · mean of 1,000 players')
    ax.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/capital_trajectories.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIG_DIR}/capital_trajectories.png')


# ── Figure 2: Mod-3 state distribution ─────────────────────
def figure_2_state_distribution(results):
    """Plot the capital mod 3 frequencies. :param results: dict[str, np.ndarray] :rtype: None"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    bar_colors = ['#f2cc60', '#3fb950', '#3fb950']
    labels = ['0 (bad coin)', '1', '2']

    b_caps = results['B'][:, 1:].flatten()
    b_counts = np.array([(b_caps % 3 == i).mean() for i in range(3)])

    ab_caps = results['AB'][:, 1:].flatten()
    ab_counts = np.array([(ab_caps % 3 == i).mean() for i in range(3)])

    uniform = np.array([1/3, 1/3, 1/3])

    panels = [
        ('game B alone', b_counts),
        ('combined (A+B)', ab_counts),
        ('uniform reference', uniform),
    ]

    for ax, (title, counts) in zip(axes, panels):
        bars = ax.bar(range(3), counts, color=bar_colors, width=0.7)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 0.5)
        ax.set_ylabel('frequency')
        ax.set_title(title)
        for bar, val in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', color='#8b949e')

    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/state_distribution.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIG_DIR}/state_distribution.png')


# ── Figure 3: Parameter sweep heatmap ──────────────────────
def figure_3_parameter_sweep(rng):
    """Plot the combined-strategy gain over a (p2, p3) grid. :param rng: np.random.Generator :rtype: None"""
    p1_fixed = 0.495
    p2_range = np.linspace(0.0, 0.5, 20)
    p3_range = np.linspace(0.5, 1.0, 20)
    n_sweep_players = 200
    n_sweep_rounds = 2_000

    gain = np.zeros((len(p3_range), len(p2_range)))

    for i, p3 in enumerate(p3_range):
        for j, p2 in enumerate(p2_range):
            caps = simulate('AB', n_sweep_rounds, n_sweep_players, rng,
                            p1=p1_fixed, p2=p2, p3=p3)
            gain[i, j] = caps[:, -1].mean() / n_sweep_rounds

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(gain, origin='lower', aspect='auto', cmap=GAIN_CMAP, norm=TwoSlopeNorm(vcenter=0.0),
                   extent=[p2_range[0], p2_range[-1], p3_range[0], p3_range[-1]])

    ax.contour(p2_range, p3_range, gain, levels=[0], colors=['#c9d1d9'], linewidths=1.0)

    ax.plot(P2, P3, 'o', color='#c9d1d9', markersize=6, markeredgecolor='#0e1116', markeredgewidth=1.0)
    ax.annotate('default', (P2, P3), textcoords='offset points', xytext=(8, -12), color='#c9d1d9')

    ax.set_xlabel('p₂ (bad coin)')
    ax.set_ylabel('p₃ (good coin)')
    ax.set_title('mean gain per round · combined strategy')
    cbar = fig.colorbar(im, ax=ax, label='gain / round', shrink=0.8)
    cbar.outline.set_edgecolor('#2d333b')
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/parameter_sweep.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIG_DIR}/parameter_sweep.png')


# ── Main ────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Running simulations...')

    results = {}
    for strategy in ['A', 'B', 'AB']:
        caps = simulate(strategy, N_ROUNDS, N_PLAYERS, RNG)
        results[strategy] = caps
        final = caps[:, -1].mean()
        print(f'  {strategy:4s}  final capital: {final:+.1f}  '
              f'(gain/round: {final / N_ROUNDS:+.5f})')

    print('\nGenerating figures...')
    figure_1_trajectories(results)
    figure_2_state_distribution(results)
    figure_3_parameter_sweep(RNG)
    print(f'\nDone. All figures saved to {FIG_DIR}')
