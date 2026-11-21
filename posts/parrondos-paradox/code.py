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

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from siteplot import ACCENT, BG, CELL_MID, DIM, FG, FIG_DIR, LINE, LINK, PEAK, PROMPT, save  # noqa: E402

# ── Config ──────────────────────────────────────────────────
RNG = np.random.default_rng(42)
N_ROUNDS = 10_000
N_PLAYERS = 1_000

# Default probabilities
P1 = 0.495       # Game A: win probability (< 0.5, losing)
P2 = 0.095       # Game B: win prob when capital % 3 == 0 (bad coin)
P3 = 0.745       # Game B: win prob when capital % 3 != 0 (good coin)

GAIN_CMAP = LinearSegmentedColormap.from_list('gain', [PEAK, LINE, PROMPT])


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
        'A':  {'color': LINK, 'ls': '-',  'label': 'game A (p=0.495)'},
        'B':  {'color': PEAK, 'ls': '--', 'label': 'game B (capital-dependent)'},
        'AB': {'color': PROMPT, 'ls': '-',  'label': 'combined (random switch)'},
    }

    for key in ['A', 'B', 'AB']:
        mean_capital = results[key].mean(axis=0)
        s = styles[key]
        ax.plot(mean_capital, color=s['color'], ls=s['ls'], label=s['label'])

    ax.axhline(0, color=DIM, lw=0.6, ls=':')
    ax.set_xlabel('round')
    ax.set_ylabel('average capital')
    ax.set_title('capital over time · mean of 1,000 players')
    ax.legend(loc='lower left')
    save(fig, 'capital_trajectories.png')


# ── Figure 2: Mod-3 state distribution ─────────────────────
def figure_2_state_distribution(results):
    """Plot the capital mod 3 frequencies. :param results: dict[str, np.ndarray] :rtype: None"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    bar_colors = [PEAK, CELL_MID, CELL_MID]
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
                    f'{val:.3f}', ha='center', va='bottom', color=ACCENT)

    save(fig, 'state_distribution.png')


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

    ax.contour(p2_range, p3_range, gain, levels=[0], colors=[FG], linewidths=1.0)

    ax.plot(P2, P3, 'o', color=FG, markersize=6, markeredgecolor=BG, markeredgewidth=1.0)
    ax.annotate('default', (P2, P3), textcoords='offset points', xytext=(8, -12), color=FG)

    ax.set_xlabel('p₂ (bad coin)')
    ax.set_ylabel('p₃ (good coin)')
    ax.set_title('mean gain per round · combined strategy')
    cbar = fig.colorbar(im, ax=ax, label='gain / round', shrink=0.8)
    cbar.outline.set_edgecolor(LINE)
    save(fig, 'parameter_sweep.png')


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
