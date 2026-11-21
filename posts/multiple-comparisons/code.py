"""
code.py — The Multiple Comparisons Problem, from a trading desk

One coin: can you tell a biased coin from a fair one after n flips?
1,000 coins: can you find the one biased coin among 999 fair ones?
1,000 strategies: what does the best backtest look like when no strategy has an edge?

Writes the figures for the blog post to the site's assets/ directory.

Usage:
    uv run --with numpy --with matplotlib --with scipy python code.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, norm

# ── Config ──────────────────────────────────────────────────
RNG = np.random.default_rng(7)
FIG_DIR = Path(__file__).resolve().parents[2] / 'assets'

N_FLIPS = 100          # flips per coin
P_BIASED = 0.6         # heads probability of the biased coin
N_COINS = 1_000        # one biased coin + 999 fair coins
N_REPS = 20_000        # repetitions of the 1,000-coin experiment

N_STRATEGIES = 1_000   # strategy variants in the backtest sweep
N_DAYS = 252           # one year of daily returns per backtest
N_SWEEPS = 2_000       # repetitions of the sweep

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
FAIR, BIASED, BEST = '#79c0ff', '#7ee787', '#f2cc60'


# ── Core functions ──────────────────────────────────────────
def coin_experiment(n_flips, n_reps, rng):
    """Return (max heads of the fair coins, heads of the biased coin) per repetition. :param n_flips: int :param n_reps: int :param rng: np.random.Generator :rtype: tuple[np.ndarray, np.ndarray]"""
    fair_heads = rng.binomial(n_flips, 0.5, size=(n_reps, N_COINS - 1))
    biased_heads = rng.binomial(n_flips, P_BIASED, size=n_reps)
    return fair_heads.max(axis=1), biased_heads


def p_found(n_flips, n_reps, rng):
    """Return the probability that the coin with the most heads is the biased one, ties split evenly. :param n_flips: int :param n_reps: int :param rng: np.random.Generator :rtype: float"""
    fair_max, biased = coin_experiment(n_flips, n_reps, rng)
    return np.mean(biased > fair_max) + 0.5 * np.mean(biased == fair_max)


def sharpe_sweep(rng):
    """Return the in-sample and out-of-sample annualised Sharpe of the best of N_STRATEGIES zero-edge strategies, per sweep. :param rng: np.random.Generator :rtype: tuple[np.ndarray, np.ndarray]"""
    best_in, best_out = np.empty(N_SWEEPS), np.empty(N_SWEEPS)
    for k in range(N_SWEEPS):
        returns = rng.standard_normal((N_STRATEGIES, 2 * N_DAYS))
        in_sample, out_sample = returns[:, :N_DAYS], returns[:, N_DAYS:]
        sr_in = in_sample.mean(axis=1) / in_sample.std(axis=1) * np.sqrt(252)
        sr_out = out_sample.mean(axis=1) / out_sample.std(axis=1) * np.sqrt(252)
        best = sr_in.argmax()
        best_in[k], best_out[k] = sr_in[best], sr_out[best]
    return best_in, best_out


def expected_max_normal(n):
    """Return the approximate expected maximum of n standard normals (Bailey & López de Prado, 2014). :param n: np.ndarray :rtype: np.ndarray"""
    gamma = 0.5772156649
    return (1 - gamma) * norm.ppf(1 - 1 / n) + gamma * norm.ppf(1 - 1 / (n * np.e))


def save(fig, name):
    """Save a figure to the assets directory. :param fig: plt.Figure :param name: str :rtype: None"""
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIG_DIR / name}')


# ── Figure 1: one coin ──────────────────────────────────────
def figure_1_one_coin():
    """Plot the heads distribution of a fair and a biased coin with the 5% rejection line. :rtype: None"""
    k = np.arange(25, 81)
    critical = int(binom.isf(0.05, N_FLIPS, 0.5)) + 1

    fig, ax = plt.subplots(figsize=(8, 3.6))
    for p, color, label in [(0.5, FAIR, 'fair coin (p = 0.5)'), (P_BIASED, BIASED, f'biased coin (p = {P_BIASED})')]:
        ax.stairs(binom.pmf(k, N_FLIPS, p), np.append(k, k[-1] + 1) - 0.5, color=color, fill=True, alpha=0.15)
        ax.stairs(binom.pmf(k, N_FLIPS, p), np.append(k, k[-1] + 1) - 0.5, color=color, label=label)
    ax.axvline(critical - 0.5, color='#c9d1d9', lw=0.8, ls=':')
    ax.text(critical + 13, 0.075, f'≥ {critical} heads:\n"biased" at 5%', color='#c9d1d9')
    ax.set_xlabel(f'heads in {N_FLIPS} flips')
    ax.set_ylabel('probability')
    ax.set_title('one coin · fair or biased?')
    ax.legend(loc='upper left')
    save(fig, 'mc_one_coin.png')

    print(f'  critical value: {critical} heads')
    print(f'  false positive rate: {binom.sf(critical - 1, N_FLIPS, 0.5):.3f}')
    print(f'  power vs p={P_BIASED}: {binom.sf(critical - 1, N_FLIPS, P_BIASED):.3f}')
    print(f'  expected fair coins over the line out of 999: {999 * binom.sf(critical - 1, N_FLIPS, 0.5):.1f}')


# ── Figure 2: 1,000 coins ───────────────────────────────────
def figure_2_thousand_coins(rng):
    """Plot the best fair coin against the biased coin, and the chance of finding the biased coin by flip count. :param rng: np.random.Generator :rtype: None"""
    fair_max, biased = coin_experiment(N_FLIPS, N_REPS, rng)
    one_fair = rng.binomial(N_FLIPS, 0.5, size=N_REPS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={'width_ratios': [3, 2]})

    bins = np.arange(25.5, 85.5)
    for data, color, label in [(one_fair, FAIR, 'one fair coin'), (biased, BIASED, 'the biased coin'),
                               (fair_max, BEST, 'best of 999 fair coins')]:
        ax1.hist(data, bins=bins, density=True, color=color, alpha=0.15)
        ax1.hist(data, bins=bins, density=True, color=color, histtype='step', lw=1.2, label=label)
    ax1.set_xlabel(f'heads in {N_FLIPS} flips')
    ax1.set_ylabel('probability')
    ax1.set_title('E[x] vs max(x) · 1,000 coins')
    ax1.legend(loc='upper left')

    flips = np.array([50, 100, 200, 300, 500, 750, 1_000, 2_000])
    found = [p_found(n, 2_000, rng) for n in flips]
    ax2.plot(flips, found, 'o-', color=BIASED, markersize=4)
    ax2.axhline(1 / N_COINS, color='#7d8590', lw=0.6, ls=':')
    ax2.set_xscale('log')
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_xlabel('flips per coin')
    ax2.set_ylabel('P(top coin is the biased one)')
    ax2.set_title('how many flips to find it?')
    save(fig, 'mc_thousand_coins.png')

    print(f'  mean heads, one fair coin: {one_fair.mean():.1f}')
    print(f'  mean heads, biased coin: {biased.mean():.1f}')
    print(f'  mean heads, best fair coin: {fair_max.mean():.1f}')
    print(f'  P(biased coin is top) at {N_FLIPS} flips: {np.mean(biased > fair_max) + 0.5 * np.mean(biased == fair_max):.3f}')
    for n, p in zip(flips, found):
        print(f'    {n:>6} flips: {p:.3f}')


# ── Figure 3: 1,000 strategies ──────────────────────────────
def figure_3_backtests(rng):
    """Plot the best backtest Sharpe in and out of sample, and the expected best Sharpe by number of trials. :param rng: np.random.Generator :rtype: None"""
    best_in, best_out = sharpe_sweep(rng)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={'width_ratios': [3, 2]})

    bins = np.linspace(-4, 6, 81)
    ax1.hist(best_out, bins=bins, density=True, color=FAIR, alpha=0.7, label='same strategy, next year')
    ax1.hist(best_in, bins=bins, density=True, color=BEST, alpha=0.7, label='best backtest of 1,000')
    ax1.axvline(0, color='#7d8590', lw=0.6, ls=':')
    ax1.set_xlabel('annualised Sharpe ratio')
    ax1.set_ylabel('density')
    ax1.set_title('1,000 strategies, zero edge · pick the best')
    ax1.legend(loc='upper left')

    n_trials = np.unique(np.logspace(0, 4, 40).astype(int))
    sims = [rng.standard_normal((4_000, n)).max(axis=1).mean() if n > 1 else 0.0 for n in n_trials]
    ax2.plot(n_trials, sims, 'o', color=BEST, markersize=3, label='simulation')
    ax2.plot(n_trials[1:], expected_max_normal(n_trials[1:]), color='#c9d1d9', lw=0.8, label='approximation')
    ax2.set_xscale('log')
    ax2.set_xlabel('strategies tried')
    ax2.set_ylabel('expected best Sharpe (1-year backtest)')
    ax2.set_title('the bar rises with every trial')
    ax2.legend(loc='upper left')
    save(fig, 'mc_backtests.png')

    print(f'  best in-sample Sharpe: mean {best_in.mean():.2f}, 5-95% {np.percentile(best_in, 5):.2f}-{np.percentile(best_in, 95):.2f}')
    print(f'  same strategy out of sample: mean {best_out.mean():.2f}, sd {best_out.std():.2f}')
    for n in [10, 100, 1_000, 10_000]:
        print(f'    E[max Sharpe] over {n:>6} trials: {expected_max_normal(n):.2f}')


# ── Main ────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Figure 1')
    figure_1_one_coin()
    print('Figure 2')
    figure_2_thousand_coins(RNG)
    print('Figure 3')
    figure_3_backtests(RNG)
