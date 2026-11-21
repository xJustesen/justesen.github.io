"""
siteplot.py — the site's matplotlib style, shared by the post scripts

Importing this module applies the site's dark terminal palette (styles.css :root)
and typeface to matplotlib.
"""

from pathlib import Path

import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parents[1] / 'assets'

# Colours named after the custom properties in styles.css :root
BG = '#0e1116'
FG = '#c9d1d9'
DIM = '#7d8590'
ACCENT = '#8b949e'
LINE = '#2d333b'
PROMPT = '#7ee787'
LINK = '#79c0ff'
PEAK = '#f2cc60'
CELL_LO = '#1f4d33'
CELL_MID = '#3fb950'

plt.rcParams.update(
    {
        'font.family': ['IBM Plex Mono', 'DejaVu Sans Mono'],
        'font.size': 8,
        'text.color': FG,
        'figure.facecolor': BG,
        'savefig.facecolor': BG,
        'axes.facecolor': BG,
        'axes.edgecolor': LINE,
        'axes.linewidth': 0.8,
        'axes.labelcolor': DIM,
        'axes.labelsize': 7,
        'axes.titlesize': 8,
        'axes.titlecolor': FG,
        'axes.titlelocation': 'left',
        'axes.titlepad': 10,
        'xtick.color': LINE,
        'ytick.color': LINE,
        'xtick.labelcolor': DIM,
        'ytick.labelcolor': DIM,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'legend.frameon': False,
        'legend.fontsize': 7,
        'legend.labelcolor': ACCENT,
        'lines.linewidth': 1.2,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
    }
)


def save(fig, name):
    """Save a figure to the site's assets directory and close it. :param fig: plt.Figure :param name: str :rtype: None"""
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIG_DIR / name}')
