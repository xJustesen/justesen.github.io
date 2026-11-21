"""
code.py — Estimating π with Monte Carlo & Importance Sampling

Demonstrates that you can estimate π using *any* 2D distribution,
not just uniform sampling, by applying importance sampling weights.

Writes the figures for the blog post to the site's assets/ directory.

Usage:
    python code.py

Requirements:
    numpy, matplotlib, scipy
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from siteplot import BG, CELL_LO, CELL_MID, DIM, FG, FIG_DIR, LINK, PEAK, PROMPT, save  # noqa: E402

# ── Config ──────────────────────────────────────────────────
RNG = np.random.default_rng(42)
N_SAMPLES = 50_000

DENSITY_CMAP = LinearSegmentedColormap.from_list("density", [BG, CELL_LO, CELL_MID, PROMPT])


# ── Helpers ─────────────────────────────────────────────────
def in_quarter_circle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return 1.0 where x² + y² ≤ 1, else 0.0. :param x: np.ndarray :param y: np.ndarray :rtype: np.ndarray"""
    return (x**2 + y**2 <= 1.0).astype(float)


def estimate_pi_uniform(x: np.ndarray, y: np.ndarray) -> float:
    """Return the plain Monte Carlo estimate 4 · hits / N. :param x: np.ndarray :param y: np.ndarray :rtype: float"""
    return 4.0 * in_quarter_circle(x, y).mean()


def estimate_pi_importance(
    x: np.ndarray,
    y: np.ndarray,
    q_x: np.ndarray,
    q_y: np.ndarray,
) -> float:
    """Return 4 · mean(g · f / q) with f = 1 on [0,1]² and q = q_x · q_y. :param x: np.ndarray :param y: np.ndarray :param q_x: np.ndarray :param q_y: np.ndarray :rtype: float"""
    mask = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
    g = in_quarter_circle(x[mask], y[mask])
    weights = 1.0 / (q_x[mask] * q_y[mask])  # f=1 on the unit square
    full_weights = np.zeros_like(x)
    full_g = np.zeros_like(x)
    full_g[mask] = g
    full_weights[mask] = weights
    return 4.0 * np.mean(full_g * full_weights)


def convergence_curve(estimates: np.ndarray) -> np.ndarray:
    """Return the running mean. :param estimates: np.ndarray :rtype: np.ndarray"""
    return np.cumsum(estimates) / np.arange(1, len(estimates) + 1)


# ── Distribution definitions ───────────────────────────────
class UniformSampler:
    name = "Uniform"
    color = FG
    ls = "-"

    def sample(self, n, rng):
        x = rng.uniform(0, 1, n)
        y = rng.uniform(0, 1, n)
        qx = np.ones(n)
        qy = np.ones(n)
        return x, y, qx, qy


class GaussianSampler:
    name = "Gaussian (μ=0.5, σ=0.3)"
    color = LINK
    ls = "--"

    def __init__(self, mu=0.5, sigma=0.3):
        self.mu = mu
        self.sigma = sigma
        self.dist = stats.norm(loc=mu, scale=sigma)

    def sample(self, n, rng):
        x = rng.normal(self.mu, self.sigma, n)
        y = rng.normal(self.mu, self.sigma, n)
        qx = self.dist.pdf(x)
        qy = self.dist.pdf(y)
        return x, y, qx, qy


class ShotgunSampler:
    """Clustered pellets: a Gaussian spread around a random aim point per shot."""

    name = "Shotgun (clustered)"
    color = PROMPT
    ls = "-."

    def __init__(self, pellets_per_shot=200, shot_sigma=0.2):
        self.pellets = pellets_per_shot
        self.shot_sigma = shot_sigma

    def sample(self, n, rng):
        n_shots = max(1, n // self.pellets)
        xs, ys = [], []
        for _ in range(n_shots):
            cx = rng.uniform(0.2, 0.8)
            cy = rng.uniform(0.2, 0.8)
            xs.append(rng.normal(cx, self.shot_sigma, self.pellets))
            ys.append(rng.normal(cy, self.shot_sigma, self.pellets))
        x = np.concatenate(xs)[:n]
        y = np.concatenate(ys)[:n]

        bins = 40
        hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, 1], [0, 1]], density=True)
        x_idx = np.clip(np.digitize(x, xedges) - 1, 0, bins - 1)
        y_idx = np.clip(np.digitize(y, yedges) - 1, 0, bins - 1)
        in_square = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
        # q comes from the same pellets it weights, so every in-square pellet has a non-empty bin.
        q_vals = np.where(in_square, hist[x_idx, y_idx], 1.0)
        # The histogram only sees pellets inside the square, so q is rescaled until the
        # in-square weights average to the square's area (1); this makes the estimate self-normalised.
        q_vals = q_vals * np.mean(np.where(in_square, 1.0 / q_vals, 0.0))

        return x, y, q_vals, np.ones_like(q_vals)


class BetaSampler:
    name = "Beta (α=2, β=5)"
    color = PEAK
    ls = ":"

    def __init__(self, a=2, b=5):
        self.a = a
        self.b = b
        self.dist = stats.beta(a, b)

    def sample(self, n, rng):
        x = rng.beta(self.a, self.b, n)
        y = rng.beta(self.a, self.b, n)
        qx = self.dist.pdf(x)
        qy = self.dist.pdf(y)
        return x, y, qx, qy


# ── Run simulations ─────────────────────────────────────────
samplers = [UniformSampler(), GaussianSampler(), ShotgunSampler(), BetaSampler()]
results = {}

for sampler in samplers:
    x, y, qx, qy = sampler.sample(N_SAMPLES, RNG)

    if isinstance(sampler, UniformSampler):
        pi_est = estimate_pi_uniform(x, y)
        per_sample = 4.0 * in_quarter_circle(x, y)
    else:
        pi_est = estimate_pi_importance(x, y, qx, qy)
        mask = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
        per_sample = np.zeros_like(x)
        g = in_quarter_circle(x[mask], y[mask])
        weights = 1.0 / (qx[mask] * qy[mask])
        per_sample[mask] = 4.0 * g * weights

    results[sampler.name] = {
        "x": x,
        "y": y,
        "qx": qx,
        "qy": qy,
        "pi": pi_est,
        "per_sample": per_sample,
        "sampler": sampler,
    }
    print(f"{sampler.name:30s}  π ≈ {pi_est:.5f}  (error: {abs(pi_est - np.pi):.5f})")


# ── Figure 1: Scatter plots comparing distributions ────────
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

for ax, sampler in zip(axes, samplers):
    r = results[sampler.name]
    x, y = r["x"], r["y"]

    vis = (x >= -0.1) & (x <= 1.1) & (y >= -0.1) & (y <= 1.1)
    xv, yv = x[vis][:5000], y[vis][:5000]
    inside = in_quarter_circle(xv, yv).astype(bool)

    ax.scatter(xv[inside], yv[inside], s=1.5, c=PROMPT, alpha=0.6, linewidths=0)
    ax.scatter(xv[~inside], yv[~inside], s=1.5, c=DIM, alpha=0.35, linewidths=0)

    theta = np.linspace(0, np.pi / 2, 100)
    ax.plot(np.cos(theta), np.sin(theta), c=FG, lw=0.8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(sampler.name.lower())
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])

fig.suptitle("sample distributions · green = inside the quarter circle", x=0.01, ha="left", y=1.02)
save(fig, 'scatter_comparison.png')


# ── Figure 2: Convergence curves ───────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))

for sampler in samplers:
    r = results[sampler.name]
    curve = convergence_curve(r["per_sample"])
    s = r["sampler"]
    n_plot = min(len(curve), N_SAMPLES)
    ax.plot(
        np.arange(1, n_plot + 1),
        curve[:n_plot],
        label=f'{s.name.lower()} (π ≈ {r["pi"]:.4f})',
        color=s.color,
        ls=s.ls,
    )

ax.axhline(np.pi, color=DIM, lw=0.6, ls=":")
ax.set_xlabel("samples")
ax.set_ylabel("running π estimate")
ax.set_xscale("log")
ax.set_ylim(2.5, 4.0)
ax.legend(loc="upper right")
ax.set_title("convergence to π by sampling strategy")
save(fig, 'convergence.png')


# ── Figure 3: Importance weights heatmap (Gaussian case) ───
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

gx = np.linspace(0, 1, 100)
gy = np.linspace(0, 1, 100)
GX, GY = np.meshgrid(gx, gy)
gauss = GaussianSampler()
Z_q = gauss.dist.pdf(GX) * gauss.dist.pdf(GY)
axes[0].contourf(GX, GY, Z_q, levels=20, cmap=DENSITY_CMAP)
axes[0].set_title("q(x,y): proposal density")
axes[0].set_aspect("equal")

Z_f = np.ones_like(GX)
Z_w = Z_f / Z_q
axes[1].contourf(GX, GY, Z_w, levels=20, cmap=DENSITY_CMAP)
axes[1].set_title("f(x,y) / q(x,y): importance weights")
axes[1].set_aspect("equal")

r = results[gauss.name]
x, y = r["x"], r["y"]
vis = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
xv, yv = x[vis][:3000], y[vis][:3000]
wv = 1.0 / (gauss.dist.pdf(xv) * gauss.dist.pdf(yv))
wv_norm = wv / wv.max()
axes[2].scatter(xv, yv, s=wv_norm * 8, c=PROMPT, alpha=0.35, linewidths=0)
theta = np.linspace(0, np.pi / 2, 100)
axes[2].plot(np.cos(theta), np.sin(theta), c=FG, lw=0.8)
axes[2].set_title("weighted samples (size ∝ weight)")
axes[2].set_aspect("equal")
axes[2].set_xlim(-0.05, 1.05)
axes[2].set_ylim(-0.05, 1.05)

for ax in axes:
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])

save(fig, 'importance_weights.png')

print(f"\nDone. All figures saved to {FIG_DIR}")
