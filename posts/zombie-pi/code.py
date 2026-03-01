"""
code.py — Estimating π with Monte Carlo & Importance Sampling

Demonstrates that you can estimate π using *any* 2D distribution,
not just uniform sampling, by applying importance sampling weights.

Generates figures for the accompanying blog post.

Usage:
    python code.py

Requirements:
    numpy, matplotlib, scipy
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ── Config ──────────────────────────────────────────────────
RNG = np.random.default_rng(42)
N_SAMPLES = 50_000
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Plot style — monochrome, minimal, matches the site
plt.rcParams.update(
    {
        "font.family": "monospace",
        "font.size": 9,
        "axes.facecolor": "#f5f5f5",
        "figure.facecolor": "#f5f5f5",
        "axes.edgecolor": "#111",
        "axes.labelcolor": "#111",
        "xtick.color": "#555",
        "ytick.color": "#555",
        "text.color": "#111",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ── Helpers ─────────────────────────────────────────────────
def in_quarter_circle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """g(x, y) = 1 if x² + y² ≤ 1, else 0."""
    return (x**2 + y**2 <= 1.0).astype(float)


def estimate_pi_uniform(x: np.ndarray, y: np.ndarray) -> float:
    """Classic Monte Carlo: π ≈ 4 * (hits / N)."""
    return 4.0 * in_quarter_circle(x, y).mean()


def estimate_pi_importance(
    x: np.ndarray,
    y: np.ndarray,
    q_x: np.ndarray,
    q_y: np.ndarray,
) -> float:
    """
    Importance sampling estimate of π.

    f(x,y) = 1 on [0,1]² (the target uniform density)
    q(x,y) = q_x * q_y  (the proposal density, factorised)

    π ≈ 4 * (1/N) Σ g(xᵢ,yᵢ) * f(xᵢ,yᵢ) / q(xᵢ,yᵢ)

    We only count samples that land in [0,1]², and the
    importance weight corrects for the non-uniform density.
    """
    mask = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
    g = in_quarter_circle(x[mask], y[mask])
    weights = 1.0 / (q_x[mask] * q_y[mask])  # f=1 on the unit square
    # Scale by (total samples / in-square samples) isn't needed if we
    # use the full-sample mean with 0 contribution outside [0,1]²
    # But cleaner: use the proper IS estimator over ALL samples
    full_weights = np.zeros_like(x)
    full_g = np.zeros_like(x)
    full_g[mask] = g
    full_weights[mask] = weights
    full_weights[~mask] = 0.0  # f(x)=0 outside unit square
    return 4.0 * np.mean(full_g * full_weights)


def convergence_curve(estimates: np.ndarray) -> np.ndarray:
    """Running cumulative mean."""
    return np.cumsum(estimates) / np.arange(1, len(estimates) + 1)


# ── Distribution definitions ───────────────────────────────
class UniformSampler:
    name = "Uniform"
    color = "#111"
    ls = "-"

    def sample(self, n, rng):
        x = rng.uniform(0, 1, n)
        y = rng.uniform(0, 1, n)
        qx = np.ones(n)
        qy = np.ones(n)
        return x, y, qx, qy


class GaussianSampler:
    name = "Gaussian (μ=0.5, σ=0.3)"
    color = "#666"
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
    """Simulates a clustered shotgun-like spread — bivariate Gaussian
    with a random center per 'shot' (batch of pellets)."""

    name = "Shotgun (clustered)"
    color = "#999"
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

        # Estimate q(x,y) with 2D histogram (like the paper)
        # Use half for density estimation
        split = n // 2
        x_est, y_est = x[:split], y[:split]
        x_eval, y_eval = x[split:], y[split:]

        bins = 20
        hist, xedges, yedges = np.histogram2d(x_est, y_est, bins=bins, range=[[0, 1], [0, 1]], density=True)

        # Smooth the histogram a bit to avoid zero bins
        from scipy.ndimage import uniform_filter

        hist = uniform_filter(hist, size=2)
        # Renormalize
        bin_area = (1.0 / bins) ** 2
        hist = hist / (hist.sum() * bin_area)

        # For each eval point, look up the bin density
        x_idx = np.clip(np.digitize(x_eval, xedges) - 1, 0, bins - 1)
        y_idx = np.clip(np.digitize(y_eval, yedges) - 1, 0, bins - 1)
        q_vals = hist[x_idx, y_idx]
        q_vals = np.maximum(q_vals, 0.01)  # floor to avoid extreme weights

        return x_eval, y_eval, q_vals, np.ones_like(q_vals)


class BetaSampler:
    name = "Beta (α=2, β=5)"
    color = "#bbb"
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
        # Per-sample estimates for convergence
        per_sample = 4.0 * in_quarter_circle(x, y)
    else:
        pi_est = estimate_pi_importance(x, y, qx, qy)
        # Per-sample weighted estimates for convergence
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

    # Clip to visible range for plotting
    vis = (x >= -0.1) & (x <= 1.1) & (y >= -0.1) & (y <= 1.1)
    xv, yv = x[vis][:5000], y[vis][:5000]
    inside = in_quarter_circle(xv, yv).astype(bool)

    ax.scatter(xv[inside], yv[inside], s=0.5, c="#111", alpha=0.3, linewidths=0)
    ax.scatter(xv[~inside], yv[~inside], s=0.5, c="#aaa", alpha=0.3, linewidths=0)

    # Quarter circle arc
    theta = np.linspace(0, np.pi / 2, 100)
    ax.plot(np.cos(theta), np.sin(theta), c="#111", lw=0.8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(sampler.name, fontsize=8, pad=8)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])

fig.suptitle("Sample distributions — dark = inside quarter circle", fontsize=9, y=1.02)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/scatter_comparison.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG_DIR}/scatter_comparison.png")


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
        label=f'{s.name} (π ≈ {r["pi"]:.4f})',
        color=s.color,
        ls=s.ls,
        lw=1.2,
    )

ax.axhline(np.pi, color="#111", lw=0.6, ls=":", alpha=0.4)
ax.set_xlabel("samples")
ax.set_ylabel("running π estimate")
ax.set_xscale("log")
ax.set_ylim(2.5, 4.0)
ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.set_title("Convergence to π across sampling strategies", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/convergence.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG_DIR}/convergence.png")


# ── Figure 3: Importance weights heatmap (Gaussian case) ───
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

# Panel 1: Gaussian density
gx = np.linspace(0, 1, 100)
gy = np.linspace(0, 1, 100)
GX, GY = np.meshgrid(gx, gy)
gauss = GaussianSampler()
Z_q = gauss.dist.pdf(GX) * gauss.dist.pdf(GY)
axes[0].contourf(GX, GY, Z_q, levels=20, cmap="Greys")
axes[0].set_title("q(x,y): proposal density", fontsize=8)
axes[0].set_aspect("equal")

# Panel 2: Importance weights f/q
Z_f = np.ones_like(GX)  # uniform on [0,1]²
Z_w = Z_f / Z_q
axes[1].contourf(GX, GY, Z_w, levels=20, cmap="Greys")
axes[1].set_title("f(x,y) / q(x,y): importance weights", fontsize=8)
axes[1].set_aspect("equal")

# Panel 3: Effective weighted samples
r = results[gauss.name]
x, y = r["x"], r["y"]
vis = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
xv, yv = x[vis][:3000], y[vis][:3000]
wv = 1.0 / (gauss.dist.pdf(xv) * gauss.dist.pdf(yv))
wv_norm = wv / wv.max()
axes[2].scatter(xv, yv, s=wv_norm * 8, c="#111", alpha=0.25, linewidths=0)
theta = np.linspace(0, np.pi / 2, 100)
axes[2].plot(np.cos(theta), np.sin(theta), c="#111", lw=0.8)
axes[2].set_title("weighted samples (size ∝ weight)", fontsize=8)
axes[2].set_aspect("equal")
axes[2].set_xlim(-0.05, 1.05)
axes[2].set_ylim(-0.05, 1.05)

for ax in axes:
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])

fig.tight_layout()
fig.savefig(f"{FIG_DIR}/importance_weights.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved {FIG_DIR}/importance_weights.png")

print("\nDone. All figures saved to figures/")
