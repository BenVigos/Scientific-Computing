import matplotlib.pyplot as plt
import numpy as np

N = 200
dt = 1.0
dx = 1.0
Du = 0.16
Dv = 0.08
noise_amp = 0.02


def laplacian(Z):
    """5-point finite-difference Laplacian with periodic BCs (via np.roll)."""
    return (
        np.roll(Z, 1, axis=0)
        + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1)
        + np.roll(Z, -1, axis=1)
        - 4 * Z
    ) / dx**2


def initialize(N, seed=42):
    """Set up initial conditions with a small central square of V."""
    rng = np.random.default_rng(seed)
    u = np.ones((N, N)) * 0.5
    v = np.zeros((N, N))

    # Small central square
    r = N // 10
    cx, cy = N // 2, N // 2
    v[cx - r : cx + r, cy - r : cy + r] = 0.25

    # Add noise
    u += noise_amp * rng.standard_normal((N, N))
    v += noise_amp * rng.standard_normal((N, N))

    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    return u, v


def simulate(f, k, n_steps, N=128, save_every=500):
    """Run the Gray-Scott simulation and return snapshots of v."""
    u, v = initialize(N)
    snapshots = []

    for step in range(n_steps + 1):
        if step % save_every == 0:
            snapshots.append((step, v.copy()))

        uvv = u * v * v
        Lu = laplacian(u)
        Lv = laplacian(v)

        u += dt * (Du * Lu - uvv + f * (1.0 - u))
        v += dt * (Dv * Lv + uvv - (f + k) * v)

    return snapshots


param_sets = [
    {"f": 0.035, "k": 0.060, "label": "Spots\nf=0.035, k=0.060", "steps": 10000},
    {"f": 0.025, "k": 0.052, "label": "Labyrinth\nf=0.025, k=0.052", "steps": 10000},
    {"f": 0.055, "k": 0.062, "label": "Worms\nf=0.055, k=0.062", "steps": 10000},
    {"f": 0.014, "k": 0.054, "label": "Solitons\nf=0.014, k=0.054", "steps": 8000},
]

print("Running Gray-Scott simulations...")
results = {}
for p in param_sets:
    print(f"  Simulating: {p['label'].replace(chr(10), ' | ')} ...")
    snaps = simulate(p["f"], p["k"], p["steps"])
    results[p["label"]] = snaps

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.patch.set_facecolor("#0d0d0d")
fig.suptitle(
    "Gray-Scott Reaction-Diffusion: Concentration of V",
    color="white",
    fontsize=15,
    fontweight="bold",
    y=0.98,
)

for ax, p in zip(axes.flat, param_sets, strict=False):
    snaps = results[p["label"]]
    v_final = snaps[-1][1]
    im = ax.imshow(
        v_final,
        cmap="inferno",
        vmin=0,
        vmax=0.4,
        origin="lower",
        interpolation="bilinear",
    )
    ax.set_title(p["label"], color="white", fontsize=11, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.yaxis.set_tick_params(
        color="white", labelcolor="white"
    )

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(
    "assignment_2/outputs/gray_scott_patterns.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)
plt.close()

f, k = 0.035, 0.060
n_steps = 10000
u0, v0 = initialize(N)
u, v = u0.copy(), v0.copy()

times_to_show = [0, 2000, 5000, 10000]
snap_dict = {}

step = 0
for t in range(n_steps + 1):
    if t in times_to_show:
        snap_dict[t] = v.copy()
    if t == n_steps:
        break
    uvv = u * v * v
    u += dt * (Du * laplacian(u) - uvv + f * (1.0 - u))
    v += dt * (Dv * laplacian(v) + uvv - (f + k) * v)

fig2, axes2 = plt.subplots(1, 4, figsize=(16, 4))
fig2.patch.set_facecolor("#0d0d0d")
fig2.suptitle(
    f"Time Evolution  (f={f}, k={k})  —  Concentration of V",
    color="white",
    fontsize=13,
    fontweight="bold",
)

for ax, t in zip(axes2, times_to_show, strict=False):
    im = ax.imshow(
        snap_dict[t],
        cmap="viridis",
        vmin=0,
        vmax=0.4,
        origin="lower",
        interpolation="bilinear",
    )
    ax.set_title(f"t = {t}", color="white", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.yaxis.set_tick_params(
        color="white", labelcolor="white"
    )

plt.tight_layout()
plt.savefig(
    "assignment_2/outputs/gray_scott_evolution.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=fig2.get_facecolor(),
)
plt.close()
