import matplotlib.pyplot as plt
import numpy as np
import os

N = 128
dt = 1.0
dx = 1.0
Du = 0.16
Dv = 0.08
noise_amp = 0.02

output_dir = "assignment_2/data/gray_scott"
os.makedirs(output_dir, exist_ok=True)

def laplacian(Z):
    """
    5-point finite-difference Laplacian with periodic boundary conditions.
    
    param Z: 2D array representing either U or V concentration.
    returns: 2D array of the same shape containing the Laplacian of Z.
    
    """
    return (
        np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
        np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) - 4 * Z
    ) / dx**2

def initialize(N, u_init=0.5, seed=42):
    """
    Set up initial conditions, allowing customizable initial U concentration.
    
    param N: Size of the grid (N x N).
    param u_init: Initial concentration for U (default 0.5).
    param seed: Random seed for reproducibility (default 42).
    returns: Tuple of 2D arrays (u, v) representing initial concentrations.
    
    """
    rng = np.random.default_rng(seed)
    
    u = np.ones((N, N)) * u_init
    v = np.zeros((N, N))

    # Small central square of V
    r = N // 10
    cx, cy = N // 2, N // 2
    v[cx - r : cx + r, cy - r : cy + r] = 0.25

    # Addition of the noise
    u += noise_amp * rng.standard_normal((N, N))
    v += noise_amp * rng.standard_normal((N, N))

    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    return u, v


def simulate_basic(f, k, n_steps):
    """
    Standard simulation for the parameter sweep.
    
    param f: Feed rate for U.
    param k: Kill rate for V.
    param n_steps: Number of time steps to simulate.
    returns: Final concentrations of U and V after simulation.
    
    """
    u, v = initialize(N, u_init=0.5)
    for _ in range(n_steps):
        uvv = u * v * v
        Lu = laplacian(u)
        Lv = laplacian(v)
        u += dt * (Du * Lu - uvv + f * (1.0 - u))
        v += dt * (Dv * Lv + uvv - (f + k) * v)
    return u, v

param_sets = [
    {"f": 0.035, "k": 0.060, "label": "Configuration 1\nf=0.035, k=0.060", "steps": 10000},
    {"f": 0.025, "k": 0.052, "label": "Configuration 2\nf=0.025, k=0.052", "steps": 10000},
    {"f": 0.055, "k": 0.062, "label": "Configuration 3\nf=0.055, k=0.062", "steps": 10000},
    {"f": 0.014, "k": 0.054, "label": "Configuration 4\nf=0.014, k=0.054", "steps": 10000},
]

fig_sweep, axes_sweep = plt.subplots(4, 2, figsize=(8, 16))
fig_sweep.suptitle("Gray-Scott Patterns (U and V Concentrations)", fontsize=16, y=0.98)

for i, p in enumerate(param_sets):
    print(f"  Simulating: {p['label'].replace(chr(10), ' | ')} ...")
    u_final, v_final = simulate_basic(p["f"], p["k"], p["steps"])
    
    # Plot U
    ax_u = axes_sweep[i, 0]
    im_u = ax_u.imshow(u_final, cmap="magma", origin="lower", interpolation="bilinear")
    ax_u.set_title(f"U Concentration\n{p['label']}", fontsize=10)
    ax_u.axis('off')
    plt.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04)

    # Plot V
    ax_v = axes_sweep[i, 1]
    im_v = ax_v.imshow(v_final, cmap="viridis", origin="lower", interpolation="bilinear")
    ax_v.set_title(f"V Concentration\n{p['label']}", fontsize=10)
    ax_v.axis('off')
    plt.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(f"{output_dir}/gray_scott_parameters_UV.png", dpi=150, bbox_inches="tight")
plt.close()

f_base, k_base = 0.035, 0.060
n_steps_base = 10000
times_to_show = [0, 2000, 5000, 10000]
snap_dict_v = {}

u, v = initialize(N, u_init=0.5)

for t in range(n_steps_base + 1):
    if t in times_to_show:
        snap_dict_v[t] = v.copy()
    if t == n_steps_base:
        break
    uvv = u * v * v
    u += dt * (Du * laplacian(u) - uvv + f_base * (1.0 - u))
    v += dt * (Dv * laplacian(v) + uvv - (f_base + k_base) * v)

fig_evol, axes_evol = plt.subplots(1, 4, figsize=(16, 4))
fig_evol.suptitle(f"Time Evolution of V (f={f_base}, k={k_base})", fontsize=14, fontweight="bold")

for ax, t in zip(axes_evol, times_to_show):
    im = ax.imshow(snap_dict_v[t], cmap="viridis", origin="lower", interpolation="bilinear")
    ax.set_title(f"Step = {t}", fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(f"{output_dir}/gray_scott_evolution_base.png", dpi=150, bbox_inches="tight")
plt.close()

f_c4, k_c4 = 0.014, 0.054
n_steps_c4 = 10000

def simulate_with_analysis(u_init):
    """
    Run simulation and track mean concentrations over time.
    
    param u_init: Initial concentration for U (0.5 for starved, 1.0 for survived).
    returns: Final V concentration and lists of mean U and V over time.
    
    """
    u, v = initialize(N, u_init=u_init)
    mean_u, mean_v = [], []
    
    for step in range(n_steps_c4 + 1):
        if step % 100 == 0:
            mean_u.append(np.mean(u))
            mean_v.append(np.mean(v))
            
        uvv = u * v * v
        Lu = laplacian(u)
        Lv = laplacian(v)
        u += dt * (Du * Lu - uvv + f_c4 * (1.0 - u))
        v += dt * (Dv * Lv + uvv - (f_c4 + k_c4) * v)

    return v, mean_u, mean_v

# Run both cases
v_starved, mu_starved, mv_starved = simulate_with_analysis(u_init=0.5)
v_survived, mu_survived, mv_survived = simulate_with_analysis(u_init=1.0)
time_axis = np.arange(0, n_steps_c4 + 1, 100)

fig_spat, axes_spat = plt.subplots(1, 2, figsize=(10, 5))
fig_spat.suptitle(f"Configuration 4 Spatial Patterns (f={f_c4}, k={k_c4})", fontsize=14, fontweight='bold')

im1 = axes_spat[0].imshow(v_starved, cmap="inferno", vmin=0, vmax=0.4, origin="lower")
axes_spat[0].set_title("Result: Extinction / Blank ($U_{init} = 0.5$)", fontsize=11)
axes_spat[0].axis('off')
plt.colorbar(im1, ax=axes_spat[0], fraction=0.046, pad=0.04).set_label('Concentration of V')

im2 = axes_spat[1].imshow(v_survived, cmap="inferno", vmin=0, vmax=0.4, origin="lower")
axes_spat[1].set_title("Result: Pattern Formation ($U_{init} = 1.0$)", fontsize=11)
axes_spat[1].axis('off')
plt.colorbar(im2, ax=axes_spat[1], fraction=0.046, pad=0.04).set_label('Concentration of V')

plt.tight_layout()
plt.savefig(f"{output_dir}/config4_analysis_spatial.png", dpi=150, bbox_inches="tight")
plt.close()

fig_stat, axes_stat = plt.subplots(1, 2, figsize=(12, 5))
fig_stat.suptitle(f"Configuration 4 Global Mean Concentration (f={f_c4}, k={k_c4})", fontsize=14, fontweight='bold')

axes_stat[0].plot(time_axis, mu_starved, label="Mean U", color='royalblue', linewidth=2)
axes_stat[0].plot(time_axis, mv_starved, label="Mean V", color='crimson', linewidth=2)
axes_stat[0].set_title("Concentration over Time ($U_{init} = 0.5$)", fontsize=11)
axes_stat[0].set_xlabel("Time steps")
axes_stat[0].set_ylabel("Global Mean Concentration")
axes_stat[0].legend()
axes_stat[0].grid(True, linestyle='--', alpha=0.6)
axes_stat[0].set_ylim(-0.05, 1.05)

axes_stat[1].plot(time_axis, mu_survived, label="Mean U", color='royalblue', linewidth=2)
axes_stat[1].plot(time_axis, mv_survived, label="Mean V", color='crimson', linewidth=2)
axes_stat[1].set_title("Concentration over Time ($U_{init} = 1.0$)", fontsize=11)
axes_stat[1].set_xlabel("Time steps")
axes_stat[1].legend()
axes_stat[1].grid(True, linestyle='--', alpha=0.6)
axes_stat[1].set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig(f"{output_dir}/config4_analysis_stats.png", dpi=150, bbox_inches="tight")
plt.close()
