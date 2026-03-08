import numpy as np
import matplotlib.pyplot as plt
import os, sys
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.abspath("../.."))

from src.dla_adapters import simulate_pde_dla, simulate_mc_dla


def omega_from_ita(ita):
    omega = 1.7 + 0.1 * ita
    return min(max(omega, 1.7), 1.9)


def plot_growth_panels(
    pde_itas=(0.02, 1.0, 2.0),
    pde_itas_big=(0.02, 1.0, 2.0),
    mc_ps=(1.0, 0.2, 0.001),
    seeds=(0, 1, 2),
    N=100,
    N_big=300,
    pde_steps=10000,
    occ_threshold=0.10,
    mc_max_steps=100000,
    cmap="viridis",
    figsize=(8, 14),
    tick_fontsize=8,
    label_fontsize=12,
    title_fontsize=13,
    show_ticks=False,
):
    """
    Creates one 9x3 panel:
      - rows 0-2: PDE optimized (N=300, parallel=True) (rows = eta values, cols = seeds)
      - rows 3-5: PDE normal (N=100) (rows = eta values, cols = seeds)
      - rows 6-8: MC  (rows = ps values,  cols = seeds)

    Colors show growth order.
    """

    n_seed = len(seeds)
    n_rows = len(pde_itas) + len(pde_itas_big) + len(mc_ps)
    n_cols = n_seed

    pde_orders = []
    pde_big_orders = []
    mc_orders = []
    max_order = 1

    # --- PDE normal (N=100) ---
    print("Simulating PDE normal (N=100)...")
    for ita in pde_itas:
        row = []
        for seed in seeds:
            _, growth = simulate_pde_dla(
                N=N,
                steps=pde_steps,
                stop_threshold=occ_threshold,
                ita=ita,
                seed=seed,
                omega=omega_from_ita(ita),
                return_growth_order=True,
            )
            row.append(growth)
            max_ord = int(growth.max()) if growth.max() > 0 else 1
            max_order = max(max_order, max_ord)
            print(f"  PDE (N={N}, ita={ita}, seed={seed}): growth max = {max_ord}")
        pde_orders.append(row)

    # --- PDE optimized (N=300, parallel=True) ---
    print("Simulating PDE optimized (N=300, parallel=True)...")
    for ita in pde_itas_big:
        row = []
        for seed in seeds:
            _, growth = simulate_pde_dla(
                N=N_big,
                steps=pde_steps,
                stop_threshold=occ_threshold,
                ita=ita,
                seed=seed,
                omega=omega_from_ita(ita),
                parallel=True,
                return_growth_order=True,
            )
            row.append(growth)
            max_ord = int(growth.max()) if growth.max() > 0 else 1
            max_order = max(max_order, max_ord)
            print(f"  PDE (N={N_big}, ita={ita}, seed={seed}, parallel=True): growth max = {max_ord}")
        pde_big_orders.append(row)

    # --- MC ---
    print("Simulating MC...")
    for ps in mc_ps:
        row = []
        for seed in seeds:
            _, growth = simulate_mc_dla(
                N=N,
                target_occupancy=occ_threshold,
                ps=ps,
                seed=seed,
                max_steps_per_walker=mc_max_steps,
                return_growth_order=True,
            )
            row.append(growth)
            max_ord = int(growth.max()) if growth.max() > 0 else 1
            max_order = max(max_order, max_ord)
            print(f"  MC (N={N}, ps={ps}, seed={seed}): growth max = {max_ord}")
        mc_orders.append(row)

    print(f"Overall max_order: {max_order}")

    # Use global norm for reference but will apply per-panel normalization
    global_norm = Normalize(vmin=1, vmax=max_order)

    cmap_obj = plt.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize[0], figsize[1] + 2)  # Add extra height for bottom spacing
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    im = None

    # ---- PDE optimized rows (0-2) ----
    for i, ita in enumerate(pde_itas_big):
        for j, seed in enumerate(seeds):
            ax = axes[i, j]
            arr = pde_big_orders[i][j].astype(float)
            arr[arr == 0] = np.nan

            # Per-panel normalization
            max_val = np.nanmax(arr)
            panel_norm = Normalize(vmin=1, vmax=max_val)
            im = ax.imshow(arr, origin="lower", cmap=cmap_obj, norm=panel_norm)

            # seed titles only once
            if i == 0:
                ax.set_title(f"seed={seed}", fontsize=title_fontsize)

            # row labels only in first column
            if j == 0:
                ax.set_ylabel(rf"PDE ($N=300$):" + "\n" + rf"$\eta={ita}$", fontsize=label_fontsize, rotation=45, ha='right', va='bottom', labelpad=10)

            if not show_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.tick_params(labelsize=tick_fontsize)

    # ---- PDE normal rows (3-5) ----
    row_offset_pde_normal = len(pde_itas_big)
    for i, ita in enumerate(pde_itas):
        for j, seed in enumerate(seeds):
            ax = axes[row_offset_pde_normal + i, j]
            arr = pde_orders[i][j].astype(float)
            arr[arr == 0] = np.nan

            # Per-panel normalization
            max_val = np.nanmax(arr)
            panel_norm = Normalize(vmin=1, vmax=max_val)
            im = ax.imshow(arr, origin="lower", cmap=cmap_obj, norm=panel_norm)

            # row labels only in first column
            if j == 0:
                ax.set_ylabel(rf"PDE (N=100):" + "\n" + rf"$\eta={ita}$", fontsize=label_fontsize, rotation=45, ha='right', va='bottom', labelpad=10)

            if not show_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.tick_params(labelsize=tick_fontsize)

    # ---- MC rows (6-8) ----
    row_offset = len(pde_itas) + len(pde_itas_big)
    for i, ps in enumerate(mc_ps):
        for j, seed in enumerate(seeds):
            ax = axes[row_offset + i, j]
            arr = mc_orders[i][j].astype(float)
            arr[arr == 0] = np.nan

            # Per-panel normalization
            max_val = np.nanmax(arr)
            panel_norm = Normalize(vmin=1, vmax=max_val)
            im = ax.imshow(arr, origin="lower", cmap=cmap_obj, norm=panel_norm)

            if j == 0:
                ax.set_ylabel(rf"MC:" + "\n" + rf"$p_s={ps}$", fontsize=label_fontsize, rotation=45, ha='right', va='bottom', labelpad=10)

            if not show_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.tick_params(labelsize=tick_fontsize)

    # make panels with increased row spacing and decreased column spacing
    fig.subplots_adjust(wspace=-0.01, hspace=0.15, left=0.25)

    # Create a colorbar with normalized values (0-1)
    norm_colorbar = Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm_colorbar)
    sm.set_array([])

    # shared colorbar - represents relative growth progression (0=first, 1=last)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.92, pad=0.01)
    cbar.set_label("Relative growth time (normalized)", fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    # fig.supxlabel("x", fontsize=label_fontsize)
    # fig.supylabel("y", fontsize=label_fontsize)


    return fig, axes

if __name__ == "__main__":
    fig, axes = plot_growth_panels(
        pde_itas=(2.0, 1.0, 0.2),
        pde_itas_big=(2.0, 1.0, 0.2),
        mc_ps=(1.0, 0.2, 0.001),
        seeds=(0, 1, 2),
        N=100,
        N_big=300,
        tick_fontsize=9,
        title_fontsize=12,
        label_fontsize=13,
        show_ticks=False
    )
    fig.savefig("../data/comparison_figures/dla_growth_panels.png", dpi=300, bbox_inches="tight")