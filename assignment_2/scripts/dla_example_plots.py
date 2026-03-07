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
    pde_itas=(0.0, 1.0, 2.0),
    mc_ps=(1.0, 0.2, 0.05),
    seeds=(0, 1, 2),
    N=100,
    pde_steps=2000,
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
    Creates one 6x3 panel:
      - rows 0-2: PDE (rows = eta values, cols = seeds)
      - rows 3-5: MC  (rows = ps values,  cols = seeds)

    Colors show growth order.
    """

    n_seed = len(seeds)
    n_rows = len(pde_itas) + len(mc_ps)
    n_cols = n_seed

    pde_orders = []
    mc_orders = []
    max_order = 1

    # --- PDE ---
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
            max_order = max(max_order, int(growth.max()))
        pde_orders.append(row)

    # --- MC ---
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
            max_order = max(max_order, int(growth.max()))
        mc_orders.append(row)

    norm = Normalize(vmin=1, vmax=max_order)

    cmap_obj = plt.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    im = None

    # ---- PDE rows ----
    for i, ita in enumerate(pde_itas):
        for j, seed in enumerate(seeds):
            ax = axes[i, j]
            arr = pde_orders[i][j].astype(float)
            arr[arr == 0] = np.nan

            im = ax.imshow(arr, origin="lower", cmap=cmap_obj, norm=norm)

            # seed titles only once
            if i == 0:
                ax.set_title(f"seed={seed}", fontsize=title_fontsize)

            # row labels only in first column
            if j == 0:
                ax.set_ylabel(rf"PDE: $\eta={ita}$", fontsize=label_fontsize)

            if not show_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.tick_params(labelsize=tick_fontsize)

    # ---- MC rows ----
    row_offset = len(pde_itas)
    for i, ps in enumerate(mc_ps):
        for j, seed in enumerate(seeds):
            ax = axes[row_offset + i, j]
            arr = mc_orders[i][j].astype(float)
            arr[arr == 0] = np.nan

            im = ax.imshow(arr, origin="lower", cmap=cmap_obj, norm=norm)

            if j == 0:
                ax.set_ylabel(rf"MC: $p_s={ps}$", fontsize=label_fontsize)

            if not show_ticks:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.tick_params(labelsize=tick_fontsize)

    # make panels touch
    fig.subplots_adjust(wspace=0.0, hspace=0.0)

    # shared colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.01)
    cbar.set_label("Growth order", fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)

    fig.supxlabel("x", fontsize=label_fontsize)
    fig.supylabel("y", fontsize=label_fontsize)


    return fig, axes

if __name__ == "__main__":
    fig, axes = plot_growth_panels(
        pde_itas=(0.0, 1.0, 2.0),
        mc_ps=(1.0, 0.2, 0.05),
        seeds=(0, 1, 2),
        N=100,
        tick_fontsize=9,
        title_fontsize=12,
        label_fontsize=13,
        show_ticks=False
    )
    fig.savefig("../data/figs/dla_growth_panels.png", dpi=300, bbox_inches="tight")