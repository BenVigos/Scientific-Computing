import numpy as np

from src.diffusion_limited_aggregation import diffusion_limited_aggregation as pde_dla
from src.mc import simulate_dla_mc_numba


def simulate_pde_dla(*, N, steps, stop_threshold, ita, seed, debug=False, omega=1.9):
    """
    PDE-DLA adapter.
    Returns grid as uint8 with convention y=0 bottom 
    """
    np.random.seed(seed)
    grid = pde_dla((N, N), steps=steps, stop_threshold=stop_threshold, debug=debug, ita=ita, omega=omega)
    grid = (grid > 0).astype(np.uint8)

    # PDE uses row 0 = top, row N-1 = bottom -> flip so y=0 becomes bottom
    grid = np.flipud(grid)
    return grid


def simulate_mc_dla(*, N, target_occupancy, ps, seed, max_steps_per_walker, top_boundary_percentage_stop=0.99):
    """
    MC-DLA adapter.
    Returns grid as uint8 with convention y=0 bottom
    """
    grid = simulate_dla_mc_numba(
        N=N,
        seed=seed,
        target_occupancy=target_occupancy,
        ps=ps,
        max_steps_per_walker=max_steps_per_walker,
        top_boundary_percentage_stop=top_boundary_percentage_stop,
    )
    return grid.astype(np.uint8)