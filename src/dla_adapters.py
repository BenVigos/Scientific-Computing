import numpy as np

from src.diffusion_limited_aggregation import diffusion_limited_aggregation as pde_dla
from src.mc import simulate_dla_mc_numba


def simulate_pde_dla(*, N, steps, stop_threshold, ita, seed, debug=False, omega=1.9, return_growth_order=False, parallel: bool = False):
    """
    PDE-DLA adapter.
    Returns grid as uint8 with convention y=0 bottom 
    """
    np.random.seed(seed)

    if return_growth_order:
        grid, growth_order = pde_dla(
            (N, N),
            steps=steps,
            stop_threshold=stop_threshold,
            debug=debug,
            ita=ita,
            omega=omega,
            return_growth_order=True
        )
        grid = (grid > 0).astype(np.uint8)
        grid = np.flipud(grid)
        growth_order = np.flipud(growth_order)
        return grid, growth_order

    grid = pde_dla(
        (N, N),
        steps=steps,
        stop_threshold=stop_threshold,
        debug=debug,
        ita=ita,
        omega=omega
    )
    grid = (grid > 0).astype(np.uint8)
    grid = np.flipud(grid)
    return grid


def simulate_mc_dla(*, N, target_occupancy, ps, seed, max_steps_per_walker, top_boundary_percentage_stop=0.99, return_growth_order=False):
    """
    MC-DLA adapter.
    Returns grid as uint8 with convention y=0 bottom
    """
    if return_growth_order:
        grid, growth_order = simulate_dla_mc_numba(
            N=N,
            seed=seed,
            target_occupancy=target_occupancy,
            ps=ps,
            max_steps_per_walker=max_steps_per_walker,
            top_boundary_percentage_stop=top_boundary_percentage_stop,
        )
        return grid.astype(np.uint8), growth_order

    grid, _ = simulate_dla_mc_numba(
        N=N,
        seed=seed,
        target_occupancy=target_occupancy,
        ps=ps,
        max_steps_per_walker=max_steps_per_walker,
        top_boundary_percentage_stop=top_boundary_percentage_stop,
    )
    return grid.astype(np.uint8)