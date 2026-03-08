import numpy as np
from src.iter_schemes import sor_numba, sor_numba_redblack
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import time

def diffusion_limited_aggregation(grid_size: tuple[int, int], steps: int = 1000, stop_threshold: float = 0.5, debug: bool = False, ita: float = 1, omega: float = 1.9, return_growth_order=False, parallel: bool = False):
    """Run a DLA simulation on a grid of given size with a specified number of particles.\n
    The process is as follows:\n
    1. Initialize an empty grid and place a seed particle at the bottom of the computational domain.\n
    2. For each step, perform a DLA step to grow the cluster.\n
    3. Stop the simulation if the percentage of occupied cells in the grid exceeds the stop_threshold or if the specified number of steps is reached.

    :param omega: over-relaxation parameter for the SOR solver (between 1 and 2, typically around 1.9 for optimal convergence).
    :param return_growth_order: boolean flag to indicate whether to return the growth order of the cluster (the order in which cells were occupied) along with the final grid.
    :param parallel: boolean flag to indicate whether to use the parallel version of the SOR solver for computing the diffusion field.
    :param grid_size: tuple (nx, ny) specifying the size of the grid.
    :param steps: number of particles to add to the cluster.
    :param stop_threshold: threshold for stopping the simulation based on the percentage of occupied cells in the grid (between 0 and 1).
    :param debug: boolean flag to print debug information during the simulation.
    :param ita: probability exponent
    :return: 2D numpy array representing the final state of the DLA cluster
    """
    #initialize grid
    grid = np.zeros(grid_size, dtype=int)
    growth_order = np.zeros(grid_size, dtype=int)
    diffusion_grid = np.zeros(grid_size, dtype=float)
    diffusion_grid[0, :] = 1  # set top boundary to concentration 1

    #place seed particle at the bottom center
    seed_x = grid_size[1] // 2
    seed_y = grid_size[0] - 1
    grid[seed_y, seed_x] = 1
    growth_order[seed_y, seed_x] = 1
    occupied = 1

    for i in range(steps):
        if debug:
            print(f"Step {i+1}/{steps} ...")

        grid, diffusion_grid, growth_order, occupied, keep_running = dla_step(
            grid, diffusion_grid, growth_order, occupied,debug=debug, ita=ita, omega=omega, parallel=parallel)

        #check stopping condition
        occupied_percentage = np.mean(grid)
        if occupied_percentage >= stop_threshold or not keep_running:
            # print(f"Stopping simulation at step {i+1} due to reaching stop threshold ({occupied_percentage:.2%} occupied).")
            break
    if return_growth_order:
        return grid, growth_order
    return grid

def dla_step(grid, prev_diffusion_grid, growth_order, occupied, debug=False, ita = 1, omega = 1.9, parallel=False):
    """Perform one step of diffusion-limited aggregation (DLA) on the grid.\n
    One step consists of:\n
    1. Solve the time-independent diffusion equation (Laplace's equation) to get the concentration field.\n
    2. Compute the sticking probability of each cell at the growth boundary based on the concentration field.\n
    3. Randomly select a cell at the growth boundary and add it to the cluster.

    :param parallel: boolean flag to indicate whether to use the parallel version of the SOR solver for computing the diffusion field.
    :param omega: over-relaxation parameter for the SOR solver (between 1 and 2, typically around 1.9 for optimal convergence).
    :param occupied: number of occupied cells in the cluster so far (used to determine growth order).
    :param growth_order: 2D numpy array that keeps track of the order in which cells were occupied (0 for unoccupied, 1 for the seed, 2 for the first added cell, etc.).
    :param prev_diffusion_grid: 2D numpy array representing the diffusion field from the previous step, which can be used as an initial guess for the SOR solver to speed up convergence.
    :param grid: 2D numpy array representing the current state of the DLA cluster (1 for occupied, 0 for empty).
    :param debug: Boolean flag to enable debugging.
    :param ita: probability exponent
    """
    insulator_mask = np.zeros(grid.shape, dtype=np.bool_)
    if parallel:
        diffusion_grid, _, _ = sor_numba_redblack(
            np.int64(len(grid)), omega=omega, c=prev_diffusion_grid,
            sink=grid, insulator=insulator_mask, max_iter=100000, tol=1e-5
        )
    else:
        diffusion_grid, _, _ = sor_numba(
            np.int64(len(grid)), omega=omega, c=prev_diffusion_grid,
            sink=grid, insulator=insulator_mask, max_iter=100000, tol=1e-5
        )

    #compute sticking probabilities at growth boundary
    neighbours = outer_neighbors(grid)
    neighbour_concentrations = neighbours  * diffusion_grid

    if debug:
        print("Neighbor concentrations at growth boundary:")
        print(neighbour_concentrations)

    probabilities = compute_stick_prob(neighbour_concentrations, neighbours, ita=ita)
    selection = select_stick_cell(probabilities)
    if selection is None:
        print("No valid cells to stick to. Ending simulation.")
        return grid, diffusion_grid, growth_order, occupied, False # return False to indicate simulation should end
    grid[selection] = 1
    occupied += 1
    growth_order[selection] = occupied

    return grid, diffusion_grid, growth_order, occupied, True

def select_stick_cell(probabilities):
    """
    Randomly select a cell to stick to based on the given probabilities.

    :param probabilities: 2D numpy array of probabilities for each cell at the growth boundary.
    :return: tuple (row, col) of the selected cell, or None if no valid cells are available.
    """
    flat_probabilities = probabilities.flatten()
    if flat_probabilities.sum() == 0:
        # assert False, "No valid cells to stick to (all probabilities are zero). Check the concentration field and growth boundary."
        return None
    selection = np.random.choice(len(flat_probabilities), p=flat_probabilities)
    row, col = np.unravel_index(selection, probabilities.shape)
    return row, col

def compute_stick_prob(concentration_field, neighbours, ita = 1, eps = 1e-6):
    """
    Compute the sticking probability for each cell at the growth boundary based on the concentration field and the presence of neighbors.

    :param concentration_field: 2D numpy array representing the concentration field at the growth boundary.
    :param neighbours: 2D boolean numpy array indicating which cells are at the growth boundary (True for cells that are neighbors to the cluster).
    :param ita: exponent for the sticking probability (default is 1, which corresponds to linear dependence on concentration). Higher values of ita will make the sticking probability more sensitive to concentration differences.
    :return: 2D numpy array of sticking probabilities for each cell at the growth boundary, normalized to sum to 1.
    """
    neighbour_mask = neighbours
    concentration_clip = np.where(concentration_field < eps, 0.0, concentration_field)
    concentration_field_exp = np.zeros_like(concentration_field)
    concentration_field_exp[neighbour_mask] = np.power(concentration_clip[neighbour_mask], ita)

    concentration_sum = np.sum(concentration_field_exp)
    prob = concentration_field_exp/concentration_sum if concentration_sum > 0 else np.zeros_like(concentration_field)
    return prob

def outer_neighbors(A):
    """
    Return a boolean matrix where True indicates cells that are adjacent to at least one occupied cell in A, but are not occupied themselves.

    :param A: the input 2D array (boolean or integer) where non-zero/True values indicate occupied cells.
    :return: matrix of the same shape as A where True indicates cells that are adjacent to at least one occupied cell in A, but are not occupied themselves.
    """

    A = A.astype(bool)

    up    = np.pad(A[:-1, :], ((1,0),(0,0)))
    down  = np.pad(A[1:,  :], ((0,1),(0,0)))

    left  = np.roll(A, shift=1, axis=1)
    right = np.roll(A, shift=-1, axis=1)

    neighbor_has_one = up | down | left | right
    B = neighbor_has_one & (~A)
    return B




if __name__ == '__main__':
    import time
    N = 300
    grid_size = (N, N)
    steps = 10000
    stop_threshold = 0.1
    debug = False
    ita = 1.5

    ts = time.perf_counter()
    final_grid = diffusion_limited_aggregation(grid_size, steps, stop_threshold, debug, ita=ita, parallel=True)
    te = time.perf_counter()
    print(f"DLA simulation completed in {te - ts:.2f} seconds.")
    print("Final DLA cluster:")
    skel = skeletonize(final_grid > 0)
    plt.imshow(skel)
    plt.title(f"DLA Cluster Skeleton (ita={ita})")
    plt.show()
    plt.imshow(final_grid)
    plt.title(f"DLA Cluster (ita={ita})")
    plt.show()