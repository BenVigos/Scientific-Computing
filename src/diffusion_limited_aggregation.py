import numpy as np
from src.iter_schemes import sor, sor_numba
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

def diffusion_limited_aggregation(grid_size: tuple[int, int], steps: int = 1000, stop_threshold: float = 0.5, debug: bool = False, ita: float = 1):
    """Run a DLA simulation on a grid of given size with a specified number of particles.\n
    The process is as follows:\n
    1. Initialize an empty grid and place a seed particle at the bottom of the computational domain.\n
    2. For each step, perform a DLA step to grow the cluster.\n
    3. Stop the simulation if the percentage of occupied cells in the grid exceeds the stop_threshold or if the specified number of steps is reached.

    :param grid_size: tuple (nx, ny) specifying the size of the grid.
    :param steps: number of particles to add to the cluster.
    :param stop_threshold: threshold for stopping the simulation based on the percentage of occupied cells in the grid (between 0 and 1).
    :param debug: boolean flag to print debug information during the simulation.
    :param ita: probability exponent
    :return: 2D numpy array representing the final state of the DLA cluster
    """
    #initialize grid
    grid = np.zeros(grid_size, dtype=int)
    diffusion_grid = np.zeros(grid_size, dtype=float)
    diffusion_grid[0, :] = 1  # set top boundary to concentration 1

    #place seed particle at the bottom center
    seed_x = grid_size[1] // 2
    seed_y = grid_size[0] - 1
    grid[seed_y, seed_x] = 1

    for i in range(steps):
        if debug:
            print(f"Step {i+1}/{steps} ...")

        grid, diffusion_grid = dla_step(grid, diffusion_grid, debug=debug, ita = ita)

        #check stopping condition
        occupied_percentage = np.mean(grid)
        if occupied_percentage >= stop_threshold:
            # print(f"Stopping simulation at step {i+1} due to reaching stop threshold ({occupied_percentage:.2%} occupied).")
            break
    return grid

def dla_step(grid, diffusion_grid, debug=False, ita = 1):
    """Perform one step of diffusion-limited aggregation (DLA) on the grid.\n
    One step consists of:\n
    1. Solve the time-independent diffusion equation (Laplace's equation) to get the concentration field.\n
    2. Compute the sticking probability of each cell at the growth boundary based on the concentration field.\n
    3. Randomly select a cell at the growth boundary and add it to the cluster.

    :param grid: 2D numpy array representing the current state of the DLA cluster (1 for occupied, 0 for empty).
    :param debug: Boolean flag to enable debugging.
    :param ita: probability exponent
    """
    insulator_mask = np.zeros(grid.shape, dtype=np.bool_)
    diffusion_grid, _, _  = sor_numba(len(grid),omega= 1.9, c=diffusion_grid, sink=grid, insulator=insulator_mask, max_iter=100000, tol=1e-5)

    #compute sticking probabilities at growth boundary
    neighbours = outer_neighbors(grid)
    neighbour_concentrations = neighbours  * diffusion_grid

    if debug:
        print("Neighbor concentrations at growth boundary:")
        print(neighbour_concentrations)

    probabilities = compute_stick_prob(neighbour_concentrations, neighbours, ita=ita)
    selection = select_stick_cell(probabilities)
    if selection is None:
        print("No valid cells to stick to. Skipping this step.")
        return grid, diffusion_grid
    grid[selection] = 1

    return grid, diffusion_grid

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

def compute_stick_prob(concentration_field, neighbours, ita = 1):
    """
    Compute the sticking probability for each cell at the growth boundary based on the concentration field and the presence of neighbors.

    :param concentration_field: 2D numpy array representing the concentration field at the growth boundary.
    :param neighbours: 2D boolean numpy array indicating which cells are at the growth boundary (True for cells that are neighbors to the cluster).
    :param ita: exponent for the sticking probability (default is 1, which corresponds to linear dependence on concentration). Higher values of ita will make the sticking probability more sensitive to concentration differences.
    :return: 2D numpy array of sticking probabilities for each cell at the growth boundary, normalized to sum to 1.
    """
    neighbour_mask = neighbours
    concentration_clip = np.clip(concentration_field, a_min=0, a_max=None)  # ensure non-negative concentrations
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
    N = 100
    grid_size = (N, N)
    steps = 1000
    stop_threshold = 0.1
    debug = False
    ita = 2

    final_grid = diffusion_limited_aggregation(grid_size, steps, stop_threshold, debug, ita=ita)
    print("Final DLA cluster:")
    skel = skeletonize(final_grid > 0)
    plt.imshow(skel)
    plt.title(f"DLA Cluster Skeleton (ita={ita})")
    plt.show()
    plt.imshow(final_grid)
    plt.title(f"DLA Cluster (ita={ita})")
    plt.show()