import numpy as np
from src.iter_schemes import sor
from tqdm import tqdm


def select_stick_cell(probabilities):
    flat_probabilities = probabilities.flatten()
    if flat_probabilities.sum() == 0:
        assert False, "No valid cells to stick to (all probabilities are zero). Check the concentration field and growth boundary."
        return
    selection = np.random.choice(len(flat_probabilities), p=flat_probabilities)
    row, col = np.unravel_index(selection, probabilities.shape)
    return row, col

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
    diffusion_grid, _, _, _ = sor(len(grid),omega= 1.9, insulator=grid, init_grid=diffusion_grid)

    #compute sticking probabilities at growth boundary
    neighbours = outer_neighbors(grid)
    neighbour_concentrations = neighbours  * diffusion_grid

    if debug:
        print("Neighbor concentrations at growth boundary:")
        print(neighbour_concentrations)

    probabilities = compute_stick_prob(neighbour_concentrations, ita=ita)
    selection = select_stick_cell(probabilities)

    grid[selection] = 1

    return grid, diffusion_grid

def diffusion_limited_aggregation(grid_size: tuple[int, int], steps: int = 10000, stop_threshold: float = 0.5, debug: bool = False, ita: float = 1):
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
    seed_x = grid_size[0] // 2
    seed_y = grid_size[1] - 1
    grid[seed_y, seed_x] = 1

    for i in tqdm(range(steps)):
        if debug:
            print(f"Step {i+1}/{steps} ...")

        grid, diffusion_grid = dla_step(grid, diffusion_grid, debug=debug, ita = ita)

        #check stopping condition
        occupied_percentage = np.mean(grid)
        if occupied_percentage >= stop_threshold:
            print(f"Stopping simulation at step {i+1} due to reaching stop threshold ({occupied_percentage:.2%} occupied).")
            break
    return grid

def compute_stick_prob(concentration_field, ita = 1):
    """Compute the sticking probability for a particle at position (x, y) based on neighboring occupied sites."""
    concentration_field_exp = np.power(concentration_field, ita)
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
    left  = np.pad(A[:, :-1], ((0,0),(1,0)))
    right = np.pad(A[:, 1: ], ((0,0),(0,1)))

    neighbor_has_one = up | down | left | right

    B = neighbor_has_one & (~A)
    return B.astype(int)


if __name__ == '__main__':
    grid_size = (5, 5)
    steps = 3
    stop_threshold = 0.5
    debug = True

    final_grid = diffusion_limited_aggregation(grid_size, steps, stop_threshold, debug)
    print("Final DLA cluster:")
    print(final_grid)
