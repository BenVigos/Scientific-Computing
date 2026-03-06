import numpy as np
from numba import njit

@njit
def _has_neighbor(cluster, x, y):
    """ Check if neighboring site is occupied, with periodic boundary conditions in x.
        Parameters:
        cluster: 2D boolean array representing the cluster
        x, y: coordinates of the site to check"""
    N = cluster.shape[0]
    #  periodic in x
    xm = x - 1
    if xm < 0: xm += N
    xp = x + 1
    if xp >= N: xp -= N

    if y > 0 and cluster[y - 1, x]: return True
    if y < N - 1 and cluster[y + 1, x]: return True
    if cluster[y, xm]: return True
    if cluster[y, xp]: return True
    return False

@njit
def simulate_dla_mc_numba(
    N,
    seed,
    target_occupancy,
    ps,
    max_steps_per_walker,
    top_boundary_percentage_stop
):
    """ Simulate DLA using a Monte Carlo method.
    Parameters:
    N: size of the grid (NxN) 
    seed: random seed for reproducibility
    target_occupancy: fraction of sites to be occupied before stopping
    ps: sticking probability (0 to 1)
    max_steps_per_walker: maximum number of steps a walker can take before being killed and restarted,
    top_boundary_percentage_stop: percentage of the top boundary that once reached, stops the simulation (0 to 1)"""
    
    np.random.seed(seed)

    cluster = np.zeros((N, N), dtype=np.bool_)
    growth_order = np.zeros((N, N), dtype=np.int32) 
    # set seed
    y0 = 0
    x0 = N // 2
    cluster[y0, x0] = True

    # for stopping
    occupied = 1
    target = int(np.ceil(target_occupancy * N * N))

    # top boundary tracking
    top_count = 0
    top_stop = top_boundary_percentage_stop > 0.0

    while occupied < target:
        x = np.random.randint(0, N)
        y = N - 1

        for _ in range(max_steps_per_walker):
            # pick an rng direcetion to move towards
            r = np.random.randint(0, 4)
            dx = 0; dy = 0
            if r == 0: dx = 1
            elif r == 1: dx = -1
            elif r == 2: dy = 1
            else: dy = -1

            # keep periodic
            xn = x + dx
            if xn < 0: xn += N
            elif xn >= N: xn -= N

            yn = y + dy

            # kill walker if ouyt of bounds on y axis
            if yn < 0 or yn >= N:
                x = np.random.randint(0, N)
                y = N - 1
                continue

            # if picked direction is occupied, try again
            if cluster[yn, xn]:
                continue

            x = xn; y = yn

            # sticking probability
            if _has_neighbor(cluster, x, y):
                if ps >= 1.0 or np.random.random() < ps:
                    cluster[y, x] = True
                    occupied += 1
                    growth_order[y, x] = occupied
                    
                    # update top row count + stopping condition
                    if top_stop and y == N - 1:
                        top_count += 1
                        if (top_count / N) >= top_boundary_percentage_stop:
                            return cluster, growth_order

                    break

    return cluster, growth_order


