from src.diffusion_limited_aggregation import diffusion_limited_aggregation as dla
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    np.random.seed(42)
    N = 100
    grid_size = (N, N)
    steps = 1000
    stop_threshold = 0.5
    debug = False
    ita = 3
    # BUG: when ita is 0 the cluster is disconnected
    final_grid = dla(grid_size, steps, stop_threshold, debug, ita=ita)

    plt.imshow(final_grid, cmap='viridis', origin='lower')
    plt.colorbar(label='Occupied (1) vs Empty (0)')
    plt.title('Final DLA Cluster')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()