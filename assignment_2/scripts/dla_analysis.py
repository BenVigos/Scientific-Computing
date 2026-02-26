from src.diffusion_limited_aggregation import diffusion_limited_aggregation as dla
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    grid_size = (100, 100)
    steps = 100
    stop_threshold = 0.1
    debug = False
    ita = 1.5

    final_grid = dla(grid_size, steps, stop_threshold, debug, ita=ita)

    plt.imshow(final_grid, cmap='viridis', origin='lower')
    plt.colorbar(label='Occupied (1) vs Empty (0)')
    plt.title('Final DLA Cluster')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()