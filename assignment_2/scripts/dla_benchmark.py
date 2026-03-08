import time
from logging.config import stopListening
from pathlib import Path

import numpy as np
from numba.np.ufunc.workqueue import parallel_for
import matplotlib.pyplot as plt

def benchmark(function, n_runs, *args, **kwargs):

    function(*args, **kwargs)

    times = []

    for _ in range(n_runs):
        t_start = time.perf_counter()

        function(*args, **kwargs)

        t_end = time.perf_counter()
        times.append(t_end - t_start)

    avg = np.mean(times)
    std = np.std(times)

    return avg, std, times

if __name__ == '__main__':
    from src.diffusion_limited_aggregation import diffusion_limited_aggregation
    from src.grid import make_grid
    from src.iter_schemes import sor_numba, sor_numba_redblack

    Ns = [25, 50, 75, 100, 125, 150, 175, 200,225, 250]
    steps = 1000
    occupancy_threshold = 0.1
    ita = 1.0
    omega = 1.9
    n_runs = 10
    normal_times = []
    parallel_times = []
    for N in Ns:
        omega = 1.8
        print(f"N = {N}")
        grid_size = (N, N)

        #benchmark normal SOR
        avg_time, std_time, times = benchmark(diffusion_limited_aggregation, n_runs, grid_size=grid_size, steps=steps, stop_threshold=occupancy_threshold, debug=False, ita=ita, parallel=False, omega=omega)
        normal_times.append((avg_time, std_time))
        print(f'Normal SOR: {avg_time:.4f} ± {std_time:.4f} seconds')

        #benchmark parallel SOR
        avg_time, std_time, times = benchmark(diffusion_limited_aggregation, n_runs, grid_size=grid_size, steps=steps, stop_threshold=occupancy_threshold, debug=False, ita=ita, parallel=True, omega=omega)
        parallel_times.append((avg_time, std_time))
        print(f'Parallel SOR: {avg_time:.4f} ± {std_time:.4f} seconds')

    # Create output directories
    data_dir = Path('../data/dla')
    figures_dir = data_dir / 'figures'
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save benchmark data to CSV
    csv_path = data_dir / 'dla_benchmark_results.csv'
    with open(csv_path, 'w') as f:
        f.write('N,normal_avg,normal_std,parallel_avg,parallel_std\n')
        for i, N in enumerate(Ns):
            f.write(f'{N},{normal_times[i][0]},{normal_times[i][1]},{parallel_times[i][0]},{parallel_times[i][1]}\n')
    print(f'\nBenchmark data saved to {csv_path}')

    # plotting results
    plt.figure(figsize=(10, 6))
    plt.errorbar(Ns, [t[0] for t in normal_times], yerr=[t[1] for t in normal_times], label='Normal SOR', fmt='-o')
    plt.errorbar(Ns, [t[0] for t in parallel_times], yerr=[t[1] for t in parallel_times], label='Parallel SOR', fmt='-o')
    plt.xlabel('Grid Size (N x N)')
    plt.ylabel('Average Time (seconds)')
    plt.legend()
    plt.title('SOR Performance Comparison: Normal vs Parallel')
    plt.grid(True, alpha=0.3)

    # Save figure
    fig_path = figures_dir / 'dla_benchmark_comparison.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f'Figure saved to {fig_path}')

    plt.show()