import os, sys
import numpy as np

sys.path.insert(0, os.path.abspath("../.."))

from src.dla_adapters import simulate_mc_dla
from dla_experiment_runner import run_experiments


def main():
    out_dir = os.path.join(os.getcwd(), "..", "data", "dla_mc")
    os.makedirs(out_dir, exist_ok=True)

    N = 100
    seed_center = (N // 2, 0)

    ps_vals = [1.0, 0.5, 0.2, 0.1, 0.05, 0.001]
    run_experiments(
        simulate_fn=simulate_mc_dla,
        param_name="ps",
        param_values=ps_vals,
        seeds_per_value=25,
        out_csv=os.path.join(out_dir, "mc_ps_metrics.csv"),
        seed_center_xy=seed_center,
        static_params={
            "N": N,
            "target_occupancy": 0.10,
            "max_steps_per_walker": 100_000
        },
    )


if __name__ == "__main__":
    main()