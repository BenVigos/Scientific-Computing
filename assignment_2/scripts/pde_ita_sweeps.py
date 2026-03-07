import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("../.."))

from src.dla_adapters import simulate_pde_dla
from dla_experiment_runner import run_experiments


def omega_from_ita(ita):
    """Linear fit from heatmap."""
    omega = 1.7 + 0.1 * ita
    return min(max(omega, 1.7), 1.9)


def main():
    out_dir = os.path.join(os.getcwd(), "..", "data", "dla")
    os.makedirs(out_dir, exist_ok=True)

    N = 100
    seed_center = (N // 2, 0)

    ita_vals = np.arange(0.0, 2.2, 0.2)

    dfs = []

    for ita in ita_vals:
        omega = omega_from_ita(ita)
        print(f"Running ita={ita:.1f}, omega={omega:.2f}")

        df = run_experiments(
            simulate_fn=simulate_pde_dla,
            param_name="ita",
            param_values=[ita],   # run one ita at a time
            seeds_per_value=25,
            out_csv=os.path.join(out_dir, f"_tmp_ita_{ita:.1f}.csv"),
            seed_center_xy=seed_center,
            static_params={
                "N": N,
                "steps": 2000,
                "stop_threshold": 0.10,
                "debug": False,
                "omega": omega,
            },
            show_progress=False,
        )
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)
    final_path = os.path.join(out_dir, "pde_ita_metrics.csv")
    final_df.to_csv(final_path, index=False)

    # optional: remove temporary files
    for ita in ita_vals:
        tmp_path = os.path.join(out_dir, f"_tmp_ita_{ita:.1f}.csv")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print("Saved final CSV to:", final_path)


if __name__ == "__main__":
    main()