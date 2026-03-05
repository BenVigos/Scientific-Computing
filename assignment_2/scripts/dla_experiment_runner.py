import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, os.path.abspath("../.."))

from src import dla_metrics as M
from src.dla_adapters import simulate_mc_dla


def compute_metrics(grid, *, seed_center_xy):
    """
    Compute a common metric set on a standardized grid.
    seed_center_xy: (x0, y0) where y0=0 for your unified convention.
    """
    bounds = M.bounding_box_metrics_seed_bottom(grid, seed_y0=seed_center_xy[1])
    fract = M.box_counting_dimension(grid)
    sk = M.skeleton_branch_stats(grid)

    occ_n = M.occupied_pixels(grid)
    perim = M.perimeter_length(grid)

    return {
        "occupied_pixels": occ_n,
        "occupancy": M.occupancy_fraction(grid),
        "bbox_density": M.bbox_density(grid),

        "perimeter": perim,
        "perimeter_per_occupied": (perim / occ_n) if occ_n > 0 else np.nan,

        "R_g": M.radius_of_gyration(grid, center=seed_center_xy),
        "D_est": fract["D"],
        "D_r": fract["r"],

        "max_width": bounds["max_width"],
        "height": bounds["height"],
        "aspect_ratio": bounds["aspect_ratio"],

        "top_row_fraction": M.top_row_fraction(grid),
        "touches_top": int(M.touches_top(grid)),

        **sk,
    }


def run_experiments(
    *,
    simulate_fn,
    param_name,
    param_values,
    seeds_per_value=20,
    out_csv,
    seed_center_xy,
    static_params=None,
    show_progress=True,
):
    """
    Unified experiment runner.

    simulate_fn: function(**kwargs)->grid (uint8), standardized y=0 bottom
    param_name: "ita" or "ps" (or any string)
    param_values: iterable of parameter values to sweep
    seeds_per_value: runs per parameter value
    out_csv: path to save results CSV
    seed_center_xy: (x0, y0) seed in unified coordinates (usually (N//2, 0))
    static_params: dict of other kwargs passed into simulate_fn (e.g. N, steps, ...)
    """
    static_params = static_params or {}
    results = []

    outer = tqdm(param_values, desc=f"sweep {param_name}") if show_progress else param_values

    for val in outer:
        for seed in range(seeds_per_value):
            kwargs = dict(static_params)
            kwargs[param_name] = val
            kwargs["seed"] = seed

            grid = simulate_fn(**kwargs)

            row = {
                "seed": int(seed),
                param_name: float(val),
                **{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in static_params.items() if k != "seed"},
            }
            row.update(compute_metrics(grid, seed_center_xy=seed_center_xy))
            results.append(row)

    df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Saved CSV to:", os.path.abspath(out_csv))

    return df


