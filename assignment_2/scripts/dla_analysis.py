"""
Batch experiment runner for DLA parameter sweep over `ita`.
Outputs per-run metrics and simple summary CSV.

Functions:
- radius_of_gyration
- radial_mass_scaling (estimate fractal D by log-log fit)
- perimeter_length (simple 4-neighbor boundary count)
- skeleton_branch_stats (requires scikit-image; graceful fallback)
- run_experiments: runs dla for each (ita, seed) and collects metrics
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from math import sqrt, log
from scipy import stats

# Import your DLA function; adjust path if necessary
from src.diffusion_limited_aggregation import diffusion_limited_aggregation as dla

# optional skimage utilities
try:
    from skimage.morphology import skeletonize
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

def radius_of_gyration(grid, center=None):
    ys, xs = np.nonzero(grid)
    if len(xs) == 0:
        return 0.0
    if center is None:
        cx = xs.mean()
        cy = ys.mean()
    else:
        cx, cy = center
    dx = xs - cx
    dy = ys - cy
    return sqrt(((dx*dx + dy*dy).mean()))

def perimeter_length(grid):
    occ = (grid > 0).astype(np.uint8)
    # 4-neighbor boundary: occupied pixel having at least one empty 4-neighbor
    up = np.pad(occ, ((1,0),(0,0)), mode='constant')[:-1,:]
    down = np.pad(occ, ((0,1),(0,0)), mode='constant')[1:,:]
    left = np.pad(occ, ((0,0),(1,0)), mode='constant')[:, :-1]
    right = np.pad(occ, ((0,0),(0,1)), mode='constant')[:, 1:]
    neighbor_sum = up + down + left + right
    boundary = (occ == 1) & (neighbor_sum < 4)
    return int(boundary.sum())

def radial_mass_scaling(grid, center=None, n_bins=30, fit_range=(0.1, 0.8)):
    ys, xs = np.nonzero(grid)
    if len(xs) == 0:
        return {"D": np.nan, "intercept": np.nan, "r": np.nan}
    if center is None:
        cx = xs.mean()
        cy = ys.mean()
    else:
        cx, cy = center
    rs = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    r_max = rs.max()
    if r_max <= 0:
        return {"D": np.nan, "intercept": np.nan, "r": np.nan}
    radii = np.linspace(1e-6, r_max, n_bins)
    mass = np.array([np.sum(rs <= rr) for rr in radii])
    # choose fitting window between fit_range fractions of r_max
    mask = (radii >= fit_range[0]*r_max) & (radii <= fit_range[1]*r_max) & (mass > 0)
    if mask.sum() < 3:
        return {"D": np.nan, "intercept": np.nan, "r": np.nan}
    logr = np.log(radii[mask])
    logm = np.log(mass[mask])
    slope, intercept, r_val, p_val, se = stats.linregress(logr, logm)
    # slope is fractal dimension D
    return {"D": float(slope), "intercept": float(intercept), "r": float(r_val)}

def occupancy_fraction(grid):
    return float((grid > 0).sum()) / grid.size

def skeleton_branch_stats(grid):
    if not SKIMAGE_AVAILABLE:
        return {"endpoints": np.nan, "branchpoints": np.nan, "skeleton_length": np.nan}
    occ = (grid > 0)
    sk = skeletonize(occ)
    sk_u8 = sk.astype(np.uint8)
    # neighbor count on skeleton (8-neighbors)
    kernel_coords = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    neigh_count = np.zeros_like(sk_u8, dtype=np.int32)
    rows, cols = sk_u8.shape
    for dy, dx in kernel_coords:
        shifted = np.zeros_like(sk_u8)
        ys_src = max(0, -dy), rows - max(0, dy)
        xs_src = max(0, -dx), cols - max(0, dx)
        ys_dst = max(0, dy), rows - max(0, -dy)
        xs_dst = max(0, dx), cols - max(0, -dx)
        shifted[ys_dst[0]:ys_dst[1], xs_dst[0]:xs_dst[1]] = sk_u8[ys_src[0]:ys_src[1], xs_src[0]:xs_src[1]]
        neigh_count += shifted
    endpoints = np.logical_and(sk, neigh_count == 1).sum()
    branchpoints = np.logical_and(sk, neigh_count > 2).sum()
    skeleton_length = sk.sum()
    return {"endpoints": int(endpoints), "branchpoints": int(branchpoints), "skeleton_length": int(skeleton_length)}

def run_experiments(ita_values, seeds_per_ita=20, grid_size=(100,100), steps=1000, out_csv="dla_metrics.csv", debug=False):
    results = []
    for ita in ita_values:
        for seed in range(seeds_per_ita):
            np.random.seed(seed)
            grid = dla(grid_size, steps, 0.5, debug, ita=ita)
            # assume seed is bottom-center; use center approx:
            center = (grid.shape[1]//2, grid.shape[0]-1)  # (x,y) if needed for other metrics
            # compute metrics
            rg = radius_of_gyration(grid, center=None)
            mass_scaling = radial_mass_scaling(grid, center=None, n_bins=40, fit_range=(0.15, 0.75))
            occ = occupancy_fraction(grid)
            perim = perimeter_length(grid)
            sk_stats = skeleton_branch_stats(grid)
            row = {
                "ita": ita,
                "seed": seed,
                "R_g": rg,
                "D_est": mass_scaling["D"],
                "D_r": mass_scaling["r"],
                "occupancy": occ,
                "perimeter": perim,
                "endpoints": sk_stats["endpoints"],
                "branchpoints": sk_stats["branchpoints"],
                "skeleton_length": sk_stats["skeleton_length"]
            }
            results.append(row)
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print("Saved CSV to:", os.path.abspath(out_csv))
    return df

if __name__ == "__main__":
    N = 10
    out_dir = os.path.join(os.getcwd(), "..", "data", "dla")
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "dla_metrics.csv")


    ita_vals = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    print("Running DLA experiments")
    df = run_experiments(ita_vals, seeds_per_ita=20, grid_size=(N,N), steps=1000, out_csv=out_csv, debug=False)
    print(df.groupby("ita").agg({"D_est":["mean","std","count"], "R_g":["mean","std"], "occupancy":["mean","std"]}))
