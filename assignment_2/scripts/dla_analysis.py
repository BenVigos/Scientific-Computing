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
from math import sqrt
from scipy import stats

# Import your DLA function; adjust path if necessary
from src.diffusion_limited_aggregation import diffusion_limited_aggregation as dla

# optional skimage utilities
try:
    from skimage.morphology import skeletonize
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False

def box_counting_dimension(grid, box_sizes=None, fit_range=(0.1, 0.9)):
    """
    Use box-counting method to estimate the fractal dimension of the occupied cluster in the grid.
    This version first crops the grid to the minimal bounding rectangle containing the cluster
    (min/max in x and y) so the box-counting focuses on the cluster extents rather than empty
    padding around it.

    :param grid: 2D numpy array representing the DLA cluster (1 for occupied, 0 for empty)
    :param box_sizes: sizes of boxes to use for counting (if None, will use powers of 2 up to the largest box that fits in the cropped region)
    :param fit_range: tuple (min_frac, max_frac) specifying the relative box size range (as a fraction of the largest box) to use for fitting the line to estimate D.
    :return: dict with keys "D" (estimated fractal dimension), "intercept" (y-intercept of the fit), and "r" (correlation coefficient of the fit)
    """
    occ_full = (grid > 0).astype(np.uint8)

    # find bounding box of occupied pixels
    ys, xs = np.nonzero(occ_full)
    if len(xs) == 0:
        # empty cluster
        return {"D": np.nan, "intercept": np.nan, "r": np.nan}

    min_y, max_y = ys.min(), ys.max()
    min_x, max_x = xs.min(), xs.max()

    # crop to bounding box (inclusive)
    occ = occ_full[min_y:max_y + 1, min_x:max_x + 1]
    rows, cols = occ.shape

    max_box = min(rows, cols)
    if max_box <= 0:
        return {"D": np.nan, "intercept": np.nan, "r": np.nan}

    if box_sizes is None:
        max_pow = int(np.floor(np.log2(max_box)))
        box_sizes = np.array([2 ** k for k in range(0, max_pow + 1)], dtype=int)
    else:
        box_sizes = np.array(box_sizes, dtype=int)
        box_sizes = box_sizes[(box_sizes >= 1) & (box_sizes <= max_box)]

    if box_sizes.size == 0:
        return {"D": np.nan, "intercept": np.nan, "r": np.nan}

    counts = []
    for s in box_sizes:
        cnt = 0
        # tile over the cropped region
        for y in range(0, rows, s):
            for x in range(0, cols, s):
                if occ[y:y + s, x:x + s].any():
                    cnt += 1
        counts.append(cnt)
    counts = np.array(counts, dtype=float)

    valid = counts > 0
    if valid.sum() < 3:
        return {"D": np.nan, "intercept": np.nan, "r": np.nan}

    # select fit window based on relative box size (relative to largest box)
    rel_size = box_sizes.astype(float) / box_sizes.max()
    mask = valid & (rel_size >= fit_range[0]) & (rel_size <= fit_range[1])

    # if too few points in requested window, fall back to all valid sizes
    if mask.sum() < 3:
        mask = valid
        if mask.sum() < 3:
            return {"D": np.nan, "intercept": np.nan, "r": np.nan}

    x = -np.log(box_sizes[mask].astype(float))   # log(1/epsilon)
    y = np.log(counts[mask])

    slope, intercept, r_val, p_val, se = stats.linregress(x, y)
    return {"D": float(slope), "intercept": float(intercept), "r": float(r_val)}


def radius_of_gyration(grid, center=None):
    """
    Compute the radius of gyration of the occupied cluster in the grid. If center is not provided, use the mean position of occupied pixels as the center.
    :param grid: 2D numpy array representing the DLA cluster (1 for occupied, 0 for empty)
    :param center: tuple (cx, cy) specifying the center of mass; if None, it will be computed from the occupied pixels
    :return: the radius of gyration (float)
    """
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

def bounding_box_metrics(grid):
    """
Compute bounding box metrics: max width (max horizontal span of occupied pixels) and height (measured from bottom seed to topmost occupied pixel). Aspect ratio is width/height.
    :param grid: 2D numpy array representing the DLA cluster (1 for occupied, 0 for empty)
    :return: dict with keys "max_width", "height", "aspect_ratio"
    """
    ys, xs = np.nonzero(grid)
    if len(xs) == 0:
        return {"max_width": 0, "height": 0, "aspect_ratio": np.nan}
    min_x, max_x = xs.min(), xs.max()
    # width in pixels
    width = int(max_x - min_x + 1)
    # height measured from bottom seed (y = rows-1) to topmost occupied pixel
    rows = grid.shape[0]
    top_y = int(ys.min())
    height_from_seed = int((rows - 1) - top_y + 1)
    aspect = float(width) / float(height_from_seed) if height_from_seed > 0 else np.nan
    return {"max_width": width, "height": height_from_seed, "aspect_ratio": aspect}


def run_experiments(ita_values, seeds_per_ita=20, grid_size=(100,100), steps=1000, out_csv="dla_metrics.csv", debug=False):
    results = []
    for ita in ita_values:
        for seed in range(seeds_per_ita):
            np.random.seed(seed)
            grid = dla(grid_size, steps, 0.1, debug, ita=ita)
            # define seed center (bottom-center) as (x,y)
            seed_center = (grid.shape[1] // 2, grid.shape[0] - 1)

            # radius of gyration and radial mass scaling using the seed center
            rg = radius_of_gyration(grid, center=seed_center)
            fractal_dim = box_counting_dimension(grid)

            # bounding / aspect metrics (max width and height measured from seed)
            bounds = bounding_box_metrics(grid)

            # occupancy, perimeter, skeleton-based topology
            occ = occupancy_fraction(grid)
            perim = perimeter_length(grid)
            sk_stats = skeleton_branch_stats(grid)

            row = {
                "ita": ita,
                "seed": seed,
                "R_g": rg,
                "D_est": fractal_dim["D"],
                "D_r": fractal_dim["r"],
                "occupancy": occ,
                "perimeter": perim,
                "max_width": bounds["max_width"],
                "height": bounds["height"],
                "aspect_ratio": bounds["aspect_ratio"],
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
    N = 20
    out_dir = os.path.join(os.getcwd(), "..", "data", "dla")
    os.makedirs(out_dir, exist_ok=True)

    occ = 0.2
    out_csv = os.path.join(out_dir, "dla_metrics.csv")


    ita_vals = np.linspace(0, 1.4, 7)
    print("Running DLA experiments")
    df = run_experiments(ita_vals, seeds_per_ita=20, grid_size=(N,N), steps=1000, out_csv=out_csv, debug=False)
    print(df.groupby("ita").agg({"D_est":["mean","std","count"], "R_g":["mean","std"], "occupancy":["mean","std"]}))
