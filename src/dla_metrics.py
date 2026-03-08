import numpy as np
from math import sqrt
from scipy import stats
from scipy.ndimage import convolve

# Optional skimage utilities
try:
    from skimage.morphology import skeletonize
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False


def occupancy_fraction(grid):
    """Fraction of occupied cells in the full grid."""
    occ = (grid > 0)
    return float(occ.mean())


def occupied_pixels(grid):
    return int((grid > 0).sum())


def bbox_density(grid):
    """
    Density inside the smallest axis-aligned bounding box enclosing the cluster:
    (#occupied in bbox) / (bbox area).
    """
    occ = (grid > 0)
    ys, xs = np.nonzero(occ)
    if xs.size == 0:
        return np.nan
    min_y, max_y = ys.min(), ys.max()
    min_x, max_x = xs.min(), xs.max()
    bbox = occ[min_y:max_y + 1, min_x:max_x + 1]
    return float(bbox.mean())


def radius_of_gyration(grid, center=None):
    """
    Radius of gyration of occupied pixels.
    center: (cx, cy) in (x, y). If None, uses COM of occupied pixels. # should stay as com or specifically from seed??
    """
    occ = (grid > 0)
    ys, xs = np.nonzero(occ)
    if xs.size == 0:
        return 0.0

    if center is None:
        cx = xs.mean()
        cy = ys.mean()
    else:
        cx, cy = center

    dx = xs - cx
    dy = ys - cy
    return float(sqrt(((dx * dx + dy * dy).mean())))


def perimeter_length(grid):
    """4-neighbor boundary count: occupied cell with at least one empty 4-neighbor."""
    occ = (grid > 0).astype(np.uint8)
    up = np.pad(occ, ((1, 0), (0, 0)), mode="constant")[:-1, :]
    down = np.pad(occ, ((0, 1), (0, 0)), mode="constant")[1:, :]
    left = np.pad(occ, ((0, 0), (1, 0)), mode="constant")[:, :-1]
    right = np.pad(occ, ((0, 0), (0, 1)), mode="constant")[:, 1:]
    neighbor_sum = up + down + left + right
    boundary = (occ == 1) & (neighbor_sum < 4)
    return int(boundary.sum())


def perimeter_per_occupied(grid):
    """Perimeter length divided by occupied pixel count (branchiness proxy)."""
    occ_n = occupied_pixels(grid)
    if occ_n == 0:
        return np.nan
    return float(perimeter_length(grid) / occ_n)


def bounding_box_metrics_seed_bottom(grid, seed_y0=0):
    """
    Width, height, aspect ratio of cluster bounding box.
    ASSUMES unified convention: y=0 at bottom (cartesian), y increases upward.
    Height is measured from seed_y0 to the highest occupied y (inclusive).
    """
    occ = (grid > 0)
    ys, xs = np.nonzero(occ)
    if xs.size == 0:
        return {"max_width": 0, "height": 0, "aspect_ratio": np.nan}

    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - seed_y0 + 1)
    aspect = float(width) / float(height) if height > 0 else np.nan
    return {"max_width": width, "height": height, "aspect_ratio": aspect}


def top_row_fraction(grid):
    """Fraction of occupied cells in the top row (y = N-1)."""
    occ = (grid > 0)
    return float(occ[-1, :].mean())


def touches_top(grid):
    occ = (grid > 0)
    return bool(occ[-1, :].any())


def box_counting_dimension(grid, box_sizes=None, fit_range=(0.1, 0.9)):
    """Box-counting fractal dimension estimate; crops to occupied bounding box."""
    occ_full = (grid > 0).astype(np.uint8)
    ys, xs = np.nonzero(occ_full)
    if xs.size == 0:
        return {"D": np.nan, "intercept": np.nan, "r": np.nan}

    min_y, max_y = ys.min(), ys.max()
    min_x, max_x = xs.min(), xs.max()
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
        for y in range(0, rows, s):
            for x in range(0, cols, s):
                if occ[y:y + s, x:x + s].any():
                    cnt += 1
        counts.append(cnt)
    counts = np.array(counts, dtype=float)

    valid = counts > 0
    if valid.sum() < 3:
        return {"D": np.nan, "intercept": np.nan, "r": np.nan}

    rel_size = box_sizes.astype(float) / box_sizes.max()
    mask = valid & (rel_size >= fit_range[0]) & (rel_size <= fit_range[1])
    if mask.sum() < 3:
        mask = valid
        if mask.sum() < 3:
            return {"D": np.nan, "intercept": np.nan, "r": np.nan}

    x = -np.log(box_sizes[mask].astype(float))
    y = np.log(counts[mask])
    slope, intercept, r_val, *_ = stats.linregress(x, y)
    return {"D": float(slope), "intercept": float(intercept), "r": float(r_val)}


def skeleton_branch_stats(grid):
    if not SKIMAGE_AVAILABLE:
        return {"endpoints": np.nan, "branchpoints": np.nan, "skeleton_length": np.nan, "branching_ratio": np.nan}

    occ = (grid > 0)
    sk = skeletonize(occ).astype(np.uint8)

    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]])
    neigh_count = convolve(sk, kernel, mode="constant", cval=0)

    endpoints = np.logical_and(sk, neigh_count == 1).sum()
    branchpoints = np.logical_and(sk, neigh_count > 2).sum()
    skeleton_length = sk.sum()
    branching_ratio = branchpoints / endpoints if endpoints > 0 else np.nan

    return {
        "endpoints": int(endpoints),
        "branchpoints": int(branchpoints),
        "skeleton_length": int(skeleton_length),
        "branching_ratio": float(branching_ratio),
    }