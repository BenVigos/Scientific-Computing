#!/usr/bin/env python
"""Grid convergence study for LBM, FDM, and FEM methods.

This script:
1) Runs a high-resolution LBM benchmark (coarse grid).
2) Runs convergence sweeps over grid sizes for selected methods.
3) Computes accuracy errors vs benchmark and runtime per method.
4) Stores results in method-specific CSV files with common schema.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from envirmonment import KarmannVortex  # noqa: E402
from solvers import LBMSolver, FDMSolver, FEMSolver  # noqa: E402


def _normalize_result(result: dict[str, Any], method: str) -> dict[str, Any]:
    """Ensure a common result schema across methods."""
    if "ux" not in result or "uy" not in result:
        missing = [k for k in ("ux", "uy") if k not in result]
        raise ValueError(f"{method} result missing keys: {missing}")

    ux = result["ux"]
    uy = result["uy"]
    if ux.shape != uy.shape:
        raise ValueError(f"{method} returned ux/uy with mismatched shapes: {ux.shape} vs {uy.shape}")

    # FDM/FEM may not return rho explicitly; use constant density from metadata or 1.0.
    if "rho" not in result:
        rho_val = float(result.get("metadata", {}).get("rho", 1.0))
        result["rho"] = np.full_like(ux, rho_val, dtype=np.float64)

    # Ensure obstacle exists for fluid masking; default to all-fluid.
    if "obstacle" not in result:
        result["obstacle"] = np.zeros_like(ux, dtype=bool)

    return result


def ensure_dirs(base: Path) -> dict[str, Path]:
    paths = {
        "root": base,
        "benchmark": base / "benchmark",
        "lbm_convergence": base / "lbm_convergence",
        "fdm_convergence": base / "fdm_convergence",
        "fem_convergence": base / "fem_convergence",
        "lbm_states": base / "lbm_convergence" / "states",
        "fdm_states": base / "fdm_convergence" / "states",
        "fem_states": base / "fem_convergence" / "states",
        "outputs": ROOT / "assignment_3" / "outputs" / "convergence_study",
        "lbm_images": ROOT / "assignment_3" / "outputs" / "convergence_study" / "lbm_final_states",
        "fdm_images": ROOT / "assignment_3" / "outputs" / "convergence_study" / "fdm_final_states",
        "fem_images": ROOT / "assignment_3" / "outputs" / "convergence_study" / "fem_final_states",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def resize_field(arr: np.ndarray, shape: tuple[int, int], order: int = 1) -> np.ndarray:
    """Resize a field to match reference shape."""
    if arr.shape == shape:
        return arr
    zx = shape[0] / arr.shape[0]
    zy = shape[1] / arr.shape[1]
    return zoom(arr, (zx, zy), order=order)


def compute_error_metrics(
    ux: np.ndarray,
    uy: np.ndarray,
    rho: np.ndarray,
    obstacle: np.ndarray,
    ref: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute L2 and relative L2 errors against reference."""
    shape = ux.shape
    ref_ux = resize_field(ref["ux"], shape, order=1)
    ref_uy = resize_field(ref["uy"], shape, order=1)
    ref_rho = resize_field(ref["rho"], shape, order=1)
    ref_obs = resize_field(ref["obstacle"].astype(np.float64), shape, order=0) > 0.5

    mask = (~obstacle) & (~ref_obs)
    if not np.any(mask):
        return {
            "l2_ux": np.nan,
            "l2_uy": np.nan,
            "l2_rho": np.nan,
            "rel_l2_ux": np.nan,
            "rel_l2_uy": np.nan,
            "rel_l2_rho": np.nan,
        }

    eps = 1e-12

    dux = ux[mask] - ref_ux[mask]
    duy = uy[mask] - ref_uy[mask]
    drho = rho[mask] - ref_rho[mask]

    den_ux = np.sqrt(np.mean(ref_ux[mask] ** 2)) + eps
    den_uy = np.sqrt(np.mean(ref_uy[mask] ** 2)) + eps
    den_rho = np.sqrt(np.mean(ref_rho[mask] ** 2)) + eps

    l2_ux = float(np.sqrt(np.mean(dux * dux)))
    l2_uy = float(np.sqrt(np.mean(duy * duy)))
    l2_rho = float(np.sqrt(np.mean(drho * drho)))

    return {
        "l2_ux": l2_ux,
        "l2_uy": l2_uy,
        "l2_rho": l2_rho,
        "rel_l2_ux": l2_ux / den_ux,
        "rel_l2_uy": l2_uy / den_uy,
        "rel_l2_rho": l2_rho / den_rho,
    }


def compute_flow_integral_metrics(
    ux: np.ndarray,
    uy: np.ndarray,
    rho: np.ndarray,
    obstacle: np.ndarray,
    dx: float,
    dy: float,
) -> dict[str, float]:

    fluid = (~obstacle) & np.isfinite(ux) & np.isfinite(uy) & np.isfinite(rho)
    if not np.any(fluid):
        return {k: np.nan for k in [
            "total_kinetic_energy",
            "enstrophy",
            "max_abs_vorticity",
            "integrated_vorticity_magnitude",
            "divergence_l2_error",
            "mass_conservation_error",
        ]}

    cell_area = dx * dy

    ux_filled = np.where(fluid, ux, 0.0)
    uy_filled = np.where(fluid, uy, 0.0)

    # --- Gradients with spacing ---
    dudy = np.gradient(ux_filled, dy, axis=1)
    dvdx = np.gradient(uy_filled, dx, axis=0)
    vorticity = dvdx - dudy

    dudx = np.gradient(ux_filled, dx, axis=0)
    dvdy = np.gradient(uy_filled, dy, axis=1)
    divergence = dudx + dvdy

    speed2 = ux * ux + uy * uy

    # --- Metrics ---
    total_kinetic_energy = float(
        np.sum(0.5 * rho[fluid] * speed2[fluid]) * cell_area
    )

    enstrophy = float(
        0.5 * np.sum(vorticity[fluid] ** 2) * cell_area
    )

    max_abs_vorticity = float(
        np.max(np.abs(vorticity[fluid]))
    )

    integrated_vorticity_magnitude = float(
        np.sum(np.abs(vorticity[fluid])) * cell_area
    )

    divergence_l2_error = float(
        np.sqrt(np.sum(divergence[fluid] ** 2) * cell_area)
    )

    mass_conservation_error = float(
        np.std(rho[fluid])
    )

    return {
        "total_kinetic_energy": total_kinetic_energy,
        "enstrophy": enstrophy,
        "max_abs_vorticity": max_abs_vorticity,
        "integrated_vorticity_magnitude": integrated_vorticity_magnitude,
        "divergence_l2_error": divergence_l2_error,
        "mass_conservation_error": mass_conservation_error,
    }


def save_final_state_image(
    ux: np.ndarray,
    uy: np.ndarray,
    obstacle: np.ndarray,
    out_file: Path,
    title: str,
) -> None:
    """Save final speed field image, masking obstacle cells."""
    speed = np.sqrt(ux * ux + uy * uy)
    speed = np.where(obstacle, np.nan, speed)

    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=140)
    im = ax.imshow(
        speed.T,
        origin="lower",
        cmap="viridis",
        aspect="auto",
        extent=(0, ux.shape[0], 0, ux.shape[1]),
    )
    ax.set_xlabel("x (grid index)")
    ax.set_ylabel("y (grid index)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Speed |u|")
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def save_final_state_data(
    ux: np.ndarray,
    uy: np.ndarray,
    rho: np.ndarray,
    obstacle: np.ndarray,
    out_file: Path,
    metadata: dict[str, Any],
) -> None:
    """Save full final state and small metadata snapshot for offline metrics."""
    np.savez_compressed(
        out_file,
        ux=ux,
        uy=uy,
        rho=rho,
        obstacle=obstacle,
    )
    meta_file = out_file.with_suffix(".json")
    with open(meta_file, "w", encoding="ascii") as f:
        json.dump(metadata, f, indent=2, default=float)


def run_lbm(env: KarmannVortex, cfg: dict[str, Any]) -> dict[str, Any]:
    solver = LBMSolver(environment=env, **cfg)
    result = solver.solve(verbose=False)
    return _normalize_result(result, "lbm")


def run_fdm(env: KarmannVortex, cfg: dict[str, Any]) -> dict[str, Any]:
    solver = FDMSolver(environment=env, **cfg)
    result = solver.solve(verbose=False)
    return _normalize_result(result, "fdm")


def run_fem(env: KarmannVortex, cfg: dict[str, Any]) -> dict[str, Any]:
    init_cfg = dict(cfg)
    export_nx = int(init_cfg.pop("export_nx", 301))
    export_ny = int(init_cfg.pop("export_ny", 121))

    solver = FEMSolver(environment=env, **init_cfg)
    result = solver.solve(verbose=False, export_nx=export_nx, export_ny=export_ny)
    return _normalize_result(result, "fem")


def run_with_timer(
    run_fn: Callable[[KarmannVortex, dict[str, Any]], dict[str, Any]],
    env: KarmannVortex,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    out = run_fn(env, cfg)
    return out, time.perf_counter() - t0


def lbm_steps_for_target_time(nx: int, target_time: float) -> int:
    """LBM step mapping requested by user: n_steps = nx * T / (2.2 * 5.33)."""
    steps = nx * target_time / (2.2 * 5.33)
    return max(1, int(round(steps)))


def load_cached_benchmark(benchmark_dir: Path) -> dict[str, np.ndarray] | None:
    ref_file = benchmark_dir / "lbm_benchmark.npz"
    if not ref_file.exists():
        return None

    data = np.load(ref_file)
    return {
        "ux": data["ux"],
        "uy": data["uy"],
        "rho": data["rho"],
        "obstacle": data["obstacle"],
    }


def run_or_load_benchmark(
    env: KarmannVortex,
    benchmark_dir: Path,
    benchmark_cfg: dict[str, Any],
    force_recompute: bool,
) -> tuple[dict[str, np.ndarray], float, bool]:
    """Return reference fields, runtime, and whether loaded from cache."""
    if not force_recompute:
        cached = load_cached_benchmark(benchmark_dir)
        if cached is not None:
            meta_file = benchmark_dir / "lbm_benchmark_config.json"
            runtime = 0.0
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="ascii") as f:
                        runtime = float(json.load(f).get("runtime_sec", 0.0))
                except Exception:
                    runtime = 0.0
            return cached, runtime, True

    benchmark_result, bench_runtime = run_with_timer(run_lbm, env, benchmark_cfg)
    np.savez_compressed(
        benchmark_dir / "lbm_benchmark.npz",
        ux=benchmark_result["ux"],
        uy=benchmark_result["uy"],
        rho=benchmark_result["rho"],
        obstacle=benchmark_result["obstacle"],
    )
    with open(benchmark_dir / "lbm_benchmark_config.json", "w", encoding="ascii") as f:
        json.dump({"config": benchmark_cfg, "runtime_sec": bench_runtime}, f, indent=2)

    ref = {
        "ux": benchmark_result["ux"],
        "uy": benchmark_result["uy"],
        "rho": benchmark_result["rho"],
        "obstacle": benchmark_result["obstacle"],
    }
    return ref, bench_runtime, False


def run_lbm_convergence(config: dict[str, Any], paths: dict[str, Path], env: KarmannVortex, ref: dict[str, np.ndarray]) -> None:
    """Run LBM convergence study over grid sizes."""
    print("\n=== LBM Convergence Study ===")

    grid_sizes = config["lbm_grid_sizes"]
    n_steps = config["lbm_convergence_steps"]
    u_inlet = config["u_inlet"]
    re = config["convergence_reynolds"]
    alpha = config["lbm_alpha"] if "lbm_alpha" in config else 1.0

    rows = []

    target_time = float(config["target_physical_time"])

    for idx, (nx, ny) in enumerate(grid_sizes, start=1):
        print(f"  LBM {idx}/{len(grid_sizes)}: grid {nx}x{ny}...", end=" ", flush=True)

        sponge_width = max(0, int(0.1 * nx))
        n_steps = lbm_steps_for_target_time(nx, target_time)
        cfg = {
            "nx": nx,
            "ny": ny,
            "u_inlet": u_inlet,
            "reynolds_number": re,
            "n_steps": n_steps,
            "vis_interval": max(100, n_steps),
            "velocity_ramp_tau": 100,
            "inlet_bc": "regularized",
            "outlet_bc": "open",
            "top_bc": "bounce_back",
            "bottom_bc": "bounce_back",
            "collision_model": "bgk",
            "alpha": alpha,
            "use_outlet_sponge": True,
            "outlet_sponge_width": sponge_width,
            "outlet_sponge_sigma_max": 0.3,
        }

        result, runtime = run_with_timer(run_lbm, env, cfg)
        ux = result["ux"]
        uy = result["uy"]
        rho = result["rho"]
        obstacle = result["obstacle"]

        stable = bool(np.isfinite(ux).any() and np.isfinite(uy).any() and np.isfinite(rho).any())
        ncells = nx * ny
        throughput = ncells / max(runtime, 1e-9)

        errors = compute_error_metrics(ux, uy, rho, obstacle, ref)
        dx = (env.x_range[1] - env.x_range[0]) / max(nx - 1, 1)
        dy = (env.y_range[1] - env.y_range[0]) / max(ny - 1, 1)
        flow_metrics = compute_flow_integral_metrics(ux, uy, rho, obstacle, dx, dy)

        row = {
            "grid_index": idx,
            "nx": nx,
            "ny": ny,
            "n_steps": n_steps,
            "target_time": target_time,
            "effective_time": target_time,
            "ncells": ncells,
            "runtime_sec": runtime,
            "cells_per_sec": throughput,
            "stable": stable,
            **errors,
            **flow_metrics,
        }
        rows.append(row)

        image_file = paths["lbm_images"] / f"lbm_run_{idx:03d}_{nx}x{ny}.png"
        save_final_state_image(ux, uy, obstacle, image_file, f"LBM final state - {nx}x{ny}")

        state_file = paths["lbm_states"] / f"lbm_run_{idx:03d}_{nx}x{ny}.npz"
        save_final_state_data(
            ux,
            uy,
            rho,
            obstacle,
            state_file,
            metadata={
                "method": "lbm",
                "grid_index": idx,
                "nx": nx,
                "ny": ny,
                "n_steps": n_steps,
                "target_time": target_time,
                "runtime_sec": runtime,
                "stable": stable,
                "metrics": {**errors, **flow_metrics},
                "config": cfg,
            },
        )
        print(f"t={runtime:.2f}s, L2_ux={errors['rel_l2_ux']:.3e}, stable={stable}")

    df = pd.DataFrame(rows)
    df.to_csv(paths["lbm_convergence"] / "results.csv", index=False)
    print(f"  Saved: {paths['lbm_convergence'] / 'results.csv'}")


def run_fdm_convergence(config: dict[str, Any], paths: dict[str, Path], env: KarmannVortex, ref: dict[str, np.ndarray]) -> None:
    """Run FDM convergence study over grid sizes."""
    print("\n=== FDM Convergence Study ===")

    grid_sizes = config["fdm_grid_sizes"]
    n_steps_cap = int(config["fdm_convergence_steps"])
    dt = config["fdm_base_dt"]
    re = config["convergence_reynolds"]

    rows = []

    target_time = float(config["target_physical_time"])

    for idx, (nx, ny) in enumerate(grid_sizes, start=1):
        print(f"  FDM {idx}/{len(grid_sizes)}: grid {nx}x{ny}...", end=" ", flush=True)

        n_steps = max(n_steps_cap, int(np.ceil(target_time / dt)) + 5)
        print(f"fdm n_steps={n_steps}")
        cfg = {
            "nx": nx,
            "ny": ny,
            "dt": dt,
            "n_steps": n_steps,
            "target_physical_time": target_time,
            "rho": 1.0,
            "adaptive_dt": True,
            "cfl_safety": 0.20,
            "diff_safety": 0.20,
            "reynolds_number": re,
            "poisson_method": "amg",
            "pressure_tol": 1e-5,
            "pressure_maxiter": 300,
            "use_preconditioner": True,
            "convection_order": "second",
            "outlet_bc": "zero_gradient",
            "outlet_convection_speed": None,
        }

        result, runtime = run_with_timer(run_fdm, env, cfg)
        ux = result["ux"]
        uy = result["uy"]
        rho = result["rho"]
        obstacle = result["obstacle"]

        stable = bool(np.isfinite(ux).any() and np.isfinite(uy).any() and np.isfinite(rho).any())
        metadata = result.get("metadata", {})
        effective_time = float(metadata.get("time_final", target_time))
        steps_used = int(metadata.get("n_steps_executed", n_steps))
        stop_reason = str(metadata.get("stop_reason", "n_steps"))
        ncells = nx * ny
        throughput = ncells / max(runtime, 1e-9)

        errors = compute_error_metrics(ux, uy, rho, obstacle, ref)
        dx = float(metadata.get("dx", (env.x_range[1] - env.x_range[0]) / max(nx - 1, 1)))
        dy = float(metadata.get("dy", (env.y_range[1] - env.y_range[0]) / max(ny - 1, 1)))
        flow_metrics = compute_flow_integral_metrics(ux, uy, rho, obstacle, dx, dy)

        row = {
            "grid_index": idx,
            "nx": nx,
            "ny": ny,
            "n_steps_cap": n_steps,
            "n_steps_executed": steps_used,
            "dt": dt,
            "target_time": target_time,
            "effective_time": effective_time,
            "stop_reason": stop_reason,
            "ncells": ncells,
            "runtime_sec": runtime,
            "cells_per_sec": throughput,
            "stable": stable,
            **errors,
            **flow_metrics,
        }
        rows.append(row)

        image_file = paths["fdm_images"] / f"fdm_run_{idx:03d}_{nx}x{ny}.png"
        save_final_state_image(ux, uy, obstacle, image_file, f"FDM final state - {nx}x{ny}")

        state_file = paths["fdm_states"] / f"fdm_run_{idx:03d}_{nx}x{ny}.npz"
        save_final_state_data(
            ux,
            uy,
            rho,
            obstacle,
            state_file,
            metadata={
                "method": "fdm",
                "grid_index": idx,
                "nx": nx,
                "ny": ny,
                "n_steps": n_steps,
                "n_steps_executed": steps_used,
                "dt": dt,
                "target_time": target_time,
                "effective_time": effective_time,
                "stop_reason": stop_reason,
                "runtime_sec": runtime,
                "stable": stable,
                "metrics": {**errors, **flow_metrics},
                "config": cfg,
            },
        )
        print(f"t={runtime:.2f}s, L2_ux={errors['rel_l2_ux']:.3e}, stable={stable}")

    df = pd.DataFrame(rows)
    df.to_csv(paths["fdm_convergence"] / "results.csv", index=False)
    print(f"  Saved: {paths['fdm_convergence'] / 'results.csv'}")


def run_fem_convergence(config: dict[str, Any], paths: dict[str, Path], env: KarmannVortex, ref: dict[str, np.ndarray]) -> None:
    """Run FEM convergence study over mesh resolutions."""
    print("\n=== FEM Convergence Study ===")

    mesh_resolutions = config["fem_mesh_resolutions"]
    n_steps_cap = int(config["fem_convergence_steps"])
    dt = config["fem_base_dt"]
    re = config["convergence_reynolds"]

    rows = []

    target_time = float(config["target_physical_time"]/10) # FEM is much slower, so we target a shorter physical time for convergence analysis.

    for idx, (global_maxh, cyl_maxh, export_nx, export_ny) in enumerate(mesh_resolutions, start=1):
        print(f"  FEM {idx}/{len(mesh_resolutions)}: maxh={global_maxh:.4f}...", end=" ", flush=True)

        n_steps = max(n_steps_cap, int(np.ceil(target_time / dt)) + 5)
        print(f"fem n_steps={n_steps}")
        cfg = {
            "dt": dt,
            "n_steps": n_steps,
            "target_physical_time": target_time,
            "global_maxh": global_maxh,
            "cyl_maxh": cyl_maxh,
            "order": 2,
            "reynolds_number": re,
            "rho": 1.0,
            "graddiv_gamma": 1e-3,
            "inlet_profile": "parabolic",
            "inlet_perturbation": 1e-3,
            "ramp_time": 0.0,
            "stokes_start": True,
            "curved_order": 3,
            "probe_point": (0.6, 0.21),
            "num_threads": config["fem_num_threads"],
            "inverse_name": "cg",
            "preconditioner_name": "amg",
            "export_nx": export_nx,
            "export_ny": export_ny,
        }

        result, runtime = run_with_timer(run_fem, env, cfg)
        ux = result["ux"]
        uy = result["uy"]
        rho = result["rho"]
        obstacle = result["obstacle"]

        stable = bool(np.isfinite(ux).any() and np.isfinite(uy).any() and np.isfinite(rho).any())
        metadata = result.get("metadata", {})
        effective_time = float(metadata.get("time_final", target_time))
        steps_used = int(metadata.get("n_steps_executed", n_steps))
        stop_reason = str(metadata.get("stop_reason", "n_steps"))
        ncells = ux.shape[0] * ux.shape[1]
        throughput = ncells / max(runtime, 1e-9)

        errors = compute_error_metrics(ux, uy, rho, obstacle, ref)
        dx = (env.x_range[1] - env.x_range[0]) / max(export_nx - 1, 1)
        dy = (env.y_range[1] - env.y_range[0]) / max(export_ny - 1, 1)
        flow_metrics = compute_flow_integral_metrics(ux, uy, rho, obstacle, dx, dy)

        row = {
            "mesh_index": idx,
            "global_maxh": global_maxh,
            "cyl_maxh": cyl_maxh,
            "export_nx": export_nx,
            "export_ny": export_ny,
            "n_steps_cap": n_steps,
            "n_steps_executed": steps_used,
            "dt": dt,
            "target_time": target_time,
            "effective_time": effective_time,
            "stop_reason": stop_reason,
            "ncells": ncells,
            "runtime_sec": runtime,
            "cells_per_sec": throughput,
            "stable": stable,
            **errors,
            **flow_metrics,
        }
        rows.append(row)

        image_file = paths["fem_images"] / f"fem_run_{idx:03d}_{export_nx}x{export_ny}.png"
        save_final_state_image(ux, uy, obstacle, image_file, f"FEM final state - {export_nx}x{export_ny}")

        state_file = paths["fem_states"] / f"fem_run_{idx:03d}_{export_nx}x{export_ny}.npz"
        save_final_state_data(
            ux,
            uy,
            rho,
            obstacle,
            state_file,
            metadata={
                "method": "fem",
                "mesh_index": idx,
                "global_maxh": global_maxh,
                "cyl_maxh": cyl_maxh,
                "export_nx": export_nx,
                "export_ny": export_ny,
                "n_steps": n_steps,
                "n_steps_executed": steps_used,
                "dt": dt,
                "target_time": target_time,
                "effective_time": effective_time,
                "stop_reason": stop_reason,
                "runtime_sec": runtime,
                "stable": stable,
                "metrics": {**errors, **flow_metrics},
                "config": cfg,
            },
        )
        print(f"t={runtime:.2f}s, L2_ux={errors['rel_l2_ux']:.3e}, stable={stable}")

    df = pd.DataFrame(rows)
    df.to_csv(paths["fem_convergence"] / "results.csv", index=False)
    print(f"  Saved: {paths['fem_convergence'] / 'results.csv'}")


def build_run_config() -> dict[str, Any]:
    """Build configuration with user-selectable method sweeps."""
    base_x = 220
    base_y = 41
    config = {
        # General
        "u_inlet": 0.12,
        "target_physical_time": 100.0,
        "convergence_reynolds": 150.0,
        "force_recompute_benchmark": False,

        # Benchmark LBM (coarse resolution reference)
        "benchmark_nx": base_x*10,
        "benchmark_ny": base_y*10,
        "benchmark_re": 150.0,
        "benchmark_velocity_ramp_tau": 1.0,

        # LBM convergence sweep (grid refinement)
        "run_lbm_convergence": False,
        "lbm_grid_sizes": [(int(base_x * s), int(base_y * s)) for s in (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5)],
        "lbm_convergence_steps": 10000,

        # FDM convergence sweep (grid refinement)
        "run_fdm_convergence": False,
        "fdm_grid_sizes": [
            (int(base_x * s), int(base_y * s)) for s in (0.5,1, 1.5, 2)
        ],
        "fdm_base_dt": 1e-1,
        "fdm_convergence_steps": 15000,


        # FEM convergence sweep (mesh refinement)
        "run_fem_convergence": True,
        "fem_mesh_resolutions": [
            (0.05, 0.005, 77, 14),
            (0.04, 0.004, 108, 20),
            (0.03, 0.003, 151, 28),
            (0.02, 0.002, 220, 41),
        ],
        "fem_base_dt": 1e-3,
        "fem_convergence_steps": 10000,
        "fem_num_threads": 8,

        # Quick mode for debugging
        "quick_mode": False,
    }

    if config["quick_mode"]:
        # Small benchmark
        config["benchmark_nx"] = 80
        config["benchmark_ny"] = 32
        config["target_physical_time"] = 10.0
        config["benchmark_velocity_ramp_tau"] = 100.0

        # Tiny sweeps
        config["lbm_grid_sizes"] = [(60, 24), (80, 32), (100, 40)]
        config["lbm_convergence_steps"] = 500
        config["lbm_alpha"] = 0.995

        config["fdm_grid_sizes"] = [(81, 32), (121, 48)]
        config["fdm_convergence_steps"] = 500

        config["fem_mesh_resolutions"] = [(0.05, 0.01, 80, 32), (0.03, 0.005, 133, 53)]
        config["fem_convergence_steps"] = 500

    return config


def run_analysis(config: dict[str, Any]) -> None:
    """Main analysis runner."""
    base = ROOT / "assignment_3" / "data" / "convergence_study"
    paths = ensure_dirs(base)

    env = KarmannVortex(v0=config["u_inlet"])

    # 1) Run or load benchmark
    print("=" * 70)
    print("STEP 1: Benchmark LBM Run")
    print("=" * 70)

    benchmark_steps = lbm_steps_for_target_time(config["benchmark_nx"], float(config["target_physical_time"]))
    benchmark_cfg = {
        "nx": config["benchmark_nx"],
        "ny": config["benchmark_ny"],
        "u_inlet": config["u_inlet"],
        "reynolds_number": config["benchmark_re"],
        "n_steps": benchmark_steps,
        "vis_interval": max(100, benchmark_steps // 20),
        "velocity_ramp_tau": config["benchmark_velocity_ramp_tau"],
        "inlet_bc": "regularized",
        "outlet_bc": "open",
        "top_bc": "bounce_back",
        "bottom_bc": "bounce_back",
        "collision_model": "bgk",
        "trt_lambda": 0.25,
        "alpha": 1.0,
        "use_outlet_sponge": True,
        "outlet_sponge_width": max(5, int(0.1 * config["benchmark_nx"])),
        "outlet_sponge_sigma_max": 0.3,
    }


    ref, bench_runtime, used_cache = run_or_load_benchmark(
        env,
        paths["benchmark"],
        benchmark_cfg,
        force_recompute=bool(config["force_recompute_benchmark"]),
    )

    benchmark_image = paths["outputs"] / "benchmark_final_state.png"
    save_final_state_image(
        ref["ux"],
        ref["uy"],
        ref["obstacle"],
        benchmark_image,
        title=f"Benchmark LBM final state - {config['benchmark_nx']}x{config['benchmark_ny']}",
    )

    benchmark_state_file = paths["benchmark"] / "lbm_benchmark_final_state.npz"
    save_final_state_data(
        ref["ux"],
        ref["uy"],
        ref["rho"],
        ref["obstacle"],
        benchmark_state_file,
        metadata={
            "method": "lbm_benchmark",
            "nx": config["benchmark_nx"],
            "ny": config["benchmark_ny"],
            "target_time": config["target_physical_time"],
            "runtime_sec": bench_runtime,
            "used_cache": used_cache,
            "config": benchmark_cfg,
        },
    )

    print(f"Benchmark: {config['benchmark_nx']}x{config['benchmark_ny']} grid")
    print(f"  Target physical time: {config['target_physical_time']}")
    print(f"  Steps (LBM mapping): {benchmark_steps}")
    print(f"  Status: {'loaded from cache' if used_cache else 'freshly computed'}")
    print(f"  Runtime: {bench_runtime:.2f}s")
    print(f"  Benchmark image: {benchmark_image}")
    print(f"  Benchmark state: {benchmark_state_file}")

    # 2) Run selected convergence sweeps
    print("\n" + "=" * 70)
    print("STEP 2: Convergence Sweeps")
    print("=" * 70)

    if config["run_lbm_convergence"]:
        run_lbm_convergence(config, paths, env, ref)

    if config["run_fdm_convergence"]:
        run_fdm_convergence(config, paths, env, ref)

    if config["run_fem_convergence"]:
        run_fem_convergence(config, paths, env, ref)

    # 3) Summary report
    print("\n" + "=" * 70)
    print("STEP 3: Summary")
    print("=" * 70)
    print(f"Benchmark: {paths['benchmark'] / 'lbm_benchmark.npz'}")
    print(f"Benchmark image: {benchmark_image}")
    print(f"Benchmark state: {benchmark_state_file}")
    if config["run_lbm_convergence"]:
        print(f"LBM convergence: {paths['lbm_convergence'] / 'results.csv'}")
        print(f"LBM final images: {paths['lbm_images']}")
        print(f"LBM final states: {paths['lbm_states']}")
    if config["run_fdm_convergence"]:
        print(f"FDM convergence: {paths['fdm_convergence'] / 'results.csv'}")
        print(f"FDM final images: {paths['fdm_images']}")
        print(f"FDM final states: {paths['fdm_states']}")
    if config["run_fem_convergence"]:
        print(f"FEM convergence: {paths['fem_convergence'] / 'results.csv'}")
        print(f"FEM final images: {paths['fem_images']}")
        print(f"FEM final states: {paths['fem_states']}")

    print("\n✓ Convergence study complete.")


if __name__ == "__main__":
    run_analysis(build_run_config())

