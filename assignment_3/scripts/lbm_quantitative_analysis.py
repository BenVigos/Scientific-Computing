#!/usr/bin/env python
"""Comprehensive quantitative comparison for LBM, FDM, and FEM methods.

This script:
1) Runs a high-resolution LBM benchmark.
2) Runs LBM/FDM/FEM across parameter sweeps.
3) Computes common quantitative metrics.
4) Stores structured outputs for reproducible analysis.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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


@dataclass
class SweepSpec:
    method: str
    grid: dict[str, list[Any]]


def ensure_dirs(base: Path) -> dict[str, Path]:
    paths = {
        "root": base,
        "benchmark": base / "benchmark",
        "runs": base / "runs",
        "fields": base / "runs" / "fields",
        "tables": base / "tables",
        "outputs": ROOT / "assignment_3" / "outputs" / "quantitative_analysis",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def resize_field(arr: np.ndarray, shape: tuple[int, int], order: int = 1) -> np.ndarray:
    zx = shape[0] / arr.shape[0]
    zy = shape[1] / arr.shape[1]
    return zoom(arr, (zx, zy), order=order)


def vorticity(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    return np.gradient(uy, axis=0) - np.gradient(ux, axis=1)


def divergence(ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    return np.gradient(ux, axis=0) + np.gradient(uy, axis=1)


def flow_metrics(ux: np.ndarray, uy: np.ndarray, rho: np.ndarray, obstacle: np.ndarray) -> dict[str, float]:
    fluid = ~obstacle
    speed = np.sqrt(ux * ux + uy * uy)
    vort = vorticity(ux, uy)
    div = divergence(ux, uy)

    return {
        "mean_speed": float(np.mean(speed[fluid])),
        "max_speed": float(np.max(speed[fluid])),
        "kinetic_energy": float(np.mean(0.5 * rho[fluid] * speed[fluid] * speed[fluid])),
        "enstrophy": float(np.mean(vort[fluid] * vort[fluid])),
        "rho_mean": float(np.mean(rho[fluid])),
        "rho_std": float(np.std(rho[fluid])),
        "mass_error": float(abs(np.mean(rho[fluid]) - 1.0)),
        "div_l2": float(np.sqrt(np.mean(div[fluid] * div[fluid]))),
        "nan_fraction": float(np.mean(np.isnan(speed[fluid]))),
    }


def benchmark_errors(
    ux: np.ndarray,
    uy: np.ndarray,
    rho: np.ndarray,
    obstacle: np.ndarray,
    ref: dict[str, np.ndarray],
) -> dict[str, float]:
    shape = ux.shape
    ref_ux = resize_field(ref["ux"], shape, order=1)
    ref_uy = resize_field(ref["uy"], shape, order=1)
    ref_rho = resize_field(ref["rho"], shape, order=1)
    ref_obs = resize_field(ref["obstacle"].astype(np.float64), shape, order=0) > 0.5

    mask = (~obstacle) & (~ref_obs)
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

def extract_history_metrics(result: dict[str, Any], method: str) -> dict[str, float]:
    """Read method-specific diagnostics from solver history."""
    history = result.get("history", {})

    if method == "fdm":
        speed_hist = np.asarray(history.get("speed_max", []), dtype=float)
        div_hist = np.asarray(history.get("div_inf", []), dtype=float)
        p_hist = np.asarray(history.get("p_max", []), dtype=float)
        dt_hist = np.asarray(history.get("dt", []), dtype=float)

        return {
            "max_speed_history": float(np.max(speed_hist)) if speed_hist.size else np.nan,
            "max_div_history": float(np.max(div_hist)) if div_hist.size else np.nan,
            "max_p_history": float(np.max(p_hist)) if p_hist.size else np.nan,
            "dt_min": float(np.min(dt_hist)) if dt_hist.size else np.nan,
            "dt_max": float(np.max(dt_hist)) if dt_hist.size else np.nan,
            "dt_final": float(dt_hist[-1]) if dt_hist.size else np.nan,
        }

    if method == "fem":
        umax_hist = np.asarray(history.get("u_max_sampled", []), dtype=float)
        div_hist = np.asarray(history.get("div_l2", []), dtype=float)

        return {
            "max_speed_history": float(np.max(umax_hist)) if umax_hist.size else np.nan,
            "max_div_history": float(np.max(div_hist)) if div_hist.size else np.nan,
            "max_p_history": np.nan,
            "dt_min": np.nan,
            "dt_max": np.nan,
            "dt_final": np.nan,
        }

    return {
        "max_speed_history": np.nan,
        "max_div_history": np.nan,
        "max_p_history": np.nan,
        "dt_min": np.nan,
        "dt_max": np.nan,
        "dt_final": np.nan,
    }

def apply_bc_once(u: np.ndarray, v: np.ndarray, bc: dict[str, Any], masks: dict[str, np.ndarray]) -> None:
    u[bc["u"]["dirichlet_mask"]] = bc["u"]["dirichlet_value"][bc["u"]["dirichlet_mask"]]
    v[bc["v"]["dirichlet_mask"]] = bc["v"]["dirichlet_value"][bc["v"]["dirichlet_mask"]]

    # Neumann at outlet/inlet approximated by first-order copy.
    right = masks["right"]
    left = masks["left"]
    if np.any(right):
        u[-1, :] = u[-2, :]
        v[-1, :] = v[-2, :]
    if np.any(left):
        # Keep inlet values from Dirichlet masks where defined.
        pass

    u[masks["obstacle"]] = 0.0
    v[masks["obstacle"]] = 0.0


def run_fdm(env: KarmannVortex, cfg: dict[str, Any]) -> dict[str, Any]:
    solver = FDMSolver(environment=env, **cfg)
    result = solver.solve(verbose=False)
    return _normalize_result(result, "fdm")


def run_fem(env: KarmannVortex, cfg: dict[str, Any]) -> dict[str, Any]:
    # export grid settings belong to solve(), not __init__
    init_cfg = dict(cfg)
    export_nx = int(init_cfg.pop("export_nx", 301))
    export_ny = int(init_cfg.pop("export_ny", 121))

    solver = FEMSolver(environment=env, **init_cfg)
    result = solver.solve(verbose=False, export_nx=export_nx, export_ny=export_ny)
    return _normalize_result(result, "fem")


def run_lbm(env: KarmannVortex, cfg: dict[str, Any]) -> dict[str, Any]:
    solver = LBMSolver(environment=env, **cfg)
    result = solver.solve(verbose=False)
    return _normalize_result(result, "lbm")


def make_sweeps(config: dict[str, Any]) -> list[SweepSpec]:
    nx = int(config["test_nx"])
    ny = int(config["test_ny"])
    n_steps = int(config["test_steps"])
    u_inlet = float(config["u_inlet"])
    sponge = max(0, int(0.05 * nx))
    return [
        SweepSpec(
            method="lbm",
            grid={
                "nx": [nx],
                "ny": [ny],
                "u_inlet": [u_inlet],
                "reynolds_number": [400, 600, 800],
                "n_steps": [n_steps],
                "velocity_ramp_tau": [200, 400],
                "inlet_bc": ["zou_he", "regularized"],
                "outlet_bc": ["open", "zou_he_pressure"],
                "collision_model": ["bgk", "trt"],
                "trt_lambda": [0.25],
                "alpha": [1.0],
                "top_bc": ["bounce_back"],
                "bottom_bc": ["bounce_back"],
                "use_outlet_sponge": [True],
                "outlet_sponge_width": [sponge],
                "outlet_sponge_sigma_max": [0.15],
                "vis_interval": [max(100, n_steps // 20)],
            },
        ),
        SweepSpec(
            method="fdm",
            grid={
                "nx": [int(config["fdm_base_nx"])],
                "ny": [int(config["fdm_base_ny"])],
                "dt": [float(config["fdm_base_dt"])],
                "n_steps": [int(config["fdm_base_n_steps"])],
                "rho": [1.0],
                "adaptive_dt": [True],
                "cfl_safety": [0.20],
                "diff_safety": [0.20],
                "reynolds_number": [150, 300, 500, 1000],
                "poisson_method": ["amg"],
                "pressure_tol": [1e-5],
                "pressure_maxiter": [300],
                "use_preconditioner": [True],
                "convection_order": ["second"],
                "outlet_bc": ["zero_gradient"],
                "outlet_convection_speed": [None],
            },
        ),
        SweepSpec(
            method="fem",
            grid={
                "dt": [float(config["fem_base_dt"])],
                "n_steps": [int(config["fem_base_n_steps"])],
                "global_maxh": [float(config["fem_base_global_maxh"])],
                "cyl_maxh": [float(config["fem_base_cyl_maxh"])],
                "order": [2],
                "reynolds_number": [150, 300, 500, 1000],
                "rho": [1.0],
                "graddiv_gamma": [1e-3],
                "inlet_profile": ["parabolic"],
                "inlet_perturbation": [1e-3],
                "ramp_time": [0.0],
                "stokes_start": [True],
                "curved_order": [3],
                "probe_point": [(0.6, 0.21)],
                "num_threads": [int(config["fem_num_threads"])],
                "inverse_name": ["cg"],
                "preconditioner_name": ["amg"],
                "export_nx": [int(config["fem_export_nx"])],
                "export_ny": [int(config["fem_export_ny"])],
            },
        ),
    ]


def expand_cases(sweeps: list[SweepSpec]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for sweep in sweeps:
        keys = list(sweep.grid.keys())
        for vals in itertools.product(*(sweep.grid[k] for k in keys)):
            cfg = dict(zip(keys, vals))
            # Filter invalid LBM combinations.
            if sweep.method == "lbm":
                if cfg["collision_model"] == "bgk":
                    cfg.pop("trt_lambda", None)
                if not cfg["use_outlet_sponge"]:
                    cfg["outlet_sponge_width"] = 0
            cases.append({"method": sweep.method, "config": cfg})
    return cases


def run_with_timer(run_fn: Callable[[KarmannVortex, dict[str, Any]], dict[str, Any]], env: KarmannVortex, cfg: dict[str, Any]) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    out = run_fn(env, cfg)
    return out, time.perf_counter() - t0


def build_metrics_plan() -> dict[str, list[str]]:
    return {
        "accuracy_vs_benchmark": ["l2_ux", "l2_uy", "l2_rho", "rel_l2_ux", "rel_l2_uy", "rel_l2_rho"],
        "stability": ["stable", "nan_fraction"],
        "physical_consistency": ["mass_error", "div_l2", "enstrophy", "kinetic_energy", "rho_std"],
        "performance": ["runtime_sec", "cells_per_sec"],
    }


def load_cached_benchmark(benchmark_dir: Path) -> dict[str, np.ndarray] | None:
    ref_file = benchmark_dir / "lbm_reference.npz"
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
    """Return reference fields, runtime and whether it was loaded from cache."""
    if not force_recompute:
        cached = load_cached_benchmark(benchmark_dir)
        if cached is not None:
            meta_file = benchmark_dir / "lbm_reference_config.json"
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
        benchmark_dir / "lbm_reference.npz",
        ux=benchmark_result["ux"],
        uy=benchmark_result["uy"],
        rho=benchmark_result["rho"],
        obstacle=benchmark_result["obstacle"],
    )
    with open(benchmark_dir / "lbm_reference_config.json", "w", encoding="ascii") as f:
        json.dump({"config": benchmark_cfg, "runtime_sec": bench_runtime}, f, indent=2)

    ref = {
        "ux": benchmark_result["ux"],
        "uy": benchmark_result["uy"],
        "rho": benchmark_result["rho"],
        "obstacle": benchmark_result["obstacle"],
    }
    return ref, bench_runtime, False


def run_analysis(config: dict[str, Any]) -> None:
    base = ROOT / "assignment_3" / "data" / "quantitative_analysis"
    paths = ensure_dirs(base)

    env = KarmannVortex(v0=config["u_inlet"])

    plan = build_metrics_plan()
    with open(paths["root"] / "metrics_plan.json", "w", encoding="ascii") as f:
        json.dump(plan, f, indent=2)

    benchmark_cfg = {
        "nx": config["benchmark_nx"],
        "ny": config["benchmark_ny"],
        "u_inlet": config["u_inlet"],
        "reynolds_number": config["benchmark_re"],
        "n_steps": config["benchmark_steps"],
        "vis_interval": max(100, config["benchmark_steps"] // 20),
        "velocity_ramp_tau": config["benchmark_velocity_ramp_tau"],
        "inlet_bc": "regularized",
        "outlet_bc": "zou_he_pressure",
        "top_bc": "bounce_back",
        "bottom_bc": "bounce_back",
        "collision_model": "trt",
        "trt_lambda": 0.25,
        "alpha": 1.0,
        "use_outlet_sponge": True,
        "outlet_sponge_width": max(10, int(1 / 8.8 * config["benchmark_nx"])),
        "outlet_sponge_sigma_max": 0.3,
    }

    ref, bench_runtime, used_cache = run_or_load_benchmark(
        env,
        paths["benchmark"],
        benchmark_cfg,
        force_recompute=bool(config["force_recompute_benchmark"]),
    )

    sweeps = make_sweeps(config)
    cases = expand_cases(sweeps)

    runners: dict[str, Callable[[KarmannVortex, dict[str, Any]], dict[str, Any]]] = {
        "lbm": run_lbm,
        "fdm": run_fdm,
        "fem": run_fem
    }

    run_rows: list[dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        method = case["method"]
        cfg = case["config"]

        result, runtime = run_with_timer(runners[method], env, cfg)
        ux = result["ux"]
        uy = result["uy"]
        rho = result["rho"]
        obstacle = result["obstacle"]

        stable = bool(np.isfinite(ux).all() and np.isfinite(uy).all() and np.isfinite(rho).all())

        row = {
            "run_id": idx,
            "method": method,
            "runtime_sec": runtime,
            "cells_per_sec": float((ux.shape[0] * ux.shape[1]) / max(runtime, 1e-9)),
            "stable": stable,
            **flow_metrics(ux, uy, rho, obstacle),
            **benchmark_errors(ux, uy, rho, obstacle, ref),
        }
        for k, v in cfg.items():
            row[f"cfg_{k}"] = v

        run_rows.append(row)

        if config["save_fields"]:
            np.savez_compressed(
                paths["fields"] / f"run_{idx:04d}_{method}.npz",
                ux=ux,
                uy=uy,
                rho=rho,
                obstacle=obstacle,
            )

    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(paths["runs"] / "all_runs.csv", index=False)

    summary = run_df.groupby("method", as_index=False).agg(
        stable_rate=("stable", "mean"),
        avg_runtime_sec=("runtime_sec", "mean"),
        avg_cells_per_sec=("cells_per_sec", "mean"),
        avg_rel_l2_ux=("rel_l2_ux", "mean"),
        avg_rel_l2_uy=("rel_l2_uy", "mean"),
        avg_rel_l2_rho=("rel_l2_rho", "mean"),
        avg_mass_error=("mass_error", "mean"),
        avg_div_l2=("div_l2", "mean"),
        avg_enstrophy=("enstrophy", "mean"),
    )
    summary.to_csv(paths["tables"] / "method_summary.csv", index=False)

    report_lines = [
        "# Quantitative Analysis: LBM vs FDM vs FEM",
        "",
        "## Metrics Plan",
        "```json",
        json.dumps(plan, indent=2),
        "```",
        "",
        "## Benchmark",
        f"- Method: High-resolution LBM {'(cached)' if used_cache else '(fresh run)'}",
        f"- Grid: {config['benchmark_nx']} x {config['benchmark_ny']}",
        f"- Steps: {config['benchmark_steps']}",
        f"- Runtime (s): {bench_runtime:.3f}",
        "",
        "## Summary Table",
        summary.to_markdown(index=False),
        "",
        "## File Structure",
        "- assignment_3/data/quantitative_analysis/benchmark/lbm_reference.npz",
        "- assignment_3/data/quantitative_analysis/benchmark/lbm_reference_config.json",
        "- assignment_3/data/quantitative_analysis/runs/all_runs.csv",
        "- assignment_3/data/quantitative_analysis/tables/method_summary.csv",
        "- assignment_3/data/quantitative_analysis/metrics_plan.json",
    ]

    with open(paths["outputs"] / "report.md", "w", encoding="ascii") as f:
        f.write("\n".join(report_lines))


def build_run_config() -> dict[str, Any]:
    """Set run settings here instead of using CLI argument parsing."""
    config = {
        # General
        "u_inlet": 0.12,
        "save_fields": False,
        "force_recompute_benchmark": False,

        # Benchmark (high-resolution LBM)
        "benchmark_nx": 640,
        "benchmark_ny": 240,
        "benchmark_steps": 7000,
        "benchmark_re": 10.0,
        "benchmark_velocity_ramp_tau": 600.0,

        # Sweep resolution
        "test_nx": 320,
        "test_ny": 120,
        "test_steps": 4000,

        # FDM 
        "fdm_base_nx": 301,
        "fdm_base_ny": 120,
        "fdm_base_dt": 5e-4,
        "fdm_base_n_steps": 4000,
        "fdm_dt_n_steps": 4000,
        "fdm_mesh_n_steps": 4000,

        # FEM 
        "fem_base_dt": 1e-3,
        "fem_base_n_steps": 4000,
        "fem_dt_n_steps": 4000,
        "fem_mesh_n_steps": 4000,
        "fem_base_global_maxh": 0.02,
        "fem_base_cyl_maxh": 0.0025,
        "fem_export_nx": 301,
        "fem_export_ny": 121,
        "fem_num_threads": 8,


        # Quick mode for fast sanity checks
        "quick_mode": True,
    }

    if config["quick_mode"]:
        config["benchmark_nx"] = 120
        config["benchmark_ny"] = 48
        config["benchmark_steps"] = 250
        config["test_nx"] = 80
        config["test_ny"] = 32
        config["test_steps"] = 180
        
        config["fdm_base_nx"] = 151
        config["fdm_base_ny"] = 60
        config["fdm_base_dt"] = 1e-3
        config["fdm_base_n_steps"] = 800
        config["fdm_dt_n_steps"] = 800
        config["fdm_mesh_n_steps"] = 800

        config["fem_base_dt"] = 2e-3
        config["fem_base_n_steps"] = 600
        config["fem_dt_n_steps"] = 600
        config["fem_mesh_n_steps"] = 600
        config["fem_export_nx"] = 151
        config["fem_export_ny"] = 61
        config["fem_num_threads"] = 4

    return config


if __name__ == "__main__":
    run_analysis(build_run_config())
