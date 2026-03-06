# Overview — Scientific Computing

This repository contains the code, notebooks, scripts and report for the assignments of the Scientific Computing course. The repository is organized so that reusable numerical routines live in `src/` and experiments, figures and animations live in `assignment_x/`.

## Setup - dependencies (uv)

This project uses the `uv` CLI to manage the virtual environment and dependencies. On Windows (PowerShell) follow this minimal workflow to get the environment ready.

- Install `uv` 

```powershell
python -m pip install uv
```

- Install / sync all project packages (creates/uses the project environment and installs pinned versions):

```powershell
uv sync
```

- Run project scripts (examples):

```powershell
python assignment_2/scripts/visualize_dla.py
python -m src.time_diff_csv
```

## Assignment 1: 1D wave equation, 2D diffusion, iterative solvers for steady Laplace

- 1D wave equation (time-dependent finite-difference solver, convergence studies, snapshots and animations)
- 2D time-dependent diffusion equation (explicit finite differences, comparisons to analytic solution, animation)
- Iterative solvers for the steady Laplace problem (Jacobi, Gauss–Seidel, SOR) and experiments to compare convergence properties

This README tells you where the code and outputs are and how to reproduce the results and animations. Each topic above is implemented in the repository; run the appropriate script(s) below to reproduce the figures/animations for that topic.

---

Repository layout (important files)

- `src/`
  - `fdm_schemes.py` — 1D wave equation solver (explicit FD). Use for time-series solutions and snapshots.
  - `wave_analysis.py` — helper utilities (analytic solution, L2 error, log-log fit, single-run helper).
  - `time_diffusion.py` — 2D diffusion explicit simulation + analytic comparison (erfc series).
  - `time_anim.py` — animated 2D diffusion visualiser (saves `equilibrium_state.png` when equilibrium is detected).
  - `time_diff_csv.py` — export diffusion time-sampled data to `diffusion_data.csv`.
  - `iter_schemes.py` — iterative solvers: `jacobi`, `gauss_seidel`, `sor` for steady-state Laplace problems.
  - `grid.py` — helper masks and grid constructors for iterative solver experiments.

- `assignment_1/`
  - `scripts/run_experiments.py` — the consolidated script that reproduces the notebook workflow for the 1D wave experiments: saves images and three interactive HTML animations by default (in `assignment_1/outputs/`). Use this to reproduce the main wave figures and videos.
  - `scripts/wave_grid_convergence.py` — runs the grid-convergence study (L2 error vs dx) and saves the convergence plot.
  - `notebooks/` — Jupyter notebooks with interactive experiments and exploratory plots for both the wave and diffusion/iterative parts.
  - `report/` — LaTeX source and images used in the assignment report.

Outputs (where you will find saved plots and videos)

- Default output directory for the scripted wave experiments: `assignment_1/outputs/`
  - `solutions_imshow.png` — heatmap (time vs x) images for the three initial conditions
  - `stacked_snapshots.png` — stacked snapshot figure with colour-graded time traces
  - `wave_animation_1.html`, `wave_animation_2.html`, `wave_animation_3.html` — interactive HTML animations (self-contained)

- Diffusion outputs (when you run the diffusion scripts):
  - `equilibrium_state.png` — saved by `src/time_anim.py` when equilibrium is detected (saved to current working directory by default)
  - `diffusion_data.csv` — saved by `src/time_diff_csv.py` when you run that script


## Assignment 2: Diffusion-limited aggregation and Gray–Scott reaction-diffusion system

Requirements

- Python 3.8+
- Recommended packages: `numpy`, `scipy`, `matplotlib`

Install the essentials (example):

```powershell
python -m venv .venv
..venvScriptsActivate.ps1
pip install numpy scipy matplotlib
```

