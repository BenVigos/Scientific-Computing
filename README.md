# Assignment 1 — Scientific Computing

This repository contains the code, notebooks, scripts and report for Assignment 1 of the Scientific Computing module. The repository is organized so that reusable numerical routines live in `src/` and experiments, figures and animations live in `assignment_1/` (scripts and notebooks).

Important: this repo is the deliverable for *Assignment 1* and contains multiple separate problems that are all part of the assignment and are equally important. The main topics covered are:

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

If you change the script `run_experiments.py` to use a different output folder with `--outdir`, the files will be produced in the folder you specify.

---

How to run (recommended order)

1) Full wave experiments and animations (reproduces the notebook flow and writes 3 animations):

```powershell
# from the repository root
python -u .\assignment_1\scripts\run_experiments.py
```

- Use `--quick` for a low-resolution, fast test run (handy during development):

```powershell
python -u .\assignment_1\scripts\run_experiments.py --quick --outdir .\assignment_1\outputs_test
```

2) Grid-convergence study (L2 error vs dx):

```powershell
python .\assignment_1\scripts\wave_grid_convergence.py
```

3) 2D diffusion experiments:

```powershell
# simulate and compare to analytical solution
python .\src\time_diffusion.py

# run the animated diffusion visualiser (saves equilibrium snapshot when reached)
python .\src\time_anim.py

# export sampled data to CSV
python .\src\time_diff_csv.py
```

4) Iterative solver experiments (from a Python REPL or notebook):

```python
from src.iter_schemes import jacobi, gauss_seidel, sor
cJ, deltasJ = jacobi(50)
cG, deltasG = gauss_seidel(50)
cS, deltasS, k, conv = sor(50, omega=1.8)
```

The notebooks under `assignment_1/notebooks/` demonstrate how to run the iterative solver sweeps and produce the convergence plots used in the report.

---

Requirements

- Python 3.8+
- Recommended packages: `numpy`, `scipy`, `matplotlib`

Install the essentials (example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy scipy matplotlib
```

---

Notes, caveats and suggestions

- All three assignment topics (1D wave, 2D diffusion, iterative solvers / steady Laplace) are part of *Assignment 1* and are treated as equally important — make sure to run and inspect outputs from each section when grading or checking results.
- The scripts save outputs to `assignment_1/outputs/` by default (the `run_experiments.py` script provides `--outdir` to override).
- For large full runs (default N=1000 and nt ~2000), memory use can be significant: use `--quick` for quick checks or reduce `N`/`T` in the script when experimenting interactively.
- If you want, I can:
  - Add CLI flags to all scripts to make N/T/dt/outdir configurable without editing the script.
  - Add a `requirements.txt` or `pyproject.toml` for reproducible environments.
  - Add small unit tests for `src/wave_analysis.py` and the iterative-schemes functions.

---

If you'd like, I can now implement any of the suggested follow-ups (add CLI flags, create a requirements file, add tests, or run the full experiments and list the produced files). Which would you like next?
