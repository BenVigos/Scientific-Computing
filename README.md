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

The structure of the repository for Assignment 2 is as follows:

```
» assignment_2
    » data
        » comparison_figures
            » bbox_density_comparison.png
            » D_est_comparison.png
            » dla_growth_panels.pdf
            » dla_growth_panels.png
            » R_g_comparison.png
        » dla
            » figures
                » omega_optimization
                    » omega_boxplots.png
                    » omega_ita_heatmap.png
                    » omega_vs_mean_iterations.png
                    » omega_vs_stability.png
                    » omega_vs_total_iterations.png
                » all_metric_trends_vs_ita.png
                » all_metric_trends.png
                » aspect_ratio_boxplot_vs_ita.png
                » aspect_ratio_boxplot.png
                » bbox_density_boxplot_vs_ita.png
                » branching_ratio_boxplot.png
                » branchpoints_boxplot.png
                » D_est_boxplot_vs_ita.png
                » D_est_boxplot.png
                » D_r_boxplot.png
                » D_vs_Rg_scatter.png
                » dla_benchmark_comparison.png
                » endpoints_boxplot.png
                » height_boxplot_vs_ita.png
                » height_boxplot.png
                » max_width_boxplot_vs_ita.png
                » max_width_boxplot.png
                » metric_trends.png
                » metrics_boxplots_vs_ita.png
                » metrics_boxplots.png
                » metrics_correlation.png
                » metrics_pairplot.png
                » occupancy_boxplot_vs_ita.png
                » occupancy_boxplot.png
                » perimeter_boxplot_vs_ita.png
                » perimeter_boxplot.png
                » perimeter_per_occupied_boxplot_vs_ita.png
                » R_g_boxplot_vs_ita.png
                » R_g_boxplot.png
                » skeleton_length_boxplot.png
            » dla_benchmark_results.csv
            » dla_omega_convergence.csv
            » pde_ita_metrics_big.csv
            » pde_ita_metrics.csv
        » dla_mc
            » figures
                » all_metric_trends_vs_ps.png
                » aspect_ratio_boxplot_vs_ps.png
                » bbox_density_boxplot_vs_ps.png
                » D_est_boxplot_vs_ps.png
                » height_boxplot_vs_ps.png
                » max_width_boxplot_vs_ps.png
                » metrics_boxplots_vs_ps.png
                » metrics_correlation.png
                » metrics_pairplot.png
                » occupancy_boxplot_vs_ps.png
                » perimeter_boxplot_vs_ps.png
                » perimeter_per_occupied_boxplot_vs_ps.png
                » R_g_boxplot_vs_ps.png
            » mc_ps_metrics.csv
        » .gitkeep
    » notebooks
        » .gitkeep
    » outputs
        » config4_analysis_spatial.png
        » config4_analysis_stats.png
        » gray_scott_evolution_base.png
        » gray_scott_parameters_UV.png
    » report
        » images
            » config4_analysis_spatial.png
            » config4_analysis_stats.png
            » gray_scott_evolution_base.png
            » gray_scott_parameters_UV.png
        » assignment_2.tex
    » scripts
        » .gitkeep
        » dla_benchmark.py
        » dla_example_plots.py
        » dla_experiment_runner.py
        » dla_metrics.csv
        » dla_omega_opt.py
        » gray_scott.py
        » mc_ps_sweeps.py
        » pde_ita_sweeps.py
        » results_table_helper.py
```

All the relevant scripts for the assignment are in `assignment_2/scripts/`. Run the scripts to reproduce the figures and animations for the DLA and Gray–Scott experiments. The scripts are designed to save outputs (images, animations) to the current working directory by default, so you can run them from anywhere.

- dla_benchmark.py — runs the DLA benchmark experiment for normal and parallel SOR
- dla_example_plots.py — runs both DLA implementations to obtain representative cluster plots.
- dla_experiment_runner.py — runs the DLA experiments with multiple random seeds and saves the computed metrics to a cvs file.
- dla_omega_opt.py — runs the SOR omega optimization experiment for the DLA problem and saves the convergence plot
- gray_scott.py — runs a single Gray–Scott simulation and saves the final concentration fields as images
- mc_ps_sweep.py — runs the Monte Carlo parameter sweep for the system and saves the computed metrics.
- pde_ita_sweeps.py — runs the PDE-ITA parameter sweeps for the system and saves the computed metrics.
- results_table_helper.py — helper script to generate a table of results from the parameter sweeps for LaTeX report.

Outputs (where you will find saved plots and videos)

The outputs for the experiments are saved to the `data` folder under `assignment_2` by default. 