#!/usr/bin/env python
"""Plot convergence study results for LBM, FDM, and FEM methods.

This script loads convergence study CSV files and generates publication-quality plots:
- Error vs. grid size (convergence rates)
- Runtime vs. grid size (computational cost)
- Performance (cells/sec) vs. grid size
- Error vs. runtime (accuracy-cost tradeoff)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_BASE = ROOT / "assignment_3" / "data" / "convergence_study"
OUTPUT_BASE = ROOT / "assignment_3" / "outputs" / "convergence_study"


def load_results(method: str) -> Optional[pd.DataFrame]:
    """Load convergence results for a method. Return None if file doesn't exist."""
    path = DATA_BASE / f"{method}_convergence" / "results.csv"
    if not path.exists():
        print(f"Warning: {path} not found. Skipping {method.upper()}.")
        return None
    return pd.read_csv(path)


def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 8,
    })


def plot_convergence_error(lbm_df: Optional[pd.DataFrame],
                          fdm_df: Optional[pd.DataFrame],
                          fem_df: Optional[pd.DataFrame]) -> None:
    """Plot relative L2 error vs. ncells (convergence rate)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (df, metric) in zip(axes, [
        (lbm_df, "rel_l2_ux"),
        (lbm_df, "rel_l2_uy"),
        (lbm_df, "rel_l2_rho"),
    ]):
        if df is not None:
            # Log-log plot for convergence visualization
            ax.loglog(df["ncells"], df[metric], "o-", label="LBM", linewidth=2, markersize=8)
            ax.set_xlabel("Number of cells")
            ax.set_ylabel(f"Relative L2 error ({metric.split('_')[2].upper()})")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()

        if fdm_df is not None:
            ax.loglog(fdm_df["ncells"], fdm_df[metric], "s-", label="FDM", linewidth=2, markersize=8)
            ax.legend()

        if fem_df is not None:
            ax.loglog(fem_df["ncells"], fem_df[metric], "^-", label="FEM", linewidth=2, markersize=8)
            ax.legend()

    fig.suptitle("Convergence: Relative L2 Error vs. Grid Size", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_BASE / "convergence_error.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {OUTPUT_BASE / 'convergence_error.png'}")
    plt.close()


def plot_runtime_vs_ncells(lbm_df: Optional[pd.DataFrame],
                           fdm_df: Optional[pd.DataFrame],
                           fem_df: Optional[pd.DataFrame]) -> None:
    """Plot runtime vs. ncells (computational cost scaling)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if lbm_df is not None:
        ax.loglog(lbm_df["ncells"], lbm_df["runtime_sec"], "o-", label="LBM", linewidth=2, markersize=8)

    if fdm_df is not None:
        ax.loglog(fdm_df["ncells"], fdm_df["runtime_sec"], "s-", label="FDM", linewidth=2, markersize=8)

    if fem_df is not None:
        ax.loglog(fem_df["ncells"], fem_df["runtime_sec"], "^-", label="FEM", linewidth=2, markersize=8)

    # Add reference lines for O(n) and O(n log n) scaling
    ncells_range = np.logspace(3, 6, 100)
    ax.loglog(ncells_range, 1e-6 * ncells_range, "k--", alpha=0.3, label="O(n)")
    ax.loglog(ncells_range, 1e-6 * ncells_range * np.log(ncells_range), "k:", alpha=0.3, label="O(n log n)")

    ax.set_xlabel("Number of cells")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Computational Cost Scaling", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_BASE / "runtime_scaling.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {OUTPUT_BASE / 'runtime_scaling.png'}")
    plt.close()


def plot_throughput(lbm_df: Optional[pd.DataFrame],
                   fdm_df: Optional[pd.DataFrame],
                   fem_df: Optional[pd.DataFrame]) -> None:
    """Plot throughput (cells/sec) vs. ncells."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if lbm_df is not None:
        ax.semilogx(lbm_df["ncells"], lbm_df["cells_per_sec"], "o-", label="LBM", linewidth=2, markersize=8)

    if fdm_df is not None:
        ax.semilogx(fdm_df["ncells"], fdm_df["cells_per_sec"], "s-", label="FDM", linewidth=2, markersize=8)

    if fem_df is not None:
        ax.semilogx(fem_df["ncells"], fem_df["cells_per_sec"], "^-", label="FEM", linewidth=2, markersize=8)

    ax.set_xlabel("Number of cells")
    ax.set_ylabel("Throughput (cells/second)")
    ax.set_title("Computational Throughput", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_BASE / "throughput.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {OUTPUT_BASE / 'throughput.png'}")
    plt.close()


def plot_accuracy_vs_cost(lbm_df: Optional[pd.DataFrame],
                         fdm_df: Optional[pd.DataFrame],
                         fem_df: Optional[pd.DataFrame]) -> None:
    """Plot relative L2 error vs. runtime (accuracy-cost tradeoff)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, metric in zip(axes, ["rel_l2_ux", "rel_l2_uy", "rel_l2_rho"]):
        if lbm_df is not None:
            ax.loglog(lbm_df["runtime_sec"], lbm_df[metric], "o-", label="LBM", linewidth=2, markersize=8)

        if fdm_df is not None:
            ax.loglog(fdm_df["runtime_sec"], fdm_df[metric], "s-", label="FDM", linewidth=2, markersize=8)

        if fem_df is not None:
            ax.loglog(fem_df["runtime_sec"], fem_df[metric], "^-", label="FEM", linewidth=2, markersize=8)

        ax.set_xlabel("Runtime (seconds)")
        ax.set_ylabel(f"Relative L2 error ({metric.split('_')[2].upper()})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle("Accuracy vs. Computational Cost Tradeoff", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_BASE / "accuracy_vs_cost.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {OUTPUT_BASE / 'accuracy_vs_cost.png'}")
    plt.close()


def plot_individual_method(df: pd.DataFrame, method: str) -> None:
    """Plot convergence details for a single method."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Error vs ncells
    ax = axes[0, 0]
    ax.loglog(df["ncells"], df["rel_l2_ux"], "o-", label="rel_L2(ux)", linewidth=2, markersize=8)
    ax.loglog(df["ncells"], df["rel_l2_uy"], "s-", label="rel_L2(uy)", linewidth=2, markersize=8)
    ax.loglog(df["ncells"], df["rel_l2_rho"], "^-", label="rel_L2(rho)", linewidth=2, markersize=8)
    ax.set_xlabel("Number of cells")
    ax.set_ylabel("Relative L2 error")
    ax.set_title(f"{method.upper()} Convergence")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # Runtime vs ncells
    ax = axes[0, 1]
    ax.loglog(df["ncells"], df["runtime_sec"], "ko-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of cells")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title(f"{method.upper()} Runtime Scaling")
    ax.grid(True, which="both", alpha=0.3)

    # Throughput vs ncells
    ax = axes[1, 0]
    ax.semilogx(df["ncells"], df["cells_per_sec"], "go-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of cells")
    ax.set_ylabel("Throughput (cells/second)")
    ax.set_title(f"{method.upper()} Throughput")
    ax.grid(True, which="both", alpha=0.3)

    # Stability
    ax = axes[1, 1]
    stable_count = df["stable"].sum()
    total_count = len(df)
    ax.bar(["Stable", "Unstable"], [stable_count, total_count - stable_count], color=["green", "red"], alpha=0.7)
    ax.set_ylabel("Count")
    ax.set_title(f"{method.upper()} Stability: {stable_count}/{total_count} runs stable")
    ax.set_ylim([0, total_count + 1])

    fig.suptitle(f"{method.upper()} Convergence Study Details", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_BASE / f"{method}_details.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {OUTPUT_BASE / f'{method}_details.png'}")
    plt.close()


def print_summary_table(lbm_df: Optional[pd.DataFrame],
                       fdm_df: Optional[pd.DataFrame],
                       fem_df: Optional[pd.DataFrame]) -> None:
    """Print summary statistics table."""
    print("\n" + "=" * 80)
    print("CONVERGENCE STUDY SUMMARY")
    print("=" * 80)

    for name, df in [("LBM", lbm_df), ("FDM", fdm_df), ("FEM", fem_df)]:
        if df is None:
            print(f"\n{name}: No data")
            continue

        print(f"\n{name}:")
        print(f"  Grid sizes tested: {len(df)}")
        print(f"  Ncells range: {df['ncells'].min():,} - {df['ncells'].max():,}")
        print(f"  Runtime range: {df['runtime_sec'].min():.2f}s - {df['runtime_sec'].max():.2f}s")
        print(f"  Throughput range: {df['cells_per_sec'].min():.2e} - {df['cells_per_sec'].max():.2e} cells/sec")
        print(f"  Stable runs: {df['stable'].sum()}/{len(df)}")
        print(f"  Rel L2(ux) range: {df['rel_l2_ux'].min():.3e} - {df['rel_l2_ux'].max():.3e}")
        print(f"  Rel L2(uy) range: {df['rel_l2_uy'].min():.3e} - {df['rel_l2_uy'].max():.3e}")
        print(f"  Rel L2(rho) range: {df['rel_l2_rho'].min():.3e} - {df['rel_l2_rho'].max():.3e}")


def main():
    """Main plotting routine."""
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CONVERGENCE STUDY PLOTTING")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    lbm_df = load_results("lbm")
    fdm_df = load_results("fdm")
    fem_df = load_results("fem")

    if lbm_df is None and fdm_df is None and fem_df is None:
        print("ERROR: No convergence data found. Run convergence_study.py first.")
        return

    # Setup plotting
    setup_plot_style()

    # Generate plots
    print("\nGenerating plots...")
    plot_convergence_error(lbm_df, fdm_df, fem_df)
    plot_runtime_vs_ncells(lbm_df, fdm_df, fem_df)
    plot_throughput(lbm_df, fdm_df, fem_df)
    plot_accuracy_vs_cost(lbm_df, fdm_df, fem_df)

    if lbm_df is not None:
        plot_individual_method(lbm_df, "lbm")
    if fdm_df is not None:
        plot_individual_method(fdm_df, "fdm")
    if fem_df is not None:
        plot_individual_method(fem_df, "fem")

    # Summary
    print_summary_table(lbm_df, fdm_df, fem_df)

    print("\n" + "=" * 80)
    print(f"✓ All plots saved to: {OUTPUT_BASE}")
    print("=" * 80)


if __name__ == "__main__":
    main()

