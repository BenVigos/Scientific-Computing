# python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_dla_metrics(csv_path=None, out_dir=None, max_pairplot_vars=6):
    """
    Visualize all numeric metrics contained in a DLA metrics CSV.

    - Detects numeric columns automatically (excludes 'seed').
    - Produces:
      * a stacked mean ± std trend plot vs 'ita' (one row per metric)
      * boxplots per metric (individual files and a combined grid)
      * a correlation heatmap
      * a pairplot for up to `max_pairplot_vars` metrics (to avoid huge plots)

    :param csv_path: path to CSV file. If None, tries common candidate locations.
    :param out_dir: output directory for figures; defaults to a 'figures' subfolder next to the CSV.
    :param max_pairplot_vars: maximum number of numeric variables to include in the seaborn pairplot.
    """
    # defaults (common locations)
    if csv_path is None:
        candidates = [
            os.path.join("..", "data", "dla", "dla_metrics.csv"),
            os.path.join("assignment_2", "data", "dla", "dla_metrics.csv"),
            os.path.join("data", "dla", "dla_metrics.csv"),
            "dla_metrics.csv",
        ]
        csv_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(csv_path), "figures")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    # ensure 'ita' exists and is numeric (otherwise try to coerce)
    if 'ita' not in df.columns:
        raise ValueError("CSV must contain an 'ita' column to plot trends.")
    df['ita'] = pd.to_numeric(df['ita'], errors='coerce')
    # drop rows where ita is NaN
    df = df.dropna(subset=['ita']).copy()

    # detect numeric metric columns (exclude 'seed' and 'ita' from metrics-to-plot lists)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metric_cols = [c for c in numeric_cols if c not in ('seed', 'ita')]
    if len(metric_cols) == 0:
        raise ValueError("No numeric metric columns found in the CSV (besides 'ita'/'seed').")

    sns.set(style='whitegrid', context='talk')

    # 1) Mean ± std trends vs ita: one subplot per metric (stacked vertically)
    n_metrics = len(metric_cols)
    ncols = 1
    nrows = n_metrics
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 3*nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_cols):
        grouped = df.groupby('ita')[metric].agg(['mean', 'std', 'count']).reset_index()
        if grouped.empty:
            ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center')
            continue
        ax.plot(grouped['ita'], grouped['mean'], marker='o', label=f'{metric} mean')
        ax.fill_between(grouped['ita'], grouped['mean'] - grouped['std'], grouped['mean'] + grouped['std'], alpha=0.25)
        ax.set_ylabel(metric)
        ax.legend()
    axes[-1].set_xlabel('ita')
    fig.tight_layout()
    trend_path = os.path.join(out_dir, 'all_metric_trends.png')
    fig.savefig(trend_path, dpi=150)
    plt.close(fig)

    # 2) Boxplots: create a combined grid and individual boxplots per metric
    # combined grid (arrange into reasonable rows/cols)
    max_cols = 3
    n_plot = len(metric_cols)
    grid_cols = min(max_cols, n_plot)
    grid_rows = int(np.ceil(n_plot / grid_cols))
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(4*grid_cols, 3*grid_rows))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for ax in axes_flat[n_plot:]:
        ax.axis('off')
    for ax, metric in zip(axes_flat, metric_cols):
        sns.boxplot(x='ita', y=metric, data=df, ax=ax, color='C0')
        ax.set_title(metric)
        ax.set_xlabel('')
    plt.tight_layout()
    combined_box_path = os.path.join(out_dir, 'metrics_boxplots.png')
    fig.savefig(combined_box_path, dpi=150)
    plt.close(fig)

    # individual boxplots
    for metric in metric_cols:
        plt.figure(figsize=(8,4))
        sns.boxplot(x='ita', y=metric, data=df, color='C0')

        # use stripplot instead of swarmplot for large datasets to avoid placement warnings
        sns.stripplot(x='ita', y=metric, data=df, color='k', alpha=0.4, size=3, jitter=0.2)

        plt.xlabel('ita')
        plt.ylabel(metric)
        plt.title(f'Distribution of {metric} per ita')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{metric}_boxplot.png'), dpi=150)
        plt.close()

    # 3) correlation heatmap for numeric metrics (excluding seed)
    numeric = df.select_dtypes(include=[np.number]).drop(columns=['seed'], errors='ignore')
    corr = numeric.corr()
    plt.figure(figsize=(max(6, 0.5*corr.shape[0]), max(5, 0.5*corr.shape[1])))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', center=0)
    plt.title('Correlation matrix (numeric metrics)')
    plt.tight_layout()
    corr_path = os.path.join(out_dir, 'metrics_correlation.png')
    plt.savefig(corr_path, dpi=150)
    plt.close()

    # 4) pairplot (only for a subset if many metrics)
    pair_cols = metric_cols.copy()
    if len(pair_cols) > max_pairplot_vars:
        # choose the most interesting: D_est, R_g, occupancy, perimeter, max_width, height if available
        preferred = ['D_est', 'R_g', 'perimeter', 'max_width', 'height']
        pair_cols = [c for c in preferred if c in metric_cols][:max_pairplot_vars]
        if len(pair_cols) < 2:
            pair_cols = metric_cols[:max_pairplot_vars]
    if len(pair_cols) >= 2:
        # seaborn pairplot can be slow for many rows; sample if very large
        sample_df = df[pair_cols + ['ita']]
        if len(sample_df) > 2000:
            sample_df = sample_df.sample(2000, random_state=0)
        g = sns.pairplot(sample_df, vars=pair_cols, hue='ita', palette='viridis', plot_kws={'s':20, 'alpha':0.6}, diag_kind='kde')
        pairplot_path = os.path.join(out_dir, 'metrics_pairplot.png')
        g.fig.suptitle('Pairwise relationships (subset)', y=1.02)
        g.fig.tight_layout()
        g.fig.savefig(pairplot_path, dpi=150)
        plt.close()

    print('Saved figures to:', os.path.abspath(out_dir))


if __name__ == '__main__':
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    visualize_dla_metrics(csv_path=csv_arg)
