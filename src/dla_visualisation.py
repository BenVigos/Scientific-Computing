import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_dla_metrics(
    csv_path=None,
    out_dir=None,
    max_pairplot_vars=6,
    param_col=None,
    model_col="model",
    preferred_metrics=None
):
    """
    Visualize numeric metrics from a sweep CSV

    Seed handling:
      - trend plots: aggregate over seeds (mean ± std)
      - boxplots: show distributions across seeds

    preferred_metrics:
      Optional list of metrics to plot. If None, all non-static numeric metrics are used.
    """
    if csv_path is None:
        candidates = [
            os.path.join("..", "data", "dla_mc", "mc_ps_metrics.csv"),
            os.path.join("..", "data", "dla", "pde_ita_metrics.csv"),
            os.path.join("assignment_2", "data", "dla_mc", "mc_ps_metrics.csv"),
            os.path.join("assignment_2", "data", "dla", "pde_ita_metrics.csv"),
            "mc_ps_metrics.csv",
            "pde_ita_metrics.csv",
        ]
        csv_path = next((p for p in candidates if os.path.exists(p)), candidates[0])

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(csv_path), "figures")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # detect sweep parameter
    if param_col is None:
        if "ita" in df.columns:
            param_col = "ita"
        elif "ps" in df.columns:
            param_col = "ps"
        else:
            raise ValueError("Could not find sweep parameter column. Expected 'ita' or 'ps'.")

    df[param_col] = pd.to_numeric(df[param_col], errors="coerce")
    df = df.dropna(subset=[param_col]).copy()

    has_model = model_col in df.columns

    # static / boring columns to exclude from plotting
    ignore_cols = {
        "seed",
        "N",
        "steps",
        "stop_threshold",
        "target_occupancy",
        "max_steps_per_walker",
        "top_boundary_percentage_stop",
        "omega",
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metric_cols = [c for c in numeric_cols if c not in ignore_cols and c != param_col]

    if preferred_metrics is not None:
        metric_cols = [c for c in preferred_metrics if c in metric_cols]

    if len(metric_cols) == 0:
        raise ValueError(f"No numeric metric columns found to plot after filtering.")

    sns.set(style="whitegrid", context="talk")

    # 1) Mean ± std trends
    n_metrics = len(metric_cols)
    fig, axes = plt.subplots(nrows=n_metrics, ncols=1, figsize=(9, 3 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_cols):
        if has_model:
            grouped = (
                df.groupby([model_col, param_col])[metric]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            for model_name, g in grouped.groupby(model_col):
                ax.plot(g[param_col], g["mean"], marker="o", label=f"{model_name}")
                ax.fill_between(
                    g[param_col],
                    g["mean"] - g["std"],
                    g["mean"] + g["std"],
                    alpha=0.2
                )
        else:
            grouped = df.groupby(param_col)[metric].agg(["mean", "std", "count"]).reset_index()
            ax.plot(grouped[param_col], grouped["mean"], marker="o", label="mean")
            ax.fill_between(
                grouped[param_col],
                grouped["mean"] - grouped["std"],
                grouped["mean"] + grouped["std"],
                alpha=0.25
            )

        ax.set_ylabel(metric)
        ax.legend()

    axes[-1].set_xlabel(param_col)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"all_metric_trends_vs_{param_col}.png"), dpi=150)
    plt.close(fig)

    # 2) Combined boxplots
    max_cols = 3
    n_plot = len(metric_cols)
    grid_cols = min(max_cols, n_plot)
    grid_rows = int(np.ceil(n_plot / grid_cols))

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(5 * grid_cols, 4 * grid_rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax in axes_flat[n_plot:]:
        ax.axis("off")

    for ax, metric in zip(axes_flat, metric_cols):
        if has_model:
            sns.boxplot(x=param_col, y=metric, hue=model_col, data=df, ax=ax)
            if ax.legend_ is not None:
                ax.legend_.remove()
        else:
            sns.boxplot(x=param_col, y=metric, data=df, ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("")

    if has_model:
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))

    fig.tight_layout(rect=(0, 0, 1, 0.96) if has_model else None)
    fig.savefig(os.path.join(out_dir, f"metrics_boxplots_vs_{param_col}.png"), dpi=150)
    plt.close(fig)

    # 3) Individual boxplots
    for metric in metric_cols:
        plt.figure(figsize=(9, 4))
        if has_model:
            sns.boxplot(x=param_col, y=metric, hue=model_col, data=df)
        else:
            sns.boxplot(x=param_col, y=metric, data=df)
        plt.xlabel(param_col)
        plt.ylabel(metric)
        plt.title(f"{metric} distribution vs {param_col}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{metric}_boxplot_vs_{param_col}.png"), dpi=150)
        plt.close()

    # 4) Correlation heatmap
    numeric = df.select_dtypes(include=[np.number]).drop(
        columns=list(ignore_cols | {param_col}),
        errors="ignore"
    )
    corr = numeric.corr()

    plt.figure(figsize=(max(7, 0.45 * corr.shape[0]), max(6, 0.45 * corr.shape[1])))
    sns.heatmap(corr, annot=False, cmap="vlag", center=0)
    plt.title("Correlation matrix (metrics only)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_correlation.png"), dpi=150)
    plt.close()

    # 5) Pairplot
    pair_cols = metric_cols.copy()
    if len(pair_cols) > max_pairplot_vars:
        preferred = ["D_est", "R_g", "bbox_density", "perimeter_per_occupied", "height", "aspect_ratio"]
        pair_cols = [c for c in preferred if c in metric_cols][:max_pairplot_vars]
        if len(pair_cols) < 2:
            pair_cols = metric_cols[:max_pairplot_vars]

    if len(pair_cols) >= 2:
        pair_df = df[pair_cols + [param_col] + ([model_col] if has_model else [])].copy()
        if len(pair_df) > 2000:
            pair_df = pair_df.sample(2000, random_state=0)

        hue_col = model_col if has_model else param_col
        g = sns.pairplot(
            pair_df,
            vars=pair_cols,
            hue=hue_col,
            plot_kws={"s": 18, "alpha": 0.6},
            diag_kind="kde"
        )
        g.fig.suptitle("Pairwise relationships (subset)", y=1.02)
        g.fig.tight_layout()
        g.fig.savefig(os.path.join(out_dir, "metrics_pairplot.png"), dpi=150)
        plt.close()



if __name__ == "__main__":
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    visualize_dla_metrics(
        csv_path=csv_arg,
        preferred_metrics=[
            "bbox_density",
            "perimeter_per_occupied",
            "D_est",
            "R_g",
            "height",
            "aspect_ratio",
            "max_width",
            "perimeter",
            "occupancy",
        ]
    )