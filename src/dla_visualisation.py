import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


METRIC_LABELS = {
    "bbox_density": "Bounding-box density",
    "D_est": "Fractal dimension",
    "R_g": "Radius of gyration",
    "height": "Cluster height",
    "max_width": "Maximum width",
    "aspect_ratio": "Aspect ratio",
    "perimeter_per_occupied": "Perimeter / occupied site",
    "perimeter": "Perimeter length",
    "occupancy": "Occupancy fraction",
}

PARAM_LABELS = {
    "ita": r"$\eta$",
    "ps": r"$p_s$",
}


def _get_metric_bounds(metric, df):
    if metric in {"bbox_density", "occupancy"}:
        return 0.0, 1.0
    if metric in {"height", "max_width"}:
        if "N" in df.columns:
            return 0.0, float(df["N"].iloc[0])
        return 0.0, None
    if metric in {"aspect_ratio", "R_g", "D_est", "perimeter", "perimeter_per_occupied"}:
        return 0.0, None
    return None, None


def _clip_band(mean, std, lower=None, upper=None):
    y_low = mean - std
    y_high = mean + std
    if lower is not None:
        y_low = np.maximum(y_low, lower)
        y_high = np.maximum(y_high, lower)
    if upper is not None:
        y_low = np.minimum(y_low, upper)
        y_high = np.minimum(y_high, upper)
    return y_low, y_high


def _prepare_grouped(df, param_col, metric):
    grouped = (
        df.groupby(param_col)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(param_col)
    )
    x = grouped[param_col].to_numpy(dtype=float)
    mean = grouped["mean"].to_numpy(dtype=float)
    std = np.nan_to_num(grouped["std"].to_numpy(dtype=float), nan=0.0)
    return x, mean, std


def plot_metric_comparison(
    metric,
    df_pde,
    df_mc,
    out_path=None,
    pde_param_col="ita",
    mc_param_col="ps",
    metric_label=None,
    pde_panel_title="PDE DLA",
    mc_panel_title="Monte Carlo DLA",
    show_title=False,
    figure_title=None,
    panel_titles=True,
    figsize=(10, 4.5),
    title_fontsize=18,
    panel_title_fontsize=14,
    label_fontsize=13,
    tick_fontsize=11,
    line_width=2.2,
    marker_size=6,
    alpha_band=0.20,
    df_pde_big=None,
    pde_big_legend=None,
):
    """
    Two-panel comparison plot with shared y-axis and clean labels.
    If df_pde_big is provided, plot both PDE datasets in the same panel with legends.
    """

    sns.set_style("whitegrid")

    df_pde = df_pde.copy()
    df_mc = df_mc.copy()

    # clean parameter columns
    df_pde[pde_param_col] = pd.to_numeric(df_pde[pde_param_col], errors="coerce").round(1)
    df_mc[mc_param_col] = pd.to_numeric(df_mc[mc_param_col], errors="coerce")

    df_pde = df_pde.dropna(subset=[pde_param_col])
    df_mc = df_mc.dropna(subset=[mc_param_col])

    if df_pde_big is not None:
        df_pde_big = df_pde_big.copy()
        df_pde_big[pde_param_col] = pd.to_numeric(df_pde_big[pde_param_col], errors="coerce").round(1)
        df_pde_big = df_pde_big.dropna(subset=[pde_param_col])

    metric_label = metric_label or METRIC_LABELS.get(metric, metric)
    pde_xlabel = PARAM_LABELS.get(pde_param_col, pde_param_col)
    mc_xlabel = PARAM_LABELS.get(mc_param_col, mc_param_col)

    # grouped data
    x_pde, mean_pde, std_pde = _prepare_grouped(df_pde, pde_param_col, metric)
    x_mc, mean_mc, std_mc = _prepare_grouped(df_mc, mc_param_col, metric)

    if df_pde_big is not None:
        x_pde_big, mean_pde_big, std_pde_big = _prepare_grouped(df_pde_big, pde_param_col, metric)

    lower_pde, upper_pde = _get_metric_bounds(metric, df_pde)
    lower_mc, upper_mc = _get_metric_bounds(metric, df_mc)

    lower = lower_pde if lower_pde is not None else lower_mc
    upper = upper_pde if upper_pde is not None else upper_mc

    y_low_pde, y_high_pde = _clip_band(mean_pde, std_pde, lower, upper)
    y_low_mc, y_high_mc = _clip_band(mean_mc, std_mc, lower, upper)

    if df_pde_big is not None:
        y_low_pde_big, y_high_pde_big = _clip_band(mean_pde_big, std_pde_big, lower, upper)

    # compute common y-limits from BOTH panels
    combined_low = min(np.min(y_low_pde), np.min(y_low_mc))
    combined_high = max(np.max(y_high_pde), np.max(y_high_mc))

    if df_pde_big is not None:
        combined_low = min(combined_low, np.min(y_low_pde_big))
        combined_high = max(combined_high, np.max(y_high_pde_big))

    if lower is not None:
        combined_low = max(combined_low, lower)
    if upper is not None:
        combined_high = min(combined_high, upper)

    # small padding
    pad = 0.03 * (combined_high - combined_low) if combined_high > combined_low else 0.05
    y_min = combined_low - pad if lower is None else max(lower, combined_low - pad)
    y_max = combined_high + pad if upper is None else min(upper, combined_high + pad)

    if df_pde_big is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

        # PDE panel
        ax = axes[0]
        ax.plot(x_pde, mean_pde, marker="o", linewidth=line_width, markersize=marker_size)
        ax.fill_between(x_pde, y_low_pde, y_high_pde, alpha=alpha_band)
        if panel_titles:
            ax.set_title(pde_panel_title, fontsize=panel_title_fontsize)
        ax.set_xlabel(pde_xlabel, fontsize=label_fontsize)
        ax.set_ylabel(metric_label, fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.set_ylim(y_min, y_max)

        # MC panel
        ax = axes[1]
        ax.plot(x_mc, mean_mc, marker="o", linewidth=line_width, markersize=marker_size)
        ax.fill_between(x_mc, y_low_mc, y_high_mc, alpha=alpha_band)
        if panel_titles:
            ax.set_title(mc_panel_title, fontsize=panel_title_fontsize)
        ax.set_xlabel(mc_xlabel, fontsize=label_fontsize)
        # ax.set_ylabel(metric_label, fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.set_ylim(y_min, y_max)
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

        # PDE panel with both datasets
        ax = axes[0]
        ax.plot(x_pde, mean_pde, marker="o", linewidth=line_width, markersize=marker_size, label="normal: N=100")
        ax.fill_between(x_pde, y_low_pde, y_high_pde, alpha=alpha_band)
        ax.plot(x_pde_big, mean_pde_big, marker="s", linewidth=line_width, markersize=marker_size, label=pde_big_legend or "optimizes: N=300")
        ax.fill_between(x_pde_big, y_low_pde_big, y_high_pde_big, alpha=alpha_band)
        if panel_titles:
            ax.set_title(pde_panel_title, fontsize=panel_title_fontsize)
        ax.set_xlabel(pde_xlabel, fontsize=label_fontsize)
        ax.set_ylabel(metric_label, fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=label_fontsize)

        # MC panel
        ax = axes[1]
        ax.plot(x_mc, mean_mc, marker="o", linewidth=line_width, markersize=marker_size)
        ax.fill_between(x_mc, y_low_mc, y_high_mc, alpha=alpha_band)
        if panel_titles:
            ax.set_title(mc_panel_title, fontsize=panel_title_fontsize)
        ax.set_xlabel(mc_xlabel, fontsize=label_fontsize)
        # ax.set_ylabel(metric_label, fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        ax.set_ylim(y_min, y_max)

    if show_title:
        fig.suptitle(figure_title or metric_label, fontsize=title_fontsize)

    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    return fig, axes

def make_comparison_figures(
    pde_csv_path,
    mc_csv_path,
    out_dir,
    metrics=("bbox_density", "D_est", "R_g"),
    show_title=False,
    panel_titles=True,
    title_fontsize=18,
    panel_title_fontsize=14,
    label_fontsize=13,
    tick_fontsize=11,
    line_width=2.2,
    marker_size=6,
    alpha_band=0.20,
    pde_big_csv_path=None,
    pde_big_legend=None,
):
    os.makedirs(out_dir, exist_ok=True)

    df_pde = pd.read_csv(pde_csv_path)
    df_mc = pd.read_csv(mc_csv_path)
    df_pde_big = pd.read_csv(pde_big_csv_path) if pde_big_csv_path is not None else None

    for metric in metrics:
        if metric not in df_pde.columns or metric not in df_mc.columns:
            print(f"Skipping {metric}: not found in both CSVs.")
            continue

        if df_pde_big is not None and metric not in df_pde_big.columns:
            print(f"Skipping {metric}: not found in pde_big CSV.")
            continue

        out_path = os.path.join(out_dir, f"{metric}_comparison.png")

        plot_metric_comparison(
            metric=metric,
            df_pde=df_pde,
            df_mc=df_mc,
            out_path=out_path,
            metric_label=METRIC_LABELS.get(metric, metric),
            pde_panel_title="PDE DLA",
            mc_panel_title="Monte Carlo DLA",
            show_title=show_title,
            figure_title=METRIC_LABELS.get(metric, metric),
            panel_titles=panel_titles,
            title_fontsize=title_fontsize,
            panel_title_fontsize=panel_title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            line_width=line_width,
            marker_size=marker_size,
            alpha_band=alpha_band,
            df_pde_big=df_pde_big,
            pde_big_legend=pde_big_legend,
        )

    print("Saved comparison figures to:", os.path.abspath(out_dir))



if __name__ == "__main__":
    make_comparison_figures(
        pde_csv_path="../assignment_2/data/dla/pde_ita_metrics.csv",
        mc_csv_path="../assignment_2/data/dla_mc/mc_ps_metrics.csv",
        out_dir="../assignment_2/data/comparison_figures",
        metrics=("bbox_density", "D_est", "R_g"),
        show_title=False,
        panel_titles=False,
        title_fontsize=18,
        panel_title_fontsize=14,
        label_fontsize=13,
        tick_fontsize=11,
        line_width=2.2,
        marker_size=6,
        alpha_band=0.18,
        pde_big_csv_path="../assignment_2/data/dla/pde_ita_metrics_big.csv",
        pde_big_legend="optimized: N=300",
    )

